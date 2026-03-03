# -*- coding: utf-8 -*-
"""
Подгрузка warmup-данных через Bybit API V5.

Для фичей с длинным окном (RSI-14, BB-20, rd_zscore-30, MACD) первые ~30 строк
каждой сессии будут NaN. Warmup загружает исторические 1-min свечи с Bybit
перед началом сессии, чтобы фичи по OHLCV вычислялись корректно.

rd_value в warmup = 0 (нейтральное), signal_barrier = NaN (Bybit не даёт разметку).
"""

from __future__ import annotations

import os
import time
from typing import Optional

import pandas as pd

from src.data.data_prep_dataset_rework import load_prepared

DEFAULT_WARMUP_SIZE = 60
DEFAULT_WARMUP_MAX = 120
RATE_LIMIT_DELAY = 0.15  # сек между запросами


def _to_bybit_symbol(symbol: str, symbol_map: Optional[dict] = None) -> str:
    """Преобразует символ в формат Bybit (BTCUSDT, XRPUSDT)."""
    s = str(symbol).strip().upper()
    if symbol_map and s in symbol_map:
        return symbol_map[s]
    if s.endswith("USDT") or s.endswith("USDC"):
        return s
    return s + "USDT"


def _fetch_klines_from_bybit(
    symbol: str,
    end_ts_ms: int,
    limit: int = 60,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    testnet: bool = False,
) -> pd.DataFrame:
    """
    Загружает 1-min свечи с Bybit API V5 до end_ts_ms.

    Args:
        symbol: Bybit symbol (BTCUSDT, XRPUSDT).
        end_ts_ms: Конечная метка времени (ms) — данные будут до этой точки.
        limit: Максимум свечей (до 1000).
        api_key, api_secret: Опционально, для больших лимитов.
        testnet: Использовать testnet.

    Returns:
        DataFrame с колонками: timestamp, open, high, low, close_price, volume.
        Пустой DataFrame при ошибке.
    """
    try:
        from pybit.unified_trading import HTTP
    except ImportError:
        raise ImportError("pybit не установлен. pip install pybit")

    session = HTTP(
        testnet=testnet,
        api_key=api_key or "",
        api_secret=api_secret or "",
    )
    # Kline: end — последняя свеча, Bybit возвращает от new to old
    result = session.get_kline(
        category="linear",
        symbol=symbol,
        interval="1",
        end=end_ts_ms,
        limit=min(limit, 1000),
    )
    if result.get("retCode") != 0:
        return pd.DataFrame()
    lst = result.get("result", {}).get("list", [])
    if not lst:
        return pd.DataFrame()
    # list: [startTime, open, high, low, close, volume, turnover] — от новых к старым
    rows = []
    for item in lst:
        ts = int(item[0])
        rows.append({
            "timestamp": ts,
            "open": float(item[1]),
            "high": float(item[2]),
            "low": float(item[3]),
            "close_price": float(item[4]),
            "volume": float(item[5]),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def add_warmup_from_bybit(
    prepared: pd.DataFrame,
    warmup_size: int = DEFAULT_WARMUP_SIZE,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    testnet: bool = False,
    rate_limit_delay: float = RATE_LIMIT_DELAY,
    symbol_map: Optional[dict] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Для каждой сессии загружает warmup-свечи с Bybit и добавляет их в DataFrame.

    Args:
        prepared: DataFrame из load_prepared() с session_key, symbol, datetime.
        warmup_size: Количество 1-min свечей до начала сессии.
        api_key, api_secret: Из env (BYBIT_API_KEY, BYBIT_API_SECRET) если не заданы.
        testnet: Использовать testnet (иначе mainnet — больше истории).
        rate_limit_delay: Пауза между запросами (сек).
        symbol_map: Маппинг внутренних символов на Bybit, напр. {"4": "4USDT", "B2": "B2USDT"}.
        verbose: Выводить статистику.

    Returns:
        DataFrame с warmup-строками, _is_warmup=1 для warmup.
        Warmup: timestamp, open, high, low, close_price, volume; rd_value, signal_barrier = NaN.
    """
    api_key = api_key or os.environ.get("BYBIT_API_KEY", "")
    api_secret = api_secret or os.environ.get("BYBIT_API_SECRET", "")
    testnet = testnet or os.environ.get("BYBIT_TESTNET", "").lower() in ("true", "1", "yes")

    prepared = prepared.copy()
    prepared["_is_warmup"] = 0
    base_cols = [c for c in prepared.columns if c != "_is_warmup"]

    rows_list = []
    symbols_seen = set()
    n_warmup_total = 0

    for skey, grp in prepared.groupby("session_key", sort=False):
        symbol_raw = grp["symbol"].iloc[0]
        symbol_bybit = _to_bybit_symbol(symbol_raw, symbol_map)
        first_ts = grp["datetime"].min()
        end_ms = int(first_ts.timestamp() * 1000) - 60_000  # 1 мин до начала сессии

        if symbol_bybit not in symbols_seen and rate_limit_delay > 0:
            time.sleep(rate_limit_delay)
        symbols_seen.add(symbol_bybit)

        try:
            klines = _fetch_klines_from_bybit(
                symbol=symbol_bybit,
                end_ts_ms=end_ms,
                limit=warmup_size,
                api_key=api_key or None,
                api_secret=api_secret or None,
                testnet=testnet,
            )
        except Exception as e:
            if verbose:
                print(f"Warmup skip {skey}: {e}")
            rows_list.append(grp.assign(_is_warmup=0))
            continue

        if klines.empty:
            rows_list.append(grp.assign(_is_warmup=0))
            continue

        klines["datetime"] = pd.to_datetime(klines["timestamp"], unit="ms", utc=True)
        klines["session_key"] = skey
        klines["symbol"] = symbol_raw
        klines["rd_value"] = 0.0  # Bybit не даёт rd; 0 — нейтральное значение для расчёта фичей
        klines["signal_barrier"] = pd.NA
        klines["time_diff_min"] = klines["datetime"].diff().dt.total_seconds() / 60
        klines["source_day"] = "bybit_warmup"
        klines["_is_warmup"] = 1
        for c in base_cols:
            if c not in klines.columns:
                klines[c] = pd.NA
        klines = klines[[c for c in base_cols] + ["_is_warmup"]]
        rows_list.append(klines)
        rows_list.append(grp.assign(_is_warmup=0))
        n_warmup_total += len(klines)

    out = pd.concat(rows_list, ignore_index=True)
    out = out.sort_values(["session_key", "datetime"]).reset_index(drop=True)

    if verbose:
        print(f"Добавлено warmup-строк (Bybit): {n_warmup_total:,}")

    return out


def remove_warmup(df: pd.DataFrame, warmup_flag_col: str = "_is_warmup") -> pd.DataFrame:
    """Удаляет warmup-строки из DataFrame."""
    if warmup_flag_col not in df.columns:
        return df
    return df[df[warmup_flag_col] == 0].drop(columns=[warmup_flag_col], errors="ignore").reset_index(drop=True)


def load_with_warmup(
    prepared_path: Optional[str] = None,
    warmup_size: int = DEFAULT_WARMUP_SIZE,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    testnet: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Загружает prepared-данные и добавляет warmup с Bybit.

    Args:
        prepared_path: Путь к prepared Parquet. По умолчанию — default.
        warmup_size: Количество warmup-свечей на сессию.
        api_key, api_secret: Опционально.
        testnet: mainnet по умолчанию (больше истории).
        verbose: Выводить статистику.

    Returns:
        DataFrame с warmup, _is_warmup=1 для warmup.
    """
    prep = load_prepared(prepared_path)
    return add_warmup_from_bybit(
        prep,
        warmup_size=warmup_size,
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
        verbose=verbose,
    )
