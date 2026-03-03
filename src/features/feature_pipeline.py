# -*- coding: utf-8 -*-
"""
Расчёт фичей для prepared dataset_rework.

Все фичи считаются внутри session_key (groupby) — границы сессий не пересекаются.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Список фич (без symbol_encoded — добавляется отдельно)
FEATURE_COLS = [
    "rd_mom_1",
    "rd_mom_5",
    "rd_mom_10",
    "rd_acceleration",
    "rd_zscore_30",
    "rd_ema_20",
    "abs_rd",
    "ret_1",
    "ret_5",
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "bb_width",
    "bb_position",
    "volatility_14",
    "volume_rel_20",
    "body_ratio",
    "close_position",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
]

def _ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Создать open/high/low из close_price при отсутствии."""
    if "close_price" not in df.columns:
        raise ValueError("Требуется колонка close_price")
    if "open" not in df.columns:
        df = df.copy()
        df["open"] = df["high"] = df["low"] = df["close_price"]
    return df


def _add_rd_features(df: pd.DataFrame, by: str = "session_key") -> pd.DataFrame:
    """RD-фичи внутри сессии."""
    grp = df.groupby(by, group_keys=False)
    rd = grp["rd_value"]

    df = df.copy()
    df["rd_mom_1"] = rd.diff(1)
    df["rd_mom_5"] = rd.diff(5)
    df["rd_mom_10"] = rd.diff(10)
    df["rd_acceleration"] = rd.diff().diff()
    # rd_zscore_30
    m = rd.transform(lambda x: x.rolling(30, min_periods=1).mean())
    s = rd.transform(lambda x: x.rolling(30, min_periods=1).std())
    df["rd_zscore_30"] = (df["rd_value"] - m) / (s.replace(0, np.nan).fillna(1e-9))
    df["rd_ema_20"] = rd.transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df["abs_rd"] = df["rd_value"].abs()
    return df


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """RSI внутри серии."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window, min_periods=1).mean()
    avg_loss = loss.rolling(window, min_periods=1).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan).fillna(1e-9))
    return 100 - (100 / (1 + rs))


def _add_price_features(df: pd.DataFrame, by: str = "session_key") -> pd.DataFrame:
    """Price-фичи внутри сессии."""
    grp = df.groupby(by, group_keys=False)
    close = grp["close_price"]

    df = df.copy()
    df["ret_1"] = close.pct_change(1)
    df["ret_5"] = close.pct_change(5)
    # RSI(14)
    df["rsi_14"] = grp["close_price"].transform(_rsi)
    # MACD(12,26,9)
    ema12 = close.transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = close.transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df["macd_line"] = ema12 - ema26
    df["macd_signal"] = df.groupby(by, group_keys=False)["macd_line"].transform(
        lambda x: x.ewm(span=9, adjust=False).mean()
    )
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]
    # Bollinger(20, 2)
    bb_mid = close.transform(lambda x: x.rolling(20, min_periods=1).mean())
    bb_std = close.transform(lambda x: x.rolling(20, min_periods=1).std())
    bb_std = bb_std.replace(0, np.nan).fillna(1e-9)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["bb_width"] = (bb_upper - bb_lower) / (bb_mid + 1e-9)
    df["bb_position"] = (df["close_price"] - bb_lower) / ((bb_upper - bb_lower) + 1e-9)
    # volatility_14 — rolling std от pct_change
    df["volatility_14"] = grp["close_price"].transform(
        lambda x: x.pct_change().rolling(14, min_periods=1).std()
    )
    return df


def _add_volume_features(df: pd.DataFrame, by: str = "session_key") -> pd.DataFrame:
    """Volume-фичи внутри сессии."""
    grp = df.groupby(by, group_keys=False)
    vol = grp["volume"]
    df = df.copy()
    vol_ma = vol.transform(lambda x: x.rolling(20, min_periods=1).mean())
    df["volume_rel_20"] = df["volume"] / (vol_ma + 1e-9)
    return df


def _add_ohlc_features(df: pd.DataFrame) -> pd.DataFrame:
    """OHLC-фичи (без groupby — локальные)."""
    df = df.copy()
    rng = df["high"] - df["low"]
    rng = rng.replace(0, np.nan)
    body = (df["close_price"] - df["open"]).abs()
    df["body_ratio"] = (body / rng).fillna(0.5).clip(0, 1)
    df["close_position"] = ((df["close_price"] - df["low"]) / rng).fillna(0.5).clip(0, 1)
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time-фичи."""
    if "datetime" not in df.columns:
        raise ValueError("Требуется колонка datetime")
    df = df.copy()
    dt = pd.to_datetime(df["datetime"], utc=True)
    hour = dt.dt.hour + dt.dt.minute / 60
    dow = dt.dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype(int)
    return df


def add_features(
    df: pd.DataFrame,
    session_key_col: str = "session_key",
    symbol_col: str = "symbol",
    fit_encoder: Optional[LabelEncoder] = None,
) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Добавить все фичи к prepared DataFrame.

    Args:
        df: Подготовленные данные с session_key.
        session_key_col: Колонка идентификатора сессии.
        symbol_col: Колонка символа.
        fit_encoder: LabelEncoder для symbol (если None — создаётся новый).

    Returns:
        (DataFrame с фичами, fitted LabelEncoder для symbol).
    """
    if session_key_col not in df.columns:
        raise ValueError(f"Колонка {session_key_col} отсутствует")
    df = _ensure_ohlc(df.copy())

    df = _add_rd_features(df, by=session_key_col)
    df = _add_price_features(df, by=session_key_col)
    df = _add_volume_features(df, by=session_key_col)
    df = _add_ohlc_features(df)
    df = _add_time_features(df)

    # symbol_encoded
    enc = fit_encoder or LabelEncoder()
    symbols = df[symbol_col].astype(str)
    if fit_encoder is None:
        enc.fit(symbols.dropna().unique())
    df["symbol_encoded"] = enc.transform(symbols)

    return df, enc


def get_feature_columns(include_symbol: bool = True) -> List[str]:
    """Возвращает список колонок-фичей."""
    cols = list(FEATURE_COLS)
    if include_symbol:
        cols.append("symbol_encoded")
    return cols
