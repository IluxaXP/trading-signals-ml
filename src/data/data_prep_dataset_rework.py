# -*- coding: utf-8 -*-
"""
Подготовка данных dataset_rework для обучения.

Фильтрация по time_diff, разбиение на сессии, сохранение в Parquet.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from .dataset_rework_loader import load_dataset_rework

SESSION_GAP_THRESHOLD_MIN = 1.5
MIN_SESSION_LENGTH = 60
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "outputs", "prepared_dataset_rework.parquet")


def prepare_for_training(
    df: Optional[pd.DataFrame] = None,
    data_dir: Optional[str] = None,
    session_gap_min: float = SESSION_GAP_THRESHOLD_MIN,
    min_session_length: int = MIN_SESSION_LENGTH,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Подготовка данных для обучения: фильтрация, сессии, только ok_for_train.

    Args:
        df: Сырые данные. Если None — загружает через load_dataset_rework(data_dir).
        data_dir: Директория dataset_rework (если df=None).
        session_gap_min: Порог разрыва (мин). time_diff > session_gap_min → новая сессия.
        min_session_length: Минимум точек в сессии. Сессии короче отбрасываются.
        verbose: Выводить ли статистику.

    Returns:
        DataFrame с колонками: session_key, symbol, datetime, timestamp, rd_value,
        open, high, low, close_price, volume, signal_barrier, time_diff_min, source_day.
    """
    if df is None:
        df = load_dataset_rework(data_dir=data_dir, verbose=verbose)

    df = df.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    df["time_diff_min"] = df.groupby("symbol")["datetime"].diff().dt.total_seconds() / 60

    # Фильтр: только строки после непрерывности (time_diff <= порог)
    mask = df["time_diff_min"].fillna(0) <= session_gap_min
    work = df[mask].copy()
    work = work.sort_values(["symbol", "datetime"]).reset_index(drop=True)

    # Пересчёт time_diff и session_id
    work["time_diff_min"] = work.groupby("symbol")["datetime"].diff().dt.total_seconds() / 60
    work["new_session"] = (work["time_diff_min"].fillna(0) > session_gap_min).astype(int)
    work["session_id"] = work.groupby("symbol")["new_session"].cumsum()
    work["session_key"] = work["symbol"].astype(str) + "_s" + work["session_id"].astype(str)

    sessions = work.groupby("session_key").agg(rows=("timestamp", "size")).reset_index()
    valid_keys = sessions[sessions["rows"] >= min_session_length]["session_key"].values
    work = work[work["session_key"].isin(valid_keys)].copy()
    work = work.drop(columns=["new_session", "session_id"], errors="ignore")

    # Recalculate time_diff within each session (first row of session → NaN)
    work["time_diff_min"] = work.groupby("session_key")["datetime"].diff().dt.total_seconds() / 60

    if verbose:
        n_sessions = work["session_key"].nunique()
        n_rows = len(work)
        n_symbols = work["symbol"].nunique()
        print(f"Подготовлено: {n_rows:,} строк, {n_sessions:,} сессий, {n_symbols} символов")

    return work


def save_prepared(
    df: pd.DataFrame,
    path: Optional[str] = None,
) -> str:
    """
    Сохраняет подготовленные данные в Parquet (при наличии pyarrow) или CSV.

    Args:
        df: DataFrame из prepare_for_training().
        path: Путь к файлу. По умолчанию outputs/prepared_dataset_rework.parquet.

    Returns:
        Путь к сохранённому файлу.
    """
    path = path or DEFAULT_OUTPUT_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # symbol может быть int или str — приводим к str для Parquet
    df = df.copy()
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str)
    try:
        df.to_parquet(path, index=False, compression="snappy")
    except ImportError:
        path_csv = path.replace(".parquet", ".csv")
        df.to_csv(path_csv, index=False, encoding="utf-8")
        path = path_csv
        print("pyarrow не установлен — сохранено в CSV. Для Parquet: pip install pyarrow")
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Сохранено: {path} ({size_mb:.2f} MB, {df.shape[0]:,} строк)")
    return path


def load_prepared(path: Optional[str] = None) -> pd.DataFrame:
    """
    Загружает подготовленные данные из Parquet или CSV.

    Args:
        path: Путь к файлу. По умолчанию outputs/prepared_dataset_rework.parquet.

    Returns:
        DataFrame с session_key, symbol, datetime, OHLC, volume, signal_barrier и др.
    """
    path = path or DEFAULT_OUTPUT_PATH
    if not os.path.exists(path):
        path_alt = path.replace(".parquet", ".csv")
        if os.path.exists(path_alt):
            path = path_alt
        else:
            raise FileNotFoundError(f"Файл не найден: {path}. Запустите prepare_for_training() и save_prepared().")
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    df = pd.read_csv(path)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


if __name__ == "__main__":
    df = prepare_for_training(verbose=True)
    save_prepared(df)
