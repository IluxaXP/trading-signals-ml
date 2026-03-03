# -*- coding: utf-8 -*-
"""Unified loader for dataset_rework CSV files."""

from __future__ import annotations

import glob
import os
from typing import Optional

import pandas as pd


def detect_separator(file_path: str) -> str:
    """Detect CSV separator from the header line."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
        header = handle.readline()
    return ";" if header.count(";") > header.count(",") else ","


def find_dataset_rework_dir(start_dir: Optional[str] = None) -> str:
    """Find dataset_rework directory from common project locations."""
    base = os.path.abspath(start_dir or os.getcwd())
    _dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "dataset_rework"),
        os.path.join(os.path.dirname(base), "dataset_rework"),
        os.path.join(_dir, "..", "..", "dataset_rework"),
    ]
    candidates = [os.path.abspath(path) for path in candidates]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"Папка dataset_rework не найдена. Проверено: {candidates}")


def load_dataset_rework(
    data_dir: Optional[str] = None,
    exclude_macos: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load dataset_rework from nested YYYY-MM-DD(-N)/SYMBOL.csv structure.

    Adds helper columns:
    - source_file: CSV filename
    - source_day: parent folder name
    - datetime: parsed UTC timestamp
    """
    resolved_dir = os.path.abspath(data_dir) if data_dir else find_dataset_rework_dir()
    files = glob.glob(os.path.join(resolved_dir, "**", "*.csv"), recursive=True)

    if exclude_macos:
        files = [
            path
            for path in files
            if "__MACOSX" not in path and not os.path.basename(path).startswith("._")
        ]

    if not files:
        raise FileNotFoundError(f"CSV файлы не найдены в: {resolved_dir}")

    dfs = []
    skipped = []

    for path in sorted(files):
        try:
            sep = detect_separator(path)
            df = pd.read_csv(
                path,
                sep=sep,
                comment="#",
                encoding="utf-8",
                encoding_errors="replace",
                on_bad_lines="skip",
            )
            df.columns = [col.strip().replace("#", "").strip() for col in df.columns]

            if "close" in df.columns and "close_price" not in df.columns:
                df = df.rename(columns={"close": "close_price"})

            required = {"timestamp", "symbol"}
            if not required.issubset(df.columns):
                skipped.append((os.path.basename(path), "нет обязательных колонок"))
                continue

            for col in ["timestamp", "rd_value", "close_price", "volume", "signal_barrier"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df["source_file"] = os.path.basename(path)
            df["source_day"] = os.path.basename(os.path.dirname(path))
            dfs.append(df)
        except Exception as error:  # pragma: no cover - defensive data-loader path
            skipped.append((os.path.basename(path), str(error)[:120]))

    if not dfs:
        raise ValueError("Не удалось прочитать ни одного CSV файла. Проверь формат/кодировку данных.")

    full = pd.concat(dfs, ignore_index=True)
    full = full.dropna(subset=["timestamp", "symbol"])

    unit = "ms" if full["timestamp"].max() > 3e10 else "s"
    full["datetime"] = pd.to_datetime(full["timestamp"], unit=unit, errors="coerce", utc=True)
    full = full.dropna(subset=["datetime"])

    full = (
        full.sort_values(["symbol", "timestamp"])
        .drop_duplicates(subset=["symbol", "timestamp"], keep="last")
        .reset_index(drop=True)
    )

    if verbose:
        print(f"Найдено CSV: {len(files)} | успешно прочитано: {len(dfs)} | пропущено: {len(skipped)}")
        if skipped:
            print("Примеры пропущенных файлов:", skipped[:5])
        print(f"Загружено: {len(full):,} строк, {full['symbol'].nunique()} символов")

    return full
