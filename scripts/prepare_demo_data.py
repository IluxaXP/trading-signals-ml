#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для подготовки демо-данных из dataset_rework.
Загружает данные через dataset_rework_loader, выбирает указанный символ,
находит самый длинный непрерывный отрезок (gap ≤ 1.5 мин) и сохраняет первые N строк в JSON.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import pandas as pd

# Добавляем корень проекта в путь, чтобы импортировать src.data
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Импортируем загрузчик (если есть)
try:
    from src.data.dataset_rework_loader import load_dataset_rework
except ImportError:
    print("Не удалось импортировать load_dataset_rework. Убедитесь, что src.data доступен.")
    sys.exit(1)

DEFAULT_SYMBOL = "SCRT"   # символ с большим количеством данных
DEFAULT_OUTPUT = BASE_DIR / "mock" / "data" / "demo_data.json"
GAP_THRESHOLD_MIN = 1.5
MAX_ROWS = 1000


def find_longest_continuous_segment(df, gap_threshold_min=GAP_THRESHOLD_MIN):
    """
    Находит самый длинный непрерывный отрезок в DataFrame по символу.
    Возвращает срез DataFrame (первые MAX_ROWS этого отрезка).
    """
    # Сортируем по времени
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Вычисляем разницу во времени между последовательными строками (в минутах)
    df["time_diff"] = df["timestamp"].diff().fillna(0) / 60000

    # Создаём метки новых сессий (где разрыв > порога)
    df["new_session"] = (df["time_diff"] > gap_threshold_min).astype(int)
    df["session_id"] = df["new_session"].cumsum()

    # Длина каждой сессии
    session_lengths = df.groupby("session_id").size()

    if len(session_lengths) == 0:
        return pd.DataFrame()

    # Выбираем самую длинную сессию
    longest_session = session_lengths.idxmax()
    longest_df = df[df["session_id"] == longest_session].copy()

    # Отбрасываем служебные колонки
    longest_df.drop(columns=["time_diff", "new_session", "session_id"], inplace=True)

    return longest_df


def main():
    parser = argparse.ArgumentParser(description="Подготовка демо-данных из dataset_rework")
    parser.add_argument(
        "--symbol",
        type=str,
        default=DEFAULT_SYMBOL,
        help=f"Символ для извлечения (по умолчанию {DEFAULT_SYMBOL})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Путь для сохранения JSON (по умолчанию {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=MAX_ROWS,
        help=f"Максимальное количество строк в выходном файле (по умолчанию {MAX_ROWS})",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Путь к папке dataset_rework (если не указан, используется стандартный)",
    )
    args = parser.parse_args()

    print("📂 Загрузка dataset_rework...")
    # Загружаем все данные
    df_all = load_dataset_rework(data_dir=args.data_dir, verbose=True)

    # Фильтруем по символу
    df_symbol = df_all[df_all["symbol"] == args.symbol].copy()
    if len(df_symbol) == 0:
        print(f"❌ Символ {args.symbol} не найден в данных.")
        print("Доступные символы (первые 20):", df_all["symbol"].unique()[:20])
        sys.exit(1)

    print(f"✅ Найдено {len(df_symbol)} строк для {args.symbol}")

    # Ищем непрерывный отрезок
    continuous_df = find_longest_continuous_segment(df_symbol)
    if len(continuous_df) == 0:
        print("❌ Не найдено непрерывных сегментов.")
        sys.exit(1)

    print(f"📏 Самый длинный непрерывный отрезок: {len(continuous_df)} строк")

    # Берём первые max_rows строк из непрерывного отрезка
    demo_df = continuous_df.head(args.max_rows).copy()

    # Переименовываем close_price -> close, если нужно
    if "close_price" in demo_df.columns and "close" not in demo_df.columns:
        demo_df.rename(columns={"close_price": "close"}, inplace=True)

    # Оставляем только нужные колонки
    keep_cols = ["timestamp", "open", "high", "low", "close", "volume", "rd_value"]
    # Проверяем наличие колонок
    available_cols = [c for c in keep_cols if c in demo_df.columns]
    if len(available_cols) != len(keep_cols):
        missing = set(keep_cols) - set(available_cols)
        print(f"⚠️  Отсутствуют колонки: {missing}. Проверьте имена колонок в данных.")
        sys.exit(1)
    demo_df = demo_df[available_cols]

    # Сбрасываем индекс и преобразуем в список словарей
    demo_data = demo_df.to_dict(orient="records")

    # Создаём папку для выходного файла, если её нет
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Сохраняем JSON
    with open(output_path, "w") as f:
        json.dump(demo_data, f, indent=2)

    print(f"✅ Демо-данные сохранены в {output_path}")
    print(f"   Строк: {len(demo_data)}")
    print(f"   Диапазон времени: {demo_df['timestamp'].min()} – {demo_df['timestamp'].max()}")


if __name__ == "__main__":
    main()