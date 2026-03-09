#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polling worker для интеграции с платформой.
Опрашивает эндпоинт /api/ml/ds/feature-windows, получает окна признаков,
вызывает модель через src.api.inference.predict и отправляет сигналы обратно.
"""

import sys
import time
import os
import csv
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.api.inference import predict
from integration.config import (
    API_BASE,
    API_USERNAME,
    API_PASSWORD,
    POLL_INTERVAL,
    REQUEST_TIMEOUT,
    SIGNAL_SOURCE,
    CONFIDENCE_THRESHOLD,   # оставлен для совместимости, но не используется в логике
    get_mode_name,
    is_production,
)

_last_signals = {}

def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}")

# ---------- Функция логирования входных данных ----------
def log_features_to_file(symbol, features, window_end):
    """
    Сохраняет полученное окно признаков в CSV-файл.
    Файлы создаются в папке logs/ с именем {symbol}_{YYYY-MM-DD}.csv.
    """
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    date_str = datetime.fromtimestamp(window_end / 1000).strftime('%Y-%m-%d')
    filename = log_dir / f"{symbol}_{date_str}.csv"
    file_exists = filename.exists()
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'rd_value', 'open', 'high', 'low', 'close', 'volume'])
        # Восстанавливаем приблизительные временные метки для каждой строки (от earliest к latest)
        for i, row in enumerate(features):
            # i=0 – самая старая минута, i=59 – самая новая (window_end)
            ts = window_end - (59 - i) * 60000
            writer.writerow([ts] + row)

def get_feature_windows():
    auth = (API_USERNAME, API_PASSWORD) if is_production() else None
    try:
        response = requests.get(
            f"{API_BASE}/api/ml/ds/feature-windows?readyOnly=true",
            timeout=REQUEST_TIMEOUT,
            auth=auth,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        log(f"❌ Ошибка получения фичей: {e}")
        return None

def send_signal(payload):
    payload["source"] = SIGNAL_SOURCE
    auth = (API_USERNAME, API_PASSWORD) if is_production() else None
    try:
        response = requests.post(
            f"{API_BASE}/api/signals/ingest",
            json=payload,
            timeout=REQUEST_TIMEOUT,
            auth=auth,
        )
        if response.status_code == 400:
            log(f"❌ Ошибка 400. Payload: {payload}")
            return False
        response.raise_for_status()
        log(f"✅ Сигнал {payload['signal']} отправлен")
        return True
    except requests.exceptions.RequestException as e:
        log(f"⚠️ Ошибка отправки: {e}")
        return False

def run_iteration():
    log("🔄 Запрос окон фичей...")
    data = get_feature_windows()
    if not data:
        return

    feature_columns = data.get("featureColumns", [])
    for item in data.get("items", []):
        if item.get("state") != "READY":
            log(f"⏩ Пропуск {item.get('symbol')} (state={item.get('state')})")
            continue

        symbol = item["symbol"]
        features = item["features"]
        window_end = item.get("windowEndTimestamp")

        # ----- Логирование сырых данных (если включено) -----
        if os.getenv('LOG_FEATURES', 'false').lower() == 'true':
            try:
                log_features_to_file(symbol, features, window_end)
            except Exception as e:
                log(f"⚠️ Ошибка при логировании фичей: {e}")

        # Создаём DataFrame из полученной матрицы
        df = pd.DataFrame(features, columns=feature_columns)
        df["symbol"] = symbol
        df["timestamp"] = [
            window_end - (len(df) - 1 - i) * 60000 for i in range(len(df))
        ]
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

        if "close" in df.columns and "close_price" not in df.columns:
            df.rename(columns={"close": "close_price"}, inplace=True)

        try:
            signal, confidence = predict(df)
        except Exception as e:
            log(f"❌ Ошибка при предсказании для {symbol}: {e}")
            continue

        if signal == 0:
            log(f"⏸️ HOLD для {symbol} (conf={confidence:.2f})")
            continue

        signal_str = "BUY" if signal == 1 else "SELL"
        prev = _last_signals.get(symbol)
        if prev == signal:
            log(f"⏸️ Сигнал не изменился для {symbol} (по-прежнему {signal_str})")
            continue

        close_price_col = "close_price" if "close_price" in df.columns else "close"
        close_price = df.iloc[-1][close_price_col]

        payload = {
            "symbol": symbol,
            "timestamp": window_end,
            "signal": signal_str,
            "price": float(close_price),
            "rating": round(confidence, 4),
        }
        if send_signal(payload):
            _last_signals[symbol] = signal

def main():
    mode = get_mode_name()
    log(f"🚀 Worker запущен в режиме: {mode}")
    if is_production():
        log(f"   API: {API_BASE} (авторизация: {API_USERNAME})")
        if API_USERNAME == 'dataset':
            log("   ⚠️ Профиль 'dataset' — отправка сигналов НЕВОЗМОЖНА (POST /ingest запрещён).")
        else:
            log(f"   ✅ Сигналы будут отправляться с источником {SIGNAL_SOURCE} (теневой режим).")
    else:
        log(f"   API: {API_BASE} (мок-сервер, без авторизации)")
        log("   ✅ Сигналы отправляются на локальный мок-сервер.")
    log(f"   Интервал опроса: {POLL_INTERVAL} сек")
    if os.getenv('LOG_FEATURES', 'false').lower() == 'true':
        log("   📁 Логирование входных окон включено (папка logs/)")
    while True:
        run_iteration()
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()