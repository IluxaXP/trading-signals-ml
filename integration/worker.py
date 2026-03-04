#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polling worker для интеграции с платформой.
Опрашивает эндпоинт /api/ml/ds/feature-windows, получает окна признаков,
вызывает модель через src.api.inference.predict и отправляет сигналы обратно.
Реализована логика удержания позиции: повторяющиеся сигналы не отправляются.
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.api.inference import predict
from integration.config import (
    API_BASE,
    POLL_INTERVAL,
    REQUEST_TIMEOUT,
    SIGNAL_SOURCE,
)

# Хранилище последних отправленных сигналов для каждого символа
_last_signals = {}

def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}")

def get_feature_windows():
    try:
        response = requests.get(
            f"{API_BASE}/api/ml/ds/feature-windows?readyOnly=true",
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        log(f"❌ Ошибка получения фичей: {e}")
        return None

def send_signal(payload):
    payload["source"] = SIGNAL_SOURCE
    try:
        response = requests.post(
            f"{API_BASE}/api/signals/ingest",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 400:
            log(f"❌ Ошибка 400 (неверные данные). Payload: {payload}")
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

        # HOLD – не отправляем, позиция сохраняется
        if signal == 0:
            log(f"⏸️ HOLD для {symbol} (conf={confidence:.2f})")
            continue

        signal_str = "BUY" if signal == 1 else "SELL"
        # Проверяем, изменился ли сигнал по сравнению с предыдущим для этого символа
        prev_signal = _last_signals.get(symbol)
        if prev_signal == signal:
            log(f"⏸️ Сигнал не изменился для {symbol} (по-прежнему {signal_str})")
            continue

        # Если сигнал новый или сменился – отправляем
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
            # Обновляем последний сигнал только при успешной отправке
            _last_signals[symbol] = signal

def main():
    log(f"🚀 Worker запущен. Опрос {API_BASE} каждые {POLL_INTERVAL} сек.")
    while True:
        run_iteration()
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()