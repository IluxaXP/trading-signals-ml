# src/api/inference.py
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.features.feature_pipeline import add_features

MODEL_PATH = BASE_DIR / 'models' / 'champion_hackathon_tp_sl_1_05.joblib'
FEATURES_FILE = BASE_DIR / 'models' / 'features_selected_tp_sl_1_05.txt'

_model = None
_scaler = None
_selected_features = None
_buy_threshold = 0.6
_sell_threshold = 0.4

def _load_artifacts():
    global _model, _scaler, _selected_features
    if _model is None:
        bundle = joblib.load(MODEL_PATH)
        _model = bundle.get("model")
        _scaler = bundle.get("scaler")
        with open(FEATURES_FILE, 'r') as f:
            _selected_features = [line.strip() for line in f if line.strip()]
        if _model is None or _scaler is None or _selected_features is None:
            raise ValueError("Не удалось загрузить компоненты модели")

def predict(window_df: pd.DataFrame) -> tuple:
    """
    Возвращает (signal, confidence), где:
        signal = 1 (BUY), -1 (SELL) или 0 (HOLD)
        confidence = вероятность соответствующего класса (для BUY/SELL) или вероятность BUY для HOLD.
    """
    _load_artifacts()

    df = window_df.copy()

    if 'close' in df.columns and 'close_price' not in df.columns:
        df.rename(columns={'close': 'close_price'}, inplace=True)

    if 'datetime' not in df.columns and 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

    if 'session_key' not in df.columns:
        df['session_key'] = 'dummy'

    df_feat, _ = add_features(df, session_key_col='session_key')
    last_row = df_feat.iloc[-1:]
    X = last_row[_selected_features].copy()

    X_scaled = pd.DataFrame(
        _scaler.transform(X),
        columns=_selected_features,
        index=X.index
    )

    proba = _model.predict_proba(X_scaled)[0]          # массив вероятностей
    classes = _model.classes_

    # Индекс класса 1 (BUY)
    buy_idx = np.where(classes == 1)[0]
    if len(buy_idx) == 0:
        raise ValueError("Класс 1 (BUY) не найден в модели")
    buy_prob = proba[buy_idx[0]]

    # Применяем пороги
    if buy_prob >= _buy_threshold:
        return 1, buy_prob
    elif buy_prob <= _sell_threshold:
        # Уверенность в SELL = 1 - buy_prob (если модель бинарная)
        return -1, 1 - buy_prob
    else:
        return 0, buy_prob   # HOLD