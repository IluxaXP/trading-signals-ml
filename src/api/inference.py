# src/api/inference.py
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Добавляем корень проекта в sys.path, чтобы импортировать src.features
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.features.feature_pipeline import add_features

# Пути к артефактам (можно переопределить через переменные окружения)
MODEL_PATH = BASE_DIR / 'models' / 'champion_hackathon_tp_sl_1_05.joblib'
SCALER_PATH = BASE_DIR / 'models' / 'scaler_tp_sl_1_05.joblib'
FEATURES_PATH = BASE_DIR / 'models' / 'features_selected_tp_sl_1_05.txt'

# Глобальные переменные для загруженных объектов (загружаются один раз при первом вызове predict)
_model = None
_scaler = None
_selected_features = None

def _load_artifacts():
    """Загружает модель, scaler и список фичей."""
    global _model, _scaler, _selected_features
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        with open(FEATURES_PATH, 'r') as f:
            _selected_features = [line.strip() for line in f if line.strip()]

def predict(window_df: pd.DataFrame) -> tuple:
    """
    Принимает DataFrame с историей (минимум 60 строк) и возвращает (signal, confidence).
    DataFrame должен содержать колонки:
        timestamp, symbol, open, high, low, close, volume, rd_value
    (колонка close будет переименована в close_price, если нужно)
    """
    _load_artifacts()

    # Создаём копию, чтобы не менять исходный DataFrame
    df = window_df.copy()

    # Переименовываем close -> close_price, если необходимо
    if 'close' in df.columns and 'close_price' not in df.columns:
        df.rename(columns={'close': 'close_price'}, inplace=True)

    # Убедимся, что есть колонка close_price
    if 'close_price' not in df.columns:
        raise ValueError("DataFrame должен содержать колонку 'close_price' (или 'close')")

    # Добавляем фичи через feature_pipeline
    # Для add_features требуется колонка session_key. Если её нет – создаём фиктивную.
    if 'session_key' not in df.columns:
        df['session_key'] = 'dummy'

    df_feat, _ = add_features(df, session_key_col='session_key')

    # Берём последнюю строку
    last_row = df_feat.iloc[-1:]

    # Выбираем только нужные фичи (порядок важен)
    X = last_row[_selected_features].values.reshape(1, -1)

    # Масштабируем
    X_scaled = _scaler.transform(X)

    # Предсказание вероятностей
    proba = _model.predict_proba(X_scaled)[0]
    classes = _model.classes_          # например [-1, 0, 1]

    pred_class = classes[np.argmax(proba)]
    confidence = np.max(proba)

    # Если класс = 0 или уверенность ниже порога (0.5), возвращаем HOLD (0)
    if pred_class == 0 or confidence < 0.5:
        return 0, confidence

    return pred_class, confidence