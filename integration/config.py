# integration/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные из .env, если файл существует
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print("✅ Загружены переменные из .env")
else:
    print("ℹ️ Файл .env не найден, используется демо-режим (мок-сервер)")

# Определяем режим работы
def is_production() -> bool:
    """Возвращает True, если заданы все необходимые production-переменные."""
    return bool(os.getenv("API_BASE_URL") and os.getenv("API_USERNAME") and os.getenv("API_PASSWORD"))

def get_mode_name() -> str:
    return "PRODUCTION" if is_production() else "DEMO (mock)"

# Основные настройки
if is_production():
    API_BASE = os.getenv("API_BASE_URL").rstrip('/')
    API_USERNAME = os.getenv("API_USERNAME")
    API_PASSWORD = os.getenv("API_PASSWORD")
    # В production используем базовый URL без порта (https)
else:
    API_BASE = os.getenv("API_BASE", "http://localhost:3000")  # по умолчанию мок
    API_USERNAME = None
    API_PASSWORD = None

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60" if is_production() else "5"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "10"))
SIGNAL_SOURCE = os.getenv("SIGNAL_SOURCE", "ml_shadow")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))