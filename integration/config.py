# integration/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print("✅ Загружены переменные из .env")
else:
    print("ℹ️ Файл .env не найден, используется демо-режим (мок-сервер)")

def is_production() -> bool:
    """Возвращает True, если не включён принудительный мок и заданы все production-переменные."""
    if os.getenv("USE_MOCK", "").lower() == "true":
        return False
    return bool(os.getenv("API_BASE_URL") and os.getenv("API_USERNAME") and os.getenv("API_PASSWORD"))

def get_mode_name() -> str:
    if is_production():
        return "PRODUCTION"
    return "DEMO (mock)"

# Определяем базовый URL для получения данных
if is_production():
    DATA_API_BASE = os.getenv("API_BASE_URL").rstrip('/')
else:
    DATA_API_BASE = os.getenv("DATA_API_BASE", "http://localhost:3000")

# Определяем базовый URL для отправки сигналов
# Если SIGNAL_TO_MOCK=true, то используем локальный мок, иначе тот же DATA_API_BASE
if os.getenv("SIGNAL_TO_MOCK", "").lower() == "true":
    SIGNAL_API_BASE = "http://localhost:3000"
else:
    SIGNAL_API_BASE = DATA_API_BASE

# Авторизация для получения данных
DATA_USERNAME = os.getenv("API_USERNAME") if is_production() else None
DATA_PASSWORD = os.getenv("API_PASSWORD") if is_production() else None

# Авторизация для отправки сигналов
if SIGNAL_API_BASE == DATA_API_BASE:
    SIGNAL_USERNAME = DATA_USERNAME
    SIGNAL_PASSWORD = DATA_PASSWORD
else:
    SIGNAL_USERNAME = None
    SIGNAL_PASSWORD = None

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60" if is_production() else "5"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "10"))
SIGNAL_SOURCE = os.getenv("SIGNAL_SOURCE", "ml_shadow")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))