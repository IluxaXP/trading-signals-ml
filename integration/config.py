import os

# Базовый URL платформы (мок или реальная)
API_BASE = os.getenv("API_BASE", "http://localhost:3000")

# Интервал опроса в секундах (для демо 5, для прода 60)
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))

# Таймауты запросов (сек)
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "5"))

# Режим работы: 'ml_shadow' или 'ml_production'
SIGNAL_SOURCE = os.getenv("SIGNAL_SOURCE", "ml_shadow")

# Порог уверенности для отправки сигнала (если < порога -> HOLD)
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))