# integration/config.py
import os

API_BASE = os.getenv("API_BASE", "http://localhost:3000")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "5"))
SIGNAL_SOURCE = os.getenv("SIGNAL_SOURCE", "ml_shadow")
# Порог больше не используется в worker, оставлен для совместимости
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "1.0"))