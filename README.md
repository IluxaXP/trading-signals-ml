## 📄 README.md (исправленный состав команды)

# 🚀 ML Trading Signals — Хакатон 2engine

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.1.0-orange.svg)](https://lightgbm.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Проект команды **№3** для хакатона магистратуры (март 2026).  
Разработка ML-модели для генерации торговых сигналов (BUY/SELL/HOLD) на криптовалютных фьючерсах Bybit.  
Решение включает в себя исследование целевых переменных, отбор признаков, обучение модели LightGBM и интеграционный слой (polling-воркер + FastAPI) для взаимодействия с платформой заказчика.

---

## 📋 Описание задачи

**Заказчик:** [2engine](https://2engine.ru/) — российская IT-компания, специализирующаяся на развитии онлайн-бизнеса с 2010 года.

**Бизнес-задача:**  
Разработать модель машинного обучения, которая заменит текущую алгоритмическую стратегию на основе пересечения скользящих средних (EMA) и повысит эффективность торговых операций на криптовалютном рынке.

**Технические требования:**
- Обработка минутных данных OHLCV + кастомного осциллятора `rd_value`.
- Классификация на 3 класса: BUY (1), SELL (-1), HOLD (0).
- Работа с множеством криптовалютных пар (более 200).
- Интеграция через API: платформа отдаёт окна признаков, ML-сервис возвращает сигнал.

**Входные данные:**  
`timestamp`, `symbol`, `open`, `high`, `low`, `close_price`, `volume`, `rd_value`.  
Целевая переменная (для обучения) создавалась нами на основе метода тройного барьера с фиксированными уровнями Take Profit / Stop Loss (TP=1%, SL=0.5%, горизонт 20 баров).  
Детальное исследование таргетов и фичей описано в [документации](docs/).

---

## 👥 Команда №3

| Участник | Роль | GitHub | Основные обязанности |
|----------|------|--------|---------------------|
| **Старинцева Наталья** | 🎯 Team Lead, EDA Lead, Backend Developer | [@Starnatvl](https://github.com/Starnatvl) | Координация проекта, коммуникация с заказчиком, разработка API и интеграционного слоя, итоговая презентация |
| **Кобзева Мария** | 📊 Data Scientist | [@Maria_Kob](https://github.com/Maria_Kob) | EDA, feature engineering, обработка данных, эксперименты с моделями, визуализация |
| **Стрик Наталья** | 🔬 Data Scientist + QA | [@StrikNa](https://github.com/StrikNa) | Feature engineering, обучение классических моделей, балансировка классов, бэктестинг |
| **Мюлинг Илья** | 🤖 ML Engineer | [@IluxaXP](https://github.com/IluxaXP) | Пайплайн обучения, подбор гиперпараметров, inference-скрипт, экспорт моделей (ONNX) |

---

## 🗂️ Структура проекта

```text
trading-signals-ml/
│
├── .gitignore
├── README.md
├── requirements.txt                # Python‑зависимости
├── docker-compose.yml               # Запуск всех сервисов одной командой
├── Dockerfile.api                    # Образ для FastAPI
├── Dockerfile.worker                  # Образ для polling‑воркера
├── Dockerfile.mock                     # Образ для мок‑сервера (Node.js)
│
├── data/                             # Данные (игнорируются git)
│   └── raw/dataset_rework/            # Исходные CSV от заказчика
│
├── notebooks/                         # Jupyter ноутбуки (EDA, эксперименты)
│
├── src/                               # Исходный код
│   ├── data/                           # Загрузчики данных
│   ├── features/                        # Feature pipeline (add_features)
│   ├── api/                              # FastAPI приложение
│   │   ├── app.py                         # Эндпоинты
│   │   └── inference.py                    # Общая функция predict
│   └── ...
│
├── models/                             # Сохранённые артефакты модели
│   ├── champion_hackathon_tp_sl_1_05.joblib   # LightGBM
│   ├── scaler_tp_sl_1_05.joblib               # StandardScaler
│   └── features_selected_tp_sl_1_05.txt       # Отобранные фичи
│
├── integration/                        # Интеграционный слой (polling)
│   ├── worker.py                         # Основной цикл опроса
│   └── config.py                          # Настройки (переменные окружения)
│
├── mock/                               # Мок‑сервер платформы (Node.js)
│   ├── server.js
│   ├── package.json
│   └── data/                             # JSON с историческими данными для демо
│
├── scripts/                             # Вспомогательные скрипты
│   └── prepare_demo_data.py               # Подготовка данных для мок‑сервера
│
├── docs/                                # Документация (исследования)
└── tests/                               # Тесты
```

---

## 📊 Ключевые результаты

- **Лучшая модель:** LightGBM с целевой переменной `tp_sl_1_05` (TP=1%, SL=0.5%).
- **Метрики:** AUC = 0.705, F1 = 0.654.
- **Прибыльность:** чистая доходность (net %) в бэктесте **+2011%** при комиссии 0.1% (2768 сделок).
- **Фичи:** 20 отобранных признаков (на основе `rd_value`, OHLCV, технических индикаторов).
- **Интеграция:** реализованы polling-воркер и FastAPI-сервер, поддерживающие спецификацию 2.1.0 (взаимодействие с платформой через REST).

---

## 🚀 Быстрый старт

### 1️⃣ Клонирование репозитория
```bash
git clone https://github.com/your-team/trading-signals-ml.git
cd trading-signals-ml
```

### 2️⃣ Запуск демо с реальными данными (локально)

#### Установка зависимостей Python
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Подготовка демо‑данных
Убедитесь, что папка `data/raw/dataset_rework` содержит исходные CSV.  
Затем выполните:
```bash
python scripts/prepare_demo_data.py --symbol SCRT --output mock/data/btcusdt_demo.json
```
Скрипт выберет самый длинный непрерывный отрезок для символа SCRT и сохранит его в JSON для мок‑сервера.

#### Запуск мок‑сервера (эмулятор платформы)
```bash
cd mock
npm install
node server.js
# Сервер будет доступен на http://localhost:3000
```

#### Запуск polling‑воркера (в другом терминале)
```bash
cd integration
python worker.py
# Воркер начнёт опрашивать мок‑сервер, вызывать модель и отправлять сигналы
```

#### (Опционально) Запуск FastAPI
```bash
uvicorn src.api.app:app --reload
# Документация: http://localhost:8000/docs
```

### 3️⃣ Запуск через Docker Compose (рекомендуется для демо)

```bash
docker-compose up --build
```
Будут подняты три контейнера:
- `mock` на порту 3000
- `worker` (без открытого порта)
- `api` на порту 8000

Логи воркера и мок‑сервера отображаются в консоли.

---

## 🧪 Тестирование

```bash
pytest tests/ -v
```

---

## 📚 Документация

Подробные исследования целевых переменных, фичей и экспериментов находятся в папке [`docs/`](docs/).  
Основные файлы:
- `01_TARGET_VARIABLES_RESEARCH.md` – обзор методов разметки.
- `02_TARGET_RESEARCH_CONCLUSIONS.md` – выбор `tp_sl_1_05`.
- `08_FEATURES_SOURCES_AND_RATIONALE.md` – описание фичей и их отбор.
- `05_PIPELINE_CONCLUSIONS.md` – итоги пайплайна.

---

## 🙏 Благодарности

- **2engine** – за предоставленный реальный кейс и данные.
- **Менторы хакатона** – за обратную связь.
- **Open Source сообщество** – за библиотеки (LightGBM, FastAPI, scikit-learn).

---

## 📌 Контакты

- **Team Lead:** Наталья Старинцева – [@Starnatvl](https://github.com/Starnatvl), star.nat.vl@gmail.com
- По вопросам API и интеграции – туда же.

```
