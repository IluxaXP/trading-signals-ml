# 🚀 ML Trading Signals — Хакатон 2engine

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.1.0-orange.svg)](https://lightgbm.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Проект команды **№3** для хакатона магистратуры (март 2026).  
Разработка ML-модели для генерации торговых сигналов (BUY/SELL/HOLD) на криптовалютных фьючерсах Bybit.  
Решение включает в себя исследование целевых переменных, отбор признаков, обучение модели LightGBM (102 признака, включая rolling-агрегаты) и интеграционный слой (polling-воркер + FastAPI) для взаимодействия с платформой заказчика.

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
| **Старинцева Наталья** | 🎯 Team Lead, Backend Developer | [@Starnatvl](https://github.com/Starnatvl) | Координация проекта, коммуникация с заказчиком, разработка API и интеграционного слоя, итоговая презентация |
| **Кобзева Мария** | 📊 Data Scientist, EDA Lead | [@Maria_Kob](https://github.com/Maria_Kob) | EDA, feature engineering, обработка данных, визуализация |
| **Стрик Наталья** | 🔬 Data Scientist + QA | [@StrikNa](https://github.com/StrikNa) | Feature engineering, балансировка классов |
| **Мюлинг Илья** | 🤖 ML Engineer | [@IluxaXP](https://github.com/IluxaXP) | Эксперименты с моделями, обучение классических моделей, бэктестинг, пайплайн обучения, подбор гиперпараметров, inference-скрипт, экспорт моделей |

---

## 🗂️ Структура проекта

```text
trading-signals-ml/
│
├── .gitignore
├── README.md
├── requirements.txt                # Python‑зависимости (минимальный набор)
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
│   │   ├── inference.py                    # Общая функция predict
│   │   └── model_bundle.py                  # Загрузка модели из bundle
│   └── ...
│
├── models/                             # Сохранённые артефакты модели
│   └── prod_lgbm_seq.joblib              # Финальная модель LightGBM (102 признака)
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

- **Модель:** LightGBM с **102 признаками** (22 базовых + 80 rolling-агрегатов по окнам 5, 15, 30, 60).
- **Метрики:** AUC на валидации = 0.907, AUC на тесте = 0.806.
- **Прибыльность:**  
  - Тестовый день: **+1146%** (840 сделок)  
  - Валидационный день: **+2554%** (1559 сделок)  
  при комиссии 0.1% round-trip и порогах BUY ≥ 0.75, SELL ≤ 0.25.
- **Интеграция:** polling-воркер + FastAPI, поддержка спецификации 2.1.0 (READY/WARMUP, теневой режим `ml_shadow`).

---

## 🔄 Алгоритм работы воркера

1. **Запрос окна признаков**  
   Воркер отправляет GET-запрос к платформе (или мок-серверу) на эндпоинт `/api/ml/ds/feature-windows?readyOnly=true`. Получает JSON с матрицей признаков для READY-инструментов.

2. **Подготовка DataFrame**  
   Из полученных данных создаётся `pandas.DataFrame`, добавляются колонки `symbol`, `timestamp` (из `windowEndTimestamp`), `datetime` и фиктивная `session_key`.

3. **Вычисление признаков (функция `predict`)**  
   - Через `feature_pipeline.add_features` рассчитываются **22 базовых признака** (включая `rd_regime`, `rd_regime_transition`).  
   - Для 10 ключевых фичей по окнам [5, 15, 30, 60] добавляются rolling mean и std – ещё **80 признаков**.  
   - Итоговый набор (102 признака) масштабируется с помощью `scaler` из загруженного bundle модели.

4. **Получение вероятности и сигнала**  
   Модель возвращает вероятность класса BUY. Применяются пороги:
   - `proba ≥ 0.75` → BUY
   - `proba ≤ 0.25` → SELL
   - иначе → HOLD

5. **Логика удержания позиции**  
   - Если сигнал HOLD – ничего не отправляется.  
   - Если сигнал BUY или SELL, сравнивается с последним отправленным сигналом для этого символа.  
     - При совпадении – сигнал не отправляется (позиция удерживается).  
     - При изменении – формируется payload и отправляется POST-запрос на `/api/signals/ingest`.

6. **Отправка сигнала**  
   Payload содержит `symbol`, `timestamp`, `signal` ("BUY"/"SELL"), `price`, `rating` (вероятность соответствующего класса), `source`.  
   При успешном ответе (200) обновляется история последних сигналов.

7. **Пауза и повтор**  
   Воркер засыпает на заданный интервал (в демо – 5 сек, в проде – 60 сек) и переходит к шагу 1.

---

## 🚀 Быстрый старт

### 1️⃣ Клонирование репозитория
```bash
git clone https://github.com/your-team/trading-signals-ml.git
cd trading-signals-ml
```

### 2️⃣ Создание и активация виртуального окружения
```bash
python3 -m venv venv
source venv/bin/activate      # для Linux/macOS
# или venv\Scripts\activate   # для Windows
```

### 3️⃣ Установка зависимостей
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ Подготовка демо‑данных (один раз)

Скрипт `prepare_demo_data.py` создаёт JSON-файл с реальными историческими данными для выбранного символа.  
Убедитесь, что папка `data/raw/dataset_rework` содержит исходные CSV.

**Пример для символа SCRT (стандартный путь):**
```bash
python scripts/prepare_demo_data.py --symbol SCRT --output mock/data/demo_data.json
```

Если данные лежат в другом месте, укажите путь через `--data-dir`:
```bash
python scripts/prepare_demo_data.py --symbol SCRT --output mock/data/demo_data.json --data-dir /полный/путь/к/dataset_rework
```

Параметры:
- `--symbol` – тикер (например, `SCRT`, `AXS`).
- `--output` – путь для сохранения JSON (рекомендуется оставить `mock/data/demo_data.json`).
- `--max-rows` – максимальное количество строк (по умолчанию 1000).

После выполнения в `mock/data/` появится файл `demo_data.json` с полями `timestamp`, `open`, `high`, `low`, `close`, `volume`, `rd_value`, `symbol`.

### 5️⃣ Запуск мок‑сервера (эмулятор платформы)
**Терминал 1**
```bash
cd mock
npm install
node server.js
```
Сервер будет доступен на `http://localhost:3000`. **Не закрывайте этот терминал!**

### 6️⃣ Запуск polling‑воркера
**Терминал 2**
```bash
cd trading-signals-ml
source venv/bin/activate
cd integration
python worker.py
```
Воркер начнёт опрашивать мок‑сервер и отправлять сигналы.

### 7️⃣ (Опционально) Запуск FastAPI
**Терминал 3**
```bash
cd trading-signals-ml
source venv/bin/activate
uvicorn src.api.app:app --reload
```
Документация API: [http://localhost:8000/docs](http://localhost:8000/docs)

### 8️⃣ Запуск через Docker Compose
Если установлены Docker и Docker Compose:
```bash
docker-compose up --build
```
Будут запущены контейнеры `mock` (порт 3000), `worker` и `api` (порт 8000).

---

## ⚠️ Важные замечания
- Все команды, кроме `cd mock`, выполняются из корня репозитория (`trading-signals-ml/`).
- Виртуальное окружение должно быть активировано перед запуском любых Python‑скриптов.
- Мок‑сервер требует Node.js (проверьте установку: `node -v`).
- Для подготовки данных убедитесь, что папка `dataset_rework` содержит подпапки с датами (например, `2026-02-01/SCRT.csv`).

---

## 📚 Документация

Подробные исследования целевых переменных, фичей и экспериментов находятся в папке [`docs/`](docs/).  
Основные файлы:
- `01_TARGET_VARIABLES_RESEARCH.md` – обзор методов разметки.
- `02_TARGET_RESEARCH_CONCLUSIONS.md` – выбор `tp_sl_1_05`.
- `08_FEATURES_SOURCES_AND_RATIONALE.md` – описание базовых фичей и их отбор.
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