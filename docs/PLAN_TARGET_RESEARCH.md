# План: Исследование целевых переменных с документацией

> Сохранённый план работ по подготовке к хакатону. Структурирован для быстрого восстановления контекста.

---

## Обзор

Провести исследование целевых переменных для ML: загрузить dataset_rework, создать rd_regime/rd_regime_transition, реализовать 2–3 target-кандидата (triple-barrier, fixed-horizon return, trend-scanning), проверить leakage, сравнить по дисбалансу/частоте/PnL и построить baseline-модель. Всё в ноутбуках с полной документацией в MD.

---

## Контекст

- **Источник данных:** `dataset_rework` (OHLCV + rd_value + signal_barrier)
- **signal_barrier** в dataset_rework = sign(rd.diff(1)) — переименовываем в **rd_regime** (фича)
- **Целевые переменные** создаём сами из OHLC, без утечки rd_value
- **Наработки fork:** `fork/feature_pipeline.py`, `fork/data_prep_dataset_rework.py`, `fork/dataset_rework_loader.py`, `fork/ML_Model_Test.ipynb` (логика PnL, backtest)

---

## Шаги выполнения

### Шаг 1. Документация: обзор целевых переменных

**Файл:** `docs/01_TARGET_VARIABLES_RESEARCH.md`

Содержание:
- Краткий обзор методов разметки (fixed-horizon, triple-barrier, trend-scanning, meta-labeling)
- Семантика rd_regime vs signal_barrier
- Описание 3 target-кандидатов: формулы, параметры, риски leakage
- Ссылки на источники (mlfinlab, RiskLab AI, статьи)

---

### Шаг 2. Ноутбук: загрузка и подготовка данных

**Файл:** `01_Load_And_Prepare_Data.ipynb`

- Импорт из fork: `dataset_rework_loader`, `data_prep_dataset_rework`
- Загрузка dataset_rework, подготовка сессий (session_key)
- Переименование `signal_barrier` → `rd_regime`
- Добавление `rd_regime_transition` (смена -1↔1)
- Добавление фичей через `feature_pipeline.add_features` (без rd_mom_1 при обучении — leakage)
- Сохранение prepared DataFrame для следующих ноутбуков (опционально parquet/cache)

---

### Шаг 3. Ноутбук: целевые переменные и проверка leakage

**Файл:** `02_Target_Variants_And_Leakage_Check.ipynb`

**3.1. Реализация target-кандидатов (внутри session_key):**

| Target      | Формула                                                       | Параметры              |
|-------------|---------------------------------------------------------------|------------------------|
| `tb_vol_5bar` | Triple-barrier: TP/SL по волатильности, horizon=5             | TP=SL=1.5×vol_14, time=5 |
| `ret_h_tau` | sign(close[t+H]/close[t] - 1) если \|ret\| > τ                | H=5, τ=0.5×vol_14      |
| `trend_scan` | Регрессия тренда в окне L, sign(slope)                        | L=5                    |

**3.2. Проверка leakage:**
- Корреляция target с rd_regime, rd_mom_1, ret_1 — не должна быть 1:1
- Таблица: target vs rd_regime (confusion-like)

**3.3. Сравнение:**
- Распределение классов (дисбаланс)
- Частота сигналов (доля BUY/SELL)
- PnL-after-fee (0.1%) по каждому target (наивная стратегия: торгуем по target)

---

### Шаг 4. Ноутбук: baseline-модель

**Файл:** `03_Baseline_Model_Comparison.ipynb`

- Выбор лучшего target по шагу 3
- Train/val/test split по session_key (как в `fork/ML_Model_Test.ipynb`)
- Модели: Dummy, LogisticRegression, XGBoost, LightGBM
- Фичи: `get_feature_columns()` без rd_mom_1
- Метрики: AUC, F1, PnL backtest
- Вывод: какой target даёт лучший баланс обучаемости и PnL

---

### Шаг 5. Итоговая документация

**Файл:** `docs/02_TARGET_RESEARCH_CONCLUSIONS.md`

- Результаты leakage check
- Сравнительная таблица target по дисбалансу, частоте, PnL
- Рекомендация по целевому target для хакатона
- Следующие шаги (дообучение, API, warmup)

---

## Структура файлов

```
c:\project\trading_bot_2Engine\
├── docs/
│   ├── PLAN_TARGET_RESEARCH.md           # этот план
│   ├── 01_TARGET_VARIABLES_RESEARCH.md   # Шаг 1
│   └── 02_TARGET_RESEARCH_CONCLUSIONS.md # Шаг 5
├── 01_Load_And_Prepare_Data.ipynb        # Шаг 2
├── 02_Target_Variants_And_Leakage_Check.ipynb  # Шаг 3
├── 03_Baseline_Model_Comparison.ipynb    # Шаг 4
├── ml_api_integration_spec.md
└── fork/                                 # без изменений
```

---

## Зависимости от fork

- `fork/dataset_rework_loader.load_dataset_rework`
- `fork/data_prep_dataset_rework.prepare_for_training`, `load_prepared`
- `fork/feature_pipeline.add_features`, `get_feature_columns`, `FEATURE_COLS`
- Логика backtest из `fork/ML_Model_Test.ipynb` (backtest_pnl, COMMISSION=0.001)

---

## Порядок выполнения

1. Создать `docs/` и `01_TARGET_VARIABLES_RESEARCH.md`
2. Создать `01_Load_And_Prepare_Data.ipynb`, выполнить
3. Создать `02_Target_Variants_And_Leakage_Check.ipynb`, выполнить
4. Создать `03_Baseline_Model_Comparison.ipynb`, выполнить
5. Заполнить `02_TARGET_RESEARCH_CONCLUSIONS.md` по результатам
