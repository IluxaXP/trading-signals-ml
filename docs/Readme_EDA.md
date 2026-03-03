# EDA: dataset_rework — результаты изучения данных

Документ суммирует всё, что изучено в EDA-ноутбуках (`01_data_prep/01_Feature_Engineering_and_Hypotheses.ipynb`, `01_data_prep/02_Load_And_Prepare_Data.ipynb`).

---

## 1. Источник данных

- **Папка:** `dataset_rework/`
- **Структура:** вложенные папки `YYYY-MM-DD(-N)/SYMBOL.csv`
- **Загрузчик:** `src/data/dataset_rework_loader.py`, `src/data/data_prep_dataset_rework.py` (load_prepared)
- **Колонки:** timestamp, symbol, rd_value, open, high, low, close (→ close_price), volume, signal_barrier
- **Без EMA:** fast_ema, slow_ema, signal_ema отсутствуют

---

## 2. Объём и качество

- **Строк:** ~395 585 (после базовой загрузки)
- **Символов:** 223
- **Период:** 2026-02-01 → 2026-02-10
- **Пропуски:** 0% по ключевым колонкам
- **Дубликаты (symbol, timestamp):** 0

---

## 3. Target в сырых данных: signal_barrier

- **HOLD убран:** в dataset_rework только BUY (1) и SELL (-1)
- **Логика:** переход BUY→SELL = продажа; переход SELL→BUY = покупка
- **Баланс классов:** ~50% BUY, ~50% SELL (примерно сбалансировано)

**Рекомендуемая целевая для обучения:** `tp_sl_1_05` (fixed TP/SL, финальный выбор в 02_targets).  
Артефакт: `outputs/prepared_with_target_tp_sl_1_05.parquet`.  
См. `docs/02_TARGET_RESEARCH_CONCLUSIONS.md`.

---

## 4. Фильтрация для модели

### 4.1 Интервалы > 1.5 мин

- Удаляем строки, где `time_diff_min > 1.5` (после большого разрыва)
- **Удалено:** ~4 879 строк (1.23%)
- **Остаётся:** ~390 706 строк (98.77%)

### 4.2 Сессии и непрерывность

- **Сессия** = непрерывный блок: gap между строками ≤ 1.5 мин
- Модель не видит разрывы — склеивание через gap даёт ложную непрерывность

**Статистика по сессиям:**
- Всего сессий: ~2 788
- Сессий < 60 точек (отбрасываем): ~1 740
- Сессий ≥ 60 точек: ~1 048 (365 375 строк, ~93.5% данных)

---

## 5. Выводы для pipeline обучения

1. **Сессия** = непрерывный блок: gap между строками ≤ 1.5 мин
2. Обучать модель **только на сессиях длиной ≥ 60 точек**
3. Строить sequences **только внутри сессии**
4. Модель не видит разрывы; склеивание через gap — ложная непрерывность

---

## 6. Признаки для модели

- **Простые сильные (quick-pass):** rd_mom_1, rd_mom_5, rd_mom_10, rd_zscore_30, rd_acceleration
- **Conditional:** time-фичи и symbol_encoded (проверять на устойчивость по OOT/grouped split)
- Производные полезнее уровней: rd_momentum, returns, abs(rd_value)

---

## 7. Cold start

- `rd_value=0` (холодный старт) в текущих данных практически нет
- Логику cold start **оставить** — в проде/API может появиться

---

## 8. Расположение компонентов

| Компонент | Путь |
|-----------|------|
| Загрузчик | `src/data/dataset_rework_loader.py`, `src/data/data_prep_dataset_rework.py` |
| Feature pipeline | `src/features/feature_pipeline.py` |
| Warmup loader | `src/features/warmup_loader.py` |
| EDA / гипотезы | `01_data_prep/` |
| Feature selection (leakage/ablation) | `03_features/06_Feature_Selection.ipynb` |
