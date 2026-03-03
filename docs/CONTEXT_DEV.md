# Контекст для разработки: features → model

Сжатая сводка для продолжения работы (когда контекст обновился).  
**Каноническая версия** — в `docs/`. Дата: 2026-03.

---

## Структура проекта

| Папка | Назначение |
|-------|------------|
| `src/data/` | Загрузчики: `dataset_rework_loader`, `data_prep_dataset_rework` |
| `src/features/` | `feature_pipeline`, `warmup_loader` |
| `01_data_prep/` | EDA, гипотезы, загрузка и подготовка данных |
| `02_targets/` | Исследование целевых переменных |
| `03_features/` | Labeling, корреляции, отбор фичей, scaling |
| `04_models/` | Обучение моделей |
| `05_experiments/` | Эксперименты (target, horizon, mult) |
| `00_contract/` | ТЗ, спецификация API |
| `docs/` | Выводы, планы, контексты |

---

## Данные (dataset_rework)

- **Загрузчик:** `src.data.data_prep_dataset_rework.load_prepared()` (использует `dataset_rework_loader`)
- **Формат:** `YYYY-MM-DD(-N)/SYMBOL.csv`; колонки: timestamp, symbol, rd_value, OHLC, volume, signal_barrier
- **Target (рекомендуемый):** `tp_sl_1_05` (fixed TP/SL, H=20).  
  Финальный parquet: `outputs/prepared_with_target_tp_sl_1_05.parquet`.  
  Альтернативы и история: см. `docs/07_TARGET_TP_SL_AND_STOCHASTIC.md`
- **Фильтры перед обучением:**
  - `time_diff_min <= 1.5` (убрать строки после больших разрывов)
  - Сессии < 60 точек — отбросить
  - Sequences строить **только внутри сессии** (gap > 1.5 мин = новая сессия)

---

## Ключевые решения (зафиксированы)

1. Сессия = непрерывный блок (gap ≤ 1.5 мин); граница сессии при gap > 1.5 мин
2. Минимум 60 точек на сессию для обучения
3. Не склеивать данные через границу сессии
4. Добавить symbol как фичу (категориальный/encoded)
5. Нет HOLD в dataset_rework; BUY↔SELL — переходы. HOLD в target-ноутбуке появляется как timeout/ambiguous (`intrabar TP+SL -> 0`)
6. EMA в данных нет — фичи только из rd_value + OHLC + volume
7. Warmup 60 баров — для корректных RSI/MACD при inference

---

## Этапы дальше

1. **Feature pipeline** — `src/features/feature_pipeline.py`. Подробности и статус фичей: `docs/08_FEATURES_SOURCES_AND_RATIONALE.md`.
2. **Session-based sequences** — собирать X/y только внутри сессий, без пересечения границ
3. **Time split** — train/val/test по времени; leakage check
4. **Модель baseline** — LightGBM (`outputs/baseline_lgbm_tp_sl_1_05.joblib`), вход: selected features, выход: buy-probability
5. **API** — POST /predict, cold-start guard (00_contract/ml_api_integration_spec.md)

---

## Ссылки

- EDA: `01_data_prep/01_Feature_Engineering_and_Hypotheses.ipynb`, `01_data_prep/02_Load_And_Prepare_Data.ipynb`
- Feature selection (leakage/ablation): `03_features/06_Feature_Selection.ipynb`
- Выводы EDA: `docs/Readme_EDA.md`
- Target research: `docs/02_TARGET_RESEARCH_CONCLUSIONS.md`, `docs/07_TARGET_TP_SL_AND_STOCHASTIC.md`
- Pipeline: `docs/05_PIPELINE_CONCLUSIONS.md`
- ТЗ: `00_contract/ml_api_integration_spec.md`
