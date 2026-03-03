# Эксперименты: текущий статус и порядок

Дата: 2026-03. Цель: развивать baseline `tp_sl_1_05 + LightGBM` и поддерживать воспроизводимость.

---

## Исторически выполненные шаги

### 10. Target: горизонт H и mult (triple-barrier)

**Ноутбук:** `10_Target_Horizon_Mult_Experiments.ipynb`

- Перебор H = 3, 4, 5, 6, 7 и mult = 1.0, 1.25, 1.5, 1.75, 2.0
- LightGBM: AUC и net PnL (full и proba>0.6) для каждой комбинации
- Результаты: `outputs/exp10_horizon_mult_results.csv`
- Лучшая конфигурация: `outputs/exp10_best_horizon_mult.joblib` (h, mult)

---

### 11. rd_regime_transition и гипотезы

**Ноутбук:** `11_rd_regime_transition_and_hypotheses.ipynb`

- Фича rd_regime_transition (момент смены режима)
- Сравнение: все фичи vs без rd_regime_transition
- Гипотеза: торговать только при transition=1 или только при transition=0
- Бектест по подвыборкам

---

### 12. Sequence-модель (окна 30, 60)

**Ноутбук:** `12_Sequence_Model_30_60.ipynb`

- Rolling mean/std по ключевым фичам для окон 30 и 60
- Избегаем LSTM (OOM на 12 GB)
- Сравнение baseline vs sequence-фичи по AUC и net PnL

---

### 13. Meta-labeling

**Ноутбук:** `13_Meta_Labeling_Experiments.ipynb`

- Meta-labeling: target = совпадение rd_regime с исходом triple-barrier
- Meta_target = 1 если rd_regime верно предсказал исход, -1 иначе
- Сравнение tb_vol vs meta_target

---

## Текущий порядок запуска (актуальный)

1. `02_targets/03_Base_Model_And_Target_Comparison.ipynb` — финальный target/model, сохранение артефактов
2. `03_features/04_Data_Labeling_And_Feature_Loading.ipynb` — подготовка `data_labeled_tp_sl_1_05.parquet`
3. `03_features/05_Correlation_Analysis.ipynb`
4. `03_features/06_Feature_Selection.ipynb` (включает leakage-check и ablation)
5. `03_features/07_Scaling_And_Normalization.ipynb`
6. `04_models/09_Model_Training_And_Analysis.ipynb` — обучение и сохранение лучшей модели
7. Дальнейшие эксперименты (robustness/meta-labeling/sequence) — только после фиксации baseline

---

## Восстановление хода действий

Все эксперименты записаны в отдельных ноутбуках. При необходимости можно:

- Воспроизвести любой шаг, запустив соответствующий ноутбук
- Использовать актуальные артефакты baseline:
  - `outputs/prepared_with_target_tp_sl_1_05.parquet`
  - `outputs/baseline_lgbm_tp_sl_1_05.joblib`
  - `outputs/target_selection_summary.csv`
  - `outputs/features_selected_tp_sl_1_05.txt`
