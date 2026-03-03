# Итоги пайплайна: фичи и модели для tp_sl_1_05

Результаты выполнения плана `04_FEATURE_MODEL_PIPELINE_PLAN.md`. Дата: 2026-03.

---

## 1. Резюме шагов

| Шаг | Ноутбук | Артефакт |
|-----|---------|----------|
| 2 | 04_Data_Labeling_And_Feature_Loading | outputs/data_labeled_tp_sl_1_05.parquet |
| 3 | 05_Correlation_Analysis | outputs/correlation_with_target_tp_sl_1_05.csv |
| 4 | 06_Feature_Selection | outputs/features_selected_tp_sl_1_05.txt |
| 5 | 07_Scaling_And_Normalization | models/scaler_tp_sl_1_05.joblib |
| 6 | 09_Model_Training_And_Analysis | models/best_model.joblib |
| 7 | 09_Backtest_Profitability | Метрики PnL |

---

## 2. Фичи: quick-pass статус

По quick-pass (ablation в `03_features/06_Feature_Selection.ipynb`):

- **Core:** `rd_mom_1`, `rd_mom_5`, `rd_mom_10`, `rd_zscore_30`, `rd_acceleration`
- **Experimental:** `rd_regime`, `rd_regime_transition`
- **Conditional:** time- и symbol-фичи (по ablation дают небольшой чувствительный эффект)

Артефакт: `outputs/features_selected_tp_sl_1_05.txt` — единственный источник правды для 07, обучения и API.

---

## 3. Лучшая модель и метрики (target=tp_sl_1_05)

**Модель:** LightGBM (AUC test около `0.705`).

- Dummy: AUC ~0.50
- Rule-based: AUC ~0.53
- CatBoost: AUC ~0.70
- LightGBM: AUC ~0.705 (лучшая)

Confusion matrix, ROC-curves — см. 09_Model_Training_And_Analysis.

---

## 4. Результаты бектеста (HOLD-зона, комиссия при смене позиции)

| Target | Model | Net % | Trades |
|--------|-------|-------|--------|
| tp_sl_1_05 | LightGBM | ~+2011 | ~2768 |
| tp_sl_1_05 | CatBoost | ~+1926 | ~2224 |
| tp_sl_1_05 | Rule-based | ~-25 | ~4827 |

**Вывод:** `tp_sl_1_05 + LightGBM` остаётся текущим baseline по сочетанию качества и turnover.

---

## 5. Рекомендации для хакатона/продакшена

1. **Target:** `tp_sl_1_05` (`ambiguous intrabar -> 0`)
2. **Фичи:** `outputs/features_selected_tp_sl_1_05.txt`
3. **Модель:** LightGBM — `outputs/baseline_lgbm_tp_sl_1_05.joblib`
4. **Scaling:** `models/scaler_tp_sl_1_05.joblib`
5. **Robustness:** контролировать OOT-группами, time/symbol считать conditional
