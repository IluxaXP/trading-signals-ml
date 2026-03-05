# Итоги пайплайна: фичи и модели для tp_sl_1_05

Результаты выполнения плана `04_FEATURE_MODEL_PIPELINE_PLAN.md`. Дата: 2026-03-04.

---

## 1. Резюме шагов

| Шаг | Ноутбук | Артефакт |
|-----|---------|----------|
| 1 | 01_Load_And_Prepare_Data | `outputs/prepared_with_rd_regime.parquet` |
| 2 | 03_Data_Labeling_And_Feature_Loading | `outputs/data_labeled_tp_sl_1_05.parquet` |
| 3 | 04_Correlation_Analysis | `outputs/correlation_with_target_tp_sl_1_05.csv` |
| 4 | 05_Feature_Selection | `outputs/features_selected_tp_sl_1_05.txt` |
| 5 | 06_Scaling_And_Normalization | `models/scaler_tp_sl_1_05.joblib` |
| 6 | 07_Model_Training_And_Analysis | `models/best_model_tp_sl_1_05.joblib` |
| 7 | 08_Backtest_Profitability | Метрики PnL |
| 8 | Эксперименты 09–15 | Rolling-фичи, пороги, логики сделок |
| **9** | **16_Production_Model_LightGBM** | **`models/prod_lgbm_seq.joblib`** |

---

## 2. Фичи

### 2.1. Базовые фичи (22)

По quick-pass (ablation в `03_features/05_Feature_Selection.ipynb`):

- **Core:** `rd_mom_1`, `rd_mom_5`, `rd_mom_10`, `rd_zscore_30`, `rd_acceleration`
- **Supporting:** `rd_ema_20`, `abs_rd`, `ret_1`, `ret_5`, `rsi_14`, `macd_signal`, `macd_hist`, `volatility_14`, `volume_rel_20`, `body_ratio`, `close_position`
- **Conditional:** `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`
- **Experimental:** `rd_regime`, `rd_regime_transition`

Артефакт: `outputs/features_selected_tp_sl_1_05.txt` — единственный источник правды.

### 2.2. Rolling sequence фичи (80)

Добавлены по результатам экспериментов NB 11, NB 13:

- 10 ключевых фичей (`rd_mom_1` … `rsi_14`) × 4 окна (5, 15, 30, 60) × 2 агрегата (mean, std)
- Вычисляются внутри `session_key` через `rolling(w, min_periods=1)`
- Итого: 22 + 80 = **102 фичи** в продовой модели

Подробности: `docs/08_FEATURES_SOURCES_AND_RATIONALE.md`.

---

## 3. Модели и метрики

### 3.1. Старый baseline (target=tp_sl_1_05, 22 фичи)

| Модель | AUC test | Net % | Trades |
|--------|----------|-------|--------|
| Dummy | ~0.50 | — | — |
| Rule-based | ~0.53 | ~-25 | ~4827 |
| CatBoost | ~0.70 | ~+1926 | ~2224 |
| LightGBM | ~0.705 | ~+2011 | ~2768 |

### 3.2. Новая продовая модель (102 фичи, порог 0.75)

- **Модель:** LightGBM (n_estimators=300, max_depth=6, lr=0.05)
- **Фичи:** 102 (22 base + 80 rolling 5/15/30/60)
- **Порог:** 0.75 / 0.25 (band 25-75)
- **Торговая логика:** signal_flip (HOLD сохраняет позицию)
- **Комиссия:** 0.1% round-trip
- **Артефакт:** `models/prod_lgbm_seq.joblib`

Метрики AUC val/test и бэктеста — см. `06_production/16_Production_Model_LightGBM.ipynb`.

---

## 4. Выбор конфигурации (обоснование)

По результатам экспериментов NB 13, 14, 15:

| Решение | Почему | Где проверено |
|---------|--------|---------------|
| LightGBM | Лучший AUC + avg/trade среди всех моделей | NB 13 §9 |
| Rolling 5/15/30/60 | Лучше 30/60 по avg/trade | NB 13 §9, NB 11 |
| Порог 25-75 | Оптимальный баланс сделок и прибыли | NB 13, NB 15 |
| signal_flip | Устойчивее exit_on_05 | NB 13, NB 15 |
| Глубокие сети отвергнуты | Не превзошли LightGBM при комиссии | NB 14 |

---

## 5. Рекомендации для продакшена

1. **Target:** `tp_sl_1_05` (`ambiguous intrabar -> 0`)
2. **Фичи:** 102 (22 базовых + 80 rolling)
3. **Модель:** LightGBM — `models/prod_lgbm_seq.joblib`
4. **Порог:** 0.75 / 0.25 (BUY/SELL), HOLD сохраняет позицию
5. **Контроль:** OOT-сплиты, мониторинг avg/trade при накоплении данных

---

## 6. Пайплайн фичей (API-ready)

- `src/features/feature_pipeline.add_features()` считает 22 базовых фичи:
  - если `rd_regime`/`rd_regime_transition` отсутствуют — автоматический фолбэк;
  - если `signal_barrier` есть — используется для `rd_regime`.
- Rolling-фичи (80) пока считаются только в ноутбуках. **TODO:** перенести в `inference_service.py` для онлайн-инференса.
- Warmup 60 баров через `src/features/warmup_loader.py`.

---

## 7. API шаг (MVP+)

- `src/api/` — FastAPI endpoints: `/health`, `/predict`.
- Worker poller: `feature-windows -> infer -> signals/ingest`.
- Retry policy: `5xx` с backoff, `400` без retry.
- HOLD не отправляется в ingest.
- **Спецификация:** `00_contract/ml_api_integration_spec_04-03-2026.md` v2.2.0 (AUTH-02).
- Документация: `docs/09_API_STEP_WORKER_AND_SERVICE.md`.
