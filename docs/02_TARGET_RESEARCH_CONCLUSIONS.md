# Выводы исследования целевых переменных

Итоги объединенного исследования в `02_targets/03_Base_Model_And_Target_Comparison.ipynb`. Дата: 2026-03.

---

## 1. Финальный выбор

- **Target:** `tp_sl_1_05`
- **Baseline model:** `LightGBM`
- **Артефакты:**
  - `outputs/prepared_with_target_tp_sl_1_05.parquet`
  - `outputs/target_selection_summary.csv`
  - `outputs/baseline_lgbm_tp_sl_1_05.joblib`

---

## 2. Leakage и корректность target

- Корреляции target с `rd_regime`, `rd_mom_1`, `ret_1` не близки к 1.
- `rd_regime` используется как признак, а не как целевая.
- Для TP/SL разметки принято правило: **`ambiguous intrabar -> 0`** (если TP и SL касаются в одном баре).

---

## 3. Сравнение target/model (baseline)

Ключевой результат по `tp_sl_1_05`:

- **LightGBM:** `AUC ~ 0.705`, `F1 ~ 0.654`, `net_% ~ +2011`, `trades ~ 2768`

На сравнении target-кандидатов:

- `tb_vol_5bar` — выше net PnL, но значительно выше turnover
- `tp_sl_2_1` — ниже AUC и net PnL
- `tp_sl_1_05` — лучший компромисс качества и частоты сделок

---

## 4. Рекомендация

Для текущего production-baseline использовать:

1. `target = tp_sl_1_05`
2. `model = LightGBM`
3. HOLD-зону и комиссию при смене позиции в backtest
4. Мониторинг метрик (`AUC`, `precision`, `net_%`, `drawdown`) на новых периодах
