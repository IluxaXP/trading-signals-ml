# Snapshot: текущие выводы по target/model

Сжатая версия текущего статуса.  
**Дата snapshot:** 2026-03-03.

---

## 1. Актуальный baseline

- `target = tp_sl_1_05`
- `model = LightGBM`
- ключевые артефакты:
  - `outputs/prepared_with_target_tp_sl_1_05.parquet`
  - `outputs/target_selection_summary.csv`
  - `outputs/baseline_lgbm_tp_sl_1_05.joblib`

---

## 2. Проверка корректности target

- Leakage-проверки не выявили прямой утечки.
- Для intrabar-случаев TP/SL используется правило: **`ambiguous -> 0`**.
- Масштаб ambiguous first-hit:
  - `tp_sl_2_1`: около `0.1%`
  - `tp_sl_1_05`: около `0.5-0.6%`

---

## 3. Метрики (ключевой срез)

Для `tp_sl_1_05 + LightGBM`:

- `AUC ~ 0.705`
- `F1 ~ 0.654`
- `net_% ~ +2011`
- `trades ~ 2768`

---

## 4. Отбор фичей (итог 05–06)

- Полный набор: 27 фичей (25 из `feature_pipeline` + `rd_regime` + `rd_regime_transition`).
- Исключены (05, корреляции): `is_weekend`, `bb_position`, `bb_width`.
- Исключены (06, мультиколлинеарность): `macd_line`.
- Исключены (06, анализ): `symbol_encoded` — AUC ≈ random, importance 0.3%, упрощает API.
- **Финальный набор:** `features_selected_tp_sl_1_05.txt` — единственный источник правды.
- Ликвидность/капитализация: не добавляется на текущем этапе (214 символов, `symbol_encoded` не значим → модель универсальна).

## 5. Что важно дальше

1. Поддерживать `03_features` на едином входе `prepared_with_target_tp_sl_1_05.parquet`
2. Контролировать robustness по OOT/сессионным сплитам
3. Актуализировать docs и артефакты при каждом пересчете target/model
4. При расширении пула монет — пересмотреть фичи ликвидности
