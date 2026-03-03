# Фичи: источники, статус и обоснование

Актуально для финального target `tp_sl_1_05` (колонка `target`) и baseline `LightGBM`.

---

## 1. Статус фичей (после quick-pass 03_features)

| Статус | Группа/фичи | Комментарий |
|--------|-------------|-------------|
| **core** | `rd_mom_1`, `rd_mom_5`, `rd_mom_10`, `rd_zscore_30`, `rd_acceleration` | Стабильно информативны в quick-pass + ablation |
| **conditional** | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` | Могут давать прирост/шум в зависимости от сплита и режима |
| **experimental** | `rd_regime`, `rd_regime_transition` | Использовать как режимные признаки, регулярно перепроверять leakage/robustness |
| **supporting** | `ret_*`, `volatility_14`, `rsi_14`, `macd_signal`, `macd_hist`, `volume_rel_20`, `body_ratio`, `close_position` | Классические признаки, роль зависит от модели и периода |
| **excluded** | `is_weekend`, `bb_position`, `bb_width` | Дублируют информацию (корреляции 05): `is_weekend` ↔ `dow_sin/cos`, `bb_position` ↔ `rsi_14/rd_zscore_30`, `bb_width` ↔ `volatility_14` |
| **excluded** | `macd_line` | Мультиколлинеарность с `macd_signal` (\|r\| > 0.85, авто в 06) |
| **excluded** | `symbol_encoded` | AUC ≈ random, importance 0.3%, ablation без него чуть лучше. Убирает зависимость от LabelEncoder в API |

---

## 2. Что показал анализ и ablation

Проверки из `03_features/06_Feature_Selection.ipynb` (leakage-check и ablation):

- `drop_rd` → резкая просадка AUC → RD-группа = ядро (`core`).
- `drop_symbol` → AUC чуть выше (0.6151 vs 0.6142) → `symbol_encoded` исключён.
- `drop_time` → AUC без изменений → time-фичи оставлены как `conditional`.
- Корреляционный анализ (05): `is_weekend`, `bb_position`, `bb_width` дублируют информацию → исключены.
- Мультиколлинеарность (06, |r|>0.85): `macd_line` → исключён.

### Про ликвидность и объём

- Из объёмных фичей используется `volume_rel_20` (объём / MA20).
- Градация по ликвидности/капитализации **не добавляется**: 214 символов в датасете, `symbol_encoded` не значим → модель работает универсально.
- Пересмотреть при расширении пула (50+ инструментов) или деградации на конкретных монетах.

---

## 3. Источники и расчёт признаков

### 3.1 RD (собственные)

Источник: `rd_value` из `dataset_rework`.

- `rd_mom_1`, `rd_mom_5`, `rd_mom_10`
- `rd_acceleration`
- `rd_zscore_30`
- `rd_ema_20`
- `abs_rd`

Назначение: режим, импульс и скорость изменения режима.

### 3.2 Price/tech/volume/time/symbol

- Price: `ret_1`, `ret_5`, `volatility_14`
- Tech: `rsi_14`, `macd_signal`, `macd_hist` (исключены: `macd_line`, `bb_width`, `bb_position`)
- Volume/OHLC: `volume_rel_20`, `body_ratio`, `close_position`
- Time: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` (исключён: `is_weekend`)
- Symbol: исключён `symbol_encoded` — не несёт информации, упрощает API

---

## 4. Важные оговорки

- Низкая линейная корреляция не означает «бесполезно» для нелинейной модели.
- Высокая importance не гарантирует устойчивость на другом периоде/символах.
- Для `rd_regime`/`rd_mom_*` обязателен регулярный leakage-check при смене target.
- При расширении пула монет → пересмотреть необходимость фичей ликвидности/капитализации.

---

## 5. Текущие артефакты 03_features

- `outputs/correlation_with_target_tp_sl_1_05.csv` — корреляции всех 27 фичей с target
- `outputs/excluded_features_tp_sl_1_05.txt` — ручные исключения из 05
- `outputs/features_selected_tp_sl_1_05.txt` — **единственный источник правды** для 07, обучения и API
- `models/scaler_tp_sl_1_05.joblib` — scaler + features + target для inference

---

## 6. Где считаются фичи

Основной модуль: `src/features/feature_pipeline.py`  
Расчёт ведется внутри `session_key`, без пересечения границ сессий.
