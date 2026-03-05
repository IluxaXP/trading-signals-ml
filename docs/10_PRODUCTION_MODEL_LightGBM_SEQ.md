## 16. Продовая модель LightGBM + SEQ (окна 5/15/30/60)

Этот документ фиксирует логику ноутбука `06_production/16_Production_Model_LightGBM.ipynb` и связь с экспериментами NB 13–15.

Цель: понять, **какую именно модель мы выводим в прод**, как считаются фичи/окна/сессии, как устроен бэктест и почему результаты честно отражают будущий запуск на реальных данных.

---

## 1. Данные, сессии и target

- Источник: `outputs/data_labeled_tp_sl_1_05.parquet`
- Target: `tp_sl_1_05` (TP=1%, SL=0.5%, H=20), колонка `target`
- Комиссия в бэктесте: `COMMISSION_RT = 0.001` (0.1% round-trip)
- Сессии:
  - уже подготовлены пайплайном (`prepare_for_training`)
  - каждая строка принадлежит `session_key` (непрерывный кусок минуток)
  - сессии **не склеиваются**: gap > 1.5 мин → новая сессия
  - сессии < 60 баров отфильтрованы ещё до экспериментов

В ноутбуке 16:

- для всех расчётов (ret_next, rolling) данные всегда группируются по `session_key` → границы сессий не пересекаются.

---

## 2. Базовые фичи (22) и связь с пайплайном

Список фичей берётся из `outputs/features_selected_tp_sl_1_05.txt` (результат `03_features/05_Feature_Selection.ipynb`):

- RD-группа: `rd_mom_1`, `rd_mom_5`, `rd_mom_10`, `rd_acceleration`, `rd_zscore_30`, `rd_ema_20`, `abs_rd`
- Price/tech: `ret_1`, `ret_5`, `rsi_14`, `macd_signal`, `macd_hist`, `volatility_14`
- Volume/OHLC: `volume_rel_20`, `body_ratio`, `close_position`
- Time: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`
- Regime: `rd_regime`, `rd_regime_transition`

Эти же фичи:

- считаются в `src/features/feature_pipeline.add_features()`
- используются и в offline (ноутбуки), и в online (API) — **единый пайплайн**.

---

## 3. Rolling sequence фичи и почему пересчитали логику

### 3.1. Какие фичи и окна

Ключевые фичи для rolling (10 штук):

- `rd_mom_1`, `rd_mom_5`, `rd_mom_10`, `rd_acceleration`, `rd_zscore_30`,
  `rd_ema_20`, `abs_rd`, `ret_1`, `ret_5`, `rsi_14`

Окна:

- 5, 15, 30, 60 баров

Агрегаты:

- mean, std

Итого rolling-фичей:

- 10 × 4 × 2 = **80**

Полный набор фичей модели:

- 22 базовых + 80 rolling = **102**

### 3.2. Как именно считается rolling (важный момент)

В исходной версии ноутбука 16 rolling считался **отдельно** в `train_df/val_df/test_df`.  
Это обрезало исторический контекст на границах сплитов и давало более слабый результат, чем в NB 15.

Сейчас логика переписана **под эксперименты NB 15**:

1. Формируем `valid`:
   - фильтруем по `target` и `ret_next`
   - считаем `ret_next` внутри `session_key`
   - добавляем `date`
2. Считаем rolling **на всём `valid`**:

   ```python
   grp = valid.groupby('session_key', group_keys=False)
   for w in SEQ_WINDOWS:  # [5, 15, 30, 60]
       for c in KEY_FEATS:  # 10 ключевых фичей
           valid[f'{c}_roll{w}_mean'] = grp[c].transform(lambda x: x.rolling(w, min_periods=1).mean())
           valid[f'{c}_roll{w}_std']  = grp[c].transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
   ```

3. **После этого** делаем temporal split:
   - train = дни 2026-02-01..2026-02-08
   - val   = 2026-02-09
   - test  = 2026-02-10

Это означает:

- фичи для val-дня используют всю историю train-дней внутри той же сессии
- фичи для test-дня используют всю историю train+val внутри сессии
- никакого заглядывания в будущее нет (в rolling используется только прошлое внутри `session_key`)

**Почему это правильно для прода:**

- в реальном онлайне у нас есть полная история по каждой монете до текущего бара;
- rolling-фичи на баре t опираются на все доступные точки ≤ t;
- в бэктесте мы должны имитировать именно такой сценарий, а не искусственно “обнулять” историю на границе train/val/test.

---

## 4. Обучение модели в ноутбуке 16

### 4.1. Split

Дни:

- train: 2026-02-01..2026-02-08 (8 дней)
- val:   2026-02-09
- test:  2026-02-10

Все срезы сортируются по `session_key` + `datetime` (или `timestamp`).

### 4.2. Масштабирование

- `StandardScaler` fit на train по всем 102 фичам
- `transform` на val/test

### 4.3. LightGBM-конфигурация

```python
model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)
```

Это тот же конфиг, который показал себя лучшим в NB 13 §9 на фичах `sequence_5_15_30_60`.

---

## 5. Бэктест: логика и связь с продом

### 5.1. Логика `signal_flip` (prod_hold)

В ноутбуке 16 используется та же логика, что:

- в продовом бэктесте (`04_models/08_Backtest_Profitability.ipynb`)
- в ноутбуках 13 и 15 под именем `backtest_prod_hold` / `signal_flip`

Правило:

- `proba >= threshold` → позиция +1 (long)
- `proba <= 1 - threshold` → позиция -1 (short)
- иначе → HOLD (оставляем предыдущую позицию)
- позиция сбрасывается в 0 при смене `session_key`
- комиссия:
  - считается только при смене позиции (`pos != pos_prev` и не оба нули)
  - каждая смена = `commission_rt / 2` (т.е. round-trip ~0.1%)

Реализация в ноутбуке 16:

```python
def backtest_pnl(proba, ret, session_ids, threshold=THRESHOLD, commission_rt=COMMISSION_RT):
    pred = np.where(proba >= threshold, 1, np.where(proba <= 1 - threshold, 0, -1))
    ...
    pos_changed = (pos != pos_prev) & ((pos != 0) | (pos_prev != 0))
    fee_total = pos_changed.sum() * (commission_rt / 2.0)
    pnl_net = (pos * ret).sum() - fee_total
    trades = int(pos_changed.sum())
    avg_trade = float((pnl_net * 100) / trades) if trades > 0 else np.nan
    return {'trades': trades, 'net_%': float(pnl_net * 100), 'avg_%_per_trade': avg_trade}
```

### 5.2. Почему это честно отражает будущую торговлю

1. **Только прошлое:** `ret_next` считается как `pct_change().shift(-1)` внутри `session_key`, rolling — только по прошлым барам.
2. **Сессии не склеиваются:** при смене `session_key` позиция сбрасывается, комиссия не берётся за “перескок”.
3. **Комиссия учтена:** 0.1% round-trip близко к Bybit taker+taker.
4. **Порог такой же, как в экспериментах:** 0.75/0.25 (band 25-75), где в NB 13 и 15 был лучший `avg_%_per_trade`.

---

## 6. Почему цифры немного отличаются от NB 15 (и почему это нормально)

Хотя мы максимально приблизили 16-й ноутбук к логике NB 15, небольшие отличия в цифрах возможны:

1. **Разные сценарии агрегации:**
   - NB 15 считает много конфигураций (разные модели, feature_set, логику, band’ы) и часто показывает **лучшие** строки,
   - NB 16 фиксирует **одну** конфигурацию (LightGBM + 102 фичи + band 25-75) и выводит её PnL целиком по дню.
2. **Технические детали train/val/test:**
   - порядок сортировки, отфильтрованные строки с `NaN` по фичам, особенности LightGBM с random_state могут давать небольшие расхождения в предсказаниях.
3. **Но в важных вещах ничего не меняется:**
   - форма распределения сделок,
   - уровни AUC,
   - порядок величины `net_%` и `avg_%_per_trade`.

Ключевое: **логика rolling, сессий и бэктеста теперь такая же, как в NB 15**, поэтому модель в 16-м — честный перенос champion-конфигурации из экспериментов в прод.

---

## 7. Итог нашей работы по 16-му шагу

1. Зафиксировали, что production‑baseline — это:
   - `target = tp_sl_1_05`
   - LightGBM + 22 базовых фичи + 80 rolling (5/15/30/60)
   - порог 0.75/0.25, логика `signal_flip`, комиссия 0.1% RT.
2. Привели расчёт rolling‑фичей в 16‑м ноутбуке к логике NB 15:
   - сначала считаем rolling на всём `valid` по `session_key`,
   - затем делаем temporal split 8/1/1.
3. Убедились, что бэктест в 16‑м:
   - использует ту же функцию, что и прод/эксперименты (комиссия, сессии, HOLD),
   - честно моделирует онлайн-торговлю.
4. Сохранили артефакт `models/prod_lgbm_seq.joblib`, полностью совместимый с API (`src/api/model_bundle.py`).

Следующий шаг для прода — перенести rolling‑фичи в online‑инференс (`inference_service.py`) и протестировать модель в shadow‑режиме (`source=ml_shadow`) на реальных данных биржи.

