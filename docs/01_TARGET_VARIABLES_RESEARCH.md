# Обзор целевых переменных для финансового ML

Документ описывает методы разметки временных рядов в торговых системах и предлагаемые target-кандидаты для dataset_rework.

---

## 1. Методы разметки (Labeling)

### 1.1 Fixed-Horizon Return

**Идея:** Метка = знак доходности за фиксированный горизонт H.

- `ret[t] = close[t+H] / close[t] - 1`
- `label = sign(ret[t])` или `label = 1 if ret > τ else (-1 if ret < -τ else 0)`

**Плюсы:** Простота, интерпретируемость.  
**Минусы:** Фиксированный порог τ может давать слишком много/мало сигналов; не учитывает волатильность.

**Улучшение:** Динамический порог τ = k × volatility (адаптация к волатильности).

---

### 1.2 Triple-Barrier Method

**Идея:** Три барьера: Take Profit (TP), Stop Loss (SL), Time (T). Метка определяется тем, какой барьер первым достигнут.

- Верхний барьер: `close[t] × (1 + TP)`
- Нижний барьер: `close[t] × (1 - SL)`
- Временной барьер: H баров

**Плюсы:** Учитывает риск и время; TP/SL можно задать волатильностью.  
**Минусы:** Сложнее реализация; параметры TP/SL/T влияют на распределение.

**Источники:** Lopez de Prado (Advances in Financial ML), mlfinlab.

---

### 1.3 Trend-Scanning Labels

**Идея:** Регрессия тренда в окне L, метка = знак наклона.

- `slope = linear_regression(close[t:t+L])`
- `label = sign(slope)`

**Плюсы:** Улавливает тренд; не требует TP/SL.  
**Минусы:** Чувствителен к шуму; L — гиперпараметр.

---

### 1.4 Meta-Labeling

**Идея:** Вторичная модель фильтрует сигналы первичной модели (размер позиции, подтверждение).

**Применение:** После выбора primary target — дополнительная модель для фильтрации.

---

## 2. Семантика rd_regime vs signal_barrier

В dataset_rework колонка `signal_barrier` имеет **другую семантику**, чем в старом ml_dataset:

| Источник | signal_barrier | Семантика |
|----------|----------------|-----------|
| ml_dataset (старый) | Triple-barrier | 1=BUY, -1=SELL, 0=HOLD |
| dataset_rework | sign(rd_value.diff(1)) | 1=long (RD растёт), -1=short (RD падает) |

**Вывод:** В dataset_rework `signal_barrier` ≈ `sign(rd_mom_1)` — это **режим RD**, а не идеальная точка входа.

**Решение:** Переименовать в `rd_regime` и использовать как **фичу**, а не как target. Целевую переменную создаём сами из OHLC.

---

## 3. Target-кандидаты для dataset_rework

### 3.1 tb_vol_5bar (Triple-Barrier, volatility-adjusted)

**Формула:**
- `vol = volatility_14` (rolling std от pct_change)
- `TP = SL = 1.5 × vol`, `T = 5` баров
- Верхний барьер: `close[t] × (1 + TP)`
- Нижний барьер: `close[t] × (1 - SL)`
- Смотрим forward: `high[t:t+5]`, `low[t:t+5]`
- Первый достигнутый: TP → 1, SL → -1, else → 0 (HOLD)

**Параметры:** TP=SL=1.5×vol_14, horizon=5

**Риск leakage:** Нет — используем только OHLC и volatility (от цены). rd_value не участвует.

---

### 3.2 ret_h_tau (Fixed-Horizon, volatility-adjusted threshold)

**Формула:**
- `ret = close[t+H] / close[t] - 1`, H=5
- `τ = 0.5 × volatility_14`
- `label = sign(ret)` если |ret| > τ, иначе 0 (HOLD)

**Параметры:** H=5, τ=0.5×vol_14

**Риск leakage:** Нет — только OHLC.

---

### 3.3 trend_scan (Trend-Scanning)

**Формула:**
- Окно L=5: `close[t], close[t+1], ..., close[t+L-1]`
- Линейная регрессия: `slope = coef` линейной модели
- `label = sign(slope)` (1=BUY, -1=SELL)

**Параметры:** L=5

**Риск leakage:** Нет — только OHLC.

---

### 3.4 tp_sl_2_1 и tp_sl_1_05 (fixed % TP/SL)

Фиксированные барьеры в стиле nikitapre.

- `tp_sl_2_1`: TP=2%, SL=1%, H=20
- `tp_sl_1_05`: TP=1%, SL=0.5%, H=20

Правило для неоднозначности внутри бара:

- если в одном баре одновременно касаются TP и SL, то метка = `0` (`ambiguous intrabar -> 0`)

Этот target-класс используется в актуальном baseline:

- финальный выбор: `tp_sl_1_05`
- артефакт: `outputs/prepared_with_target_tp_sl_1_05.parquet`

---

## 4. Актуальный выбор target (production baseline)

По итогам `02_targets/03_Base_Model_And_Target_Comparison.ipynb`:

1. `target = tp_sl_1_05`
2. `model = LightGBM`
3. HOLД-зона в backtest и комиссия при смене позиции

Исторические target (`tb_vol_5bar`, `ret_h_tau`, `trend_scan`) сохраняются для сравнительных экспериментов.

---

## 5. Sources

- [mlfinlab](https://github.com/hudson-and-thames/mlfinlab) — Triple Barrier, Trend Scanning
- [RiskLab AI](https://risklab.ai/) — Lopez de Prado, Advances in Financial ML
- [Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) — M. Lopez de Prado
- [Triple-Barrier Method](https://mlfinlab.readthedocs.io/en/latest/labeling/tb_meta_labeling.html) — mlfinlab docs
