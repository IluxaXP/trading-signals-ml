# План экспериментов для улучшения PnL (после фиксации tp_sl_1_05)

Дата: 2026-03-01

## Цели

1. Улучшить устойчивость baseline `tp_sl_1_05 + LightGBM` на новых периодах
2. Подготовить базу для сложных моделей (sequence/LSTM)
3. Документировать каждый шаг в отдельном ноутбуке для воспроизводимости

---

## Шаг 1: Robustness baseline на новых срезах

**Ноутбук:** `10_Target_Horizon_Mult_Experiments.ipynb` (исторический), затем отдельный robust-pass

| Параметр | Варианты |
|----------|----------|
| H (horizon) | 3, 4, 5, 6, 7 |
| mult (TP/SL) | 1.0, 1.25, 1.5, 1.75, 2.0 |

Для актуального baseline:
- OOT проверка по периодам
- Grouped проверка по symbol/session
- Контроль drift по классам target и ключевым фичам

**Гипотеза:** основной риск теперь не в выборе target, а в стабильности фичей/модели на новых данных.

---

## Шаг 2: Feature robustness и pruning

**Ноутбук:** `11_rd_regime_transition_and_hypotheses.ipynb`

1. Проверить устойчивость `time` и `symbol_encoded` (conditional-фичи).
2. Повторить ablation на OOT/grouped split.
3. Зафиксировать окончательный `core/conditional/experimental` список.

---

## Шаг 3: Meta-labeling (после стабилизации baseline)

**Ноутбук:** `12_Meta_Labeling_and_Alternatives.ipynb`

1. Primary: `tp_sl_1_05`-классификатор.
2. Secondary: фильтр входа/размера позиции.
3. Сравнить прирост net PnL и просадки против чистого baseline.

---

## Шаг 4: Sequence-модели (опционально)

**Ноутбук:** `13_Sequence_Model_30_60.ipynb`

1. Подготовка последовательностей:
   - Окно: 30 и 60 баров
   - Разбивка по session_key (без пересечения)

2. Модели:
   - LSTM / GRU (упрощённая архитектура для экономии памяти)
   - 1D-CNN по временной оси
   - LightGBM на flatten-последовательностях (baseline)

3. Метрики: AUC, net PnL на backtest.

---

## Порядок выполнения

1. Robustness baseline (`tp_sl_1_05 + LightGBM`)
2. Feature robustness / pruning
3. Meta-labeling
4. Sequence-модели (при наличии ресурса и при доказанной пользе)
5. Обновление документации итогами
