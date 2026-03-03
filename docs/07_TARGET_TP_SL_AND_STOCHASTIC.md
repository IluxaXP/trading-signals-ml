# Target: TP/SL фиксированный % и стохастическая аномальная доходность

Актуальный статус целевых переменных по итогам `02_targets/03_Base_Model_And_Target_Comparison.ipynb`. Дата: 2026-03.

---

## 1. Финальный target в проекте

- **Выбранный target:** `tp_sl_1_05` (TP=1%, SL=0.5%, horizon=20)
- **Финальная модель baseline:** `LightGBM`
- **Артефакты:**
  - `outputs/prepared_with_target_tp_sl_1_05.parquet`
  - `outputs/target_selection_summary.csv`
  - `outputs/baseline_lgbm_tp_sl_1_05.joblib`

### Почему выбран `tp_sl_1_05`

- Лучший AUC среди сравниваемых target в baseline-сценарии.
- Высокий net PnL при заметно меньшем turnover, чем у `tb_vol_5bar`.
- Хороший компромисс между частотой сделок и качеством сигналов.

---

## 2. Логика TP/SL fixed (подход nikitapre)

**Источник идеи:** [nikitapre/trading_project_2_0](https://github.com/nikitapre/trading_project_2_0)

### Правила разметки

- Для long:
  - TP: `entry * (1 + TP_pct)`
  - SL: `entry * (1 - SL_pct)`
- Для short:
  - TP: `entry * (1 - TP_pct)`
  - SL: `entry * (1 + SL_pct)`
- Метка:
  - `1` — long-сценарий выигрывает
  - `-1` — short-сценарий выигрывает
  - `0` — timeout/неоднозначность

### Важное обновление (intrabar ambiguity)

Если в **одном и том же баре** одновременно касаются TP и SL (по `high`/`low`), порядок событий внутри минуты неизвестен.  
В проекте принято правило: **`ambiguous intrabar -> 0`**.

### Масштаб неоднозначности (по текущим данным)

- `tp_sl_2_1`: ambiguous first-hit около `0.1%` entry-points
- `tp_sl_1_05`: ambiguous first-hit около `0.5-0.6%` entry-points

Вывод: влияние на общие метрики ограниченное, но правило повышает корректность разметки.

---

## 3. Исследовательские target (исторически)

| Target | Назначение | Статус |
|--------|------------|--------|
| `tb_vol_5bar` | volatility-adjusted triple barrier | baseline для сравнения |
| `tp_sl_2_1` | fixed TP/SL (2%/1%) | baseline для сравнения |
| `tp_sl_1_05` | fixed TP/SL (1%/0.5%) | **production baseline** |
| stochastic anomalous return | альтернативная схема | исследовательский |

Источники исследовательских ноутбуков:
- `fork/01_Target_TP_SL_Fixed_Research.ipynb`
- `fork/02_Target_Stochastic_Anomalous_Research.ipynb`

---

## 4. Формула безубыточности (справочно)

$$w > \frac{SL + c}{TP + SL}$$

- $w$ — precision выигрышных сделок
- $c$ — издержки (комиссия + проскальзывание)
- Для TP=1%, SL=0.5%, c=0.1%: порог precision около `40%`

---

## 5. Связанные документы

- Фичи и их статус: `docs/08_FEATURES_SOURCES_AND_RATIONALE.md`
- Итоги выбора target/model: `docs/02_TARGET_RESEARCH_CONCLUSIONS.md`
