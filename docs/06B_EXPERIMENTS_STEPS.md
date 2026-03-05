# Эксперименты: текущий статус и порядок

Дата: 2026-03-04. Цель: развивать baseline `tp_sl_1_05 + LightGBM` и поддерживать воспроизводимость.

---

## Исторически выполненные шаги

### 09. Threshold Strategy

**Ноутбук:** `05_experiments/09B_Threshold_Strategy_Experiments.ipynb`

- Подбор порога HOLD-зоны для baseline LightGBM (22 фичи)
- Сравнение net PnL при разных порогах

---

### 10. Target: горизонт H и mult (triple-barrier)

**Ноутбук:** `05_experiments/09_Target_Horizon_Mult_Experiments.ipynb`

- Перебор H = 3, 4, 5, 6, 7 и mult = 1.0, 1.25, 1.5, 1.75, 2.0
- LightGBM: AUC и net PnL для каждой комбинации
- Результаты: `outputs/exp10_horizon_mult_results.csv`
- Лучшая конфигурация: `outputs/exp10_best_horizon_mult.joblib`

---

### 11. Sequence-модель (окна 30, 60)

**Ноутбук:** `05_experiments/11_Sequence_Model_30_60.ipynb`

- Rolling mean/std по 10 ключевым фичам для окон 30 и 60
- Сравнение baseline (22) vs sequence (62 фичи) по AUC и net PnL
- Вывод: sequence-фичи дают улучшение, `lgb_seq` — кандидат для прода

---

### 12. Meta-labeling

**Ноутбук:** `05_experiments/12_Meta_Labeling_Experiments.ipynb`

- Meta-labeling: target = совпадение rd_regime с исходом
- Сравнение tb_vol vs meta_target
- Вывод: незначительный прирост, не оправдывает сложность

---

### 13. Complex Models, Ensemble и Entry/Exit Logic

**Ноутбук:** `05_experiments/13_Complex_Models_Ensemble_And_Entry_Exit_Logic.ipynb`

- Сравнение моделей: LogReg, RandomForest, XGBoost, LightGBM, CatBoost, MLP
- Набор фичей: baseline (22), old_30_60 (62), new_5_15_30_60 (102)
- Пороги: 20-80, 25-75, 30-70, 35-65, 40-60
- Две торговых логики: `signal_flip` и `exit_on_05`
- Day-by-day stability по дням val/test
- **Вывод:** LightGBM + seq_5_15_30_60 + 25-75 — лучший avg_%_per_trade

---

### 14. Deep Networks and Sequences (PyTorch)

**Ноутбук:** `05_experiments/14_Deep_Networks_And_Sequences.ipynb`

- Модели: LSTM, GRU, 1D-CNN, Transformer (MultiHeadAttention)
- Окна: 5, 15, 30, 60, 120
- Пороги: 20-80 … 40-60
- Фреймворк: PyTorch (CUDA, RTX 3060)
- **Вывод:** глубокие сети не превзошли LightGBM при учёте комиссии 0.1% и проскальзывания. GRU/LSTM лучше на коротких окнах (15-30), но avg/trade ниже LightGBM

---

### 15. Entry/Exit 0.5 Logic Experiment

**Ноутбук:** `05_experiments/15_Entry_Exit_05_Logic_Experiment.ipynb`

- Альтернативная логика: выход при пересечении уверенности 0.5 (вместо ожидания противоположного сигнала)
- Те же модели и фичи что в NB 13
- Сравнение `prod_hold` vs `entry_exit_05` по всем порогам
- **Вывод:** `signal_flip` (prod_hold) устойчивее; exit_on_05 даёт больше сделок, но ниже avg/trade

---

### 16. Production Model (финальная сборка)

**Ноутбук:** `06_production/16_Production_Model_LightGBM.ipynb`

- Сборка продовой модели: данные → фичи → rolling → scale → train → backtest → save
- Модель: LightGBM + 102 фичи + порог 0.75/0.25
- Артефакт: `models/prod_lgbm_seq.joblib`
- Валидация через `src/api/model_bundle.py`
- **Вывод:** артефакт готов к деплою, требуется обновление `inference_service.py` для rolling-фичей

---

## Текущий порядок запуска (актуальный)

### Пайплайн данных и фичей
1. `01_data_prep/01_Load_And_Prepare_Data.ipynb` — загрузка, подготовка, warmup
2. `02_targets/02_Base_Model_And_Target_Comparison.ipynb` — финальный target
3. `03_features/03_Data_Labeling_And_Feature_Loading.ipynb` — labeling
4. `03_features/04_Correlation_Analysis.ipynb` — корреляции
5. `03_features/05_Feature_Selection.ipynb` — отбор фичей, ablation
6. `03_features/06_Scaling_And_Normalization.ipynb` — стандартизация

### Обучение моделей
7. `04_models/07_Model_Training_And_Analysis.ipynb` — baseline модели (22 фичи)
8. `04_models/08_Backtest_Profitability.ipynb` — бэктест baseline

### Эксперименты
9. `05_experiments/09B_Threshold_Strategy_Experiments.ipynb` — пороги
10. `05_experiments/09_Target_Horizon_Mult_Experiments.ipynb` — H и mult
11. `05_experiments/11_Sequence_Model_30_60.ipynb` — rolling 30/60
12. `05_experiments/12_Meta_Labeling_Experiments.ipynb` — meta-labeling
13. `05_experiments/13_Complex_Models_Ensemble_And_Entry_Exit_Logic.ipynb` — полный эксперимент
14. `05_experiments/14_Deep_Networks_And_Sequences.ipynb` — глубокие сети
15. `05_experiments/15_Entry_Exit_05_Logic_Experiment.ipynb` — альтернативная логика

### Production
16. `06_production/16_Production_Model_LightGBM.ipynb` — финальная модель

### API
17. `src/api/*` — online-inference, см. `docs/09_API_STEP_WORKER_AND_SERVICE.md`

---

## Восстановление хода действий

Все эксперименты записаны в отдельных ноутбуках. Актуальные артефакты:

| Артефакт | Путь | Создан в |
|----------|------|----------|
| Размеченные данные | `outputs/data_labeled_tp_sl_1_05.parquet` | NB 03 |
| Отобранные фичи | `outputs/features_selected_tp_sl_1_05.txt` | NB 05 |
| Baseline модель | `outputs/baseline_lgbm_tp_sl_1_05.joblib` | NB 02 |
| Старая прод модель | `models/best_model_tp_sl_1_05.joblib` | NB 07 |
| **Новая прод модель** | `models/prod_lgbm_seq.joblib` | NB 16 |
