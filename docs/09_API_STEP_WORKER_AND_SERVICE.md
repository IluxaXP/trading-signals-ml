# Шаг API: worker + сервис

Дата: 2026-03-04.

Этот шаг переводит модель из оффлайн-ноутбуков в online-inference по контракту
`00_contract/ml_api_integration_spec_04-03-2026.md` (v2.2.0).

---

## 1. Что реализовано

- `src/api/app.py`:
  - `GET /health` — проверка загрузки model bundle.
  - `POST /predict` — инференс по окну `[N][6]`:
    `rd_value, open, high, low, close, volume`.
- `src/api/poller.py`:
  - цикл: `GET /api/ml/ds/feature-windows?readyOnly=true` → inference →
    `POST /api/signals/ingest`.
  - `HOLD` не отправляется в ingest.
- `src/api/client_node.py`:
  - REST-клиент с retry/backoff для `5xx`.
  - для `400` retry не выполняется.
- `src/api/model_bundle.py`:
  - безопасная загрузка model bundle (joblib).
  - проверка обязательных ключей: `model`, `scaler`, `features`.
- `src/api/inference_service.py`:
  - window → DataFrame → `add_features()` → scaler/model → BUY/SELL/HOLD.
  - использует фолбэк `rd_regime`/`rd_regime_transition` из feature pipeline.
- `src/api/config.py`, `src/api/contracts.py`:
  - env-конфиг и pydantic-контракты.

---

## 2. Model bundle (артефакты)

### Старый baseline (22 фичи)
- Файл: `models/best_model_tp_sl_1_05.joblib` (alias `models/best_model.joblib`)
- Фичи: 22 базовых из `features_selected_tp_sl_1_05.txt`
- Порог: 0.45 (из Val F1 tuning)

### Новая продовая модель (102 фичи)
- Файл: `models/prod_lgbm_seq.joblib`
- Фичи: 102 (22 base + 80 rolling 5/15/30/60)
- Порог: 0.75 / 0.25 (band 25-75)
- Создан в: `06_production/16_Production_Model_LightGBM.ipynb`

**Важно:** для использования новой модели требуется обновить `inference_service.py` — добавить расчёт rolling-фичей (80 новых). Текущий `add_features()` считает только 22 базовых.

---

## 3. Как запускать

### 3.1 FastAPI сервис

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Проверка:

```bash
curl http://localhost:8000/health
```

### 3.2 Worker poller

```bash
python -m src.api.poller
```

---

## 4. ENV-переменные

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `MODEL_PATH` | `models/best_model.joblib` | Путь к артефакту модели |
| `NODE_API_BASE_URL` | `http://localhost:3000` | URL Node.js API платформы |
| `API_TIMEOUT_S` | `10` | Таймаут HTTP-запросов |
| `INGEST_SOURCE` | `ml_shadow` | `ml` или `ml_shadow` |
| `THRESHOLD_HI_DEFAULT` | `0.55` | Порог BUY (переопределяется из bundle) |
| `LOOKBACK_STEPS` | `60` | Размер окна фичей |
| `POLL_INTERVAL_S` | `60` | Интервал поллинга (сек) |
| `RETRY_5XX_MAX` | `3` | Макс. retry при 5xx |
| `RETRY_BACKOFF_S` | `1.0` | Базовый backoff (сек) |

Для новой модели: `MODEL_PATH=models/prod_lgbm_seq.joblib`.

---

## 5. Спецификация API v2.2.0 (AUTH-02)

Изменения от v2.1.0:

- Добавлены профили доступа через Basic Auth (Nginx):
  - `operator` — полный доступ к публичному периметру
  - `dataset` — только `GET /api/ml/ds/*`
- `POST /api/signals/ingest` недоступен для `dataset`
- Для поллинга и отправки сигналов использовать профиль `operator`

Контракт данных не изменился:
- `timeframe = "1m"`, `lookbackSteps = 60`
- Порядок строк: `oldest → newest`
- `signal_barrier` только `1|-1`
- В training-dataset строки с `rd_value = 0` не включаются

Файл: `00_contract/ml_api_integration_spec_04-03-2026.md`.

---

## 6. Smoke-тесты

```bash
python -m unittest discover -s tests/smoke -p "test_*.py" -v
```

Покрывается:
- `/health` и `/predict`
- retry-policy (`5xx` → retry, `400` → no-retry)
- poller-логика для `READY/WARMUP` и правила HOLD

---

## 7. Важные замечания

1. Порядок колонок окна строго: `rd_value, open, high, low, close, volume`.
2. Для online-пайплайна использовать фичи из `bundle['features']`.
3. При переходе на новый bundle (`prod_lgbm_seq.joblib`) нужно:
   - обновить `inference_service.py` — добавить rolling-фичи;
   - обновить `MODEL_PATH` в env;
   - сохранить буфер 60 точек per session для rolling.
4. Рекомендуемый rollout: старт с `source=ml_shadow`, мониторинг, затем `source=ml`.
