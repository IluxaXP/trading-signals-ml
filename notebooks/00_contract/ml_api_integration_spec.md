# Спецификация интеграции ML API

Версия: `2.1.0`  
Дата: `2026-03-02`  

## 1. Назначение

Документ фиксирует production-контракт между:
1. Node.js API платформы (поставщик данных и приемник сигналов)
2. ML-сервисом (потребитель данных для обучения и online-инференса)

Режим интеграции:
1. `REST pull` для получения обучающего датасета и online-окон фич
2. `REST push` для отправки сигналов модели обратно в платформу

## 2. Фиксированный контракт

Эти параметры обязательны и не должны переопределяться на стороне ML:
1. `timeframe = "1m"`
2. `lookbackSteps = 60`
3. Порядок строк в окне фич строго `oldest -> newest`
4. Gaps в данных для online-окон допустимы; readiness не зависит от strict `delta_timestamp == 60000`
5. Обучающие строки всегда содержат бинарный `signal_barrier` (`1|-1`)
6. `signal_ema` не входит в новый DS API
7. Online-окна формируются только для символов из текущего `watchlist`
8. В training-dataset строки с `rd_value = 0` не включаются

## 3. Endpoints

### 3.1 Обучающий датасет

`GET /api/ml/ds/training-dataset`

Query-параметры:
1. `fromDate` (optional, ISO8601)
2. `toDate` (optional, ISO8601)
3. `symbols` (optional, CSV short symbols, например `BTCUSDT,ETHUSDT`)
4. `format` (optional: `csv|json`, default `csv`)

Схема строки ответа:
1. `timestamp` (`number`, ms)
2. `symbol` (`string`)
3. `rd_value` (`number`)
4. `open` (`number`)
5. `high` (`number`)
6. `low` (`number`)
7. `close` (`number`)
8. `volume` (`number`)
9. `signal_barrier` (`1|-1`)

Правило генерации `signal_barrier`:
1. Данные группируются по `symbol` и упорядочиваются по `timestamp` (ASC).
2. Из каждой группы удаляются строки с `rd_value = 0`.
3. Для каждой соседней пары вычисляется `delta = rd_value(i) - rd_value(i-1)` и направление `sign(delta)`:
   - `delta > 0` => `1`
   - `delta < 0` => `-1`
   - `delta = 0` => без смены направления
4. Точки старта сегмента фиксируются при смене ненулевого направления относительно предыдущего ненулевого направления.
5. `full-segment`: каждая строка группы получает текущее направление сегмента (`1` или `-1`), то есть в выходе нет `0|null`.
6. Если после удаления `rd_value = 0` в группе нет ни одного ненулевого `delta(rd_value)`, строки этого `symbol` не включаются в ответ.

### 3.2 Online-окна фич

`GET /api/ml/ds/feature-windows`

Query-параметры:
1. `readyOnly` (optional boolean, default `false`)

Схема ответа:
1. `timeframe: "1m"`
2. `lookbackSteps: 60`
3. `featureColumns: ["rd_value","open","high","low","close","volume"]`
4. `generatedAt` (ISO8601 string)
5. `items[]`

Схема `items[]`:
1. `symbol` (`string`)
2. `state` (`READY|WARMUP`)
3. `reason` (`null|not_enough_points|stale_data|missing_joined_rows`)
4. `pointsCollected` (`number`)
5. `expectedPoints` (`60`)
6. `windowStartTimestamp` (`number|null`)
7. `windowEndTimestamp` (`number|null`)
8. `features` (только при `state=READY`, форма `[60][6]`)

Логика readiness:
1. Используются только join-строки, где одновременно есть `rating_history.rd` и `trading_data(timeframe='1m')` на тот же `symbol+timestamp`.
2. Для `READY` достаточно последних 60 join-точек (строгая непрерывность по timestamp не требуется).
3. Любые gaps внутри истории не переводят символ в `WARMUP`, если есть минимум 60 точек.
4. `WARMUP` выставляется, если:
   - точек меньше 60 (`reason=not_enough_points`), или
   - последняя точка по символу старше 15 минут относительно текущего времени сервера (`reason=stale_data`).

### 3.3 Прием сигналов (из ML в платформу)

`POST /api/signals/ingest`

Это официальный endpoint для получения сигналов модели.

Пример request JSON:
```json
{
  "symbol": "BTCUSDT",
  "timestamp": 1700000000000,
  "signal": "BUY",
  "price": 62123.5,
  "rating": 0.78,
  "source": "ml_shadow"
}
```

Ограничения валидации:
1. Обязательные поля: `symbol`, `timestamp`, `signal`, `price`, `rating`, `source`
2. `signal` допустимые значения: `BUY|SELL`
3. `source` допустимые значения: `ema|ml|ml_shadow` (для ML использовать `ml` или `ml_shadow`)

Рекомендуемое маппирование результата инференса:
1. `timestamp = windowEndTimestamp`
2. `price = close` последней строки окна фич
3. `rating = confidence/probability` модели
4. При выходе модели `HOLD` (`0`) запрос в ingest не отправлять

## 4. Rollout и режим эксплуатации

Рекомендуемый rollout:
1. Старт с `source=ml_shadow` для теневой проверки.
2. Мониторинг поведения и quality-метрик.
3. Переключение на `source=ml` по операционной процедуре cutover.

## 5. Рекомендуемый polling workflow

Каждую минуту:
1. Вызвать `GET /api/ml/ds/feature-windows?readyOnly=true`
2. Для каждого `item` со `state=READY` выполнить инференс по `features`
3. Если результат `BUY|SELL`, отправить `POST /api/signals/ingest`
4. Если результат `HOLD`, сигнал не отправлять

## 6. Контракт обработки ошибок

1. `400` - некорректные параметры запроса или payload
2. `500` - внутренняя ошибка сервера

Требования к ML-сервису:
1. Retry с backoff для временных `5xx`
2. Без retry для `400`, пока не исправлены запрос/данные
3. Элементы `WARMUP` информационные и не должны запускать инференс

## 7. Примечания по совместимости

1. Legacy endpoint `GET /api/ml/export-dataset` остается доступным.
2. Контракт `POST /api/signals/ingest` не изменяется.
3. Версия `2.0.0` содержит **breaking change** для `GET /api/ml/ds/training-dataset`: `signal_barrier` теперь только `1|-1`, логика triple-barrier `0|null` удалена.
4. Версия `2.1.0` обновляет readiness `GET /api/ml/ds/feature-windows`: gaps больше не приводят к `WARMUP`; добавлен `stale_data` при отсутствии новых точек более 15 минут.
