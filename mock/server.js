const express = require('express');
const fs = require('fs');
const path = require('path');
const app = express();
app.use(express.json());

// Загружаем демо-данные
const dataPath = path.join(__dirname, 'data', 'btcusdt_demo.json');
const rawData = JSON.parse(fs.readFileSync(dataPath, 'utf8'));

// Преобразуем в удобный формат: массив объектов с полями
// Убедимся, что все числа в нужном типе
const historicalData = rawData.map(item => ({
    timestamp: item.timestamp,
    open: item.open,
    high: item.high,
    low: item.low,
    close: item.close,
    volume: item.volume,
    rd_value: item.rd_value
}));

// Переменная для хранения текущего индекса (имитация реального времени)
let currentIndex = 60; // будем увеличивать при каждом запросе

// Эндпоинт для получения окна фичей
app.get('/api/ml/ds/feature-windows', (req, res) => {
    const readyOnly = req.query.readyOnly === 'true';
    
    // Проверяем, хватает ли данных для окна
    if (currentIndex < 60) {
        // Ещё не накопилось 60 баров — возвращаем статус WARMUP
        return res.json({
            timeframe: "1m",
            lookbackSteps: 60,
            featureColumns: ["rd_value", "open", "high", "low", "close", "volume"],
            generatedAt: new Date().toISOString(),
            items: [
                {
                    symbol: "BTCUSDT",
                    state: "WARMUP",
                    reason: "not_enough_points",
                    pointsCollected: currentIndex,
                    expectedPoints: 60,
                    // можно не отдавать features
                }
            ]
        });
    }

    // Берём окно из последних 60 записей (от currentIndex-60 до currentIndex-1)
    const start = currentIndex - 60;
    const windowData = historicalData.slice(start, currentIndex);
    
    // Извлекаем матрицу фичей в нужном порядке
    const features = windowData.map(row => [
        row.rd_value,
        row.open,
        row.high,
        row.low,
        row.close,
        row.volume
    ]);

    // Формируем ответ
    const response = {
        timeframe: "1m",
        lookbackSteps: 60,
        featureColumns: ["rd_value", "open", "high", "low", "close", "volume"],
        generatedAt: new Date().toISOString(),
        items: [
            {
                symbol: "BTCUSDT",
                state: "READY",
                reason: null,
                pointsCollected: 60,
                expectedPoints: 60,
                windowStartTimestamp: windowData[0].timestamp,
                windowEndTimestamp: windowData[59].timestamp,
                features: features
            }
        ]
    };

    // Увеличиваем currentIndex для следующего запроса (имитация нового бара)
    if (currentIndex < historicalData.length) {
        currentIndex++;
    }

    res.json(response);
});

// Эндпоинт приёма сигналов (без изменений)
app.post('/api/signals/ingest', (req, res) => {
    const { symbol, timestamp, signal, price, rating, source } = req.body;
    if (!symbol || !timestamp || !signal || !price || !rating || !source) {
        return res.status(400).json({ error: "Missing required fields" });
    }
    console.log(`\n[PLATFORM] 🚀 ПОЛУЧЕН СИГНАЛ ОТ ML!`);
    console.log(`Торговая пара: ${symbol} | Сигнал: ${signal} | Цена: ${price} | Уверенность: ${(rating*100).toFixed(1)}% | Режим: ${source}`);
    res.status(200).json({ success: true, message: "Signal accepted" });
});

app.listen(3000, () => console.log('✅ Мок-платформа с реальными данными запущена на http://localhost:3000'));