const express = require('express');
const fs = require('fs');
const path = require('path');
const app = express();
app.use(express.json());

// Функция для получения временной метки в ISO формате (UTC)
const getTimestamp = () => new Date().toISOString();

// Загружаем демо-данные
const dataPath = path.join(__dirname, 'data', 'demo_data.json');
const rawData = JSON.parse(fs.readFileSync(dataPath, 'utf8'));

// Извлекаем символ из первой записи
const symbol = rawData[0]?.symbol || "UNKNOWN";

const historicalData = rawData.map(item => ({
    timestamp: item.timestamp,
    open: item.open,
    high: item.high,
    low: item.low,
    close: item.close,
    volume: item.volume,
    rd_value: item.rd_value
}));

let currentIndex = 60; // сразу READY

app.get('/api/ml/ds/feature-windows', (req, res) => {
    if (currentIndex < 60) {
        return res.json({
            timeframe: "1m",
            lookbackSteps: 60,
            featureColumns: ["rd_value", "open", "high", "low", "close", "volume"],
            generatedAt: getTimestamp(),
            items: [
                {
                    symbol: symbol,
                    state: "WARMUP",
                    reason: "not_enough_points",
                    pointsCollected: currentIndex,
                    expectedPoints: 60,
                }
            ]
        });
    }

    const start = currentIndex - 60;
    const windowData = historicalData.slice(start, currentIndex);
    
    const features = windowData.map(row => [
        row.rd_value,
        row.open,
        row.high,
        row.low,
        row.close,
        row.volume
    ]);

    const response = {
        timeframe: "1m",
        lookbackSteps: 60,
        featureColumns: ["rd_value", "open", "high", "low", "close", "volume"],
        generatedAt: getTimestamp(),
        items: [
            {
                symbol: symbol,
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

    if (currentIndex < historicalData.length) {
        currentIndex++;
    }

    res.json(response);
});

app.post('/api/signals/ingest', (req, res) => {
    const { symbol, timestamp, signal, price, rating, source } = req.body;
    if (!symbol || !timestamp || !signal || !price || !rating || !source) {
        return res.status(400).json({ error: "Missing required fields" });
    }
    console.log(`\n[${getTimestamp()}] 🚀 ПОЛУЧЕН СИГНАЛ ОТ ML!`);
    console.log(`[${getTimestamp()}] Торговая пара: ${symbol} | Сигнал: ${signal} | Цена: ${price} | Уверенность: ${(rating*100).toFixed(1)}% | Режим: ${source}`);
    res.status(200).json({ success: true, message: "Signal accepted" });
});

app.listen(3000, () => console.log(`[${getTimestamp()}] ✅ Мок-платформа с реальными данными запущена на http://localhost:3000`));