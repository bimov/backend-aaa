# Результаты бенчмарка

- Время замера: 2026-04-05T10:43:39.754390+00:00
- Платформа: macOS-14.4-arm64-arm-64bit
- Python: 3.11.9
- CPU count: 8
- Модель: sergeyzh/rubert-mini-frida
- Контейнер: rubert-inference-service
- Устройство: cpu
- Размер эмбеддинга: 312
- Сценарий: 120 запросов, concurrency=6, warmup=12

## Метрики

| Метрика | Значение |
| --- | ---: |
| Latency mean, ms | 149.774 |
| Latency p50, ms | 147.895 |
| Latency p95, ms | 174.195 |
| Latency p99, ms | 181.042 |
| Throughput, RPS | 39.007 |
| Inference time mean, ms | 25.46 |
| Inference time p95, ms | 29.208 |
| Peak RSS, MB | 278.5 |
| Delta RSS, MB | 8.1 |

## Проверка порогов

| Порог | Статус |
| --- | --- |
| latency_p95_ms <= 200.0 | OK |
| latency_p99_ms <= 250.0 | OK |
| throughput_rps >= 8.0 | OK |
| inference_time_p95_ms <= 150.0 | OK |
| peak_rss_mb <= 900.0 | OK |
