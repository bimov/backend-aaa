from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

import httpx
import psutil

from benchmarks.samples import SAMPLE_TEXTS


@dataclass(frozen=True)
class Thresholds:
    """Целевые пороги для оценки результатов бенчмарка."""

    latency_p95_ms: float = 200.0
    latency_p99_ms: float = 250.0
    throughput_rps: float = 8.0
    inference_time_p95_ms: float = 150.0
    peak_rss_mb: float = 900.0


@dataclass(frozen=True)
class BenchmarkConfig:
    """Параметры запуска нагрузочного сценария."""

    base_url: str
    container_name: str
    total_requests: int
    concurrency: int
    warmup_requests: int
    results_path: Path
    markdown_path: Path
    thresholds: Thresholds


class MemorySampler:
    """Фоновый сборщик RSS процесса сервиса."""

    def __init__(self, container_name: str, interval_seconds: float = 0.5) -> None:
        """Создает объект для периодического чтения памяти контейнера."""
        self._container_name = container_name
        self._interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._samples_mb: list[float] = []

    def start(self) -> None:
        """Запускает фоновый поток сбора метрик памяти."""
        self._thread.start()

    def stop(self) -> None:
        """Останавливает фоновый поток и дожидается его завершения."""
        self._stop_event.set()
        self._thread.join(timeout=2)

    @property
    def peak_mb(self) -> float:
        """Возвращает максимальное зафиксированное значение RSS в мегабайтах."""
        if not self._samples_mb:
            return 0.0
        return max(self._samples_mb)

    def _run(self) -> None:
        """Считывает память контейнера с заданным интервалом до остановки."""
        while not self._stop_event.is_set():
            try:
                self._samples_mb.append(read_container_memory_mb(self._container_name))
            except RuntimeError:
                return
            time.sleep(self._interval_seconds)


def percentile(values: list[float], ratio: float) -> float:
    """Вычисляет процентиль по списку чисел с линейной интерполяцией."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * ratio
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    return lower + (upper - lower) * fraction


def bytes_to_mb(value: int) -> float:
    """Переводит размер из байт в мегабайты."""
    return value / (1024 * 1024)


def parse_args() -> BenchmarkConfig:
    """Читает аргументы командной строки и формирует конфигурацию запуска."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--container-name", default="rubert-inference-service")
    parser.add_argument("--total-requests", type=int, default=160)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup-requests", type=int, default=20)
    parser.add_argument(
        "--results-path",
        default="benchmarks/results/docker_cpu.json",
    )
    parser.add_argument(
        "--markdown-path",
        default="benchmarks/results/docker_cpu.md",
    )
    args = parser.parse_args()

    return BenchmarkConfig(
        base_url=args.base_url,
        container_name=args.container_name,
        total_requests=args.total_requests,
        concurrency=args.concurrency,
        warmup_requests=args.warmup_requests,
        results_path=Path(args.results_path),
        markdown_path=Path(args.markdown_path),
        thresholds=Thresholds(),
    )


async def wait_for_health(base_url: str, timeout_seconds: float = 180.0) -> dict:
    """Ожидает, пока сервис перейдет в состояние готовности."""
    deadline = time.perf_counter() + timeout_seconds
    async with httpx.AsyncClient(timeout=10.0) as client:
        while time.perf_counter() < deadline:
            try:
                response = await client.get(f"{base_url}/health")
                if response.status_code == 200:
                    payload = response.json()
                    if payload.get("status") == "ok":
                        return payload
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)
    raise TimeoutError("Сервис не успел перейти в состояние ok")


async def warmup(client: httpx.AsyncClient, warmup_requests: int) -> None:
    """Выполняет прогрев сервиса перед основным измерением."""
    for index in range(warmup_requests):
        text = SAMPLE_TEXTS[index % len(SAMPLE_TEXTS)]
        response = await client.post("/embed", json={"text": text})
        response.raise_for_status()


async def run_load(client: httpx.AsyncClient, total_requests: int, concurrency: int) -> tuple[list[float], list[float], float]:
    """Отправляет основную серию запросов и собирает сырые метрики."""
    latencies_ms: list[float] = []
    inference_times_ms: list[float] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def send_request(index: int) -> None:
        """Отправляет один запрос и сохраняет измерения по нему."""
        async with semaphore:
            text = SAMPLE_TEXTS[index % len(SAMPLE_TEXTS)]
            started_at = time.perf_counter()
            response = await client.post("/embed", json={"text": text})
            finished_at = time.perf_counter()
            response.raise_for_status()
            latencies_ms.append((finished_at - started_at) * 1000)
            inference_times_ms.append(float(response.headers["X-Inference-Time-Ms"]))

    started_at = time.perf_counter()
    await asyncio.gather(*(send_request(index) for index in range(total_requests)))
    total_time_seconds = time.perf_counter() - started_at
    return latencies_ms, inference_times_ms, total_time_seconds


def build_results(
    config: BenchmarkConfig,
    health_payload: dict,
    latencies_ms: list[float],
    inference_times_ms: list[float],
    total_time_seconds: float,
    baseline_rss_mb: float,
    peak_rss_mb: float,
) -> dict:
    """Собирает итоговую структуру результатов бенчмарка."""
    throughput_rps = config.total_requests / total_time_seconds if total_time_seconds else 0.0
    metrics = {
        "latency_ms": {
            "mean": round(mean(latencies_ms), 3),
            "p50": round(percentile(latencies_ms, 0.50), 3),
            "p95": round(percentile(latencies_ms, 0.95), 3),
            "p99": round(percentile(latencies_ms, 0.99), 3),
            "min": round(min(latencies_ms), 3),
            "max": round(max(latencies_ms), 3),
        },
        "inference_time_ms": {
            "mean": round(mean(inference_times_ms), 3),
            "p50": round(percentile(inference_times_ms, 0.50), 3),
            "p95": round(percentile(inference_times_ms, 0.95), 3),
            "p99": round(percentile(inference_times_ms, 0.99), 3),
        },
        "throughput_rps": round(throughput_rps, 3),
        "memory_mb": {
            "baseline_rss": round(baseline_rss_mb, 3),
            "peak_rss": round(peak_rss_mb, 3),
            "delta_rss": round(peak_rss_mb - baseline_rss_mb, 3),
        },
    }

    thresholds = asdict(config.thresholds)
    verdicts = {
        "latency_p95_ms": metrics["latency_ms"]["p95"] <= config.thresholds.latency_p95_ms,
        "latency_p99_ms": metrics["latency_ms"]["p99"] <= config.thresholds.latency_p99_ms,
        "throughput_rps": metrics["throughput_rps"] >= config.thresholds.throughput_rps,
        "inference_time_p95_ms": metrics["inference_time_ms"]["p95"] <= config.thresholds.inference_time_p95_ms,
        "peak_rss_mb": metrics["memory_mb"]["peak_rss"] <= config.thresholds.peak_rss_mb,
    }

    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "system": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
            "memory_total_mb": round(bytes_to_mb(psutil.virtual_memory().total), 3),
        },
        "service": {
            "base_url": config.base_url,
            "model": health_payload["model"],
            "device": health_payload["device"],
            "dimensions": health_payload["dimensions"],
        },
        "scenario": {
            "warmup_requests": config.warmup_requests,
            "total_requests": config.total_requests,
            "concurrency": config.concurrency,
        },
        "metrics": metrics,
        "thresholds": thresholds,
        "verdicts": verdicts,
    }


def save_results(results: dict, json_path: Path, markdown_path: Path) -> None:
    """Сохраняет результаты бенчмарка в JSON и Markdown."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(results), encoding="utf-8")


def render_markdown(results: dict) -> str:
    """Преобразует результаты бенчмарка в Markdown-отчет."""
    latency = results["metrics"]["latency_ms"]
    inference = results["metrics"]["inference_time_ms"]
    memory = results["metrics"]["memory_mb"]
    verdicts = results["verdicts"]

    lines = [
        "# Результаты бенчмарка",
        "",
        f"- Время замера: {results['timestamp_utc']}",
        f"- Платформа: {results['system']['platform']}",
        f"- Python: {results['system']['python']}",
        f"- CPU count: {results['system']['cpu_count']}",
        f"- Модель: {results['service']['model']}",
        f"- Контейнер: {results['service']['container_name']}",
        f"- Устройство: {results['service']['device']}",
        f"- Размер эмбеддинга: {results['service']['dimensions']}",
        f"- Сценарий: {results['scenario']['total_requests']} запросов, concurrency={results['scenario']['concurrency']}, warmup={results['scenario']['warmup_requests']}",
        "",
        "## Метрики",
        "",
        "| Метрика | Значение |",
        "| --- | ---: |",
        f"| Latency mean, ms | {latency['mean']} |",
        f"| Latency p50, ms | {latency['p50']} |",
        f"| Latency p95, ms | {latency['p95']} |",
        f"| Latency p99, ms | {latency['p99']} |",
        f"| Throughput, RPS | {results['metrics']['throughput_rps']} |",
        f"| Inference time mean, ms | {inference['mean']} |",
        f"| Inference time p95, ms | {inference['p95']} |",
        f"| Peak RSS, MB | {memory['peak_rss']} |",
        f"| Delta RSS, MB | {memory['delta_rss']} |",
        "",
        "## Проверка порогов",
        "",
        "| Порог | Статус |",
        "| --- | --- |",
        f"| latency_p95_ms <= {results['thresholds']['latency_p95_ms']} | {'OK' if verdicts['latency_p95_ms'] else 'FAIL'} |",
        f"| latency_p99_ms <= {results['thresholds']['latency_p99_ms']} | {'OK' if verdicts['latency_p99_ms'] else 'FAIL'} |",
        f"| throughput_rps >= {results['thresholds']['throughput_rps']} | {'OK' if verdicts['throughput_rps'] else 'FAIL'} |",
        f"| inference_time_p95_ms <= {results['thresholds']['inference_time_p95_ms']} | {'OK' if verdicts['inference_time_p95_ms'] else 'FAIL'} |",
        f"| peak_rss_mb <= {results['thresholds']['peak_rss_mb']} | {'OK' if verdicts['peak_rss_mb'] else 'FAIL'} |",
        "",
    ]
    return "\n".join(lines)


async def run_benchmark(config: BenchmarkConfig) -> dict:
    """Полностью выполняет сценарий бенчмарка и возвращает результаты."""
    health_payload = await wait_for_health(config.base_url)

    async with httpx.AsyncClient(base_url=config.base_url, timeout=120.0) as client:
        await warmup(client, config.warmup_requests)
        baseline_rss_mb = read_container_memory_mb(config.container_name)
        sampler = MemorySampler(config.container_name)
        sampler.start()
        try:
            latencies_ms, inference_times_ms, total_time_seconds = await run_load(
                client=client,
                total_requests=config.total_requests,
                concurrency=config.concurrency,
            )
        finally:
            sampler.stop()

    results = build_results(
        config=config,
        health_payload=health_payload,
        latencies_ms=latencies_ms,
        inference_times_ms=inference_times_ms,
        total_time_seconds=total_time_seconds,
        baseline_rss_mb=baseline_rss_mb,
        peak_rss_mb=sampler.peak_mb,
    )
    results["service"]["container_name"] = config.container_name
    return results


def read_container_memory_mb(container_name: str) -> float:
    """Считывает текущее потребление памяти контейнера в мегабайтах."""
    completed = subprocess.run(
        [
            "docker",
            "stats",
            "--no-stream",
            "--format",
            "{{.MemUsage}}",
            container_name,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "Не удалось прочитать память контейнера")
    usage = completed.stdout.strip().split("/", maxsplit=1)[0].strip()
    return parse_size_to_mb(usage)


def parse_size_to_mb(raw_value: str) -> float:
    """Преобразует строковое значение размера Docker в мегабайты."""
    normalized = raw_value.strip().replace("iB", "B")
    units = [
        ("GB", 1024),
        ("MB", 1),
        ("kB", 1 / 1024),
        ("KB", 1 / 1024),
        ("B", 1 / (1024 * 1024)),
    ]
    for suffix, multiplier in units:
        if normalized.endswith(suffix):
            value = float(normalized[: -len(suffix)].strip())
            return value * multiplier
    raise RuntimeError(f"Неизвестный формат размера памяти: {raw_value}")


def main() -> None:
    """Запускает бенчмарк из командной строки."""
    config = parse_args()
    results = asyncio.run(run_benchmark(config))
    save_results(results, config.results_path, config.markdown_path)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
