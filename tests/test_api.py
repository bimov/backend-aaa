from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

from app.main import create_app


@dataclass(frozen=True)
class FakeResult:
    """Результат работы тестовой заглушки энкодера."""

    embedding: list[float]
    dimensions: int
    inference_time_ms: float
    prefix_used: str


class FakeEncoder:
    """Тестовый энкодер для API."""

    is_ready = True
    model_name = "fake/rubert-mini-frida"
    device = "cpu"
    dimensions = 3

    def embed(self, text: str, prefix: str | None = None) -> FakeResult:
        """Возвращает заранее подготовленный эмбеддинг для тестов."""
        prefix_used = "categorize: " if prefix is None else prefix
        return FakeResult(
            embedding=[0.1, 0.2, 0.3],
            dimensions=3,
            inference_time_ms=1.25,
            prefix_used=prefix_used,
        )


def test_health() -> None:
    """Проверяет, что эндпоинт health возвращает корректный статус."""
    app = create_app(load_model_on_startup=False, initial_encoder=FakeEncoder())
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": "fake/rubert-mini-frida",
        "dimensions": 3,
        "device": "cpu",
    }


def test_embed() -> None:
    """Проверяет успешный ответ эндпоинта embed."""
    app = create_app(load_model_on_startup=False, initial_encoder=FakeEncoder())
    with TestClient(app) as client:
        response = client.post("/embed", json={"text": "тест"})

    assert response.status_code == 200
    assert response.json() == {
        "embedding": [0.1, 0.2, 0.3],
        "dimensions": 3,
        "model": "fake/rubert-mini-frida",
        "prefix_used": "categorize: ",
    }
    assert response.headers["X-Inference-Time-Ms"] == "1.250"
