from __future__ import annotations

from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    """Тело запроса на получение эмбеддинга."""

    text: str = Field(min_length=1)
    prefix: str | None = None


class EmbedResponse(BaseModel):
    """Ответ сервиса с эмбеддингом и метаданными модели."""

    embedding: list[float]
    dimensions: int
    model: str
    prefix_used: str


class HealthResponse(BaseModel):
    """Ответ эндпоинта проверки готовности сервиса."""

    status: str
    model: str
    dimensions: int
    device: str
