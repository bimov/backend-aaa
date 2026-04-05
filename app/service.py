from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import perf_counter

import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer

from app.config import Settings


@dataclass(frozen=True)
class EmbeddingResult:
    """Результат вычисления эмбеддинга для одного текста."""

    embedding: list[float]
    dimensions: int
    inference_time_ms: float
    prefix_used: str


class EmbeddingService:
    """Сервис для построения эмбеддингов."""

    def __init__(self, settings: Settings) -> None:
        """Инициализирует сервис и подготавливает внутреннее состояние."""
        self._settings = settings
        self._tokenizer = None
        self._model = None
        self._device = settings.resolved_device
        self._dimensions = 0
        self._load_lock = Lock()
        self._infer_lock = Lock()

    @property
    def is_ready(self) -> bool:
        """Показывает, загружены ли токенизатор и модель."""
        return self._tokenizer is not None and self._model is not None

    @property
    def model_name(self) -> str:
        """Возвращает имя модели из конфигурации."""
        return self._settings.model_name

    @property
    def device(self) -> str:
        """Возвращает устройство, на котором размещена модель."""
        return self._device

    @property
    def dimensions(self) -> int:
        """Возвращает размерность выходного эмбеддинга."""
        return self._dimensions

    def load(self) -> None:
        """Загружает токенизатор и модель в память один раз."""
        with self._load_lock:
            if self.is_ready:
                return
            tokenizer = AutoTokenizer.from_pretrained(self._settings.model_name)
            model = AutoModel.from_pretrained(self._settings.model_name)
            model.to(self._device)
            model.eval()
            self._tokenizer = tokenizer
            self._model = model
            self._dimensions = int(getattr(model.config, "hidden_size", 0))

    def unload(self) -> None:
        """Освобождает модель и очищает память при остановке сервиса."""
        with self._load_lock:
            self._tokenizer = None
            self._model = None
            self._dimensions = 0
            if self._device == "cuda":
                torch.cuda.empty_cache()

    def embed(self, text: str, prefix: str | None = None) -> EmbeddingResult:
        """Строит нормализованный эмбеддинг для входного текста."""
        if not self.is_ready:
            raise RuntimeError("Модель еще не загружена")

        prepared_text, prefix_used = self._prepare_text(text=text, prefix=prefix)

        with self._infer_lock:
            started_at = perf_counter()
            tokenized = self._tokenizer(
                [prepared_text],
                max_length=self._settings.model_max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self._device)

            with torch.inference_mode():
                outputs = self._model(**tokenized)
                pooled = self._mean_pool(
                    hidden_state=outputs.last_hidden_state,
                    attention_mask=tokenized["attention_mask"],
                )
                normalized = functional.normalize(pooled, p=2, dim=1)

            embedding = normalized[0].detach().cpu().tolist()
            inference_time_ms = (perf_counter() - started_at) * 1000

        return EmbeddingResult(
            embedding=embedding,
            dimensions=len(embedding),
            inference_time_ms=inference_time_ms,
            prefix_used=prefix_used,
        )

    def _prepare_text(self, text: str, prefix: str | None) -> tuple[str, str]:
        """Добавляет префикс задачи к тексту и возвращает итоговую строку."""
        prefix_value = self._settings.default_prefix if prefix is None else prefix
        if prefix_value:
            return f"{prefix_value}{text}", prefix_value
        return text, ""

    @staticmethod
    def _mean_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Вычисляет mean pooling по непустым токенам последовательности."""
        masked = hidden_state * attention_mask.unsqueeze(-1).float()
        summed = masked.sum(dim=1)
        counts = attention_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        return summed / counts
