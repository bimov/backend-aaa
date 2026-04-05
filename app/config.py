from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Настройки запуска inference service."""

    model_name: str
    model_max_length: int
    model_device: str
    default_prefix: str

    @classmethod
    def from_env(cls) -> "Settings":
        """Считывает настройки сервиса из переменных окружения."""
        return cls(
            model_name=os.getenv("MODEL_NAME", "sergeyzh/rubert-mini-frida"),
            model_max_length=int(os.getenv("MODEL_MAX_LENGTH", "512")),
            model_device=os.getenv("MODEL_DEVICE", "auto"),
            default_prefix=os.getenv("MODEL_DEFAULT_PREFIX", "categorize: "),
        )

    @property
    def resolved_device(self) -> str:
        """Возвращает вычислительное устройство для запуска модели."""
        if self.model_device != "auto":
            return self.model_device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
