from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool

from app.config import Settings
from app.schemas import EmbedRequest, EmbedResponse, HealthResponse


def create_app(
    load_model_on_startup: bool = True,
    initial_encoder: Any | None = None,
) -> FastAPI:
    """Создает и настраивает экземпляр FastAPI приложения."""
    settings = Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        """Управляет жизненным циклом модели при старте и остановке сервиса."""
        application.state.settings = settings
        if load_model_on_startup:
            from app.service import EmbeddingService

            encoder = EmbeddingService(settings=settings)
            encoder.load()
            application.state.encoder = encoder
        else:
            application.state.encoder = initial_encoder
        yield
        current_encoder = getattr(application.state, "encoder", None)
        if load_model_on_startup and hasattr(current_encoder, "unload"):
            current_encoder.unload()

    application = FastAPI(
        title="rubert-mini-frida Inference Service",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        """Возвращает статус готовности сервиса и параметры модели."""
        encoder = getattr(request.app.state, "encoder", None)
        if encoder is None or not getattr(encoder, "is_ready", False):
            return HealthResponse(
                status="starting",
                model=settings.model_name,
                dimensions=0,
                device=settings.resolved_device,
            )
        return HealthResponse(
            status="ok",
            model=encoder.model_name,
            dimensions=encoder.dimensions,
            device=encoder.device,
        )

    @application.post("/embed", response_model=EmbedResponse)
    async def embed(
        payload: EmbedRequest,
        response: Response,
        request: Request,
    ) -> EmbedResponse:
        """Возвращает эмбеддинг для переданного текста."""
        encoder = getattr(request.app.state, "encoder", None)
        if encoder is None or not getattr(encoder, "is_ready", False):
            raise HTTPException(status_code=503, detail="Модель еще не загружена")

        result = await run_in_threadpool(encoder.embed, payload.text, payload.prefix)
        response.headers["X-Inference-Time-Ms"] = f"{result.inference_time_ms:.3f}"

        return EmbedResponse(
            embedding=result.embedding,
            dimensions=result.dimensions,
            model=encoder.model_name,
            prefix_used=result.prefix_used,
        )

    return application


app = create_app()
