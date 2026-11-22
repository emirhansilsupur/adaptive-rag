"""FastAPI application factory and ASGI entrypoint for AdaptiveRAG."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from adaptiverag.api.dependencies import AppContainer, create_app_container
from adaptiverag.api.routes import router as v1_router
from adaptiverag.config import get_settings
from adaptiverag.observability.tracing import configure_logging

logger = structlog.get_logger(__name__)


def create_app(container: AppContainer | None = None) -> FastAPI:
    """Create and configure the AdaptiveRAG FastAPI application.

    Passing a pre-built ``container`` enables deterministic dependency injection in tests
    without connecting to live Qdrant or external LLM APIs.
    """

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if container is not None:
            application.state.container = container
            logger.info("adaptiverag_api_started", mode="injected_container")
            yield
            return

        settings = get_settings()
        configure_logging(settings.log_level)
        managed_container = create_app_container(settings)
        application.state.container = managed_container
        logger.info(
            "adaptiverag_api_started",
            mode="managed_lifespan",
            collection=managed_container.store.collection_name,
            llm_provider=settings.llm_provider,
        )
        try:
            yield
        finally:
            await managed_container.aclose()
            logger.info("adaptiverag_api_shutdown")

    application = FastAPI(
        title="AdaptiveRAG API",
        version="0.1.0",
        description=(
            "Query-aware Agentic RAG engine with native Qdrant hybrid search (Dense + BM25 RRF), "
            "cross-encoder reranking, 3-stage dynamic top-k, multi-signal confidence + CRAG gating, "
            "web fallback, and two-tier citation validation."
        ),
        lifespan=lifespan,
    )
    if container is not None:
        application.state.container = container

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(v1_router)

    @application.get("/", tags=["Meta"], summary="API service metadata")
    async def root() -> dict[str, Any]:
        return {
            "service": "AdaptiveRAG",
            "version": "0.1.0",
            "docs": "/docs",
            "endpoints": [
                "GET /v1/health",
                "POST /v1/query",
                "POST /v1/query/stream",
                "POST /v1/ingest",
                "POST /v1/ingest/upload",
                "GET /v1/documents",
                "DELETE /v1/documents/{doc_id}",
            ],
        }

    return application


app = create_app()
