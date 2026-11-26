"""FastAPI application factory and ASGI entrypoint for AdaptiveRAG."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from adaptiverag.api.dependencies import AppContainer, create_app_container
from adaptiverag.api.routes import router as v1_router
from adaptiverag.api.security import (
    ApiSecurityMiddleware,
    LocalOnlyGuardMiddleware,
    SlidingWindowRateLimiter,
    api_auth_enabled,
    local_only_mode,
    parse_cors_origins,
    parse_host_list,
)
from adaptiverag.config import Settings, get_settings
from adaptiverag.observability.tracing import configure_logging
from adaptiverag.retrieval.qdrant_store import EmbeddingModelMismatchError

logger = structlog.get_logger(__name__)

_SERVICE_METADATA: dict[str, Any] = {
    "service": "AdaptiveRAG",
    "version": "0.1.0",
    "docs": "/docs",
    "endpoints": [
        "GET /v1/health",
        "GET /v1/models",
        "POST /v1/query",
        "POST /v1/query/stream",
        "POST /v1/feedback",
        "POST /v1/ingest",
        "POST /v1/ingest/upload",
        "GET /v1/documents",
        "DELETE /v1/documents/{doc_id}",
    ],
}


def _resolve_static_ui_root(
    settings: Settings,
    explicit_dir: Path | str | None,
) -> Path | None:
    """Return the resolved built SPA directory when enabled and containing ``index.html``."""
    candidate_dir = Path(explicit_dir) if explicit_dir is not None else settings.static_ui_dir
    if not settings.serve_static_ui or candidate_dir is None:
        return None
    resolved = candidate_dir.resolve()
    if (resolved / "index.html").is_file():
        return resolved
    logger.warning("static_ui_dir_missing_index", static_ui_dir=str(resolved))
    return None


def _warn_about_open_access(settings: Settings) -> None:
    """Log a warning for every setting that lets someone other than the local user in."""
    any_origin = "*" in parse_cors_origins(settings.cors_allow_origins)
    if api_auth_enabled(settings):
        if any_origin:
            logger.warning(
                "cors_allows_any_origin",
                detail="Any website can call this API from a visitor's browser; "
                "requests still need the API key.",
            )
        return

    if "*" in parse_host_list(settings.allowed_hosts_without_auth):
        logger.warning(
            "api_open_without_auth",
            detail="API_AUTH_KEY is unset and ALLOWED_HOSTS_WITHOUT_AUTH=*, so anyone who can "
            "reach this port can query, upload and delete documents.",
        )
    if any_origin:
        logger.warning(
            "cors_allows_any_origin_without_auth",
            detail="API_AUTH_KEY is unset and CORS_ALLOW_ORIGINS=*, so any website open in a "
            "browser on this machine can query this API and upload to it.",
        )


def create_app(
    container: AppContainer | None = None,
    *,
    static_ui_dir: Path | str | None = None,
    rate_limiter: SlidingWindowRateLimiter | None = None,
) -> FastAPI:
    """Create and configure the AdaptiveRAG FastAPI application.

    Passing a pre-built ``container`` enables deterministic dependency injection in tests
    without connecting to live Qdrant or external LLM APIs.
    """
    active_settings = container.settings if container is not None else get_settings()

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
            api_auth_enabled=api_auth_enabled(settings),
            local_only_mode=local_only_mode(settings),
            cors_allow_origins=parse_cors_origins(settings.cors_allow_origins),
            rate_limit_enabled=settings.rate_limit_enabled,
            guardrails_enabled=settings.guardrails_enabled,
        )
        _warn_about_open_access(settings)
        try:
            yield
        finally:
            await managed_container.aclose()
            logger.info("adaptiverag_api_shutdown")

    application = FastAPI(
        title="AdaptiveRAG API",
        version="0.1.0",
        summary="Query-aware Agentic RAG engine with verifiable citations and glass-box observability.",
        description=(
            "Domain-agnostic Agentic RAG engine featuring native Qdrant hybrid search (Dense + BM25 RRF), "
            "local ONNX cross-encoder reranking, 3-stage dynamic top-k & hierarchical parent-child retrieval, "
            "multi-signal CRAG confidence gating, web fallback, multi-layer safety guardrails, "
            "two-tier citation auditing, and Langfuse v5 OpenTelemetry observability."
        ),
        contact={
            "name": "Emirhan Silsüpür",
            "url": "https://github.com/emirhansilsupur/adaptive-rag",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
        lifespan=lifespan,
    )
    if container is not None:
        application.state.container = container

    # Configure optional API Key Auth & Per-IP Sliding-Window Rate Limiting.
    # Added BEFORE CORSMiddleware so CORS runs outermost and attaches CORS headers even on 401/429.
    api_key_value = (
        active_settings.api_auth_key.get_secret_value()
        if api_auth_enabled(active_settings) and active_settings.api_auth_key is not None
        else None
    )
    cors_origins = parse_cors_origins(active_settings.cors_allow_origins)
    effective_limiter = rate_limiter
    if effective_limiter is None and active_settings.rate_limit_enabled:
        effective_limiter = SlidingWindowRateLimiter(
            max_requests=active_settings.rate_limit_requests_per_minute,
            window_seconds=60.0,
        )
    if api_key_value or effective_limiter is not None:
        application.add_middleware(
            ApiSecurityMiddleware,
            api_key=api_key_value,
            rate_limiter=effective_limiter,
            trust_proxy_headers=active_settings.rate_limit_trust_proxy_headers,
        )
    if not api_key_value:
        # No key: answer only this machine (loopback Host) and the allowed browser origins.
        application.add_middleware(
            LocalOnlyGuardMiddleware,
            allowed_origins=cors_origins,
            extra_hosts=parse_host_list(active_settings.allowed_hosts_without_auth),
        )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "Retry-After",
            "X-Guardrail-PII-Redacted",
            "X-Guardrail-Status",
        ],
    )

    application.include_router(v1_router)

    @application.exception_handler(EmbeddingModelMismatchError)
    async def embedding_models_changed(
        _request: Request, exc: EmbeddingModelMismatchError
    ) -> JSONResponse:
        # Searching or adding to the collection needs a re-index first; the message says how.
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"detail": str(exc)})

    @application.get("/api/meta", tags=["Meta"], summary="API service metadata")
    async def api_meta() -> dict[str, Any]:
        return dict(_SERVICE_METADATA)

    static_root = _resolve_static_ui_root(active_settings, static_ui_dir)
    if static_root is not None:
        index_html = static_root / "index.html"

        @application.get("/", include_in_schema=False)
        async def serve_spa_index() -> FileResponse:
            return FileResponse(index_html)

        @application.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa_assets_or_fallback(full_path: str) -> FileResponse:
            # Never shadow unknown API routes with HTML.
            if full_path == "v1" or full_path.startswith("v1/"):
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")

            candidate = (static_root / full_path).resolve()
            if not candidate.is_relative_to(static_root):
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")

            if candidate.is_file():
                headers = (
                    {"Cache-Control": "public, max-age=31536000, immutable"}
                    if full_path.startswith("assets/")
                    else None
                )
                return FileResponse(candidate, headers=headers)

            # Return 404 for missing static bundle assets rather than HTML.
            if full_path.startswith("assets/"):
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")

            return FileResponse(index_html)

        logger.info("static_spa_mounted", static_ui_dir=str(static_root))
    else:

        @application.get("/", tags=["Meta"], summary="API service metadata")
        async def root() -> dict[str, Any]:
            return dict(_SERVICE_METADATA)

    return application


app = create_app()
