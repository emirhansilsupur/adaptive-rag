"""Versioned (/v1) HTTP and Server-Sent Events (SSE) API endpoints."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

import structlog
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from adaptiverag.api.dependencies import AppContainer, get_container
from adaptiverag.api.schemas import (
    DeleteDocumentResponse,
    DocumentListResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    IndexedDocumentInfo,
    IngestRequest,
    IngestResponse,
    LLMOverride,
    ModelsCatalogResponse,
    ProviderCatalogItem,
    QueryRequest,
)
from adaptiverag.api.security import api_auth_enabled, local_only_mode
from adaptiverag.graph.workflow import AdaptiveRAGWorkflow
from adaptiverag.ingestion.chunker import ChunkingConfig
from adaptiverag.ingestion.indexer import (
    index_corpus,
    index_documents,
    index_web_crawl,
    index_web_urls,
)
from adaptiverag.ingestion.loader import (
    LoadedDocument,
    aload_document_bytes,
    detect_document_language,
)
from adaptiverag.ingestion.pdf_parser import PdfOcrMode, health_pdf_ocr_summary
from adaptiverag.ingestion.web_loader import (
    FirecrawlError,
    FirecrawlUnavailableError,
    UnsafeURLError,
)
from adaptiverag.llm.cost import BudgetExceededError, CostTracker
from adaptiverag.llm.factory import (
    PROVIDER_CATALOG,
    SUPPORTED_PROVIDERS,
    ModelBundle,
    check_provider_configured,
    check_provider_sdk_installed,
)
from adaptiverag.schemas.response import FinalAnswer
from adaptiverag.verification.guardrails import (
    GuardrailDecision,
    InputGuardrailConfig,
    screen_query_input,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["AdaptiveRAG"])


def _resolve_workflow_or_503(
    container: AppContainer,
    *,
    models_override: ModelBundle | None = None,
) -> AdaptiveRAGWorkflow:
    try:
        workflow = container.ensure_workflow()
    except (ValueError, ImportError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    if models_override is None and workflow.runtime.models is None:
        # The server has no LLM of its own and this request didn't bring a key.
        _, reason = container.check_llm_configured()
        detail = reason or "This server has no LLM configured."
        if container.settings.allow_client_provider_keys:
            detail += " API clients can send their own key in the X-Provider-API-Key header."
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
    return workflow


def _resolve_llm_override_or_400(
    override: LLMOverride | None,
    container: AppContainer,
    *,
    provider_api_key: str | None = None,
) -> tuple[ModelBundle | None, float | None]:
    cleaned_key = provider_api_key.strip() if provider_api_key else None
    if (override is None or override.is_empty) and not cleaned_key:
        return None, None
    try:
        return container.resolve_model_bundle(
            provider=override.provider if override is not None else None,
            fast_model=override.fast_model if override is not None else None,
            generation_model=override.generation_model if override is not None else None,
            temperature=override.temperature if override is not None else None,
            api_key_override=cleaned_key,
        )
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    except (ValueError, ImportError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


def _screen_query_or_400(payload: QueryRequest, container: AppContainer) -> GuardrailDecision:
    """Apply $0 deterministic prompt-injection and PII guardrails before invoking LLMs."""
    settings = container.settings
    cfg = InputGuardrailConfig(
        enabled=settings.guardrails_enabled,
        block_prompt_injection=settings.guardrails_block_prompt_injection,
        pii_mode=settings.guardrails_pii_mode,
    )
    decision = screen_query_input(
        payload.query,
        payload.chat_history,
        custom_instructions=payload.custom_instructions,
        config=cfg,
    )
    if not decision.allowed:
        logger.warning(
            "input_guardrail_blocked",
            rules=decision.triggered_rules,
            pii_types=decision.pii_types,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=decision.blocked_reason or "Request blocked by input safety guardrails.",
            headers={"X-Guardrail-Status": "blocked"},
        )
    if decision.redacted:
        logger.info(
            "input_guardrail_pii_redacted",
            pii_counts={finding.entity_type: finding.count for finding in decision.pii_findings},
        )
    return decision


@router.get("/health", response_model=HealthResponse, summary="Check API, Qdrant, and LLM health")
async def health_check(
    container: AppContainer = Depends(get_container),  # noqa: B008
) -> HealthResponse:
    """Return service status, Qdrant collection readiness, and active model configuration."""
    settings = container.settings
    try:
        qdrant_info = await container.store.health_check()
        qdrant_reachable = True
    except Exception as exc:
        logger.warning("qdrant_health_check_failed", error=str(exc))
        qdrant_reachable = False
        qdrant_info = {
            "status": "unreachable",
            "collection": container.store.collection_name,
            "collection_exists": False,
            "points_count": 0,
            "dense_model": settings.dense_embedding_model,
            "sparse_model": settings.sparse_embedding_model,
            "error": str(exc),
        }

    llm_ok, llm_error = container.check_llm_configured()
    overall_status: Literal["ok", "degraded"] = (
        "ok" if (qdrant_reachable and llm_ok) else "degraded"
    )

    fallback_providers = list(getattr(container.fallback_retriever, "provider_names", []))
    fallback_enabled = bool(
        getattr(container.fallback_retriever, "enabled", settings.web_fallback_enabled)
    )

    if container.firecrawl_client is not None:
        firecrawl_info = await container.firecrawl_client.ahealth_check(timeout_seconds=1.5)
    else:
        firecrawl_info = {
            "status": "disabled",
            "base_url": settings.firecrawl_local_url,
        }

    langfuse_info = container.get_observability().health_summary()
    langfuse_info["prompt_management"] = container.get_prompt_manager().health_summary()

    security_info = {
        "api_key_auth_enabled": api_auth_enabled(settings),
        # Without a key, only loopback hosts (plus ALLOWED_HOSTS_WITHOUT_AUTH) are answered.
        "local_only_mode": local_only_mode(settings),
        "rate_limit_enabled": settings.rate_limit_enabled,
        "rate_limit_requests_per_minute": settings.rate_limit_requests_per_minute,
        "guardrails_enabled": settings.guardrails_enabled,
        "prompt_injection_protection": (
            settings.guardrails_enabled and settings.guardrails_block_prompt_injection
        ),
        "pii_mode": settings.guardrails_pii_mode if settings.guardrails_enabled else "off",
    }

    parent_child_info = {
        "enabled": settings.enable_parent_child_retrieval,
        "parent_max_tokens": settings.parent_max_tokens,
        "child_max_tokens": settings.child_max_tokens,
        "child_overlap_tokens": settings.child_overlap_tokens,
    }
    pdf_ocr_info = await asyncio.to_thread(health_pdf_ocr_summary, settings)

    return HealthResponse(
        status=overall_status,
        qdrant=qdrant_info,
        firecrawl=firecrawl_info,
        langfuse=langfuse_info,
        security=security_info,
        parent_child_retrieval=parent_child_info,
        pdf_ocr=pdf_ocr_info,
        llm_configured=llm_ok,
        llm_error=llm_error,
        llm_provider=settings.llm_provider,
        fast_model=settings.llm_fast_model,
        generation_model=settings.llm_generation_model,
        dense_embedding_model=settings.dense_embedding_model,
        sparse_embedding_model=settings.sparse_embedding_model,
        reranker_model=settings.reranker_model,
        web_fallback_enabled=fallback_enabled,
        web_fallback_providers=fallback_providers,
        langfuse_enabled=bool(langfuse_info.get("enabled", settings.langfuse_enabled)),
        llm_cache_enabled=settings.llm_cache_enabled,
        default_max_run_cost_usd=settings.max_run_cost_usd,
    )


@router.get(
    "/models",
    response_model=ModelsCatalogResponse,
    summary="List configured LLM providers, curated models, and live local models",
)
async def list_models_catalog(
    container: AppContainer = Depends(get_container),  # noqa: B008
) -> ModelsCatalogResponse:
    """Return safe, non-secret provider readiness and recommended/detected model lists."""
    settings = container.settings
    local_reachable, local_detected, local_err = await container.probe_local_models()

    items: list[ProviderCatalogItem] = []
    for provider_id in SUPPORTED_PROVIDERS:
        entry = PROVIDER_CATALOG[provider_id]
        sdk_ok, _ = check_provider_sdk_installed(provider_id)
        configured, cfg_detail = check_provider_configured(settings, provider_id)
        is_default = provider_id == settings.llm_provider

        rec_fast = list(entry.recommended_fast_models)
        rec_gen = list(entry.recommended_generation_models)
        if is_default:
            if settings.llm_fast_model and settings.llm_fast_model not in rec_fast:
                rec_fast.insert(0, settings.llm_fast_model)
            if settings.llm_generation_model and settings.llm_generation_model not in rec_gen:
                rec_gen.insert(0, settings.llm_generation_model)

        if provider_id == "ollama":
            for detected_id in reversed(local_detected):
                if detected_id not in rec_fast:
                    rec_fast.insert(0, detected_id)
                if detected_id not in rec_gen:
                    rec_gen.insert(0, detected_id)
            status_detail = (
                cfg_detail if not configured else (None if local_reachable else local_err)
            )
            items.append(
                ProviderCatalogItem(
                    id=provider_id,
                    label=entry.label,
                    is_local=True,
                    requires_api_key=False,
                    sdk_installed=sdk_ok,
                    configured=configured,
                    reachable=local_reachable,
                    status_detail=status_detail,
                    default_fast_model=(
                        settings.llm_fast_model
                        if is_default
                        else (local_detected[0] if local_detected else entry.default_fast_model)
                    ),
                    default_generation_model=(
                        settings.llm_generation_model
                        if is_default
                        else (
                            local_detected[0] if local_detected else entry.default_generation_model
                        )
                    ),
                    recommended_fast_models=rec_fast,
                    recommended_generation_models=rec_gen,
                    detected_models=local_detected,
                )
            )
        else:
            items.append(
                ProviderCatalogItem(
                    id=provider_id,
                    label=entry.label,
                    is_local=False,
                    requires_api_key=True,
                    sdk_installed=sdk_ok,
                    configured=configured,
                    reachable=None,
                    status_detail=cfg_detail,
                    default_fast_model=(
                        settings.llm_fast_model if is_default else entry.default_fast_model
                    ),
                    default_generation_model=(
                        settings.llm_generation_model
                        if is_default
                        else entry.default_generation_model
                    ),
                    recommended_fast_models=rec_fast,
                    recommended_generation_models=rec_gen,
                    detected_models=[],
                )
            )

    return ModelsCatalogResponse(
        default_provider=settings.llm_provider,
        default_fast_model=settings.llm_fast_model,
        default_generation_model=settings.llm_generation_model,
        default_temperature=settings.llm_temperature,
        ollama_base_url=settings.ollama_base_url,
        allow_client_provider_keys=settings.allow_client_provider_keys,
        providers=items,
    )


@router.post("/query", response_model=FinalAnswer, summary="Run AdaptiveRAG query (batch JSON)")
async def run_query(
    payload: QueryRequest,
    response: Response,
    provider_api_key: str | None = Header(default=None, alias="X-Provider-API-Key"),
    container: AppContainer = Depends(get_container),  # noqa: B008
) -> FinalAnswer:
    """Execute the full AdaptiveRAG agentic pipeline and return a Glass-Box FinalAnswer."""
    guardrail = _screen_query_or_400(payload, container)
    if guardrail.redacted:
        response.headers["X-Guardrail-PII-Redacted"] = ",".join(guardrail.pii_types)

    models_override, temp_override = _resolve_llm_override_or_400(
        payload.llm_override,
        container,
        provider_api_key=provider_api_key,
    )
    workflow = _resolve_workflow_or_503(container, models_override=models_override)
    ablation = payload.resolve_ablation()
    effective_max_cost = (
        payload.max_cost_usd
        if payload.max_cost_usd is not None
        else container.settings.max_run_cost_usd
    )
    request_tracker = CostTracker(max_cost_usd=effective_max_cost)

    try:
        answer: FinalAnswer = await workflow.arun(
            guardrail.sanitized_query,
            chat_history=guardrail.sanitized_history,
            ablation=ablation,
            custom_instructions=guardrail.sanitized_custom_instructions,
            models=models_override,
            temperature=temp_override,
            cost_tracker=request_tracker,
            session_id=payload.session_id,
            user_id=payload.user_id,
            preset_name=payload.resolve_preset_label(),
            pre_screened=True,
        )
        return answer
    except BudgetExceededError as exc:
        logger.warning("query_budget_exceeded", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=str(exc),
        ) from exc


@router.post(
    "/query/stream",
    summary="Run AdaptiveRAG query with Server-Sent Events (SSE) node trace streaming",
)
async def stream_query(
    payload: QueryRequest,
    http_request: Request,
    provider_api_key: str | None = Header(default=None, alias="X-Provider-API-Key"),
    container: AppContainer = Depends(get_container),  # noqa: B008
) -> EventSourceResponse:
    """Stream live ``node_update`` events followed by the complete ``final_answer`` over SSE."""
    guardrail = _screen_query_or_400(payload, container)
    models_override, temp_override = _resolve_llm_override_or_400(
        payload.llm_override,
        container,
        provider_api_key=provider_api_key,
    )
    workflow = _resolve_workflow_or_503(container, models_override=models_override)
    ablation = payload.resolve_ablation()
    effective_max_cost = (
        payload.max_cost_usd
        if payload.max_cost_usd is not None
        else container.settings.max_run_cost_usd
    )
    request_tracker = CostTracker(max_cost_usd=effective_max_cost)

    async def event_generator() -> AsyncIterator[ServerSentEvent]:
        try:
            async for item in workflow.astream_events(
                guardrail.sanitized_query,
                chat_history=guardrail.sanitized_history,
                ablation=ablation,
                custom_instructions=guardrail.sanitized_custom_instructions,
                models=models_override,
                temperature=temp_override,
                cost_tracker=request_tracker,
                session_id=payload.session_id,
                user_id=payload.user_id,
                preset_name=payload.resolve_preset_label(),
                pre_screened=True,
            ):
                if await http_request.is_disconnected():
                    logger.info("sse_client_disconnected")
                    break

                event_type = str(item.get("event", "message"))
                if event_type == "final_answer":
                    data_payload: Any = item.get("data", {})
                else:
                    data_payload = {
                        "node": item.get("node", ""),
                        "trace": item.get("trace", []),
                        "usage": item.get("usage", {}),
                    }

                yield ServerSentEvent(
                    event=event_type,
                    data=json.dumps(data_payload, ensure_ascii=False),
                )
        except BudgetExceededError as exc:
            logger.warning("sse_query_budget_exceeded", error=str(exc))
            yield ServerSentEvent(
                event="error",
                data=json.dumps(
                    {"code": "budget_exceeded", "message": str(exc)},
                    ensure_ascii=False,
                ),
            )
        except Exception as exc:
            logger.exception("sse_query_stream_error", error=str(exc))
            yield ServerSentEvent(
                event="error",
                data=json.dumps(
                    {"code": "internal_error", "message": str(exc)},
                    ensure_ascii=False,
                ),
            )

    sse_headers: dict[str, str] = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    if guardrail.redacted:
        sse_headers["X-Guardrail-PII-Redacted"] = ",".join(guardrail.pii_types)

    return EventSourceResponse(
        event_generator(),
        headers=sse_headers,
    )


def _build_chunking_config(
    container: AppContainer,
    *,
    max_tokens: int,
    overlap_tokens: int,
    parent_child: bool | None = None,
    parent_max_tokens: int | None = None,
    child_max_tokens: int | None = None,
    child_overlap_tokens: int | None = None,
) -> ChunkingConfig:
    settings = container.settings
    use_pc = parent_child if parent_child is not None else settings.enable_parent_child_retrieval
    try:
        return ChunkingConfig(
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            parent_child_enabled=use_pc,
            parent_max_tokens=(
                parent_max_tokens if parent_max_tokens is not None else settings.parent_max_tokens
            ),
            child_max_tokens=(
                child_max_tokens if child_max_tokens is not None else settings.child_max_tokens
            ),
            child_overlap_tokens=(
                child_overlap_tokens
                if child_overlap_tokens is not None
                else settings.child_overlap_tokens
            ),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc


def _report_to_ingest_response(report: Any) -> IngestResponse:
    return IngestResponse(
        collection=report.collection_name,
        source=report.data_dir,
        documents_processed=report.documents_loaded,
        chunks_created=report.chunks_created,
        points_upserted=report.points_upserted,
        total_collection_points=report.total_collection_points,
        collection_created=report.collection_created,
        elapsed_ms=round(report.elapsed_seconds * 1000.0, 2),
        source_urls=report.source_urls,
        parent_child_enabled=report.parent_child_enabled,
        unique_parents_created=report.unique_parents_created,
        ocr_pages_processed=report.ocr_pages_processed,
        ocr_backends_used=report.ocr_backends_used,
    )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest local corpus, inline docs, or live Firecrawl URLs/docs sites",
)
async def ingest_documents(
    payload: IngestRequest,
    container: AppContainer = Depends(get_container),  # noqa: B008
) -> IngestResponse:
    """Chunk, embed (Dense + BM25), and upsert local files, inline docs, or Firecrawl web pages."""
    chunk_cfg = _build_chunking_config(
        container,
        max_tokens=payload.max_tokens,
        overlap_tokens=payload.overlap_tokens,
        parent_child=payload.parent_child,
        parent_max_tokens=payload.parent_max_tokens,
        child_max_tokens=payload.child_max_tokens,
        child_overlap_tokens=payload.child_overlap_tokens,
    )

    if payload.documents is not None:
        started = time.perf_counter()
        loaded_docs = [
            LoadedDocument(
                doc_id=doc.doc_id,
                title=doc.title,
                content=doc.content,
                source_path=f"inline://{doc.doc_id}",
                doc_type=doc.doc_type,
                language=doc.language or detect_document_language(doc.content, doc.doc_id),
                metadata=dict(doc.metadata),
            )
            for doc in payload.documents
        ]
        report = await index_documents(
            loaded_docs,
            container.store,
            source="inline",
            chunk_config=chunk_cfg,
            recreate=payload.recreate_collection,
            batch_size=payload.batch_size,
            started_at=started,
        )
        return _report_to_ingest_response(report)

    if payload.urls is not None or payload.crawl_url is not None:
        if container.firecrawl_client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Firecrawl Local ingestion is disabled. Set FIRECRAWL_LOCAL_URL "
                    "(e.g., http://localhost:3002) and run `firecrawl-local serve`."
                ),
            )
        try:
            if payload.urls is not None:
                report = await index_web_urls(
                    urls=payload.urls,
                    store=container.store,
                    client=container.firecrawl_client,
                    chunk_config=chunk_cfg,
                    recreate=payload.recreate_collection,
                    batch_size=payload.batch_size,
                    only_main_content=payload.only_main_content,
                )
            else:
                assert payload.crawl_url is not None
                report = await index_web_crawl(
                    crawl_url=payload.crawl_url,
                    store=container.store,
                    client=container.firecrawl_client,
                    limit=payload.crawl_limit,
                    max_depth=payload.crawl_max_depth,
                    chunk_config=chunk_cfg,
                    recreate=payload.recreate_collection,
                    batch_size=payload.batch_size,
                    only_main_content=payload.only_main_content,
                )
        except UnsafeURLError as exc:
            logger.warning("api_web_ingest_unsafe_url", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc
        except FirecrawlUnavailableError as exc:
            logger.warning("api_web_ingest_firecrawl_unavailable", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        except FirecrawlError as exc:
            logger.warning("api_web_ingest_firecrawl_error", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
            ) from exc

        return _report_to_ingest_response(report)

    allowed_root = container.settings.allowed_corpus_root.resolve()
    raw_target = (
        Path(payload.corpus_dir)
        if payload.corpus_dir is not None
        else (container.settings.allowed_corpus_root / "raw")
    )
    resolved_dir = raw_target.resolve()
    if not resolved_dir.is_relative_to(allowed_root):
        logger.warning(
            "api_ingest_corpus_dir_blocked",
            requested_dir=str(raw_target),
            allowed_root=str(allowed_root),
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Corpus directory must reside inside the configured allowed corpus root "
                f"('{container.settings.allowed_corpus_root}')."
            ),
        )
    if not resolved_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corpus directory not found: {raw_target}",
        )

    ingest_tracker = CostTracker(max_cost_usd=container.settings.max_run_cost_usd)
    try:
        report = await index_corpus(
            data_dir=resolved_dir,
            store=container.store,
            chunk_config=chunk_cfg,
            recreate=payload.recreate_collection,
            batch_size=payload.batch_size,
            ocr_mode=payload.ocr_mode,
            vision_ocr_fn=container.vision_ocr_fn,
            settings=container.settings,
            cost_tracker=ingest_tracker,
        )
    except BudgetExceededError as exc:
        logger.warning("ingest_corpus_budget_exceeded", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    return _report_to_ingest_response(report)


MAX_UPLOAD_FILES = 20
MAX_UPLOAD_FILE_BYTES = 15 * 1024 * 1024  # 15 MiB per file


@router.post(
    "/ingest/upload",
    response_model=IngestResponse,
    summary="Upload and index .md, .txt, or .pdf files into Qdrant",
)
async def upload_and_ingest_files(
    files: list[UploadFile] = File(...),  # noqa: B008
    recreate_collection: bool = Form(default=False),
    max_tokens: int = Form(default=512, ge=64, le=4096),
    overlap_tokens: int = Form(default=64, ge=0, le=512),
    parent_child: bool | None = Form(default=None),
    ocr_mode: PdfOcrMode | None = Form(default=None),  # noqa: B008
    container: AppContainer = Depends(get_container),  # noqa: B008
) -> IngestResponse:
    """Parse uploaded Markdown, plain text, or PDF files and upsert chunks into Qdrant."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="At least one file must be uploaded.",
        )
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"At most {MAX_UPLOAD_FILES} files may be uploaded per request.",
        )
    if overlap_tokens >= max_tokens:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"overlap_tokens ({overlap_tokens}) must be < max_tokens ({max_tokens}).",
        )

    started = time.perf_counter()
    upload_tracker = CostTracker(max_cost_usd=container.settings.max_run_cost_usd)
    loaded_docs: list[LoadedDocument] = []
    for upload in files:
        filename = upload.filename or "untitled"
        raw = await upload.read()
        await upload.close()
        if len(raw) > MAX_UPLOAD_FILE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=(
                    f"File '{filename}' exceeds maximum size of "
                    f"{MAX_UPLOAD_FILE_BYTES // (1024 * 1024)} MB."
                ),
            )
        try:
            doc = await aload_document_bytes(
                filename,
                raw,
                ocr_mode=ocr_mode,
                vision_ocr_fn=container.vision_ocr_fn,
                settings=container.settings,
                cost_tracker=upload_tracker,
            )
            loaded_docs.append(doc)
        except BudgetExceededError as exc:
            logger.warning("upload_ocr_budget_exceeded", filename=filename, error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=str(exc),
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc

    chunk_cfg = _build_chunking_config(
        container,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        parent_child=parent_child,
    )
    report = await index_documents(
        loaded_docs,
        container.store,
        source="upload",
        chunk_config=chunk_cfg,
        recreate=recreate_collection,
        started_at=started,
    )
    return _report_to_ingest_response(report)


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List indexed documents in the active Qdrant collection",
)
async def list_indexed_documents(
    container: AppContainer = Depends(get_container),  # noqa: B008
) -> DocumentListResponse:
    """Return all indexed documents grouped by doc_id with chunk and token statistics."""
    try:
        raw_docs = await container.store.list_documents()
    except Exception as exc:
        logger.warning("list_documents_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Unable to read documents from Qdrant: {exc}",
        ) from exc

    docs = [IndexedDocumentInfo.model_validate(item) for item in raw_docs]
    total_chunks = sum(d.chunk_count for d in docs)
    return DocumentListResponse(
        collection=container.store.collection_name,
        total_documents=len(docs),
        total_chunks=total_chunks,
        documents=docs,
    )


@router.delete(
    "/documents/{doc_id:path}",
    response_model=DeleteDocumentResponse,
    summary="Delete all chunks of an indexed document from Qdrant",
)
async def delete_indexed_document(
    doc_id: str,
    container: AppContainer = Depends(get_container),  # noqa: B008
) -> DeleteDocumentResponse:
    """Remove every Qdrant point belonging to ``doc_id``."""
    cleaned_id = doc_id.strip()
    if not cleaned_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="doc_id must not be empty.",
        )

    try:
        deleted_chunks, remaining = await container.store.delete_document(cleaned_id)
    except Exception as exc:
        logger.warning("delete_document_failed", doc_id=cleaned_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Unable to delete document from Qdrant: {exc}",
        ) from exc

    if deleted_chunks == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{cleaned_id}' not found in collection '{container.store.collection_name}'.",
        )

    return DeleteDocumentResponse(
        collection=container.store.collection_name,
        doc_id=cleaned_id,
        deleted_chunks=deleted_chunks,
        remaining_collection_points=remaining,
    )


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Record human-in-the-loop feedback score for a traced answer",
)
async def submit_feedback(
    payload: FeedbackRequest,
    container: AppContainer = Depends(get_container),  # noqa: B008
) -> FeedbackResponse:
    """Record thumbs-up/down feedback on a trace_id and forward to Langfuse when enabled."""
    outcome = await asyncio.to_thread(
        container.get_observability().record_user_feedback,
        trace_id=payload.trace_id,
        score=payload.score,
        comment=payload.comment,
        session_id=payload.session_id,
        query=payload.query,
    )
    return FeedbackResponse(
        status="recorded",
        trace_id=outcome.trace_id,
        score=outcome.score,
        comment=outcome.comment,
        persisted_to_langfuse=outcome.persisted_to_langfuse,
        langfuse_error=outcome.langfuse_error,
    )
