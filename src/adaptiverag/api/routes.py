"""Versioned (/v1) HTTP and Server-Sent Events (SSE) API endpoints."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from adaptiverag.api.dependencies import AppContainer, get_container
from adaptiverag.api.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
)
from adaptiverag.ingestion.chunker import ChunkingConfig, chunk_documents
from adaptiverag.ingestion.indexer import index_corpus
from adaptiverag.ingestion.loader import LoadedDocument, detect_document_language
from adaptiverag.llm.cost import BudgetExceededError, CostTracker
from adaptiverag.schemas.response import FinalAnswer

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["AdaptiveRAG"])


def _resolve_workflow_or_503(container: AppContainer) -> Any:
    try:
        return container.ensure_workflow()
    except (ValueError, ImportError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


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

    return HealthResponse(
        status=overall_status,
        qdrant=qdrant_info,
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
        langfuse_enabled=settings.langfuse_enabled,
        llm_cache_enabled=settings.llm_cache_enabled,
        default_max_run_cost_usd=settings.max_run_cost_usd,
    )


@router.post("/query", response_model=FinalAnswer, summary="Run AdaptiveRAG query (batch JSON)")
async def run_query(
    payload: QueryRequest,
    container: AppContainer = Depends(get_container),  # noqa: B008
) -> FinalAnswer:
    """Execute the full AdaptiveRAG agentic pipeline and return a Glass-Box FinalAnswer."""
    workflow = _resolve_workflow_or_503(container)
    ablation = payload.resolve_ablation()
    request_tracker = (
        CostTracker(max_cost_usd=payload.max_cost_usd) if payload.max_cost_usd is not None else None
    )

    try:
        answer: FinalAnswer = await workflow.arun(
            payload.query,
            chat_history=payload.chat_history,
            ablation=ablation,
            cost_tracker=request_tracker,
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
    container: AppContainer = Depends(get_container),  # noqa: B008
) -> EventSourceResponse:
    """Stream live ``node_update`` events followed by the complete ``final_answer`` over SSE."""
    workflow = _resolve_workflow_or_503(container)
    ablation = payload.resolve_ablation()
    request_tracker = (
        CostTracker(max_cost_usd=payload.max_cost_usd) if payload.max_cost_usd is not None else None
    )

    async def event_generator() -> AsyncIterator[ServerSentEvent]:
        try:
            async for item in workflow.astream_events(
                payload.query,
                chat_history=payload.chat_history,
                ablation=ablation,
                cost_tracker=request_tracker,
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

    return EventSourceResponse(
        event_generator(),
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/ingest", response_model=IngestResponse, summary="Ingest local corpus or inline docs")
async def ingest_documents(
    payload: IngestRequest,
    container: AppContainer = Depends(get_container),  # noqa: B008
) -> IngestResponse:
    """Chunk, embed (Dense + BM25), and upsert local corpus files or inline documents into Qdrant."""
    chunk_cfg = ChunkingConfig(
        max_tokens=payload.max_tokens,
        overlap_tokens=payload.overlap_tokens,
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
        chunks = chunk_documents(loaded_docs, config=chunk_cfg)
        created = await container.store.ensure_collection(recreate=payload.recreate_collection)
        upserted = await container.store.upsert_chunks(
            chunks,
            batch_size=payload.batch_size,
            replace_existing_docs=not payload.recreate_collection,
        )
        total_points = await container.store.count_points()
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)

        logger.info(
            "api_inline_ingest_complete",
            documents=len(loaded_docs),
            chunks=len(chunks),
            upserted=upserted,
            total_points=total_points,
        )
        return IngestResponse(
            collection=container.store.collection_name,
            source="inline",
            documents_processed=len(loaded_docs),
            chunks_created=len(chunks),
            points_upserted=upserted,
            total_collection_points=total_points,
            collection_created=created,
            elapsed_ms=elapsed_ms,
        )

    data_dir = Path(payload.corpus_dir) if payload.corpus_dir else Path("data/raw")
    if not data_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corpus directory not found: {data_dir}",
        )

    report = await index_corpus(
        data_dir=data_dir,
        store=container.store,
        chunk_config=chunk_cfg,
        recreate=payload.recreate_collection,
        batch_size=payload.batch_size,
    )
    return IngestResponse(
        collection=report.collection_name,
        source=report.data_dir,
        documents_processed=report.documents_loaded,
        chunks_created=report.chunks_created,
        points_upserted=report.points_upserted,
        total_collection_points=report.total_collection_points,
        collection_created=report.collection_created,
        elapsed_ms=round(report.elapsed_seconds * 1000.0, 2),
    )
