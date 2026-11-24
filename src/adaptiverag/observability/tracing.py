"""Structured logging (structlog) and Langfuse v5 OpenTelemetry observability layer.

Provides:
- Fail-open Langfuse v5 root trace management compatible with both batch (``arun``)
  and async generator SSE streaming (``astream_events``) without OTEL contextvar leaks.
- Rich child observations for retrieval/Dynamic-K, SQLite cache hits, CRAG confidence
  evaluation, and two-tier citation guardrails.
- Automated online quality score publishing (CRAG confidence, citation precision,
  claim support rate, fabricated citation count, abstention/fallback flags, latency, USD cost).
- Human-in-the-loop user feedback score recording (``POST /v1/feedback``).
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import structlog

from adaptiverag.config import Settings
from adaptiverag.schemas.response import FinalAnswer

logger = structlog.get_logger(__name__)

ScoreDataType = Literal["NUMERIC", "CATEGORICAL", "BOOLEAN"]


@dataclass(frozen=True)
class OnlineScoreSpec:
    """Typed score payload pushed to Langfuse after each pipeline execution."""

    name: str
    value: float | str
    data_type: ScoreDataType = "NUMERIC"
    comment: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceRunHandle:
    """Active per-request trace handle returned by ``LangfuseObservability.start_trace``."""

    trace_id: str
    trace_url: str | None = None
    root_span: Any | None = None
    callback_handler: Any | None = None
    langfuse_metadata: dict[str, Any] = field(default_factory=dict)
    langfuse_active: bool = False


@dataclass(frozen=True)
class FeedbackRecordResult:
    """Outcome of recording human-in-the-loop feedback for a trace."""

    trace_id: str
    score: float
    comment: str | None
    persisted_to_langfuse: bool
    langfuse_error: str | None = None


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structured console logging via structlog."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        cache_logger_on_first_use=True,
    )


def _generate_trace_id(client: Any | None = None) -> str:
    """Generate a valid 32-character lowercase hex OpenTelemetry trace ID."""
    if client is not None:
        create_fn = getattr(client, "create_trace_id", None)
        if callable(create_fn):
            try:
                generated = create_fn()
                if isinstance(generated, str) and generated.strip():
                    return generated.strip()
            except Exception:
                pass
    return uuid.uuid4().hex


def build_online_quality_scores(answer: FinalAnswer) -> list[OnlineScoreSpec]:
    """Derive AdaptiveRAG's online quality, grounding, and FinOps scores from a FinalAnswer."""
    scores: list[OnlineScoreSpec] = []

    if answer.confidence is not None:
        conf = answer.confidence
        scores.append(
            OnlineScoreSpec(
                name="crag_confidence",
                value=round(float(conf.final_score), 6),
                data_type="NUMERIC",
                comment=conf.reasoning[:240] if conf.reasoning else None,
                metadata={
                    "top1_score": conf.top1_score,
                    "mean_topk_score": conf.mean_topk_score,
                    "subquery_coverage": conf.subquery_coverage,
                    "deterministic_score": conf.deterministic_score,
                    "crag_grade": conf.crag_grade.value if conf.crag_grade else None,
                    "recommended_action": conf.recommended_action.value,
                },
            )
        )
        scores.append(
            OnlineScoreSpec(
                name="crag_level",
                value=conf.level.value,
                data_type="CATEGORICAL",
                comment=f"Action: {conf.recommended_action.value}",
            )
        )

    if answer.citation_report is not None:
        cite = answer.citation_report
        scores.extend(
            [
                OnlineScoreSpec(
                    name="citation_precision",
                    value=round(float(cite.citation_precision), 6),
                    data_type="NUMERIC",
                ),
                OnlineScoreSpec(
                    name="claim_support_rate",
                    value=round(float(cite.claim_support_rate), 6),
                    data_type="NUMERIC",
                ),
                OnlineScoreSpec(
                    name="citation_passed",
                    value=1.0 if cite.passed else 0.0,
                    data_type="BOOLEAN",
                    comment=(
                        "Passed two-tier citation audit"
                        if cite.passed
                        else f"Invalid IDs: {cite.invalid_citation_ids}"
                    ),
                ),
                OnlineScoreSpec(
                    name="fabricated_citations",
                    value=float(len(cite.invalid_citation_ids)),
                    data_type="NUMERIC",
                ),
            ]
        )

    scores.extend(
        [
            OnlineScoreSpec(
                name="selected_k",
                value=float(len(answer.evidence_chunks)),
                data_type="NUMERIC",
            ),
            OnlineScoreSpec(
                name="fallback_used",
                value=1.0 if answer.fallback_used else 0.0,
                data_type="BOOLEAN",
            ),
            OnlineScoreSpec(
                name="abstained",
                value=1.0 if answer.abstained else 0.0,
                data_type="BOOLEAN",
            ),
            OnlineScoreSpec(
                name="estimated_cost_usd",
                value=round(float(answer.usage.estimated_cost_usd), 8),
                data_type="NUMERIC",
            ),
            OnlineScoreSpec(
                name="latency_ms",
                value=round(float(answer.total_latency_ms), 2),
                data_type="NUMERIC",
            ),
            OnlineScoreSpec(
                name="cache_hits",
                value=float(answer.usage.cache_hits),
                data_type="NUMERIC",
            ),
        ]
    )
    return scores


class LangfuseObservability:
    """Managed, fail-open Langfuse v5 observability adapter for AdaptiveRAG.

    Can be constructed from ``Settings`` (lazy client initialization) or with an injected
    mock/fake client for deterministic ``$0`` offline tests.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        host: str = "https://cloud.langfuse.com",
        public_key: str = "",
        secret_key: str | None = None,
        environment: str = "development",
        release: str | None = "0.1.0",
        client: Any | None = None,
        callback_factory: Callable[[str | None, dict[str, Any]], Any] | None = None,
    ) -> None:
        self._enabled = bool(enabled)
        self._host = host.strip().rstrip("/") or "https://cloud.langfuse.com"
        self._public_key = public_key.strip()
        self._secret_key = (secret_key or "").strip()
        self._environment = environment.strip() or "development"
        self._release = release.strip() if release else None
        self._client: Any | None = client
        self._client_initialized = client is not None
        self._callback_factory = callback_factory

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        client: Any | None = None,
        callback_factory: Callable[[str | None, dict[str, Any]], Any] | None = None,
    ) -> LangfuseObservability:
        """Build a LangfuseObservability manager from application Settings."""
        secret = (
            settings.langfuse_secret_key.get_secret_value()
            if settings.langfuse_secret_key is not None
            else None
        )
        return cls(
            enabled=settings.langfuse_enabled,
            host=settings.langfuse_host,
            public_key=settings.langfuse_public_key,
            secret_key=secret,
            environment=settings.langfuse_environment,
            release=settings.langfuse_release,
            client=client,
            callback_factory=callback_factory,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def configured(self) -> bool:
        if self._client is not None:
            return self._enabled
        return bool(self._enabled and self._public_key and self._secret_key)

    @property
    def host(self) -> str:
        return self._host

    @property
    def environment(self) -> str:
        return self._environment

    def health_summary(self) -> dict[str, Any]:
        """Return non-secret observability readiness info for GET /v1/health."""
        is_cloud = "cloud.langfuse.com" in self._host.lower()
        mode = "disabled"
        if self._enabled:
            if not self.configured:
                mode = "misconfigured"
            elif is_cloud:
                mode = "cloud"
            else:
                mode = "self_hosted"
        return {
            "enabled": self._enabled,
            "configured": self.configured,
            "mode": mode,
            "host": self._host,
            "environment": self._environment,
        }

    def get_client(self) -> Any | None:
        """Lazily initialize and return the Langfuse v5 client when enabled and configured."""
        if not self.configured:
            return None
        if self._client_initialized:
            return self._client

        self._client_initialized = True
        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=self._public_key,
                secret_key=self._secret_key,
                host=self._host,
                environment=self._environment,
                release=self._release,
            )
            return self._client
        except Exception as exc:
            logger.warning("langfuse_client_init_failed", host=self._host, error=str(exc))
            self._client = None
            return None

    def _build_callback_handler(
        self,
        *,
        trace_id: str,
        parent_span_id: str | None = None,
    ) -> Any | None:
        trace_context: dict[str, Any] = {"trace_id": trace_id}
        if parent_span_id:
            trace_context["parent_span_id"] = parent_span_id

        if self._callback_factory is not None:
            try:
                return self._callback_factory(self._public_key or None, trace_context)
            except Exception as exc:
                logger.warning("langfuse_callback_factory_failed", error=str(exc))
                return None

        client = self.get_client()
        if client is None:
            return None

        # Only build the real LangChain CallbackHandler when backed by a real Langfuse client.
        if not self._public_key:
            return None

        try:
            from langfuse.langchain import CallbackHandler

            return CallbackHandler(
                public_key=self._public_key,
                trace_context=trace_context,  # type: ignore[arg-type]
            )
        except Exception as exc:
            logger.warning("langfuse_callback_init_failed", error=str(exc))
            return None

    def resolve_trace_url(self, trace_id: str) -> str | None:
        """Resolve the Langfuse web UI deep-link URL for a trace ID."""
        client = self.get_client()
        if client is None:
            return None
        get_url = getattr(client, "get_trace_url", None)
        if callable(get_url):
            try:
                resolved = get_url(trace_id=trace_id)
                if isinstance(resolved, str) and resolved.strip():
                    return resolved.strip()
            except Exception as exc:
                logger.debug("langfuse_get_trace_url_fallback", error=str(exc))
        return f"{self._host}/trace/{trace_id}"

    def start_trace(
        self,
        *,
        query: str,
        session_id: str | None = None,
        user_id: str | None = None,
        preset_name: str | None = None,
        ablation: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceRunHandle:
        """Start a manual-lifecycle root trace handle safe for both batch and SSE streams."""
        client = self.get_client()
        trace_id = _generate_trace_id(client)

        effective_tags = [
            "adaptiverag",
            f"env:{self._environment}",
            *(tags or []),
        ]
        if preset_name:
            effective_tags.append(f"preset:{preset_name}")
        # Preserve order while deduplicating tags
        deduped_tags = list(dict.fromkeys(t for t in effective_tags if t))

        langfuse_metadata: dict[str, Any] = {
            "langfuse_trace_name": "adaptiverag.query",
            "langfuse_tags": deduped_tags,
            "environment": self._environment,
        }
        if session_id:
            langfuse_metadata["langfuse_session_id"] = session_id
        if user_id:
            langfuse_metadata["langfuse_user_id"] = user_id
        if preset_name:
            langfuse_metadata["preset"] = preset_name
        # Note: full `ablation` dict is stored on root `span_meta` below (not propagated
        # via RunnableConfig metadata, where Langfuse v5 enforces a 200-char value cap).
        if metadata:
            langfuse_metadata.update(metadata)

        if client is None:
            return TraceRunHandle(
                trace_id=trace_id,
                trace_url=None,
                root_span=None,
                callback_handler=None,
                langfuse_metadata={},
                langfuse_active=False,
            )

        root_span: Any | None = None
        parent_span_id: str | None = None
        try:
            span_meta: dict[str, Any] = {
                "environment": self._environment,
                "tags": deduped_tags,
            }
            if session_id:
                span_meta["session_id"] = session_id
            if user_id:
                span_meta["user_id"] = user_id
            if preset_name:
                span_meta["preset"] = preset_name
            if ablation:
                span_meta["ablation"] = ablation
            if metadata:
                span_meta.update(metadata)

            root_span = client.start_observation(
                trace_context={"trace_id": trace_id},
                name="adaptiverag.query",
                as_type="agent",
                input={"query": query},
                metadata=span_meta,
            )
            parent_span_id = getattr(root_span, "id", None)
        except Exception as exc:
            logger.warning("langfuse_root_span_start_failed", trace_id=trace_id, error=str(exc))
            root_span = None

        callback_handler = self._build_callback_handler(
            trace_id=trace_id,
            parent_span_id=parent_span_id,
        )
        trace_url = self.resolve_trace_url(trace_id)
        return TraceRunHandle(
            trace_id=trace_id,
            trace_url=trace_url,
            root_span=root_span,
            callback_handler=callback_handler,
            langfuse_metadata=langfuse_metadata,
            langfuse_active=True,
        )

    def _record_domain_observations(
        self,
        client: Any,
        handle: TraceRunHandle,
        answer: FinalAnswer,
    ) -> None:
        """Emit rich domain child spans that plain LangChain LLM callbacks miss."""
        trace_ctx: dict[str, Any] = {"trace_id": handle.trace_id}
        parent_id = getattr(handle.root_span, "id", None)
        if parent_id:
            trace_ctx["parent_span_id"] = parent_id

        def _emit_child(
            *,
            name: str,
            as_type: str,
            input_payload: Any,
            output_payload: Any,
            metadata: dict[str, Any] | None = None,
            level: Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"] = "DEFAULT",
        ) -> None:
            try:
                child = client.start_observation(
                    trace_context=trace_ctx,
                    name=name,
                    as_type=as_type,
                    input=input_payload,
                    output=output_payload,
                    metadata=metadata or {},
                    level=level,
                )
                end_fn = getattr(child, "end", None)
                if callable(end_fn):
                    end_fn()
            except Exception as exc:
                logger.debug("langfuse_child_span_failed", name=name, error=str(exc))

        # 1. Retrieval & 3-Stage Dynamic Top-K summary
        if answer.dynamic_k_decision is not None or answer.evidence_chunks:
            sub_queries = (
                answer.query_plan.sub_queries if answer.query_plan is not None else [answer.query]
            )
            chunk_summaries = [
                {
                    "citation_id": ch.citation_id,
                    "doc_id": ch.doc_id,
                    "title": ch.title,
                    "source_type": ch.source_type.value,
                    "rerank_score": ch.rerank_score,
                    "fusion_score": ch.fusion_score,
                    "token_count": ch.token_count,
                }
                for ch in answer.evidence_chunks
            ]
            _emit_child(
                name="retrieval.hybrid_dynamic_k",
                as_type="retriever",
                input_payload={"sub_queries": sub_queries},
                output_payload={"selected_chunks": chunk_summaries},
                metadata=(
                    answer.dynamic_k_decision.model_dump(mode="json")
                    if answer.dynamic_k_decision is not None
                    else {"selected_count": len(answer.evidence_chunks)}
                ),
            )

        # 2. SQLite LLM Cache Hit visibility (otherwise invisible to LangChain LLM callbacks)
        if answer.usage.cache_hits > 0:
            _emit_child(
                name="llm.sqlite_cache_hits",
                as_type="span",
                input_payload={"query": answer.query},
                output_payload={
                    "cache_hits": answer.usage.cache_hits,
                    "estimated_cost_usd": answer.usage.estimated_cost_usd,
                },
                metadata={"cache_backend": "sqlite", "cost_saved": True},
            )

        # 3. CRAG & Multi-Signal Confidence Evaluator
        if answer.confidence is not None:
            _emit_child(
                name="verification.crag_confidence",
                as_type="evaluator",
                input_payload={"selected_k": len(answer.evidence_chunks)},
                output_payload=answer.confidence.model_dump(mode="json"),
                level="WARNING" if answer.confidence.level.value == "low" else "DEFAULT",
            )

        # 4. Two-Tier Citation Audit Guardrail
        if answer.citation_report is not None:
            cite = answer.citation_report
            _emit_child(
                name="verification.citation_audit",
                as_type="guardrail",
                input_payload={
                    "total_citations_found": cite.total_citations_found,
                    "regeneration_attempts": answer.regeneration_attempts,
                },
                output_payload={
                    "passed": cite.passed,
                    "citation_precision": cite.citation_precision,
                    "claim_support_rate": cite.claim_support_rate,
                    "valid_citation_ids": cite.valid_citation_ids,
                    "invalid_citation_ids": cite.invalid_citation_ids,
                    "regeneration_triggered": cite.regeneration_triggered,
                },
                level="DEFAULT" if cite.passed else "WARNING",
            )

    def finish_trace(
        self,
        handle: TraceRunHandle,
        *,
        answer: FinalAnswer | None = None,
        error: Exception | None = None,
    ) -> None:
        """Finalize root span, emit domain spans + online scores, and flush fail-open."""
        if not handle.langfuse_active:
            return
        client = self.get_client()
        if client is None:
            return

        try:
            if answer is not None:
                self._record_domain_observations(client, handle, answer)

                if handle.root_span is not None:
                    update_fn = getattr(handle.root_span, "update", None)
                    if callable(update_fn):
                        prompts_used: dict[str, dict[str, Any]] = {}
                        for event in answer.trace:
                            prompt_meta = event.metadata.get("prompt")
                            if isinstance(prompt_meta, dict) and prompt_meta.get("name"):
                                prompts_used[str(prompt_meta["name"])] = dict(prompt_meta)

                        root_meta: dict[str, Any] = {
                            "query_type": answer.query_type.value,
                            "language": answer.language,
                            "selected_k": len(answer.evidence_chunks),
                            "rewrite_attempts": answer.rewrite_attempts,
                            "regeneration_attempts": answer.regeneration_attempts,
                            "total_latency_ms": answer.total_latency_ms,
                            "usage": answer.usage.model_dump(mode="json"),
                            "node_count": len(answer.trace),
                        }
                        if prompts_used:
                            root_meta["prompts_used"] = prompts_used

                        update_fn(
                            output={
                                "answer": answer.answer,
                                "query_type": answer.query_type.value,
                                "language": answer.language,
                                "abstained": answer.abstained,
                                "fallback_used": answer.fallback_used,
                            },
                            metadata=root_meta,
                            level="WARNING" if answer.abstained else "DEFAULT",
                        )

                for score_spec in build_online_quality_scores(answer):
                    try:
                        client.create_score(
                            trace_id=handle.trace_id,
                            name=score_spec.name,
                            value=score_spec.value,
                            data_type=score_spec.data_type,
                            comment=score_spec.comment,
                            metadata=score_spec.metadata or None,
                        )
                    except Exception as exc:
                        logger.warning(
                            "langfuse_online_score_failed",
                            trace_id=handle.trace_id,
                            score_name=score_spec.name,
                            error=str(exc),
                        )
            elif error is not None and handle.root_span is not None:
                update_fn = getattr(handle.root_span, "update", None)
                if callable(update_fn):
                    update_fn(
                        level="ERROR",
                        status_message=str(error)[:500],
                        metadata={"error_type": type(error).__name__},
                    )
        except Exception as exc:
            logger.warning("langfuse_finish_trace_failed", trace_id=handle.trace_id, error=str(exc))
        finally:
            if handle.root_span is not None:
                try:
                    end_fn = getattr(handle.root_span, "end", None)
                    if callable(end_fn):
                        end_fn()
                except Exception as exc:
                    logger.debug("langfuse_root_span_end_failed", error=str(exc))

    def record_user_feedback(
        self,
        *,
        trace_id: str,
        score: float,
        comment: str | None = None,
        session_id: str | None = None,
        query: str | None = None,
    ) -> FeedbackRecordResult:
        """Record human-in-the-loop thumbs up/down feedback to structlog and Langfuse."""
        cleaned_comment = comment.strip() if comment and comment.strip() else None
        logger.info(
            "user_feedback_recorded",
            trace_id=trace_id,
            score=score,
            comment=cleaned_comment,
            session_id=session_id,
            langfuse_configured=self.configured,
        )

        client = self.get_client()
        if client is None:
            return FeedbackRecordResult(
                trace_id=trace_id,
                score=score,
                comment=cleaned_comment,
                persisted_to_langfuse=False,
                langfuse_error=None,
            )

        meta: dict[str, Any] = {"source": "react_glass_box_studio"}
        if session_id:
            meta["session_id"] = session_id
        if query:
            meta["query_preview"] = query[:240]

        try:
            client.create_score(
                trace_id=trace_id,
                name="user_feedback",
                value=float(score),
                data_type="NUMERIC",
                comment=cleaned_comment,
                metadata=meta,
            )
            flush_fn = getattr(client, "flush", None)
            if callable(flush_fn):
                flush_fn()
            return FeedbackRecordResult(
                trace_id=trace_id,
                score=score,
                comment=cleaned_comment,
                persisted_to_langfuse=True,
                langfuse_error=None,
            )
        except Exception as exc:
            logger.warning(
                "langfuse_feedback_score_failed",
                trace_id=trace_id,
                error=str(exc),
            )
            return FeedbackRecordResult(
                trace_id=trace_id,
                score=score,
                comment=cleaned_comment,
                persisted_to_langfuse=False,
                langfuse_error=str(exc),
            )

    def flush(self) -> None:
        """Flush buffered Langfuse telemetry safely (no-op when disabled)."""
        if not self._client_initialized or self._client is None:
            return
        try:
            flush_fn = getattr(self._client, "flush", None)
            if callable(flush_fn):
                flush_fn()
        except Exception as exc:
            logger.debug("langfuse_flush_failed", error=str(exc))


def get_langfuse_callback(
    settings: Settings,
    *,
    trace_context: dict[str, Any] | None = None,
) -> Any | None:
    """Return a Langfuse v5 LangChain CallbackHandler when enabled, else None."""
    obs = LangfuseObservability.from_settings(settings)
    if obs.get_client() is None:
        return None
    if not settings.langfuse_public_key:
        return None
    try:
        from langfuse.langchain import CallbackHandler

        if trace_context is not None:
            return CallbackHandler(
                public_key=settings.langfuse_public_key,
                trace_context=trace_context,  # type: ignore[arg-type]
            )
        return CallbackHandler(public_key=settings.langfuse_public_key)
    except Exception:
        return None
