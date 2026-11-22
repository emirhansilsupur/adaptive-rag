"""HTTP request and response schemas for the AdaptiveRAG FastAPI service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from adaptiverag.graph.state import AblationConfig
from adaptiverag.schemas.query import ChatMessage

AblationPreset = Literal["full", "adaptive_routing", "hybrid_rerank", "naive_dense"]


class QueryRequest(BaseModel):
    """Payload for synchronous (/v1/query) and streaming (/v1/query/stream) RAG queries."""

    query: str = Field(min_length=1, max_length=4000, description="User question in TR or EN.")
    chat_history: list[ChatMessage] = Field(
        default_factory=list,
        description="Optional prior conversation turns for coreference resolution.",
    )
    preset: AblationPreset | None = Field(
        default=None,
        description="Optional named ablation preset (ignored when explicit ablation is supplied).",
    )
    ablation: AblationConfig | None = Field(
        default=None,
        description="Optional fine-grained feature switches overriding preset defaults.",
    )
    max_cost_usd: float | None = Field(
        default=None,
        gt=0.0,
        le=50.0,
        description="Optional per-request hard USD cost cap (FinOps guardrail).",
    )

    @field_validator("query")
    @classmethod
    def _strip_and_validate_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be empty or whitespace-only.")
        return cleaned

    def resolve_ablation(self) -> AblationConfig:
        """Resolve the effective AblationConfig from explicit overrides or preset name."""
        if self.ablation is not None:
            return self.ablation
        if self.preset == "naive_dense":
            return AblationConfig.naive_dense()
        if self.preset == "hybrid_rerank":
            return AblationConfig.hybrid_rerank()
        if self.preset == "adaptive_routing":
            return AblationConfig.adaptive_routing()
        return AblationConfig.full()


class InlineDocument(BaseModel):
    """Inline document payload for direct API ingestion without local disk files."""

    doc_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=400)
    content: str = Field(min_length=1)
    doc_type: Literal["markdown", "text"] = "markdown"
    language: Literal["en", "tr"] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("doc_id", "title", "content")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field must not be empty or whitespace-only.")
        return cleaned


class IngestRequest(BaseModel):
    """Payload for POST /v1/ingest supporting either corpus directory or inline documents."""

    corpus_dir: str | None = Field(
        default=None,
        description="Directory containing .md/.txt/.pdf files (defaults to data/raw when documents is omitted).",
    )
    documents: list[InlineDocument] | None = Field(
        default=None,
        description="Optional list of inline documents to chunk and upsert directly.",
    )
    recreate_collection: bool = Field(
        default=False,
        description="If true, drop and recreate the Qdrant collection before upserting.",
    )
    max_tokens: int = Field(default=512, ge=64, le=4096)
    overlap_tokens: int = Field(default=64, ge=0, le=512)
    batch_size: int = Field(default=64, ge=1, le=512)

    @model_validator(mode="after")
    def _validate_ingest_options(self) -> IngestRequest:
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError(
                f"overlap_tokens ({self.overlap_tokens}) must be < max_tokens ({self.max_tokens})"
            )
        if self.documents is not None and self.corpus_dir is not None:
            raise ValueError("Provide either 'documents' or 'corpus_dir', not both.")
        if self.documents is not None and len(self.documents) == 0:
            raise ValueError("'documents' must contain at least one document when provided.")
        return self


class IngestResponse(BaseModel):
    """Summary returned after corpus or inline document ingestion."""

    collection: str
    source: str
    documents_processed: int = Field(ge=0)
    chunks_created: int = Field(ge=0)
    points_upserted: int = Field(ge=0)
    total_collection_points: int = Field(ge=0)
    collection_created: bool = False
    elapsed_ms: float = Field(ge=0.0)


class HealthResponse(BaseModel):
    """Operational readiness & configuration summary for GET /v1/health."""

    status: Literal["ok", "degraded"]
    version: str = "0.1.0"
    qdrant: dict[str, Any]
    llm_configured: bool
    llm_error: str | None = None
    llm_provider: str
    fast_model: str
    generation_model: str
    dense_embedding_model: str
    sparse_embedding_model: str
    reranker_model: str
    web_fallback_enabled: bool
    web_fallback_providers: list[str] = Field(default_factory=list)
    langfuse_enabled: bool
    llm_cache_enabled: bool
    default_max_run_cost_usd: float
