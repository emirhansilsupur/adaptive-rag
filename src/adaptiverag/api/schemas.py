"""HTTP request and response schemas for the AdaptiveRAG FastAPI service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from adaptiverag.graph.state import AblationConfig
from adaptiverag.ingestion.pdf_parser import PdfOcrMode
from adaptiverag.llm.factory import LLMProvider
from adaptiverag.schemas.query import ChatMessage

AblationPreset = Literal["full", "adaptive_routing", "hybrid_rerank", "naive_dense"]


class LLMOverride(BaseModel):
    """Optional concurrency-safe per-request LLM provider, tier models & temperature override."""

    provider: LLMProvider | None = Field(
        default=None,
        description="Override server default LLM provider (must be configured in server .env).",
    )
    fast_model: str | None = Field(
        default=None,
        max_length=200,
        description="Override fast-tier model ID (Router, Decompose, CRAG, Citation Audit).",
    )
    generation_model: str | None = Field(
        default=None,
        max_length=200,
        description="Override generation-tier model ID (Grounded Synthesis & Strict Regen).",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Override LLM sampling temperature (0.0 for grounded RAG, 1.0 for reasoning models).",
    )

    @field_validator("fast_model", "generation_model")
    @classmethod
    def _strip_optional_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @property
    def is_empty(self) -> bool:
        return (
            self.provider is None
            and self.fast_model is None
            and self.generation_model is None
            and self.temperature is None
        )


class QueryRequest(BaseModel):
    """Payload for synchronous (/v1/query) and streaming (/v1/query/stream) RAG queries."""

    query: str = Field(min_length=1, max_length=4000, description="User question in TR or EN.")
    chat_history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=50,
        description="Optional prior conversation turns (max 50) for coreference resolution.",
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
    custom_instructions: str | None = Field(
        default=None,
        max_length=1500,
        description=(
            "Optional Layer-2 thread style/format instructions applied only to synthesis "
            "and conversational turns (subordinate to locked grounding & citation rules)."
        ),
    )
    doc_ids: list[str] | None = Field(
        default=None,
        max_length=100,
        description=(
            "Optional list of indexed doc_ids scoping retrieval for this request/thread "
            "(works with both named presets and custom ablation)."
        ),
    )
    llm_override: LLMOverride | None = Field(
        default=None,
        description=(
            "Optional per-request LLM provider/model/temperature override "
            "(uses server-managed API keys from .env; never accepts raw secrets or custom URLs)."
        ),
    )
    session_id: str | None = Field(
        default=None,
        max_length=128,
        description="Optional UI/client session identifier grouped in Langfuse Sessions.",
    )
    user_id: str | None = Field(
        default=None,
        max_length=128,
        description="Optional end-user identifier attached to observability traces.",
    )

    @field_validator("query")
    @classmethod
    def _strip_and_validate_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be empty or whitespace-only.")
        return cleaned

    @field_validator("session_id", "user_id", "custom_instructions")
    @classmethod
    def _strip_optional_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("doc_ids")
    @classmethod
    def _normalize_doc_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = list(dict.fromkeys(item.strip() for item in value if item and item.strip()))
        return cleaned or None

    def resolve_preset_label(self) -> str:
        """Return the effective preset label for trace tagging and analytics."""
        if self.ablation is not None:
            return "custom"
        return self.preset or "full"

    def resolve_ablation(self) -> AblationConfig:
        """Resolve the effective AblationConfig from explicit overrides or preset name."""
        if self.ablation is not None:
            base = self.ablation
        elif self.preset == "naive_dense":
            base = AblationConfig.naive_dense()
        elif self.preset == "hybrid_rerank":
            base = AblationConfig.hybrid_rerank()
        elif self.preset == "adaptive_routing":
            base = AblationConfig.adaptive_routing()
        else:
            base = AblationConfig.full()

        if self.doc_ids:
            return base.model_copy(update={"doc_ids": list(self.doc_ids)})
        return base


class InlineDocument(BaseModel):
    """Inline document payload for direct API ingestion without local disk files."""

    doc_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=400)
    content: str = Field(min_length=1, max_length=1_000_000)
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
    """Payload for POST /v1/ingest supporting local files, inline docs, or Firecrawl URLs."""

    corpus_dir: str | None = Field(
        default=None,
        max_length=512,
        description=(
            "Directory containing .md/.txt/.pdf files "
            "(defaults to data/raw when no other source is provided)."
        ),
    )
    documents: list[InlineDocument] | None = Field(
        default=None,
        max_length=50,
        description="Optional list of inline documents (max 50) to chunk and upsert directly.",
    )
    urls: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of public HTTP/HTTPS URLs (max 25) to scrape via firecrawl-local "
            "and index into Qdrant."
        ),
    )
    crawl_url: str | None = Field(
        default=None,
        max_length=2048,
        description=(
            "Optional public root documentation/website URL to crawl via firecrawl-local "
            "and index into Qdrant."
        ),
    )
    crawl_limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of pages to crawl when 'crawl_url' is provided.",
    )
    crawl_max_depth: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description="Optional maximum link discovery depth when 'crawl_url' is provided.",
    )
    only_main_content: bool = Field(
        default=True,
        description="Ask Firecrawl to strip navbars/sidebars/footers and keep primary content.",
    )
    recreate_collection: bool = Field(
        default=False,
        description="If true, drop and recreate the Qdrant collection before upserting.",
    )
    max_tokens: int = Field(default=512, ge=64, le=4096)
    overlap_tokens: int = Field(default=64, ge=0, le=512)
    parent_child: bool | None = Field(
        default=None,
        description="Enable hierarchical Parent-Child (Small-to-Big) chunking (defaults to server setting).",
    )
    parent_max_tokens: int | None = Field(default=None, ge=128, le=8192)
    child_max_tokens: int | None = Field(default=None, ge=64, le=2048)
    child_overlap_tokens: int | None = Field(default=None, ge=0, le=512)
    ocr_mode: PdfOcrMode | None = Field(
        default=None,
        description="Optional PDF OCR routing override ('auto', 'ocria_nano', 'vision_llm', 'off').",
    )
    batch_size: int = Field(default=64, ge=1, le=512)

    @model_validator(mode="after")
    def _validate_ingest_options(self) -> IngestRequest:
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError(
                f"overlap_tokens ({self.overlap_tokens}) must be < max_tokens ({self.max_tokens})"
            )
        if (
            self.child_max_tokens is not None
            and self.parent_max_tokens is not None
            and self.child_max_tokens >= self.parent_max_tokens
        ):
            raise ValueError(
                f"child_max_tokens ({self.child_max_tokens}) must be < "
                f"parent_max_tokens ({self.parent_max_tokens})"
            )
        if (
            self.child_overlap_tokens is not None
            and self.child_max_tokens is not None
            and self.child_overlap_tokens >= self.child_max_tokens
        ):
            raise ValueError(
                f"child_overlap_tokens ({self.child_overlap_tokens}) must be < "
                f"child_max_tokens ({self.child_max_tokens})"
            )

        if self.urls is not None:
            cleaned_urls = [u.strip() for u in self.urls]
            if not cleaned_urls:
                raise ValueError("'urls' must contain at least one URL when provided.")
            if any(not u for u in cleaned_urls):
                raise ValueError("'urls' must not contain empty or whitespace-only entries.")
            if len(cleaned_urls) > 25:
                raise ValueError(
                    "'urls' supports at most 25 URLs per request; use 'crawl_url' for larger sites."
                )
            self.urls = cleaned_urls

        if self.crawl_url is not None:
            cleaned_crawl = self.crawl_url.strip()
            if not cleaned_crawl:
                raise ValueError("'crawl_url' must not be empty or whitespace-only.")
            self.crawl_url = cleaned_crawl

        active_sources = [
            name
            for name, val in (
                ("corpus_dir", self.corpus_dir),
                ("documents", self.documents),
                ("urls", self.urls),
                ("crawl_url", self.crawl_url),
            )
            if val is not None
        ]
        if len(active_sources) > 1:
            joined = ", ".join(f"'{s}'" for s in active_sources)
            raise ValueError(f"Provide at most one ingestion source, got multiple: {joined}.")
        if self.documents is not None and len(self.documents) == 0:
            raise ValueError("'documents' must contain at least one document when provided.")
        return self


class IngestResponse(BaseModel):
    """Summary returned after corpus, inline, or live Firecrawl URL ingestion."""

    collection: str
    source: str
    documents_processed: int = Field(ge=0)
    chunks_created: int = Field(ge=0)
    points_upserted: int = Field(ge=0)
    total_collection_points: int = Field(ge=0)
    collection_created: bool = False
    elapsed_ms: float = Field(ge=0.0)
    source_urls: list[str] = Field(default_factory=list)
    parent_child_enabled: bool = False
    unique_parents_created: int = Field(default=0, ge=0)
    ocr_pages_processed: int = Field(default=0, ge=0)
    ocr_backends_used: list[str] = Field(default_factory=list)


class IndexedDocumentInfo(BaseModel):
    """Aggregated metadata for one indexed document in the Qdrant collection."""

    doc_id: str
    title: str
    source_origin: Literal["local", "web"] = "local"
    doc_type: str = "markdown"
    language: str = "en"
    chunk_strategy: Literal["flat", "parent_child"] = "flat"
    chunk_count: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    source_url: str | None = None
    source_path: str | None = None
    file_name: str | None = None


class DocumentListResponse(BaseModel):
    """Response payload for GET /v1/documents."""

    collection: str
    total_documents: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
    documents: list[IndexedDocumentInfo] = Field(default_factory=list)


class DeleteDocumentResponse(BaseModel):
    """Response payload for DELETE /v1/documents/{doc_id}."""

    collection: str
    doc_id: str
    deleted_chunks: int = Field(ge=0)
    remaining_collection_points: int = Field(ge=0)


class ProviderCatalogItem(BaseModel):
    """Non-secret provider readiness and curated/detected models for GET /v1/models."""

    id: LLMProvider
    label: str
    is_local: bool
    requires_api_key: bool = True
    sdk_installed: bool = True
    configured: bool
    reachable: bool | None = None
    status_detail: str | None = None
    default_fast_model: str
    default_generation_model: str
    recommended_fast_models: list[str] = Field(default_factory=list)
    recommended_generation_models: list[str] = Field(default_factory=list)
    detected_models: list[str] = Field(default_factory=list)


class ModelsCatalogResponse(BaseModel):
    """Response payload for GET /v1/models exposing safe runtime defaults & configured providers."""

    default_provider: LLMProvider
    default_fast_model: str
    default_generation_model: str
    default_temperature: float
    ollama_base_url: str
    allow_client_provider_keys: bool = True
    providers: list[ProviderCatalogItem] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    """Payload for POST /v1/feedback recording human-in-the-loop answer quality scores."""

    trace_id: str = Field(
        min_length=8,
        max_length=128,
        description="Trace ID returned on FinalAnswer.trace_id.",
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalized user rating (1.0 = thumbs up / helpful, 0.0 = thumbs down).",
    )
    comment: str | None = Field(
        default=None,
        max_length=1000,
        description="Optional qualitative explanation from the reviewer or user.",
    )
    session_id: str | None = Field(
        default=None,
        max_length=128,
        description="Optional session identifier for correlation.",
    )
    query: str | None = Field(
        default=None,
        max_length=500,
        description="Optional query preview attached to feedback metadata.",
    )

    @field_validator("trace_id")
    @classmethod
    def _strip_trace_id(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 8:
            raise ValueError("trace_id must contain at least 8 non-whitespace characters.")
        return cleaned


class FeedbackResponse(BaseModel):
    """Response payload for POST /v1/feedback."""

    status: Literal["recorded"] = "recorded"
    trace_id: str
    score: float
    comment: str | None = None
    persisted_to_langfuse: bool = False
    langfuse_error: str | None = None


class HealthResponse(BaseModel):
    """Operational readiness & configuration summary for GET /v1/health."""

    status: Literal["ok", "degraded"]
    version: str = "0.1.0"
    qdrant: dict[str, Any]
    firecrawl: dict[str, Any] = Field(default_factory=dict)
    langfuse: dict[str, Any] = Field(default_factory=dict)
    security: dict[str, Any] = Field(default_factory=dict)
    parent_child_retrieval: dict[str, Any] = Field(default_factory=dict)
    pdf_ocr: dict[str, Any] = Field(default_factory=dict)
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
