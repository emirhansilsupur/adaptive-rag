"""Application settings and external YAML strategy configuration loader."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from adaptiverag.schemas.query import QueryType, StrategyProfile


class ConfidenceConfig(BaseModel):
    """Weights and decision thresholds for multi-signal retrieval confidence."""

    weight_top1: float = Field(default=0.50, ge=0.0, le=1.0)
    weight_mean_topk: float = Field(default=0.30, ge=0.0, le=1.0)
    weight_subquery_coverage: float = Field(default=0.20, ge=0.0, le=1.0)
    high_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    low_threshold: float = Field(default=0.38, ge=0.0, le=1.0)
    enable_crag_grader_in_ambiguous_band: bool = True
    max_corrective_rewrites: int = Field(default=1, ge=0, le=3)

    @model_validator(mode="after")
    def _validate_weights_and_thresholds(self) -> ConfidenceConfig:
        total = self.weight_top1 + self.weight_mean_topk + self.weight_subquery_coverage
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Confidence weights must sum to 1.0, got {total:.4f}")
        if self.low_threshold >= self.high_threshold:
            raise ValueError(
                f"low_threshold ({self.low_threshold}) must be < high_threshold ({self.high_threshold})"
            )
        return self


class CitationValidationConfig(BaseModel):
    """Policy configuration for two-tier citation verification."""

    enable_semantic_entailment: bool = True
    min_claim_support_rate: float = Field(default=0.85, ge=0.0, le=1.0)
    max_regenerations: int = Field(default=1, ge=0, le=3)
    allow_partial_as_supported: bool = False


class StrategiesConfig(BaseModel):
    """Root schema matching configs/strategies.yaml."""

    strategies: dict[QueryType, StrategyProfile]
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    citation_validation: CitationValidationConfig = Field(default_factory=CitationValidationConfig)

    @model_validator(mode="after")
    def _ensure_all_query_types_present(self) -> StrategiesConfig:
        missing = [qt.value for qt in QueryType if qt not in self.strategies]
        if missing:
            raise ValueError(f"Missing strategy definitions for query types: {missing}")
        return self

    def for_query_type(self, query_type: QueryType) -> StrategyProfile:
        return self.strategies[query_type]


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM Provider ---
    llm_provider: Literal["openrouter", "openai", "ollama", "gemini", "anthropic"] = "openrouter"
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openai_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434/v1"

    llm_fast_model: str = "qwen/qwen3-30b-a3b-instruct-2507"
    llm_generation_model: str = "deepseek/deepseek-v3.2"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_request_timeout: float = Field(default=45.0, gt=1.0)
    allow_client_provider_keys: bool = True

    # --- FinOps Budget & Cache ---
    llm_cache_enabled: bool = True
    llm_cache_path: Path = Path(".adaptiverag_cache/llm_cache.sqlite")
    max_run_cost_usd: float = Field(default=0.50, gt=0.0)

    # --- Multilingual Embeddings & Cross-Encoder (FastEmbed ONNX, $0 API cost) ---
    dense_embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    sparse_embedding_model: str = "Qdrant/bm25"
    reranker_model: str = "BAAI/bge-reranker-v2-m3-int8"

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "adaptiverag_docs"

    # --- Hierarchical Parent-Child (Small-to-Big) Chunking & Retrieval ---
    enable_parent_child_retrieval: bool = False
    parent_max_tokens: int = Field(default=1000, ge=128, le=8192)
    child_max_tokens: int = Field(default=256, ge=64, le=2048)
    child_overlap_tokens: int = Field(default=40, ge=0, le=512)

    # --- Adaptive PDF Parsing & 2-Tier OCR (LiteParse + OCRia-nano-1 + Vision-LLM) ---
    pdf_ocr_mode: Literal["auto", "ocria_nano", "vision_llm", "off"] = "auto"
    pdf_ocr_auto_pull: bool = True
    pdf_ocr_model_dir: Path = Path(".adaptiverag_cache/liteparse")
    pdf_ocr_vision_fallback_enabled: bool = True
    pdf_ocr_vision_model: str = "google/gemini-2.5-flash"
    pdf_ocr_dpi: int = Field(default=200, ge=72, le=400)
    pdf_ocr_max_pages: int = Field(default=25, ge=1, le=500)
    pdf_ocr_min_chars: int = Field(default=32, ge=1, le=500)

    # --- Web Fallback ---
    tavily_api_key: SecretStr | None = None
    web_fallback_enabled: bool = True
    web_fallback_max_results: int = Field(default=4, ge=1, le=10)

    # --- Live URL & Docs Site Ingestion (Firecrawl Local v2 API) ---
    firecrawl_local_url: str | None = Field(
        default="http://localhost:3002",
        description="Base URL for a local firecrawl-local serve instance (empty to disable).",
    )
    firecrawl_api_key: SecretStr | None = None
    firecrawl_timeout_seconds: float = Field(default=60.0, ge=5.0, le=300.0)
    firecrawl_default_crawl_limit: int = Field(default=10, ge=1, le=100)
    firecrawl_allow_private_urls: bool = False

    # --- Observability & Prompt Management (Langfuse v5 OTEL SDK: Cloud or Self-Hosted v4) ---
    langfuse_enabled: bool = False
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_public_key: str = ""
    langfuse_secret_key: SecretStr | None = None
    langfuse_environment: str = "development"
    langfuse_release: str | None = "0.1.0"
    langfuse_prompt_management_enabled: bool = False
    langfuse_prompt_label: str = "production"
    langfuse_prompt_cache_ttl_seconds: int = Field(default=300, ge=0, le=86400)

    # --- Production API Security, Rate Limiting & Static SPA Serving ---
    # Without a key the API runs local-only (see api.security.LocalOnlyGuardMiddleware).
    api_auth_key: SecretStr | None = None
    # Browser origins allowed to call the API. "*" lets any website read answers from a
    # visitor's browser, so it must be set explicitly; empty means the local Studio dev origins.
    cors_allow_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    # Hosts answered without API_AUTH_KEY besides localhost/127.0.0.1/[::1]; "*" answers any host.
    allowed_hosts_without_auth: str = ""
    rate_limit_enabled: bool = False
    rate_limit_requests_per_minute: int = Field(default=60, ge=1, le=10000)
    rate_limit_trust_proxy_headers: bool = False
    serve_static_ui: bool = True
    static_ui_dir: Path | None = None

    # --- Input Safety Guardrails ($0 deterministic screening before LLM calls) ---
    guardrails_enabled: bool = True
    guardrails_block_prompt_injection: bool = True
    guardrails_pii_mode: Literal["off", "redact", "block"] = "redact"

    # --- Paths & Logging ---
    allowed_corpus_root: Path = Path("data")
    strategies_config_path: Path = Path("configs/strategies.yaml")
    log_level: str = "INFO"

    @field_validator("allowed_corpus_root", mode="before")
    @classmethod
    def _normalize_allowed_corpus_root(cls, value: str | Path | None) -> Path:
        if value is None:
            return Path("data")
        cleaned = str(value).strip()
        return Path(cleaned) if cleaned else Path("data")

    @field_validator("api_auth_key", mode="before")
    @classmethod
    def _normalize_api_auth_key(cls, value: str | SecretStr | None) -> str | SecretStr | None:
        if value is None:
            return None
        raw = value.get_secret_value() if isinstance(value, SecretStr) else str(value)
        cleaned = raw.strip()
        return value if (cleaned and isinstance(value, SecretStr)) else (cleaned or None)

    @field_validator("static_ui_dir", mode="before")
    @classmethod
    def _normalize_static_ui_dir(cls, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return Path(cleaned) if cleaned else None

    @field_validator("firecrawl_local_url", mode="before")
    @classmethod
    def _normalize_firecrawl_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip().rstrip("/")
        return cleaned or None

    @field_validator("langfuse_host", mode="before")
    @classmethod
    def _normalize_langfuse_host(cls, value: str | None) -> str:
        if value is None:
            return "https://cloud.langfuse.com"
        cleaned = str(value).strip().rstrip("/")
        return cleaned or "https://cloud.langfuse.com"

    @field_validator("langfuse_prompt_label", mode="before")
    @classmethod
    def _normalize_langfuse_prompt_label(cls, value: str | None) -> str:
        if value is None:
            return "production"
        cleaned = str(value).strip()
        return cleaned or "production"

    @model_validator(mode="after")
    def _validate_parent_child_bounds(self) -> Settings:
        if self.child_overlap_tokens >= self.child_max_tokens:
            raise ValueError(
                f"child_overlap_tokens ({self.child_overlap_tokens}) must be < "
                f"child_max_tokens ({self.child_max_tokens})"
            )
        if self.child_max_tokens >= self.parent_max_tokens:
            raise ValueError(
                f"child_max_tokens ({self.child_max_tokens}) must be < "
                f"parent_max_tokens ({self.parent_max_tokens})"
            )
        return self


def load_strategies(path: Path | str = Path("configs/strategies.yaml")) -> StrategiesConfig:
    """Load and validate strategy profiles from YAML."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Strategies configuration not found at: {config_path.resolve()}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return StrategiesConfig.model_validate(raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached runtime settings."""
    return Settings()
