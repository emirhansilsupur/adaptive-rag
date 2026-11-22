"""Application settings and external YAML strategy configuration loader."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, model_validator
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

    # --- Web Fallback ---
    tavily_api_key: SecretStr | None = None
    web_fallback_enabled: bool = True
    web_fallback_max_results: int = Field(default=4, ge=1, le=10)

    # --- Observability (Langfuse) ---
    langfuse_enabled: bool = False
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: SecretStr | None = None

    # --- Paths & Logging ---
    strategies_config_path: Path = Path("configs/strategies.yaml")
    log_level: str = "INFO"


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
