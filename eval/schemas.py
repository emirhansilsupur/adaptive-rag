"""Pydantic v2 schemas for golden evaluation datasets and 4-way ablation benchmark reports."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from adaptiverag.schemas.query import QueryType


class EvalStratum(StrEnum):
    """Evaluation stratum isolating a specific AdaptiveRAG capability."""

    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    CONVERSATIONAL = "conversational"
    WEB_FALLBACK = "web_fallback"
    UNANSWERABLE = "unanswerable"


class GoldenQuery(BaseModel):
    """Single labeled evaluation item in a JSONL or adapted external benchmark dataset."""

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    language: str = Field(default="en", min_length=2, max_length=8)
    stratum: EvalStratum = EvalStratum.FACTUAL
    expected_query_type: QueryType = QueryType.FACTUAL
    needs_retrieval: bool = True
    expected_doc_ids: list[str] = Field(default_factory=list)
    reference_answer: str = Field(default="")
    key_facts: list[str] = Field(default_factory=list)
    expect_web_fallback: bool = False
    expect_abstention: bool = False

    @model_validator(mode="after")
    def _validate_invariants(self) -> GoldenQuery:
        self.id = self.id.strip()
        self.query = self.query.strip()
        self.language = self.language.strip().lower()
        if self.stratum == EvalStratum.CONVERSATIONAL and self.needs_retrieval:
            raise ValueError(
                f"Conversational golden query '{self.id}' must have needs_retrieval=False."
            )
        if self.expect_web_fallback and self.expect_abstention:
            raise ValueError(
                f"Golden query '{self.id}' cannot expect both web_fallback and abstention."
            )
        return self


class IRScores(BaseModel):
    """Document-level Information Retrieval ranking metrics for a single query."""

    k_eval: int = Field(default=0, ge=0)
    recall_at_k: float = Field(default=0.0, ge=0.0, le=1.0)
    precision_at_k: float = Field(default=0.0, ge=0.0, le=1.0)
    mrr_at_k: float = Field(default=0.0, ge=0.0, le=1.0)
    ndcg_at_k: float = Field(default=0.0, ge=0.0, le=1.0)
    hit_at_k: bool = False


class RagasScores(BaseModel):
    """Optional RAGAS LLM-as-a-Judge quality metrics for a single query."""

    faithfulness: float | None = Field(default=None, ge=0.0, le=1.0)
    answer_relevancy: float | None = Field(default=None, ge=0.0, le=1.0)
    context_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    context_recall: float | None = Field(default=None, ge=0.0, le=1.0)


class QueryEvalRecord(BaseModel):
    """Per-query evaluation trace and metrics emitted by the benchmark runner."""

    query_id: str
    stratum: EvalStratum
    language: str
    config_name: str
    predicted_query_type: QueryType | None = None
    routing_correct: bool | None = None
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    selected_k: int = Field(default=0, ge=0)
    context_tokens: int = Field(default=0, ge=0)
    ir_scores: IRScores | None = None
    fact_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_level: str | None = None
    fallback_used: bool = False
    abstained: bool = False
    rewrite_attempts: int = Field(default=0, ge=0)
    regeneration_attempts: int = Field(default=0, ge=0)
    citation_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    claim_support_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    citation_passed: bool | None = None
    # "completed" | "skipped" | "failed"; "failed" means the claim judge errored.
    citation_semantic_audit: str | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    cache_hits: int = Field(default=0, ge=0)
    ragas: RagasScores | None = None
    answer_preview: str = ""
    error: str | None = None


class StratumSummary(BaseModel):
    """Aggregated performance metrics for a single stratum or language slice."""

    name: str
    count: int = Field(default=0, ge=0)
    retrieval_count: int = Field(default=0, ge=0)
    routing_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_mrr: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_ndcg: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_fact_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    avg_selected_k: float = Field(default=0.0, ge=0.0)
    avg_context_tokens: float = Field(default=0.0, ge=0.0)
    mean_latency_ms: float = Field(default=0.0, ge=0.0)
    fallback_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    abstention_rate: float = Field(default=0.0, ge=0.0, le=1.0)


class ConfigBenchmarkSummary(BaseModel):
    """Aggregated benchmark scorecard for one ablation configuration."""

    config_name: str
    mode: Literal["offline-ir", "live"]
    query_count: int = Field(default=0, ge=0)
    error_count: int = Field(default=0, ge=0)
    retrieval_query_count: int = Field(default=0, ge=0)

    # IR metrics (averaged over queries with non-empty expected_doc_ids)
    mean_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_mrr: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_ndcg: float | None = Field(default=None, ge=0.0, le=1.0)
    hit_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_fact_coverage: float | None = Field(default=None, ge=0.0, le=1.0)

    # Agentic control & grounding metrics
    routing_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    fallback_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    fallback_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    abstention_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    abstention_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_citation_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_claim_support_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    citation_pass_rate: float | None = Field(default=None, ge=0.0, le=1.0)

    # Context efficiency & latency / FinOps
    avg_selected_k: float = Field(default=0.0, ge=0.0)
    avg_context_tokens: float = Field(default=0.0, ge=0.0)
    mean_latency_ms: float = Field(default=0.0, ge=0.0)
    latency_p50_ms: float = Field(default=0.0, ge=0.0)
    latency_p95_ms: float = Field(default=0.0, ge=0.0)
    total_tokens: int = Field(default=0, ge=0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    cost_per_query_usd: float = Field(default=0.0, ge=0.0)
    total_cache_hits: int = Field(default=0, ge=0)
    rewrite_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    regeneration_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    # Optional RAGAS judge means
    ragas_faithfulness: float | None = Field(default=None, ge=0.0, le=1.0)
    ragas_answer_relevancy: float | None = Field(default=None, ge=0.0, le=1.0)
    ragas_context_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    ragas_context_recall: float | None = Field(default=None, ge=0.0, le=1.0)

    by_stratum: dict[str, StratumSummary] = Field(default_factory=dict)
    by_language: dict[str, StratumSummary] = Field(default_factory=dict)


class BenchmarkReport(BaseModel):
    """Complete multi-configuration ablation report persisted as JSON & Markdown."""

    run_id: str
    timestamp_utc: str
    mode: Literal["offline-ir", "live"]
    dataset_source: str
    total_queries: int = Field(ge=0)
    configs: list[str]
    budget_exceeded: bool = False
    summaries: list[ConfigBenchmarkSummary] = Field(default_factory=list)
    records: list[QueryEvalRecord] = Field(default_factory=list)
