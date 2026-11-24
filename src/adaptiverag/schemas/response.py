"""Final response, token/cost accounting, and trace event schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from adaptiverag.schemas.citation import CitationReport
from adaptiverag.schemas.query import QueryPlan, QueryType
from adaptiverag.schemas.retrieval import ConfidenceAssessment, DynamicKDecision, EvidenceChunk


class TokenUsage(BaseModel):
    """Aggregated token and USD cost accounting across a pipeline execution."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    cache_hits: int = Field(default=0, ge=0)

    def add(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated_cost_usd=round(self.estimated_cost_usd + other.estimated_cost_usd, 8),
            cache_hits=self.cache_hits + other.cache_hits,
        )


class NodeTraceEvent(BaseModel):
    """Structured telemetry event emitted by each LangGraph node."""

    node: str
    latency_ms: float = Field(ge=0.0)
    summary: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class FinalAnswer(BaseModel):
    """Complete glass-box output returned by AdaptiveRAG."""

    query: str
    answer: str
    query_type: QueryType
    language: str = "en"
    query_plan: QueryPlan | None = None
    evidence_chunks: list[EvidenceChunk] = Field(default_factory=list)
    dynamic_k_decision: DynamicKDecision | None = None
    confidence: ConfidenceAssessment | None = None
    citation_report: CitationReport | None = None
    fallback_used: bool = False
    abstained: bool = False
    rewrite_attempts: int = Field(default=0, ge=0)
    regeneration_attempts: int = Field(default=0, ge=0)
    total_latency_ms: float = Field(default=0.0, ge=0.0)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    trace: list[NodeTraceEvent] = Field(default_factory=list)
    trace_id: str | None = None
    trace_url: str | None = None
