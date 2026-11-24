"""LangGraph state definitions and ablation configuration for AdaptiveRAG."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field

from adaptiverag.retrieval.qdrant_store import RetrievalMode
from adaptiverag.schemas.citation import CitationReport
from adaptiverag.schemas.query import ChatMessage, QueryPlan
from adaptiverag.schemas.response import NodeTraceEvent, TokenUsage
from adaptiverag.schemas.retrieval import (
    ConfidenceAssessment,
    DynamicKDecision,
    EvidenceChunk,
)


def merge_token_usage(left: TokenUsage | None, right: TokenUsage | None) -> TokenUsage:
    """LangGraph reducer accumulating TokenUsage deltas across nodes."""
    if left is None:
        return right if right is not None else TokenUsage()
    if right is None:
        return left
    return left.add(right)


class AblationConfig(BaseModel):
    """Runtime feature switches supporting 4-way ablation benchmarking & API overrides."""

    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    enable_query_routing: bool = True
    enable_multi_query: bool = True
    enable_rerank: bool = True
    enable_dynamic_k: bool = True
    fixed_top_k: int = Field(default=5, ge=1, le=30)
    enable_confidence_gating: bool = True
    enable_web_fallback: bool = True
    enable_citation_validation: bool = True
    enable_parent_child: bool | None = Field(
        default=None,
        description=(
            "Override runtime small-to-big parent expansion (None defers to "
            "Settings.enable_parent_child_retrieval)."
        ),
    )
    doc_ids: list[str] | None = None
    language_filter: str | None = None

    @classmethod
    def naive_dense(cls, *, fixed_top_k: int = 5) -> AblationConfig:
        """Baseline 1: Dense-only static top-k RAG with no routing, reranking, or verification."""
        return cls(
            retrieval_mode=RetrievalMode.DENSE_ONLY,
            enable_query_routing=False,
            enable_multi_query=False,
            enable_rerank=False,
            enable_dynamic_k=False,
            fixed_top_k=fixed_top_k,
            enable_confidence_gating=False,
            enable_web_fallback=False,
            enable_citation_validation=False,
        )

    @classmethod
    def hybrid_rerank(cls, *, fixed_top_k: int = 5) -> AblationConfig:
        """Baseline 2: Hybrid BM25+Dense + Cross-Encoder Rerank with static top-k."""
        return cls(
            retrieval_mode=RetrievalMode.HYBRID,
            enable_query_routing=False,
            enable_multi_query=False,
            enable_rerank=True,
            enable_dynamic_k=False,
            fixed_top_k=fixed_top_k,
            enable_confidence_gating=False,
            enable_web_fallback=False,
            enable_citation_validation=False,
        )

    @classmethod
    def adaptive_routing(cls) -> AblationConfig:
        """Baseline 3: Query routing + multi-query + hybrid + rerank + dynamic top-k."""
        return cls(
            retrieval_mode=RetrievalMode.HYBRID,
            enable_query_routing=True,
            enable_multi_query=True,
            enable_rerank=True,
            enable_dynamic_k=True,
            enable_confidence_gating=False,
            enable_web_fallback=False,
            enable_citation_validation=False,
        )

    @classmethod
    def full(cls) -> AblationConfig:
        """Full AdaptiveRAG with all agentic self-correction & verification loops enabled."""
        return cls()


class AgentState(TypedDict, total=False):
    """Shared state flowing through the AdaptiveRAG LangGraph cyclic workflow."""

    # Inputs
    query: str
    chat_history: list[ChatMessage]
    ablation: AblationConfig

    # Planning & Query Transformation
    query_plan: QueryPlan
    active_queries: list[str]

    # Retrieval, Reranking & Dynamic Top-K
    retrieved_chunks: list[EvidenceChunk]
    selected_chunks: list[EvidenceChunk]
    dynamic_k_decision: DynamicKDecision

    # Confidence Gating & Recovery
    confidence: ConfidenceAssessment
    rewrite_attempts: int
    fallback_used: bool

    # Generation & Citation Verification
    draft_answer: str
    citation_report: CitationReport
    regeneration_attempts: int
    abstained: bool

    # Reducers for per-node telemetry
    usage: Annotated[TokenUsage, merge_token_usage]
    trace: Annotated[list[NodeTraceEvent], operator.add]
