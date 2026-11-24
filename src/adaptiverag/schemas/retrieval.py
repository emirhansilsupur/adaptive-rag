"""Retrieval evidence, dynamic top-k, and confidence assessment schemas."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(StrEnum):
    """Origin of an evidence chunk."""

    LOCAL = "local"
    WEB = "web"


class EvidenceChunk(BaseModel):
    """Unified representation of a retrieved local or web evidence passage."""

    citation_id: str = Field(
        default="",
        description="Short prompt citation label assigned prior to generation, e.g., C1 or W1.",
    )
    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_type: SourceType = SourceType.LOCAL
    doc_id: str = ""
    title: str = ""
    section_path: str = ""
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Retrieval & reranking scores
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    rerank_raw_score: float | None = None
    rerank_score: float | None = Field(default=None, ge=0.0, le=1.0)

    matched_sub_queries: list[str] = Field(default_factory=list)
    token_count: int = Field(default=0, ge=0)

    # Hierarchical Parent-Child (Small-to-Big) lineage
    parent_id: str | None = None
    parent_text: str | None = Field(
        default=None,
        exclude=True,
        description="Full parent passage retained in memory for small-to-big expansion.",
    )
    parent_section_path: str | None = None
    parent_token_count: int | None = Field(default=None, ge=0)
    expanded_to_parent: bool = False
    matched_child_ids: list[str] = Field(default_factory=list)


class DynamicKDecision(BaseModel):
    """Audit record explaining how dynamic top-k selected the final context chunks."""

    candidate_count: int = Field(ge=0)
    selected_count: int = Field(ge=0)
    cutoff_reason: str
    top_score: float = Field(default=0.0, ge=0.0, le=1.0)
    cutoff_score: float = Field(default=0.0, ge=0.0, le=1.0)
    total_context_tokens: int = Field(default=0, ge=0)
    dropped_chunk_ids: list[str] = Field(default_factory=list)
    parent_expansion_applied: bool = False
    collapsed_child_count: int = Field(default=0, ge=0)
    expanded_parent_count: int = Field(default=0, ge=0)


class ConfidenceLevel(StrEnum):
    """Discrete retrieval confidence band."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CRAGGrade(StrEnum):
    """Corrective RAG sufficiency verdict for ambiguous retrieval bands."""

    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class CRAGJudgment(BaseModel):
    """Structured output from the lightweight ambiguous-band CRAG grader."""

    grade: CRAGGrade
    missing_aspects: list[str] = Field(default_factory=list)
    reasoning: str = Field(min_length=1)


class ConfidenceAction(StrEnum):
    """Control-flow action selected after confidence scoring."""

    PROCEED = "proceed"
    REWRITE = "rewrite"
    WEB_FALLBACK = "web_fallback"
    ABSTAIN = "abstain"


class ConfidenceAssessment(BaseModel):
    """Multi-signal retrieval confidence breakdown and control-flow decision."""

    top1_score: float = Field(ge=0.0, le=1.0)
    mean_topk_score: float = Field(ge=0.0, le=1.0)
    subquery_coverage: float = Field(ge=0.0, le=1.0)
    deterministic_score: float = Field(ge=0.0, le=1.0)
    crag_grade: CRAGGrade | None = None
    final_score: float = Field(ge=0.0, le=1.0)
    level: ConfidenceLevel
    recommended_action: ConfidenceAction
    reasoning: str = ""
