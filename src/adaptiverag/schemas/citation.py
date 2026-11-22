"""Two-tier citation verification and grounding schemas."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ClaimVerdict(StrEnum):
    """Entailment verdict for a single factual claim against its cited chunks."""

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


class ClaimVerification(BaseModel):
    """Verification result for one atomic claim extracted from the draft answer."""

    claim: str = Field(min_length=1)
    cited_ids: list[str] = Field(default_factory=list)
    verdict: ClaimVerdict
    explanation: str = ""


class ClaimAuditBatch(BaseModel):
    """Structured LLM output for Tier-2 claim-level entailment auditing."""

    verifications: list[ClaimVerification] = Field(default_factory=list)


class CitationReport(BaseModel):
    """Combined Tier-1 deterministic and Tier-2 semantic citation validation report."""

    total_citations_found: int = Field(default=0, ge=0)
    valid_citation_ids: list[str] = Field(default_factory=list)
    invalid_citation_ids: list[str] = Field(default_factory=list)
    uncited_factual_sentences: list[str] = Field(default_factory=list)
    claim_verifications: list[ClaimVerification] = Field(default_factory=list)
    citation_precision: float = Field(default=1.0, ge=0.0, le=1.0)
    claim_support_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    passed: bool = True
    regeneration_triggered: bool = False
