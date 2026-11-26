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


class SemanticAuditStatus(StrEnum):
    """Whether the Tier-2 entailment judge produced the claim verdicts behind a report."""

    COMPLETED = "completed"
    """The judge returned claim verdicts, and they count toward ``passed``."""

    SKIPPED = "skipped"
    """Not run by design: disabled, no judge model, nothing to audit, or Tier-1 already failed."""

    FAILED = "failed"
    """It should have run, but the judge errored or returned no verdicts for a cited answer."""


class CitationReport(BaseModel):
    """Combined Tier-1 deterministic and Tier-2 semantic citation validation report.

    ``passed`` fails closed: when the semantic audit was required but ``FAILED``, the answer
    is not reported as passed, even if its citation structure is fine. ``structural_passed``
    keeps the Tier-1 verdict on its own, so callers can tell a broken answer from a broken judge.
    ``claim_support_rate`` is the entailment support rate only when ``semantic_audit`` is
    ``COMPLETED``; otherwise it is the share of substantive sentences that carry a citation.
    ``empty_answer`` marks an answer with no text: there was nothing to audit, so it never passes.
    """

    total_citations_found: int = Field(default=0, ge=0)
    valid_citation_ids: list[str] = Field(default_factory=list)
    invalid_citation_ids: list[str] = Field(default_factory=list)
    uncited_factual_sentences: list[str] = Field(default_factory=list)
    claim_verifications: list[ClaimVerification] = Field(default_factory=list)
    citation_precision: float = Field(default=1.0, ge=0.0, le=1.0)
    claim_support_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    passed: bool = True
    regeneration_triggered: bool = False
    structural_passed: bool = True
    semantic_audit: SemanticAuditStatus = SemanticAuditStatus.SKIPPED
    semantic_audit_error: str | None = None
    empty_answer: bool = False
