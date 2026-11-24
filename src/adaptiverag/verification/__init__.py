"""Multi-signal retrieval confidence scoring and two-tier citation verification."""

from adaptiverag.verification.citation_validator import (
    extract_citations,
    find_uncited_factual_sentences,
    validate_citations,
    validate_citations_structural,
)
from adaptiverag.verification.confidence import (
    DeterministicConfidenceSignals,
    assess_retrieval_confidence,
    compute_deterministic_confidence,
)
from adaptiverag.verification.guardrails import (
    GuardrailBlockedError,
    GuardrailDecision,
    InputGuardrailConfig,
    PiiFinding,
    detect_prompt_injection,
    sanitize_untrusted_passage,
    screen_query_input,
)

__all__ = [
    "DeterministicConfidenceSignals",
    "GuardrailBlockedError",
    "GuardrailDecision",
    "InputGuardrailConfig",
    "PiiFinding",
    "assess_retrieval_confidence",
    "compute_deterministic_confidence",
    "detect_prompt_injection",
    "extract_citations",
    "find_uncited_factual_sentences",
    "sanitize_untrusted_passage",
    "screen_query_input",
    "validate_citations",
    "validate_citations_structural",
]
