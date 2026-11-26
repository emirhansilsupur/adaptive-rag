"""Two-tier citation verification: Tier-1 deterministic structural + Tier-2 semantic entailment."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import structlog
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from adaptiverag.config import CitationValidationConfig
from adaptiverag.llm.cache import SQLiteLLMCache
from adaptiverag.llm.cost import BudgetExceededError, CostTracker
from adaptiverag.llm.structured import failed_call_usage, invoke_structured
from adaptiverag.observability.prompts import (
    DEFAULT_PROMPTS,
    PROMPT_CITATION_ENTAILMENT,
    PromptManager,
    ResolvedPrompt,
)
from adaptiverag.schemas.citation import (
    CitationReport,
    ClaimAuditBatch,
    ClaimVerdict,
    SemanticAuditStatus,
)
from adaptiverag.schemas.response import TokenUsage
from adaptiverag.schemas.retrieval import EvidenceChunk

logger = structlog.get_logger(__name__)

# Matches [C1], [W2], [C1, C2], [C1; W1], while ignoring Markdown links like [C1](https://...)
_CITATION_GROUP_RE = re.compile(
    r"\[(\s*[CW]\d+(?:\s*[,;]\s*[CW]\d+)*\s*)\](?!\()",
    re.IGNORECASE,
)
_SINGLE_CITATION_RE = re.compile(r"[CW]\d+", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ0-9`\"'\-\*])")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s+")
_TABLE_SEP_RE = re.compile(r"^\s*\|?(?:\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?\s*$")
_INLINE_MD_RE = re.compile(r"[`*_~]+")

_ABSTENTION_MARKERS = (
    "cannot verify",
    "cannot answer",
    "could not find",
    "not enough evidence",
    "insufficient evidence",
    "no verified evidence",
    "not covered in the provided",
    "not mentioned in the provided",
    "yeterli kanıt",
    "yeterli doğrulanmış",
    "doğrulanmış kaynak",
    "bilgi bulunamadı",
    "yanıt veremiyorum",
    "doğrulayamıyorum",
)

# Matches _CONTEXT_CHUNK_CHAR_LIMIT (6000) so Tier-2 entailment sees the full 1000-token Parent
# passage and never flags grounded facts from later H3/H4 subsections as UNSUPPORTED.
_ENTAILMENT_EXCERPT_LIMIT = 6000

_TIER2_SYSTEM_PROMPT = DEFAULT_PROMPTS[PROMPT_CITATION_ENTAILMENT].template


def extract_citations(text: str) -> list[str]:
    """Extract all inline citation IDs (e.g., C1, W2) in order of appearance, expanding groups."""
    cleaned = _CODE_FENCE_RE.sub("", text)
    found: list[str] = []
    for match in _CITATION_GROUP_RE.finditer(cleaned):
        group_body = match.group(1)
        for token in _SINGLE_CITATION_RE.findall(group_body):
            found.append(token.upper())
    return found


def _is_abstention_or_meta_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(marker in lowered for marker in _ABSTENTION_MARKERS)


def _extract_substantive_sentences(answer: str) -> list[str]:
    """Split markdown prose into substantive declarative sentences, ignoring code & headings."""
    without_code = _CODE_FENCE_RE.sub("\n", answer)
    sentences: list[str] = []

    for raw_line in without_code.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _HEADING_RE.match(line) or _TABLE_SEP_RE.match(line):
            continue
        if line in {"---", "***", "___"}:
            continue

        # Strip markdown list markers so bullet points are inspected cleanly.
        line = _LIST_PREFIX_RE.sub("", line).strip()
        if not line:
            continue

        # Split paragraph/bullet lines into individual sentences.
        for candidate in _SENTENCE_SPLIT_RE.split(line):
            sent = candidate.strip()
            if not sent:
                continue
            # Skip short lead-in lines like "Key differences:" or "Temel farklar:"
            if sent.endswith(":"):
                continue

            plain = _INLINE_MD_RE.sub("", sent).strip()
            # Ignore trivial transitions/labels with too few words or characters.
            words = plain.split()
            if len(plain) < 25 or len(words) < 5:
                continue
            if _is_abstention_or_meta_sentence(plain):
                continue

            sentences.append(sent)

    return sentences


def find_uncited_factual_sentences(answer: str) -> list[str]:
    """Identify substantive factual statements in the answer that lack any [C#]/[W#] citation."""
    uncited: list[str] = []
    for sentence in _extract_substantive_sentences(answer):
        if not _CITATION_GROUP_RE.search(sentence):
            uncited.append(sentence)
    return uncited


def validate_citations_structural(
    answer: str,
    evidence_chunks: Sequence[EvidenceChunk],
    *,
    config: CitationValidationConfig | None = None,
) -> CitationReport:
    """Tier-1 zero-LLM deterministic verification of citation IDs and uncited statements."""
    policy = config or CitationValidationConfig()
    valid_id_set = {
        chunk.citation_id.strip().upper()
        for chunk in evidence_chunks
        if chunk.citation_id and chunk.citation_id.strip()
    }

    all_citations = extract_citations(answer)
    total_found = len(all_citations)

    valid_seen: list[str] = []
    invalid_seen: list[str] = []
    valid_occurrences = 0

    for cid in all_citations:
        if cid in valid_id_set:
            valid_occurrences += 1
            if cid not in valid_seen:
                valid_seen.append(cid)
        elif cid not in invalid_seen:
            invalid_seen.append(cid)

    substantive_sentences = _extract_substantive_sentences(answer)
    uncited_sentences = [
        sent for sent in substantive_sentences if not _CITATION_GROUP_RE.search(sent)
    ]

    if total_found > 0:
        citation_precision = round(valid_occurrences / total_found, 4)
    elif uncited_sentences and valid_id_set:
        citation_precision = 0.0
    else:
        citation_precision = 1.0

    if not substantive_sentences:
        structural_support_rate = 1.0 if not invalid_seen else 0.0
    else:
        cited_sentences = len(substantive_sentences) - len(uncited_sentences)
        if valid_id_set and valid_occurrences == 0:
            structural_support_rate = 0.0
        else:
            structural_support_rate = round(cited_sentences / len(substantive_sentences), 4)

    missing_all_citations = bool(valid_id_set) and total_found == 0 and bool(uncited_sentences)
    passed = (
        len(invalid_seen) == 0
        and not missing_all_citations
        and structural_support_rate >= policy.min_claim_support_rate
    )

    return CitationReport(
        total_citations_found=total_found,
        valid_citation_ids=valid_seen,
        invalid_citation_ids=invalid_seen,
        uncited_factual_sentences=uncited_sentences,
        claim_verifications=[],
        citation_precision=citation_precision,
        claim_support_rate=structural_support_rate,
        passed=passed,
        regeneration_triggered=False,
        structural_passed=passed,
        semantic_audit=SemanticAuditStatus.SKIPPED,
    )


def _describe_judge_failure(exc: Exception) -> str:
    """Name why the entailment judge failed without echoing provider error text to clients."""
    if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
        return "The judge model timed out."
    if isinstance(exc, ValueError):
        # invoke_structured raises ValueError once JSON repair attempts are exhausted.
        return "The judge model's reply could not be parsed into claim verdicts."
    return f"The judge model call failed ({type(exc).__name__})."


def _semantic_audit_failed(tier1: CitationReport, reason: str) -> CitationReport:
    """Fail closed: keep the Tier-1 findings, but never report unchecked claims as passed."""
    return tier1.model_copy(
        update={
            "passed": False,
            "semantic_audit": SemanticAuditStatus.FAILED,
            "semantic_audit_error": reason,
        }
    )


def _build_entailment_user_prompt(
    answer: str,
    evidence_chunks: Sequence[EvidenceChunk],
) -> str:
    lines: list[str] = ["Draft Answer to Audit:", answer.strip(), "", "Evidence Passages:"]
    for idx, chunk in enumerate(evidence_chunks, start=1):
        label = chunk.citation_id.strip().upper() if chunk.citation_id else f"C{idx}"
        header = " | ".join(part for part in (chunk.title, chunk.section_path) if part)
        text = chunk.text.strip()
        if len(text) > _ENTAILMENT_EXCERPT_LIMIT:
            text = text[:_ENTAILMENT_EXCERPT_LIMIT].rstrip() + "..."
        lines.append(f"[{label}] ({header or chunk.doc_id})\n{text}\n")
    return "\n".join(lines)


async def validate_citations(
    *,
    answer: str,
    evidence_chunks: Sequence[EvidenceChunk],
    config: CitationValidationConfig,
    fast_model: BaseChatModel | None = None,
    fast_model_name: str = "",
    fast_model_provider: str | None = None,
    temperature: float = 0.0,
    cache: SQLiteLLMCache | None = None,
    cost_tracker: CostTracker | None = None,
    callbacks: list[Any] | None = None,
    prompt_manager: PromptManager | None = None,
    prompt_out: list[ResolvedPrompt] | None = None,
) -> tuple[CitationReport, TokenUsage]:
    """Run Tier-1 deterministic check followed by optional Tier-2 semantic entailment audit."""
    tier1 = validate_citations_structural(answer, evidence_chunks, config=config)

    if not answer.strip() or not evidence_chunks:
        return tier1, TokenUsage()

    if not config.enable_semantic_entailment or fast_model is None:
        return tier1, TokenUsage()

    # FinOps fast-fail: if the answer made factual claims with ZERO valid citations,
    # skip Tier-2 LLM entailment because regeneration is already guaranteed.
    if not tier1.valid_citation_ids and (
        tier1.uncited_factual_sentences or tier1.invalid_citation_ids
    ):
        failed = tier1.model_copy(
            update={"claim_support_rate": 0.0, "passed": False, "structural_passed": False}
        )
        return failed, TokenUsage()

    resolved_prompt = (
        await prompt_manager.aresolve(PROMPT_CITATION_ENTAILMENT)
        if prompt_manager is not None
        else ResolvedPrompt(
            name=PROMPT_CITATION_ENTAILMENT,
            text=_TIER2_SYSTEM_PROMPT,
            source="local_fallback",
        )
    )
    if prompt_out is not None:
        prompt_out.append(resolved_prompt)

    messages = [
        SystemMessage(content=resolved_prompt.text),
        HumanMessage(content=_build_entailment_user_prompt(answer, evidence_chunks)),
    ]

    try:
        audit_result = await invoke_structured(
            fast_model,
            ClaimAuditBatch,
            messages,
            model_name=fast_model_name or "unknown",
            provider=fast_model_provider,
            temperature=temperature,
            max_repair_attempts=1,
            cache=cache,
            cost_tracker=cost_tracker,
            callbacks=callbacks,
            langfuse_prompt=resolved_prompt.langfuse_prompt,
            prompt_metadata=resolved_prompt.telemetry_metadata(),
        )
    except BudgetExceededError:
        raise
    except Exception as exc:
        logger.warning(
            "citation_tier2_entailment_failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        failed = _semantic_audit_failed(tier1, _describe_judge_failure(exc))
        return failed, failed_call_usage(exc)

    verifications = audit_result.parsed.verifications
    if not verifications and tier1.valid_citation_ids:
        # The answer cites passages, so there are claims to check; an empty batch means
        # the judge skipped its job, not that every claim is fine.
        logger.warning(
            "citation_tier2_entailment_returned_no_verdicts",
            valid_citation_ids=tier1.valid_citation_ids,
        )
        failed = _semantic_audit_failed(
            tier1, "The judge model returned no claim verdicts for a cited answer."
        )
        return failed, audit_result.usage

    if verifications:
        supported_verdicts = {ClaimVerdict.SUPPORTED}
        if config.allow_partial_as_supported:
            supported_verdicts.add(ClaimVerdict.PARTIALLY_SUPPORTED)

        supported_count = sum(1 for item in verifications if item.verdict in supported_verdicts)
        claim_support_rate = round(supported_count / len(verifications), 4)
        has_contradiction = any(item.verdict == ClaimVerdict.CONTRADICTED for item in verifications)
    else:
        claim_support_rate = tier1.claim_support_rate
        has_contradiction = False

    missing_all_citations = tier1.total_citations_found == 0 and bool(
        tier1.uncited_factual_sentences
    )
    passed = (
        len(tier1.invalid_citation_ids) == 0
        and not has_contradiction
        and not missing_all_citations
        and claim_support_rate >= config.min_claim_support_rate
    )

    combined = tier1.model_copy(
        update={
            "claim_verifications": verifications,
            "claim_support_rate": claim_support_rate,
            "passed": passed,
            "semantic_audit": SemanticAuditStatus.COMPLETED,
        }
    )
    return combined, audit_result.usage
