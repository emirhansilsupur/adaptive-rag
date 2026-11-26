"""Multi-signal retrieval confidence scoring with ambiguous-band CRAG grading."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import structlog
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from adaptiverag.config import ConfidenceConfig
from adaptiverag.llm.cache import SQLiteLLMCache
from adaptiverag.llm.cost import BudgetExceededError, CostTracker
from adaptiverag.llm.structured import failed_call_usage, invoke_structured
from adaptiverag.observability.prompts import (
    DEFAULT_PROMPTS,
    PROMPT_CRAG_GRADER,
    PromptManager,
    ResolvedPrompt,
)
from adaptiverag.schemas.response import TokenUsage
from adaptiverag.schemas.retrieval import (
    ConfidenceAction,
    ConfidenceAssessment,
    ConfidenceLevel,
    CRAGGrade,
    CRAGJudgment,
    EvidenceChunk,
    SourceType,
)

logger = structlog.get_logger(__name__)

_CRAG_MAX_CHUNKS_IN_PROMPT = 6
_CRAG_EXCERPT_CHAR_LIMIT = 2400
_CRAG_PARENT_EXCERPT_CHAR_LIMIT = 4500

_CRAG_SYSTEM_PROMPT = DEFAULT_PROMPTS[PROMPT_CRAG_GRADER].template


@dataclass(frozen=True)
class DeterministicConfidenceSignals:
    """Breakdown of zero-LLM deterministic retrieval quality signals."""

    top1_score: float
    mean_topk_score: float
    subquery_coverage: float
    deterministic_score: float


def _chunk_score(chunk: EvidenceChunk) -> float:
    if chunk.rerank_score is not None:
        return min(1.0, max(0.0, float(chunk.rerank_score)))
    if chunk.fusion_score is not None:
        return min(1.0, max(0.0, float(chunk.fusion_score)))
    return 0.0


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def compute_deterministic_confidence(
    selected_chunks: Sequence[EvidenceChunk],
    sub_queries: Sequence[str],
    config: ConfidenceConfig,
) -> DeterministicConfidenceSignals:
    """Compute composite deterministic retrieval confidence from calibrated scores & coverage.

    Formula:
        C_det = w_top1 * S_top1 + w_mean * S_mean_topk + w_cov * S_subquery_coverage
    """
    if not selected_chunks:
        return DeterministicConfidenceSignals(
            top1_score=0.0,
            mean_topk_score=0.0,
            subquery_coverage=0.0,
            deterministic_score=0.0,
        )

    scores = [_chunk_score(chunk) for chunk in selected_chunks]
    top1 = max(scores)
    mean_topk = sum(scores) / len(scores)

    unique_sub_queries = list(
        dict.fromkeys(q.strip().lower() for q in sub_queries if q and q.strip())
    )
    if not unique_sub_queries:
        coverage = 1.0
    else:
        matched_norm = {
            q.strip().lower()
            for chunk in selected_chunks
            for q in chunk.matched_sub_queries
            if q and q.strip()
        }
        covered = sum(1 for sq in unique_sub_queries if sq in matched_norm)
        coverage = covered / len(unique_sub_queries)

    composite = (
        config.weight_top1 * top1
        + config.weight_mean_topk * mean_topk
        + config.weight_subquery_coverage * coverage
    )

    return DeterministicConfidenceSignals(
        top1_score=round(_clamp01(top1), 4),
        mean_topk_score=round(_clamp01(mean_topk), 4),
        subquery_coverage=round(_clamp01(coverage), 4),
        deterministic_score=round(_clamp01(composite), 4),
    )


def _select_recovery_action(
    *,
    rewrite_attempts: int,
    config: ConfidenceConfig,
    web_fallback_enabled: bool,
    allow_proceed_when_exhausted: bool = False,
) -> ConfidenceAction:
    if rewrite_attempts < config.max_corrective_rewrites:
        return ConfidenceAction.REWRITE
    if web_fallback_enabled:
        return ConfidenceAction.WEB_FALLBACK
    if allow_proceed_when_exhausted:
        return ConfidenceAction.PROCEED
    return ConfidenceAction.ABSTAIN


def _build_crag_user_prompt(
    query: str,
    sub_queries: Sequence[str],
    selected_chunks: Sequence[EvidenceChunk],
) -> str:
    lines: list[str] = [f"User Question: {query.strip()}"]
    cleaned_subs = [q.strip() for q in sub_queries if q and q.strip()]
    if len(cleaned_subs) > 1:
        lines.append("Analytical Sub-Queries:")
        for idx, sq in enumerate(cleaned_subs, start=1):
            lines.append(f"  {idx}. {sq}")

    lines.append("\nRetrieved Evidence Excerpts:")
    for idx, chunk in enumerate(selected_chunks[:_CRAG_MAX_CHUNKS_IN_PROMPT], start=1):
        label = chunk.citation_id or f"P{idx}"
        header = " | ".join(p for p in (chunk.title, chunk.section_path) if p)
        score = _chunk_score(chunk)
        excerpt = chunk.text.strip()
        char_cap = (
            _CRAG_PARENT_EXCERPT_CHAR_LIMIT
            if chunk.expanded_to_parent
            else _CRAG_EXCERPT_CHAR_LIMIT
        )
        if len(excerpt) > char_cap:
            excerpt = excerpt[:char_cap].rstrip() + "..."
        lines.append(f"[{label}] ({header or chunk.doc_id}, score={score:.2f})\n{excerpt}")

    return "\n".join(lines)


async def assess_retrieval_confidence(
    *,
    query: str,
    selected_chunks: Sequence[EvidenceChunk],
    sub_queries: Sequence[str],
    config: ConfidenceConfig,
    rewrite_attempts: int = 0,
    web_fallback_enabled: bool = True,
    requires_fresh_web_info: bool = False,
    fast_model: BaseChatModel | None = None,
    fast_model_name: str = "",
    fast_model_provider: str | None = None,
    temperature: float = 0.0,
    cache: SQLiteLLMCache | None = None,
    cost_tracker: CostTracker | None = None,
    callbacks: list[Any] | None = None,
    prompt_manager: PromptManager | None = None,
    prompt_out: list[ResolvedPrompt] | None = None,
) -> tuple[ConfidenceAssessment, TokenUsage]:
    """Evaluate retrieval quality via deterministic fast paths and optional CRAG grading.

    - ``deterministic_score >= high_threshold``: zero-cost ``HIGH`` / ``PROCEED`` fast path.
    - ``deterministic_score < low_threshold`` or empty chunks: zero-cost ``LOW`` recovery path.
    - ``low_threshold <= deterministic_score < high_threshold``: invokes lightweight
      structured ``CRAGJudgment`` grader when enabled and a fast model is provided.
    - Queries explicitly marked ``requires_fresh_web_info=True`` route to ``WEB_FALLBACK``
      immediately if web evidence has not yet been gathered and web fallback is enabled.
    """
    signals = compute_deterministic_confidence(selected_chunks, sub_queries, config)
    has_web_evidence = any(chunk.source_type == SourceType.WEB for chunk in selected_chunks)

    # Freshness override: static local docs cannot guarantee current/live release facts.
    if requires_fresh_web_info and web_fallback_enabled and not has_web_evidence:
        level = (
            ConfidenceLevel.MEDIUM
            if signals.deterministic_score >= config.low_threshold
            else ConfidenceLevel.LOW
        )
        return (
            ConfidenceAssessment(
                top1_score=signals.top1_score,
                mean_topk_score=signals.mean_topk_score,
                subquery_coverage=signals.subquery_coverage,
                deterministic_score=signals.deterministic_score,
                crag_grade=None,
                final_score=signals.deterministic_score,
                level=level,
                recommended_action=ConfidenceAction.WEB_FALLBACK,
                reasoning=(
                    "Query requires fresh external information not guaranteed in the static "
                    "local corpus; routing to web fallback."
                ),
            ),
            TokenUsage(),
        )

    # Empty retrieval after tau_min cutoff
    if not selected_chunks:
        action = _select_recovery_action(
            rewrite_attempts=rewrite_attempts,
            config=config,
            web_fallback_enabled=web_fallback_enabled,
            allow_proceed_when_exhausted=False,
        )
        return (
            ConfidenceAssessment(
                top1_score=0.0,
                mean_topk_score=0.0,
                subquery_coverage=0.0,
                deterministic_score=0.0,
                crag_grade=None,
                final_score=0.0,
                level=ConfidenceLevel.LOW,
                recommended_action=action,
                reasoning=(
                    f"No evidence chunks survived relevance floor; recommending {action.value}."
                ),
            ),
            TokenUsage(),
        )

    # Zero-cost HIGH confidence fast path
    if signals.deterministic_score >= config.high_threshold:
        return (
            ConfidenceAssessment(
                top1_score=signals.top1_score,
                mean_topk_score=signals.mean_topk_score,
                subquery_coverage=signals.subquery_coverage,
                deterministic_score=signals.deterministic_score,
                crag_grade=None,
                final_score=signals.deterministic_score,
                level=ConfidenceLevel.HIGH,
                recommended_action=ConfidenceAction.PROCEED,
                reasoning=(
                    f"Deterministic confidence {signals.deterministic_score:.3f} >= "
                    f"high threshold {config.high_threshold:.2f}; bypassing CRAG grader."
                ),
            ),
            TokenUsage(),
        )

    # Zero-cost LOW confidence fast path
    if signals.deterministic_score < config.low_threshold:
        action = _select_recovery_action(
            rewrite_attempts=rewrite_attempts,
            config=config,
            web_fallback_enabled=web_fallback_enabled,
            allow_proceed_when_exhausted=False,
        )
        return (
            ConfidenceAssessment(
                top1_score=signals.top1_score,
                mean_topk_score=signals.mean_topk_score,
                subquery_coverage=signals.subquery_coverage,
                deterministic_score=signals.deterministic_score,
                crag_grade=None,
                final_score=signals.deterministic_score,
                level=ConfidenceLevel.LOW,
                recommended_action=action,
                reasoning=(
                    f"Deterministic confidence {signals.deterministic_score:.3f} < "
                    f"low threshold {config.low_threshold:.2f}; recommending {action.value}."
                ),
            ),
            TokenUsage(),
        )

    # Ambiguous band [low_threshold, high_threshold): invoke CRAG grader if enabled
    grader_usage = TokenUsage()
    if config.enable_crag_grader_in_ambiguous_band and fast_model is not None:
        resolved_prompt = (
            await prompt_manager.aresolve(PROMPT_CRAG_GRADER)
            if prompt_manager is not None
            else ResolvedPrompt(
                name=PROMPT_CRAG_GRADER,
                text=_CRAG_SYSTEM_PROMPT,
                source="local_fallback",
            )
        )
        if prompt_out is not None:
            prompt_out.append(resolved_prompt)
        messages = [
            SystemMessage(content=resolved_prompt.text),
            HumanMessage(content=_build_crag_user_prompt(query, sub_queries, selected_chunks)),
        ]
        try:
            crag_result = await invoke_structured(
                fast_model,
                CRAGJudgment,
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
            judgment = crag_result.parsed

            if judgment.grade == CRAGGrade.SUFFICIENT:
                upgraded = round(
                    _clamp01(max(signals.deterministic_score, config.high_threshold + 0.08)), 4
                )
                return (
                    ConfidenceAssessment(
                        top1_score=signals.top1_score,
                        mean_topk_score=signals.mean_topk_score,
                        subquery_coverage=signals.subquery_coverage,
                        deterministic_score=signals.deterministic_score,
                        crag_grade=judgment.grade,
                        final_score=upgraded,
                        level=ConfidenceLevel.HIGH,
                        recommended_action=ConfidenceAction.PROCEED,
                        reasoning=f"CRAG SUFFICIENT: {judgment.reasoning}",
                    ),
                    crag_result.usage,
                )

            if judgment.grade == CRAGGrade.PARTIAL:
                action = _select_recovery_action(
                    rewrite_attempts=rewrite_attempts,
                    config=config,
                    web_fallback_enabled=web_fallback_enabled,
                    allow_proceed_when_exhausted=True,
                )
                missing_note = (
                    f" Missing: {', '.join(judgment.missing_aspects)}."
                    if judgment.missing_aspects
                    else ""
                )
                return (
                    ConfidenceAssessment(
                        top1_score=signals.top1_score,
                        mean_topk_score=signals.mean_topk_score,
                        subquery_coverage=signals.subquery_coverage,
                        deterministic_score=signals.deterministic_score,
                        crag_grade=judgment.grade,
                        final_score=signals.deterministic_score,
                        level=ConfidenceLevel.MEDIUM,
                        recommended_action=action,
                        reasoning=f"CRAG PARTIAL: {judgment.reasoning}{missing_note}",
                    ),
                    crag_result.usage,
                )

            # CRAGGrade.INSUFFICIENT
            downgraded = round(
                _clamp01(min(signals.deterministic_score, max(0.0, config.low_threshold - 0.05))), 4
            )
            action = _select_recovery_action(
                rewrite_attempts=rewrite_attempts,
                config=config,
                web_fallback_enabled=web_fallback_enabled,
                allow_proceed_when_exhausted=False,
            )
            return (
                ConfidenceAssessment(
                    top1_score=signals.top1_score,
                    mean_topk_score=signals.mean_topk_score,
                    subquery_coverage=signals.subquery_coverage,
                    deterministic_score=signals.deterministic_score,
                    crag_grade=judgment.grade,
                    final_score=downgraded,
                    level=ConfidenceLevel.LOW,
                    recommended_action=action,
                    reasoning=f"CRAG INSUFFICIENT: {judgment.reasoning}",
                ),
                crag_result.usage,
            )
        except BudgetExceededError:
            raise
        except Exception as exc:
            grader_usage = failed_call_usage(exc)
            logger.warning(
                "crag_grader_failed_falling_back_to_deterministic",
                error=str(exc),
                deterministic_score=signals.deterministic_score,
            )

    # Deterministic fallback when CRAG grader is disabled, unprovided, or errored
    action = _select_recovery_action(
        rewrite_attempts=rewrite_attempts,
        config=config,
        web_fallback_enabled=web_fallback_enabled,
        allow_proceed_when_exhausted=True,
    )
    return (
        ConfidenceAssessment(
            top1_score=signals.top1_score,
            mean_topk_score=signals.mean_topk_score,
            subquery_coverage=signals.subquery_coverage,
            deterministic_score=signals.deterministic_score,
            crag_grade=None,
            final_score=signals.deterministic_score,
            level=ConfidenceLevel.MEDIUM,
            recommended_action=action,
            reasoning=(
                f"Ambiguous deterministic confidence {signals.deterministic_score:.3f} "
                f"(CRAG grader inactive); recommending {action.value}."
            ),
        ),
        grader_usage,
    )
