"""Three-stage adaptive dynamic top-k context selection and citation labeling."""

from __future__ import annotations

from collections.abc import Sequence

from adaptiverag.ingestion.chunker import count_tokens
from adaptiverag.schemas.query import StrategyProfile
from adaptiverag.schemas.retrieval import DynamicKDecision, EvidenceChunk


def _effective_score(chunk: EvidenceChunk) -> float:
    """Return calibrated [0.0, 1.0] relevance score, falling back to clamped fusion score."""
    if chunk.rerank_score is not None:
        return min(1.0, max(0.0, float(chunk.rerank_score)))
    if chunk.fusion_score is not None:
        return min(1.0, max(0.0, float(chunk.fusion_score)))
    return 0.0


def _chunk_token_count(chunk: EvidenceChunk) -> int:
    """Return precomputed token count or estimate deterministically on demand."""
    if chunk.token_count > 0:
        return chunk.token_count
    return max(1, count_tokens(chunk.text))


def assign_citation_ids(
    chunks: Sequence[EvidenceChunk],
    *,
    prefix: str = "C",
    start_index: int = 1,
) -> list[EvidenceChunk]:
    """Return copies of chunks stamped with sequential prompt citation labels (e.g., C1..Ck)."""
    return [
        chunk.model_copy(update={"citation_id": f"{prefix}{start_index + idx}"})
        for idx, chunk in enumerate(chunks)
    ]


def select_dynamic_top_k(
    chunks: Sequence[EvidenceChunk],
    strategy: StrategyProfile,
) -> tuple[list[EvidenceChunk], DynamicKDecision]:
    """Select an adaptive number of evidence chunks using a 3-stage cutoff algorithm.

    Stage 1 — Absolute Quality Floor (``tau_min``):
        Drop noisy candidates below ``strategy.tau_min``. If zero candidates survive,
        return an empty list rather than force-padding irrelevant chunks, allowing
        downstream Confidence Gating to detect out-of-corpus queries.

    Stage 2 — Relative Drop & Elbow Gap (``alpha_relative``, ``delta_gap``):
        Keep consecutive descending-ranked chunks while
        ``score >= alpha_relative * top1`` and ``(prev_score - score) <= delta_gap``.

    Stage 3 — ``[rerank_min_k, rerank_max_k]`` Clamp & Token Budget Guard:
        Ensure at least ``min(rerank_min_k, len(above_floor))`` quality-passing chunks
        (preserving multi-aspect recall for analytical queries), cap at ``rerank_max_k``,
        and enforce ``strategy.max_context_tokens`` (always preserving the top-1 valid chunk).
    """
    candidate_count = len(chunks)
    if candidate_count == 0:
        return [], DynamicKDecision(
            candidate_count=0,
            selected_count=0,
            cutoff_reason="no_candidates",
            top_score=0.0,
            cutoff_score=0.0,
            total_context_tokens=0,
            dropped_chunk_ids=[],
        )

    if not strategy.needs_retrieval or strategy.rerank_max_k <= 0:
        return [], DynamicKDecision(
            candidate_count=candidate_count,
            selected_count=0,
            cutoff_reason="retrieval_disabled",
            top_score=0.0,
            cutoff_score=0.0,
            total_context_tokens=0,
            dropped_chunk_ids=[c.chunk_id for c in chunks],
        )

    # Sort defensively descending by calibrated score so callers cannot break Stage 2.
    sorted_chunks = sorted(
        chunks,
        key=lambda c: (
            _effective_score(c),
            c.rerank_raw_score if c.rerank_raw_score is not None else float("-inf"),
            c.fusion_score if c.fusion_score is not None else 0.0,
        ),
        reverse=True,
    )
    top_score = _effective_score(sorted_chunks[0])

    # Stage 1: Absolute relevance floor
    above_floor = [c for c in sorted_chunks if _effective_score(c) >= strategy.tau_min]
    if not above_floor:
        return [], DynamicKDecision(
            candidate_count=candidate_count,
            selected_count=0,
            cutoff_reason="below_tau_min",
            top_score=round(top_score, 6),
            cutoff_score=0.0,
            total_context_tokens=0,
            dropped_chunk_ids=[c.chunk_id for c in sorted_chunks],
        )

    # Stage 2: Relative drop & adjacent elbow gap
    selected: list[EvidenceChunk] = [above_floor[0]]
    cutoff_reason = (
        "below_tau_min" if len(above_floor) < len(sorted_chunks) else "all_candidates_retained"
    )
    stage2_triggered = False

    for idx in range(1, len(above_floor)):
        prev_score = _effective_score(above_floor[idx - 1])
        curr_score = _effective_score(above_floor[idx])
        gap_exceeded = (prev_score - curr_score) > strategy.delta_gap + 1e-12
        rel_failed = curr_score < (strategy.alpha_relative * top_score) - 1e-12

        if gap_exceeded or rel_failed:
            cutoff_reason = "elbow_gap" if gap_exceeded else "relative_drop"
            stage2_triggered = True
            break
        selected.append(above_floor[idx])

    # Stage 3a: min_k floor (only draws from chunks that already passed tau_min)
    target_min_k = min(strategy.rerank_min_k, len(above_floor))
    if len(selected) < target_min_k:
        selected = above_floor[:target_min_k]
        if stage2_triggered:
            cutoff_reason = "min_k_floor"

    # Stage 3b: max_k cap
    if len(selected) > strategy.rerank_max_k:
        selected = selected[: strategy.rerank_max_k]
        cutoff_reason = "max_k_cap"

    # Stage 3c: Context token budget guard (always retain at least the #1 valid chunk)
    budgeted: list[EvidenceChunk] = []
    running_tokens = 0
    for idx, chunk in enumerate(selected):
        chunk_tokens = _chunk_token_count(chunk)
        if idx > 0 and (running_tokens + chunk_tokens) > strategy.max_context_tokens:
            cutoff_reason = "token_budget"
            break
        budgeted.append(chunk)
        running_tokens += chunk_tokens

    selected_id_set = {c.chunk_id for c in budgeted}
    dropped_ids = [c.chunk_id for c in sorted_chunks if c.chunk_id not in selected_id_set]
    cutoff_score = _effective_score(budgeted[-1]) if budgeted else 0.0

    decision = DynamicKDecision(
        candidate_count=candidate_count,
        selected_count=len(budgeted),
        cutoff_reason=cutoff_reason,
        top_score=round(top_score, 6),
        cutoff_score=round(cutoff_score, 6),
        total_context_tokens=running_tokens,
        dropped_chunk_ids=dropped_ids,
    )
    return budgeted, decision
