"""Small-to-Big Parent-Child passage deduplication and token-budgeted expansion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from adaptiverag.ingestion.chunker import count_tokens
from adaptiverag.schemas.retrieval import DynamicKDecision, EvidenceChunk


def _max_optional(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _effective_score(chunk: EvidenceChunk) -> float:
    if chunk.rerank_score is not None:
        return min(1.0, max(0.0, float(chunk.rerank_score)))
    if chunk.fusion_score is not None:
        return min(1.0, max(0.0, float(chunk.fusion_score)))
    return 0.0


def _estimate_tokens(chunk: EvidenceChunk) -> int:
    return chunk.token_count if chunk.token_count > 0 else max(1, count_tokens(chunk.text))


def _has_parent_payload(chunk: EvidenceChunk) -> bool:
    return bool(chunk.parent_id and chunk.parent_text and chunk.parent_text.strip())


def expand_to_parent_chunks(
    selected_children: Sequence[EvidenceChunk],
    *,
    max_context_tokens: int,
    decision: DynamicKDecision | None = None,
) -> tuple[list[EvidenceChunk], DynamicKDecision | None]:
    """Deduplicate selected child chunks by ``parent_id`` and expand to full parent passages.

    Design invariants:
    - **Zero-regression pass-through:** Legacy flat chunks (without ``parent_id`` / ``parent_text``)
      and web chunks pass through untouched in rank order.
    - **Rank & provenance preservation:** Sibling child chunks sharing a ``parent_id`` collapse
      into one parent passage positioned at the rank of the highest-scoring child, retaining
      the peak retrieval/rerank scores and the union of ``matched_sub_queries``.
    - **Recall-preserving token budget guard:** Always keeps the #1 passage. For subsequent
      passages, expands to the full parent if it fits ``max_context_tokens``; if the expanded
      parent would overflow the budget, gracefully retains the compact primary child chunk when
      it fits so multi-document recall is never sacrificed for a single oversized section.
    """
    if not selected_children:
        return [], decision

    if not any(_has_parent_payload(c) for c in selected_children):
        return list(selected_children), decision

    # Group selected chunks preserving first-seen (highest-ranked) order.
    ordered_keys: list[str] = []
    grouped: dict[str, list[EvidenceChunk]] = {}
    for chunk in selected_children:
        key = (
            f"parent::{chunk.parent_id}"
            if _has_parent_payload(chunk)
            else f"chunk::{chunk.chunk_id}"
        )
        if key not in grouped:
            ordered_keys.append(key)
            grouped[key] = [chunk]
        else:
            grouped[key].append(chunk)

    budgeted: list[EvidenceChunk] = []
    dropped_ids: list[str] = []
    running_tokens = 0
    budget_clipped = False

    for idx, key in enumerate(ordered_keys):
        siblings = grouped[key]
        primary = siblings[0]

        if not _has_parent_payload(primary):
            tok = _estimate_tokens(primary)
            if idx > 0 and (running_tokens + tok) > max_context_tokens:
                budget_clipped = True
                dropped_ids.extend(c.chunk_id for c in siblings)
                break
            budgeted.append(primary)
            running_tokens += tok
            continue

        assert primary.parent_id is not None
        assert primary.parent_text is not None
        parent_text = primary.parent_text.strip()
        parent_tokens = (
            primary.parent_token_count
            if primary.parent_token_count is not None and primary.parent_token_count > 0
            else max(1, count_tokens(parent_text))
        )

        # Merge scores & provenance across all matched sibling children
        best_dense: float | None = None
        best_sparse: float | None = None
        best_fusion: float | None = None
        best_rerank_raw: float | None = None
        best_rerank: float | None = None
        merged_sub_queries: list[str] = []
        matched_child_ids: list[str] = []
        matched_child_sections: list[str] = []

        for child in siblings:
            best_dense = _max_optional(best_dense, child.dense_score)
            best_sparse = _max_optional(best_sparse, child.sparse_score)
            best_fusion = _max_optional(best_fusion, child.fusion_score)
            best_rerank_raw = _max_optional(best_rerank_raw, child.rerank_raw_score)
            best_rerank = _max_optional(best_rerank, child.rerank_score)
            for sq in child.matched_sub_queries:
                if sq not in merged_sub_queries:
                    merged_sub_queries.append(sq)
            matched_child_ids.append(child.chunk_id)
            if child.section_path and child.section_path not in matched_child_sections:
                matched_child_sections.append(child.section_path)

        # Check context token budget: always preserve passage #1; for idx > 0 try parent -> child fallback
        if idx == 0 or (running_tokens + parent_tokens) <= max_context_tokens:
            merged_meta: dict[str, Any] = {
                **primary.metadata,
                "parent_expansion_applied": True,
                "matched_child_count": len(siblings),
                "matched_child_sections": matched_child_sections,
            }
            expanded = primary.model_copy(
                update={
                    "chunk_id": primary.parent_id,
                    "text": parent_text,
                    "section_path": primary.parent_section_path or primary.section_path,
                    "metadata": merged_meta,
                    "dense_score": best_dense,
                    "sparse_score": best_sparse,
                    "fusion_score": best_fusion,
                    "rerank_raw_score": best_rerank_raw,
                    "rerank_score": best_rerank,
                    "matched_sub_queries": merged_sub_queries,
                    "token_count": parent_tokens,
                    "parent_text": None,
                    "expanded_to_parent": True,
                    "matched_child_ids": matched_child_ids,
                }
            )
            budgeted.append(expanded)
            running_tokens += parent_tokens
            continue

        # Parent passage would exceed remaining budget: fall back to compact primary child if it fits
        child_tokens = _estimate_tokens(primary)
        if (running_tokens + child_tokens) <= max_context_tokens:
            fallback_meta: dict[str, Any] = {
                **primary.metadata,
                "parent_expansion_skipped": "token_budget",
                "matched_child_count": len(siblings),
            }
            compact = primary.model_copy(
                update={
                    "metadata": fallback_meta,
                    "dense_score": best_dense,
                    "sparse_score": best_sparse,
                    "fusion_score": best_fusion,
                    "rerank_raw_score": best_rerank_raw,
                    "rerank_score": best_rerank,
                    "matched_sub_queries": merged_sub_queries,
                    "token_count": child_tokens,
                    "parent_text": None,
                    "expanded_to_parent": False,
                    "matched_child_ids": matched_child_ids,
                }
            )
            budgeted.append(compact)
            running_tokens += child_tokens
        else:
            budget_clipped = True
            dropped_ids.extend(matched_child_ids)
            break

    expanded_parent_count = sum(1 for chunk in budgeted if chunk.expanded_to_parent)
    expansion_applied = expanded_parent_count > 0 or len(budgeted) < len(selected_children)

    updated_decision = decision
    if decision is not None:
        combined_dropped = list(dict.fromkeys([*decision.dropped_chunk_ids, *dropped_ids]))
        new_reason = "parent_token_budget" if budget_clipped else decision.cutoff_reason
        cutoff_score = _effective_score(budgeted[-1]) if budgeted else 0.0
        updated_decision = decision.model_copy(
            update={
                "selected_count": len(budgeted),
                "cutoff_reason": new_reason,
                "cutoff_score": round(cutoff_score, 6),
                "total_context_tokens": running_tokens,
                "dropped_chunk_ids": combined_dropped,
                "parent_expansion_applied": expansion_applied,
                "collapsed_child_count": len(selected_children),
                "expanded_parent_count": expanded_parent_count,
            }
        )

    return budgeted, updated_decision
