"""Unit tests for Cross-Encoder sigmoid calibration, reranking, and 3-stage Dynamic Top-K."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest

from adaptiverag.config import load_strategies
from adaptiverag.retrieval import (
    FastEmbedCrossEncoderReranker,
    Reranker,
    assign_citation_ids,
    format_passage_for_rerank,
    select_dynamic_top_k,
    sigmoid,
)
from adaptiverag.schemas import EvidenceChunk, QueryType, StrategyProfile


class _FakeOnnxCrossEncoder:
    """Deterministic stub mimicking fastembed TextCrossEncoder.rerank raw logits."""

    def __init__(self, logit_by_substring: dict[str, float], default_logit: float = -3.0) -> None:
        self._logit_by_substring = logit_by_substring
        self._default_logit = default_logit
        self.seen_passages: list[str] = []

    def rerank(
        self,
        query: str,
        documents: Iterable[str],
        batch_size: int = 64,
        **kwargs: Any,
    ) -> Iterable[float]:
        self.seen_passages = list(documents)
        scores: list[float] = []
        for doc in self.seen_passages:
            matched = self._default_logit
            for needle, logit in self._logit_by_substring.items():
                if needle in doc:
                    matched = logit
                    break
            scores.append(matched)
        return scores


def _make_chunk(
    chunk_id: str,
    *,
    rerank_score: float | None = None,
    fusion_score: float | None = 0.5,
    token_count: int = 200,
    text: str = "Sample technical evidence chunk.",
    title: str = "Doc",
    section_path: str = "Section",
    matched_sub_queries: list[str] | None = None,
) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        doc_id="doc_1",
        title=title,
        section_path=section_path,
        text=text,
        fusion_score=fusion_score,
        rerank_score=rerank_score,
        token_count=token_count,
        matched_sub_queries=matched_sub_queries or ["q1"],
    )


def test_sigmoid_numerical_stability_and_temperature() -> None:
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(1000.0) == 1.0
    assert sigmoid(-1000.0) == 0.0
    assert sigmoid(2.0, temperature=2.0) == pytest.approx(sigmoid(1.0))

    with pytest.raises(ValueError, match="temperature"):
        sigmoid(1.0, temperature=0.0)


def test_format_passage_for_rerank_prepends_breadcrumb_once() -> None:
    chunk = _make_chunk("c1", title="Qdrant HNSW", section_path="Parameters", text="ef_construct=100")
    formatted = format_passage_for_rerank(chunk)
    assert formatted == "[Qdrant HNSW | Parameters]\nef_construct=100"

    already_prefixed = chunk.model_copy(update={"text": formatted})
    assert format_passage_for_rerank(already_prefixed) == formatted


async def test_fastembed_reranker_calibrates_sorts_and_preserves_metadata() -> None:
    reranker = FastEmbedCrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3-int8")
    assert isinstance(reranker, Reranker)

    fake_onnx = _FakeOnnxCrossEncoder(
        logit_by_substring={"ef_construct": 3.0, "PagedAttention": 0.5, "MVCC": -4.0}
    )
    reranker._model = fake_onnx  # type: ignore[assignment]

    candidates = [
        _make_chunk("c_low", text="PostgreSQL MVCC visibility rules.", matched_sub_queries=["q_mvcc"]),
        _make_chunk("c_mid", text="vLLM PagedAttention KV cache blocks.", matched_sub_queries=["q_vllm"]),
        _make_chunk("c_top", text="Qdrant HNSW ef_construct defaults to 100.", matched_sub_queries=["q_qdrant", "q_hnsw"]),
    ]

    ranked = await reranker.arerank("Qdrant ef_construct varsayılan değeri nedir?", candidates)

    assert [c.chunk_id for c in ranked] == ["c_top", "c_mid", "c_low"]
    assert ranked[0].rerank_raw_score == pytest.approx(3.0)
    assert ranked[0].rerank_score == pytest.approx(sigmoid(3.0), abs=1e-5)
    assert ranked[2].rerank_score == pytest.approx(sigmoid(-4.0), abs=1e-5)
    assert ranked[0].matched_sub_queries == ["q_qdrant", "q_hnsw"]
    assert any(p.startswith("[Doc | Section]\n") for p in fake_onnx.seen_passages)

    top2 = await reranker.arerank("ef_construct", candidates, top_n=2)
    assert [c.chunk_id for c in top2] == ["c_top", "c_mid"]
    assert await reranker.arerank("ef_construct", []) == []


def test_dynamic_k_returns_empty_when_all_chunks_below_tau_min() -> None:
    strategies = load_strategies()
    factual = strategies.for_query_type(QueryType.FACTUAL)  # tau_min=0.35, min_k=2

    garbage_chunks = [
        _make_chunk("g1", rerank_score=0.22),
        _make_chunk("g2", rerank_score=0.18),
        _make_chunk("g3", rerank_score=0.05),
    ]

    selected, decision = select_dynamic_top_k(garbage_chunks, factual)

    assert selected == []
    assert decision.selected_count == 0
    assert decision.cutoff_reason == "below_tau_min"
    assert decision.top_score == pytest.approx(0.22)
    assert decision.dropped_chunk_ids == ["g1", "g2", "g3"]


def test_dynamic_k_handles_conversational_and_empty_candidates() -> None:
    strategies = load_strategies()
    conv = strategies.for_query_type(QueryType.CONVERSATIONAL)
    factual = strategies.for_query_type(QueryType.FACTUAL)

    selected_empty, dec_empty = select_dynamic_top_k([], factual)
    assert selected_empty == []
    assert dec_empty.cutoff_reason == "no_candidates"

    selected_conv, dec_conv = select_dynamic_top_k([_make_chunk("c1", rerank_score=0.95)], conv)
    assert selected_conv == []
    assert dec_conv.cutoff_reason == "retrieval_disabled"


def test_dynamic_k_triggers_elbow_gap_and_relative_drop() -> None:
    strategies = load_strategies()
    factual = strategies.for_query_type(QueryType.FACTUAL)
    # factual: min_k=2, max_k=5, tau_min=0.35, alpha_relative=0.65, delta_gap=0.25

    # Sharp elbow gap after index 2 (0.86 -> 0.56, gap=0.30 > 0.25)
    elbow_chunks = [
        _make_chunk("c3", rerank_score=0.56),  # intentionally unsorted input
        _make_chunk("c1", rerank_score=0.92),
        _make_chunk("c4", rerank_score=0.50),
        _make_chunk("c2", rerank_score=0.86),
    ]
    sel_elbow, dec_elbow = select_dynamic_top_k(elbow_chunks, factual)
    assert [c.chunk_id for c in sel_elbow] == ["c1", "c2"]
    assert dec_elbow.cutoff_reason == "elbow_gap"
    assert dec_elbow.cutoff_score == pytest.approx(0.86)
    assert dec_elbow.dropped_chunk_ids == ["c3", "c4"]

    # Gradual relative drop below 0.65 * 0.90 = 0.585 (step gaps <= 0.14 <= 0.25)
    rel_chunks = [
        _make_chunk("r1", rerank_score=0.90),
        _make_chunk("r2", rerank_score=0.76),
        _make_chunk("r3", rerank_score=0.63),
        _make_chunk("r4", rerank_score=0.52),
    ]
    sel_rel, dec_rel = select_dynamic_top_k(rel_chunks, factual)
    assert [c.chunk_id for c in sel_rel] == ["r1", "r2", "r3"]
    assert dec_rel.cutoff_reason == "relative_drop"
    assert dec_rel.dropped_chunk_ids == ["r4"]


def test_dynamic_k_respects_analytical_min_k_and_max_k_profiles() -> None:
    strategies = load_strategies()
    factual = strategies.for_query_type(QueryType.FACTUAL)  # max_k=5
    analytical = strategies.for_query_type(QueryType.ANALYTICAL)  # min_k=4, max_k=10

    # 8 strong chunks: factual caps at 5, analytical keeps all 8
    strong_chunks = [_make_chunk(f"s{i}", rerank_score=0.92 - i * 0.02) for i in range(8)]
    sel_f, dec_f = select_dynamic_top_k(strong_chunks, factual)
    sel_a, dec_a = select_dynamic_top_k(strong_chunks, analytical)

    assert len(sel_f) == 5
    assert dec_f.cutoff_reason == "max_k_cap"
    assert len(sel_a) == 8
    assert dec_a.cutoff_reason == "all_candidates_retained"

    # Analytical early gap rescued to min_k=4 ONLY from chunks passing tau_min=0.30
    multi_aspect = [
        _make_chunk("a1", rerank_score=0.95),
        _make_chunk("a2", rerank_score=0.52),  # gap 0.43 > 0.35
        _make_chunk("a3", rerank_score=0.49),
        _make_chunk("a4", rerank_score=0.45),
        _make_chunk("a5", rerank_score=0.42),
        _make_chunk("a6", rerank_score=0.20),  # below tau_min=0.30
    ]
    sel_rescued, dec_rescued = select_dynamic_top_k(multi_aspect, analytical)
    assert [c.chunk_id for c in sel_rescued] == ["a1", "a2", "a3", "a4"]
    assert dec_rescued.cutoff_reason == "min_k_floor"


def test_dynamic_k_enforces_token_budget_while_keeping_top1() -> None:
    tight_strategy = StrategyProfile(
        needs_retrieval=True,
        rerank_min_k=2,
        rerank_max_k=5,
        tau_min=0.30,
        alpha_relative=0.50,
        delta_gap=0.35,
        max_context_tokens=500,
    )
    chunks = [
        _make_chunk("t1", rerank_score=0.92, token_count=300),
        _make_chunk("t2", rerank_score=0.88, token_count=180),
        _make_chunk("t3", rerank_score=0.85, token_count=150),
    ]
    selected, decision = select_dynamic_top_k(chunks, tight_strategy)
    assert [c.chunk_id for c in selected] == ["t1", "t2"]
    assert decision.cutoff_reason == "token_budget"
    assert decision.total_context_tokens == 480

    # Even if top-1 alone exceeds budget, never drop the single best valid chunk
    oversized = [_make_chunk("huge", rerank_score=0.95, token_count=900)]
    sel_huge, dec_huge = select_dynamic_top_k(oversized, tight_strategy)
    assert [c.chunk_id for c in sel_huge] == ["huge"]
    assert dec_huge.total_context_tokens == 900


def test_assign_citation_ids_does_not_mutate_originals() -> None:
    originals = [_make_chunk("c1", rerank_score=0.9), _make_chunk("c2", rerank_score=0.8)]
    labeled = assign_citation_ids(originals, prefix="C")
    assert [c.citation_id for c in labeled] == ["C1", "C2"]
    assert [c.citation_id for c in originals] == ["", ""]
