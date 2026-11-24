"""$0 offline tests for Hierarchical Parent-Child (Small-to-Big) chunking, expansion, API & eval."""

from __future__ import annotations

import json
import math
import re
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from adaptiverag.api.dependencies import AppContainer
from adaptiverag.api.main import create_app
from adaptiverag.config import Settings, load_strategies
from adaptiverag.graph import AblationConfig, AdaptiveRAGWorkflow, GraphRuntime
from adaptiverag.ingestion import (
    ChunkingConfig,
    LoadedDocument,
    chunk_document,
    chunk_documents,
    load_directory,
)
from adaptiverag.llm import CostTracker, ModelBundle
from adaptiverag.retrieval import (
    QdrantHybridStore,
    SparseVectorData,
    expand_to_parent_chunks,
)
from adaptiverag.schemas import DynamicKDecision, EvidenceChunk
from eval.datasets import load_golden_jsonl
from eval.metrics import compute_fact_coverage, render_markdown_report
from eval.run_benchmark import (
    PARENT_CHILD_COMPARE_CONFIGS,
    run_parent_child_ablation_benchmark,
)

# ---------------------------------------------------------------------------
# Deterministic $0 test doubles
# ---------------------------------------------------------------------------

_VOCAB = [
    "qdrant",
    "quantization",
    "scalar",
    "binary",
    "product",
    "oversampling",
    "rescore",
    "hnsw",
    "ef_construct",
    "rrf",
    "dbsf",
    "bm25",
    "langgraph",
    "vllm",
]


def _tokenize(text: str) -> list[str]:
    norm = (
        text.lower()
        .replace("ü", "u")
        .replace("ö", "o")
        .replace("ğ", "g")
        .replace("ş", "s")
        .replace("ı", "i")
        .replace("ç", "c")
    )
    return re.findall(r"[a-z0-9_]+", norm)


class _StubDenseEmbedder:
    @property
    def model_name(self) -> str:
        return "stub-dense-pc"

    @property
    def dimension(self) -> int:
        return len(_VOCAB)

    def _encode(self, text: str) -> list[float]:
        tokens = _tokenize(text)
        vec = [float(tokens.count(term)) for term in _VOCAB]
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return [1.0 / math.sqrt(len(_VOCAB))] * len(_VOCAB)
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text)

    async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


class _StubSparseEmbedder:
    @property
    def model_name(self) -> str:
        return "stub-bm25-pc"

    def _encode(self, text: str) -> SparseVectorData:
        counts: dict[int, float] = {}
        for tok in _tokenize(text):
            idx = abs(hash(tok)) % 10000
            counts[idx] = counts.get(idx, 0.0) + 1.0
        indices = sorted(counts.keys())
        return SparseVectorData(indices=indices, values=[counts[i] for i in indices])

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVectorData]:
        return [self._encode(t) for t in texts]

    def embed_query(self, text: str) -> SparseVectorData:
        return self._encode(text)

    async def aembed_documents(self, texts: Sequence[str]) -> list[SparseVectorData]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> SparseVectorData:
        return self.embed_query(text)


class _LexicalStubReranker:
    @property
    def model_name(self) -> str:
        return "stub-reranker-pc"

    def rerank(
        self,
        query: str,
        chunks: Sequence[EvidenceChunk],
        *,
        top_n: int | None = None,
    ) -> list[EvidenceChunk]:
        q_tokens = set(_tokenize(query))
        scored: list[EvidenceChunk] = []
        for idx, chunk in enumerate(chunks):
            c_tokens = set(_tokenize(f"{chunk.section_path} {chunk.text}"))
            overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))
            score = round(min(0.98, max(0.15, 0.30 + overlap * 0.65 - idx * 0.01)), 4)
            scored.append(
                chunk.model_copy(
                    update={"rerank_raw_score": score * 4.0 - 1.5, "rerank_score": score}
                )
            )
        scored.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
        return scored[:top_n] if top_n is not None else scored

    async def arerank(
        self,
        query: str,
        chunks: Sequence[EvidenceChunk],
        *,
        top_n: int | None = None,
    ) -> list[EvidenceChunk]:
        return self.rerank(query, chunks, top_n=top_n)


class _ScriptedLLM(BaseChatModel):
    responses: list[str] = Field(default_factory=list)
    call_index: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-parent-child-test"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        text = self.responses[min(self.call_index, len(self.responses) - 1)]
        self.call_index += 1
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=text,
                        usage_metadata={
                            "input_tokens": 25,
                            "output_tokens": 15,
                            "total_tokens": 40,
                        },
                    )
                )
            ]
        )


class _NoopFallback:
    enabled = False
    provider_names: ClassVar[list[str]] = []

    async def asearch(
        self,
        query: str,
        *,
        max_results: int = 4,
        start_citation_index: int = 1,
    ) -> list[EvidenceChunk]:
        del query, max_results, start_citation_index
        return []


# ---------------------------------------------------------------------------
# 1. Config & Hierarchical Parent-Child Chunker Tests
# ---------------------------------------------------------------------------


def test_parent_child_config_and_settings_validation() -> None:
    with pytest.raises(ValueError, match="child_overlap_tokens"):
        ChunkingConfig(parent_child_enabled=True, child_max_tokens=128, child_overlap_tokens=128)

    with pytest.raises(ValueError, match="child_max_tokens"):
        ChunkingConfig(parent_child_enabled=True, child_max_tokens=1000, parent_max_tokens=1000)

    with pytest.raises(ValueError, match="min_chunk_tokens"):
        ChunkingConfig(
            parent_child_enabled=True,
            min_chunk_tokens=200,
            child_max_tokens=128,
            child_overlap_tokens=16,
            parent_max_tokens=512,
        )

    with pytest.raises(ValueError, match="child_max_tokens"):
        Settings(_env_file=None, child_max_tokens=1000, parent_max_tokens=800)  # type: ignore[call-arg]


def test_hierarchical_parent_child_groups_h3_subsections_into_cohesive_parent() -> None:
    md = """# Qdrant Quantization Guide

Overview of quantization methods in vector databases.

## Vector Quantization Strategies

Storing 768d float32 vectors consumes 3,072 bytes per vector before graph overhead.

### Scalar Quantization (int8)

Scalar quantization achieves 4x memory compression with oversampling 1.5 to 2.0 and rescore=True.

### Binary Quantization (1-bit)

Binary quantization achieves 32x memory compression with oversampling 3.0 to 4.0 and rescore=True.

### Product Quantization (PQ)

Product quantization splits vectors into sub-vectors for 8x to 64x compression.
"""
    doc = LoadedDocument(
        doc_id="qdrant_quant",
        title="Qdrant Quantization Guide",
        content=md,
        source_path="qdrant_quant.md",
        doc_type="markdown",
        language="en",
    )

    cfg = ChunkingConfig(
        parent_child_enabled=True,
        parent_max_tokens=1000,
        child_max_tokens=120,
        child_overlap_tokens=20,
        min_chunk_tokens=10,
    )
    chunks = chunk_document(doc, cfg)
    quant_children = [
        c
        for c in chunks
        if c.parent_section_path == "Qdrant Quantization Guide > Vector Quantization Strategies"
    ]

    # Intro + 3 H3 subsections should share ONE cohesive H2 parent passage!
    assert len(quant_children) == 4
    parent_ids = {c.parent_id for c in quant_children}
    assert len(parent_ids) == 1
    shared_parent_id = next(iter(parent_ids))
    assert shared_parent_id is not None
    assert uuid.UUID(shared_parent_id).version == 5

    # Children retain fine-grained H3 section paths while parent_text preserves all H3 subsections
    child_paths = [c.section_path for c in quant_children]
    assert (
        "Qdrant Quantization Guide > Vector Quantization Strategies > Scalar Quantization (int8)"
        in child_paths
    )
    assert (
        "Qdrant Quantization Guide > Vector Quantization Strategies > Binary Quantization (1-bit)"
        in child_paths
    )

    parent_text = quant_children[0].parent_text or ""
    assert "### Scalar Quantization (int8)" in parent_text
    assert "### Binary Quantization (1-bit)" in parent_text
    assert "### Product Quantization (PQ)" in parent_text
    for child in quant_children:
        assert child.parent_text == parent_text
        assert child.child_count == 4
        assert child.metadata["chunk_strategy"] == "parent_child"
        assert (child.parent_token_count or 0) >= child.token_count


def test_long_section_splits_into_multiple_parents_preserving_tables_and_code() -> None:
    rows = "\n".join(
        f"| `param_{i}` | `{i * 10}` | Detailed production explanation for HNSW tuning parameter {i} |"
        for i in range(1, 30)
    )
    md = f"""# Long Spec

## Detailed Tuning Reference

Introductory paragraph discussing production deployment constraints and latency budgets.

| Parameter | Default | Notes |
| :--- | :--- | :--- |
{rows}

```python
# Keep code fences atomic
cfg = {{"m": 16, "ef_construct": 128}}
```
"""
    doc = LoadedDocument(
        doc_id="long_spec",
        title="Long Spec",
        content=md,
        source_path="long_spec.md",
        doc_type="markdown",
        language="en",
    )

    cfg = ChunkingConfig(
        parent_child_enabled=True,
        parent_max_tokens=420,
        child_max_tokens=160,
        child_overlap_tokens=20,
        min_chunk_tokens=10,
    )
    chunks = chunk_document(doc, cfg)
    unique_parents = {c.parent_id for c in chunks}
    assert len(unique_parents) >= 2
    assert len(chunks) > len(unique_parents)

    table_children = [c for c in chunks if c.metadata.get("has_table")]
    assert len(table_children) >= 2
    for tc in table_children:
        assert "| Parameter | Default | Notes |" in tc.text

    code_children = [c for c in chunks if c.metadata.get("has_code_block")]
    assert len(code_children) == 1
    assert code_children[0].text.count("```") == 2


def test_parent_child_chunk_ids_are_deterministic_across_curated_corpus() -> None:
    docs = load_directory(Path("data/raw"))
    cfg = ChunkingConfig(
        parent_child_enabled=True,
        parent_max_tokens=1000,
        child_max_tokens=256,
        child_overlap_tokens=40,
    )
    run_a = chunk_documents(docs, cfg)
    run_b = chunk_documents(docs, cfg)

    assert len(run_a) > 62  # finer child chunks than flat 512-tok baseline (62)
    assert [c.chunk_id for c in run_a] == [c.chunk_id for c in run_b]
    assert [c.parent_id for c in run_a] == [c.parent_id for c in run_b]
    assert all(c.parent_id and c.parent_text for c in run_a)
    # Child chunk IDs must be strictly disjoint from flat chunk IDs
    flat_ids = {c.chunk_id for c in chunk_documents(docs)}
    assert flat_ids.isdisjoint({c.chunk_id for c in run_a})


# ---------------------------------------------------------------------------
# 2. Small-to-Big Expansion & Token Budget Guard Tests
# ---------------------------------------------------------------------------


def test_expand_to_parent_deduplicates_siblings_preserves_scores_and_hides_parent_text_in_json() -> (
    None
):
    parent_text = (
        "Full quantization section covering Scalar Quantization (4x) and Binary Quantization (32x)."
    )
    c1 = EvidenceChunk(
        chunk_id="child-1",
        doc_id="doc_1",
        title="Qdrant",
        section_path="Qdrant > Quantization > Scalar",
        text="Scalar Quantization achieves 4x compression.",
        rerank_score=0.94,
        rerank_raw_score=2.8,
        fusion_score=0.032,
        matched_sub_queries=["scalar quantization"],
        token_count=12,
        parent_id="parent-q",
        parent_text=parent_text,
        parent_section_path="Qdrant > Quantization",
        parent_token_count=30,
    )
    c2 = EvidenceChunk(
        chunk_id="child-2",
        doc_id="doc_1",
        title="Qdrant",
        section_path="Qdrant > Quantization > Binary",
        text="Binary Quantization achieves 32x compression.",
        rerank_score=0.81,
        rerank_raw_score=1.4,
        fusion_score=0.041,
        matched_sub_queries=["binary quantization"],
        token_count=12,
        parent_id="parent-q",
        parent_text=parent_text,
        parent_section_path="Qdrant > Quantization",
        parent_token_count=30,
    )
    legacy = EvidenceChunk(
        chunk_id="flat-chunk",
        doc_id="doc_2",
        title="Legacy Doc",
        section_path="Legacy",
        text="Flat chunk without parent payload.",
        rerank_score=0.72,
        token_count=15,
    )
    decision = DynamicKDecision(
        candidate_count=10,
        selected_count=3,
        cutoff_reason="relative_drop",
        top_score=0.94,
        cutoff_score=0.72,
        total_context_tokens=39,
    )

    expanded, updated = expand_to_parent_chunks(
        [c1, c2, legacy],
        max_context_tokens=500,
        decision=decision,
    )

    assert len(expanded) == 2
    assert expanded[0].chunk_id == "parent-q"
    assert expanded[0].expanded_to_parent is True
    assert expanded[0].text == parent_text
    assert expanded[0].section_path == "Qdrant > Quantization"
    assert expanded[0].rerank_score == pytest.approx(0.94)
    assert expanded[0].fusion_score == pytest.approx(0.041)
    assert expanded[0].matched_sub_queries == ["scalar quantization", "binary quantization"]
    assert expanded[0].matched_child_ids == ["child-1", "child-2"]
    assert expanded[0].token_count == 30
    assert "parent_text" not in expanded[0].model_dump()

    # Legacy flat chunk passes through cleanly at rank #2
    assert expanded[1].chunk_id == "flat-chunk"
    assert expanded[1].expanded_to_parent is False

    assert updated is not None
    assert updated.parent_expansion_applied is True
    assert updated.collapsed_child_count == 3
    assert updated.expanded_parent_count == 1
    assert updated.selected_count == 2
    assert updated.total_context_tokens == 45


def test_expand_to_parent_noop_for_legacy_flat_chunks_and_respects_budget_fallback() -> None:
    flat_chunks = [
        EvidenceChunk(
            chunk_id="f1", doc_id="d1", text="Flat text 1.", rerank_score=0.9, token_count=50
        ),
        EvidenceChunk(
            chunk_id="f2", doc_id="d2", text="Flat text 2.", rerank_score=0.8, token_count=60
        ),
    ]
    base_decision = DynamicKDecision(
        candidate_count=2,
        selected_count=2,
        cutoff_reason="all_candidates_retained",
        top_score=0.9,
        cutoff_score=0.8,
        total_context_tokens=110,
    )
    out_flat, dec_flat = expand_to_parent_chunks(
        flat_chunks, max_context_tokens=500, decision=base_decision
    )
    assert out_flat == flat_chunks
    assert dec_flat is base_decision

    # Budget test: Parent #1 (800 tok) always kept; Parent #2 (900 tok) overflows 1050 budget,
    # so compact child #2 (180 tok) is retained to preserve multi-doc recall; #3 (200 tok) clips!
    p1 = EvidenceChunk(
        chunk_id="c1",
        doc_id="d1",
        text="Child 1",
        rerank_score=0.95,
        token_count=150,
        parent_id="p1",
        parent_text="Full Parent 1",
        parent_token_count=800,
    )
    p2 = EvidenceChunk(
        chunk_id="c2",
        doc_id="d2",
        text="Compact Child 2",
        rerank_score=0.85,
        token_count=180,
        parent_id="p2",
        parent_text="Oversized Parent 2",
        parent_token_count=900,
    )
    p3 = EvidenceChunk(
        chunk_id="c3",
        doc_id="d3",
        text="Child 3 does not fit",
        rerank_score=0.75,
        token_count=200,
        parent_id="p3",
        parent_text="Parent 3",
        parent_token_count=500,
    )

    out_budget, dec_budget = expand_to_parent_chunks(
        [p1, p2, p3],
        max_context_tokens=1050,
        decision=DynamicKDecision(
            candidate_count=5,
            selected_count=3,
            cutoff_reason="all_candidates_retained",
            top_score=0.95,
            cutoff_score=0.75,
            total_context_tokens=530,
        ),
    )

    assert len(out_budget) == 2
    assert out_budget[0].chunk_id == "p1" and out_budget[0].expanded_to_parent is True
    assert out_budget[1].chunk_id == "c2" and out_budget[1].expanded_to_parent is False
    assert out_budget[1].metadata["parent_expansion_skipped"] == "token_budget"
    assert dec_budget is not None
    assert dec_budget.cutoff_reason == "parent_token_budget"
    assert dec_budget.total_context_tokens == 980
    assert "c3" in dec_budget.dropped_chunk_ids


# ---------------------------------------------------------------------------
# 3. End-to-End Qdrant + LangGraph + FastAPI Integration Tests ($0)
# ---------------------------------------------------------------------------


async def test_qdrant_and_workflow_parent_child_end_to_end() -> None:
    md = """# Qdrant Quantization

## Vector Quantization & Memory Optimization

Overview of quantization memory tradeoffs.

### Scalar Quantization (int8)

Scalar quantization gives 4x compression with oversampling 1.5 to 2.0.

### Binary Quantization (1-bit)

Binary quantization gives 32x compression with oversampling 3.0 to 4.0.
"""
    doc = LoadedDocument(
        doc_id="01_qdrant_quant",
        title="Qdrant Quantization",
        content=md,
        source_path="01_qdrant_quant.md",
        doc_type="markdown",
        language="en",
    )
    chunks = chunk_document(
        doc,
        ChunkingConfig(
            parent_child_enabled=True,
            parent_max_tokens=1000,
            child_max_tokens=100,
            child_overlap_tokens=16,
            min_chunk_tokens=10,
        ),
    )

    settings = Settings(_env_file=None, enable_parent_child_retrieval=False)  # type: ignore[call-arg]
    store = QdrantHybridStore.from_settings(
        settings,
        collection_name="test_pc_workflow",
        dense_embedder=_StubDenseEmbedder(),
        sparse_embedder=_StubSparseEmbedder(),
        in_memory=True,
    )
    await store.ensure_collection(recreate=True)
    await store.upsert_chunks(chunks)

    # Verify Qdrant payload rehydration
    hits = await store.search_single("Scalar Quantization compression", fusion_limit=5)
    assert len(hits) >= 2
    assert all(h.parent_id and h.parent_text for h in hits)

    router_json = json.dumps(
        {
            "query_type": "factual",
            "needs_retrieval": True,
            "standalone_query": "Compare Scalar Quantization and Binary Quantization compression",
            "language": "en",
            "requires_fresh_web_info": False,
            "reasoning": "Factual quantization lookup.",
        }
    )
    answer_text = "Scalar Quantization achieves 4x compression while Binary Quantization achieves 32x compression [C1]."

    fast_llm = _ScriptedLLM(responses=[router_json])
    gen_llm = _ScriptedLLM(responses=[answer_text])
    runtime = GraphRuntime(
        settings=settings,
        strategies=load_strategies(),
        models=ModelBundle(
            fast_model=fast_llm,
            generation_model=gen_llm,
            fast_model_name="stub-fast",
            generation_model_name="stub-gen",
        ),
        store=store,
        reranker=_LexicalStubReranker(),
        fallback_retriever=_NoopFallback(),
        cost_tracker=CostTracker(max_cost_usd=1.0),
    )
    workflow = AdaptiveRAGWorkflow(runtime)

    # Run WITH parent-child expansion enabled via ablation override
    ans_pc = await workflow.arun(
        "Compare Scalar Quantization and Binary Quantization compression",
        ablation=AblationConfig(
            enable_parent_child=True,
            enable_confidence_gating=False,
            enable_citation_validation=False,
        ),
    )
    assert ans_pc.dynamic_k_decision is not None
    assert ans_pc.dynamic_k_decision.parent_expansion_applied is True
    assert ans_pc.dynamic_k_decision.expanded_parent_count >= 1
    assert ans_pc.evidence_chunks[0].citation_id == "C1"
    assert ans_pc.evidence_chunks[0].expanded_to_parent is True
    assert "4x compression" in ans_pc.evidence_chunks[0].text
    assert "32x compression" in ans_pc.evidence_chunks[0].text

    # Run WITH parent-child expansion explicitly disabled -> stays unexpanded child chunks
    fast_llm_off = _ScriptedLLM(responses=[router_json])
    gen_llm_off = _ScriptedLLM(responses=[answer_text])
    runtime.models = ModelBundle(
        fast_model=fast_llm_off,
        generation_model=gen_llm_off,
        fast_model_name="stub-fast",
        generation_model_name="stub-gen",
    )
    ans_flat = await workflow.arun(
        "Compare Scalar Quantization and Binary Quantization compression",
        ablation=AblationConfig(
            enable_parent_child=False,
            enable_confidence_gating=False,
            enable_citation_validation=False,
        ),
    )
    assert ans_flat.dynamic_k_decision is not None
    assert ans_flat.dynamic_k_decision.parent_expansion_applied is False
    assert all(not c.expanded_to_parent for c in ans_flat.evidence_chunks)

    await store.close()


async def test_api_ingest_and_health_expose_parent_child_controls() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        enable_parent_child_retrieval=True,
        parent_max_tokens=900,
        child_max_tokens=200,
        child_overlap_tokens=30,
        firecrawl_local_url="",
        langfuse_enabled=False,
    )
    store = QdrantHybridStore.from_settings(
        settings,
        collection_name="test_api_pc",
        dense_embedder=_StubDenseEmbedder(),
        sparse_embedder=_StubSparseEmbedder(),
        in_memory=True,
    )
    router_json = json.dumps(
        {
            "query_type": "factual",
            "needs_retrieval": True,
            "standalone_query": "Compare RRF and DBSF fusion methods",
            "language": "en",
            "requires_fresh_web_info": False,
            "reasoning": "Factual comparison.",
        }
    )
    workflow = AdaptiveRAGWorkflow(
        GraphRuntime(
            settings=settings,
            strategies=load_strategies(),
            models=ModelBundle(
                fast_model=_ScriptedLLM(responses=[router_json]),
                generation_model=_ScriptedLLM(
                    responses=["RRF uses k=60 while DBSF normalizes mean and std [C1]."]
                ),
                fast_model_name="stub-fast",
                generation_model_name="stub-gen",
            ),
            store=store,
            reranker=_LexicalStubReranker(),
            fallback_retriever=_NoopFallback(),
            cost_tracker=CostTracker(max_cost_usd=1.0),
        )
    )
    container = AppContainer(
        settings=settings,
        strategies=load_strategies(),
        store=store,
        fallback_retriever=_NoopFallback(),
        workflow=workflow,
    )
    app = create_app(container=container)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        health = await client.get("/v1/health")
        assert health.status_code == 200
        pc_health = health.json()["parent_child_retrieval"]
        assert pc_health["enabled"] is True
        assert pc_health["parent_max_tokens"] == 900
        assert pc_health["child_max_tokens"] == 200

        ingest_resp = await client.post(
            "/v1/ingest",
            json={
                "documents": [
                    {
                        "doc_id": "inline_pc_doc",
                        "title": "Hybrid Fusion",
                        "content": (
                            "# Hybrid Fusion\n\n## Fusion Methods\n\nIntro.\n\n"
                            "### RRF\n\nReciprocal Rank Fusion uses k=60.\n\n"
                            "### DBSF\n\nDistribution-Based Score Fusion normalizes mean and std.\n"
                        ),
                    }
                ],
                "parent_child": True,
                "parent_max_tokens": 800,
                "child_max_tokens": 120,
                "child_overlap_tokens": 20,
            },
        )
        assert ingest_resp.status_code == 200
        body = ingest_resp.json()
        assert body["parent_child_enabled"] is True
        assert body["unique_parents_created"] >= 1
        assert body["chunks_created"] >= 3

        query_resp = await client.post(
            "/v1/query",
            json={
                "query": "Compare RRF and DBSF fusion methods",
                "ablation": {
                    "enable_parent_child": True,
                    "enable_confidence_gating": False,
                    "enable_citation_validation": False,
                },
            },
        )
        assert query_resp.status_code == 200
        q_body = query_resp.json()
        assert q_body["dynamic_k_decision"]["parent_expansion_applied"] is True
        assert q_body["evidence_chunks"][0]["expanded_to_parent"] is True
        assert "Reciprocal Rank Fusion" in q_body["evidence_chunks"][0]["text"]
        assert "Distribution-Based Score Fusion" in q_body["evidence_chunks"][0]["text"]
        assert "parent_text" not in q_body["evidence_chunks"][0]

    await store.close()


# ---------------------------------------------------------------------------
# 4. Fact Coverage Metric & Parent-Child Offline Ablation Runner Tests
# ---------------------------------------------------------------------------


def test_compute_fact_coverage_handles_en_tr_and_strict_numbers() -> None:
    passages = [
        "Scalar Quantization achieves 4x memory compression with oversampling 1.5 to 2.0.",
        "Türkçenin sondan eklemeli yapısı vektörlerin kök-ek ayrımını zorlaştırır.",
        "vLLM default block_size is 160 in a fictional test.",
    ]
    assert compute_fact_coverage(passages, []) is None
    assert compute_fact_coverage([], ["Scalar Quantization 4x"]) == 0.0

    # First two facts match (EN technical + TR inflected); numeric '16' must NOT fuzzy-match '160'
    cov = compute_fact_coverage(
        passages,
        [
            "Scalar Quantization 4x",
            "eklemeli yapı vektör ayrımını",
            "block_size=16 tokens",
        ],
    )
    assert cov == pytest.approx(2 / 3, abs=1e-5)


async def test_run_parent_child_ablation_benchmark_offline_in_memory() -> None:
    docs = load_directory(Path("data/raw"))
    queries = load_golden_jsonl(Path("data/eval/golden_set.jsonl"))[:8]
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    dense = _StubDenseEmbedder()
    sparse = _StubSparseEmbedder()

    flat_store = QdrantHybridStore.from_settings(
        settings,
        collection_name="bench_flat",
        dense_embedder=dense,
        sparse_embedder=sparse,
        in_memory=True,
    )
    pc_store = QdrantHybridStore.from_settings(
        settings,
        collection_name="bench_pc",
        dense_embedder=dense,
        sparse_embedder=sparse,
        in_memory=True,
    )
    await flat_store.ensure_collection(recreate=True)
    await flat_store.upsert_chunks(chunk_documents(docs))

    await pc_store.ensure_collection(recreate=True)
    await pc_store.upsert_chunks(
        chunk_documents(
            docs,
            ChunkingConfig(
                parent_child_enabled=True,
                parent_max_tokens=1000,
                child_max_tokens=256,
                child_overlap_tokens=40,
            ),
        )
    )

    report = await run_parent_child_ablation_benchmark(
        queries,
        flat_store=flat_store,
        parent_child_store=pc_store,
        reranker=_LexicalStubReranker(),
        strategies=load_strategies(),
        fixed_top_k=5,
    )

    assert tuple(report.configs) == PARENT_CHILD_COMPARE_CONFIGS
    assert len(report.summaries) == 4
    assert all(s.mean_fact_coverage is not None for s in report.summaries)
    md = render_markdown_report(report)
    assert "Fact Cov" in md
    assert "parent_child_adaptive_dynamic_k" in md

    await flat_store.close()
    await pc_store.close()


async def test_parent_passage_long_tail_preserved_and_store_reupsert_cleans_stale_ids() -> None:
    from adaptiverag.graph.prompts import format_evidence_context
    from adaptiverag.verification.citation_validator import _build_entailment_user_prompt

    # 1. Verify ~4,000-char (1000-token) parent passage tail facts are preserved in both
    #    grounded generation and Tier-2 citation entailment prompts (no 1400/2400-char clipping).
    long_body = ("Detailed HNSW quantization background prose. " * 75) + "TAIL_FACT_MARKER_4096"
    assert len(long_body) > 3300
    parent_chunk = EvidenceChunk(
        citation_id="C1",
        chunk_id="parent-long-1",
        doc_id="qdrant_quant",
        title="Qdrant Quantization",
        section_path="Qdrant Quantization > Vector Quantization",
        text=long_body,
        token_count=920,
        parent_id="parent-long-1",
        expanded_to_parent=True,
    )
    gen_ctx = format_evidence_context([parent_chunk])
    entail_ctx = _build_entailment_user_prompt("Answer [C1].", [parent_chunk])
    assert "TAIL_FACT_MARKER_4096" in gen_ctx
    assert "TAIL_FACT_MARKER_4096" in entail_ctx

    # 2. Verify Qdrant upsert_chunks transitions cleanly from flat IDs to parent-child IDs
    #    without leaving stale flat points, and list_documents reports non-overlapping parent tokens.
    doc = LoadedDocument(
        doc_id="transition_doc",
        title="Transition Guide",
        content=(
            "# Transition Guide\n\n## Section A\n\n"
            "### Sub 1\n\nAlpha scalar quantization details for Qdrant hybrid search.\n\n"
            "### Sub 2\n\nBeta binary quantization details with oversampling and rescore.\n"
        ),
        source_path="transition_doc.md",
        doc_type="markdown",
        language="en",
    )
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    store = QdrantHybridStore.from_settings(
        settings,
        collection_name="transition_test",
        dense_embedder=_StubDenseEmbedder(),
        sparse_embedder=_StubSparseEmbedder(),
        in_memory=True,
    )
    await store.ensure_collection(recreate=True)
    flat_chunks = chunk_documents([doc])
    await store.upsert_chunks(flat_chunks)
    flat_docs = await store.list_documents()
    assert len(flat_docs) == 1
    assert flat_docs[0]["chunk_strategy"] == "flat"

    pc_chunks = chunk_documents(
        [doc],
        ChunkingConfig(
            parent_child_enabled=True,
            parent_max_tokens=1000,
            child_max_tokens=80,
            child_overlap_tokens=20,
            min_chunk_tokens=5,
        ),
    )
    assert len(pc_chunks) >= 2
    await store.upsert_chunks(pc_chunks)

    # Stale flat IDs must be cleaned up after parent-child upsert completes
    assert await store.count_points() == len(pc_chunks)
    pc_docs = await store.list_documents()
    assert len(pc_docs) == 1
    assert pc_docs[0]["chunk_strategy"] == "parent_child"
    assert pc_docs[0]["chunk_count"] == len(pc_chunks)
    expected_parent_tokens = sum(
        {c.parent_id: c.parent_token_count or c.token_count for c in pc_chunks}.values()
    )
    assert pc_docs[0]["total_tokens"] == expected_parent_tokens
    await store.close()
