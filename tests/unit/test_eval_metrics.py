"""$0 unit tests for golden dataset integrity, dataset adapters, IR metrics, RAGAS bridge, and 4-way benchmark runner."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from adaptiverag.config import Settings, load_strategies
from adaptiverag.graph import AdaptiveRAGWorkflow, GraphRuntime
from adaptiverag.ingestion.chunker import chunk_documents
from adaptiverag.ingestion.loader import load_directory
from adaptiverag.llm import CostTracker, ModelBundle, SQLiteLLMCache
from adaptiverag.retrieval.embeddings import SparseVectorData
from adaptiverag.retrieval.qdrant_store import QdrantHybridStore
from adaptiverag.schemas import EvidenceChunk, FinalAnswer, QueryType, SourceType
from eval.datasets import (
    adapt_hotpotqa_row,
    adapt_ragbench_row,
    filter_queries,
    load_external_jsonl,
    load_golden_jsonl,
)
from eval.metrics import (
    compute_binary_precision_recall,
    compute_ir_scores,
    deduplicate_doc_ids,
    mrr_at_k,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    render_markdown_report,
)
from eval.ragas_runner import FastEmbedRagasEmbeddings, build_ragas_samples
from eval.run_benchmark import (
    LIVE_CONFIG_NAMES,
    OFFLINE_IR_CONFIGS,
    run_live_ablation_benchmark,
    run_offline_ir_benchmark,
    save_benchmark_artifacts,
)
from eval.schemas import EvalStratum, GoldenQuery

_VOCAB = [
    "qdrant",
    "hnsw",
    "ef_construct",
    "hybrid",
    "bm25",
    "rrf",
    "langgraph",
    "vllm",
    "pagedattention",
    "fastapi",
    "sse",
    "postgresql",
    "mvcc",
    "pgvector",
    "crag",
    "ragas",
    "turkce",
    "finops",
    "chunking",
    "dinamik",
]


def _tokenize(text: str) -> list[str]:
    normalized = (
        text.lower()
        .replace("ü", "u")
        .replace("ö", "o")
        .replace("ğ", "g")
        .replace("ş", "s")
        .replace("ı", "i")
        .replace("ç", "c")
    )
    return re.findall(r"[a-z0-9_]+", normalized)


class _StubDenseEmbedder:
    @property
    def model_name(self) -> str:
        return "stub-dense-20d"

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
        return "stub-bm25"

    def _encode(self, text: str) -> SparseVectorData:
        tokens = _tokenize(text)
        counts: dict[int, float] = {}
        for tok in tokens:
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
    """Deterministic $0 reranker assigning calibrated [0, 1] scores based on term overlap."""

    @property
    def model_name(self) -> str:
        return "stub-lexical-reranker"

    def rerank(
        self,
        query: str,
        chunks: Sequence[EvidenceChunk],
        *,
        top_n: int | None = None,
    ) -> list[EvidenceChunk]:
        q_tokens = {t for t in _tokenize(query) if len(t) > 2}
        scored: list[EvidenceChunk] = []
        for chunk in chunks:
            c_tokens = set(_tokenize(f"{chunk.title} {chunk.section_path} {chunk.text}"))
            overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))
            # Fictional/unanswerable keywords stay below tau_min (0.35)
            calibrated = round(min(0.96, overlap * 1.4), 4)
            scored.append(
                chunk.model_copy(
                    update={"rerank_raw_score": calibrated * 4.0 - 1.5, "rerank_score": calibrated}
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


class _StubWebFallback:
    @property
    def provider_names(self) -> list[str]:
        return ["stub-web"]

    async def asearch(
        self,
        query: str,
        *,
        max_results: int = 3,
        start_citation_index: int = 1,
    ) -> list[EvidenceChunk]:
        return [
            EvidenceChunk(
                citation_id=f"W{start_citation_index}",
                chunk_id="web_stub_1",
                text=f"Fresh web evidence answering: {query}",
                source_type=SourceType.WEB,
                doc_id="web_live_result",
                title="Web Live Result",
                url="https://example.com/live",
                rerank_score=0.82,
                token_count=16,
            )
        ][:max_results]


class _BenchmarkFakeLLM(BaseChatModel):
    """Deterministic $0 fake chat model handling all AdaptiveRAG prompt roles."""

    model_name: str = Field(default="qwen/qwen3-30b-a3b-instruct-2507")
    prompt_tokens_per_call: int = Field(default=100)
    completion_tokens_per_call: int = Field(default=40)

    @property
    def _llm_type(self) -> str:
        return "benchmark-fake"

    def _reply(self, messages: list[BaseMessage]) -> str:
        sys_text = (
            "\n".join(str(m.content) for m in messages[:-1]).lower() if len(messages) > 1 else ""
        )
        user_text = str(messages[-1].content) if messages else ""
        user_lower = user_text.lower()

        if "query router for adaptiverag" in sys_text:
            # Extract the actual query line after "Current User Query:"
            marker = "current user query:"
            raw_q = (
                user_text[user_lower.rfind(marker) + len(marker) :].strip()
                if marker in user_lower
                else user_text.strip()
            )
            q_lower = raw_q.lower()
            if any(w in q_lower for w in ("hi there", "merhaba", "thanks", "teşekkür")):
                qtype, needs = "conversational", False
            elif any(w in q_lower for w in ("compare", "kıyasla", "trade-off", "how do ")):
                qtype, needs = "analytical", True
            else:
                qtype, needs = "factual", True
            lang = "tr" if any(c in raw_q for c in "ğşçıöüĞŞÇİÖÜ") else "en"
            return json.dumps(
                {
                    "query_type": qtype,
                    "needs_retrieval": needs,
                    "language": lang,
                    "standalone_query": raw_q[:200] or "query",
                    "requires_fresh_web_info": False,
                    "reasoning": "Deterministic benchmark classification.",
                }
            )

        if "query decomposition specialist" in sys_text:
            return json.dumps(
                {
                    "sub_queries": [
                        "Qdrant Filterable HNSW and Quantization",
                        "PostgreSQL pgvector filtered search and hybrid trade-offs",
                    ],
                    "rationale": "Split comparison across both vector engines.",
                }
            )

        if "retrieval sufficiency grader" in sys_text:
            grade = "insufficient" if "quantum_hyper_mesh" in user_lower else "sufficient"
            return json.dumps({"grade": grade, "missing_aspects": [], "reasoning": "Fake CRAG."})

        if "corrective retrieval rewriter" in sys_text:
            return json.dumps(
                {
                    "rewritten_queries": ["Qdrant HNSW ef_construct default value"],
                    "diagnosis": "Broadening lexical terms for higher recall.",
                }
            )

        if "citation entailment auditor" in sys_text:
            cited = ["W1"] if "[W1]" in user_text else ["C1"]
            return json.dumps(
                {
                    "verifications": [
                        {
                            "claim": "Verified grounded technical statement.",
                            "cited_ids": cited,
                            "verdict": "supported",
                            "explanation": "Directly entailed by cited passage.",
                        }
                    ]
                }
            )

        if "[W1]" in user_text:
            return "According to live web search results, this information was retrieved externally [W1]."
        if "[C1]" in user_text:
            return "According to the corpus documentation, ef_construct defaults to 100 and Modifier.IDF enables live IDF [C1]."
        return "Hello! Happy to help with AdaptiveRAG."

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        content = self._reply(messages)
        ai_msg = AIMessage(
            content=content,
            usage_metadata={
                "input_tokens": self.prompt_tokens_per_call,
                "output_tokens": self.completion_tokens_per_call,
                "total_tokens": self.prompt_tokens_per_call + self.completion_tokens_per_call,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])


# ---------------------------------------------------------------------------
# 1. Golden Dataset Integrity & Corpus Alignment Tests
# ---------------------------------------------------------------------------


def test_golden_dataset_integrity_and_corpus_alignment() -> None:
    golden_path = Path("data/eval/golden_set.jsonl")
    corpus_docs = load_directory(Path("data/raw"))
    valid_doc_ids = {doc.doc_id for doc in corpus_docs}
    en_doc_ids = {doc.doc_id for doc in corpus_docs if doc.language == "en"}

    queries = load_golden_jsonl(golden_path)
    assert len(queries) >= 60

    strata_counts: dict[EvalStratum, int] = {}
    lang_counts: dict[str, int] = {}
    referenced_doc_ids: set[str] = set()
    cross_lingual_tr_to_en = 0

    for q in queries:
        strata_counts[q.stratum] = strata_counts.get(q.stratum, 0) + 1
        lang_counts[q.language] = lang_counts.get(q.language, 0) + 1
        assert q.reference_answer.strip()

        for doc_id in q.expected_doc_ids:
            assert doc_id in valid_doc_ids, (
                f"Golden query '{q.id}' references unknown doc_id '{doc_id}'. "
                f"Valid IDs: {sorted(valid_doc_ids)}"
            )
            referenced_doc_ids.add(doc_id)

        if q.language == "tr" and any(d in en_doc_ids for d in q.expected_doc_ids):
            cross_lingual_tr_to_en += 1

        if q.stratum == EvalStratum.CONVERSATIONAL:
            assert q.needs_retrieval is False
            assert q.expected_doc_ids == []
        elif q.stratum == EvalStratum.WEB_FALLBACK:
            assert q.expect_web_fallback is True
            assert q.expect_abstention is False
        elif q.stratum == EvalStratum.UNANSWERABLE:
            assert q.expect_abstention is True
            assert q.expect_web_fallback is False
        else:
            assert len(q.expected_doc_ids) >= 1

    # All 5 strata must be populated
    assert set(strata_counts.keys()) == set(EvalStratum)
    assert strata_counts[EvalStratum.FACTUAL] >= 20
    assert strata_counts[EvalStratum.ANALYTICAL] >= 15
    assert strata_counts[EvalStratum.CONVERSATIONAL] >= 5
    assert strata_counts[EvalStratum.WEB_FALLBACK] >= 5
    assert strata_counts[EvalStratum.UNANSWERABLE] >= 5

    # Multilingual TR + EN + Cross-lingual coverage
    assert lang_counts.get("en", 0) >= 35
    assert lang_counts.get("tr", 0) >= 20
    assert cross_lingual_tr_to_en >= 5

    # Every single corpus document in data/raw/ must be evaluated by the golden set
    assert referenced_doc_ids == valid_doc_ids


# ---------------------------------------------------------------------------
# 2. External JSONL & HuggingFace Adapter Tests ($0 Offline)
# ---------------------------------------------------------------------------


def test_external_jsonl_and_huggingface_adapters(tmp_path: Path) -> None:
    ext_file = tmp_path / "external.jsonl"
    ext_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "qid": "ext-1",
                        "question": "What is HNSW?",
                        "ground_truth": "Hierarchical Navigable Small World graph.",
                        "doc_ids": ["doc_hnsw"],
                    }
                ),
                json.dumps(
                    {
                        "question": "Compare HNSW and IVF.",
                        "answer": "HNSW gives higher recall at higher RAM.",
                        "stratum": "analytical",
                        "relevant_doc_ids": ["doc_hnsw", "doc_ivf"],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_external_jsonl(ext_file)
    assert len(loaded) == 2
    assert loaded[0].id == "ext-1"
    assert loaded[0].expected_doc_ids == ["doc_hnsw"]
    assert loaded[1].stratum == EvalStratum.ANALYTICAL
    assert loaded[1].expected_query_type == QueryType.ANALYTICAL

    filtered = filter_queries(loaded, strata=["analytical"], language="en", limit=1)
    assert len(filtered) == 1
    assert filtered[0].id == "ext_0002"

    # HotpotQA adapter (offline row fixture)
    hotpot_row = {
        "id": "hp_42",
        "question": "Which engine written in Rust supports Filterable HNSW?",
        "answer": "Qdrant",
        "type": "bridge",
        "supporting_facts": {"title": ["Qdrant Engine", "HNSW Graphs"], "sent_id": [0, 1]},
        "context": {
            "title": ["Qdrant Engine", "HNSW Graphs", "Distractor Doc"],
            "sentences": [
                ["Qdrant is written in Rust."],
                ["Filterable HNSW augments graph edges with payload filters."],
                ["Unrelated weather report."],
            ],
        },
    }
    hp_query, hp_docs = adapt_hotpotqa_row(hotpot_row, index=1)
    assert hp_query.id == "hp_42"
    assert hp_query.stratum == EvalStratum.ANALYTICAL
    assert hp_query.expected_doc_ids == ["qdrant_engine", "hnsw_graphs"]
    assert len(hp_docs) == 3
    assert set(hp_query.expected_doc_ids).issubset({d.doc_id for d in hp_docs})

    # RAGBench adapter (offline row fixture)
    ragbench_row = {
        "id": "rb_7",
        "question": "Explain PagedAttention and Prefix Caching.",
        "response": "PagedAttention pages KV blocks; Prefix Caching reuses shared prompt blocks.",
        "documents": [
            "PagedAttention splits KV cache into 16-token blocks.",
            "Prefix Caching hashes shared prompt KV blocks to cut TTFT.",
        ],
    }
    rb_query, rb_docs = adapt_ragbench_row(ragbench_row, index=1)
    assert rb_query.stratum == EvalStratum.ANALYTICAL
    assert len(rb_docs) == 2
    assert rb_query.expected_doc_ids == [d.doc_id for d in rb_docs]


# ---------------------------------------------------------------------------
# 3. Deterministic IR & Agentic Metric Math Tests
# ---------------------------------------------------------------------------


def test_ir_metrics_math_and_deduplication() -> None:
    # Duplicate chunks from same doc collapse while keeping first rank position
    raw_retrieved = ["doc_b", "doc_b", "doc_a", "doc_c", "doc_a"]
    assert deduplicate_doc_ids(raw_retrieved) == ["doc_b", "doc_a", "doc_c"]

    expected = ["doc_a", "doc_d"]
    assert recall_at_k(raw_retrieved, expected, k=3) == pytest.approx(0.5)
    assert precision_at_k(raw_retrieved, expected, k=3) == pytest.approx(1.0 / 3.0)
    # doc_a is at rank 2 after deduplication -> MRR = 1/2
    assert mrr_at_k(raw_retrieved, expected, k=3) == pytest.approx(0.5)

    # NDCG@3: DCG = 1/log2(3), IDCG (2 relevant docs) = 1/log2(2) + 1/log2(3)
    expected_ndcg = (1.0 / math.log2(3)) / (1.0 / math.log2(2) + 1.0 / math.log2(3))
    assert ndcg_at_k(raw_retrieved, expected, k=3) == pytest.approx(expected_ndcg, rel=1e-5)

    scores = compute_ir_scores(["doc_a", "doc_d"], expected)
    assert scores.recall_at_k == 1.0
    assert scores.precision_at_k == 1.0
    assert scores.mrr_at_k == 1.0
    assert scores.ndcg_at_k == 1.0
    assert scores.hit_at_k is True

    # Empty retrieval handles safely
    empty_scores = compute_ir_scores([], expected)
    assert empty_scores.recall_at_k == 0.0
    assert empty_scores.hit_at_k is False

    # Percentile & binary precision/recall
    assert percentile([10.0, 20.0, 30.0, 40.0, 50.0], 50.0) == pytest.approx(30.0)
    assert percentile([10.0, 20.0, 30.0, 40.0, 50.0], 95.0) == pytest.approx(48.0)

    prec, rec = compute_binary_precision_recall(
        [True, True, False, False],
        [True, False, True, False],
    )
    assert prec == pytest.approx(0.5)
    assert rec == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 4. RAGAS Bridge & Sample Builder Tests ($0 Offline)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ragas_embeddings_bridge_and_sample_builder() -> None:
    bridge = FastEmbedRagasEmbeddings(_StubDenseEmbedder())
    sync_q = bridge.embed_query("qdrant hybrid")
    async_q = await bridge.aembed_query("qdrant hybrid")
    assert len(sync_q) == len(_VOCAB)
    assert sync_q == async_q
    docs_vecs = await bridge.aembed_documents(["qdrant", "vllm"])
    assert len(docs_vecs) == 2

    queries = [
        GoldenQuery(
            id="q1",
            query="What is ef_construct?",
            stratum=EvalStratum.FACTUAL,
            expected_query_type=QueryType.FACTUAL,
            needs_retrieval=True,
            expected_doc_ids=["01_qdrant_architecture_and_hnsw"],
            reference_answer="Default ef_construct is 100.",
        ),
        GoldenQuery(
            id="q2",
            query="Hello!",
            stratum=EvalStratum.CONVERSATIONAL,
            expected_query_type=QueryType.CONVERSATIONAL,
            needs_retrieval=False,
            expected_doc_ids=[],
            reference_answer="Hi there!",
        ),
    ]
    answers = [
        FinalAnswer(
            query="What is ef_construct?",
            answer="ef_construct defaults to 100 [C1].",
            query_type=QueryType.FACTUAL,
            evidence_chunks=[
                EvidenceChunk(
                    citation_id="C1",
                    chunk_id="c1",
                    text="ef_construct default value is 100.",
                    doc_id="01_qdrant_architecture_and_hnsw",
                )
            ],
        ),
        FinalAnswer(
            query="Hello!",
            answer="Hello! How can I help?",
            query_type=QueryType.CONVERSATIONAL,
            evidence_chunks=[],
        ),
    ]

    included_ids, samples = build_ragas_samples(queries, answers)
    assert included_ids == ["q1"]
    assert len(samples) == 1
    assert samples[0].user_input == "What is ef_construct?"
    assert samples[0].reference == "Default ef_construct is 100."


# ---------------------------------------------------------------------------
# 5. End-to-End $0 Offline IR & Live 4-Way Ablation Runner Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_offline_ir_benchmark_end_to_end_zero_cost(tmp_path: Path) -> None:
    settings = Settings()
    strategies = load_strategies(settings.strategies_config_path)

    store = QdrantHybridStore.from_settings(
        settings,
        collection_name="eval_offline_ir_test",
        dense_embedder=_StubDenseEmbedder(),
        sparse_embedder=_StubSparseEmbedder(),
        in_memory=True,
    )
    try:
        docs = load_directory(Path("data/raw"))
        chunks = chunk_documents(docs)
        await store.ensure_collection(recreate=True)
        await store.upsert_chunks(chunks)

        all_golden = load_golden_jsonl(Path("data/eval/golden_set.jsonl"))
        # Pick 1 representative query per stratum (plus 1 TR factual) = 6 queries
        pick_ids = {
            "fact_en_01",
            "fact_tr_01",
            "anal_en_01",
            "conv_en_01",
            "web_en_01",
            "unans_en_01",
        }
        sample_queries = [q for q in all_golden if q.id in pick_ids]
        assert len(sample_queries) == 6

        report = await run_offline_ir_benchmark(
            sample_queries,
            store=store,
            reranker=_LexicalStubReranker(),
            strategies=strategies,
            fixed_top_k=5,
        )
    finally:
        await store.close()

    assert report.mode == "offline-ir"
    assert report.configs == list(OFFLINE_IR_CONFIGS)
    assert len(report.summaries) == 4
    assert len(report.records) == 4 * len(sample_queries)

    # Verify adaptive_dynamic_k skips conversational retrieval (selected_k == 0)
    conv_adaptive = next(
        r
        for r in report.records
        if r.config_name == "adaptive_dynamic_k" and r.query_id == "conv_en_01"
    )
    assert conv_adaptive.selected_k == 0
    assert conv_adaptive.context_tokens == 0
    assert conv_adaptive.routing_correct is True

    # Verify naive_dense still retrieves on conversational query (over-retrieval baseline)
    conv_naive = next(
        r for r in report.records if r.config_name == "naive_dense" and r.query_id == "conv_en_01"
    )
    assert conv_naive.selected_k == 5

    # Save & inspect artifacts (JSON, Markdown, Plotly HTML)
    artifacts = save_benchmark_artifacts(report, tmp_path / "reports")
    assert artifacts["json"].is_file()
    assert artifacts["markdown"].is_file()
    markdown = render_markdown_report(report)
    for cfg in OFFLINE_IR_CONFIGS:
        assert f"`{cfg}`" in markdown


@pytest.mark.asyncio
async def test_live_4way_ablation_and_budget_guardrail(tmp_path: Path) -> None:
    settings = Settings(llm_cache_enabled=False, web_fallback_enabled=True)
    strategies = load_strategies(settings.strategies_config_path)

    store = QdrantHybridStore.from_settings(
        settings,
        collection_name="eval_live_ablation_test",
        dense_embedder=_StubDenseEmbedder(),
        sparse_embedder=_StubSparseEmbedder(),
        in_memory=True,
    )
    try:
        docs = load_directory(Path("data/raw"))
        await store.ensure_collection(recreate=True)
        await store.upsert_chunks(chunk_documents(docs))

        fake_fast = _BenchmarkFakeLLM(model_name="qwen/qwen3-30b-a3b-instruct-2507")
        fake_gen = _BenchmarkFakeLLM(model_name="deepseek/deepseek-v3.2")
        runtime = GraphRuntime(
            settings=settings,
            strategies=strategies,
            models=ModelBundle(
                fast_model=fake_fast,
                fast_model_name="qwen/qwen3-30b-a3b-instruct-2507",
                generation_model=fake_gen,
                generation_model_name="deepseek/deepseek-v3.2",
            ),
            store=store,
            reranker=_LexicalStubReranker(),
            fallback_retriever=_StubWebFallback(),
            cache=SQLiteLLMCache(tmp_path / "cache.sqlite"),
            cost_tracker=CostTracker(max_cost_usd=1.0),
        )
        workflow = AdaptiveRAGWorkflow(runtime)

        all_golden = load_golden_jsonl(Path("data/eval/golden_set.jsonl"))
        sample_queries = [
            q for q in all_golden if q.id in {"fact_en_01", "anal_en_01", "conv_en_01"}
        ]

        report = await run_live_ablation_benchmark(
            sample_queries,
            workflow=workflow,
            max_cost_usd=0.50,
            with_ragas=False,
        )
        assert report.mode == "live"
        assert report.budget_exceeded is False
        assert report.configs == list(LIVE_CONFIG_NAMES)
        assert len(report.summaries) == 4
        assert all(s.error_count == 0 for s in report.summaries)

        full_summary = next(s for s in report.summaries if s.config_name == "full")
        assert full_summary.routing_accuracy == pytest.approx(1.0)
        assert full_summary.mean_citation_precision is not None

        # Now enforce a near-zero FinOps cap to verify graceful BudgetExceededError stop
        # Use fresh uncached workflow so every call bills real token estimates
        runtime_uncached = GraphRuntime(
            settings=settings,
            strategies=strategies,
            models=ModelBundle(
                fast_model=_BenchmarkFakeLLM(
                    prompt_tokens_per_call=50000,
                    completion_tokens_per_call=20000,
                ),
                fast_model_name="qwen/qwen3-30b-a3b-instruct-2507",
                generation_model=_BenchmarkFakeLLM(
                    model_name="deepseek/deepseek-v3.2",
                    prompt_tokens_per_call=50000,
                    completion_tokens_per_call=20000,
                ),
                generation_model_name="deepseek/deepseek-v3.2",
            ),
            store=store,
            reranker=_LexicalStubReranker(),
            fallback_retriever=_StubWebFallback(),
            cache=None,
            cost_tracker=CostTracker(max_cost_usd=0.001),
        )
        capped_report = await run_live_ablation_benchmark(
            sample_queries,
            workflow=AdaptiveRAGWorkflow(runtime_uncached),
            max_cost_usd=0.001,
        )
        assert capped_report.budget_exceeded is True
    finally:
        await store.close()
