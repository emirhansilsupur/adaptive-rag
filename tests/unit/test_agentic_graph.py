"""End-to-end $0 unit tests for the AdaptiveRAG LangGraph cyclic workflow across all 7 paths."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from adaptiverag.config import Settings, load_strategies
from adaptiverag.graph import AblationConfig, AdaptiveRAGWorkflow, GraphRuntime
from adaptiverag.graph.edges import route_after_citation_validation
from adaptiverag.ingestion.chunker import DocumentChunk
from adaptiverag.llm import ModelBundle, SQLiteLLMCache
from adaptiverag.retrieval.embeddings import SparseVectorData
from adaptiverag.retrieval.qdrant_store import QdrantHybridStore
from adaptiverag.schemas import (
    ChatMessage,
    CitationReport,
    ConfidenceLevel,
    EvidenceChunk,
    QueryType,
    SemanticAuditStatus,
    SourceType,
)

_TEST_NAMESPACE = UUID("12345678-1234-5678-1234-567812345678")

# ---------------------------------------------------------------------------
# Deterministic $0 test doubles (embedders, reranker, web fallback, LLMs)
# ---------------------------------------------------------------------------

_VOCAB = [
    "qdrant",
    "hybrid",
    "bm25",
    "rrf",
    "langgraph",
    "stategraph",
    "vllm",
    "pagedattention",
    "fastapi",
    "sse",
    "crag",
    "reranker",
    "turkce",
    "vektor",
    "veritabani",
    "release",
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
    return re.findall(r"[a-z0-9]+", normalized)


class _StubDenseEmbedder:
    @property
    def model_name(self) -> str:
        return "stub-dense-16d"

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


class _ControllableReranker:
    """Reranker that supports per-call scripted score sequences or default high scores."""

    def __init__(self, score_schedule: list[float] | None = None) -> None:
        self._schedule = list(score_schedule) if score_schedule is not None else []
        self.call_count = 0
        self.seen_queries: list[str] = []

    @property
    def model_name(self) -> str:
        return "stub-reranker"

    def rerank(
        self,
        query: str,
        chunks: Sequence[EvidenceChunk],
        *,
        top_n: int | None = None,
    ) -> list[EvidenceChunk]:
        call_idx = self.call_count
        self.call_count += 1
        self.seen_queries.append(query)
        base_score = (
            self._schedule[min(call_idx, len(self._schedule) - 1)] if self._schedule else 0.88
        )
        scored: list[EvidenceChunk] = []
        for idx, chunk in enumerate(chunks):
            # Decay slightly per rank while staying within [0.0, 1.0]
            score = max(0.01, min(0.99, round(base_score - idx * 0.04, 4)))
            scored.append(
                chunk.model_copy(
                    update={"rerank_raw_score": score * 5.0 - 2.0, "rerank_score": score}
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
    def __init__(self, chunks: list[EvidenceChunk] | None = None) -> None:
        self._chunks = chunks or []
        self.calls: list[str] = []

    async def asearch(
        self,
        query: str,
        *,
        max_results: int | None = None,
        start_citation_index: int = 1,
    ) -> list[EvidenceChunk]:
        self.calls.append(query)
        limit = max_results or len(self._chunks)
        out: list[EvidenceChunk] = []
        for idx, chunk in enumerate(self._chunks[:limit]):
            out.append(chunk.model_copy(update={"citation_id": f"W{start_citation_index + idx}"}))
        return out


class _ScriptedChatModel(BaseChatModel):
    responses: list[str] = Field(default_factory=list)
    call_count: int = 0
    prompt_tokens: int = 100
    completion_tokens: int = 50

    @property
    def _llm_type(self) -> str:
        return "scripted-graph-llm"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        msg = AIMessage(
            content=self.responses[idx],
            usage_metadata={
                "input_tokens": self.prompt_tokens,
                "output_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


async def _build_in_memory_store() -> QdrantHybridStore:
    settings = Settings()
    store = QdrantHybridStore.from_settings(
        settings,
        collection_name="test_graph_docs",
        dense_embedder=_StubDenseEmbedder(),
        sparse_embedder=_StubSparseEmbedder(),
        in_memory=True,
    )
    await store.ensure_collection(recreate=True)

    docs = [
        (
            "qdrant_hybrid",
            "Qdrant Hybrid Search",
            "RRF Fusion",
            "Qdrant combines dense cosine vectors and BM25 sparse vectors natively via Prefetch and Reciprocal Rank Fusion (RRF).",
            "en",
        ),
        (
            "langgraph_state",
            "LangGraph StateGraph",
            "Cyclic Agents",
            "LangGraph models cyclic agentic workflows using typed state reducers, conditional edges, and bounded retry loops.",
            "en",
        ),
        (
            "vllm_paged",
            "vLLM PagedAttention",
            "KV Cache",
            "vLLM uses PagedAttention to eliminate KV-cache memory fragmentation and increase LLM serving throughput.",
            "en",
        ),
        (
            "tr_rag_guide",
            "Üretim Ortamında Hibrit RAG",
            "Hibrit Arama ve Reranker",
            "Türkçe teknik dokümanlarda BM25 ve yoğun vektör aramasının RRF ile birleştirilmesi ve cross-encoder reranker kullanılması doğruluğu artırır.",
            "tr",
        ),
    ]

    chunks: list[DocumentChunk] = []
    for idx, (doc_id, title, section, text, lang) in enumerate(docs):
        chunks.append(
            DocumentChunk(
                chunk_id=str(uuid5(_TEST_NAMESPACE, f"{doc_id}::{idx}")),
                doc_id=doc_id,
                title=title,
                section_path=section,
                text=text,
                embedding_text=f"{title} > {section}\n{text}",
                chunk_index=idx,
                token_count=32,
                metadata={"language": lang},
            )
        )
    await store.upsert_chunks(chunks)
    return store


def _make_workflow(
    *,
    store: QdrantHybridStore,
    fast_responses: list[str],
    gen_responses: list[str],
    reranker_scores: list[float] | None = None,
    web_chunks: list[EvidenceChunk] | None = None,
    cache_path: Path | None = None,
) -> tuple[
    AdaptiveRAGWorkflow,
    _ScriptedChatModel,
    _ScriptedChatModel,
    _ControllableReranker,
    _StubWebFallback,
]:
    settings = Settings(
        llm_cache_enabled=cache_path is not None,
        web_fallback_enabled=True,
        langfuse_enabled=False,
    )
    strategies = load_strategies("configs/strategies.yaml")
    fast_model = _ScriptedChatModel(responses=fast_responses)
    gen_model = _ScriptedChatModel(responses=gen_responses)
    reranker = _ControllableReranker(score_schedule=reranker_scores)
    fallback = _StubWebFallback(chunks=web_chunks)

    runtime = GraphRuntime(
        settings=settings,
        strategies=strategies,
        models=ModelBundle(
            fast_model=fast_model,
            fast_model_name="qwen/qwen3-30b-a3b-instruct-2507",
            generation_model=gen_model,
            generation_model_name="deepseek/deepseek-v3.2",
        ),
        store=store,
        reranker=reranker,
        fallback_retriever=fallback,
        cache=SQLiteLLMCache(cache_path) if cache_path else None,
    )
    return AdaptiveRAGWorkflow(runtime), fast_model, gen_model, reranker, fallback


# ---------------------------------------------------------------------------
# 1. Conversational fast path (0 retrieval calls)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_conversational_path_skips_retrieval() -> None:
    store = await _build_in_memory_store()
    try:
        router_json = json.dumps(
            {
                "query_type": "conversational",
                "needs_retrieval": False,
                "language": "tr",
                "standalone_query": "Merhaba, nasılsın?",
                "requires_fresh_web_info": False,
                "reasoning": "Simple Turkish greeting.",
            }
        )
        workflow, fast_model, gen_model, reranker, fallback = _make_workflow(
            store=store,
            fast_responses=[
                router_json,
                "Merhaba! Size teknik dokümanlarınızla ilgili nasıl yardımcı olabilirim?",
            ],
            gen_responses=[],
        )

        result = await workflow.arun("Merhaba, nasılsın?")

        assert result.query_type == QueryType.CONVERSATIONAL
        assert result.language == "tr"
        assert "Merhaba" in result.answer
        assert result.evidence_chunks == []
        assert result.citation_report is None
        assert result.abstained is False
        assert result.fallback_used is False
        assert reranker.call_count == 0
        assert fallback.calls == []
        assert gen_model.call_count == 0
        assert fast_model.call_count == 2  # router + conversational reply
        assert [t.node for t in result.trace] == ["classify_and_plan", "conversational_respond"]
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# 2. Factual happy path with coreference resolution + high confidence + valid citations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_factual_happy_path_with_coreference() -> None:
    store = await _build_in_memory_store()
    try:
        router_json = json.dumps(
            {
                "query_type": "factual",
                "needs_retrieval": True,
                "language": "en",
                "standalone_query": "How does Qdrant combine dense and BM25 sparse vectors via RRF?",
                "requires_fresh_web_info": False,
                "reasoning": "Direct factual lookup resolving 'it' to Qdrant.",
            }
        )
        entailment_json = json.dumps(
            {
                "verifications": [
                    {
                        "claim": "Qdrant fuses dense cosine vectors and BM25 sparse vectors natively via Prefetch and RRF.",
                        "cited_ids": ["C1"],
                        "verdict": "supported",
                        "explanation": "Directly supported by C1.",
                    }
                ]
            }
        )
        workflow, fast_model, gen_model, reranker, fallback = _make_workflow(
            store=store,
            fast_responses=[router_json, entailment_json],
            gen_responses=[
                "Qdrant combines dense cosine vectors and BM25 sparse vectors natively using Prefetch and Reciprocal Rank Fusion (RRF) [C1]."
            ],
            reranker_scores=[0.91],
        )

        history = [
            ChatMessage(role="user", content="Tell me about Qdrant."),
            ChatMessage(role="assistant", content="Qdrant is a vector search engine."),
        ]
        result = await workflow.arun(
            "How does it fuse dense and sparse search?", chat_history=history
        )

        assert result.query_type == QueryType.FACTUAL
        assert result.query_plan is not None
        assert "Qdrant" in result.query_plan.standalone_query
        assert result.confidence is not None
        assert result.confidence.level == ConfidenceLevel.HIGH
        assert result.confidence.crag_grade is None  # zero-cost HIGH fast path bypassed CRAG!
        assert result.citation_report is not None
        assert result.citation_report.passed is True
        assert result.citation_report.valid_citation_ids == ["C1"]
        assert result.fallback_used is False
        assert result.abstained is False
        assert result.rewrite_attempts == 0
        assert result.regeneration_attempts == 0
        assert reranker.call_count == 1
        assert fallback.calls == []
        assert gen_model.call_count == 1
        assert fast_model.call_count == 2  # router + citation Tier-2 audit
        assert result.usage.total_tokens > 0
        assert [t.node for t in result.trace] == [
            "classify_and_plan",
            "hybrid_retrieve",
            "rerank_and_select_k",
            "score_confidence",
            "generate_answer",
            "validate_citations",
        ]
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# 3. Analytical multi-query path in Turkish
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_analytical_multiquery_turkish_synthesis() -> None:
    store = await _build_in_memory_store()
    try:
        router_json = json.dumps(
            {
                "query_type": "analytical",
                "needs_retrieval": True,
                "language": "tr",
                "standalone_query": "Qdrant hibrit arama ile LangGraph döngüsel ajan mimarisini birlikte kullanmanın avantajları nelerdir?",
                "requires_fresh_web_info": False,
                "reasoning": "Multi-system architectural analysis.",
            }
        )
        decompose_json = json.dumps(
            {
                "sub_queries": [
                    "Qdrant hybrid search BM25 RRF",
                    "LangGraph StateGraph cyclic agents",
                    "Türkçe hibrit RAG reranker",
                ],
                "rationale": "Separate retrieval, orchestration, and Turkish reranking facets.",
            }
        )
        entailment_json = json.dumps(
            {
                "verifications": [
                    {
                        "claim": "Qdrant, BM25 ve yoğun vektörleri yerel RRF ile birleştirir.",
                        "cited_ids": ["C1"],
                        "verdict": "supported",
                        "explanation": "Supported by C1.",
                    },
                    {
                        "claim": "LangGraph döngüsel ajan akışlarını sınırlı tekrar döngüleriyle yönetir.",
                        "cited_ids": ["C2"],
                        "verdict": "supported",
                        "explanation": "Supported by C2.",
                    },
                ]
            }
        )
        workflow, _, _, _, _ = _make_workflow(
            store=store,
            fast_responses=[router_json, decompose_json, entailment_json],
            gen_responses=[
                "Qdrant, yoğun vektörler ile BM25 seyrek vektörlerini yerel RRF ile birleştirir [C1]. "
                "LangGraph ise döngüsel ajan iş akışlarını tipli state ve sınırlı tekrar döngüleriyle yönetir [C2]."
            ],
            reranker_scores=[0.90],
        )

        result = await workflow.arun(
            "Qdrant hibrit arama ile LangGraph döngüsel ajan mimarisini birlikte kullanmanın avantajları nelerdir?"
        )

        assert result.query_type == QueryType.ANALYTICAL
        assert result.language == "tr"
        assert result.query_plan is not None
        assert len(result.query_plan.sub_queries) == 3
        assert result.citation_report is not None and result.citation_report.passed is True
        assert "decompose_queries" in [t.node for t in result.trace]
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# 4. Corrective rewrite recovery loop (weak retrieval -> rewrite -> strong retrieval)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_corrective_rewrite_recovers_before_web_fallback() -> None:
    store = await _build_in_memory_store()
    try:
        router_json = json.dumps(
            {
                "query_type": "factual",
                "needs_retrieval": True,
                "language": "en",
                "standalone_query": "How does vLLM avoid KV memory fragmentation?",
                "requires_fresh_web_info": False,
                "reasoning": "Single technical mechanism lookup.",
            }
        )
        rewrite_json = json.dumps(
            {
                "rewritten_queries": ["vLLM PagedAttention KV cache memory fragmentation"],
                "diagnosis": "Initial query missed canonical term PagedAttention.",
            }
        )
        entailment_json = json.dumps(
            {
                "verifications": [
                    {
                        "claim": "vLLM uses PagedAttention to eliminate KV-cache memory fragmentation.",
                        "cited_ids": ["C1"],
                        "verdict": "supported",
                        "explanation": "Directly supported.",
                    }
                ]
            }
        )
        # 1st rerank returns 0.25 (< tau_min 0.35 -> 0 selected -> LOW -> REWRITE)
        # 2nd rerank (after rewrite) returns 0.90 -> HIGH -> PROCEED
        workflow, _, _, reranker, fallback = _make_workflow(
            store=store,
            fast_responses=[router_json, rewrite_json, entailment_json],
            gen_responses=[
                "vLLM uses PagedAttention to eliminate KV-cache memory fragmentation and boost throughput [C1]."
            ],
            reranker_scores=[0.25, 0.90],
        )

        result = await workflow.arun("How does vLLM avoid KV memory fragmentation?")

        assert result.rewrite_attempts == 1
        assert result.fallback_used is False
        assert result.abstained is False
        assert reranker.call_count == 2
        assert fallback.calls == []
        assert result.citation_report is not None and result.citation_report.passed is True
        nodes_visited = [t.node for t in result.trace]
        assert "corrective_rewrite" in nodes_visited
        assert nodes_visited.count("hybrid_retrieve") == 2
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# 5. Web fallback path (fresh web query -> W1 citation)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_web_fallback_for_fresh_web_query() -> None:
    store = await _build_in_memory_store()
    try:
        router_json = json.dumps(
            {
                "query_type": "factual",
                "needs_retrieval": True,
                "language": "en",
                "standalone_query": "What is the latest Qdrant release version?",
                "requires_fresh_web_info": True,
                "reasoning": "Asks for latest release version.",
            }
        )
        entailment_json = json.dumps(
            {
                "verifications": [
                    {
                        "claim": "Qdrant 1.13 introduced improved sparse vector quantization.",
                        "cited_ids": ["W1"],
                        "verdict": "supported",
                        "explanation": "Stated in W1.",
                    }
                ]
            }
        )
        web_hit = EvidenceChunk(
            citation_id="W1",
            chunk_id="web-qdrant-113",
            doc_id="web::qdrant.tech",
            title="Qdrant Releases",
            section_path="qdrant.tech",
            url="https://qdrant.tech/releases",
            text="Qdrant 1.13 is the latest release introducing improved sparse vector quantization.",
            source_type=SourceType.WEB,
            fusion_score=1.0,
            token_count=20,
        )
        workflow, _, _, _, fallback = _make_workflow(
            store=store,
            fast_responses=[router_json, entailment_json],
            gen_responses=[
                "According to the latest release notes, Qdrant 1.13 introduced improved sparse vector quantization [W1]."
            ],
            reranker_scores=[0.80, 0.92],
            web_chunks=[web_hit],
        )

        result = await workflow.arun("What is the latest Qdrant release version?")

        assert result.fallback_used is True
        assert result.abstained is False
        assert len(fallback.calls) == 1
        assert any(
            c.source_type == SourceType.WEB and c.citation_id == "W1"
            for c in result.evidence_chunks
        )
        assert result.citation_report is not None and result.citation_report.passed is True
        assert "W1" in result.citation_report.valid_citation_ids
        assert result.dynamic_k_decision is not None
        assert result.dynamic_k_decision.cutoff_reason == "web_fallback"
        assert result.dynamic_k_decision.selected_count == len(result.evidence_chunks)
        assert result.dynamic_k_decision.total_context_tokens >= 20
        assert "web_fallback_search" in [t.node for t in result.trace]
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# 6. Citation failure -> bounded strict regeneration loop (pass & bounded fail)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_citation_failure_triggers_bounded_regeneration() -> None:
    store = await _build_in_memory_store()
    try:
        router_json = json.dumps(
            {
                "query_type": "factual",
                "needs_retrieval": True,
                "language": "en",
                "standalone_query": "How does Qdrant perform hybrid search?",
                "requires_fresh_web_info": False,
                "reasoning": "Direct factual lookup.",
            }
        )
        # First draft uses ghost [C99] (fails Tier-1 + Tier-2); second draft uses valid [C1] (passes)
        audit_fail_json = json.dumps(
            {
                "verifications": [
                    {
                        "claim": "Qdrant merges vectors natively via RRF.",
                        "cited_ids": ["C1"],
                        "verdict": "supported",
                        "explanation": "Supported by C1.",
                    },
                    {
                        "claim": "Qdrant runs quantum annealing automatically.",
                        "cited_ids": ["C99"],
                        "verdict": "unsupported",
                        "explanation": "C99 does not exist.",
                    },
                ]
            }
        )
        audit_pass_json = json.dumps(
            {
                "verifications": [
                    {
                        "claim": "Qdrant merges dense and BM25 sparse vectors natively via RRF.",
                        "cited_ids": ["C1"],
                        "verdict": "supported",
                        "explanation": "Supported by C1.",
                    }
                ]
            }
        )
        workflow, _, gen_model, _, _ = _make_workflow(
            store=store,
            fast_responses=[router_json, audit_fail_json, audit_pass_json],
            gen_responses=[
                "Qdrant merges vectors natively via RRF [C1]. It also runs quantum annealing [C99].",
                "Qdrant merges dense cosine vectors and BM25 sparse vectors natively via Prefetch and RRF [C1].",
            ],
            reranker_scores=[0.91],
        )

        result = await workflow.arun("How does Qdrant perform hybrid search?")

        assert result.regeneration_attempts == 1
        assert gen_model.call_count == 2
        assert result.citation_report is not None
        assert result.citation_report.passed is True
        assert result.citation_report.regeneration_triggered is True
        assert result.citation_report.invalid_citation_ids == []
        assert "[C99]" not in result.answer
        assert [t.node for t in result.trace].count("validate_citations") == 2
        assert "regenerate_strict" in [t.node for t in result.trace]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_graph_regeneration_stops_at_max_regenerations_bound() -> None:
    store = await _build_in_memory_store()
    try:
        router_json = json.dumps(
            {
                "query_type": "factual",
                "needs_retrieval": True,
                "language": "en",
                "standalone_query": "How does Qdrant perform hybrid search?",
                "requires_fresh_web_info": False,
                "reasoning": "Direct factual lookup.",
            }
        )
        # Both initial draft and strict regen omit citations completely (FinOps fast-fail -> 0 Tier-2 calls)
        workflow, _, gen_model, _, _ = _make_workflow(
            store=store,
            fast_responses=[router_json],
            gen_responses=[
                "Qdrant performs hybrid search natively by fusing dense and sparse vectors without citations.",
                "Qdrant still performs hybrid search natively by fusing dense and sparse vectors without citations.",
            ],
            reranker_scores=[0.91],
        )

        result = await workflow.arun("How does Qdrant perform hybrid search?")

        assert result.regeneration_attempts == 1  # max_regenerations=1 enforced
        assert gen_model.call_count == 2
        assert result.citation_report is not None
        assert result.citation_report.passed is False
        assert result.citation_report.regeneration_triggered is True
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_graph_judge_failure_is_unverified_not_regenerated() -> None:
    store = await _build_in_memory_store()
    try:
        router_json = json.dumps(
            {
                "query_type": "factual",
                "needs_retrieval": True,
                "language": "en",
                "standalone_query": "How does Qdrant perform hybrid search?",
                "requires_fresh_web_info": False,
                "reasoning": "Direct factual lookup.",
            }
        )
        # The judge never returns JSON, so the semantic audit fails while the draft is well cited.
        workflow, _, gen_model, _, _ = _make_workflow(
            store=store,
            fast_responses=[router_json, "sorry, I cannot produce JSON"],
            gen_responses=[
                "Qdrant merges dense cosine vectors and BM25 sparse vectors natively via Prefetch and RRF [C1].",
            ],
            reranker_scores=[0.91],
        )

        result = await workflow.arun("How does Qdrant perform hybrid search?")

        report = result.citation_report
        assert report is not None
        assert report.passed is False
        assert report.structural_passed is True
        assert report.semantic_audit == SemanticAuditStatus.FAILED
        assert result.regeneration_attempts == 0
        assert gen_model.call_count == 1
        nodes = [t.node for t in result.trace]
        assert "regenerate_strict" not in nodes
        audit_event = next(t for t in result.trace if t.node == "validate_citations")
        assert audit_event.summary.startswith("Citation audit UNVERIFIED")
        assert audit_event.metadata["semantic_audit"] == "failed"
    finally:
        await store.close()


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        (CitationReport(passed=True, semantic_audit=SemanticAuditStatus.COMPLETED), "end"),
        (
            CitationReport(passed=False, semantic_audit=SemanticAuditStatus.COMPLETED),
            "regenerate_strict",
        ),
        (
            CitationReport(
                passed=False,
                structural_passed=True,
                semantic_audit=SemanticAuditStatus.FAILED,
            ),
            "end",
        ),
        (
            CitationReport(
                passed=False,
                structural_passed=False,
                invalid_citation_ids=["C9"],
                semantic_audit=SemanticAuditStatus.FAILED,
            ),
            "regenerate_strict",
        ),
    ],
    ids=["passed", "flagged-by-judge", "judge-failed", "judge-failed-and-broken-citations"],
)
def test_citation_router_regenerates_only_for_answer_problems(
    report: CitationReport, expected: str
) -> None:
    state: Any = {"citation_report": report, "regeneration_attempts": 0}
    assert route_after_citation_validation(state, max_regenerations=1) == expected


# ---------------------------------------------------------------------------
# 7. Honest abstention when local + rewrite + web fallback all fail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_honest_abstention_when_all_evidence_exhausted() -> None:
    store = await _build_in_memory_store()
    try:
        router_json = json.dumps(
            {
                "query_type": "factual",
                "needs_retrieval": True,
                "language": "tr",
                "standalone_query": "Mars kolonisi 2099 bütçe dağılımı nedir?",
                "requires_fresh_web_info": False,
                "reasoning": "Factual question outside local corpus.",
            }
        )
        rewrite_json = json.dumps(
            {
                "rewritten_queries": ["Mars colony 2099 budget allocation"],
                "diagnosis": "No matching technical docs found.",
            }
        )
        # All local rerank passes score 0.12 (< tau_min) and web fallback returns []
        workflow, _, gen_model, _, fallback = _make_workflow(
            store=store,
            fast_responses=[router_json, rewrite_json],
            gen_responses=[],
            reranker_scores=[0.12, 0.10],
            web_chunks=[],
        )

        result = await workflow.arun("Mars kolonisi 2099 bütçe dağılımı nedir?")

        assert result.abstained is True
        assert result.fallback_used is True
        assert result.rewrite_attempts == 1
        assert len(fallback.calls) == 1
        assert gen_model.call_count == 0  # Zero wasted generation tokens!
        assert "yeterli doğrulanmış kanıt" in result.answer
        assert "abstain_response" in [t.node for t in result.trace]
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# 8. Ablation preset (naive_dense) + SSE streaming (astream_events)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_ablation_naive_dense_and_astream_events() -> None:
    store = await _build_in_memory_store()
    try:
        workflow, fast_model, gen_model, reranker, fallback = _make_workflow(
            store=store,
            fast_responses=[],
            gen_responses=["Qdrant combines dense and sparse vectors natively via RRF [C1]."],
            reranker_scores=[0.90],
        )

        events: list[dict[str, Any]] = []
        async for ev in workflow.astream_events(
            "How does Qdrant hybrid search work?",
            ablation=AblationConfig.naive_dense(fixed_top_k=3),
        ):
            events.append(ev)

        node_events = [e for e in events if e["event"] == "node_update"]
        final_events = [e for e in events if e["event"] == "final_answer"]

        assert len(final_events) == 1
        final_data = final_events[0]["data"]
        assert final_data["query_type"] == "factual"
        assert final_data["dynamic_k_decision"]["cutoff_reason"] == "static_top_k"
        assert final_data["dynamic_k_decision"]["selected_count"] <= 3
        assert final_data["abstained"] is False
        assert fast_model.call_count == 0  # router, CRAG, Tier-2 entailment all skipped
        assert reranker.call_count == 0  # reranker skipped
        assert fallback.calls == []
        assert gen_model.call_count == 1
        assert [e["node"] for e in node_events] == [
            "classify_and_plan",
            "hybrid_retrieve",
            "rerank_and_select_k",
            "score_confidence",
            "generate_answer",
            "validate_citations",
        ]
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# 9. Rewrite-aware reranking & web-fallback noise filtering regression tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_reranks_against_rewritten_queries_not_stale_standalone() -> None:
    store = await _build_in_memory_store()
    try:
        router_json = json.dumps(
            {
                "query_type": "factual",
                "needs_retrieval": True,
                "language": "tr",
                "standalone_query": "vLLM KV onbellek bellek israfini nasil azaltir?",
                "requires_fresh_web_info": False,
                "reasoning": "Cross-lingual technical lookup.",
            }
        )
        rewrite_json = json.dumps(
            {
                "rewritten_queries": [
                    "vLLM PagedAttention KV cache memory fragmentation",
                    "vLLM PagedAttention bellek parcalanmasi",
                ],
                "diagnosis": "Added canonical English query for cross-lingual reranking.",
            }
        )
        entailment_json = json.dumps(
            {
                "verifications": [
                    {
                        "claim": "vLLM PagedAttention kullanarak KV-cache bellek parçalanmasını azaltır.",
                        "cited_ids": ["C1"],
                        "verdict": "supported",
                        "explanation": "Supported by C1.",
                    }
                ]
            }
        )
        # 1st pass (original TR query) scores 0.20 (< tau_min -> rewrite)
        # 2nd pass scores the 2 rewritten queries (0.91 and 0.72 -> max-score fusion keeps 0.91)
        workflow, _, _, reranker, fallback = _make_workflow(
            store=store,
            fast_responses=[router_json, rewrite_json, entailment_json],
            gen_responses=[
                "vLLM, PagedAttention mekanizmasıyla KV-cache bellek parçalanmasını azaltır [C1]."
            ],
            reranker_scores=[0.20, 0.91, 0.72],
        )

        result = await workflow.arun("vLLM KV onbellek bellek israfini nasil azaltir?")

        assert result.abstained is False
        assert result.fallback_used is False
        assert result.rewrite_attempts == 1
        assert fallback.calls == []
        assert "vLLM PagedAttention KV cache memory fragmentation" in reranker.seen_queries
        assert "vLLM PagedAttention bellek parcalanmasi" in reranker.seen_queries
        assert result.confidence is not None and result.confidence.final_score >= 0.72
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_graph_web_fallback_filters_irrelevant_snippets_and_abstains() -> None:
    store = await _build_in_memory_store()
    try:
        router_json = json.dumps(
            {
                "query_type": "factual",
                "needs_retrieval": True,
                "language": "en",
                "standalone_query": "What is the official website URL of FastAPI?",
                "requires_fresh_web_info": False,
                "reasoning": "External documentation lookup.",
            }
        )
        rewrite_json = json.dumps(
            {
                "rewritten_queries": ["FastAPI official documentation URL"],
                "diagnosis": "Try canonical query.",
            }
        )
        noisy_web_hit = EvidenceChunk(
            citation_id="W1",
            chunk_id="web-dict-noise",
            doc_id="web::dictionary.example",
            title="Dictionary definition of official",
            text="Official means relating to an authority or public body.",
            source_type=SourceType.WEB,
            url="https://dictionary.example/official",
            token_count=12,
        )
        # Local pass 1 (0.10), local pass 2 (0.10), web rerank (0.04 < min_web_score 0.175 -> filtered)
        workflow, _, gen_model, _, fallback = _make_workflow(
            store=store,
            fast_responses=[router_json, rewrite_json],
            gen_responses=[],
            reranker_scores=[0.10, 0.10, 0.04],
            web_chunks=[noisy_web_hit],
        )

        result = await workflow.arun("What is the official website URL of FastAPI?")

        assert result.fallback_used is True
        assert result.abstained is True
        assert result.evidence_chunks == []
        assert result.dynamic_k_decision is not None
        assert result.dynamic_k_decision.cutoff_reason == "web_fallback_empty"
        assert result.dynamic_k_decision.selected_count == 0
        assert result.dynamic_k_decision.total_context_tokens == 0
        assert len(fallback.calls) == 1
        assert gen_model.call_count == 0
        assert "abstain_response" in [t.node for t in result.trace]
    finally:
        await store.close()
