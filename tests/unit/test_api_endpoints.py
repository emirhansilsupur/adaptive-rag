"""$0 unit tests for FastAPI endpoints (/v1/health, /v1/ingest, /v1/query, /v1/query/stream)."""

from __future__ import annotations

import importlib.util
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

import httpx
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from qdrant_client import models

from adaptiverag.api.dependencies import AppContainer
from adaptiverag.api.main import create_app
from adaptiverag.config import Settings, load_strategies
from adaptiverag.graph import AdaptiveRAGWorkflow, GraphRuntime
from adaptiverag.ingestion.chunker import DocumentChunk
from adaptiverag.ingestion.web_loader import FirecrawlClient
from adaptiverag.llm import CostTracker, LLMProvider, ModelBundle, SQLiteLLMCache
from adaptiverag.retrieval.embeddings import SparseVectorData
from adaptiverag.retrieval.qdrant_store import QdrantHybridStore, RetrievalMode
from adaptiverag.schemas import EvidenceChunk, FinalAnswer, QueryType

_TEST_NAMESPACE = UUID("22345678-1234-5678-1234-567812345678")
_VOCAB = [
    "qdrant",
    "hybrid",
    "bm25",
    "rrf",
    "langgraph",
    "fastapi",
    "sse",
    "reranker",
    "turkce",
    "vektor",
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
        return "stub-dense-10d"

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


class _StubReranker:
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
        scored = [
            chunk.model_copy(
                update={
                    "rerank_raw_score": 2.4 - idx * 0.2,
                    "rerank_score": max(0.1, round(0.90 - idx * 0.05, 4)),
                }
            )
            for idx, chunk in enumerate(chunks)
        ]
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
    enabled: bool = True

    @property
    def provider_names(self) -> list[str]:
        return ["stub_web"]

    async def asearch(
        self,
        query: str,
        *,
        max_results: int | None = None,
        start_citation_index: int = 1,
    ) -> list[EvidenceChunk]:
        return []


class _ScriptedChatModel(BaseChatModel):
    responses: list[str] = Field(default_factory=list)
    cursor: int = 0
    prompt_tokens: int = 120
    completion_tokens: int = 60

    @property
    def _llm_type(self) -> str:
        return "scripted-api-test-chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.responses:
            raise RuntimeError("No scripted responses left")
        idx = min(self.cursor, len(self.responses) - 1)
        self.cursor += 1
        msg = AIMessage(
            content=self.responses[idx],
            usage_metadata={
                "input_tokens": self.prompt_tokens,
                "output_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


class _FailingStore:
    collection_name = "broken_collection"

    async def health_check(self) -> dict[str, Any]:
        raise ConnectionError("Qdrant connection refused on localhost:6333")

    async def close(self) -> None:
        return None


def _parse_sse_text(body: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse raw SSE response text into (event, json_payload) tuples."""
    events: list[tuple[str, dict[str, Any]]] = []
    event_name = "message"
    data_lines: list[str] = []

    for raw_line in body.splitlines():
        line = raw_line.rstrip("\r\n")
        if not line:
            if data_lines:
                events.append((event_name, json.loads("\n".join(data_lines))))
                event_name = "message"
                data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())

    if data_lines:
        events.append((event_name, json.loads("\n".join(data_lines))))
    return events


async def _build_test_container(
    tmp_path: Path,
    *,
    fast_responses: list[str] | None = None,
    gen_responses: list[str] | None = None,
    fast_model_name: str = "fake/fast",
    gen_model_name: str = "fake/gen",
    cost_tracker: CostTracker | None = None,
    seed_docs: bool = True,
    firecrawl_client: FirecrawlClient | None = None,
    provider: LLMProvider = "openrouter",
    max_run_cost_usd: float = 0.50,
) -> AppContainer:
    settings = Settings(
        llm_provider=provider,
        openrouter_api_key=None,
        llm_fast_model=fast_model_name,
        llm_generation_model=gen_model_name,
        qdrant_collection="api_test_docs",
        llm_cache_enabled=True,
        llm_cache_path=tmp_path / "api_cache.sqlite",
        allowed_corpus_root=tmp_path,
        max_run_cost_usd=max_run_cost_usd,
    )
    strategies = load_strategies(Path("configs/strategies.yaml"))
    store = QdrantHybridStore.from_settings(
        settings,
        dense_embedder=_StubDenseEmbedder(),
        sparse_embedder=_StubSparseEmbedder(),
        in_memory=True,
    )
    if seed_docs:
        await store.ensure_collection(recreate=True)
        chunks = [
            DocumentChunk(
                chunk_id=str(uuid5(_TEST_NAMESPACE, "qdrant::0")),
                doc_id="qdrant_hybrid",
                title="Qdrant Hybrid Search Architecture",
                section_path="Qdrant > RRF Fusion",
                text=(
                    "Qdrant combines dense cosine vectors and BM25 sparse vectors with "
                    "server-side Prefetch and Reciprocal Rank Fusion (RRF, default k=60)."
                ),
                chunk_index=0,
                token_count=26,
                metadata={"language": "en"},
            ),
            DocumentChunk(
                chunk_id=str(uuid5(_TEST_NAMESPACE, "fastapi::0")),
                doc_id="fastapi_sse",
                title="FastAPI SSE Streaming",
                section_path="FastAPI > Server-Sent Events",
                text=(
                    "FastAPI streams LangGraph node updates over Server-Sent Events (SSE) "
                    "using EventSourceResponse and chunked HTTP transfer."
                ),
                chunk_index=0,
                token_count=22,
                metadata={"language": "en"},
            ),
        ]
        await store.upsert_chunks(chunks)

    fast_llm = _ScriptedChatModel(
        responses=fast_responses
        or [
            json.dumps(
                {
                    "query_type": "factual",
                    "needs_retrieval": True,
                    "language": "en",
                    "standalone_query": "How does Qdrant hybrid search combine dense and BM25?",
                    "requires_fresh_web_info": False,
                    "reasoning": "Factual technical question answerable from local Qdrant docs.",
                }
            ),
            json.dumps(
                {
                    "verifications": [
                        {
                            "claim": "Qdrant combines dense and BM25 sparse vectors with Prefetch and RRF.",
                            "cited_ids": ["C1"],
                            "verdict": "supported",
                            "explanation": "Directly supported by [C1].",
                        }
                    ]
                }
            ),
        ]
    )
    gen_llm = _ScriptedChatModel(
        responses=gen_responses
        or [
            "Qdrant combines dense cosine vectors and BM25 sparse vectors via server-side Prefetch and RRF (k=60) [C1]."
        ]
    )
    fallback = _StubWebFallback()
    runtime = GraphRuntime(
        settings=settings,
        strategies=strategies,
        models=ModelBundle(
            fast_model=fast_llm,
            fast_model_name=fast_model_name,
            generation_model=gen_llm,
            generation_model_name=gen_model_name,
            provider=provider,
        ),
        store=store,
        reranker=_StubReranker(),
        fallback_retriever=fallback,
        cache=SQLiteLLMCache(tmp_path / "api_cache.sqlite"),
        cost_tracker=cost_tracker or CostTracker(max_cost_usd=1.0),
    )
    workflow = AdaptiveRAGWorkflow(runtime)
    return AppContainer(
        settings=settings,
        strategies=strategies,
        store=store,
        fallback_retriever=fallback,
        workflow=workflow,
        firecrawl_client=firecrawl_client,
        owns_store=True,
        owns_firecrawl=firecrawl_client is not None,
    )


@pytest.mark.asyncio
async def test_root_and_health_ok(tmp_path: Path) -> None:
    container = await _build_test_container(tmp_path)
    app = create_app(container=container)

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://localhost",
        ) as client:
            root_resp = await client.get("/")
            assert root_resp.status_code == 200
            assert root_resp.json()["service"] == "AdaptiveRAG"

            health_resp = await client.get("/v1/health")
            assert health_resp.status_code == 200
            data = health_resp.json()
            assert data["status"] == "ok"
            assert data["llm_configured"] is True
            assert data["qdrant"]["status"] == "ok"
            assert data["qdrant"]["points_count"] == 2
            assert data["web_fallback_providers"] == ["stub_web"]
    finally:
        await container.aclose()


@pytest.mark.asyncio
async def test_health_degraded_and_query_503_when_unconfigured() -> None:
    settings = Settings(
        llm_provider="openrouter",
        openrouter_api_key=None,
        qdrant_collection="broken_collection",
    )
    strategies = load_strategies(Path("configs/strategies.yaml"))
    container = AppContainer(
        settings=settings,
        strategies=strategies,
        store=_FailingStore(),  # type: ignore[arg-type]
        fallback_retriever=_StubWebFallback(),
        workflow=None,
        owns_store=False,
    )
    app = create_app(container=container)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://localhost",
    ) as client:
        health_resp = await client.get("/v1/health")
        assert health_resp.status_code == 200
        body = health_resp.json()
        assert body["status"] == "degraded"
        assert body["qdrant"]["status"] == "unreachable"
        assert body["llm_configured"] is False
        assert "OPENROUTER_API_KEY" in (body["llm_error"] or "")

        query_resp = await client.post("/v1/query", json={"query": "What is RRF?"})
        assert query_resp.status_code == 503
        assert "OPENROUTER_API_KEY" in query_resp.json()["detail"]


@pytest.mark.asyncio
async def test_a_collection_indexed_with_another_model_is_refused_until_replaced(
    tmp_path: Path,
) -> None:
    container = await _build_test_container(tmp_path)
    # As if a server configured with another dense model had indexed these passages.
    await container.store.client.set_payload(
        collection_name=container.store.collection_name,
        payload={"dense_model": "old/dense-model"},
        points=models.FilterSelector(filter=models.Filter()),
    )
    fast_llm = container.workflow.runtime.models.fast_model  # type: ignore[union-attr]
    app = create_app(container=container)

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://localhost",
        ) as client:
            health = (await client.get("/v1/health")).json()
            assert health["status"] == "degraded"
            assert health["qdrant"]["status"] == "embedding_mismatch"
            assert health["qdrant"]["indexed_dense_model"] == "old/dense-model"
            assert health["qdrant"]["dense_model"] == "stub-dense-10d"
            assert "old/dense-model" in health["qdrant"]["embedding_mismatch"]

            question = {"query": "How does Qdrant hybrid search combine dense and BM25?"}
            for path in ("/v1/query", "/v1/query/stream"):
                resp = await client.post(path, json=question)
                assert resp.status_code == 409
                assert "old/dense-model" in resp.json()["detail"]
            assert fast_llm.cursor == 0  # refused before any LLM call

            doc = {"doc_id": "notes", "title": "Notes", "content": "Qdrant hybrid search notes."}
            add_resp = await client.post("/v1/ingest", json={"documents": [doc]})
            assert add_resp.status_code == 409
            upload_resp = await client.post(
                "/v1/ingest/upload",
                files={
                    "files": (
                        "notes.md",
                        b"# Notes\n\nQdrant hybrid search notes.",
                        "text/markdown",
                    )
                },
            )
            assert upload_resp.status_code == 409
            assert "old/dense-model" in upload_resp.json()["detail"]
            assert (await client.get("/v1/health")).json()["qdrant"]["points_count"] == 2

            replace_resp = await client.post(
                "/v1/ingest", json={"documents": [doc], "recreate_collection": True}
            )
            assert replace_resp.status_code == 200
            health = (await client.get("/v1/health")).json()
            assert health["status"] == "ok"
            assert health["qdrant"]["indexed_dense_model"] == "stub-dense-10d"
            assert health["qdrant"]["points_count"] == replace_resp.json()["points_upserted"]
    finally:
        await container.aclose()


@pytest.mark.asyncio
async def test_a_qdrant_outage_still_reaches_the_stream_as_an_error_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The up-front model check steps aside on other errors, so the run reports them as before."""
    container = await _build_test_container(tmp_path)

    async def _refuse_connection(*args: Any, **kwargs: Any) -> bool:
        raise ConnectionError("Qdrant connection refused on localhost:6333")

    monkeypatch.setattr(container.store.client, "collection_exists", _refuse_connection)
    app = create_app(container=container)

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://localhost",
        ) as client:
            resp = await client.post(
                "/v1/query/stream",
                json={"query": "How does Qdrant hybrid search combine dense and BM25?"},
            )
            assert resp.status_code == 200
            event, data = _parse_sse_text(resp.text)[-1]
            assert event == "error"
            assert "connection refused" in data["message"]
    finally:
        await container.aclose()


@pytest.mark.asyncio
async def test_ingest_inline_directory_and_validation_errors(tmp_path: Path) -> None:
    container = await _build_test_container(tmp_path, seed_docs=False)
    app = create_app(container=container)

    # Create a small local Markdown directory for directory-mode ingestion test
    corpus_dir = tmp_path / "mini_corpus"
    corpus_dir.mkdir()
    (corpus_dir / "tr_notes.md").write_text(
        "# Türkçe Vektör Veritabanı Notları\n\n"
        "Hibrit arama yoğun vektörlerle BM25 seyrek vektörlerini RRF ile birleştirir.\n",
        encoding="utf-8",
    )

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://localhost",
        ) as client:
            # 1. Inline ingestion
            inline_resp = await client.post(
                "/v1/ingest",
                json={
                    "documents": [
                        {
                            "doc_id": "inline_en_1",
                            "title": "Inline Hybrid Guide",
                            "content": "# Hybrid\n\nDense and BM25 vectors fused via RRF.",
                            "language": "en",
                        },
                        {
                            "doc_id": "inline_tr_1",
                            "title": "Türkçe Rehber",
                            "content": "# Türkçe\n\nÇapraz kodlayıcı yeniden sıralama hassasiyeti artırır.",
                        },
                    ]
                },
            )
            assert inline_resp.status_code == 200
            inline_data = inline_resp.json()
            assert inline_data["source"] == "inline"
            assert inline_data["documents_processed"] == 2
            assert inline_data["points_upserted"] >= 2
            assert inline_data["total_collection_points"] >= 2

            # 2. Directory ingestion (appends idempotently)
            dir_resp = await client.post(
                "/v1/ingest",
                json={"corpus_dir": str(corpus_dir)},
            )
            assert dir_resp.status_code == 200
            dir_data = dir_resp.json()
            assert dir_data["documents_processed"] == 1
            assert dir_data["total_collection_points"] > inline_data["total_collection_points"]

            # 3. Missing directory returns 400
            missing_resp = await client.post(
                "/v1/ingest",
                json={"corpus_dir": str(tmp_path / "does_not_exist")},
            )
            assert missing_resp.status_code == 400

            # 4. Providing both documents and corpus_dir returns 422
            conflict_resp = await client.post(
                "/v1/ingest",
                json={
                    "corpus_dir": str(corpus_dir),
                    "documents": [
                        {"doc_id": "x", "title": "X", "content": "Hello"},
                    ],
                },
            )
            assert conflict_resp.status_code == 422
    finally:
        await container.aclose()


@pytest.mark.asyncio
async def test_query_batch_full_preset_and_validation(tmp_path: Path) -> None:
    container = await _build_test_container(tmp_path)
    app = create_app(container=container)

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://localhost",
        ) as client:
            # Full AdaptiveRAG
            resp = await client.post(
                "/v1/query",
                json={"query": "How does Qdrant hybrid search work?"},
            )
            assert resp.status_code == 200
            parsed = FinalAnswer.model_validate(resp.json())
            assert parsed.query_type == QueryType.FACTUAL
            assert "[C1]" in parsed.answer
            assert parsed.citation_report is not None
            assert parsed.citation_report.passed is True
            assert len(parsed.trace) >= 5

            # Preset override: naive_dense uses dense-only static top-k and disables agentic gates
            naive_resp = await client.post(
                "/v1/query",
                json={
                    "query": "Qdrant hybrid search dense",
                    "preset": "naive_dense",
                },
            )
            assert naive_resp.status_code == 200
            naive_parsed = FinalAnswer.model_validate(naive_resp.json())
            assert naive_parsed.dynamic_k_decision is not None
            assert "static" in naive_parsed.dynamic_k_decision.cutoff_reason
            assert naive_parsed.citation_report is not None
            assert naive_parsed.citation_report.claim_verifications == []
            trace_by_node = {evt.node: evt.metadata for evt in naive_parsed.trace}
            assert trace_by_node["hybrid_retrieve"]["mode"] == "dense_only"
            assert trace_by_node["validate_citations"]["validation_enabled"] is False

            # Whitespace-only query or oversized chat_history (>50 turns) fails validation (422)
            bad_resp = await client.post("/v1/query", json={"query": "   "})
            assert bad_resp.status_code == 422

            oversized_history = await client.post(
                "/v1/query",
                json={
                    "query": "What is RRF?",
                    "chat_history": [{"role": "user", "content": f"msg {i}"} for i in range(51)],
                },
            )
            assert oversized_history.status_code == 422
    finally:
        await container.aclose()


@pytest.mark.asyncio
async def test_query_budget_exceeded_returns_402(tmp_path: Path) -> None:
    # Real pricing model name so estimated cost > $0.000001 triggers BudgetExceededError
    container = await _build_test_container(
        tmp_path,
        fast_model_name="deepseek/deepseek-v3.2",
        gen_model_name="deepseek/deepseek-v3.2",
    )
    app = create_app(container=container)

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://localhost",
        ) as client:
            resp = await client.post(
                "/v1/query",
                json={
                    "query": "How does Qdrant hybrid search work?",
                    "max_cost_usd": 0.000001,
                },
            )
            assert resp.status_code == 402
            assert "budget" in resp.json()["detail"].lower()
    finally:
        await container.aclose()


@pytest.mark.asyncio
async def test_a_request_cannot_raise_the_servers_spend_cap(tmp_path: Path) -> None:
    """A request's max_cost_usd used to replace MAX_RUN_COST_USD, up to $50 (backlog #7)."""
    container = await _build_test_container(
        tmp_path,
        fast_model_name="deepseek/deepseek-v3.2",
        gen_model_name="deepseek/deepseek-v3.2",
        max_run_cost_usd=0.000001,
    )
    app = create_app(container=container)

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://localhost",
        ) as client:
            resp = await client.post(
                "/v1/query",
                json={"query": "How does Qdrant hybrid search work?", "max_cost_usd": 50},
            )
            assert resp.status_code == 402
            assert "budget" in resp.json()["detail"].lower()
    finally:
        await container.aclose()


@pytest.mark.asyncio
async def test_a_local_model_costs_nothing_whatever_it_is_called(tmp_path: Path) -> None:
    """Prices went by model name, so local models got the fallback rate (backlog #7)."""
    container = await _build_test_container(
        tmp_path,
        provider="ollama",
        fast_model_name="inkling-2-mini",
        gen_model_name="inkling-2-mini",
    )
    assert container.workflow is not None

    try:
        # A tiny cap: phantom prices used to stop local runs with BudgetExceededError.
        answer = await container.workflow.arun(
            "How does Qdrant hybrid search combine dense and BM25?",
            cost_tracker=CostTracker(max_cost_usd=0.000001),
        )
        assert answer.usage.total_tokens > 0
        assert answer.usage.estimated_cost_usd == 0.0
    finally:
        await container.aclose()


@pytest.mark.asyncio
async def test_an_answer_counts_the_tokens_of_llm_calls_that_failed(tmp_path: Path) -> None:
    """Failed structured calls were billed to the budget but left out of the answer (#7)."""
    container = await _build_test_container(
        tmp_path,
        fast_responses=["not json"],
        fast_model_name="deepseek/deepseek-v3.2",
        gen_model_name="deepseek/deepseek-v3.2",
    )
    assert container.workflow is not None
    tracker = CostTracker(max_cost_usd=1.0)

    try:
        answer = await container.workflow.arun(
            "How does Qdrant hybrid search combine dense and BM25?", cost_tracker=tracker
        )
        # The router's two unparseable replies led to the factual fallback.
        assert answer.query_type == QueryType.FACTUAL
        spent = tracker.usage
        assert answer.usage.total_tokens == spent.total_tokens
        assert answer.usage.estimated_cost_usd == pytest.approx(spent.estimated_cost_usd)
    finally:
        await container.aclose()


@pytest.mark.asyncio
async def test_query_stream_sse_emits_node_updates_and_final_answer(tmp_path: Path) -> None:
    container = await _build_test_container(tmp_path)
    app = create_app(container=container)

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://localhost",
        ) as client:
            resp = await client.post(
                "/v1/query/stream",
                json={"query": "How does Qdrant hybrid search work?"},
            )
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            assert "no-cache" in resp.headers.get("cache-control", "")

            events = _parse_sse_text(resp.text)
            event_names = [name for name, _ in events]
            assert "node_update" in event_names
            assert event_names[-1] == "final_answer"

            streamed_nodes = [payload["node"] for name, payload in events if name == "node_update"]
            assert streamed_nodes == [
                "classify_and_plan",
                "hybrid_retrieve",
                "rerank_and_select_k",
                "score_confidence",
                "generate_answer",
                "validate_citations",
            ]

            final_payload = events[-1][1]
            final_answer = FinalAnswer.model_validate(final_payload)
            assert final_answer.citation_report is not None
            assert final_answer.citation_report.passed is True
            assert "[C1]" in final_answer.answer
    finally:
        await container.aclose()


@pytest.mark.asyncio
async def test_query_stream_sse_emits_error_event_on_budget_cap(tmp_path: Path) -> None:
    container = await _build_test_container(
        tmp_path,
        fast_model_name="deepseek/deepseek-v3.2",
        gen_model_name="deepseek/deepseek-v3.2",
    )
    app = create_app(container=container)

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://localhost",
        ) as client:
            resp = await client.post(
                "/v1/query/stream",
                json={
                    "query": "How does Qdrant hybrid search work?",
                    "max_cost_usd": 0.000001,
                },
            )
            assert resp.status_code == 200
            events = _parse_sse_text(resp.text)
            error_events = [payload for name, payload in events if name == "error"]
            assert len(error_events) == 1
            assert error_events[0]["code"] == "budget_exceeded"
    finally:
        await container.aclose()


def test_ui_sse_parser_and_payload_builder() -> None:
    """Verify ui/app.py pure helpers (iter_sse_events, build_query_payload) without Streamlit runtime."""
    ui_path = Path("ui/app.py").resolve()
    spec = importlib.util.spec_from_file_location("adaptiverag_ui_app", ui_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    raw_lines = [
        ": keep-alive comment",
        "event: node_update",
        'data: {"node": "classify_and_plan", "trace": [], "usage": {}}',
        "",
        "event: final_answer",
        'data: {"query": "test", "answer": "ok [C1]"}',
        "",
    ]
    parsed = list(module.iter_sse_events(raw_lines))
    assert len(parsed) == 2
    assert parsed[0][0] == "node_update"
    assert parsed[0][1]["node"] == "classify_and_plan"
    assert parsed[1][0] == "final_answer"
    assert parsed[1][1]["answer"] == "ok [C1]"

    preset_payload = module.build_query_payload(
        "What is RRF?",
        [{"role": "user", "content": "hi"}],
        preset="hybrid_rerank",
        custom_ablation=None,
        max_cost_usd=0.05,
    )
    assert preset_payload["preset"] == "hybrid_rerank"
    assert preset_payload["max_cost_usd"] == 0.05
    assert "ablation" not in preset_payload

    custom_payload = module.build_query_payload(
        "What is RRF?",
        [],
        preset="custom",
        custom_ablation={"retrieval_mode": "dense_only", "enable_rerank": False},
        max_cost_usd=None,
    )
    assert "preset" not in custom_payload
    assert custom_payload["ablation"]["retrieval_mode"] == "dense_only"

    parsed_urls = module.parse_url_lines(
        "  https://docs.qdrant.tech/concepts/hybrid-queries/ \n\n"
        "https://docs.qdrant.tech/concepts/hybrid-queries/\n"
        "https://docs.firecrawl.dev/quickstarts/firecrawl-local  "
    )
    assert parsed_urls == [
        "https://docs.qdrant.tech/concepts/hybrid-queries/",
        "https://docs.firecrawl.dev/quickstarts/firecrawl-local",
    ]

    scrape_payload = module.build_web_ingest_payload(
        mode="scrape",
        urls_text="https://docs.qdrant.tech/concepts/hybrid-queries/",
        only_main_content=True,
        recreate_collection=False,
    )
    assert scrape_payload["urls"] == ["https://docs.qdrant.tech/concepts/hybrid-queries/"]
    assert "crawl_url" not in scrape_payload

    crawl_payload = module.build_web_ingest_payload(
        mode="crawl",
        crawl_url=" https://docs.qdrant.tech/concepts/ ",
        crawl_limit=8,
        crawl_max_depth=2,
    )
    assert crawl_payload["crawl_url"] == "https://docs.qdrant.tech/concepts/"
    assert crawl_payload["crawl_limit"] == 8
    assert crawl_payload["crawl_max_depth"] == 2
    assert module.build_auth_headers("") == {}
    assert module.build_auth_headers("  secret-123  ") == {"X-API-Key": "secret-123"}


@pytest.mark.asyncio
async def test_ingest_firecrawl_urls_crawl_and_error_handling(tmp_path: Path) -> None:
    def mock_firecrawl_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.method == "POST" and request.url.path == "/v2/scrape":
            body = json.loads(request.content.decode("utf-8"))
            url = body["url"]
            if "fail-upstream" in url:
                return httpx.Response(500, json={"success": False, "error": "Upstream 500"})
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        "markdown": (
                            "# Qdrant Hybrid Guide\n\n"
                            "## Reciprocal Rank Fusion\n\n"
                            "Qdrant combines dense vectors and BM25 sparse vectors with RRF."
                        ),
                        "metadata": {
                            "title": "Qdrant Hybrid Guide",
                            "sourceURL": url,
                            "statusCode": 200,
                        },
                    },
                },
            )
        if request.method == "POST" and request.url.path == "/v2/crawl":
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "id": "crawl-1",
                    "url": "http://localhost:3002/v2/crawl/crawl-1",
                },
            )
        if request.method == "GET" and request.url.path == "/v2/crawl/crawl-1":
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "data": [
                        {
                            "markdown": "# FastAPI SSE\n\nFastAPI streams SSE updates cleanly.",
                            "metadata": {
                                "title": "FastAPI SSE Docs",
                                "sourceURL": "https://docs.example.com/fastapi-sse",
                            },
                        },
                        {
                            "markdown": "# LangGraph Cyclic Agents\n\nLangGraph manages bounded state loops.",
                            "metadata": {
                                "title": "LangGraph Docs",
                                "sourceURL": "https://docs.example.com/langgraph",
                            },
                        },
                    ],
                },
            )
        raise AssertionError(f"Unexpected Firecrawl request: {request.method} {request.url}")

    fc_client = FirecrawlClient(
        "http://localhost:3002",
        resolve_dns=False,
        transport=httpx.MockTransport(mock_firecrawl_handler),
    )
    container = await _build_test_container(
        tmp_path,
        seed_docs=False,
        firecrawl_client=fc_client,
    )
    app = create_app(container=container)

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://localhost",
        ) as client:
            # 1. Health endpoint reports Firecrawl Local as ok
            health = await client.get("/v1/health")
            assert health.status_code == 200
            assert health.json()["firecrawl"]["status"] == "ok"

            # 2. Live URL scrape ingestion + idempotency + EvidenceChunk.url propagation
            target_url = "https://docs.qdrant.tech/concepts/hybrid-queries/"
            scrape_resp = await client.post(
                "/v1/ingest",
                json={"urls": [target_url]},
            )
            assert scrape_resp.status_code == 200
            scrape_data = scrape_resp.json()
            assert scrape_data["source"] == "firecrawl_scrape"
            assert scrape_data["documents_processed"] == 1
            assert scrape_data["source_urls"] == [target_url]
            points_after_first = scrape_data["total_collection_points"]
            assert points_after_first >= 1

            # Verify stored chunk carries source_url onto EvidenceChunk.url
            hits = await container.store.search_single(
                "Qdrant hybrid BM25 RRF",
                mode=RetrievalMode.HYBRID,
                fusion_limit=5,
            )
            assert any(h.url == target_url for h in hits)

            # Re-scraping the exact same URL is idempotent (zero duplicate points)
            rescrape_resp = await client.post("/v1/ingest", json={"urls": [target_url]})
            assert rescrape_resp.status_code == 200
            assert rescrape_resp.json()["total_collection_points"] == points_after_first

            # 3. Live docs site crawl ingestion
            crawl_resp = await client.post(
                "/v1/ingest",
                json={
                    "crawl_url": "https://docs.example.com/",
                    "crawl_limit": 5,
                    "crawl_max_depth": 2,
                },
            )
            assert crawl_resp.status_code == 200
            crawl_data = crawl_resp.json()
            assert crawl_data["documents_processed"] == 2
            assert crawl_data["source_urls"] == [
                "https://docs.example.com/fastapi-sse",
                "https://docs.example.com/langgraph",
            ]
            assert crawl_data["total_collection_points"] > points_after_first

            # 4. SSRF URL returns HTTP 422
            ssrf_resp = await client.post(
                "/v1/ingest",
                json={"urls": ["http://169.254.169.254/latest/meta-data/"]},
            )
            assert ssrf_resp.status_code == 422
            assert "SSRF" in ssrf_resp.json()["detail"]

            # 5. Multiple conflicting sources return HTTP 422
            conflict_resp = await client.post(
                "/v1/ingest",
                json={
                    "urls": [target_url],
                    "crawl_url": "https://docs.example.com/",
                },
            )
            assert conflict_resp.status_code == 422

            # 6. Upstream Firecrawl 500 maps cleanly to HTTP 502
            bad_gw_resp = await client.post(
                "/v1/ingest",
                json={"urls": ["https://docs.example.com/fail-upstream"]},
            )
            assert bad_gw_resp.status_code == 502

            # 7. Unreachable Firecrawl Local maps cleanly to HTTP 503
            def _connect_fail(req: httpx.Request) -> httpx.Response:
                raise httpx.ConnectError("Connection refused", request=req)

            down_client = FirecrawlClient(
                "http://localhost:3002",
                resolve_dns=False,
                transport=httpx.MockTransport(_connect_fail),
            )
            container.firecrawl_client = down_client
            try:
                down_resp = await client.post("/v1/ingest", json={"urls": [target_url]})
                assert down_resp.status_code == 503
                assert "firecrawl-local serve" in down_resp.json()["detail"]
            finally:
                await down_client.aclose()
                container.firecrawl_client = fc_client
    finally:
        await container.aclose()


@pytest.mark.asyncio
async def test_documents_list_delete_and_file_upload_endpoints(tmp_path: Path) -> None:
    container = await _build_test_container(tmp_path, seed_docs=False)
    app = create_app(container=container)

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://localhost",
        ) as client:
            # 1. Empty collection returns zero documents cleanly
            empty_resp = await client.get("/v1/documents")
            assert empty_resp.status_code == 200
            empty_data = empty_resp.json()
            assert empty_data["total_documents"] == 0
            assert empty_data["total_chunks"] == 0
            assert empty_data["documents"] == []

            # 2. Multipart file upload (.md + .txt, TR + EN)
            md_bytes = (
                b"# Hybrid Search Architecture\n\n"
                b"Dense cosine search and BM25 sparse vectors merge via Reciprocal Rank Fusion.\n"
            )
            txt_bytes = (
                "Türkçe RAG değerlendirme notları: çapraz kodlayıcı yeniden sıralama "
                "ve güven eşiği doğruluk oranını önemli ölçüde artırır.\n"
            ).encode()

            upload_resp = await client.post(
                "/v1/ingest/upload",
                files=[
                    ("files", ("hybrid_guide.md", md_bytes, "text/markdown")),
                    ("files", ("tr_notlar.txt", txt_bytes, "text/plain")),
                ],
                data={"recreate_collection": "false"},
            )
            assert upload_resp.status_code == 200
            upload_data = upload_resp.json()
            assert upload_data["source"] == "upload"
            assert upload_data["documents_processed"] == 2
            assert upload_data["chunks_created"] >= 2
            total_points = upload_data["total_collection_points"]
            assert total_points == upload_data["chunks_created"]

            # 3. Also ingest a web-metadata inline document so web vs local classification is tested
            web_inline_resp = await client.post(
                "/v1/ingest",
                json={
                    "documents": [
                        {
                            "doc_id": "web_docs_demo",
                            "title": "Firecrawl Docs Demo",
                            "content": "# Firecrawl Local\n\nScrapes web pages into clean Markdown.",
                            "language": "en",
                            "metadata": {
                                "source": "firecrawl_local",
                                "source_url": "https://docs.firecrawl.dev/quickstarts/firecrawl-local",
                            },
                        }
                    ]
                },
            )
            assert web_inline_resp.status_code == 200

            # 4. List documents verifies grouping, source_origin, language, and totals
            list_resp = await client.get("/v1/documents")
            assert list_resp.status_code == 200
            list_data = list_resp.json()
            assert list_data["total_documents"] == 3
            assert list_data["total_chunks"] >= 3
            by_id = {d["doc_id"]: d for d in list_data["documents"]}
            assert by_id["hybrid_guide"]["title"] == "Hybrid Search Architecture"
            assert by_id["hybrid_guide"]["source_origin"] == "local"
            assert by_id["hybrid_guide"]["doc_type"] == "markdown"
            assert by_id["hybrid_guide"]["language"] == "en"
            assert by_id["tr_notlar"]["language"] == "tr"
            assert by_id["tr_notlar"]["doc_type"] == "text"
            assert by_id["web_docs_demo"]["source_origin"] == "web"
            assert (
                by_id["web_docs_demo"]["source_url"]
                == "https://docs.firecrawl.dev/quickstarts/firecrawl-local"
            )

            # 5. Delete an indexed document and confirm points drop accordingly
            del_resp = await client.delete("/v1/documents/hybrid_guide")
            assert del_resp.status_code == 200
            del_data = del_resp.json()
            assert del_data["doc_id"] == "hybrid_guide"
            assert del_data["deleted_chunks"] >= 1
            assert del_data["remaining_collection_points"] == (
                list_data["total_chunks"] - del_data["deleted_chunks"]
            )

            # Deleting the same document again returns HTTP 404
            missing_del = await client.delete("/v1/documents/hybrid_guide")
            assert missing_del.status_code == 404

            # 6. Upload validation: unsupported extension and empty file return HTTP 422
            bad_ext = await client.post(
                "/v1/ingest/upload",
                files=[("files", ("malware.exe", b"MZ\x90\x00", "application/octet-stream"))],
            )
            assert bad_ext.status_code == 422
            assert "Unsupported document extension" in bad_ext.json()["detail"]

            empty_file = await client.post(
                "/v1/ingest/upload",
                files=[("files", ("empty.md", b"", "text/markdown"))],
            )
            assert empty_file.status_code == 422
            assert "empty" in empty_file.json()["detail"].lower()
    finally:
        await container.aclose()
