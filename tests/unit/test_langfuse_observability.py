"""$0 offline unit tests for Langfuse v5 observability, /v1/feedback API, and eval sync."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid5

import httpx
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from adaptiverag.api.dependencies import AppContainer
from adaptiverag.api.main import create_app
from adaptiverag.config import Settings, load_strategies
from adaptiverag.graph import AdaptiveRAGWorkflow, GraphRuntime
from adaptiverag.ingestion.chunker import DocumentChunk
from adaptiverag.llm import CostTracker, ModelBundle, SQLiteLLMCache
from adaptiverag.observability.tracing import (
    LangfuseObservability,
    build_online_quality_scores,
)
from adaptiverag.retrieval.embeddings import SparseVectorData
from adaptiverag.retrieval.qdrant_store import QdrantHybridStore
from adaptiverag.schemas.citation import CitationReport, ClaimVerdict, ClaimVerification
from adaptiverag.schemas.query import QueryType
from adaptiverag.schemas.response import FinalAnswer, NodeTraceEvent, TokenUsage
from adaptiverag.schemas.retrieval import (
    ConfidenceAction,
    ConfidenceAssessment,
    ConfidenceLevel,
    CRAGGrade,
    DynamicKDecision,
    EvidenceChunk,
    SourceType,
)
from eval.langfuse_eval import (
    build_record_evaluations,
    build_summary_evaluations,
    deterministic_dataset_item_id,
    publish_benchmark_report_to_langfuse,
    sync_golden_dataset,
)
from eval.schemas import (
    BenchmarkReport,
    ConfigBenchmarkSummary,
    EvalStratum,
    GoldenQuery,
    IRScores,
    QueryEvalRecord,
    RagasScores,
)

_NS = UUID("33445566-1234-5678-1234-567812345678")


@dataclass
class _FakeSpan:
    id: str
    trace_id: str
    name: str
    as_type: str
    input: Any = None
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    level: str = "DEFAULT"
    status_message: str | None = None
    ended: bool = False
    updates: list[dict[str, Any]] = field(default_factory=list)

    def update(self, **kwargs: Any) -> _FakeSpan:
        self.updates.append(dict(kwargs))
        if "output" in kwargs:
            self.output = kwargs["output"]
        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            self.metadata.update(kwargs["metadata"])
        if "level" in kwargs and kwargs["level"] is not None:
            self.level = str(kwargs["level"])
        if "status_message" in kwargs:
            self.status_message = kwargs["status_message"]
        return self

    def end(self, **kwargs: Any) -> _FakeSpan:
        self.ended = True
        return self


class _FakeDataset:
    def __init__(self, client: _FakeLangfuseClient, name: str, items: list[dict[str, Any]]) -> None:
        self._client = client
        self.name = name
        self.items = [SimpleNamespace(**item) for item in items]

    def run_experiment(self, **kwargs: Any) -> Any:
        return self._client.run_experiment(data=self.items, **kwargs)


class _FakeLangfuseClient:
    """In-memory fake of the Langfuse v5 SDK client for deterministic $0 tests."""

    def __init__(self, *, fail_score_names: set[str] | None = None) -> None:
        self.observations: list[_FakeSpan] = []
        self.scores: list[dict[str, Any]] = []
        self.datasets: dict[str, dict[str, Any]] = {}
        self.dataset_items: dict[str, dict[str, dict[str, Any]]] = {}
        self.experiments: list[dict[str, Any]] = []
        self.flush_count: int = 0
        self._trace_counter = 0
        self._span_counter = 0
        self._fail_score_names = fail_score_names or set()

    def create_trace_id(self, *, seed: str | None = None) -> str:
        self._trace_counter += 1
        return f"{self._trace_counter:032x}"

    def get_trace_url(self, *, trace_id: str | None = None) -> str:
        return f"https://cloud.langfuse.com/project/demo/traces/{trace_id}"

    def start_observation(
        self,
        *,
        trace_context: dict[str, Any] | None = None,
        name: str,
        as_type: str = "span",
        input: Any = None,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
        level: str = "DEFAULT",
        **kwargs: Any,
    ) -> _FakeSpan:
        self._span_counter += 1
        span = _FakeSpan(
            id=f"span_{self._span_counter:08x}",
            trace_id=(trace_context or {}).get("trace_id", ""),
            name=name,
            as_type=as_type,
            input=input,
            output=output,
            metadata=dict(metadata or {}),
            level=level,
        )
        self.observations.append(span)
        return span

    def create_score(self, **kwargs: Any) -> None:
        if kwargs.get("name") in self._fail_score_names:
            raise RuntimeError(f"Simulated score upload failure for {kwargs.get('name')}")
        self.scores.append(dict(kwargs))

    def create_dataset(
        self,
        *,
        name: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {"name": name, "description": description, "metadata": metadata or {}}
        self.datasets[name] = payload
        self.dataset_items.setdefault(name, {})
        return payload

    def create_dataset_item(
        self,
        *,
        dataset_name: str,
        id: str | None = None,
        input: Any = None,
        expected_output: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        items_map = self.dataset_items.setdefault(dataset_name, {})
        item_id = id or f"item_{len(items_map) + 1}"
        item = {
            "id": item_id,
            "dataset_name": dataset_name,
            "input": input,
            "expected_output": expected_output,
            "metadata": metadata or {},
        }
        items_map[item_id] = item
        return item

    def get_dataset(self, name: str) -> _FakeDataset:
        items = list(self.dataset_items.get(name, {}).values())
        return _FakeDataset(self, name, items)

    def run_experiment(
        self,
        *,
        name: str,
        run_name: str | None = None,
        description: str | None = None,
        data: list[Any],
        task: Any,
        evaluators: list[Any] | None = None,
        run_evaluators: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        item_results: list[dict[str, Any]] = []
        for item in data:
            output = task(item=item)
            item_input = item["input"] if isinstance(item, dict) else item.input
            expected = item["expected_output"] if isinstance(item, dict) else item.expected_output
            item_meta = item["metadata"] if isinstance(item, dict) else item.metadata
            evals: list[Any] = []
            for ev in evaluators or []:
                res = ev(
                    input=item_input,
                    output=output,
                    expected_output=expected,
                    metadata=item_meta,
                )
                if isinstance(res, list):
                    evals.extend(res)
                elif res is not None:
                    evals.append(res)
            item_results.append({"input": item_input, "output": output, "evaluations": evals})

        run_evals: list[Any] = []
        for rev in run_evaluators or []:
            res = rev(item_results=item_results)
            if isinstance(res, list):
                run_evals.extend(res)
            elif res is not None:
                run_evals.append(res)

        record = {
            "name": name,
            "run_name": run_name,
            "description": description,
            "metadata": metadata or {},
            "item_results": item_results,
            "run_evaluations": run_evals,
        }
        self.experiments.append(record)
        return SimpleNamespace(
            dataset_run_url=f"https://cloud.langfuse.com/project/demo/datasets/runs/{name}"
        )

    def flush(self) -> None:
        self.flush_count += 1


def _sample_final_answer() -> FinalAnswer:
    return FinalAnswer(
        query="How does Qdrant hybrid search work?",
        answer="Qdrant fuses dense vectors and BM25 sparse vectors with RRF [C1].",
        query_type=QueryType.FACTUAL,
        language="en",
        evidence_chunks=[
            EvidenceChunk(
                citation_id="C1",
                chunk_id="chunk_1",
                text="Qdrant fuses dense and BM25 sparse vectors with Reciprocal Rank Fusion.",
                source_type=SourceType.LOCAL,
                doc_id="qdrant_hybrid",
                title="Qdrant Hybrid Search",
                section_path="Hybrid > RRF",
                rerank_score=0.94,
                fusion_score=0.032,
                token_count=20,
            )
        ],
        dynamic_k_decision=DynamicKDecision(
            candidate_count=8,
            selected_count=1,
            cutoff_reason="elbow_gap",
            top_score=0.94,
            cutoff_score=0.94,
            total_context_tokens=20,
            dropped_chunk_ids=["chunk_2"],
        ),
        confidence=ConfidenceAssessment(
            top1_score=0.94,
            mean_topk_score=0.94,
            subquery_coverage=1.0,
            deterministic_score=0.952,
            crag_grade=CRAGGrade.SUFFICIENT,
            final_score=0.952,
            level=ConfidenceLevel.HIGH,
            recommended_action=ConfidenceAction.PROCEED,
            reasoning="Strong local rerank confidence.",
        ),
        citation_report=CitationReport(
            total_citations_found=1,
            valid_citation_ids=["C1"],
            invalid_citation_ids=[],
            uncited_factual_sentences=[],
            claim_verifications=[
                ClaimVerification(
                    claim="Qdrant fuses dense vectors and BM25 sparse vectors with RRF.",
                    cited_ids=["C1"],
                    verdict=ClaimVerdict.SUPPORTED,
                    explanation="Directly supported by [C1].",
                )
            ],
            citation_precision=1.0,
            claim_support_rate=1.0,
            passed=True,
            regeneration_triggered=False,
        ),
        fallback_used=False,
        abstained=False,
        rewrite_attempts=0,
        regeneration_attempts=0,
        total_latency_ms=412.5,
        usage=TokenUsage(
            prompt_tokens=300,
            completion_tokens=45,
            total_tokens=345,
            estimated_cost_usd=0.00042,
            cache_hits=2,
        ),
        trace=[
            NodeTraceEvent(
                node="classify_and_plan",
                latency_ms=80.0,
                summary="factual (en)",
            )
        ],
    )


def test_online_quality_scores_extraction() -> None:
    answer = _sample_final_answer()
    scores = {s.name: s for s in build_online_quality_scores(answer)}

    assert scores["crag_confidence"].value == pytest.approx(0.952)
    assert scores["crag_confidence"].data_type == "NUMERIC"
    assert scores["crag_level"].value == "high"
    assert scores["crag_level"].data_type == "CATEGORICAL"
    assert scores["citation_precision"].value == 1.0
    assert scores["claim_support_rate"].value == 1.0
    assert scores["citation_passed"].value == 1.0
    assert scores["citation_passed"].data_type == "BOOLEAN"
    assert scores["fabricated_citations"].value == 0.0
    assert scores["selected_k"].value == 1.0
    assert scores["fallback_used"].value == 0.0
    assert scores["abstained"].value == 0.0
    assert scores["estimated_cost_usd"].value == pytest.approx(0.00042)
    assert scores["latency_ms"].value == pytest.approx(412.5)
    assert scores["cache_hits"].value == 2.0


def test_observability_root_trace_domain_spans_and_fail_open_scores() -> None:
    fake_client = _FakeLangfuseClient(fail_score_names={"crag_level"})
    obs = LangfuseObservability(
        enabled=True,
        host="https://cloud.langfuse.com",
        environment="test",
        client=fake_client,
    )

    handle = obs.start_trace(
        query="How does Qdrant hybrid search work?",
        session_id="sess_123",
        user_id="user_42",
        preset_name="full",
        tags=["mode:batch"],
    )
    assert handle.langfuse_active is True
    assert len(handle.trace_id) == 32
    assert handle.trace_url == f"https://cloud.langfuse.com/project/demo/traces/{handle.trace_id}"
    assert handle.langfuse_metadata["langfuse_session_id"] == "sess_123"
    assert handle.langfuse_metadata["langfuse_user_id"] == "user_42"
    assert "preset:full" in handle.langfuse_metadata["langfuse_tags"]

    obs.finish_trace(handle, answer=_sample_final_answer())

    obs_by_name = {o.name: o for o in fake_client.observations}
    assert "adaptiverag.query" in obs_by_name
    assert obs_by_name["adaptiverag.query"].ended is True
    assert obs_by_name["adaptiverag.query"].as_type == "agent"
    assert "retrieval.hybrid_dynamic_k" in obs_by_name
    assert obs_by_name["retrieval.hybrid_dynamic_k"].as_type == "retriever"
    assert "llm.sqlite_cache_hits" in obs_by_name
    assert "verification.crag_confidence" in obs_by_name
    assert obs_by_name["verification.crag_confidence"].as_type == "evaluator"
    assert "verification.citation_audit" in obs_by_name
    assert obs_by_name["verification.citation_audit"].as_type == "guardrail"

    # Fail-open: crag_level failed, but all other 11 online scores still succeeded
    recorded_names = {s["name"] for s in fake_client.scores}
    assert "crag_level" not in recorded_names
    assert {"crag_confidence", "citation_precision", "claim_support_rate", "cache_hits"}.issubset(
        recorded_names
    )


def test_observability_health_modes_and_feedback_graceful_degradation() -> None:
    disabled = LangfuseObservability(enabled=False)
    assert disabled.health_summary()["mode"] == "disabled"
    handle = disabled.start_trace(query="hi")
    assert len(handle.trace_id) == 32
    assert handle.trace_url is None
    assert handle.langfuse_active is False

    fb_disabled = disabled.record_user_feedback(
        trace_id=handle.trace_id,
        score=1.0,
        comment="Great citation accuracy",
        session_id="sess_1",
    )
    assert fb_disabled.persisted_to_langfuse is False
    assert fb_disabled.langfuse_error is None

    misconfigured = LangfuseObservability(enabled=True, public_key="", secret_key=None)
    assert misconfigured.health_summary()["mode"] == "misconfigured"

    self_hosted = LangfuseObservability(
        enabled=True,
        host="http://localhost:3000",
        public_key="pk-lf-local",
        secret_key="sk-lf-local",
    )
    assert self_hosted.health_summary()["mode"] == "self_hosted"

    fake_client = _FakeLangfuseClient()
    cloud_obs = LangfuseObservability(
        enabled=True,
        host="https://cloud.langfuse.com",
        client=fake_client,
    )
    assert cloud_obs.health_summary()["mode"] == "cloud"

    fb_ok = cloud_obs.record_user_feedback(
        trace_id="0" * 31 + "1",
        score=0.0,
        comment="Missed second source",
        session_id="sess_99",
        query="Compare Qdrant and pgvector",
    )
    assert fb_ok.persisted_to_langfuse is True
    assert fake_client.flush_count == 1
    assert len(fake_client.scores) == 1
    assert fake_client.scores[0]["name"] == "user_feedback"
    assert fake_client.scores[0]["value"] == 0.0
    assert fake_client.scores[0]["comment"] == "Missed second source"


class _MiniDense:
    model_name = "stub-dense"
    dimension = 4

    def _enc(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._enc(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._enc(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


class _MiniSparse:
    model_name = "stub-bm25"

    def _enc(self, text: str) -> SparseVectorData:
        return SparseVectorData(indices=[1], values=[1.0])

    def embed_documents(self, texts: list[str]) -> list[SparseVectorData]:
        return [self._enc(t) for t in texts]

    def embed_query(self, text: str) -> SparseVectorData:
        return self._enc(text)

    async def aembed_documents(self, texts: list[str]) -> list[SparseVectorData]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> SparseVectorData:
        return self.embed_query(text)


class _MiniReranker:
    model_name = "stub-reranker"

    def rerank(
        self, query: str, chunks: list[EvidenceChunk], *, top_n: int | None = None
    ) -> list[EvidenceChunk]:
        out = [c.model_copy(update={"rerank_score": 0.92, "rerank_raw_score": 2.4}) for c in chunks]
        return out[:top_n] if top_n else out

    async def arerank(
        self, query: str, chunks: list[EvidenceChunk], *, top_n: int | None = None
    ) -> list[EvidenceChunk]:
        return self.rerank(query, chunks, top_n=top_n)


class _MiniFallback:
    enabled = True

    @property
    def provider_names(self) -> list[str]:
        return ["stub_web"]

    async def asearch(self, query: str, **kwargs: Any) -> list[EvidenceChunk]:
        return []


class _MiniChat(BaseChatModel):
    responses: list[str] = Field(default_factory=list)
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return "mini-chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        idx = min(self.cursor, len(self.responses) - 1)
        self.cursor += 1
        msg = AIMessage(
            content=self.responses[idx],
            usage_metadata={"input_tokens": 100, "output_tokens": 40, "total_tokens": 140},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.mark.asyncio
async def test_workflow_and_api_end_to_end_langfuse_tracing_and_feedback(tmp_path: Path) -> None:
    fake_client = _FakeLangfuseClient()
    obs = LangfuseObservability(
        enabled=True,
        host="https://cloud.langfuse.com",
        environment="ci-test",
        client=fake_client,
    )
    settings = Settings(
        llm_provider="openrouter",
        qdrant_collection="lf_test_docs",
        llm_cache_enabled=True,
        llm_cache_path=tmp_path / "lf_cache.sqlite",
        langfuse_enabled=True,
    )
    strategies = load_strategies(Path("configs/strategies.yaml"))
    store = QdrantHybridStore.from_settings(
        settings,
        dense_embedder=_MiniDense(),  # type: ignore[arg-type]
        sparse_embedder=_MiniSparse(),  # type: ignore[arg-type]
        in_memory=True,
    )
    await store.ensure_collection(recreate=True)
    await store.upsert_chunks(
        [
            DocumentChunk(
                chunk_id=str(uuid5(_NS, "doc::0")),
                doc_id="qdrant_hybrid",
                title="Qdrant Hybrid Search",
                section_path="RRF",
                text="Qdrant fuses dense vectors and BM25 sparse vectors with RRF.",
                chunk_index=0,
                token_count=16,
                metadata={"language": "en"},
            )
        ]
    )

    router_json = json.dumps(
        {
            "query_type": "factual",
            "needs_retrieval": True,
            "language": "en",
            "standalone_query": "How does Qdrant hybrid search work?",
            "requires_fresh_web_info": False,
            "reasoning": "Factual Qdrant question.",
        }
    )
    entail_json = json.dumps(
        {
            "verifications": [
                {
                    "claim": "Qdrant fuses dense and BM25 sparse vectors with RRF.",
                    "cited_ids": ["C1"],
                    "verdict": "supported",
                    "explanation": "Supported by [C1].",
                }
            ]
        }
    )

    fast_llm = _MiniChat(responses=[router_json, entail_json, router_json, entail_json])
    gen_llm = _MiniChat(
        responses=[
            "Qdrant fuses dense and BM25 sparse vectors with RRF [C1].",
            "Qdrant fuses dense and BM25 sparse vectors with RRF [C1].",
        ]
    )
    runtime = GraphRuntime(
        settings=settings,
        strategies=strategies,
        models=ModelBundle(
            fast_model=fast_llm,
            fast_model_name="fake/fast",
            generation_model=gen_llm,
            generation_model_name="fake/gen",
        ),
        store=store,
        reranker=_MiniReranker(),  # type: ignore[arg-type]
        fallback_retriever=_MiniFallback(),  # type: ignore[arg-type]
        cache=SQLiteLLMCache(tmp_path / "lf_cache.sqlite"),
        cost_tracker=CostTracker(max_cost_usd=1.0),
        observability=obs,
    )
    workflow = AdaptiveRAGWorkflow(runtime)
    container = AppContainer(
        settings=settings,
        strategies=strategies,
        store=store,
        fallback_retriever=_MiniFallback(),  # type: ignore[arg-type]
        workflow=workflow,
        observability=obs,
        owns_store=True,
        owns_firecrawl=False,
    )
    app = create_app(container=container)

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            # 1. Health reports Langfuse mode=cloud & configured=True
            health = await client.get("/v1/health")
            assert health.status_code == 200
            h_body = health.json()
            assert h_body["langfuse_enabled"] is True
            assert h_body["langfuse"]["mode"] == "cloud"
            assert h_body["langfuse"]["configured"] is True

            # 2. Batch query returns trace_id & trace_url and publishes online scores
            q_resp = await client.post(
                "/v1/query",
                json={
                    "query": "How does Qdrant hybrid search work?",
                    "preset": "full",
                    "session_id": "sess_portfolio_demo",
                },
            )
            assert q_resp.status_code == 200
            ans = FinalAnswer.model_validate(q_resp.json())
            assert ans.trace_id is not None and len(ans.trace_id) == 32
            assert ans.trace_url is not None and ans.trace_id in ans.trace_url

            # 3. Streaming query (2nd call hits SQLite cache -> verifies cache-hit span too!)
            s_resp = await client.post(
                "/v1/query/stream",
                json={
                    "query": "How does Qdrant hybrid search work?",
                    "preset": "full",
                    "session_id": "sess_portfolio_demo",
                },
            )
            assert s_resp.status_code == 200
            assert "final_answer" in s_resp.text
            assert "llm.sqlite_cache_hits" in {o.name for o in fake_client.observations}

            # 4. Human-in-the-loop feedback endpoint attaches score to exact trace_id
            fb_resp = await client.post(
                "/v1/feedback",
                json={
                    "trace_id": ans.trace_id,
                    "score": 1.0,
                    "comment": "Accurate and well cited",
                    "session_id": "sess_portfolio_demo",
                    "query": ans.query,
                },
            )
            assert fb_resp.status_code == 200
            fb_body = fb_resp.json()
            assert fb_body["status"] == "recorded"
            assert fb_body["persisted_to_langfuse"] is True
            assert any(
                s["name"] == "user_feedback" and s["trace_id"] == ans.trace_id and s["value"] == 1.0
                for s in fake_client.scores
            )

            # 5. Validation rejects invalid score or blank trace_id
            bad_fb = await client.post(
                "/v1/feedback",
                json={"trace_id": "short", "score": 1.5},
            )
            assert bad_fb.status_code == 422
    finally:
        await container.aclose()
        assert fake_client.flush_count >= 2


def test_langfuse_dataset_sync_and_experiment_publish() -> None:
    fake_client = _FakeLangfuseClient()
    queries = [
        GoldenQuery(
            id="fact_en_01",
            query="What is Qdrant's default ef_construct?",
            language="en",
            stratum=EvalStratum.FACTUAL,
            expected_query_type=QueryType.FACTUAL,
            expected_doc_ids=["01_qdrant_hybrid"],
            reference_answer="Default ef_construct is 100.",
        ),
        GoldenQuery(
            id="conv_tr_01",
            query="Merhaba, bana nasıl yardımcı olabilirsin?",
            language="tr",
            stratum=EvalStratum.CONVERSATIONAL,
            expected_query_type=QueryType.CONVERSATIONAL,
            needs_retrieval=False,
            reference_answer="AdaptiveRAG teknik belgelerinizde kaynaklı yanıtlar verir.",
        ),
    ]

    # Deterministic IDs guarantee idempotent re-syncs
    id1 = deterministic_dataset_item_id("adaptiverag-golden-set", "fact_en_01")
    id2 = deterministic_dataset_item_id("adaptiverag-golden-set", "fact_en_01")
    assert id1 == id2

    sync1 = sync_golden_dataset(fake_client, queries, dataset_name="adaptiverag-golden-set")
    sync2 = sync_golden_dataset(fake_client, queries, dataset_name="adaptiverag-golden-set")
    assert sync1.items_synced == 2
    assert sync2.item_ids == sync1.item_ids
    assert len(fake_client.dataset_items["adaptiverag-golden-set"]) == 2

    records = [
        QueryEvalRecord(
            query_id="fact_en_01",
            stratum=EvalStratum.FACTUAL,
            language="en",
            config_name="full",
            predicted_query_type=QueryType.FACTUAL,
            routing_correct=True,
            retrieved_doc_ids=["01_qdrant_hybrid"],
            selected_k=2,
            context_tokens=320,
            ir_scores=IRScores(
                k_eval=2,
                recall_at_k=1.0,
                precision_at_k=0.5,
                mrr_at_k=1.0,
                ndcg_at_k=1.0,
                hit_at_k=True,
            ),
            confidence_score=0.93,
            confidence_level="high",
            citation_precision=1.0,
            claim_support_rate=1.0,
            citation_passed=True,
            latency_ms=480.0,
            estimated_cost_usd=0.0005,
            ragas=RagasScores(
                faithfulness=0.99,
                answer_relevancy=0.91,
                context_precision=0.98,
                context_recall=1.0,
            ),
            answer_preview="Default ef_construct is 100 [C1].",
        ),
        QueryEvalRecord(
            query_id="conv_tr_01",
            stratum=EvalStratum.CONVERSATIONAL,
            language="tr",
            config_name="full",
            predicted_query_type=QueryType.CONVERSATIONAL,
            routing_correct=True,
            selected_k=0,
            context_tokens=0,
            latency_ms=110.0,
            estimated_cost_usd=0.0001,
            answer_preview="Merhaba! Teknik sorularınızı yanıtlayabilirim.",
        ),
    ]
    summary = ConfigBenchmarkSummary(
        config_name="full",
        mode="live",
        query_count=2,
        retrieval_query_count=1,
        mean_recall=1.0,
        mean_precision=0.5,
        mean_mrr=1.0,
        mean_ndcg=1.0,
        hit_rate=1.0,
        routing_accuracy=1.0,
        mean_citation_precision=1.0,
        mean_claim_support_rate=1.0,
        citation_pass_rate=1.0,
        avg_selected_k=1.0,
        avg_context_tokens=160.0,
        mean_latency_ms=295.0,
        latency_p50_ms=295.0,
        latency_p95_ms=480.0,
        total_cost_usd=0.0006,
        ragas_faithfulness=0.99,
        ragas_answer_relevancy=0.91,
        ragas_context_precision=0.98,
        ragas_context_recall=1.0,
    )
    report = BenchmarkReport(
        run_id="live_test001",
        timestamp_utc="2026-11-24T10:00:00+00:00",
        mode="live",
        dataset_source="data/eval/golden_set.jsonl",
        total_queries=2,
        configs=["full"],
        summaries=[summary],
        records=records,
    )

    assert len(build_record_evaluations(records[0])) >= 14
    assert len(build_summary_evaluations(summary)) >= 14

    publish_res = publish_benchmark_report_to_langfuse(
        fake_client,
        report,
        queries,
        dataset_name="adaptiverag-golden-set",
        experiment_prefix="v0.1-ablation",
    )
    assert publish_res.items_synced == 2
    assert len(publish_res.runs) == 1
    assert publish_res.runs[0].config_name == "full"
    assert publish_res.runs[0].items_evaluated == 2
    assert fake_client.flush_count == 1
    assert len(fake_client.experiments) == 1

    exp_payload = fake_client.experiments[0]
    assert exp_payload["run_name"] == "v0.1-ablation::full"
    assert len(exp_payload["item_results"]) == 2
    run_eval_names = {ev.name for ev in exp_payload["run_evaluations"]}
    assert {"avg_mrr", "ragas_faithfulness", "total_cost_usd", "routing_accuracy"}.issubset(
        run_eval_names
    )
