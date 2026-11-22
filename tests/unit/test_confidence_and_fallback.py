"""Unit tests for Multi-Signal Confidence + CRAG Grader and Web Fallback Retriever."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, SecretStr

from adaptiverag.config import ConfidenceConfig, Settings
from adaptiverag.llm import BudgetExceededError, CostTracker, SQLiteLLMCache
from adaptiverag.retrieval import (
    WebFallbackRetriever,
    WebSearchResult,
)
from adaptiverag.schemas import (
    ConfidenceAction,
    ConfidenceLevel,
    CRAGGrade,
    EvidenceChunk,
    SourceType,
)
from adaptiverag.verification import (
    assess_retrieval_confidence,
    compute_deterministic_confidence,
)


class _ScriptedModel(BaseChatModel):
    responses: list[str] = Field(default_factory=list)
    call_count: int = 0
    prompt_tokens: int = 120
    completion_tokens: int = 40

    @property
    def _llm_type(self) -> str:
        return "scripted-crag"

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


class _StubProvider:
    def __init__(
        self,
        name: str,
        *,
        results: list[WebSearchResult] | None = None,
        error: Exception | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        self._name = name
        self._results = results or []
        self._error = error
        self._delay_seconds = delay_seconds
        self.calls: list[str] = []

    @property
    def provider_name(self) -> str:
        return self._name

    def search(self, query: str, *, max_results: int = 4) -> list[WebSearchResult]:
        self.calls.append(query)
        if self._delay_seconds > 0:
            time.sleep(self._delay_seconds)
        if self._error is not None:
            raise self._error
        return self._results[:max_results]


def _chunk(
    cid: str,
    score: float,
    *,
    matched: list[str] | None = None,
    source_type: SourceType = SourceType.LOCAL,
    text: str = "Qdrant HNSW uses ef_construct=100 by default.",
) -> EvidenceChunk:
    return EvidenceChunk(
        citation_id="C1" if source_type == SourceType.LOCAL else "W1",
        chunk_id=cid,
        doc_id="doc_qdrant",
        title="Qdrant Docs",
        section_path="HNSW",
        text=text,
        source_type=source_type,
        rerank_score=score,
        matched_sub_queries=matched if matched is not None else ["q1"],
        token_count=120,
    )


def test_deterministic_confidence_formula_and_subquery_coverage() -> None:
    cfg = ConfidenceConfig()  # 0.50*top1 + 0.30*mean + 0.20*coverage
    chunks = [
        _chunk("c1", 0.80, matched=["qdrant hnsw"]),
        _chunk("c2", 0.60, matched=["qdrant hnsw", "vllm pagedattention"]),
    ]
    sub_queries = ["Qdrant HNSW", "vLLM PagedAttention", "PostgreSQL MVCC", "RAGAS Faithfulness"]

    signals = compute_deterministic_confidence(chunks, sub_queries, cfg)

    assert signals.top1_score == pytest.approx(0.80)
    assert signals.mean_topk_score == pytest.approx(0.70)
    assert signals.subquery_coverage == pytest.approx(0.50)  # 2 of 4 covered
    # 0.50*0.80 + 0.30*0.70 + 0.20*0.50 = 0.40 + 0.21 + 0.10 = 0.71
    assert signals.deterministic_score == pytest.approx(0.71)

    empty_signals = compute_deterministic_confidence([], sub_queries, cfg)
    assert empty_signals.deterministic_score == 0.0


async def test_confidence_high_and_low_fast_paths_make_zero_llm_calls() -> None:
    cfg = ConfidenceConfig()  # high=0.72, low=0.38
    model = _ScriptedModel(
        responses=['{"grade": "sufficient", "missing_aspects": [], "reasoning": "unused"}']
    )

    # HIGH fast path: 0.5*0.92 + 0.3*0.86 + 0.2*1.0 = 0.918 >= 0.72
    high_chunks = [_chunk("h1", 0.92, matched=["q1"]), _chunk("h2", 0.80, matched=["q1"])]
    assessment, usage = await assess_retrieval_confidence(
        query="Qdrant ef_construct nedir?",
        selected_chunks=high_chunks,
        sub_queries=["q1"],
        config=cfg,
        fast_model=model,
        fast_model_name="qwen/qwen3-30b-a3b-instruct-2507",
    )
    assert assessment.level == ConfidenceLevel.HIGH
    assert assessment.recommended_action == ConfidenceAction.PROCEED
    assert assessment.crag_grade is None
    assert model.call_count == 0
    assert usage.total_tokens == 0

    # LOW fast path: 0.5*0.30 + 0.3*0.25 + 0.2*0.0 = 0.225 < 0.38
    low_chunks = [_chunk("l1", 0.30, matched=[]), _chunk("l2", 0.20, matched=[])]
    low_first, _ = await assess_retrieval_confidence(
        query="Bilinmeyen konu",
        selected_chunks=low_chunks,
        sub_queries=["q_missing"],
        config=cfg,
        rewrite_attempts=0,
        web_fallback_enabled=True,
        fast_model=model,
    )
    assert low_first.level == ConfidenceLevel.LOW
    assert low_first.recommended_action == ConfidenceAction.REWRITE

    low_after_rewrite, _ = await assess_retrieval_confidence(
        query="Bilinmeyen konu",
        selected_chunks=low_chunks,
        sub_queries=["q_missing"],
        config=cfg,
        rewrite_attempts=1,
        web_fallback_enabled=True,
        fast_model=model,
    )
    assert low_after_rewrite.recommended_action == ConfidenceAction.WEB_FALLBACK

    low_no_web, _ = await assess_retrieval_confidence(
        query="Bilinmeyen konu",
        selected_chunks=[],
        sub_queries=["q_missing"],
        config=cfg,
        rewrite_attempts=1,
        web_fallback_enabled=False,
        fast_model=model,
    )
    assert low_no_web.recommended_action == ConfidenceAction.ABSTAIN
    assert model.call_count == 0


async def test_confidence_ambiguous_band_invokes_crag_grader_and_caches(tmp_path: Path) -> None:
    cfg = ConfidenceConfig()
    cache = SQLiteLLMCache(tmp_path / "crag_cache.sqlite")
    # Ambiguous score: 0.5*0.60 + 0.3*0.55 + 0.2*0.50 = 0.565 in [0.38, 0.72)
    ambiguous_chunks = [
        _chunk("a1", 0.60, matched=["qdrant"]),
        _chunk("a2", 0.50, matched=["qdrant"]),
    ]

    # 1. CRAG SUFFICIENT upgrades to HIGH / PROCEED + caches repeat call
    sufficient_model = _ScriptedModel(
        responses=[
            '{"grade": "sufficient", "missing_aspects": [], "reasoning": "Kanıt soruyu doğrudan karşılıyor."}'
        ]
    )
    res1, usage1 = await assess_retrieval_confidence(
        query="Qdrant ile vLLM karşılaştırması",
        selected_chunks=ambiguous_chunks,
        sub_queries=["qdrant", "vllm"],
        config=cfg,
        fast_model=sufficient_model,
        fast_model_name="qwen/qwen3-30b-a3b-instruct-2507",
        cache=cache,
    )
    res2, usage2 = await assess_retrieval_confidence(
        query="Qdrant ile vLLM karşılaştırması",
        selected_chunks=ambiguous_chunks,
        sub_queries=["qdrant", "vllm"],
        config=cfg,
        fast_model=sufficient_model,
        fast_model_name="qwen/qwen3-30b-a3b-instruct-2507",
        cache=cache,
    )
    assert res1.crag_grade == CRAGGrade.SUFFICIENT
    assert res1.level == ConfidenceLevel.HIGH
    assert res1.recommended_action == ConfidenceAction.PROCEED
    assert res1.final_score >= cfg.high_threshold
    assert usage1.total_tokens == 160
    assert sufficient_model.call_count == 1
    assert usage2.cache_hits == 1
    assert res2.crag_grade == CRAGGrade.SUFFICIENT

    # 2. CRAG PARTIAL triggers REWRITE -> WEB_FALLBACK -> PROCEED when web disabled
    partial_model = _ScriptedModel(
        responses=[
            '{"grade": "partial", "missing_aspects": ["vLLM memory numbers"], "reasoning": "Only Qdrant is covered."}'
        ]
    )
    partial_rewrite, _ = await assess_retrieval_confidence(
        query="Compare Qdrant vs vLLM",
        selected_chunks=ambiguous_chunks,
        sub_queries=["qdrant", "vllm"],
        config=cfg,
        rewrite_attempts=0,
        web_fallback_enabled=True,
        fast_model=partial_model,
        fast_model_name="qwen/qwen3-30b-a3b-instruct-2507",
    )
    assert partial_rewrite.level == ConfidenceLevel.MEDIUM
    assert partial_rewrite.recommended_action == ConfidenceAction.REWRITE
    assert "vLLM memory numbers" in partial_rewrite.reasoning

    partial_no_web, _ = await assess_retrieval_confidence(
        query="Compare Qdrant vs vLLM",
        selected_chunks=ambiguous_chunks,
        sub_queries=["qdrant", "vllm"],
        config=cfg,
        rewrite_attempts=1,
        web_fallback_enabled=False,
        fast_model=partial_model,
        fast_model_name="qwen/qwen3-30b-a3b-instruct-2507",
    )
    assert partial_no_web.recommended_action == ConfidenceAction.PROCEED

    # 3. CRAG INSUFFICIENT downgrades to LOW / ABSTAIN when rewrites & web disabled
    insuff_model = _ScriptedModel(
        responses=[
            '{"grade": "insufficient", "missing_aspects": ["everything"], "reasoning": "Off-topic passages."}'
        ]
    )
    insuff_res, _ = await assess_retrieval_confidence(
        query="Compare Qdrant vs vLLM",
        selected_chunks=ambiguous_chunks,
        sub_queries=["qdrant", "vllm"],
        config=cfg,
        rewrite_attempts=1,
        web_fallback_enabled=False,
        fast_model=insuff_model,
        fast_model_name="qwen/qwen3-30b-a3b-instruct-2507",
    )
    assert insuff_res.crag_grade == CRAGGrade.INSUFFICIENT
    assert insuff_res.level == ConfidenceLevel.LOW
    assert insuff_res.final_score < cfg.low_threshold
    assert insuff_res.recommended_action == ConfidenceAction.ABSTAIN


async def test_confidence_freshness_override_and_budget_error_propagation() -> None:
    cfg = ConfidenceConfig()
    high_local = [_chunk("h1", 0.95, matched=["q1"], source_type=SourceType.LOCAL)]

    # Requires fresh web info overrides high static local score -> WEB_FALLBACK
    fresh_res, _ = await assess_retrieval_confidence(
        query="What changed in Qdrant latest release this week?",
        selected_chunks=high_local,
        sub_queries=["q1"],
        config=cfg,
        requires_fresh_web_info=True,
        web_fallback_enabled=True,
    )
    assert fresh_res.recommended_action == ConfidenceAction.WEB_FALLBACK

    # Once web chunk is already present, proceed normally
    high_web = [_chunk("w1", 0.90, matched=["q1"], source_type=SourceType.WEB)]
    after_web_res, _ = await assess_retrieval_confidence(
        query="What changed in Qdrant latest release this week?",
        selected_chunks=high_web,
        sub_queries=["q1"],
        config=cfg,
        requires_fresh_web_info=True,
        web_fallback_enabled=True,
    )
    assert after_web_res.recommended_action == ConfidenceAction.PROCEED

    # BudgetExceededError must propagate rather than being swallowed
    ambiguous = [_chunk("a1", 0.55, matched=["q1"])]
    tracker = CostTracker(max_cost_usd=0.000001)
    expensive_model = _ScriptedModel(
        responses=['{"grade": "sufficient", "missing_aspects": [], "reasoning": "ok"}'],
        prompt_tokens=100_000,
        completion_tokens=10_000,
    )
    with pytest.raises(BudgetExceededError):
        await assess_retrieval_confidence(
            query="Budget test",
            selected_chunks=ambiguous,
            sub_queries=["q1"],
            config=cfg,
            fast_model=expensive_model,
            fast_model_name="qwen/qwen3-30b-a3b-instruct-2507",
            cost_tracker=tracker,
        )

    # Non-budget CRAG parse failure degrades gracefully to deterministic MEDIUM policy
    broken_model = _ScriptedModel(responses=["not valid json at all", "still broken"])
    degraded, _ = await assess_retrieval_confidence(
        query="Malformed grader output test",
        selected_chunks=ambiguous,
        sub_queries=["q1"],
        config=cfg,
        rewrite_attempts=0,
        fast_model=broken_model,
        fast_model_name="qwen/qwen3-30b-a3b-instruct-2507",
    )
    assert degraded.level == ConfidenceLevel.MEDIUM
    assert degraded.crag_grade is None
    assert degraded.recommended_action == ConfidenceAction.REWRITE

    # Disabled CRAG grader skips LLM entirely in ambiguous band
    cfg_no_crag = ConfidenceConfig(enable_crag_grader_in_ambiguous_band=False)
    unused_model = _ScriptedModel(
        responses=['{"grade": "sufficient", "missing_aspects": [], "reasoning": "x"}']
    )
    det_only, det_usage = await assess_retrieval_confidence(
        query="CRAG disabled test",
        selected_chunks=ambiguous,
        sub_queries=["q1"],
        config=cfg_no_crag,
        rewrite_attempts=1,
        web_fallback_enabled=True,
        fast_model=unused_model,
    )
    assert unused_model.call_count == 0
    assert det_usage.total_tokens == 0
    assert det_only.level == ConfidenceLevel.MEDIUM
    assert det_only.recommended_action == ConfidenceAction.WEB_FALLBACK


async def test_web_fallback_primary_dedup_failover_and_exhaustion() -> None:
    # 1. Primary succeeds & deduplicates URLs
    tavily_stub = _StubProvider(
        "tavily",
        results=[
            WebSearchResult(
                title="Qdrant Release Notes",
                url="https://qdrant.tech/releases",
                content="Qdrant 1.14 introduces improved quantization and query throughput.",
                score=0.91,
                provider="tavily",
            ),
            WebSearchResult(
                title="Duplicate Release Notes",
                url="https://qdrant.tech/releases",
                content="Duplicate content that should be skipped.",
                score=0.80,
                provider="tavily",
            ),
            WebSearchResult(
                title="GitHub Changelog",
                url="https://www.github.com/qdrant/qdrant/releases",
                content="Detailed changelog for Qdrant vector search engine.",
                score=0.84,
                provider="tavily",
            ),
        ],
    )
    ddg_stub = _StubProvider("duckduckgo", results=[])
    retriever = WebFallbackRetriever([tavily_stub, ddg_stub], default_max_results=4)

    chunks = await retriever.asearch("Qdrant latest release features")
    assert len(chunks) == 2
    assert [c.citation_id for c in chunks] == ["W1", "W2"]
    assert all(c.source_type == SourceType.WEB for c in chunks)
    assert chunks[0].rerank_score == pytest.approx(0.91)
    assert chunks[1].section_path == "github.com"
    assert chunks[0].metadata["provider"] == "tavily"
    assert ddg_stub.calls == []

    # 2. Primary raises error -> seamless failover to DuckDuckGo
    failing_primary = _StubProvider("tavily", error=RuntimeError("429 Too Many Requests"))
    working_backup = _StubProvider(
        "duckduckgo",
        results=[
            WebSearchResult(
                title="LangGraph Docs",
                url="https://langchain-ai.github.io/langgraph/",
                content="LangGraph supports stateful cyclic agent workflows.",
                score=None,
                provider="duckduckgo",
            )
        ],
    )
    failover_retriever = WebFallbackRetriever([failing_primary, working_backup])
    failover_chunks = await failover_retriever.asearch("LangGraph release")
    assert len(failover_chunks) == 1
    assert failover_chunks[0].citation_id == "W1"
    assert failover_chunks[0].metadata["provider"] == "duckduckgo"
    assert failover_chunks[0].rerank_score is None
    assert failover_chunks[0].fusion_score == pytest.approx(1.0)

    # 3. Primary times out -> seamless failover to DuckDuckGo
    slow_primary = _StubProvider("tavily", delay_seconds=1.0, results=[])
    timeout_retriever = WebFallbackRetriever(
        [slow_primary, working_backup],
        timeout_seconds=0.5,
    )
    timeout_chunks = await timeout_retriever.asearch("LangGraph release")
    assert len(timeout_chunks) == 1
    assert timeout_chunks[0].metadata["provider"] == "duckduckgo"

    # 4. All providers fail -> returns empty list cleanly
    exhausted = WebFallbackRetriever(
        [
            _StubProvider("tavily", error=RuntimeError("down")),
            _StubProvider("duckduckgo", error=RuntimeError("down")),
        ]
    )
    assert await exhausted.asearch("anything") == []


def test_web_fallback_from_settings_provider_order() -> None:
    no_key_settings = Settings(_env_file=None, tavily_api_key=None)  # type: ignore[call-arg]
    r_no_key = WebFallbackRetriever.from_settings(no_key_settings)
    assert r_no_key.provider_names == ["duckduckgo"]

    with_key_settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        tavily_api_key=SecretStr("tvly-test-key"),
    )
    r_with_key = WebFallbackRetriever.from_settings(with_key_settings)
    assert r_with_key.provider_names == ["tavily", "duckduckgo"]
