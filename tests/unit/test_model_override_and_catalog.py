"""$0 offline unit & API tests for GET /v1/models catalog and per-request llm_override."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

import httpx
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, SecretStr

from adaptiverag.api.dependencies import AppContainer
from adaptiverag.api.main import create_app
from adaptiverag.config import Settings, load_strategies
from adaptiverag.graph import AdaptiveRAGWorkflow, GraphRuntime
from adaptiverag.ingestion.chunker import DocumentChunk
from adaptiverag.llm import CostTracker, ModelBundle, resolve_effective_settings
from adaptiverag.retrieval.embeddings import SparseVectorData
from adaptiverag.retrieval.qdrant_store import QdrantHybridStore
from adaptiverag.schemas.response import FinalAnswer
from adaptiverag.schemas.retrieval import EvidenceChunk

_NS = UUID("11223344-5566-7788-99aa-bbccddeeff00")


class _TaggingChatModel(BaseChatModel):
    """Scripted fake LLM tracking invocation counts for default vs override bundles."""

    responses: list[str] = Field(default_factory=list)
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "tagging-chat"

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
            usage_metadata={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


class _StubDense:
    model_name = "stub-dense"
    dimension = 4

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


class _StubSparse:
    model_name = "stub-bm25"

    def embed_documents(self, texts: list[str]) -> list[SparseVectorData]:
        return [SparseVectorData(indices=[1], values=[1.0]) for _ in texts]

    def embed_query(self, text: str) -> SparseVectorData:
        return SparseVectorData(indices=[1], values=[1.0])

    async def aembed_documents(self, texts: list[str]) -> list[SparseVectorData]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> SparseVectorData:
        return self.embed_query(text)


class _HighScoreReranker:
    model_name = "stub-high-reranker"

    def rerank(
        self, query: str, chunks: list[EvidenceChunk], *, top_n: int | None = None
    ) -> list[EvidenceChunk]:
        out = [c.model_copy(update={"rerank_score": 0.91, "rerank_raw_score": 2.3}) for c in chunks]
        return out[:top_n] if top_n else out

    async def arerank(
        self, query: str, chunks: list[EvidenceChunk], *, top_n: int | None = None
    ) -> list[EvidenceChunk]:
        return self.rerank(query, chunks, top_n=top_n)


class _EmptyFallback:
    enabled = True

    @property
    def provider_names(self) -> list[str]:
        return ["stub_web"]

    async def asearch(self, query: str, **kwargs: Any) -> list[EvidenceChunk]:
        return []


def test_resolve_effective_settings_switches_provider_defaults_without_mutation() -> None:
    base = Settings(
        llm_provider="ollama",
        ollama_base_url="http://localhost:8080/v1",
        llm_fast_model="inkling-2-mini",
        llm_generation_model="inkling-2-mini",
        llm_temperature=1.0,
        openrouter_api_key=SecretStr("sk-or-v1-testsecret1234567890"),
    )

    # Switching provider without model names selects that provider's catalog defaults
    switched = resolve_effective_settings(base, provider="openrouter")
    assert switched.llm_provider == "openrouter"
    assert switched.llm_fast_model == "qwen/qwen3-30b-a3b-instruct-2507"
    assert switched.llm_generation_model == "deepseek/deepseek-v3.2"
    assert switched.llm_temperature == 1.0

    # Original settings object stays immutable
    assert base.llm_provider == "ollama"
    assert base.llm_fast_model == "inkling-2-mini"

    # Explicit overrides win
    explicit = resolve_effective_settings(
        base,
        provider="openrouter",
        fast_model="google/gemini-2.5-flash",
        generation_model="anthropic/claude-sonnet-4",
        temperature=0.2,
    )
    assert explicit.llm_fast_model == "google/gemini-2.5-flash"
    assert explicit.llm_generation_model == "anthropic/claude-sonnet-4"
    assert explicit.llm_temperature == 0.2


@pytest.mark.asyncio
async def test_models_catalog_and_per_request_llm_override_api(tmp_path: Path) -> None:
    secret_val = "sk-or-v1-supersecret-never-leak-9999"
    settings = Settings(
        llm_provider="ollama",
        ollama_base_url="http://localhost:8080/v1",
        openrouter_api_key=SecretStr(secret_val),
        openai_api_key=None,
        gemini_api_key=None,
        anthropic_api_key=None,
        llm_fast_model="inkling-2-mini",
        llm_generation_model="inkling-2-mini",
        llm_temperature=1.0,
        qdrant_collection="model_override_test",
        llm_cache_enabled=False,
    )
    strategies = load_strategies(Path("configs/strategies.yaml"))
    store = QdrantHybridStore.from_settings(
        settings,
        dense_embedder=_StubDense(),  # type: ignore[arg-type]
        sparse_embedder=_StubSparse(),  # type: ignore[arg-type]
        in_memory=True,
    )
    await store.ensure_collection(recreate=True)
    await store.upsert_chunks(
        [
            DocumentChunk(
                chunk_id=str(uuid5(_NS, "doc1::0")),
                doc_id="qdrant_hybrid",
                title="Qdrant Hybrid Search",
                section_path="RRF",
                text="Qdrant fuses dense and BM25 sparse rankings with RRF.",
                chunk_index=0,
                token_count=14,
                metadata={"language": "en"},
            )
        ]
    )

    router_factual = json.dumps(
        {
            "query_type": "factual",
            "needs_retrieval": True,
            "language": "en",
            "standalone_query": "How does Qdrant hybrid search work?",
            "requires_fresh_web_info": False,
            "reasoning": "Direct lookup.",
        }
    )
    entail_ok = json.dumps(
        {
            "verifications": [
                {
                    "claim": "Qdrant fuses dense and BM25 sparse rankings with RRF.",
                    "cited_ids": ["C1"],
                    "verdict": "supported",
                    "explanation": "Explicitly supported by [C1].",
                }
            ]
        }
    )

    default_fast = _TaggingChatModel(responses=[router_factual, entail_ok])
    default_gen = _TaggingChatModel(
        responses=["Default local model answer: Qdrant fuses dense and sparse via RRF [C1]."]
    )

    override_fast = _TaggingChatModel(
        responses=[router_factual, entail_ok, router_factual, entail_ok, router_factual, entail_ok]
    )
    override_gen = _TaggingChatModel(
        responses=[
            "Override cloud answer: Qdrant fuses dense and sparse via RRF [C1].",
            "Override cloud answer #2: Qdrant fuses dense and sparse via RRF [C1].",
            "BYOK OpenAI answer: Qdrant fuses dense and sparse via RRF [C1].",
        ]
    )

    factory_invocations: list[tuple[str, str, str, float, str | None]] = []

    def _fake_bundle_factory(eff: Settings) -> ModelBundle:
        effective_openai_key = (
            eff.openai_api_key.get_secret_value() if eff.openai_api_key is not None else None
        )
        factory_invocations.append(
            (
                eff.llm_provider,
                eff.llm_fast_model,
                eff.llm_generation_model,
                eff.llm_temperature,
                effective_openai_key,
            )
        )
        return ModelBundle(
            fast_model=override_fast,
            fast_model_name=eff.llm_fast_model,
            generation_model=override_gen,
            generation_model_name=eff.llm_generation_model,
            provider=eff.llm_provider,
            temperature=eff.llm_temperature,
        )

    async def _fake_local_probe(_base_url: str) -> tuple[bool, list[str], str | None]:
        return True, ["inkling-2-mini"], None

    runtime = GraphRuntime(
        settings=settings,
        strategies=strategies,
        models=ModelBundle(
            fast_model=default_fast,
            fast_model_name="inkling-2-mini",
            generation_model=default_gen,
            generation_model_name="inkling-2-mini",
            provider="ollama",
            temperature=1.0,
        ),
        store=store,
        reranker=_HighScoreReranker(),  # type: ignore[arg-type]
        fallback_retriever=_EmptyFallback(),  # type: ignore[arg-type]
        cache=None,
        cost_tracker=CostTracker(max_cost_usd=1.0),
    )
    workflow = AdaptiveRAGWorkflow(runtime)
    container = AppContainer(
        settings=settings,
        strategies=strategies,
        store=store,
        fallback_retriever=_EmptyFallback(),  # type: ignore[arg-type]
        workflow=workflow,
        local_models_probe_fn=_fake_local_probe,
        model_bundle_factory=_fake_bundle_factory,
        owns_store=True,
        owns_firecrawl=False,
    )
    app = create_app(container=container)

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://localhost",
        ) as client:
            # 1. GET /v1/models returns safe readiness + detected local models, zero secret leakage
            models_resp = await client.get("/v1/models")
            assert models_resp.status_code == 200
            raw_text = models_resp.text
            assert secret_val not in raw_text

            catalog = models_resp.json()
            assert catalog["default_provider"] == "ollama"
            assert catalog["default_fast_model"] == "inkling-2-mini"
            assert catalog["default_temperature"] == 1.0
            assert catalog["allow_client_provider_keys"] is True

            by_id = {p["id"]: p for p in catalog["providers"]}
            assert by_id["openrouter"]["configured"] is True
            assert by_id["openrouter"]["requires_api_key"] is True
            assert by_id["openrouter"]["sdk_installed"] is True
            assert by_id["ollama"]["configured"] is True
            assert by_id["ollama"]["requires_api_key"] is False
            assert by_id["ollama"]["reachable"] is True
            assert by_id["ollama"]["detected_models"] == ["inkling-2-mini"]
            assert by_id["openai"]["configured"] is False
            assert by_id["openai"]["sdk_installed"] is True
            assert "OPENAI_API_KEY" in (by_id["openai"]["status_detail"] or "")

            # 2. Request without llm_override uses default runtime models
            r_default = await client.post(
                "/v1/query",
                json={"query": "How does Qdrant hybrid search work?", "preset": "full"},
            )
            assert r_default.status_code == 200
            ans_default = FinalAnswer.model_validate(r_default.json())
            assert "Default local model answer" in ans_default.answer
            assert default_gen.call_count == 1
            assert override_gen.call_count == 0

            # 3. Request with llm_override switches seamlessly to OpenRouter override bundle
            override_payload = {
                "provider": "openrouter",
                "fast_model": "google/gemini-2.5-flash",
                "generation_model": "deepseek/deepseek-v3.2",
                "temperature": 0.25,
            }
            r_override = await client.post(
                "/v1/query",
                json={
                    "query": "How does Qdrant hybrid search work?",
                    "preset": "full",
                    "llm_override": override_payload,
                },
            )
            assert r_override.status_code == 200
            ans_override = FinalAnswer.model_validate(r_override.json())
            assert "Override cloud answer" in ans_override.answer
            gen_trace = next(ev for ev in ans_override.trace if ev.node == "generate_answer")
            assert gen_trace.metadata["model"] == "deepseek/deepseek-v3.2"
            assert len(factory_invocations) == 1
            assert factory_invocations[0] == (
                "openrouter",
                "google/gemini-2.5-flash",
                "deepseek/deepseek-v3.2",
                0.25,
                None,
            )

            # 4. Repeating identical override hits container bundle LRU cache (0 extra factory calls)
            r_override_2 = await client.post(
                "/v1/query",
                json={
                    "query": "How does Qdrant hybrid search work?",
                    "preset": "full",
                    "llm_override": override_payload,
                },
            )
            assert r_override_2.status_code == 200
            assert len(factory_invocations) == 1
            assert override_gen.call_count == 2

            # 5. Requesting an unconfigured provider without BYOK header fails cleanly with HTTP 400
            r_bad = await client.post(
                "/v1/query",
                json={
                    "query": "How does Qdrant hybrid search work?",
                    "preset": "full",
                    "llm_override": {"provider": "openai", "generation_model": "gpt-4.1"},
                },
            )
            assert r_bad.status_code == 400
            assert "OPENAI_API_KEY" in r_bad.json()["detail"]

            # 6. Supplying session-only X-Provider-API-Key enables OpenAI without mutating Settings
            byok_secret = "sk-proj-byok-ephemeral-key-987654321"
            r_byok = await client.post(
                "/v1/query",
                headers={"X-Provider-API-Key": byok_secret},
                json={
                    "query": "How does Qdrant hybrid search work?",
                    "preset": "full",
                    "llm_override": {
                        "provider": "openai",
                        "fast_model": "gpt-4o-mini",
                        "generation_model": "gpt-4.1",
                        "temperature": 0.1,
                    },
                },
            )
            assert r_byok.status_code == 200
            assert byok_secret not in r_byok.text
            assert "BYOK OpenAI answer" in FinalAnswer.model_validate(r_byok.json()).answer
            assert container.settings.openai_api_key is None
            assert factory_invocations[-1] == (
                "openai",
                "gpt-4o-mini",
                "gpt-4.1",
                0.1,
                byok_secret,
            )
            # Raw client secret must NEVER appear in container._bundle_cache keys
            assert all(byok_secret not in str(cache_key) for cache_key in container._bundle_cache)

            # 7. Disabling ALLOW_CLIENT_PROVIDER_KEYS blocks client header with HTTP 403
            container.settings = container.settings.model_copy(
                update={"allow_client_provider_keys": False}
            )
            r_forbidden = await client.post(
                "/v1/query",
                headers={"X-Provider-API-Key": byok_secret},
                json={
                    "query": "How does Qdrant hybrid search work?",
                    "preset": "full",
                    "llm_override": {"provider": "openai", "generation_model": "gpt-4.1"},
                },
            )
            assert r_forbidden.status_code == 403
            assert "ALLOW_CLIENT_PROVIDER_KEYS" in r_forbidden.json()["detail"]
    finally:
        await container.aclose()
