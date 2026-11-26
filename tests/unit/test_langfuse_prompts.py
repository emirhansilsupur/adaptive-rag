"""$0 offline unit tests for Langfuse v5 Prompt Management, generation linking, and CLI sync."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, Field

from adaptiverag.api.dependencies import AppContainer
from adaptiverag.api.main import create_app
from adaptiverag.config import Settings, load_strategies
from adaptiverag.graph import AdaptiveRAGWorkflow, GraphRuntime
from adaptiverag.ingestion.chunker import DocumentChunk
from adaptiverag.llm import CostTracker, ModelBundle, SQLiteLLMCache
from adaptiverag.llm.structured import invoke_structured, invoke_text
from adaptiverag.observability.prompts import (
    DEFAULT_PROMPTS,
    PROMPT_CITATION_ENTAILMENT,
    PROMPT_CRAG_GRADER,
    PROMPT_GROUNDED_SYNTHESIS,
    PROMPT_ROUTER,
    PromptManager,
    compile_mustache_template,
    sync_prompts_to_langfuse,
)
from adaptiverag.observability.prompts import main as prompts_cli_main
from adaptiverag.observability.tracing import LangfuseObservability
from adaptiverag.retrieval.embeddings import SparseVectorData
from adaptiverag.retrieval.qdrant_store import QdrantHybridStore
from adaptiverag.schemas.response import FinalAnswer
from adaptiverag.schemas.retrieval import EvidenceChunk

_NS = UUID("55667788-1234-5678-1234-567812345678")
_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


class _FakeTextPromptClient:
    """In-memory fake mirroring ``langfuse.model.TextPromptClient``."""

    def __init__(
        self,
        *,
        name: str,
        prompt: str,
        version: int,
        labels: list[str] | None = None,
        is_fallback: bool = False,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.prompt = prompt
        self.version = version
        self.labels = list(labels or [])
        self.is_fallback = is_fallback
        self.config = dict(config or {})

    def compile(self, **kwargs: Any) -> str:
        return compile_mustache_template(self.prompt, kwargs)


class _FakePromptLangfuseClient:
    """In-memory Langfuse v5 client supporting Prompt Management + tracing."""

    def __init__(self, *, fail_get: bool = False, return_fallback: bool = False) -> None:
        self._store: dict[str, list[_FakeTextPromptClient]] = {}
        self.get_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []
        self.clear_cache_count = 0
        self.flush_count = 0
        self.fail_get = fail_get
        self.return_fallback = return_fallback
        self.observations: list[Any] = []
        self.scores: list[dict[str, Any]] = []
        self._trace_counter = 0

    def seed_remote(
        self,
        name: str,
        prompt: str,
        *,
        version: int = 1,
        labels: list[str] | None = None,
    ) -> _FakeTextPromptClient:
        obj = _FakeTextPromptClient(
            name=name,
            prompt=prompt,
            version=version,
            labels=labels or ["production"],
        )
        self._store.setdefault(name, []).append(obj)
        return obj

    def get_prompt(
        self,
        name: str,
        *,
        version: int | None = None,
        label: str | None = None,
        type: str = "text",
        cache_ttl_seconds: int | None = None,
        fallback: str | None = None,
        fetch_timeout_seconds: int | None = None,
    ) -> _FakeTextPromptClient:
        self.get_calls.append(
            {
                "name": name,
                "version": version,
                "label": label,
                "type": type,
                "cache_ttl_seconds": cache_ttl_seconds,
                "fallback": fallback,
                "fetch_timeout_seconds": fetch_timeout_seconds,
            }
        )
        if self.fail_get:
            raise TimeoutError("Simulated Langfuse prompt fetch timeout")
        if self.return_fallback:
            return _FakeTextPromptClient(
                name=name,
                prompt=fallback or "",
                version=0,
                is_fallback=True,
            )

        versions = self._store.get(name, [])
        if version is not None:
            for item in versions:
                if item.version == version:
                    return item
            raise KeyError(f"Prompt {name} v{version} not found")

        target_label = label or "production"
        for item in reversed(versions):
            if target_label in item.labels:
                return item
        raise KeyError(f"Prompt {name} with label {target_label} not found")

    def create_prompt(
        self,
        *,
        name: str,
        prompt: str,
        type: str = "text",
        labels: list[str] | None = None,
        tags: list[str] | None = None,
        config: dict[str, Any] | None = None,
        commit_message: str | None = None,
    ) -> _FakeTextPromptClient:
        self.create_calls.append(
            {
                "name": name,
                "prompt": prompt,
                "type": type,
                "labels": list(labels or []),
                "tags": list(tags or []),
                "config": dict(config or {}),
                "commit_message": commit_message,
            }
        )
        versions = self._store.setdefault(name, [])
        next_version = len(versions) + 1
        obj = _FakeTextPromptClient(
            name=name,
            prompt=prompt,
            version=next_version,
            labels=list(labels or []),
            config=config,
        )
        versions.append(obj)
        return obj

    def clear_prompt_cache(self) -> None:
        self.clear_cache_count += 1

    def create_trace_id(self, *, seed: str | None = None) -> str:
        self._trace_counter += 1
        return f"{self._trace_counter:032x}"

    def get_trace_url(self, *, trace_id: str | None = None) -> str:
        return f"https://cloud.langfuse.com/project/demo/traces/{trace_id}"

    def start_observation(self, **kwargs: Any) -> Any:
        record = {
            "id": f"span_{len(self.observations) + 1}",
            "name": kwargs.get("name"),
            "as_type": kwargs.get("as_type"),
            "metadata": dict(kwargs.get("metadata") or {}),
            "ended": False,
        }

        class _SpanHandle:
            id = record["id"]

            def update(self, **u_kwargs: Any) -> _SpanHandle:
                if isinstance(u_kwargs.get("metadata"), dict):
                    record["metadata"].update(u_kwargs["metadata"])
                return self

            def end(self, **e_kwargs: Any) -> _SpanHandle:
                record["ended"] = True
                return self

        self.observations.append(record)
        return _SpanHandle()

    def create_score(self, **kwargs: Any) -> None:
        self.scores.append(dict(kwargs))

    def flush(self) -> None:
        self.flush_count += 1


class _RecordingPromptCallback(BaseCallbackHandler):
    """Captures chain & chat-model callback events to verify Langfuse prompt linking contract."""

    def __init__(self) -> None:
        self.prompt_registrations: dict[UUID, Any] = {}
        self.linked_chat_prompts: list[Any] = []

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is not None and metadata and metadata.get("langfuse_prompt") is not None:
            self.prompt_registrations[parent_run_id] = metadata["langfuse_prompt"]

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        linked = self.prompt_registrations.get(parent_run_id) if parent_run_id is not None else None
        self.linked_chat_prompts.append(linked)


class _InspectingChat(BaseChatModel):
    """Scripted fake LLM that records exact system prompts received."""

    responses: list[str] = Field(default_factory=list)
    seen_system_prompts: list[str] = Field(default_factory=list)
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return "inspecting-chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        sys_texts = [str(m.content) for m in messages if m.type == "system"]
        self.seen_system_prompts.append("\n---\n".join(sys_texts))
        idx = min(self.cursor, len(self.responses) - 1)
        self.cursor += 1
        msg = AIMessage(
            content=self.responses[idx],
            usage_metadata={"input_tokens": 80, "output_tokens": 30, "total_tokens": 110},
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


class _AmbiguousBandReranker:
    """Assigns rerank_score=0.50 so score_confidence enters the ambiguous CRAG grader band."""

    model_name = "stub-ambiguous-reranker"

    def rerank(
        self, query: str, chunks: list[EvidenceChunk], *, top_n: int | None = None
    ) -> list[EvidenceChunk]:
        out = [c.model_copy(update={"rerank_score": 0.50, "rerank_raw_score": 0.0}) for c in chunks]
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


def test_default_prompt_registry_and_mustache_compiler() -> None:
    assert len(DEFAULT_PROMPTS) == 8

    for name, defn in DEFAULT_PROMPTS.items():
        assert defn.name == name
        assert defn.description.strip()
        assert defn.template.strip()
        found_vars = set(_VAR_RE.findall(defn.template))
        assert found_vars == set(defn.variables), f"Mismatch in declared variables for {name}"

    rendered = compile_mustache_template(
        'Schema: {"grade": "ok"} | Lang={{ language }} | Max={{max_sub_queries}} | {{unknown}}',
        {"language": "tr", "max_sub_queries": 3},
    )
    assert rendered == 'Schema: {"grade": "ok"} | Lang=tr | Max=3 | {{unknown}}'


def test_prompt_manager_local_fallback_when_disabled_unconfigured_or_failing() -> None:
    # 1. Disabled explicitly -> local fallback
    pm_disabled = PromptManager(enabled=False)
    assert pm_disabled.active is False
    assert pm_disabled.health_summary()["mode"] == "local_fallback"
    r_disabled = pm_disabled.resolve(
        PROMPT_GROUNDED_SYNTHESIS,
        variables={"language_instruction": "Respond in English.", "style_hint": "Be concise."},
    )
    assert r_disabled.source == "local_fallback"
    assert r_disabled.version is None
    assert r_disabled.langfuse_prompt is None
    assert "Respond in English." in r_disabled.text
    assert "Be concise." in r_disabled.text

    # 2. Enabled, but observability unconfigured -> local fallback
    obs_unconfigured = LangfuseObservability(enabled=True, public_key="", secret_key=None)
    pm_unconf = PromptManager(observability=obs_unconfigured, enabled=True)
    assert pm_unconf.active is False
    assert pm_unconf.resolve(PROMPT_ROUTER).source == "local_fallback"

    # 3. SDK returns fallback object (is_fallback=True) -> reported as local_fallback, not linked
    fallback_client = _FakePromptLangfuseClient(return_fallback=True)
    obs_fb = LangfuseObservability(enabled=True, client=fallback_client)
    pm_fb = PromptManager(observability=obs_fb, enabled=True, label="production")
    res_fb = pm_fb.resolve(PROMPT_ROUTER)
    assert res_fb.source == "local_fallback"
    assert res_fb.version is None
    assert res_fb.langfuse_prompt is None

    # 4. Network error during get_prompt -> fail-open to local template
    err_client = _FakePromptLangfuseClient(fail_get=True)
    obs_err = LangfuseObservability(enabled=True, client=err_client)
    pm_err = PromptManager(observability=obs_err, enabled=True, label="production")
    res_err = pm_err.resolve(PROMPT_ROUTER)
    assert res_err.source == "local_fallback"
    assert "Query Router for AdaptiveRAG" in res_err.text


def test_prompt_manager_live_resolution_version_override_and_cache_clear() -> None:
    fake_client = _FakePromptLangfuseClient()
    v2 = fake_client.seed_remote(
        PROMPT_GROUNDED_SYNTHESIS,
        "V2 Prompt: {{language_instruction}} | {{style_hint}}",
        version=2,
        labels=["staging"],
    )
    v3 = fake_client.seed_remote(
        PROMPT_GROUNDED_SYNTHESIS,
        "V3 PROD Prompt: {{language_instruction}} | {{style_hint}}",
        version=3,
        labels=["production"],
    )
    obs = LangfuseObservability(enabled=True, client=fake_client)
    pm = PromptManager(
        observability=obs,
        enabled=True,
        label="production",
        cache_ttl_seconds=120,
    )

    health = pm.health_summary()
    assert health == {
        "enabled": True,
        "active": True,
        "mode": "langfuse",
        "label": "production",
        "cache_ttl_seconds": 120,
        "managed_prompts": 8,
    }

    resolved_prod = pm.resolve(
        PROMPT_GROUNDED_SYNTHESIS,
        variables={"language_instruction": "Answer in TR.", "style_hint": "Bullet points."},
    )
    assert resolved_prod.source == "langfuse"
    assert resolved_prod.version == 3
    assert resolved_prod.label == "production"
    assert resolved_prod.langfuse_prompt is v3
    assert resolved_prod.text == "V3 PROD Prompt: Answer in TR. | Bullet points."
    assert resolved_prod.telemetry_metadata() == {
        "name": PROMPT_GROUNDED_SYNTHESIS,
        "source": "langfuse",
        "version": 3,
        "label": "production",
    }

    # Explicit version override must omit label when calling Langfuse get_prompt
    resolved_v2 = pm.resolve(
        PROMPT_GROUNDED_SYNTHESIS,
        variables={"language_instruction": "Answer in EN.", "style_hint": "Short."},
        version=2,
    )
    assert resolved_v2.version == 2
    assert resolved_v2.label is None
    assert resolved_v2.langfuse_prompt is v2
    assert fake_client.get_calls[-1]["version"] == 2
    assert fake_client.get_calls[-1]["label"] is None

    pm.clear_cache()
    assert fake_client.clear_cache_count == 1


class _MiniSchema(BaseModel):
    ok: bool


@pytest.mark.asyncio
async def test_invoke_helpers_link_langfuse_prompt_and_invalidate_cache_on_version_change(
    tmp_path: Path,
) -> None:
    prompt_v1 = _FakeTextPromptClient(
        name=PROMPT_GROUNDED_SYNTHESIS,
        prompt="SYSTEM_V1",
        version=1,
        labels=["production"],
    )
    prompt_v2 = _FakeTextPromptClient(
        name=PROMPT_GROUNDED_SYNTHESIS,
        prompt="SYSTEM_V2_UPDATED",
        version=2,
        labels=["production"],
    )

    cache = SQLiteLLMCache(tmp_path / "prompt_cache.sqlite")
    llm = _InspectingChat(responses=["Answer from v1", "Answer from v2", '{"ok": true}'])
    cb = _RecordingPromptCallback()

    from langchain_core.messages import HumanMessage, SystemMessage

    # 1. Text call with v1 links prompt_v1 on parent chain -> chat model
    res1 = await invoke_text(
        llm,
        [SystemMessage(content=prompt_v1.prompt), HumanMessage(content="Hello")],
        model_name="fake/gen",
        cache=cache,
        callbacks=[cb],
        langfuse_prompt=prompt_v1,
        prompt_metadata={"name": prompt_v1.name, "source": "langfuse", "version": 1},
    )
    assert res1.text == "Answer from v1"
    assert res1.from_cache is False
    assert cb.linked_chat_prompts[-1] is prompt_v1

    # 2. Same v1 call hits SQLite cache ($0)
    res1_cached = await invoke_text(
        llm,
        [SystemMessage(content=prompt_v1.prompt), HumanMessage(content="Hello")],
        model_name="fake/gen",
        cache=cache,
        callbacks=[cb],
        langfuse_prompt=prompt_v1,
    )
    assert res1_cached.from_cache is True

    # 3. Updating prompt text to v2 automatically invalidates cache & links prompt_v2
    res2 = await invoke_text(
        llm,
        [SystemMessage(content=prompt_v2.prompt), HumanMessage(content="Hello")],
        model_name="fake/gen",
        cache=cache,
        callbacks=[cb],
        langfuse_prompt=prompt_v2,
        prompt_metadata={"name": prompt_v2.name, "source": "langfuse", "version": 2},
    )
    assert res2.text == "Answer from v2"
    assert res2.from_cache is False
    assert cb.linked_chat_prompts[-1] is prompt_v2

    # 4. Structured call also binds prompt_v2 via CallbackHandler protocol
    res_struct = await invoke_structured(
        llm,
        _MiniSchema,
        [SystemMessage(content=prompt_v2.prompt), HumanMessage(content="Check")],
        model_name="fake/fast",
        callbacks=[cb],
        langfuse_prompt=prompt_v2,
        prompt_metadata={"name": PROMPT_ROUTER, "source": "langfuse", "version": 2},
    )
    assert res_struct.parsed.ok is True
    assert cb.linked_chat_prompts[-1] is prompt_v2


@pytest.mark.asyncio
async def test_workflow_and_health_end_to_end_with_managed_prompts(tmp_path: Path) -> None:
    fake_client = _FakePromptLangfuseClient()
    for name, defn in DEFAULT_PROMPTS.items():
        fake_client.seed_remote(
            name,
            f"[LANGFUSE_V4:{name}]\n{defn.template}",
            version=4,
            labels=["staging"],
        )

    obs = LangfuseObservability(
        enabled=True,
        host="https://cloud.langfuse.com",
        environment="staging",
        client=fake_client,
    )
    pm = PromptManager(observability=obs, enabled=True, label="staging", cache_ttl_seconds=60)
    settings = Settings(
        llm_provider="openrouter",
        qdrant_collection="lf_prompts_e2e",
        llm_cache_enabled=False,
        langfuse_enabled=True,
        langfuse_prompt_management_enabled=True,
        langfuse_prompt_label="staging",
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
            "reasoning": "Direct technical lookup.",
        }
    )
    crag_json = json.dumps(
        {
            "grade": "sufficient",
            "missing_aspects": [],
            "reasoning": "Passage directly explains Qdrant hybrid search.",
        }
    )
    entail_json = json.dumps(
        {
            "verifications": [
                {
                    "claim": "Qdrant fuses dense vectors and BM25 sparse vectors with RRF.",
                    "cited_ids": ["C1"],
                    "verdict": "supported",
                    "explanation": "Explicitly supported by [C1].",
                }
            ]
        }
    )

    fast_llm = _InspectingChat(responses=[router_json, crag_json, entail_json])
    gen_llm = _InspectingChat(
        responses=["Qdrant fuses dense vectors and BM25 sparse vectors with RRF [C1]."]
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
        reranker=_AmbiguousBandReranker(),  # type: ignore[arg-type]
        fallback_retriever=_EmptyFallback(),  # type: ignore[arg-type]
        cache=None,
        cost_tracker=CostTracker(max_cost_usd=1.0),
        observability=obs,
        prompt_manager=pm,
    )
    workflow = AdaptiveRAGWorkflow(runtime)
    container = AppContainer(
        settings=settings,
        strategies=strategies,
        store=store,
        fallback_retriever=_EmptyFallback(),  # type: ignore[arg-type]
        workflow=workflow,
        observability=obs,
        prompt_manager=pm,
        owns_store=True,
        owns_firecrawl=False,
    )
    app = create_app(container=container)

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://localhost",
        ) as client:
            health = await client.get("/v1/health")
            assert health.status_code == 200
            pm_health = health.json()["langfuse"]["prompt_management"]
            assert pm_health["active"] is True
            assert pm_health["mode"] == "langfuse"
            assert pm_health["label"] == "staging"
            assert pm_health["managed_prompts"] == 8

            resp = await client.post(
                "/v1/query",
                json={"query": "How does Qdrant hybrid search work?", "preset": "full"},
            )
            assert resp.status_code == 200
            answer = FinalAnswer.model_validate(resp.json())

            # Every invoked LLM received the remote v4 managed system prompt
            assert any(f"[LANGFUSE_V4:{PROMPT_ROUTER}]" in p for p in fast_llm.seen_system_prompts)
            assert any(
                f"[LANGFUSE_V4:{PROMPT_CRAG_GRADER}]" in p for p in fast_llm.seen_system_prompts
            )
            assert any(
                f"[LANGFUSE_V4:{PROMPT_CITATION_ENTAILMENT}]" in p
                for p in fast_llm.seen_system_prompts
            )
            assert any(
                f"[LANGFUSE_V4:{PROMPT_GROUNDED_SYNTHESIS}]" in p
                for p in gen_llm.seen_system_prompts
            )

            # Trace events expose versioned prompt telemetry
            trace_by_node = {ev.node: ev for ev in answer.trace}
            for node_name, expected_prompt in [
                ("classify_and_plan", PROMPT_ROUTER),
                ("score_confidence", PROMPT_CRAG_GRADER),
                ("generate_answer", PROMPT_GROUNDED_SYNTHESIS),
                ("validate_citations", PROMPT_CITATION_ENTAILMENT),
            ]:
                p_meta = trace_by_node[node_name].metadata.get("prompt")
                assert p_meta == {
                    "name": expected_prompt,
                    "source": "langfuse",
                    "version": 4,
                    "label": "staging",
                }

            # Root trace span aggregates prompts_used for trace-level filtering in Langfuse
            root_span = next(
                o for o in fake_client.observations if o["name"] == "adaptiverag.query"
            )
            assert set(root_span["metadata"]["prompts_used"].keys()) == {
                PROMPT_ROUTER,
                PROMPT_CRAG_GRADER,
                PROMPT_GROUNDED_SYNTHESIS,
                PROMPT_CITATION_ENTAILMENT,
            }
    finally:
        await container.aclose()


def test_idempotent_sync_prompts_to_langfuse_and_cli(capsys: pytest.CaptureFixture[str]) -> None:
    fake_client = _FakePromptLangfuseClient()

    # 1. Dry-run previews all 8 creations without writing anything
    dry = sync_prompts_to_langfuse(fake_client, label="production", dry_run=True)
    assert dry.total == 8
    assert dry.created == 0
    assert dry.unchanged == 0
    assert all(item.action == "would_create" for item in dry.items)
    assert len(fake_client.create_calls) == 0

    # 2. First real seed creates all 8 prompts at v1
    first = sync_prompts_to_langfuse(fake_client, label="production")
    assert first.total == 8
    assert first.created == 8
    assert first.unchanged == 0
    assert first.errors == 0
    assert all(item.version == 1 for item in first.items)
    assert fake_client.flush_count >= 1

    # 3. Second identical seed is 100% idempotent (0 duplicate versions created)
    second = sync_prompts_to_langfuse(fake_client, label="production")
    assert second.created == 0
    assert second.unchanged == 8
    assert len(fake_client.create_calls) == 8

    # 4. Simulate drift in 1 prompt -> only that 1 prompt bumps to v2
    fake_client.seed_remote(
        PROMPT_ROUTER, "Modified remote prompt", version=2, labels=["production"]
    )
    third = sync_prompts_to_langfuse(fake_client, label="production")
    assert third.created == 1
    assert third.unchanged == 7
    router_item = next(item for item in third.items if item.name == PROMPT_ROUTER)
    assert router_item.action == "created"
    assert router_item.version == 3

    # 5. Force pushes a fresh version for every prompt
    forced = sync_prompts_to_langfuse(fake_client, label="production", force=True)
    assert forced.created == 8
    assert forced.unchanged == 0

    # 6. Offline CLI --list --json outputs all 8 canonical prompts at $0
    exit_code = prompts_cli_main(["--list", "--json"])
    assert exit_code == 0
    captured = capsys.readouterr()
    listed = json.loads(captured.out)
    assert len(listed) == 8
    assert {row["name"] for row in listed} == set(DEFAULT_PROMPTS.keys())
