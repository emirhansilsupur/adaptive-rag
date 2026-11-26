"""Compiled LangGraph StateGraph and high-level AdaptiveRAGWorkflow facade."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence
from functools import partial
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from adaptiverag.config import Settings, StrategiesConfig, load_strategies
from adaptiverag.graph.edges import (
    route_after_citation_validation,
    route_after_confidence,
    route_after_plan,
    route_after_web_fallback,
)
from adaptiverag.graph.nodes import GraphRuntime, create_nodes
from adaptiverag.graph.prompts import build_empty_answer_message
from adaptiverag.graph.state import AblationConfig, AgentState, merge_token_usage
from adaptiverag.llm.cache import SQLiteLLMCache
from adaptiverag.llm.cost import CostTracker
from adaptiverag.llm.factory import ModelBundle, check_provider_configured, create_model_bundle
from adaptiverag.observability.prompts import PromptManager
from adaptiverag.observability.tracing import LangfuseObservability, TraceRunHandle
from adaptiverag.retrieval.fallback import FallbackRetriever, WebFallbackRetriever
from adaptiverag.retrieval.qdrant_store import QdrantHybridStore
from adaptiverag.retrieval.reranker import FastEmbedCrossEncoderReranker, Reranker
from adaptiverag.schemas.query import ChatMessage, QueryType
from adaptiverag.schemas.response import FinalAnswer, NodeTraceEvent, TokenUsage
from adaptiverag.verification.guardrails import (
    GuardrailBlockedError,
    InputGuardrailConfig,
    screen_query_input,
)


def build_adaptive_rag_graph(
    runtime: GraphRuntime,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Assemble and compile the bounded cyclic AdaptiveRAG LangGraph state machine."""
    nodes = create_nodes(runtime)
    builder: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(AgentState)

    for node_name, node_fn in nodes.items():
        builder.add_node(node_name, cast(Any, node_fn))

    builder.add_edge(START, "classify_and_plan")

    builder.add_conditional_edges(
        "classify_and_plan",
        route_after_plan,
        {
            "conversational_respond": "conversational_respond",
            "decompose_queries": "decompose_queries",
            "hybrid_retrieve": "hybrid_retrieve",
        },
    )

    builder.add_edge("conversational_respond", END)
    builder.add_edge("decompose_queries", "hybrid_retrieve")
    builder.add_edge("hybrid_retrieve", "rerank_and_select_k")
    builder.add_edge("rerank_and_select_k", "score_confidence")

    confidence_router = partial(
        route_after_confidence,
        max_rewrites=runtime.strategies.confidence.max_corrective_rewrites,
        web_fallback_enabled=runtime.settings.web_fallback_enabled,
    )
    builder.add_conditional_edges(
        "score_confidence",
        confidence_router,
        {
            "generate_answer": "generate_answer",
            "corrective_rewrite": "corrective_rewrite",
            "web_fallback_search": "web_fallback_search",
            "abstain_response": "abstain_response",
        },
    )

    builder.add_edge("corrective_rewrite", "hybrid_retrieve")

    builder.add_conditional_edges(
        "web_fallback_search",
        route_after_web_fallback,
        {
            "generate_answer": "generate_answer",
            "abstain_response": "abstain_response",
        },
    )

    builder.add_edge("generate_answer", "validate_citations")

    citation_router = partial(
        route_after_citation_validation,
        max_regenerations=runtime.strategies.citation_validation.max_regenerations,
    )
    builder.add_conditional_edges(
        "validate_citations",
        citation_router,
        {
            "regenerate_strict": "regenerate_strict",
            "end": END,
        },
    )

    builder.add_edge("regenerate_strict", "validate_citations")
    builder.add_edge("abstain_response", END)

    return builder.compile()


def _state_to_final_answer(
    state: dict[str, Any],
    total_latency_ms: float,
    *,
    trace_id: str | None = None,
    trace_url: str | None = None,
) -> FinalAnswer:
    plan = state.get("query_plan")
    query_type = plan.query_type if plan is not None else QueryType.FACTUAL
    language = plan.language if plan is not None else "en"
    regeneration_attempts = int(state.get("regeneration_attempts", 0))
    answer = str(state.get("draft_answer", ""))
    if not answer.strip():
        # The model returned no text, also on the strict retry when one ran.
        answer = build_empty_answer_message(language=language, retried=regeneration_attempts > 0)

    return FinalAnswer(
        query=str(state.get("query", "")),
        answer=answer,
        query_type=query_type,
        language=language,
        query_plan=plan,
        evidence_chunks=list(state.get("selected_chunks") or []),
        dynamic_k_decision=state.get("dynamic_k_decision"),
        confidence=state.get("confidence"),
        citation_report=state.get("citation_report"),
        fallback_used=bool(state.get("fallback_used", False)),
        abstained=bool(state.get("abstained", False)),
        rewrite_attempts=int(state.get("rewrite_attempts", 0)),
        regeneration_attempts=regeneration_attempts,
        total_latency_ms=round(total_latency_ms, 2),
        usage=state.get("usage") or TokenUsage(),
        trace=list(state.get("trace") or []),
        trace_id=trace_id,
        trace_url=trace_url,
    )


class AdaptiveRAGWorkflow:
    """High-level async orchestrator exposing batch (arun) and streaming (astream_events) APIs."""

    def __init__(self, runtime: GraphRuntime) -> None:
        self._runtime = runtime
        self._graph = build_adaptive_rag_graph(runtime)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        strategies: StrategiesConfig | None = None,
        models: ModelBundle | None = None,
        store: QdrantHybridStore | None = None,
        reranker: Reranker | None = None,
        fallback_retriever: FallbackRetriever | None = None,
        cache: SQLiteLLMCache | None = None,
        cost_tracker: CostTracker | None = None,
        observability: LangfuseObservability | None = None,
        prompt_manager: PromptManager | None = None,
        in_memory_store: bool = False,
        require_models: bool = True,
    ) -> AdaptiveRAGWorkflow:
        """Instantiate a full AdaptiveRAGWorkflow wired from Settings & strategies.yaml.

        Without ``models``, the server's own models are built from ``settings``. With
        ``require_models=False`` the workflow is built even when the server's provider isn't
        configured; it then has no default models, and every run must pass its own.
        """
        resolved_strategies = strategies or load_strategies(settings.strategies_config_path)
        resolved_models = models
        if resolved_models is None:
            server_llm_ready, _ = check_provider_configured(settings, settings.llm_provider)
            if server_llm_ready or require_models:
                resolved_models = create_model_bundle(settings)
        resolved_store = store or QdrantHybridStore.from_settings(
            settings,
            in_memory=in_memory_store,
        )
        resolved_reranker: Reranker = reranker or FastEmbedCrossEncoderReranker(
            model_name=settings.reranker_model,
        )
        resolved_fallback: FallbackRetriever = (
            fallback_retriever or WebFallbackRetriever.from_settings(settings)
        )
        resolved_cache = (
            cache
            if cache is not None
            else (SQLiteLLMCache(settings.llm_cache_path) if settings.llm_cache_enabled else None)
        )
        # Keep runtime.cost_tracker None unless explicitly injected so each run/request
        # creates its own isolated CostTracker(max_cost_usd=settings.max_run_cost_usd).
        resolved_cost_tracker = cost_tracker
        resolved_observability = observability or LangfuseObservability.from_settings(settings)
        resolved_prompts = prompt_manager or PromptManager.from_settings(
            settings,
            observability=resolved_observability,
        )

        runtime = GraphRuntime(
            settings=settings,
            strategies=resolved_strategies,
            models=resolved_models,
            store=resolved_store,
            reranker=resolved_reranker,
            fallback_retriever=resolved_fallback,
            cache=resolved_cache,
            cost_tracker=resolved_cost_tracker,
            observability=resolved_observability,
            prompt_manager=resolved_prompts,
        )
        return cls(runtime)

    @property
    def runtime(self) -> GraphRuntime:
        return self._runtime

    @property
    def graph(self) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
        return self._graph

    def _build_initial_state(
        self,
        query: str,
        *,
        chat_history: Sequence[ChatMessage] | None = None,
        ablation: AblationConfig | None = None,
        custom_instructions: str | None = None,
    ) -> AgentState:
        cleaned_instructions = (
            custom_instructions.strip()
            if custom_instructions and custom_instructions.strip()
            else None
        )
        return AgentState(
            query=query.strip(),
            chat_history=list(chat_history) if chat_history else [],
            ablation=ablation or AblationConfig(),
            custom_instructions=cleaned_instructions,
            usage=TokenUsage(),
            trace=[],
        )

    def _get_observability(self) -> LangfuseObservability:
        if self._runtime.observability is None:
            self._runtime.observability = LangfuseObservability.from_settings(
                self._runtime.settings
            )
        return self._runtime.observability

    def _resolve_run_cost_tracker(self, cost_tracker: CostTracker | None) -> CostTracker:
        """Return an isolated per-run CostTracker unless a shared tracker was explicitly supplied."""
        if cost_tracker is not None:
            return cost_tracker
        if self._runtime.cost_tracker is not None:
            return self._runtime.cost_tracker
        return CostTracker(max_cost_usd=self._runtime.settings.max_run_cost_usd)

    def _screen_inputs_if_needed(
        self,
        query: str,
        chat_history: Sequence[ChatMessage] | None,
        *,
        custom_instructions: str | None = None,
        pre_screened: bool,
    ) -> tuple[str, list[ChatMessage], str | None]:
        history_list = list(chat_history) if chat_history else []
        cleaned_instructions = (
            custom_instructions.strip()
            if custom_instructions and custom_instructions.strip()
            else None
        )
        if pre_screened or not self._runtime.settings.guardrails_enabled:
            return query, history_list, cleaned_instructions
        settings = self._runtime.settings
        cfg = InputGuardrailConfig(
            enabled=settings.guardrails_enabled,
            block_prompt_injection=settings.guardrails_block_prompt_injection,
            pii_mode=settings.guardrails_pii_mode,
        )
        decision = screen_query_input(
            query,
            history_list,
            custom_instructions=cleaned_instructions,
            config=cfg,
        )
        if not decision.allowed:
            raise GuardrailBlockedError(decision)
        return (
            decision.sanitized_query,
            decision.sanitized_history,
            decision.sanitized_custom_instructions,
        )

    def _build_run_config(
        self,
        *,
        models: ModelBundle | None = None,
        temperature: float | None = None,
        cost_tracker: CostTracker | None = None,
        callbacks: Sequence[Any] | None = None,
        trace_handle: TraceRunHandle | None = None,
    ) -> RunnableConfig:
        merged_callbacks: list[Any] = []
        if callbacks:
            merged_callbacks.extend(callbacks)
        elif self._runtime.callbacks:
            merged_callbacks.extend(self._runtime.callbacks)

        if trace_handle is not None and trace_handle.callback_handler is not None:
            merged_callbacks.append(trace_handle.callback_handler)

        resolved_tracker = self._resolve_run_cost_tracker(cost_tracker)
        configurable: dict[str, Any] = {"cost_tracker": resolved_tracker}
        if models is not None:
            configurable["models"] = models
        if temperature is not None:
            configurable["temperature"] = float(temperature)

        config: RunnableConfig = {"recursion_limit": 25}
        if merged_callbacks:
            config["callbacks"] = merged_callbacks
        if configurable:
            config["configurable"] = configurable
        if trace_handle is not None and trace_handle.langfuse_metadata:
            config["metadata"] = dict(trace_handle.langfuse_metadata)
        return config

    async def arun(
        self,
        query: str,
        *,
        chat_history: Sequence[ChatMessage] | None = None,
        ablation: AblationConfig | None = None,
        custom_instructions: str | None = None,
        models: ModelBundle | None = None,
        temperature: float | None = None,
        cost_tracker: CostTracker | None = None,
        callbacks: Sequence[Any] | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        preset_name: str | None = None,
        trace_tags: Sequence[str] | None = None,
        pre_screened: bool = False,
    ) -> FinalAnswer:
        """Execute the full AdaptiveRAG graph and return a Glass-Box FinalAnswer."""
        started = time.perf_counter()
        safe_query, safe_history, safe_instructions = self._screen_inputs_if_needed(
            query,
            chat_history,
            custom_instructions=custom_instructions,
            pre_screened=pre_screened,
        )
        resolved_ablation = ablation or AblationConfig()
        initial_state = self._build_initial_state(
            safe_query,
            chat_history=safe_history,
            ablation=resolved_ablation,
            custom_instructions=safe_instructions,
        )
        observability = self._get_observability()
        trace_handle = observability.start_trace(
            query=initial_state["query"],
            session_id=session_id,
            user_id=user_id,
            preset_name=preset_name,
            ablation=resolved_ablation.model_dump(mode="json"),
            tags=list(trace_tags) if trace_tags else ["mode:batch"],
        )
        run_config = self._build_run_config(
            models=models,
            temperature=temperature,
            cost_tracker=cost_tracker,
            callbacks=callbacks,
            trace_handle=trace_handle,
        )
        try:
            final_state = await self._graph.ainvoke(initial_state, config=run_config)
        except Exception as exc:
            observability.finish_trace(trace_handle, error=exc)
            raise

        total_ms = (time.perf_counter() - started) * 1000.0
        final_answer = _state_to_final_answer(
            dict(final_state),
            total_ms,
            trace_id=trace_handle.trace_id,
            trace_url=trace_handle.trace_url,
        )
        observability.finish_trace(trace_handle, answer=final_answer)
        return final_answer

    async def astream_events(
        self,
        query: str,
        *,
        chat_history: Sequence[ChatMessage] | None = None,
        ablation: AblationConfig | None = None,
        custom_instructions: str | None = None,
        models: ModelBundle | None = None,
        temperature: float | None = None,
        cost_tracker: CostTracker | None = None,
        callbacks: Sequence[Any] | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        preset_name: str | None = None,
        trace_tags: Sequence[str] | None = None,
        pre_screened: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream structured node_update events followed by a final_answer payload (SSE-ready)."""
        started = time.perf_counter()
        safe_query, safe_history, safe_instructions = self._screen_inputs_if_needed(
            query,
            chat_history,
            custom_instructions=custom_instructions,
            pre_screened=pre_screened,
        )
        resolved_ablation = ablation or AblationConfig()
        initial_state = self._build_initial_state(
            safe_query,
            chat_history=safe_history,
            ablation=resolved_ablation,
            custom_instructions=safe_instructions,
        )
        observability = self._get_observability()
        trace_handle = observability.start_trace(
            query=initial_state["query"],
            session_id=session_id,
            user_id=user_id,
            preset_name=preset_name,
            ablation=resolved_ablation.model_dump(mode="json"),
            tags=list(trace_tags) if trace_tags else ["mode:stream"],
        )
        run_config = self._build_run_config(
            models=models,
            temperature=temperature,
            cost_tracker=cost_tracker,
            callbacks=callbacks,
            trace_handle=trace_handle,
        )

        accumulated: dict[str, Any] = dict(initial_state)
        accumulated["trace"] = []
        accumulated["usage"] = TokenUsage()
        completed = False

        try:
            async for chunk in self._graph.astream(
                initial_state,
                config=run_config,
                stream_mode="updates",
            ):
                if not isinstance(chunk, dict):
                    continue
                for node_name, delta in chunk.items():
                    if not isinstance(delta, dict):
                        continue

                    node_traces: list[NodeTraceEvent] = list(delta.get("trace") or [])
                    node_usage: TokenUsage = delta.get("usage") or TokenUsage()

                    for key, value in delta.items():
                        if key == "trace":
                            accumulated["trace"].extend(node_traces)
                        elif key == "usage":
                            accumulated["usage"] = merge_token_usage(
                                accumulated["usage"], node_usage
                            )
                        else:
                            accumulated[key] = value

                    yield {
                        "event": "node_update",
                        "node": node_name,
                        "trace": [t.model_dump(mode="json") for t in node_traces],
                        "usage": node_usage.model_dump(mode="json"),
                    }

            total_ms = (time.perf_counter() - started) * 1000.0
            final_answer = _state_to_final_answer(
                accumulated,
                total_ms,
                trace_id=trace_handle.trace_id,
                trace_url=trace_handle.trace_url,
            )
            observability.finish_trace(trace_handle, answer=final_answer)
            completed = True
            yield {
                "event": "final_answer",
                "data": final_answer.model_dump(mode="json"),
            }
        except Exception as exc:
            observability.finish_trace(trace_handle, error=exc)
            completed = True
            raise
        finally:
            if not completed:
                # Ensure open root spans are closed if the SSE client disconnects mid-stream.
                observability.finish_trace(trace_handle)
