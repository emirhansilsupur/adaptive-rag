"""Async LangGraph node implementations with per-node latency, cost & trace telemetry."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig

from adaptiverag.config import Settings, StrategiesConfig
from adaptiverag.graph.prompts import (
    build_abstention_message,
    detect_language_heuristic,
    resolve_conversational_prompt,
    resolve_corrective_rewrite_prompt,
    resolve_decompose_prompt,
    resolve_generation_prompt,
    resolve_router_prompt,
    resolve_strict_regeneration_prompt,
)
from adaptiverag.graph.state import AblationConfig, AgentState
from adaptiverag.ingestion.chunker import count_tokens
from adaptiverag.llm.cache import SQLiteLLMCache
from adaptiverag.llm.cost import BudgetExceededError, CostTracker
from adaptiverag.llm.factory import ModelBundle
from adaptiverag.llm.structured import invoke_structured, invoke_text
from adaptiverag.observability.prompts import PromptManager, ResolvedPrompt
from adaptiverag.observability.tracing import LangfuseObservability
from adaptiverag.retrieval.dynamic_k import assign_citation_ids, select_dynamic_top_k
from adaptiverag.retrieval.fallback import FallbackRetriever
from adaptiverag.retrieval.parent_child import expand_to_parent_chunks
from adaptiverag.retrieval.qdrant_store import QdrantHybridStore
from adaptiverag.retrieval.reranker import Reranker
from adaptiverag.schemas.citation import CitationReport
from adaptiverag.schemas.query import (
    CorrectiveRewriteResult,
    QueryPlan,
    QueryType,
    RouterClassification,
    SubQueryDecomposition,
)
from adaptiverag.schemas.response import NodeTraceEvent, TokenUsage
from adaptiverag.schemas.retrieval import (
    ConfidenceAction,
    ConfidenceAssessment,
    ConfidenceLevel,
    CRAGGrade,
    DynamicKDecision,
    EvidenceChunk,
)
from adaptiverag.verification.citation_validator import (
    validate_citations,
    validate_citations_structural,
)
from adaptiverag.verification.confidence import (
    assess_retrieval_confidence,
    compute_deterministic_confidence,
)

logger = structlog.get_logger(__name__)

NodeFn = Callable[[AgentState, RunnableConfig], Awaitable[dict[str, Any]]]


@dataclass
class GraphRuntime:
    """Dependency container injected into all AdaptiveRAG LangGraph nodes."""

    settings: Settings
    strategies: StrategiesConfig
    models: ModelBundle
    store: QdrantHybridStore
    reranker: Reranker
    fallback_retriever: FallbackRetriever
    cache: SQLiteLLMCache | None = None
    cost_tracker: CostTracker | None = None
    callbacks: list[Any] | None = None
    observability: LangfuseObservability | None = None
    prompt_manager: PromptManager | None = None


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000.0, 2)


def _resolve_controls(
    runtime: GraphRuntime,
    config: RunnableConfig | None,
) -> tuple[CostTracker | None, list[Any] | None]:
    configurable = (config or {}).get("configurable") or {}
    cost_tracker = configurable.get("cost_tracker", runtime.cost_tracker)

    # When graph-level callbacks are already configured on RunnableConfig, LangChain
    # automatically propagates them via contextvars; only forward explicit callbacks
    # if none are already attached at the graph level.
    graph_callbacks = (config or {}).get("callbacks")
    if graph_callbacks:
        return cost_tracker, None

    req_callbacks = configurable.get("callbacks")
    if req_callbacks is not None:
        return cost_tracker, list(req_callbacks)
    if runtime.callbacks:
        return cost_tracker, list(runtime.callbacks)
    return cost_tracker, None


def _effective_chunk_score(chunk: EvidenceChunk) -> float:
    if chunk.rerank_score is not None:
        return min(1.0, max(0.0, float(chunk.rerank_score)))
    if chunk.fusion_score is not None:
        return min(1.0, max(0.0, float(chunk.fusion_score)))
    return 0.0


def _chunk_tokens(chunk: EvidenceChunk) -> int:
    return chunk.token_count if chunk.token_count > 0 else max(1, count_tokens(chunk.text))


def _deduplicate_queries(queries: Sequence[str], *, limit: int) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in queries:
        q = raw.strip()
        key = q.lower()
        if not q or key in seen:
            continue
        seen.add(key)
        cleaned.append(q)
        if len(cleaned) >= limit:
            break
    return cleaned


async def _rerank_against_active_queries(
    reranker: Reranker,
    *,
    standalone_query: str,
    active_queries: Sequence[str],
    candidates: Sequence[EvidenceChunk],
    per_query_cap: int = 12,
) -> list[EvidenceChunk]:
    """Rerank candidates against active (decomposed or rewritten) queries using max-score fusion.

    Single-query runs execute in one cross-encoder pass. Multi-query analytical runs and
    corrective rewrites score each active query against the candidates it matched (falling back
    to the top candidates when sub-query provenance is absent) and retain each chunk's peak
    relevance score so multi-hop evidence and cross-lingual rewrites are never penalized against
    a stale initial query.
    """
    queries = _deduplicate_queries(active_queries, limit=5) or [standalone_query]
    if len(queries) == 1:
        return await reranker.arerank(queries[0], candidates)

    best_by_id: dict[str, EvidenceChunk] = {}
    for q in queries:
        matched = [c for c in candidates if q in c.matched_sub_queries]
        sub_candidates = matched[:per_query_cap] if matched else list(candidates[:per_query_cap])
        if not sub_candidates:
            continue
        scored = await reranker.arerank(q, sub_candidates)
        for chunk in scored:
            prev = best_by_id.get(chunk.chunk_id)
            if prev is None or (chunk.rerank_score or 0.0) > (prev.rerank_score or 0.0):
                best_by_id[chunk.chunk_id] = chunk

    for chunk in candidates:
        best_by_id.setdefault(chunk.chunk_id, chunk)

    ranked = list(best_by_id.values())
    ranked.sort(
        key=lambda c: (
            c.rerank_score if c.rerank_score is not None else -1.0,
            c.fusion_score if c.fusion_score is not None else -1.0,
        ),
        reverse=True,
    )
    return ranked


async def _resolve_node_prompt[R](
    runtime: GraphRuntime,
    resolver: Callable[..., R],
    *args: Any,
    **kwargs: Any,
) -> R:
    """Offload prompt resolution to a worker thread only when remote Langfuse fetching is active."""
    pm = runtime.prompt_manager
    if pm is not None and pm.active:
        return await asyncio.to_thread(resolver, *args, prompt_manager=pm, **kwargs)
    return resolver(*args, prompt_manager=pm, **kwargs)


def create_nodes(runtime: GraphRuntime) -> dict[str, NodeFn]:
    """Create all bound async LangGraph node callables for the AdaptiveRAG state machine."""

    async def classify_and_plan(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        started = time.perf_counter()
        query = state["query"].strip()
        chat_history = state.get("chat_history") or []
        ablation = state.get("ablation") or AblationConfig()
        cost_tracker, callbacks = _resolve_controls(runtime, config)

        node_usage = TokenUsage()

        if not ablation.enable_query_routing:
            lang = detect_language_heuristic(query)
            strategy = runtime.strategies.for_query_type(QueryType.FACTUAL)
            plan = QueryPlan(
                original_query=query,
                standalone_query=query,
                query_type=QueryType.FACTUAL,
                needs_retrieval=True,
                language=lang,
                requires_fresh_web_info=False,
                reasoning="Query routing disabled by ablation config; defaulting to factual strategy.",
                strategy=strategy,
                sub_queries=[query],
            )
            event = NodeTraceEvent(
                node="classify_and_plan",
                latency_ms=_elapsed_ms(started),
                summary=f"Static factual route (routing disabled, lang={lang})",
                metadata={
                    "query_type": plan.query_type.value,
                    "needs_retrieval": plan.needs_retrieval,
                    "language": plan.language,
                    "routing_enabled": False,
                },
            )
            return {
                "query_plan": plan,
                "active_queries": [query],
                "rewrite_attempts": 0,
                "regeneration_attempts": 0,
                "fallback_used": False,
                "abstained": False,
                "trace": [event],
            }

        messages, resolved_prompt = await _resolve_node_prompt(
            runtime,
            resolve_router_prompt,
            query,
            chat_history,
        )
        try:
            res = await invoke_structured(
                runtime.models.fast_model,
                RouterClassification,
                messages,
                model_name=runtime.models.fast_model_name,
                temperature=runtime.settings.llm_temperature,
                max_repair_attempts=1,
                cache=runtime.cache,
                cost_tracker=cost_tracker,
                callbacks=callbacks,
                langfuse_prompt=resolved_prompt.langfuse_prompt,
                prompt_metadata=resolved_prompt.telemetry_metadata(),
            )
            node_usage = res.usage
            cls = res.parsed
            strategy = runtime.strategies.for_query_type(cls.query_type)
            needs_retrieval = (
                False
                if cls.query_type == QueryType.CONVERSATIONAL
                else bool(cls.needs_retrieval and strategy.needs_retrieval)
            )
            standalone = cls.standalone_query.strip() or query
            plan = QueryPlan(
                original_query=query,
                standalone_query=standalone,
                query_type=cls.query_type,
                needs_retrieval=needs_retrieval,
                language=cls.language,
                requires_fresh_web_info=cls.requires_fresh_web_info,
                reasoning=cls.reasoning,
                strategy=strategy,
                sub_queries=[standalone] if needs_retrieval else [],
            )
        except BudgetExceededError:
            raise
        except Exception as exc:
            logger.warning("query_router_failed_falling_back_to_factual", error=str(exc))
            lang = detect_language_heuristic(query)
            strategy = runtime.strategies.for_query_type(QueryType.FACTUAL)
            plan = QueryPlan(
                original_query=query,
                standalone_query=query,
                query_type=QueryType.FACTUAL,
                needs_retrieval=True,
                language=lang,
                requires_fresh_web_info=False,
                reasoning=f"Router fallback to factual after error: {type(exc).__name__}",
                strategy=strategy,
                sub_queries=[query],
            )

        event = NodeTraceEvent(
            node="classify_and_plan",
            latency_ms=_elapsed_ms(started),
            summary=(
                f"Classified as {plan.query_type.value} "
                f"(retrieval={plan.needs_retrieval}, lang={plan.language}, "
                f"fresh_web={plan.requires_fresh_web_info})"
            ),
            metadata={
                "query_type": plan.query_type.value,
                "needs_retrieval": plan.needs_retrieval,
                "language": plan.language,
                "standalone_query": plan.standalone_query,
                "requires_fresh_web_info": plan.requires_fresh_web_info,
                "reasoning": plan.reasoning,
                "prompt": resolved_prompt.telemetry_metadata(),
            },
        )
        return {
            "query_plan": plan,
            "active_queries": list(plan.sub_queries),
            "rewrite_attempts": 0,
            "regeneration_attempts": 0,
            "fallback_used": False,
            "abstained": False,
            "usage": node_usage,
            "trace": [event],
        }

    async def conversational_respond(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        started = time.perf_counter()
        plan = state["query_plan"]
        chat_history = state.get("chat_history") or []
        custom_instructions = state.get("custom_instructions")
        cost_tracker, callbacks = _resolve_controls(runtime, config)

        messages, resolved_prompt = await _resolve_node_prompt(
            runtime,
            resolve_conversational_prompt,
            query=state["query"],
            language=plan.language,
            chat_history=chat_history,
            custom_instructions=custom_instructions,
        )
        res = await invoke_text(
            runtime.models.fast_model,
            messages,
            model_name=runtime.models.fast_model_name,
            temperature=runtime.settings.llm_temperature,
            cache=runtime.cache,
            cost_tracker=cost_tracker,
            callbacks=callbacks,
            langfuse_prompt=resolved_prompt.langfuse_prompt,
            prompt_metadata=resolved_prompt.telemetry_metadata(),
        )

        event = NodeTraceEvent(
            node="conversational_respond",
            latency_ms=_elapsed_ms(started),
            summary=f"Direct conversational reply via {runtime.models.fast_model_name} (0 retrieval calls)",
            metadata={
                "model": runtime.models.fast_model_name,
                "from_cache": res.from_cache,
                "custom_instructions_active": bool(
                    custom_instructions and custom_instructions.strip()
                ),
                "prompt": resolved_prompt.telemetry_metadata(),
            },
        )
        return {
            "draft_answer": res.text.strip(),
            "abstained": False,
            "usage": res.usage,
            "trace": [event],
        }

    async def decompose_queries(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        started = time.perf_counter()
        plan = state["query_plan"]
        cost_tracker, callbacks = _resolve_controls(runtime, config)
        max_subs = max(1, plan.strategy.max_sub_queries)

        node_usage = TokenUsage()
        sub_queries = [plan.standalone_query]
        rationale = ""

        messages, resolved_prompt = await _resolve_node_prompt(
            runtime,
            resolve_decompose_prompt,
            plan.standalone_query,
            language=plan.language,
            max_sub_queries=max_subs,
        )
        try:
            res = await invoke_structured(
                runtime.models.fast_model,
                SubQueryDecomposition,
                messages,
                model_name=runtime.models.fast_model_name,
                temperature=runtime.settings.llm_temperature,
                max_repair_attempts=1,
                cache=runtime.cache,
                cost_tracker=cost_tracker,
                callbacks=callbacks,
                langfuse_prompt=resolved_prompt.langfuse_prompt,
                prompt_metadata=resolved_prompt.telemetry_metadata(),
            )
            node_usage = res.usage
            deduped = _deduplicate_queries(res.parsed.sub_queries, limit=max_subs)
            if deduped:
                sub_queries = deduped
            rationale = res.parsed.rationale
        except BudgetExceededError:
            raise
        except Exception as exc:
            logger.warning("decompose_queries_failed_using_standalone", error=str(exc))

        updated_plan = plan.model_copy(update={"sub_queries": sub_queries})
        event = NodeTraceEvent(
            node="decompose_queries",
            latency_ms=_elapsed_ms(started),
            summary=f"Decomposed analytical query into {len(sub_queries)} sub-queries",
            metadata={
                "sub_queries": sub_queries,
                "rationale": rationale,
                "prompt": resolved_prompt.telemetry_metadata(),
            },
        )
        return {
            "query_plan": updated_plan,
            "active_queries": sub_queries,
            "usage": node_usage,
            "trace": [event],
        }

    async def hybrid_retrieve(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        del config
        started = time.perf_counter()
        plan = state["query_plan"]
        ablation = state.get("ablation") or AblationConfig()
        queries = state.get("active_queries") or [plan.standalone_query]
        strategy = plan.strategy

        max_total = max(strategy.fusion_limit * 2, 30)
        chunks = await runtime.store.search_multi_query(
            queries,
            mode=ablation.retrieval_mode,
            dense_limit=strategy.dense_prefetch_limit,
            sparse_limit=strategy.sparse_prefetch_limit,
            fusion_limit=strategy.fusion_limit,
            max_total_candidates=max_total,
            doc_ids=ablation.doc_ids,
            language=ablation.language_filter,
        )

        scope_suffix = f" [scoped to {len(ablation.doc_ids)} doc(s)]" if ablation.doc_ids else ""
        event = NodeTraceEvent(
            node="hybrid_retrieve",
            latency_ms=_elapsed_ms(started),
            summary=(
                f"Retrieved {len(chunks)} candidates via {ablation.retrieval_mode.value} "
                f"across {len(queries)} query(s){scope_suffix}"
            ),
            metadata={
                "mode": ablation.retrieval_mode.value,
                "queries": list(queries),
                "candidate_count": len(chunks),
                "rewrite_attempt": state.get("rewrite_attempts", 0),
                "doc_ids": list(ablation.doc_ids) if ablation.doc_ids else None,
            },
        )
        return {
            "retrieved_chunks": chunks,
            "trace": [event],
        }

    async def rerank_and_select_k(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        del config
        started = time.perf_counter()
        plan = state["query_plan"]
        ablation = state.get("ablation") or AblationConfig()
        candidates = state.get("retrieved_chunks") or []

        active_queries = state.get("active_queries") or [plan.standalone_query]
        if ablation.enable_rerank and candidates:
            ranked = await _rerank_against_active_queries(
                runtime.reranker,
                standalone_query=plan.standalone_query,
                active_queries=active_queries,
                candidates=candidates,
            )
        else:
            ranked = list(candidates)

        if ablation.enable_dynamic_k:
            selected, decision = select_dynamic_top_k(ranked, plan.strategy)
        else:
            k = max(1, ablation.fixed_top_k)
            selected = ranked[:k]
            top_score = _effective_chunk_score(selected[0]) if selected else 0.0
            cutoff_score = _effective_chunk_score(selected[-1]) if selected else 0.0
            total_tokens = sum(_chunk_tokens(c) for c in selected)
            decision = DynamicKDecision(
                candidate_count=len(ranked),
                selected_count=len(selected),
                cutoff_reason="static_top_k" if selected else "no_candidates",
                top_score=round(top_score, 6),
                cutoff_score=round(cutoff_score, 6),
                total_context_tokens=total_tokens,
                dropped_chunk_ids=[c.chunk_id for c in ranked[k:]],
            )

        use_parent_child = (
            ablation.enable_parent_child
            if ablation.enable_parent_child is not None
            else runtime.settings.enable_parent_child_retrieval
        )
        if use_parent_child and selected:
            selected, expanded_decision = expand_to_parent_chunks(
                selected,
                max_context_tokens=plan.strategy.max_context_tokens,
                decision=decision,
            )
            if expanded_decision is not None:
                decision = expanded_decision

        labeled = assign_citation_ids(selected, prefix="C", start_index=1)
        parent_suffix = (
            f", expanded {decision.collapsed_child_count} child->{decision.expanded_parent_count} parent"
            if decision.parent_expansion_applied
            else ""
        )
        event = NodeTraceEvent(
            node="rerank_and_select_k",
            latency_ms=_elapsed_ms(started),
            summary=(
                f"Selected {len(labeled)}/{len(candidates)} chunks "
                f"(reason={decision.cutoff_reason}, top={decision.top_score:.3f}, "
                f"tokens={decision.total_context_tokens}{parent_suffix})"
            ),
            metadata={
                "rerank_enabled": ablation.enable_rerank,
                "dynamic_k_enabled": ablation.enable_dynamic_k,
                "parent_child_enabled": use_parent_child,
                "parent_expansion_applied": decision.parent_expansion_applied,
                "collapsed_child_count": decision.collapsed_child_count,
                "expanded_parent_count": decision.expanded_parent_count,
                "cutoff_reason": decision.cutoff_reason,
                "selected_count": decision.selected_count,
                "candidate_count": decision.candidate_count,
                "top_score": decision.top_score,
                "cutoff_score": decision.cutoff_score,
                "total_context_tokens": decision.total_context_tokens,
            },
        )
        return {
            "selected_chunks": labeled,
            "dynamic_k_decision": decision,
            "trace": [event],
        }

    async def score_confidence(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        started = time.perf_counter()
        plan = state["query_plan"]
        ablation = state.get("ablation") or AblationConfig()
        selected = state.get("selected_chunks") or []
        active_queries = state.get("active_queries") or [plan.standalone_query]
        rewrite_attempts = state.get("rewrite_attempts", 0)
        fallback_used = state.get("fallback_used", False)
        cost_tracker, callbacks = _resolve_controls(runtime, config)

        node_usage = TokenUsage()
        crag_prompts: list[ResolvedPrompt] = []
        if not ablation.enable_confidence_gating:
            signals = compute_deterministic_confidence(
                selected, active_queries, runtime.strategies.confidence
            )
            action = ConfidenceAction.PROCEED if selected else ConfidenceAction.ABSTAIN
            level = (
                ConfidenceLevel.HIGH
                if signals.deterministic_score >= runtime.strategies.confidence.high_threshold
                else (
                    ConfidenceLevel.MEDIUM
                    if signals.deterministic_score >= runtime.strategies.confidence.low_threshold
                    else ConfidenceLevel.LOW
                )
            )
            assessment = ConfidenceAssessment(
                top1_score=signals.top1_score,
                mean_topk_score=signals.mean_topk_score,
                subquery_coverage=signals.subquery_coverage,
                deterministic_score=signals.deterministic_score,
                crag_grade=None,
                final_score=signals.deterministic_score,
                level=level,
                recommended_action=action,
                reasoning="Confidence gating disabled by ablation config.",
            )
        else:
            web_enabled = (
                runtime.settings.web_fallback_enabled
                and ablation.enable_web_fallback
                and not fallback_used
            )
            assessment, node_usage = await assess_retrieval_confidence(
                query=plan.standalone_query,
                selected_chunks=selected,
                sub_queries=active_queries,
                config=runtime.strategies.confidence,
                rewrite_attempts=rewrite_attempts,
                web_fallback_enabled=web_enabled,
                requires_fresh_web_info=plan.requires_fresh_web_info,
                fast_model=runtime.models.fast_model,
                fast_model_name=runtime.models.fast_model_name,
                temperature=runtime.settings.llm_temperature,
                cache=runtime.cache,
                cost_tracker=cost_tracker,
                callbacks=callbacks,
                prompt_manager=runtime.prompt_manager,
                prompt_out=crag_prompts,
            )

        confidence_meta: dict[str, Any] = {
            "deterministic_score": assessment.deterministic_score,
            "final_score": assessment.final_score,
            "top1_score": assessment.top1_score,
            "mean_topk_score": assessment.mean_topk_score,
            "subquery_coverage": assessment.subquery_coverage,
            "crag_grade": assessment.crag_grade.value if assessment.crag_grade else None,
            "level": assessment.level.value,
            "recommended_action": assessment.recommended_action.value,
        }
        if crag_prompts:
            confidence_meta["prompt"] = crag_prompts[-1].telemetry_metadata()

        event = NodeTraceEvent(
            node="score_confidence",
            latency_ms=_elapsed_ms(started),
            summary=(
                f"Confidence {assessment.level.value.upper()} ({assessment.final_score:.3f}) -> "
                f"{assessment.recommended_action.value}"
            ),
            metadata=confidence_meta,
        )
        return {
            "confidence": assessment,
            "usage": node_usage,
            "trace": [event],
        }

    async def corrective_rewrite(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        started = time.perf_counter()
        plan = state["query_plan"]
        prior_queries = state.get("active_queries") or [plan.standalone_query]
        confidence = state.get("confidence")
        attempt = state.get("rewrite_attempts", 0) + 1
        cost_tracker, callbacks = _resolve_controls(runtime, config)

        node_usage = TokenUsage()
        rewritten = list(prior_queries)
        diagnosis = ""

        messages, resolved_prompt = await _resolve_node_prompt(
            runtime,
            resolve_corrective_rewrite_prompt,
            original_query=plan.original_query,
            standalone_query=plan.standalone_query,
            prior_queries=prior_queries,
            confidence=confidence,
            language=plan.language,
        )
        try:
            res = await invoke_structured(
                runtime.models.fast_model,
                CorrectiveRewriteResult,
                messages,
                model_name=runtime.models.fast_model_name,
                temperature=runtime.settings.llm_temperature,
                max_repair_attempts=1,
                cache=runtime.cache,
                cost_tracker=cost_tracker,
                callbacks=callbacks,
                langfuse_prompt=resolved_prompt.langfuse_prompt,
                prompt_metadata=resolved_prompt.telemetry_metadata(),
            )
            node_usage = res.usage
            max_rewrite_queries = max(2, min(3, max(plan.strategy.max_sub_queries, 2)))
            deduped = _deduplicate_queries(res.parsed.rewritten_queries, limit=max_rewrite_queries)
            if deduped:
                rewritten = deduped
            diagnosis = res.parsed.diagnosis
        except BudgetExceededError:
            raise
        except Exception as exc:
            logger.warning("corrective_rewrite_failed_retaining_queries", error=str(exc))

        event = NodeTraceEvent(
            node="corrective_rewrite",
            latency_ms=_elapsed_ms(started),
            summary=f"Corrective rewrite #{attempt}: generated {len(rewritten)} reformulated query(s)",
            metadata={
                "attempt": attempt,
                "rewritten_queries": rewritten,
                "diagnosis": diagnosis,
                "prompt": resolved_prompt.telemetry_metadata(),
            },
        )
        return {
            "active_queries": rewritten,
            "rewrite_attempts": attempt,
            "usage": node_usage,
            "trace": [event],
        }

    async def web_fallback_search(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        del config
        started = time.perf_counter()
        plan = state["query_plan"]
        ablation = state.get("ablation") or AblationConfig()
        existing_local = state.get("selected_chunks") or []
        confidence = state.get("confidence")

        web_chunks = await runtime.fallback_retriever.asearch(
            plan.standalone_query,
            max_results=runtime.settings.web_fallback_max_results,
            start_citation_index=1,
        )

        if web_chunks and ablation.enable_rerank:
            web_chunks = await runtime.reranker.arerank(plan.standalone_query, web_chunks)
            min_web_score = max(0.15, plan.strategy.tau_min * 0.5)
            web_chunks = [
                c for c in web_chunks if c.rerank_score is None or c.rerank_score >= min_web_score
            ]
        web_chunks = assign_citation_ids(web_chunks, prefix="W", start_index=1)

        # Keep local chunks only when they were NOT graded insufficient/low-confidence.
        local_is_usable = confidence is None or (
            confidence.level != ConfidenceLevel.LOW
            and confidence.crag_grade != CRAGGrade.INSUFFICIENT
        )
        kept_local: list[EvidenceChunk] = []
        if local_is_usable and existing_local:
            local_limit = max(2, plan.strategy.rerank_min_k)
            qualified_local = [
                c for c in existing_local if _effective_chunk_score(c) >= plan.strategy.tau_min
            ][:local_limit]
            kept_local = assign_citation_ids(qualified_local, prefix="C", start_index=1)

        # Prioritize web chunks in token budget so fresh/fallback web evidence is never crowded out.
        prioritized = [*web_chunks, *kept_local]
        budgeted: list[EvidenceChunk] = []
        running_tokens = 0
        for idx, chunk in enumerate(prioritized):
            tokens = _chunk_tokens(chunk)
            if idx > 0 and (running_tokens + tokens) > plan.strategy.max_context_tokens:
                continue
            budgeted.append(chunk)
            running_tokens += tokens

        prev_decision = state.get("dynamic_k_decision")
        top_score = max((_effective_chunk_score(c) for c in budgeted), default=0.0)
        cutoff_score = _effective_chunk_score(budgeted[-1]) if budgeted else 0.0
        prior_candidates = prev_decision.candidate_count if prev_decision is not None else 0
        fallback_decision = DynamicKDecision(
            candidate_count=max(prior_candidates + len(web_chunks), len(budgeted)),
            selected_count=len(budgeted),
            cutoff_reason="web_fallback" if budgeted else "web_fallback_empty",
            top_score=round(top_score, 6),
            cutoff_score=round(cutoff_score, 6),
            total_context_tokens=running_tokens,
            dropped_chunk_ids=list(prev_decision.dropped_chunk_ids) if prev_decision else [],
        )

        event = NodeTraceEvent(
            node="web_fallback_search",
            latency_ms=_elapsed_ms(started),
            summary=(
                f"Web fallback returned {len(web_chunks)} chunk(s); "
                f"assembled {len(budgeted)} total evidence chunk(s) ({running_tokens} tokens)"
            ),
            metadata={
                "web_chunk_count": len(web_chunks),
                "retained_local_count": len(kept_local),
                "final_chunk_count": len(budgeted),
                "total_tokens": running_tokens,
            },
        )
        return {
            "selected_chunks": budgeted,
            "dynamic_k_decision": fallback_decision,
            "fallback_used": True,
            "trace": [event],
        }

    async def generate_answer(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        started = time.perf_counter()
        plan = state["query_plan"]
        chunks = state.get("selected_chunks") or []
        chat_history = state.get("chat_history") or []
        custom_instructions = state.get("custom_instructions")
        cost_tracker, callbacks = _resolve_controls(runtime, config)

        messages, resolved_prompt = await _resolve_node_prompt(
            runtime,
            resolve_generation_prompt,
            query=state["query"],
            standalone_query=plan.standalone_query,
            language=plan.language,
            chunks=chunks,
            chat_history=chat_history,
            query_type=plan.query_type,
            custom_instructions=custom_instructions,
        )
        res = await invoke_text(
            runtime.models.generation_model,
            messages,
            model_name=runtime.models.generation_model_name,
            temperature=runtime.settings.llm_temperature,
            cache=runtime.cache,
            cost_tracker=cost_tracker,
            callbacks=callbacks,
            langfuse_prompt=resolved_prompt.langfuse_prompt,
            prompt_metadata=resolved_prompt.telemetry_metadata(),
        )

        event = NodeTraceEvent(
            node="generate_answer",
            latency_ms=_elapsed_ms(started),
            summary=(
                f"Generated grounded answer via {runtime.models.generation_model_name} "
                f"using {len(chunks)} evidence chunk(s)"
            ),
            metadata={
                "model": runtime.models.generation_model_name,
                "evidence_count": len(chunks),
                "from_cache": res.from_cache,
                "custom_instructions_active": bool(
                    custom_instructions and custom_instructions.strip()
                ),
                "prompt": resolved_prompt.telemetry_metadata(),
            },
        )
        return {
            "draft_answer": res.text.strip(),
            "abstained": False,
            "usage": res.usage,
            "trace": [event],
        }

    async def validate_citations_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        started = time.perf_counter()
        ablation = state.get("ablation") or AblationConfig()
        draft_answer = state.get("draft_answer", "")
        chunks = state.get("selected_chunks") or []
        regen_attempts = state.get("regeneration_attempts", 0)
        cost_tracker, callbacks = _resolve_controls(runtime, config)

        node_usage = TokenUsage()
        entail_prompts: list[ResolvedPrompt] = []
        if not ablation.enable_citation_validation:
            report = validate_citations_structural(
                draft_answer,
                chunks,
                config=runtime.strategies.citation_validation,
            )
        else:
            report, node_usage = await validate_citations(
                answer=draft_answer,
                evidence_chunks=chunks,
                config=runtime.strategies.citation_validation,
                fast_model=runtime.models.fast_model,
                fast_model_name=runtime.models.fast_model_name,
                temperature=runtime.settings.llm_temperature,
                cache=runtime.cache,
                cost_tracker=cost_tracker,
                callbacks=callbacks,
                prompt_manager=runtime.prompt_manager,
                prompt_out=entail_prompts,
            )

        if regen_attempts > 0:
            report = report.model_copy(update={"regeneration_triggered": True})

        cite_meta: dict[str, Any] = {
            "passed": report.passed,
            "citation_precision": report.citation_precision,
            "claim_support_rate": report.claim_support_rate,
            "valid_citation_ids": report.valid_citation_ids,
            "invalid_citation_ids": report.invalid_citation_ids,
            "uncited_count": len(report.uncited_factual_sentences),
            "audited_claims": len(report.claim_verifications),
            "regeneration_attempts": regen_attempts,
            "validation_enabled": ablation.enable_citation_validation,
        }
        if entail_prompts:
            cite_meta["prompt"] = entail_prompts[-1].telemetry_metadata()

        event = NodeTraceEvent(
            node="validate_citations",
            latency_ms=_elapsed_ms(started),
            summary=(
                f"Citation audit {'PASSED' if report.passed else 'FAILED'} "
                f"(precision={report.citation_precision:.2f}, "
                f"support={report.claim_support_rate:.2f}, "
                f"invalid={len(report.invalid_citation_ids)})"
            ),
            metadata=cite_meta,
        )
        return {
            "citation_report": report,
            "usage": node_usage,
            "trace": [event],
        }

    async def regenerate_strict(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        started = time.perf_counter()
        plan = state["query_plan"]
        chunks = state.get("selected_chunks") or []
        previous_draft = state.get("draft_answer", "")
        report = state.get("citation_report") or CitationReport(passed=False)
        custom_instructions = state.get("custom_instructions")
        attempt = state.get("regeneration_attempts", 0) + 1
        cost_tracker, callbacks = _resolve_controls(runtime, config)

        messages, resolved_prompt = await _resolve_node_prompt(
            runtime,
            resolve_strict_regeneration_prompt,
            query=state["query"],
            standalone_query=plan.standalone_query,
            language=plan.language,
            chunks=chunks,
            previous_draft=previous_draft,
            report=report,
            custom_instructions=custom_instructions,
        )
        res = await invoke_text(
            runtime.models.generation_model,
            messages,
            model_name=runtime.models.generation_model_name,
            temperature=runtime.settings.llm_temperature,
            cache=runtime.cache,
            cost_tracker=cost_tracker,
            callbacks=callbacks,
            langfuse_prompt=resolved_prompt.langfuse_prompt,
            prompt_metadata=resolved_prompt.telemetry_metadata(),
        )

        event = NodeTraceEvent(
            node="regenerate_strict",
            latency_ms=_elapsed_ms(started),
            summary=(
                f"Strict regeneration #{attempt} via {runtime.models.generation_model_name} "
                f"to fix citation violations"
            ),
            metadata={
                "attempt": attempt,
                "model": runtime.models.generation_model_name,
                "from_cache": res.from_cache,
                "custom_instructions_active": bool(
                    custom_instructions and custom_instructions.strip()
                ),
                "prompt": resolved_prompt.telemetry_metadata(),
            },
        )
        return {
            "draft_answer": res.text.strip(),
            "regeneration_attempts": attempt,
            "usage": res.usage,
            "trace": [event],
        }

    async def abstain_response(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        del config
        started = time.perf_counter()
        plan = state.get("query_plan")
        language = plan.language if plan is not None else detect_language_heuristic(state["query"])
        confidence = state.get("confidence")
        fallback_used = state.get("fallback_used", False)

        answer = build_abstention_message(
            query=state["query"],
            language=language,
            confidence=confidence,
            fallback_used=fallback_used,
        )
        event = NodeTraceEvent(
            node="abstain_response",
            latency_ms=_elapsed_ms(started),
            summary=f"Honest abstention returned (lang={language}, fallback_used={fallback_used})",
            metadata={"language": language, "fallback_used": fallback_used},
        )
        return {
            "draft_answer": answer,
            "abstained": True,
            "trace": [event],
        }

    return {
        "classify_and_plan": classify_and_plan,
        "conversational_respond": conversational_respond,
        "decompose_queries": decompose_queries,
        "hybrid_retrieve": hybrid_retrieve,
        "rerank_and_select_k": rerank_and_select_k,
        "score_confidence": score_confidence,
        "corrective_rewrite": corrective_rewrite,
        "web_fallback_search": web_fallback_search,
        "generate_answer": generate_answer,
        "validate_citations": validate_citations_node,
        "regenerate_strict": regenerate_strict,
        "abstain_response": abstain_response,
    }
