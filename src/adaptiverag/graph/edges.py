"""Pure conditional routing functions enforcing bounded cycles in the AdaptiveRAG graph."""

from __future__ import annotations

from typing import Literal

from adaptiverag.graph.state import AblationConfig, AgentState
from adaptiverag.schemas.citation import SemanticAuditStatus
from adaptiverag.schemas.query import QueryType
from adaptiverag.schemas.retrieval import ConfidenceAction

PlanRoute = Literal["conversational_respond", "decompose_queries", "hybrid_retrieve"]
ConfidenceRoute = Literal[
    "generate_answer",
    "corrective_rewrite",
    "web_fallback_search",
    "abstain_response",
]
WebFallbackRoute = Literal["generate_answer", "abstain_response"]
CitationRoute = Literal["regenerate_strict", "end"]


def route_after_plan(state: AgentState) -> PlanRoute:
    """Route conversational queries directly, analytical to decomposition, factual to hybrid search."""
    plan = state.get("query_plan")
    ablation = state.get("ablation") or AblationConfig()
    if plan is None:
        return "hybrid_retrieve"

    if not plan.needs_retrieval or plan.query_type == QueryType.CONVERSATIONAL:
        return "conversational_respond"

    if (
        plan.query_type == QueryType.ANALYTICAL
        and ablation.enable_multi_query
        and plan.strategy.max_sub_queries > 1
    ):
        return "decompose_queries"

    return "hybrid_retrieve"


def route_after_confidence(
    state: AgentState,
    *,
    max_rewrites: int,
    web_fallback_enabled: bool,
) -> ConfidenceRoute:
    """Select next control-flow branch after confidence scoring with strict loop guards."""
    ablation = state.get("ablation") or AblationConfig()
    selected = state.get("selected_chunks") or []
    confidence = state.get("confidence")
    rewrite_attempts = state.get("rewrite_attempts", 0)
    fallback_used = state.get("fallback_used", False)
    can_web_fallback = web_fallback_enabled and ablation.enable_web_fallback and not fallback_used

    if not ablation.enable_confidence_gating:
        return "generate_answer" if selected else "abstain_response"

    if confidence is None:
        return "generate_answer" if selected else "abstain_response"

    action = confidence.recommended_action

    if action == ConfidenceAction.PROCEED:
        return "generate_answer" if selected else "abstain_response"

    if action == ConfidenceAction.REWRITE:
        if rewrite_attempts < max_rewrites:
            return "corrective_rewrite"
        if can_web_fallback:
            return "web_fallback_search"
        return "generate_answer" if selected else "abstain_response"

    if action == ConfidenceAction.WEB_FALLBACK:
        if can_web_fallback:
            return "web_fallback_search"
        return "generate_answer" if selected else "abstain_response"

    return "abstain_response"


def route_after_web_fallback(state: AgentState) -> WebFallbackRoute:
    """Proceed to generation when web fallback yielded evidence; otherwise abstain honestly."""
    selected = state.get("selected_chunks") or []
    return "generate_answer" if selected else "abstain_response"


def route_after_citation_validation(
    state: AgentState,
    *,
    max_regenerations: int,
) -> CitationRoute:
    """Trigger bounded strict regeneration when citation verification fails."""
    ablation = state.get("ablation") or AblationConfig()
    if not ablation.enable_citation_validation:
        return "end"

    if state.get("abstained", False):
        return "end"

    report = state.get("citation_report")
    if report is None or report.passed:
        return "end"

    # A failed judge is not a failed answer: regenerating cannot fix it and only adds cost.
    # Regenerate only when the answer itself has problems the strict prompt can address.
    if report.semantic_audit == SemanticAuditStatus.FAILED and report.structural_passed:
        return "end"

    regeneration_attempts = state.get("regeneration_attempts", 0)
    if regeneration_attempts < max_regenerations:
        return "regenerate_strict"

    return "end"
