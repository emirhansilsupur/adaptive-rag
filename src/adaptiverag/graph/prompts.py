"""Multilingual (TR + EN) prompt builders for routing, transformation, synthesis & recovery.

Supports both Langfuse-managed versioned prompts (via ``PromptManager``) and zero-downtime
local canonical fallback templates defined in ``adaptiverag.observability.prompts``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from adaptiverag.observability.prompts import (
    DEFAULT_PROMPTS,
    PROMPT_CONVERSATIONAL,
    PROMPT_CORRECTIVE_REWRITE,
    PROMPT_DECOMPOSE,
    PROMPT_GROUNDED_SYNTHESIS,
    PROMPT_ROUTER,
    PROMPT_STRICT_REGENERATION,
    PromptManager,
    ResolvedPrompt,
    compile_mustache_template,
)
from adaptiverag.schemas.citation import CitationReport, ClaimVerdict
from adaptiverag.schemas.query import ChatMessage, QueryType
from adaptiverag.schemas.retrieval import ConfidenceAssessment, EvidenceChunk, SourceType

_TR_DIACRITICS_RE = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
_TR_HINT_WORDS = {
    "nedir",
    "nasil",
    "neden",
    "hangi",
    "karsilastir",
    "arasindaki",
    "fark",
    "icin",
    "nelerdir",
    "merhaba",
    "selam",
    "tesekkur",
}

_CONTEXT_CHUNK_CHAR_LIMIT = 2400


def _resolve_system_prompt(
    name: str,
    *,
    variables: Mapping[str, Any] | None = None,
    prompt_manager: PromptManager | None = None,
) -> ResolvedPrompt:
    if prompt_manager is not None:
        return prompt_manager.resolve(name, variables=variables)
    defn = DEFAULT_PROMPTS[name]
    return ResolvedPrompt(
        name=name,
        text=compile_mustache_template(defn.template, variables),
        source="local_fallback",
        version=None,
        label=None,
        langfuse_prompt=None,
    )


def detect_language_heuristic(text: str) -> Literal["tr", "en"]:
    """Fast zero-cost language heuristic used when router is disabled or falls back."""
    if _TR_DIACRITICS_RE.search(text):
        return "tr"
    tokens = set(re.findall(r"[a-zA-Z]+", text.lower()))
    if tokens & _TR_HINT_WORDS:
        return "tr"
    return "en"


def _language_instruction(language: str) -> str:
    if language == "tr":
        return "Respond in natural, clear technical Turkish (Türkçe)."
    if language == "en":
        return "Respond in natural, clear technical English."
    return "Respond in the exact same language as the user's question."


def format_chat_history(chat_history: Sequence[ChatMessage], *, max_turns: int = 6) -> str:
    """Render recent conversation turns for coreference resolution or conversational replies."""
    if not chat_history:
        return "(no prior conversation)"
    recent = list(chat_history)[-max_turns:]
    return "\n".join(f"{msg.role.upper()}: {msg.content.strip()}" for msg in recent)


def format_evidence_context(chunks: Sequence[EvidenceChunk]) -> str:
    """Format local [C#] and web [W#] evidence passages with structural metadata."""
    if not chunks:
        return "(no evidence passages provided)"

    blocks: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        prefix = "W" if chunk.source_type == SourceType.WEB else "C"
        label = chunk.citation_id.strip().upper() if chunk.citation_id else f"{prefix}{idx}"
        header_parts = [p for p in (chunk.title, chunk.section_path) if p]
        if chunk.source_type == SourceType.WEB and chunk.url:
            header_parts.append(chunk.url)
        header = " | ".join(header_parts) if header_parts else chunk.doc_id
        body = chunk.text.strip()
        if len(body) > _CONTEXT_CHUNK_CHAR_LIMIT:
            body = body[:_CONTEXT_CHUNK_CHAR_LIMIT].rstrip() + "..."
        source_tag = "WEB" if chunk.source_type == SourceType.WEB else "LOCAL"
        blocks.append(f"[{label}] ({source_tag} | {header})\n{body}")

    return "\n\n---\n\n".join(blocks)


def resolve_router_prompt(
    query: str,
    chat_history: Sequence[ChatMessage],
    *,
    prompt_manager: PromptManager | None = None,
) -> tuple[list[BaseMessage], ResolvedPrompt]:
    """Resolve router prompt and build messages for classification & coreference resolution."""
    resolved = _resolve_system_prompt(PROMPT_ROUTER, prompt_manager=prompt_manager)
    user_prompt = (
        f"Conversation History:\n{format_chat_history(chat_history)}\n\n"
        f"Current User Query:\n{query.strip()}"
    )
    return [SystemMessage(content=resolved.text), HumanMessage(content=user_prompt)], resolved


def build_router_messages(
    query: str,
    chat_history: Sequence[ChatMessage],
    *,
    prompt_manager: PromptManager | None = None,
) -> list[BaseMessage]:
    """Build messages for query classification, coreference resolution & freshness detection."""
    messages, _ = resolve_router_prompt(query, chat_history, prompt_manager=prompt_manager)
    return messages


def resolve_decompose_prompt(
    standalone_query: str,
    *,
    language: str,
    max_sub_queries: int,
    prompt_manager: PromptManager | None = None,
) -> tuple[list[BaseMessage], ResolvedPrompt]:
    """Resolve decomposition prompt and build messages for analytical multi-query splitting."""
    target_max = max(2, min(5, max_sub_queries))
    resolved = _resolve_system_prompt(
        PROMPT_DECOMPOSE,
        variables={"max_sub_queries": target_max, "language": language},
        prompt_manager=prompt_manager,
    )
    user_prompt = f"Analytical Question:\n{standalone_query.strip()}"
    return [SystemMessage(content=resolved.text), HumanMessage(content=user_prompt)], resolved


def build_decompose_messages(
    standalone_query: str,
    *,
    language: str,
    max_sub_queries: int,
    prompt_manager: PromptManager | None = None,
) -> list[BaseMessage]:
    """Build messages for analytical multi-query decomposition."""
    messages, _ = resolve_decompose_prompt(
        standalone_query,
        language=language,
        max_sub_queries=max_sub_queries,
        prompt_manager=prompt_manager,
    )
    return messages


def resolve_corrective_rewrite_prompt(
    *,
    original_query: str,
    standalone_query: str,
    prior_queries: Sequence[str],
    confidence: ConfidenceAssessment | None,
    language: str,
    prompt_manager: PromptManager | None = None,
) -> tuple[list[BaseMessage], ResolvedPrompt]:
    """Resolve corrective rewrite prompt and build messages after low/medium confidence."""
    cross_lingual_rule = (
        f"- Because technical knowledge bases often mix Turkish and English documentation, "
        f"write AT LEAST ONE query in canonical English technical terminology and keep the remaining "
        f"query(s) in {language} using standard English acronyms/identifiers."
        if language != "en"
        else "- Use canonical English technical terminology, exact parameter/class names, and synonyms."
    )
    resolved = _resolve_system_prompt(
        PROMPT_CORRECTIVE_REWRITE,
        variables={"cross_lingual_rule": cross_lingual_rule},
        prompt_manager=prompt_manager,
    )

    prior_lines = "\n".join(f"  - {q}" for q in prior_queries) if prior_queries else "  - (none)"
    diag_text = confidence.reasoning if confidence is not None else "Weak initial retrieval scores."
    score_text = f"{confidence.final_score:.2f}" if confidence is not None else "unknown"

    user_prompt = (
        f"Original User Question: {original_query.strip()}\n"
        f"Standalone Query: {standalone_query.strip()}\n"
        f"Prior Search Queries:\n{prior_lines}\n"
        f"Confidence Score: {score_text}\n"
        f"Confidence Diagnosis: {diag_text}"
    )
    return [SystemMessage(content=resolved.text), HumanMessage(content=user_prompt)], resolved


def build_corrective_rewrite_messages(
    *,
    original_query: str,
    standalone_query: str,
    prior_queries: Sequence[str],
    confidence: ConfidenceAssessment | None,
    language: str,
    prompt_manager: PromptManager | None = None,
) -> list[BaseMessage]:
    """Build messages for corrective query reformulation after low/medium confidence."""
    messages, _ = resolve_corrective_rewrite_prompt(
        original_query=original_query,
        standalone_query=standalone_query,
        prior_queries=prior_queries,
        confidence=confidence,
        language=language,
        prompt_manager=prompt_manager,
    )
    return messages


def resolve_generation_prompt(
    *,
    query: str,
    standalone_query: str,
    language: str,
    chunks: Sequence[EvidenceChunk],
    chat_history: Sequence[ChatMessage],
    query_type: QueryType,
    prompt_manager: PromptManager | None = None,
) -> tuple[list[BaseMessage], ResolvedPrompt]:
    """Resolve grounded synthesis prompt and build strictly cited generation messages."""
    style_hint = (
        "Structure your response clearly to cover comparisons, trade-offs, and key dimensions."
        if query_type == QueryType.ANALYTICAL
        else "Keep the response direct, precise, and technically accurate."
    )
    resolved = _resolve_system_prompt(
        PROMPT_GROUNDED_SYNTHESIS,
        variables={
            "language_instruction": _language_instruction(language),
            "style_hint": style_hint,
        },
        prompt_manager=prompt_manager,
    )

    history_block = (
        f"\nRecent Conversation Context:\n{format_chat_history(chat_history, max_turns=4)}\n"
        if chat_history
        else ""
    )
    user_prompt = (
        f"User Question: {query.strip()}\n"
        f"Resolved Standalone Question: {standalone_query.strip()}"
        f"{history_block}\n"
        f"Evidence Passages:\n{format_evidence_context(chunks)}"
    )
    return [SystemMessage(content=resolved.text), HumanMessage(content=user_prompt)], resolved


def build_generation_messages(
    *,
    query: str,
    standalone_query: str,
    language: str,
    chunks: Sequence[EvidenceChunk],
    chat_history: Sequence[ChatMessage],
    query_type: QueryType,
    prompt_manager: PromptManager | None = None,
) -> list[BaseMessage]:
    """Build strictly grounded synthesis messages requiring inline [C#]/[W#] citations."""
    messages, _ = resolve_generation_prompt(
        query=query,
        standalone_query=standalone_query,
        language=language,
        chunks=chunks,
        chat_history=chat_history,
        query_type=query_type,
        prompt_manager=prompt_manager,
    )
    return messages


def resolve_strict_regeneration_prompt(
    *,
    query: str,
    standalone_query: str,
    language: str,
    chunks: Sequence[EvidenceChunk],
    previous_draft: str,
    report: CitationReport,
    prompt_manager: PromptManager | None = None,
) -> tuple[list[BaseMessage], ResolvedPrompt]:
    """Resolve strict regeneration prompt and build messages with citation violation feedback."""
    issues: list[str] = []
    if report.invalid_citation_ids:
        issues.append(
            f"- Hallucinated/invalid citation IDs removed: {', '.join(report.invalid_citation_ids)}"
        )
    if report.uncited_factual_sentences:
        sample = "; ".join(s[:120] for s in report.uncited_factual_sentences[:4])
        issues.append(f"- Uncited factual statements needing citations or removal: {sample}")
    for cv in report.claim_verifications:
        if cv.verdict in {
            ClaimVerdict.UNSUPPORTED,
            ClaimVerdict.CONTRADICTED,
            ClaimVerdict.PARTIALLY_SUPPORTED,
        }:
            issues.append(
                f"- [{cv.verdict.value.upper()}] {cv.claim} "
                f"(cited={cv.cited_ids}): {cv.explanation}"
            )

    feedback_block = "\n".join(issues) if issues else "- Improve inline citation coverage."
    valid_labels = (
        ", ".join(
            f"[{c.citation_id.upper()}]" for c in chunks if c.citation_id and c.citation_id.strip()
        )
        or "(none)"
    )

    resolved = _resolve_system_prompt(
        PROMPT_STRICT_REGENERATION,
        variables={
            "valid_labels": valid_labels,
            "language_instruction": _language_instruction(language),
        },
        prompt_manager=prompt_manager,
    )

    user_prompt = (
        f"User Question: {query.strip()}\n"
        f"Resolved Question: {standalone_query.strip()}\n\n"
        f"Previous Draft (FAILED VERIFICATION):\n{previous_draft.strip()}\n\n"
        f"Verification Violations to Fix:\n{feedback_block}\n\n"
        f"Evidence Passages:\n{format_evidence_context(chunks)}"
    )
    return [SystemMessage(content=resolved.text), HumanMessage(content=user_prompt)], resolved


def build_strict_regeneration_messages(
    *,
    query: str,
    standalone_query: str,
    language: str,
    chunks: Sequence[EvidenceChunk],
    previous_draft: str,
    report: CitationReport,
    prompt_manager: PromptManager | None = None,
) -> list[BaseMessage]:
    """Build strict corrective regeneration messages with exact citation violation feedback."""
    messages, _ = resolve_strict_regeneration_prompt(
        query=query,
        standalone_query=standalone_query,
        language=language,
        chunks=chunks,
        previous_draft=previous_draft,
        report=report,
        prompt_manager=prompt_manager,
    )
    return messages


def resolve_conversational_prompt(
    *,
    query: str,
    language: str,
    chat_history: Sequence[ChatMessage],
    prompt_manager: PromptManager | None = None,
) -> tuple[list[BaseMessage], ResolvedPrompt]:
    """Resolve conversational prompt and build messages for zero-retrieval turns."""
    resolved = _resolve_system_prompt(
        PROMPT_CONVERSATIONAL,
        variables={"language_instruction": _language_instruction(language)},
        prompt_manager=prompt_manager,
    )
    user_prompt = (
        f"Conversation History:\n{format_chat_history(chat_history)}\n\n"
        f"User Message:\n{query.strip()}"
    )
    return [SystemMessage(content=resolved.text), HumanMessage(content=user_prompt)], resolved


def build_conversational_messages(
    *,
    query: str,
    language: str,
    chat_history: Sequence[ChatMessage],
    prompt_manager: PromptManager | None = None,
) -> list[BaseMessage]:
    """Build messages for zero-retrieval conversational turns."""
    messages, _ = resolve_conversational_prompt(
        query=query,
        language=language,
        chat_history=chat_history,
        prompt_manager=prompt_manager,
    )
    return messages


def build_abstention_message(
    *,
    query: str,
    language: str,
    confidence: ConfidenceAssessment | None,
    fallback_used: bool,
) -> str:
    """Return a deterministic, honest TR/EN abstention when evidence is insufficient."""
    score_str = f"{confidence.final_score:.2f}" if confidence is not None else "0.00"
    if language == "tr":
        scope = (
            "hem yerel bilgi tabanında hem de web arama yedeğinde"
            if fallback_used
            else "yerel bilgi tabanında"
        )
        return (
            f'"{query.strip()}" sorusunu güvenilir biçimde yanıtlamak için {scope} '
            f"yeterli doğrulanmış kanıt bulunamadı (güven skoru: {score_str}). "
            "Yanıltıcı bilgi üretmemek adına tahminde bulunmuyorum; soruyu daha spesifik teknik "
            "terimlerle daraltabilir veya ilgili dokümanları indeksleyebilirsiniz."
        )

    scope_en = (
        "both the local knowledge base and web search fallback"
        if fallback_used
        else "the local knowledge base"
    )
    return (
        f'I cannot provide a verified answer to "{query.strip()}" because {scope_en} '
        f"did not return sufficient supporting evidence (confidence score: {score_str}). "
        "Rather than speculating, I am abstaining; please try narrowing the technical terms "
        "or ingesting the relevant documentation."
    )
