"""Multilingual (TR + EN) prompt builders for routing, transformation, synthesis & recovery."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

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


def build_router_messages(
    query: str,
    chat_history: Sequence[ChatMessage],
) -> list[BaseMessage]:
    """Build messages for query classification, coreference resolution & freshness detection."""
    system_prompt = """You are the Query Router for AdaptiveRAG, a multilingual (Turkish & English) technical knowledge assistant.

Classify the user query into exactly one `query_type`:
- "conversational": Greetings, thanks, small talk, or pure meta-requests about prior assistant turns (set `needs_retrieval=false`).
- "factual": Direct definition, specific API/parameter/default lookup, exact formula, or single-concept question (set `needs_retrieval=true`).
- "analytical": Comparisons, trade-offs, architectural design choices, multi-step troubleshooting, or questions combining multiple systems/concepts (set `needs_retrieval=true`).

Also produce:
- `language`: "tr" for Turkish, "en" for English, or "other".
- `standalone_query`: Rewrite the user query so it is completely self-contained by resolving any pronouns ("it", "this", "onun", "bunlar") using chat history. Preserve the query's language and technical terms.
- `requires_fresh_web_info`: Set `true` ONLY when the query explicitly asks for current/latest releases, live benchmarks, recent news, or real-time prices unlikely to exist in static documentation. Otherwise `false`.
- `reasoning`: One concise sentence justifying your routing choice."""

    user_prompt = (
        f"Conversation History:\n{format_chat_history(chat_history)}\n\n"
        f"Current User Query:\n{query.strip()}"
    )
    return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]


def build_decompose_messages(
    standalone_query: str,
    *,
    language: str,
    max_sub_queries: int,
) -> list[BaseMessage]:
    """Build messages for analytical multi-query decomposition."""
    target_max = max(2, min(5, max_sub_queries))
    system_prompt = f"""You are a Query Decomposition Specialist for a hybrid BM25 + Vector search engine.

Decompose the complex analytical question into 2 to {target_max} distinct, focused sub-queries that cover orthogonal aspects (e.g., each system/concept being compared, specific mechanisms, latency/memory/operational trade-offs).

Rules:
- Keep every sub-query concise and keyword-rich so both BM25 lexical search and dense embeddings perform well.
- Avoid redundant duplicates.
- Preserve exact technical terms, library names, and acronyms.
- Write sub-queries primarily in the language of the input question (language={language}), keeping standard English technical identifiers intact."""

    user_prompt = f"Analytical Question:\n{standalone_query.strip()}"
    return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]


def build_corrective_rewrite_messages(
    *,
    original_query: str,
    standalone_query: str,
    prior_queries: Sequence[str],
    confidence: ConfidenceAssessment | None,
    language: str,
) -> list[BaseMessage]:
    """Build messages for corrective query reformulation after low/medium confidence."""
    cross_lingual_rule = (
        f"- Because technical knowledge bases often mix Turkish and English documentation, "
        f"write AT LEAST ONE query in canonical English technical terminology and keep the remaining "
        f"query(s) in {language} using standard English acronyms/identifiers."
        if language != "en"
        else "- Use canonical English technical terminology, exact parameter/class names, and synonyms."
    )
    system_prompt = f"""You are a Corrective Retrieval Rewriter (CRAG query optimizer) for a multilingual (TR/EN) hybrid BM25 + dense vector search engine.

The initial retrieval produced low or partial confidence. Reformulate 2 to 3 improved search queries that use better technical synonyms, canonical terminology, or clearer aspect separation.

Rules:
- Do NOT simply repeat the prior queries verbatim.
- Target any missing aspects identified in the confidence diagnosis.
{cross_lingual_rule}
- Keep every query concise and retrieval-oriented."""

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
    return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]


def build_generation_messages(
    *,
    query: str,
    standalone_query: str,
    language: str,
    chunks: Sequence[EvidenceChunk],
    chat_history: Sequence[ChatMessage],
    query_type: QueryType,
) -> list[BaseMessage]:
    """Build strictly grounded synthesis messages requiring inline [C#]/[W#] citations."""
    style_hint = (
        "Structure your response clearly to cover comparisons, trade-offs, and key dimensions."
        if query_type == QueryType.ANALYTICAL
        else "Keep the response direct, precise, and technically accurate."
    )

    system_prompt = f"""You are AdaptiveRAG, a strictly grounded technical AI assistant.

Grounding & Citation Rules (MANDATORY):
1. Answer ONLY using facts explicitly stated in the provided Evidence Passages below. Do NOT rely on unverified parametric memory.
2. Attach exact inline citations like [C1], [W1], or [C1, C2] to EVERY factual sentence or bullet point.
3. NEVER invent, guess, or alter citation IDs. Only use IDs that appear in the Evidence Passages.
4. If the evidence only partially covers the question, answer the supported part with citations and explicitly note what is not covered in the provided sources.
5. {_language_instruction(language)}
6. {style_hint}"""

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
    return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]


def build_strict_regeneration_messages(
    *,
    query: str,
    standalone_query: str,
    language: str,
    chunks: Sequence[EvidenceChunk],
    previous_draft: str,
    report: CitationReport,
) -> list[BaseMessage]:
    """Build strict corrective regeneration messages with exact citation violation feedback."""
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
    valid_labels = ", ".join(
        f"[{c.citation_id.upper()}]" for c in chunks if c.citation_id and c.citation_id.strip()
    )

    system_prompt = f"""You are revising a draft answer that FAILED strict citation verification.

Mandatory Revision Instructions:
1. Rewrite the answer so EVERY factual statement is 100% entailed by the Evidence Passages and cites valid passage IDs inline.
2. Allowed citation IDs ONLY: {valid_labels or "(none)"}. Never emit any other ID.
3. REMOVE or explicitly qualify any claim that was flagged as unsupported or contradicted.
4. Do NOT add new unsupported facts. Prefer a shorter, 100% verified answer over a longer speculative one.
5. {_language_instruction(language)}"""

    user_prompt = (
        f"User Question: {query.strip()}\n"
        f"Resolved Question: {standalone_query.strip()}\n\n"
        f"Previous Draft (FAILED VERIFICATION):\n{previous_draft.strip()}\n\n"
        f"Verification Violations to Fix:\n{feedback_block}\n\n"
        f"Evidence Passages:\n{format_evidence_context(chunks)}"
    )
    return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]


def build_conversational_messages(
    *,
    query: str,
    language: str,
    chat_history: Sequence[ChatMessage],
) -> list[BaseMessage]:
    """Build messages for zero-retrieval conversational turns."""
    system_prompt = (
        "You are AdaptiveRAG, a helpful and concise AI engineering assistant. "
        "Handle conversational messages, greetings, or meta-questions about prior turns naturally. "
        "Do NOT fabricate document citations like [C1] or [W1] because no retrieval was performed. "
        f"{_language_instruction(language)}"
    )
    user_prompt = (
        f"Conversation History:\n{format_chat_history(chat_history)}\n\n"
        f"User Message:\n{query.strip()}"
    )
    return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]


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
