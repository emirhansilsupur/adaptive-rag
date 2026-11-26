"""Streamlit Glass-Box RAG Inspector UI for AdaptiveRAG.

Connects to the FastAPI backend (/v1/health, /v1/query/stream, /v1/query, /v1/ingest)
and visualizes every agentic decision in real time: routing, multi-query decomposition,
hybrid search, cross-encoder reranking & dynamic top-k, confidence + CRAG gating,
web fallback, and two-tier citation auditing.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from typing import Any

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DEFAULT_API_URL = os.environ.get("ADAPTIVERAG_API_URL", "http://localhost:8000").rstrip("/")
DEFAULT_API_KEY = os.environ.get("ADAPTIVERAG_API_KEY", "").strip()

NODE_METADATA: dict[str, tuple[str, str, str]] = {
    "classify_and_plan": ("🧭", "Classify & Plan", "#2563EB"),
    "conversational_respond": ("💬", "Conversational Respond", "#0EA5E9"),
    "decompose_queries": ("🧩", "Decompose Sub-Queries", "#6366F1"),
    "hybrid_retrieve": ("🔎", "Hybrid Retrieve (Dense + BM25)", "#0D9488"),
    "rerank_and_select_k": ("📊", "Rerank & Dynamic Top-K", "#14B8A6"),
    "score_confidence": ("🎯", "Confidence & CRAG Gate", "#D97706"),
    "corrective_rewrite": ("♻️", "Corrective Query Rewrite", "#EA580C"),
    "web_fallback_search": ("🌐", "Web Fallback Search", "#9333EA"),
    "generate_answer": ("✍️", "Grounded Generation", "#16A34A"),
    "validate_citations": ("✅", "Two-Tier Citation Audit", "#15803D"),
    "regenerate_strict": ("🔒", "Strict Grounded Regeneration", "#DC2626"),
    "abstain_response": ("🛑", "Honest Abstention", "#64748B"),
}

PRESET_LABELS: dict[str, str] = {
    "full": "Full AdaptiveRAG (All Agentic Loops)",
    "adaptive_routing": "Adaptive Routing + Dynamic Top-K (No Self-Correction)",
    "hybrid_rerank": "Hybrid BM25+Dense + Rerank (Static Top-K)",
    "naive_dense": "Naive Dense Baseline (Static Top-K)",
    "custom": "Custom Feature Switches",
}

DEMO_QUERIES: list[tuple[str, str]] = [
    (
        "🇬🇧 Factual · RRF",
        "What is the Reciprocal Rank Fusion (RRF) formula and how does Qdrant combine dense and sparse prefetch results?",
    ),
    (
        "🇹🇷 Analytical · Hibrit & Rerank",
        "RAG sistemlerinde hibrit arama ve cross-encoder reranking neden birlikte kullanılır; gecikme ve doğruluk dengesini karşılaştır?",
    ),
    (
        "🇬🇧 Analytical · LangGraph",
        "Compare cyclic LangGraph state graphs against linear RAG chains for corrective retrieval and citation verification.",
    ),
    (
        "💬 Conversational · TR",
        "Merhaba! Bugün nasılsın, hangi konularda yardımcı olabilirsin?",
    ),
    (
        "🌐 Web Fallback · Fresh Info",
        "What is the latest stable release version of Qdrant available today?",
    ),
]


# ---------------------------------------------------------------------------
# HTTP & SSE Client Helpers
# ---------------------------------------------------------------------------


def iter_sse_events(lines: Iterable[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Parse standard Server-Sent Events lines into ``(event_name, json_payload)`` pairs."""
    event_name = "message"
    data_lines: list[str] = []

    for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        if not line:
            if data_lines:
                raw_data = "\n".join(data_lines)
                try:
                    parsed: dict[str, Any] = json.loads(raw_data)
                except json.JSONDecodeError:
                    parsed = {"raw": raw_data}
                yield event_name, parsed
                event_name = "message"
                data_lines = []
            continue

        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip() or "message"
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())

    if data_lines:
        raw_data = "\n".join(data_lines)
        try:
            parsed = json.loads(raw_data)
        except json.JSONDecodeError:
            parsed = {"raw": raw_data}
        yield event_name, parsed


def build_auth_headers(api_key: str = "") -> dict[str, str]:
    """Build optional X-API-Key header dictionary when API_AUTH_KEY is configured."""
    cleaned = api_key.strip()
    return {"X-API-Key": cleaned} if cleaned else {}


def fetch_health(
    api_base_url: str,
    *,
    api_key: str = "",
) -> tuple[dict[str, Any] | None, str | None]:
    """Fetch GET /v1/health from the FastAPI backend."""
    url = f"{api_base_url.rstrip('/')}/v1/health"
    try:
        with httpx.Client(timeout=8.0, headers=build_auth_headers(api_key)) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return dict(resp.json()), None
    except Exception as exc:
        return None, str(exc)


def parse_url_lines(raw_text: str) -> list[str]:
    """Split a newline/comma-separated URL input into deduplicated non-empty strings."""
    urls: list[str] = []
    seen: set[str] = set()
    for line in raw_text.splitlines():
        cleaned = line.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            urls.append(cleaned)
    return urls


def build_web_ingest_payload(
    *,
    mode: str,
    urls_text: str = "",
    crawl_url: str = "",
    crawl_limit: int = 10,
    crawl_max_depth: int | None = None,
    only_main_content: bool = True,
    recreate_collection: bool = False,
) -> dict[str, Any]:
    """Build a validated POST /v1/ingest JSON payload for Firecrawl URL scrape or site crawl."""
    payload: dict[str, Any] = {
        "only_main_content": only_main_content,
        "recreate_collection": recreate_collection,
    }
    if mode == "crawl":
        cleaned = crawl_url.strip()
        if not cleaned:
            raise ValueError("Enter a root documentation URL to crawl.")
        payload["crawl_url"] = cleaned
        payload["crawl_limit"] = int(crawl_limit)
        if crawl_max_depth is not None:
            payload["crawl_max_depth"] = int(crawl_max_depth)
    else:
        parsed_urls = parse_url_lines(urls_text)
        if not parsed_urls:
            raise ValueError("Enter at least one public http(s) URL to scrape.")
        payload["urls"] = parsed_urls
    return payload


def trigger_ingest(
    api_base_url: str,
    *,
    recreate_collection: bool = False,
    api_key: str = "",
) -> tuple[dict[str, Any] | None, str | None]:
    """Trigger POST /v1/ingest for the default local corpus (data/raw)."""
    return trigger_ingest_payload(
        api_base_url,
        {"recreate_collection": recreate_collection},
        api_key=api_key,
    )


def trigger_ingest_payload(
    api_base_url: str,
    payload: dict[str, Any],
    *,
    api_key: str = "",
) -> tuple[dict[str, Any] | None, str | None]:
    """Send an arbitrary POST /v1/ingest request (local corpus, inline docs, or Firecrawl URLs)."""
    url = f"{api_base_url.rstrip('/')}/v1/ingest"
    try:
        with httpx.Client(timeout=240.0, headers=build_auth_headers(api_key)) as client:
            resp = client.post(url, json=payload)
            if resp.status_code >= 400:
                detail = resp.text
                try:
                    parsed = resp.json()
                    if isinstance(parsed, dict) and parsed.get("detail"):
                        detail = str(parsed["detail"])
                except ValueError:
                    pass
                return None, f"HTTP {resp.status_code}: {detail}"
            return dict(resp.json()), None
    except Exception as exc:
        return None, str(exc)


def stream_query_events(
    api_base_url: str,
    payload: dict[str, Any],
    *,
    api_key: str = "",
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Stream SSE events from POST /v1/query/stream."""
    url = f"{api_base_url.rstrip('/')}/v1/query/stream"
    timeout = httpx.Timeout(180.0, connect=10.0)
    with (
        httpx.Client(timeout=timeout, headers=build_auth_headers(api_key)) as client,
        client.stream("POST", url, json=payload) as response,
    ):
        if response.status_code >= 400:
            body = response.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API returned HTTP {response.status_code}: {body}")
        yield from iter_sse_events(response.iter_lines())


def run_query_batch(
    api_base_url: str,
    payload: dict[str, Any],
    *,
    api_key: str = "",
) -> dict[str, Any]:
    """Execute non-streaming POST /v1/query and return FinalAnswer dict."""
    url = f"{api_base_url.rstrip('/')}/v1/query"
    with httpx.Client(
        timeout=httpx.Timeout(180.0, connect=10.0),
        headers=build_auth_headers(api_key),
    ) as client:
        resp = client.post(url, json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(f"API returned HTTP {resp.status_code}: {resp.text}")
        return dict(resp.json())


# ---------------------------------------------------------------------------
# Styling & Badge Rendering
# ---------------------------------------------------------------------------


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .ar-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin: 0.35rem 0 0.75rem 0;
        }
        .ar-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            border: 1px solid rgba(148, 163, 184, 0.35);
            background: rgba(148, 163, 184, 0.12);
        }
        .ar-badge-blue { background: rgba(37, 99, 235, 0.14); border-color: rgba(37, 99, 235, 0.4); }
        .ar-badge-green { background: rgba(22, 163, 74, 0.14); border-color: rgba(22, 163, 74, 0.4); }
        .ar-badge-amber { background: rgba(217, 119, 6, 0.15); border-color: rgba(217, 119, 6, 0.45); }
        .ar-badge-red { background: rgba(220, 38, 38, 0.15); border-color: rgba(220, 38, 38, 0.45); }
        .ar-badge-purple { background: rgba(147, 51, 234, 0.14); border-color: rgba(147, 51, 234, 0.4); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_answer_badges(result: dict[str, Any]) -> None:
    """Render compact telemetry badges summarizing a FinalAnswer."""
    badges: list[tuple[str, str]] = []

    q_type = str(result.get("query_type", "factual")).upper()
    lang = str(result.get("language", "en")).upper()
    badges.append((f"🧭 {q_type}", "ar-badge-blue"))
    badges.append((f"🌍 {lang}", "ar-badge"))

    confidence = result.get("confidence")
    if isinstance(confidence, dict):
        level = str(confidence.get("level", "")).upper()
        score = float(confidence.get("final_score", 0.0))
        css = (
            "ar-badge-green"
            if level == "HIGH"
            else ("ar-badge-amber" if level == "MEDIUM" else "ar-badge-red")
        )
        badges.append((f"🎯 Conf: {score:.2f} ({level})", css))

    citation = result.get("citation_report")
    if isinstance(citation, dict):
        passed = bool(citation.get("passed", False))
        support = float(citation.get("claim_support_rate", 0.0)) * 100.0
        if _citation_unverified(citation):
            badges.append(("⚠️ Citations: unverified", "ar-badge-amber"))
        else:
            css = "ar-badge-green" if passed else "ar-badge-red"
            icon = "✅" if passed else "⚠️"
            badges.append((f"{icon} Citations: {support:.0f}%", css))

    if result.get("fallback_used"):
        badges.append(("🌐 Web Fallback", "ar-badge-purple"))
    if result.get("abstained"):
        badges.append(("🛑 Abstained", "ar-badge-red"))
    if int(result.get("rewrite_attempts", 0)) > 0:
        badges.append((f"♻️ Rewrites: {result['rewrite_attempts']}", "ar-badge-amber"))
    if int(result.get("regeneration_attempts", 0)) > 0:
        badges.append((f"🔒 Regens: {result['regeneration_attempts']}", "ar-badge-amber"))

    latency_ms = float(result.get("total_latency_ms", 0.0))
    usage = result.get("usage") or {}
    total_tokens = int(usage.get("total_tokens", 0))
    cost_usd = float(usage.get("estimated_cost_usd", 0.0))
    cache_hits = int(usage.get("cache_hits", 0))

    badges.append((f"⏱️ {latency_ms:.0f} ms", "ar-badge"))
    badges.append((f"🪙 {total_tokens:,} tok · ${cost_usd:.5f}", "ar-badge"))
    if cache_hits > 0:
        badges.append((f"⚡ Cache Hits: {cache_hits}", "ar-badge-green"))

    html = "".join(f'<span class="ar-badge {css}">{label}</span>' for label, css in badges)
    st.markdown(f'<div class="ar-badge-row">{html}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Glass-Box Inspector Tabs
# ---------------------------------------------------------------------------


def _render_trace_and_plan_tab(result: dict[str, Any]) -> None:
    plan = result.get("query_plan")
    if isinstance(plan, dict):
        st.markdown("#### Query Plan & Active Strategy")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Query Type", str(plan.get("query_type", "-")).title())
        col2.metric("Language", str(plan.get("language", "-")).upper())
        col3.metric("Needs Retrieval", "Yes" if plan.get("needs_retrieval") else "No")
        col4.metric("Fresh Web Flag", "Yes" if plan.get("requires_fresh_web_info") else "No")

        st.markdown(
            f"**Standalone Query:** `{plan.get('standalone_query', result.get('query', ''))}`"
        )
        if plan.get("reasoning"):
            st.caption(f"Router reasoning: {plan['reasoning']}")

        sub_queries = plan.get("sub_queries") or []
        if sub_queries:
            st.markdown("**Sub-Queries Executed:**")
            for idx, sq in enumerate(sub_queries, start=1):
                st.markdown(f"{idx}. `{sq}`")

        strategy = plan.get("strategy")
        if isinstance(strategy, dict):
            with st.expander(
                "Active Strategy Hyperparameters (configs/strategies.yaml)", expanded=False
            ):
                st.json(strategy)

    trace = result.get("trace") or []
    st.markdown("#### Execution Latency Breakdown")
    if not trace:
        st.info("No node trace events recorded for this turn.")
        return

    labels: list[str] = []
    latencies: list[float] = []
    colors: list[str] = []
    summaries: list[str] = []

    for step_idx, evt in enumerate(trace, start=1):
        node_key = str(evt.get("node", "unknown"))
        icon, pretty, color = NODE_METADATA.get(node_key, ("⚙️", node_key, "#64748B"))
        labels.append(f"{step_idx}. {icon} {pretty}")
        latencies.append(float(evt.get("latency_ms", 0.0)))
        colors.append(color)
        summaries.append(str(evt.get("summary", "")))

    fig = go.Figure(
        go.Bar(
            x=latencies,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{ms:.1f} ms" for ms in latencies],
            textposition="outside",
            cliponaxis=False,
            hovertext=summaries,
            hovertemplate="<b>%{y}</b><br>Latency: %{x:.1f} ms<br>%{hovertext}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(240, 40 * len(labels) + 60),
        margin={"l": 10, "r": 85, "t": 15, "b": 30},
        xaxis_title="Latency (ms)",
        yaxis={"autorange": "reversed"},
    )
    st.plotly_chart(fig, width="stretch")

    trace_df = pd.DataFrame(
        [
            {
                "Step": i,
                "Node": evt.get("node", ""),
                "Latency (ms)": round(float(evt.get("latency_ms", 0.0)), 2),
                "Summary": evt.get("summary", ""),
                "Metadata": json.dumps(evt.get("metadata") or {}, ensure_ascii=False),
            }
            for i, evt in enumerate(trace, start=1)
        ]
    )
    st.dataframe(trace_df, width="stretch", hide_index=True)


def _render_dynamic_k_tab(result: dict[str, Any]) -> None:
    dk = result.get("dynamic_k_decision")
    chunks = result.get("evidence_chunks") or []
    plan = result.get("query_plan") or {}
    strategy = plan.get("strategy") if isinstance(plan, dict) else {}

    if not isinstance(dk, dict) and not chunks:
        st.info("Retrieval was skipped for this turn (conversational route).")
        return

    if isinstance(dk, dict):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Selected / Candidates",
            f"{dk.get('selected_count', len(chunks))} / {dk.get('candidate_count', len(chunks))}",
        )
        c2.metric("Top Score", f"{float(dk.get('top_score', 0.0)):.3f}")
        c3.metric("Cutoff Score", f"{float(dk.get('cutoff_score', 0.0)):.3f}")
        c4.metric("Context Tokens", f"{int(dk.get('total_context_tokens', 0)):,}")
        st.caption(f"Cutoff reason: `{dk.get('cutoff_reason', 'n/a')}`")

    if not chunks:
        st.warning("No evidence chunks survived the dynamic top-k filter.")
        return

    use_rerank = any(ch.get("rerank_score") is not None for ch in chunks)
    metric_label = "Calibrated Rerank Score (Sigmoid)" if use_rerank else "RRF / Retrieval Score"

    x_labels: list[str] = []
    y_scores: list[float] = []
    colors: list[str] = []
    hovers: list[str] = []

    for idx, ch in enumerate(chunks, start=1):
        cid = ch.get("citation_id") or f"#{idx}"
        src = str(ch.get("source_type", "local")).lower()
        score = (
            float(ch["rerank_score"])
            if use_rerank and ch.get("rerank_score") is not None
            else float(ch.get("fusion_score") or ch.get("dense_score") or 0.0)
        )
        x_labels.append(f"[{cid}]")
        y_scores.append(score)
        colors.append("#9333EA" if src == "web" else "#0D9488")
        title = ch.get("title") or ch.get("doc_id") or "Untitled"
        section = ch.get("section_path") or "-"
        hovers.append(f"{title}<br>Section: {section}<br>Source: {src.upper()}")

    fig = go.Figure(
        go.Bar(
            x=x_labels,
            y=y_scores,
            marker_color=colors,
            text=[f"{s:.3f}" for s in y_scores],
            textposition="outside",
            hovertext=hovers,
            hovertemplate="<b>%{x}</b><br>Score: %{y:.4f}<br>%{hovertext}<extra></extra>",
        )
    )
    if use_rerank and isinstance(strategy, dict) and "tau_min" in strategy:
        tau_min = float(strategy["tau_min"])
        fig.add_hline(
            y=tau_min,
            line_dash="dash",
            line_color="#DC2626",
            annotation_text=f"tau_min = {tau_min:.2f}",
            annotation_position="top right",
        )

    fig.update_layout(
        height=320,
        margin={"l": 10, "r": 20, "t": 30, "b": 35},
        yaxis_title=metric_label,
        yaxis={"range": [0.0, 1.12]} if use_rerank else None,
        xaxis_title="Selected Evidence Passage",
    )
    st.plotly_chart(fig, width="stretch")


def _render_confidence_tab(result: dict[str, Any]) -> None:
    conf = result.get("confidence")
    if not isinstance(conf, dict):
        st.info("Confidence assessment was not run for this turn.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Score", f"{float(conf.get('final_score', 0.0)):.3f}")
    c2.metric("Confidence Level", str(conf.get("level", "-")).upper())
    c3.metric(
        "CRAG Grade",
        str(conf.get("crag_grade") or "Not Needed").upper(),
    )
    c4.metric(
        "Recommended Action",
        str(conf.get("recommended_action", "-")).replace("_", " ").upper(),
    )

    if conf.get("reasoning"):
        st.info(f"**Confidence Gate Reasoning:** {conf['reasoning']}")

    signals = [
        ("Top-1 Rerank (w=0.50)", float(conf.get("top1_score", 0.0)), "#2563EB"),
        ("Mean Top-K (w=0.30)", float(conf.get("mean_topk_score", 0.0)), "#0D9488"),
        ("Sub-Query Coverage (w=0.20)", float(conf.get("subquery_coverage", 0.0)), "#6366F1"),
        ("Deterministic Composite", float(conf.get("deterministic_score", 0.0)), "#D97706"),
        ("Final Calibrated Score", float(conf.get("final_score", 0.0)), "#16A34A"),
    ]

    fig = go.Figure(
        go.Bar(
            x=[val for _, val, _ in signals],
            y=[name for name, _, _ in signals],
            orientation="h",
            marker_color=[col for _, _, col in signals],
            text=[f"{val:.3f}" for _, val, _ in signals],
            textposition="outside",
        )
    )
    fig.add_vline(
        x=0.38,
        line_dash="dot",
        line_color="#DC2626",
        annotation_text="low (0.38)",
        annotation_position="top",
    )
    fig.add_vline(
        x=0.72,
        line_dash="dot",
        line_color="#16A34A",
        annotation_text="high (0.72)",
        annotation_position="top",
    )
    fig.update_layout(
        height=290,
        margin={"l": 10, "r": 50, "t": 35, "b": 30},
        xaxis={"range": [0.0, 1.12], "title": "Confidence Signal (0.0 – 1.0)"},
        yaxis={"autorange": "reversed"},
    )
    st.plotly_chart(fig, width="stretch")


def _render_evidence_tab(result: dict[str, Any]) -> None:
    chunks = result.get("evidence_chunks") or []
    if not chunks:
        st.info("No evidence passages attached to this response.")
        return

    filter_choice = st.radio(
        "Filter evidence source",
        options=["All", "Local", "Web"],
        horizontal=True,
        key=f"evidence_filter_{id(result)}",
    )
    filtered = [
        ch
        for ch in chunks
        if filter_choice == "All"
        or str(ch.get("source_type", "local")).lower() == filter_choice.lower()
    ]

    if not filtered:
        st.info(f"No {filter_choice.lower()} passages in this turn.")
        return

    for idx, ch in enumerate(filtered, start=1):
        cid = ch.get("citation_id") or f"#{idx}"
        src = str(ch.get("source_type", "local")).lower()
        src_icon = "🌐 WEB" if src == "web" else "📄 LOCAL"
        title = ch.get("title") or ch.get("doc_id") or "Untitled"
        rerank = ch.get("rerank_score")
        fusion = ch.get("fusion_score")
        score_str = (
            f"rerank={float(rerank):.3f}"
            if rerank is not None
            else (f"fusion={float(fusion):.4f}" if fusion is not None else "n/a")
        )
        header = f"[{cid}] {src_icon} · {title} ({score_str})"
        with st.expander(header, expanded=(idx <= 2)):
            meta_parts: list[str] = []
            if ch.get("section_path"):
                meta_parts.append(f"**Section:** {ch['section_path']}")
            if ch.get("doc_id"):
                meta_parts.append(f"**Doc ID:** `{ch['doc_id']}`")
            if ch.get("token_count"):
                meta_parts.append(f"**Tokens:** {ch['token_count']}")
            if meta_parts:
                st.markdown(" · ".join(meta_parts))

            if ch.get("url"):
                st.markdown(f"**URL:** [{ch['url']}]({ch['url']})")

            matched = ch.get("matched_sub_queries") or []
            if matched:
                st.caption("Matched queries: " + " | ".join(f"`{q}`" for q in matched))

            st.markdown("---")
            st.markdown(str(ch.get("text", "")))


def _citation_unverified(report: dict[str, Any]) -> bool:
    """True when the claim judge failed but the citation structure itself was fine."""
    return report.get("semantic_audit") == "failed" and bool(report.get("structural_passed", True))


def _render_citation_tab(result: dict[str, Any]) -> None:
    report = result.get("citation_report")
    if not isinstance(report, dict):
        st.info("Citation validation was skipped for this turn.")
        return

    passed = bool(report.get("passed", False))
    unverified = _citation_unverified(report)
    if report.get("semantic_audit") == "failed":
        st.warning(
            "The claim check didn't run: "
            f"{report.get('semantic_audit_error') or 'the judge model failed.'} "
            "Claims were not checked against their passages."
        )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Audit Status", "PASSED" if passed else ("UNVERIFIED" if unverified else "FAILED"))
    c2.metric("Citation Precision", f"{float(report.get('citation_precision', 0.0)) * 100:.1f}%")
    c3.metric("Claim Support Rate", f"{float(report.get('claim_support_rate', 0.0)) * 100:.1f}%")
    c4.metric("Citations Found", int(report.get("total_citations_found", 0)))

    valid_ids = report.get("valid_citation_ids") or []
    invalid_ids = report.get("invalid_citation_ids") or []
    uncited = report.get("uncited_factual_sentences") or []

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            "**Valid Citations:** "
            + (", ".join(f"`[{cid}]`" for cid in valid_ids) if valid_ids else "_None_")
        )
    with col_b:
        if invalid_ids:
            st.error("Invalid / Fabricated IDs: " + ", ".join(f"`[{cid}]`" for cid in invalid_ids))
        else:
            st.success("Zero fabricated citation IDs detected (Tier-1).")

    if uncited:
        with st.expander(f"⚠️ Uncited Factual Sentences ({len(uncited)})", expanded=True):
            for sent in uncited:
                st.markdown(f"- {sent}")

    verifications = report.get("claim_verifications") or []
    if verifications:
        st.markdown("#### Tier-2 Claim-Level Semantic Entailment")
        verdict_icons = {
            "supported": "✅ Supported",
            "partially_supported": "🟡 Partial",
            "unsupported": "⚠️ Unsupported",
            "contradicted": "❌ Contradicted",
        }
        rows = [
            {
                "Verdict": verdict_icons.get(
                    str(v.get("verdict", "")).lower(), str(v.get("verdict", ""))
                ),
                "Cited IDs": ", ".join(f"[{cid}]" for cid in (v.get("cited_ids") or [])),
                "Claim": v.get("claim", ""),
                "Explanation": v.get("explanation", ""),
            }
            for v in verifications
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.caption("No Tier-2 claim-level entailment rows returned (skipped or zero citations).")


def render_glass_box_inspector(result: dict[str, Any]) -> None:
    """Render the 5-tab Glass-Box Deep-Dive Inspector for a FinalAnswer payload."""
    st.markdown("### 🔬 Glass-Box Pipeline Inspector")
    tabs = st.tabs(
        [
            "🧭 Trace & Query Plan",
            "📊 Dynamic Top-K & Rerank",
            "🎯 Confidence & CRAG",
            "📚 Evidence Passages",
            "✅ Citation Audit",
        ]
    )
    with tabs[0]:
        _render_trace_and_plan_tab(result)
    with tabs[1]:
        _render_dynamic_k_tab(result)
    with tabs[2]:
        _render_confidence_tab(result)
    with tabs[3]:
        _render_evidence_tab(result)
    with tabs[4]:
        _render_citation_tab(result)


# ---------------------------------------------------------------------------
# Sidebar Controls & Main Application
# ---------------------------------------------------------------------------


def build_query_payload(
    query: str,
    chat_history: list[dict[str, str]],
    *,
    preset: str,
    custom_ablation: dict[str, Any] | None,
    max_cost_usd: float | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": query,
        "chat_history": chat_history,
    }
    if preset == "custom" and custom_ablation is not None:
        payload["ablation"] = custom_ablation
    elif preset != "custom":
        payload["preset"] = preset
    if max_cost_usd is not None:
        payload["max_cost_usd"] = max_cost_usd
    return payload


def _render_sidebar_health(
    container: Any,
    health_data: dict[str, Any] | None,
    health_err: str | None,
) -> None:
    """Render (or re-render in place) the sidebar health badge and Backend & Model Status panel."""
    with container.container():
        if health_data is not None:
            status_val = str(health_data.get("status", "degraded"))
            qdrant = health_data.get("qdrant") or {}
            points = int(qdrant.get("points_count", 0))
            if status_val == "ok":
                st.success(f"API Online · {points} chunks")
            else:
                st.warning(f"API Degraded · {points} chunks")

            with st.expander("Backend & Model Status", expanded=(status_val != "ok")):
                st.markdown(
                    f"- **Collection:** `{qdrant.get('collection', '-')}` ({qdrant.get('status', '-')})"
                )
                st.markdown(f"- **Fast Model:** `{health_data.get('fast_model', '-')}`")
                st.markdown(f"- **Gen Model:** `{health_data.get('generation_model', '-')}`")
                st.markdown(f"- **Reranker:** `{health_data.get('reranker_model', '-')}`")
                providers = ", ".join(health_data.get("web_fallback_providers") or []) or "none"
                st.markdown(f"- **Web Fallback:** `{providers}`")
                firecrawl = health_data.get("firecrawl") or {}
                if firecrawl:
                    fc_status = firecrawl.get("status", "unknown")
                    fc_url = firecrawl.get("base_url") or "-"
                    st.markdown(f"- **Firecrawl Local:** `{fc_status}` (`{fc_url}`)")
                if not health_data.get("llm_configured") and health_data.get("llm_error"):
                    st.error(str(health_data["llm_error"]))
                if qdrant.get("status") == "unreachable":
                    st.error(f"Qdrant unreachable: {qdrant.get('error', 'Start `make up`')}")
        else:
            st.error("API Offline")
            st.caption(f"Start backend with `make api`. Error: {health_err}")


def render_sidebar() -> dict[str, Any]:
    """Render sidebar controls and return current runtime settings dictionary."""
    with st.sidebar:
        st.title("🧠 AdaptiveRAG")
        st.caption("Query-Aware Agentic RAG · Glass-Box Inspector")

        api_url = st.text_input(
            "FastAPI Backend URL",
            value=st.session_state.get("api_url", DEFAULT_API_URL),
            help="Address where `make api` (uvicorn) is running.",
        ).rstrip("/")
        st.session_state["api_url"] = api_url

        api_key = st.text_input(
            "API Key (X-API-Key, optional)",
            value=st.session_state.get("api_key", DEFAULT_API_KEY),
            type="password",
            help="Set when backend API_AUTH_KEY is enabled.",
        ).strip()
        st.session_state["api_key"] = api_key

        if st.button("↻ Refresh Backend Health", width="stretch") or (
            "health_cache" not in st.session_state
        ):
            st.session_state["health_cache"] = fetch_health(api_url, api_key=api_key)

        health_placeholder = st.empty()
        health_data, health_err = st.session_state["health_cache"]
        _render_sidebar_health(health_placeholder, health_data, health_err)

        st.divider()
        st.subheader("Ablation & Pipeline Controls")
        preset = st.selectbox(
            "Pipeline Mode / Ablation Preset",
            options=list(PRESET_LABELS.keys()),
            format_func=lambda k: PRESET_LABELS[k],
            index=0,
        )

        custom_ablation: dict[str, Any] | None = None
        if preset == "custom":
            retrieval_mode = st.selectbox(
                "Retrieval Mode",
                options=["hybrid", "dense_only", "sparse_only"],
                index=0,
            )
            enable_routing = st.checkbox("Query Routing (Factual/Analytical/Conv)", value=True)
            enable_multi_query = st.checkbox("Analytical Multi-Query Decomposition", value=True)
            enable_rerank = st.checkbox("Cross-Encoder Reranking", value=True)
            enable_dynamic_k = st.checkbox("3-Stage Dynamic Top-K", value=True)
            fixed_top_k = st.slider(
                "Static Top-K (when Dynamic Top-K off)",
                min_value=1,
                max_value=15,
                value=5,
                disabled=enable_dynamic_k,
            )
            enable_confidence = st.checkbox("Confidence + CRAG Self-Correction", value=True)
            enable_web_fallback = st.checkbox("Web Search Fallback", value=True)
            enable_citations = st.checkbox("Two-Tier Citation Validation", value=True)
            lang_choice = st.selectbox(
                "Corpus Language Filter", options=["All", "en", "tr"], index=0
            )

            custom_ablation = {
                "retrieval_mode": retrieval_mode,
                "enable_query_routing": enable_routing,
                "enable_multi_query": enable_multi_query,
                "enable_rerank": enable_rerank,
                "enable_dynamic_k": enable_dynamic_k,
                "fixed_top_k": fixed_top_k,
                "enable_confidence_gating": enable_confidence,
                "enable_web_fallback": enable_web_fallback,
                "enable_citation_validation": enable_citations,
                "language_filter": None if lang_choice == "All" else lang_choice,
            }

        use_streaming = st.toggle("Live SSE Trace Streaming", value=True)
        enable_budget_override = st.checkbox("Per-Request USD Cap Override", value=False)
        max_cost_usd: float | None = None
        if enable_budget_override:
            max_cost_usd = float(
                st.slider(
                    "Max Request Cost (USD)",
                    min_value=0.01,
                    max_value=0.50,
                    value=0.05,
                    step=0.01,
                )
            )

        st.divider()
        st.subheader("Corpus Ingestion")
        recreate_col = st.checkbox("Recreate Qdrant Collection", value=False)
        if st.button("📥 Index Local Corpus (data/raw)", width="stretch"):
            with st.spinner("Chunking & indexing data/raw into Qdrant..."):
                ingest_res, ingest_err = trigger_ingest(
                    api_url,
                    recreate_collection=recreate_col,
                    api_key=api_key,
                )
            if ingest_err:
                st.error(f"Ingest failed: {ingest_err}")
            elif ingest_res:
                st.success(
                    f"Indexed {ingest_res['documents_processed']} docs -> "
                    f"{ingest_res['chunks_created']} chunks "
                    f"({ingest_res['total_collection_points']} total points, "
                    f"{ingest_res['elapsed_ms']:.0f} ms)."
                )
                st.session_state["health_cache"] = fetch_health(api_url, api_key=api_key)
                _render_sidebar_health(health_placeholder, *st.session_state["health_cache"])

        with st.expander("🌐 Index Web URL / Docs Site (Firecrawl Local)", expanded=False):
            fc_info = (health_data or {}).get("firecrawl") or {}
            fc_status = str(fc_info.get("status", "unknown"))
            if fc_status == "ok":
                st.caption(f"🟢 Firecrawl Local online at `{fc_info.get('base_url', '')}`")
            elif fc_status == "unreachable":
                st.caption(
                    f"🟡 Firecrawl Local offline (`{fc_info.get('base_url', 'http://localhost:3002')}`). "
                    "Run `firecrawl-local serve` to enable live scraping."
                )

            web_mode = st.radio(
                "Web Ingest Mode",
                options=["scrape", "crawl"],
                format_func=lambda m: "Scrape URL(s)" if m == "scrape" else "Crawl Docs Site",
                horizontal=True,
                key="firecrawl_web_mode",
            )
            urls_text = ""
            crawl_url = ""
            crawl_limit = 8
            crawl_depth: int | None = None
            if web_mode == "scrape":
                urls_text = st.text_area(
                    "Public URLs (one per line)",
                    placeholder=(
                        "https://docs.qdrant.tech/concepts/hybrid-queries/\n"
                        "https://docs.firecrawl.dev/quickstarts/firecrawl-local"
                    ),
                    height=90,
                    key="firecrawl_urls_text",
                )
            else:
                crawl_url = st.text_input(
                    "Root Docs URL",
                    placeholder="https://docs.qdrant.tech/concepts/",
                    key="firecrawl_crawl_url",
                )
                crawl_limit = int(
                    st.slider(
                        "Max Pages to Crawl",
                        min_value=1,
                        max_value=25,
                        value=8,
                        key="firecrawl_crawl_limit",
                    )
                )
                crawl_depth = int(
                    st.slider(
                        "Max Link Depth",
                        min_value=1,
                        max_value=4,
                        value=2,
                        key="firecrawl_crawl_depth",
                    )
                )
            only_main = st.checkbox(
                "Extract main content only",
                value=True,
                key="firecrawl_only_main",
            )

            if st.button("🚀 Scrape & Index into Qdrant", width="stretch"):
                try:
                    web_payload = build_web_ingest_payload(
                        mode=web_mode,
                        urls_text=urls_text,
                        crawl_url=crawl_url,
                        crawl_limit=crawl_limit,
                        crawl_max_depth=crawl_depth,
                        only_main_content=only_main,
                        recreate_collection=recreate_col,
                    )
                except ValueError as exc:
                    st.warning(str(exc))
                else:
                    spinner_msg = (
                        f"Crawling up to {crawl_limit} pages via Firecrawl Local..."
                        if web_mode == "crawl"
                        else "Scraping URL(s) via Firecrawl Local & indexing into Qdrant..."
                    )
                    with st.spinner(spinner_msg):
                        web_res, web_err = trigger_ingest_payload(
                            api_url,
                            web_payload,
                            api_key=api_key,
                        )
                    if web_err:
                        st.error(f"Web ingest failed: {web_err}")
                    elif web_res:
                        st.success(
                            f"Indexed {web_res['documents_processed']} page(s) -> "
                            f"{web_res['chunks_created']} chunks "
                            f"({web_res['total_collection_points']} total points, "
                            f"{web_res['elapsed_ms']:.0f} ms)."
                        )
                        indexed_urls = web_res.get("source_urls") or []
                        if indexed_urls:
                            st.caption("Indexed URLs:")
                            for u in indexed_urls[:10]:
                                st.markdown(f"- [{u}]({u})")
                        st.session_state["health_cache"] = fetch_health(api_url, api_key=api_key)
                _render_sidebar_health(health_placeholder, *st.session_state["health_cache"])

        st.divider()
        st.subheader("Quick Demo Queries (TR + EN)")
        for btn_label, demo_text in DEMO_QUERIES:
            if st.button(btn_label, width="stretch"):
                st.session_state["pending_query"] = demo_text

        st.divider()
        if st.button("🗑️ Clear Conversation", width="stretch"):
            st.session_state["turns"] = []
            st.session_state["selected_turn_idx"] = None
            st.rerun()

    return {
        "api_url": api_url,
        "api_key": api_key,
        "preset": preset,
        "custom_ablation": custom_ablation,
        "use_streaming": use_streaming,
        "max_cost_usd": max_cost_usd,
    }


def execute_user_query(query: str, config: dict[str, Any]) -> None:
    """Execute a query against FastAPI (SSE streaming or batch) and save the turn."""
    turns: list[dict[str, Any]] = st.session_state.setdefault("turns", [])
    chat_history: list[dict[str, str]] = []
    for turn in turns:
        chat_history.append({"role": "user", "content": turn["query"]})
        chat_history.append({"role": "assistant", "content": turn["result"].get("answer", "")})

    payload = build_query_payload(
        query,
        chat_history,
        preset=config["preset"],
        custom_ablation=config["custom_ablation"],
        max_cost_usd=config["max_cost_usd"],
    )

    with st.chat_message("user"):
        st.markdown(query)

    final_result: dict[str, Any] | None = None
    with st.chat_message("assistant"):
        if config["use_streaming"]:
            with st.status("Running AdaptiveRAG agentic pipeline...", expanded=True) as status_box:
                try:
                    for event_name, event_data in stream_query_events(
                        config["api_url"],
                        payload,
                        api_key=config.get("api_key", ""),
                    ):
                        if event_name == "node_update":
                            node_key = str(event_data.get("node", "unknown"))
                            icon, pretty, _ = NODE_METADATA.get(
                                node_key, ("⚙️", node_key, "#64748B")
                            )
                            traces = event_data.get("trace") or []
                            if traces:
                                t0 = traces[0]
                                ms = float(t0.get("latency_ms", 0.0))
                                summary = str(t0.get("summary", ""))
                                st.markdown(f"{icon} **{pretty}** (`{ms:.1f} ms`) — {summary}")
                                status_box.update(label=f"{icon} Completed: {pretty}")
                            else:
                                st.markdown(f"{icon} **{pretty}**")
                        elif event_name == "final_answer":
                            final_result = event_data
                        elif event_name == "error":
                            msg = event_data.get("message", "Unknown streaming error")
                            status_box.update(label=f"Pipeline error: {msg}", state="error")
                            st.error(f"Pipeline error ({event_data.get('code', 'error')}): {msg}")
                            return

                    if final_result is not None:
                        total_ms = float(final_result.get("total_latency_ms", 0.0))
                        status_box.update(
                            label=f"Completed in {total_ms:.0f} ms",
                            state="complete",
                            expanded=False,
                        )
                except Exception as exc:
                    status_box.update(label="Connection / API error", state="error")
                    st.error(str(exc))
                    return
        else:
            with st.spinner("Running AdaptiveRAG pipeline..."):
                try:
                    final_result = run_query_batch(
                        config["api_url"],
                        payload,
                        api_key=config.get("api_key", ""),
                    )
                except Exception as exc:
                    st.error(str(exc))
                    return

        if final_result is None:
            st.error("Stream finished without a final_answer payload.")
            return

        render_answer_badges(final_result)
        st.markdown(str(final_result.get("answer", "")))

    turns.append({"query": query, "result": final_result})
    st.session_state["selected_turn_idx"] = len(turns) - 1


def main() -> None:
    st.set_page_config(
        page_title="AdaptiveRAG — Glass-Box Inspector",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    config = render_sidebar()

    st.title("AdaptiveRAG · Glass-Box Agentic Inspector")
    st.caption(
        "Route → Decompose → Hybrid RRF (Dense + BM25) → Cross-Encoder Rerank & Dynamic Top-K "
        "→ Confidence / CRAG Gate → Web Fallback → Two-Tier Citation Audit"
    )

    turns: list[dict[str, Any]] = st.session_state.setdefault("turns", [])

    # Render conversation history
    for idx, turn in enumerate(turns):
        with st.chat_message("user"):
            st.markdown(turn["query"])
        with st.chat_message("assistant"):
            render_answer_badges(turn["result"])
            st.markdown(str(turn["result"].get("answer", "")))
            if st.button(
                f"🔬 Inspect Turn #{idx + 1}",
                key=f"inspect_turn_{idx}",
                type="tertiary",
            ):
                st.session_state["selected_turn_idx"] = idx

    # Handle chat input or sidebar demo button
    pending_query = st.session_state.pop("pending_query", None)
    user_input = st.chat_input("Ask a technical question in English or Turkish...")
    active_query = pending_query or user_input

    if active_query:
        execute_user_query(active_query, config)

    # Render Glass-Box Inspector for the selected (or latest) turn
    if turns:
        st.divider()
        selected_idx = st.session_state.get("selected_turn_idx")
        if selected_idx is None or not (0 <= selected_idx < len(turns)):
            selected_idx = len(turns) - 1

        if len(turns) > 1:
            selected_idx = st.selectbox(
                "Inspecting Conversation Turn",
                options=list(range(len(turns))),
                index=selected_idx,
                format_func=lambda i: f"Turn #{i + 1}: {turns[i]['query'][:80]}",
            )
            st.session_state["selected_turn_idx"] = selected_idx

        render_glass_box_inspector(turns[selected_idx]["result"])
    else:
        st.info(
            "Ask a question below or pick one of the **TR / EN Quick Demo Queries** in the sidebar "
            "to watch the live LangGraph agentic trace and Glass-Box diagnostics."
        )


if __name__ == "__main__":
    main()
