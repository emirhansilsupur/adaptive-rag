"""Deterministic Information Retrieval, Agentic Control, FinOps metrics, and report renderers."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from eval.schemas import (
    BenchmarkReport,
    ConfigBenchmarkSummary,
    EvalStratum,
    GoldenQuery,
    IRScores,
    QueryEvalRecord,
    StratumSummary,
)


def deduplicate_doc_ids(doc_ids: Sequence[str]) -> list[str]:
    """Preserve first-rank order while collapsing multiple chunks from the same document."""
    seen: set[str] = set()
    ordered: list[str] = []
    for doc_id in doc_ids:
        cleaned = doc_id.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            ordered.append(cleaned)
    return ordered


def recall_at_k(
    retrieved_doc_ids: Sequence[str],
    expected_doc_ids: Sequence[str],
    *,
    k: int | None = None,
) -> float:
    """Fraction of ground-truth documents retrieved within the top-k unique document ranks."""
    expected = set(deduplicate_doc_ids(expected_doc_ids))
    if not expected:
        return 0.0
    ranked = deduplicate_doc_ids(retrieved_doc_ids)
    if k is not None:
        ranked = ranked[:k]
    hits = sum(1 for doc_id in ranked if doc_id in expected)
    return hits / len(expected)


def precision_at_k(
    retrieved_doc_ids: Sequence[str],
    expected_doc_ids: Sequence[str],
    *,
    k: int | None = None,
) -> float:
    """Fraction of top-k unique retrieved documents that belong to ground truth."""
    expected = set(deduplicate_doc_ids(expected_doc_ids))
    ranked = deduplicate_doc_ids(retrieved_doc_ids)
    if k is not None:
        ranked = ranked[:k]
    if not ranked or not expected:
        return 0.0
    hits = sum(1 for doc_id in ranked if doc_id in expected)
    return hits / len(ranked)


def mrr_at_k(
    retrieved_doc_ids: Sequence[str],
    expected_doc_ids: Sequence[str],
    *,
    k: int | None = None,
) -> float:
    """Reciprocal rank (1 / rank) of the first relevant document in top-k unique ranks."""
    expected = set(deduplicate_doc_ids(expected_doc_ids))
    if not expected:
        return 0.0
    ranked = deduplicate_doc_ids(retrieved_doc_ids)
    if k is not None:
        ranked = ranked[:k]
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in expected:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_doc_ids: Sequence[str],
    expected_doc_ids: Sequence[str],
    *,
    k: int | None = None,
) -> float:
    """Binary Normalized Discounted Cumulative Gain over top-k unique document ranks."""
    expected = set(deduplicate_doc_ids(expected_doc_ids))
    if not expected:
        return 0.0
    ranked = deduplicate_doc_ids(retrieved_doc_ids)
    effective_k = len(ranked) if k is None else min(k, len(ranked))
    if effective_k == 0:
        return 0.0

    dcg = 0.0
    for rank, doc_id in enumerate(ranked[:effective_k], start=1):
        if doc_id in expected:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(expected), k if k is not None else len(expected))
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return min(1.0, dcg / idcg) if idcg > 0.0 else 0.0


_FACT_TOKEN_RE = re.compile(r"[a-z0-9_]+(?:[.:/-][a-z0-9_]+)*")
_FACT_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "per",
        "the",
        "to",
        "vs",
        "with",
        "bir",
        "bu",
        "da",
        "de",
        "en",
        "icin",
        "ile",
        "ve",
        "veya",
    }
)


def _normalize_fact_text(text: str) -> str:
    lowered = text.lower()
    tr_map = str.maketrans(
        {
            "ü": "u",
            "ö": "o",
            "ğ": "g",
            "ş": "s",
            "ı": "i",
            "İ": "i",
            "ç": "c",
        }
    )
    return lowered.translate(tr_map)


def _extract_fact_tokens(text: str) -> list[str]:
    norm = _normalize_fact_text(text)
    raw = _FACT_TOKEN_RE.findall(norm)
    tokens: list[str] = []
    for tok in raw:
        # Also break dotted/hyphenated compound tokens into sub-tokens while retaining exact token
        parts = [tok]
        if any(sep in tok for sep in (":", "/", "-")):
            parts.extend(p for p in re.split(r"[:/-]", tok) if p)
        for piece in parts:
            if piece in _FACT_STOPWORDS:
                continue
            if len(piece) <= 1 and not piece.isdigit():
                continue
            tokens.append(piece)
    return list(dict.fromkeys(tokens))


def compute_fact_coverage(
    passage_texts: Sequence[str],
    key_facts: Sequence[str],
    *,
    token_match_threshold: float = 0.70,
) -> float | None:
    """Deterministic $0 passage completeness metric: fraction of key_facts supported by retrieved text."""
    valid_facts = [fact.strip() for fact in key_facts if fact and fact.strip()]
    if not valid_facts:
        return None
    if not passage_texts:
        return 0.0

    combined_norm = _normalize_fact_text("\n".join(passage_texts))
    ctx_tokens = set(_extract_fact_tokens(combined_norm))
    ctx_alpha_stems = {t[:5] for t in ctx_tokens if len(t) >= 5 and t.isalpha()}

    covered = 0
    for fact in valid_facts:
        fact_norm = _normalize_fact_text(fact)
        if fact_norm in combined_norm:
            covered += 1
            continue

        fact_tokens = _extract_fact_tokens(fact)
        if not fact_tokens:
            continue

        matched = 0
        for tok in fact_tokens:
            if tok in ctx_tokens:
                matched += 1
            elif len(tok) >= 5 and tok.isalpha() and tok[:5] in ctx_alpha_stems:
                # Handles agglutinative Turkish/English inflections without fuzzy-matching numbers
                matched += 1

        if (matched / len(fact_tokens)) >= token_match_threshold:
            covered += 1

    return round(covered / len(valid_facts), 6)


def compute_ir_scores(
    retrieved_doc_ids: Sequence[str],
    expected_doc_ids: Sequence[str],
    *,
    k: int | None = None,
) -> IRScores:
    """Compute Recall@k, Precision@k, MRR@k, NDCG@k, and Hit@k for one retrieval query."""
    ranked = deduplicate_doc_ids(retrieved_doc_ids)
    k_eval = len(ranked) if k is None else min(k, len(ranked))
    rec = recall_at_k(ranked, expected_doc_ids, k=k)
    prec = precision_at_k(ranked, expected_doc_ids, k=k)
    mrr = mrr_at_k(ranked, expected_doc_ids, k=k)
    ndcg = ndcg_at_k(ranked, expected_doc_ids, k=k)
    return IRScores(
        k_eval=k_eval,
        recall_at_k=round(rec, 6),
        precision_at_k=round(prec, 6),
        mrr_at_k=round(mrr, 6),
        ndcg_at_k=round(ndcg, 6),
        hit_at_k=rec > 0.0,
    )


def percentile(values: Sequence[float], p: float) -> float:
    """Compute linear-interpolated percentile p in [0, 100] without external dependencies."""
    if not values:
        return 0.0
    if not (0.0 <= p <= 100.0):
        raise ValueError(f"Percentile p must be in [0, 100], got {p}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (p / 100.0) * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] + frac * (ordered[hi] - ordered[lo]))


def compute_binary_precision_recall(
    y_true: Sequence[bool],
    y_pred: Sequence[bool],
) -> tuple[float | None, float | None]:
    """Return (precision, recall) for boolean event detection (fallback / abstention)."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have identical lengths.")
    tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if (not t) and p)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t and (not p))

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else None
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else None
    return precision, recall


def _mean_optional(values: Sequence[float | None]) -> float | None:
    valid = [v for v in values if v is not None and not math.isnan(v)]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 6)


def _summarize_slice(
    name: str,
    records: Sequence[QueryEvalRecord],
) -> StratumSummary:
    count = len(records)
    if count == 0:
        return StratumSummary(name=name)

    ir_records = [r.ir_scores for r in records if r.ir_scores is not None]
    routing_flags = [
        1.0 if r.routing_correct else 0.0 for r in records if r.routing_correct is not None
    ]

    return StratumSummary(
        name=name,
        count=count,
        retrieval_count=len(ir_records),
        routing_accuracy=_mean_optional(routing_flags),
        mean_recall=_mean_optional([s.recall_at_k for s in ir_records]),
        mean_precision=_mean_optional([s.precision_at_k for s in ir_records]),
        mean_mrr=_mean_optional([s.mrr_at_k for s in ir_records]),
        mean_ndcg=_mean_optional([s.ndcg_at_k for s in ir_records]),
        mean_fact_coverage=_mean_optional([r.fact_coverage for r in records]),
        avg_selected_k=round(sum(r.selected_k for r in records) / count, 3),
        avg_context_tokens=round(sum(r.context_tokens for r in records) / count, 2),
        mean_latency_ms=round(sum(r.latency_ms for r in records) / count, 2),
        fallback_rate=round(sum(1 for r in records if r.fallback_used) / count, 6),
        abstention_rate=round(sum(1 for r in records if r.abstained) / count, 6),
    )


def aggregate_config_summary(
    *,
    config_name: str,
    mode: str,
    queries: Sequence[GoldenQuery],
    records: Sequence[QueryEvalRecord],
) -> ConfigBenchmarkSummary:
    """Aggregate per-query evaluation records into a full ablation configuration scorecard."""
    query_by_id = {q.id: q for q in queries}
    count = len(records)
    if count == 0:
        return ConfigBenchmarkSummary(
            config_name=config_name,
            mode="live" if mode == "live" else "offline-ir",
        )

    ir_records = [r.ir_scores for r in records if r.ir_scores is not None]
    routing_flags = [
        1.0 if r.routing_correct else 0.0 for r in records if r.routing_correct is not None
    ]

    fb_true: list[bool] = []
    fb_pred: list[bool] = []
    abs_true: list[bool] = []
    abs_pred: list[bool] = []
    for rec in records:
        q = query_by_id.get(rec.query_id)
        if q is not None:
            fb_true.append(q.expect_web_fallback)
            fb_pred.append(rec.fallback_used)
            abs_true.append(q.expect_abstention)
            abs_pred.append(rec.abstained)

    fb_prec, fb_rec = compute_binary_precision_recall(fb_true, fb_pred)
    abs_prec, abs_rec = compute_binary_precision_recall(abs_true, abs_pred)

    latencies = [r.latency_ms for r in records]
    total_tokens = sum(r.total_tokens for r in records)
    total_cost = round(sum(r.estimated_cost_usd for r in records), 8)
    total_cache_hits = sum(r.cache_hits for r in records)

    # Group slices by stratum and language
    by_stratum: dict[str, StratumSummary] = {}
    for stratum in EvalStratum:
        slice_recs = [r for r in records if r.stratum == stratum]
        if slice_recs:
            by_stratum[stratum.value] = _summarize_slice(stratum.value, slice_recs)

    by_language: dict[str, StratumSummary] = {}
    for lang in sorted({r.language for r in records}):
        lang_recs = [r for r in records if r.language == lang]
        by_language[lang] = _summarize_slice(lang, lang_recs)

    ragas_items = [r.ragas for r in records if r.ragas is not None]

    return ConfigBenchmarkSummary(
        config_name=config_name,
        mode="live" if mode == "live" else "offline-ir",
        query_count=count,
        error_count=sum(1 for r in records if r.error is not None),
        retrieval_query_count=len(ir_records),
        mean_recall=_mean_optional([s.recall_at_k for s in ir_records]),
        mean_precision=_mean_optional([s.precision_at_k for s in ir_records]),
        mean_mrr=_mean_optional([s.mrr_at_k for s in ir_records]),
        mean_ndcg=_mean_optional([s.ndcg_at_k for s in ir_records]),
        hit_rate=_mean_optional([1.0 if s.hit_at_k else 0.0 for s in ir_records]),
        mean_fact_coverage=_mean_optional([r.fact_coverage for r in records]),
        routing_accuracy=_mean_optional(routing_flags),
        fallback_precision=round(fb_prec, 6) if fb_prec is not None else None,
        fallback_recall=round(fb_rec, 6) if fb_rec is not None else None,
        abstention_precision=round(abs_prec, 6) if abs_prec is not None else None,
        abstention_recall=round(abs_rec, 6) if abs_rec is not None else None,
        mean_citation_precision=_mean_optional([r.citation_precision for r in records]),
        mean_claim_support_rate=_mean_optional([r.claim_support_rate for r in records]),
        citation_pass_rate=_mean_optional(
            [1.0 if r.citation_passed else 0.0 for r in records if r.citation_passed is not None]
        ),
        avg_selected_k=round(sum(r.selected_k for r in records) / count, 3),
        avg_context_tokens=round(sum(r.context_tokens for r in records) / count, 2),
        mean_latency_ms=round(sum(latencies) / count, 2),
        latency_p50_ms=round(percentile(latencies, 50.0), 2),
        latency_p95_ms=round(percentile(latencies, 95.0), 2),
        total_tokens=total_tokens,
        total_cost_usd=total_cost,
        cost_per_query_usd=round(total_cost / count, 8),
        total_cache_hits=total_cache_hits,
        rewrite_rate=round(sum(1 for r in records if r.rewrite_attempts > 0) / count, 6),
        regeneration_rate=round(sum(1 for r in records if r.regeneration_attempts > 0) / count, 6),
        ragas_faithfulness=_mean_optional([x.faithfulness for x in ragas_items]),
        ragas_answer_relevancy=_mean_optional([x.answer_relevancy for x in ragas_items]),
        ragas_context_precision=_mean_optional([x.context_precision for x in ragas_items]),
        ragas_context_recall=_mean_optional([x.context_recall for x in ragas_items]),
        by_stratum=by_stratum,
        by_language=by_language,
    )


def _fmt_pct(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "—"


def _fmt_float(value: float | None, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value is not None else "—"


def render_markdown_report(report: BenchmarkReport) -> str:
    """Render a GitHub-Flavored Markdown ablation summary suitable for README or CI artifacts."""
    lines: list[str] = [
        f"# AdaptiveRAG Ablation Benchmark Report (`{report.mode}`)",
        "",
        f"- **Run ID:** `{report.run_id}`",
        f"- **Timestamp (UTC):** `{report.timestamp_utc}`",
        f"- **Dataset:** `{report.dataset_source}` (`{report.total_queries}` queries)",
        f"- **Configs Evaluated:** `{', '.join(report.configs)}`",
        f"- **Budget Cap Triggered:** `{'Yes' if report.budget_exceeded else 'No'}`",
        "",
        "## 1. Ablation Scorecard Overview",
        "",
        "| Config | Recall@k | Precision@k | MRR | NDCG@k | Fact Cov | Routing Acc | Citation Prec | Claim Support | Avg k | Avg Ctx Tok | p50 / p95 (ms) | Cost (USD) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for s in report.summaries:
        lines.append(
            f"| `{s.config_name}` | {_fmt_pct(s.mean_recall)} | {_fmt_pct(s.mean_precision)} | "
            f"{_fmt_float(s.mean_mrr)} | {_fmt_float(s.mean_ndcg)} | {_fmt_pct(s.mean_fact_coverage)} | "
            f"{_fmt_pct(s.routing_accuracy)} | "
            f"{_fmt_pct(s.mean_citation_precision)} | {_fmt_pct(s.mean_claim_support_rate)} | "
            f"{s.avg_selected_k:.2f} | {s.avg_context_tokens:.0f} | "
            f"{s.latency_p50_ms:.1f} / {s.latency_p95_ms:.1f} | ${s.total_cost_usd:.5f} |"
        )

    # Optional RAGAS section if any config scored RAGAS metrics
    has_ragas = any(
        s.ragas_faithfulness is not None
        or s.ragas_answer_relevancy is not None
        or s.ragas_context_precision is not None
        or s.ragas_context_recall is not None
        for s in report.summaries
    )
    if has_ragas:
        lines.extend(
            [
                "",
                "## 2. RAGAS LLM-as-a-Judge Metrics",
                "",
                "| Config | Faithfulness | Answer Relevancy | Context Precision | Context Recall |",
                "| :--- | :---: | :---: | :---: | :---: |",
            ]
        )
        for s in report.summaries:
            lines.append(
                f"| `{s.config_name}` | {_fmt_float(s.ragas_faithfulness)} | "
                f"{_fmt_float(s.ragas_answer_relevancy)} | "
                f"{_fmt_float(s.ragas_context_precision)} | "
                f"{_fmt_float(s.ragas_context_recall)} |"
            )

    lines.extend(
        [
            "",
            "## 3. Agentic Control Breakdown (Fallback, Abstention, Loops)",
            "",
            "| Config | Fallback Prec | Fallback Rec | Abstention Prec | Abstention Rec | Rewrite Rate | Regen Rate | Cache Hits |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for s in report.summaries:
        lines.append(
            f"| `{s.config_name}` | {_fmt_pct(s.fallback_precision)} | {_fmt_pct(s.fallback_recall)} | "
            f"{_fmt_pct(s.abstention_precision)} | {_fmt_pct(s.abstention_recall)} | "
            f"{_fmt_pct(s.rewrite_rate)} | {_fmt_pct(s.regeneration_rate)} | {s.total_cache_hits} |"
        )

    lines.extend(
        [
            "",
            "## 4. Per-Stratum Breakdown",
            "",
            "| Config | Stratum | N | Recall@k | MRR | NDCG@k | Avg k | Avg Ctx Tok | Fallback Rate | Abstention Rate |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for s in report.summaries:
        for stratum_name, st in s.by_stratum.items():
            lines.append(
                f"| `{s.config_name}` | `{stratum_name}` | {st.count} | {_fmt_pct(st.mean_recall)} | "
                f"{_fmt_float(st.mean_mrr)} | {_fmt_float(st.mean_ndcg)} | {st.avg_selected_k:.2f} | "
                f"{st.avg_context_tokens:.0f} | {_fmt_pct(st.fallback_rate)} | {_fmt_pct(st.abstention_rate)} |"
            )

    lines.extend(
        [
            "",
            "## 5. Multilingual Breakdown (EN vs TR)",
            "",
            "| Config | Language | N | Recall@k | MRR | NDCG@k | Avg k | Mean Latency (ms) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for s in report.summaries:
        for lang, st in s.by_language.items():
            lines.append(
                f"| `{s.config_name}` | `{lang}` | {st.count} | {_fmt_pct(st.mean_recall)} | "
                f"{_fmt_float(st.mean_mrr)} | {_fmt_float(st.mean_ndcg)} | {st.avg_selected_k:.2f} | "
                f"{st.mean_latency_ms:.1f} |"
            )

    return "\n".join(lines) + "\n"


def build_ablation_chart_html(report: BenchmarkReport) -> str | None:
    """Build an interactive Plotly HTML comparison chart if plotly is installed."""
    if not report.summaries:
        return None
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    configs = [s.config_name for s in report.summaries]
    fig = go.Figure()
    metric_specs = [
        ("Recall@k", [s.mean_recall or 0.0 for s in report.summaries]),
        ("Precision@k", [s.mean_precision or 0.0 for s in report.summaries]),
        ("MRR", [s.mean_mrr or 0.0 for s in report.summaries]),
        ("NDCG@k", [s.mean_ndcg or 0.0 for s in report.summaries]),
    ]
    for label, values in metric_specs:
        fig.add_trace(go.Bar(name=label, x=configs, y=values))

    fig.update_layout(
        title=f"AdaptiveRAG Ablation Comparison ({report.mode})",
        barmode="group",
        yaxis={"title": "Score", "range": [0, 1.05]},
        xaxis={"title": "Ablation Configuration"},
        template="plotly_white",
    )
    return str(fig.to_html(full_html=True, include_plotlyjs="cdn"))
