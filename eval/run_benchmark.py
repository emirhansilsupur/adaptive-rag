"""4-Way Ablation Benchmark Runner for AdaptiveRAG (offline $0 IR + live Agentic/RAGAS).

Usage examples:
    # Zero-API-cost offline IR ablation (in-memory Qdrant seeded from data/raw):
    uv run python -m eval.run_benchmark --mode offline-ir --in-memory

    # Full 4-way live Agentic ablation with strict $0.50 FinOps budget cap:
    uv run python -m eval.run_benchmark --mode live --in-memory --max-cost-usd 0.50

    # Live ablation + optional RAGAS LLM-as-a-Judge pass on a 12-query slice:
    uv run python -m eval.run_benchmark --mode live --limit 12 --with-ragas --max-cost-usd 0.50

    # External HuggingFace benchmark (HotpotQA / RAGBench) adapter:
    uv run python -m eval.run_benchmark --mode offline-ir --in-memory \\
        --hf-dataset hotpot_qa --hf-subset distractor --limit 25
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import structlog

from adaptiverag.config import Settings, StrategiesConfig, get_settings, load_strategies
from adaptiverag.graph.state import AblationConfig
from adaptiverag.graph.workflow import AdaptiveRAGWorkflow
from adaptiverag.ingestion.chunker import ChunkingConfig, chunk_documents, count_tokens
from adaptiverag.ingestion.loader import LoadedDocument, load_directory
from adaptiverag.llm.cost import BudgetExceededError, CostTracker
from adaptiverag.observability.tracing import LangfuseObservability, configure_logging
from adaptiverag.retrieval.dynamic_k import select_dynamic_top_k
from adaptiverag.retrieval.embeddings import (
    DenseEmbedder,
    FastEmbedDenseEmbedder,
    FastEmbedSparseEmbedder,
)
from adaptiverag.retrieval.parent_child import expand_to_parent_chunks
from adaptiverag.retrieval.qdrant_store import QdrantHybridStore, RetrievalMode
from adaptiverag.retrieval.reranker import FastEmbedCrossEncoderReranker, Reranker
from adaptiverag.schemas.query import QueryType
from adaptiverag.schemas.response import FinalAnswer
from adaptiverag.schemas.retrieval import EvidenceChunk
from adaptiverag.verification.confidence import compute_deterministic_confidence
from eval.datasets import (
    filter_queries,
    load_external_jsonl,
    load_golden_jsonl,
    load_huggingface_dataset,
)
from eval.langfuse_eval import publish_benchmark_report_to_langfuse
from eval.metrics import (
    aggregate_config_summary,
    build_ablation_chart_html,
    compute_fact_coverage,
    compute_ir_scores,
    render_markdown_report,
)
from eval.ragas_runner import aevaluate_with_ragas
from eval.schemas import (
    BenchmarkReport,
    ConfigBenchmarkSummary,
    EvalStratum,
    GoldenQuery,
    QueryEvalRecord,
    RagasScores,
)

logger = structlog.get_logger(__name__)

OFFLINE_IR_CONFIGS: tuple[str, ...] = (
    "naive_dense",
    "hybrid_static",
    "hybrid_rerank",
    "adaptive_dynamic_k",
)

PARENT_CHILD_COMPARE_CONFIGS: tuple[str, ...] = (
    "flat_hybrid_rerank",
    "flat_adaptive_dynamic_k",
    "parent_child_hybrid_rerank",
    "parent_child_adaptive_dynamic_k",
)

LIVE_CONFIG_NAMES: tuple[str, ...] = (
    "naive_dense",
    "hybrid_rerank",
    "adaptive_routing",
    "full",
)


def resolve_live_ablation(config_name: str, *, fixed_top_k: int = 5) -> AblationConfig:
    """Resolve a benchmark preset name into its corresponding AblationConfig."""
    normalized = config_name.strip().lower()
    if normalized == "naive_dense":
        return AblationConfig.naive_dense(fixed_top_k=fixed_top_k)
    if normalized == "hybrid_rerank":
        return AblationConfig.hybrid_rerank(fixed_top_k=fixed_top_k)
    if normalized == "adaptive_routing":
        return AblationConfig.adaptive_routing()
    if normalized == "full":
        return AblationConfig.full()
    raise ValueError(
        f"Unknown live ablation config '{config_name}'. Expected one of {LIVE_CONFIG_NAMES}."
    )


def _estimate_context_tokens(chunks: Sequence[EvidenceChunk]) -> int:
    total = 0
    for chunk in chunks:
        total += chunk.token_count if chunk.token_count > 0 else count_tokens(chunk.text)
    return total


class _BenchmarkRerankerCache:
    """Bit-exact content-addressed memoization avoiding duplicate ONNX scoring within one index."""

    def __init__(self, inner: Reranker) -> None:
        self._inner = inner
        self._scores: dict[tuple[str, str, str, str], tuple[float | None, float | None]] = {}

    @staticmethod
    def _key(query: str, chunk: EvidenceChunk) -> tuple[str, str, str, str]:
        return (query, chunk.title, chunk.section_path, chunk.text)

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    def rerank(
        self,
        query: str,
        chunks: Sequence[EvidenceChunk],
        *,
        top_n: int | None = None,
    ) -> list[EvidenceChunk]:
        return self._inner.rerank(query, chunks, top_n=top_n)

    async def arerank(
        self,
        query: str,
        chunks: Sequence[EvidenceChunk],
        *,
        top_n: int | None = None,
    ) -> list[EvidenceChunk]:
        if not chunks:
            return []
        missing = [c for c in chunks if self._key(query, c) not in self._scores]
        if missing:
            scored_missing = await self._inner.arerank(query, missing)
            for item in scored_missing:
                self._scores[self._key(query, item)] = (item.rerank_raw_score, item.rerank_score)

        rehydrated = [
            chunk.model_copy(
                update={
                    "rerank_raw_score": self._scores[self._key(query, chunk)][0],
                    "rerank_score": self._scores[self._key(query, chunk)][1],
                }
            )
            for chunk in chunks
        ]
        rehydrated.sort(
            key=lambda c: (
                c.rerank_score if c.rerank_score is not None else -1.0,
                c.rerank_raw_score if c.rerank_raw_score is not None else float("-inf"),
                c.fusion_score if c.fusion_score is not None else 0.0,
            ),
            reverse=True,
        )
        return rehydrated[:top_n] if top_n is not None else rehydrated


async def _evaluate_single_offline_ir(
    query: GoldenQuery,
    *,
    config_name: str,
    store: QdrantHybridStore,
    reranker: Reranker,
    strategies: StrategiesConfig,
    fixed_top_k: int,
    enable_parent_child: bool = False,
    record_config_name: str | None = None,
) -> QueryEvalRecord:
    started = time.perf_counter()
    output_config_name = record_config_name or config_name

    if config_name not in OFFLINE_IR_CONFIGS:
        raise ValueError(
            f"Unknown offline-ir config '{config_name}'. Expected one of {OFFLINE_IR_CONFIGS}."
        )

    # Adaptive config skips retrieval completely on conversational queries
    if config_name == "adaptive_dynamic_k" and (
        query.stratum == EvalStratum.CONVERSATIONAL or not query.needs_retrieval
    ):
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return QueryEvalRecord(
            query_id=query.id,
            stratum=query.stratum,
            language=query.language,
            config_name=output_config_name,
            predicted_query_type=QueryType.CONVERSATIONAL,
            routing_correct=query.expected_query_type == QueryType.CONVERSATIONAL,
            retrieved_doc_ids=[],
            selected_k=0,
            context_tokens=0,
            ir_scores=None,
            fact_coverage=None,
            confidence_score=None,
            confidence_level=None,
            fallback_used=False,
            abstained=False,
            latency_ms=round(elapsed_ms, 2),
        )

    strategy = (
        strategies.for_query_type(query.expected_query_type)
        if config_name == "adaptive_dynamic_k"
        else strategies.for_query_type(QueryType.FACTUAL)
    )

    if config_name == "naive_dense":
        candidates = await store.search_single(
            query.query,
            mode=RetrievalMode.DENSE_ONLY,
            dense_limit=max(fixed_top_k, 20),
            sparse_limit=0,
            fusion_limit=fixed_top_k,
        )
        selected = candidates[:fixed_top_k]
        predicted_qtype = QueryType.FACTUAL
    elif config_name == "hybrid_static":
        candidates = await store.search_single(
            query.query,
            mode=RetrievalMode.HYBRID,
            dense_limit=20,
            sparse_limit=25,
            fusion_limit=fixed_top_k,
        )
        selected = candidates[:fixed_top_k]
        predicted_qtype = QueryType.FACTUAL
    elif config_name == "hybrid_rerank":
        candidates = await store.search_single(
            query.query,
            mode=RetrievalMode.HYBRID,
            dense_limit=20,
            sparse_limit=25,
            fusion_limit=20,
        )
        reranked = await reranker.arerank(query.query, candidates)
        selected = reranked[:fixed_top_k]
        predicted_qtype = QueryType.FACTUAL
    else:
        # adaptive_dynamic_k: query-intent prefetch + Hybrid RRF + Cross-Encoder + 3-stage Dynamic Top-K
        candidates = await store.search_single(
            query.query,
            mode=RetrievalMode.HYBRID,
            dense_limit=strategy.dense_prefetch_limit,
            sparse_limit=strategy.sparse_prefetch_limit,
            fusion_limit=strategy.fusion_limit,
        )
        reranked = await reranker.arerank(query.query, candidates)
        selected, _decision = select_dynamic_top_k(reranked, strategy)
        predicted_qtype = query.expected_query_type

    if enable_parent_child and selected:
        selected, _ = expand_to_parent_chunks(
            selected,
            max_context_tokens=strategy.max_context_tokens,
        )

    signals = compute_deterministic_confidence(
        selected,
        sub_queries=[query.query],
        config=strategies.confidence,
    )
    conf_cfg = strategies.confidence
    if signals.deterministic_score >= conf_cfg.high_threshold:
        conf_level = "high"
    elif signals.deterministic_score < conf_cfg.low_threshold:
        conf_level = "low"
    else:
        conf_level = "medium"

    retrieved_doc_ids = [chunk.doc_id for chunk in selected if chunk.doc_id]
    ir_scores = (
        compute_ir_scores(retrieved_doc_ids, query.expected_doc_ids)
        if query.expected_doc_ids
        else None
    )
    fact_cov = (
        compute_fact_coverage([chunk.text for chunk in selected], query.key_facts)
        if query.key_facts
        else None
    )

    # In offline IR mode (no live web search), adaptive_dynamic_k flags LOW confidence as
    # abstention/fallback-candidate when evidence drops below relevance floor.
    low_evidence = config_name == "adaptive_dynamic_k" and (
        len(selected) == 0 or signals.deterministic_score < conf_cfg.low_threshold
    )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return QueryEvalRecord(
        query_id=query.id,
        stratum=query.stratum,
        language=query.language,
        config_name=output_config_name,
        predicted_query_type=predicted_qtype,
        routing_correct=(predicted_qtype == query.expected_query_type)
        if config_name == "adaptive_dynamic_k"
        else None,
        retrieved_doc_ids=retrieved_doc_ids,
        selected_k=len(selected),
        context_tokens=_estimate_context_tokens(selected),
        ir_scores=ir_scores,
        fact_coverage=fact_cov,
        confidence_score=round(signals.deterministic_score, 6),
        confidence_level=conf_level,
        fallback_used=False,
        abstained=low_evidence,
        latency_ms=round(elapsed_ms, 2),
    )


async def run_offline_ir_benchmark(
    queries: Sequence[GoldenQuery],
    *,
    store: QdrantHybridStore,
    reranker: Reranker,
    strategies: StrategiesConfig,
    configs: Sequence[str] | None = None,
    fixed_top_k: int = 5,
    enable_parent_child: bool = False,
    dataset_source: str = "data/eval/golden_set.jsonl",
) -> BenchmarkReport:
    """Run a deterministic $0 LLM-cost IR ablation benchmark across retrieval configurations."""
    active_configs = list(configs) if configs else list(OFFLINE_IR_CONFIGS)
    all_records: list[QueryEvalRecord] = []
    summaries: list[ConfigBenchmarkSummary] = []

    for config_name in active_configs:
        logger.info(
            "benchmark_offline_config_start",
            config=config_name,
            queries=len(queries),
            parent_child=enable_parent_child,
        )
        cfg_records: list[QueryEvalRecord] = []
        for query in queries:
            try:
                record = await _evaluate_single_offline_ir(
                    query,
                    config_name=config_name,
                    store=store,
                    reranker=reranker,
                    strategies=strategies,
                    fixed_top_k=fixed_top_k,
                    enable_parent_child=enable_parent_child,
                )
            except Exception as exc:
                logger.exception("benchmark_offline_query_error", query_id=query.id, error=str(exc))
                record = QueryEvalRecord(
                    query_id=query.id,
                    stratum=query.stratum,
                    language=query.language,
                    config_name=config_name,
                    error=str(exc),
                )
            cfg_records.append(record)

        all_records.extend(cfg_records)
        summaries.append(
            aggregate_config_summary(
                config_name=config_name,
                mode="offline-ir",
                queries=queries,
                records=cfg_records,
            )
        )

    return BenchmarkReport(
        run_id=f"offline_{uuid.uuid4().hex[:8]}",
        timestamp_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        mode="offline-ir",
        dataset_source=dataset_source,
        total_queries=len(queries),
        configs=active_configs,
        budget_exceeded=False,
        summaries=summaries,
        records=all_records,
    )


async def run_parent_child_ablation_benchmark(
    queries: Sequence[GoldenQuery],
    *,
    flat_store: QdrantHybridStore,
    parent_child_store: QdrantHybridStore,
    reranker: Reranker,
    strategies: StrategiesConfig,
    fixed_top_k: int = 5,
    dataset_source: str = "data/eval/golden_set.jsonl",
) -> BenchmarkReport:
    """Head-to-head $0 IR ablation comparing flat structure-aware chunks vs Small-to-Big Parent-Child."""
    flat_reranker: Reranker = _BenchmarkRerankerCache(reranker)
    pc_reranker: Reranker = _BenchmarkRerankerCache(reranker)
    plan_specs: list[tuple[str, str, QdrantHybridStore, Reranker, bool]] = [
        ("flat_hybrid_rerank", "hybrid_rerank", flat_store, flat_reranker, False),
        ("flat_adaptive_dynamic_k", "adaptive_dynamic_k", flat_store, flat_reranker, False),
        ("parent_child_hybrid_rerank", "hybrid_rerank", parent_child_store, pc_reranker, True),
        (
            "parent_child_adaptive_dynamic_k",
            "adaptive_dynamic_k",
            parent_child_store,
            pc_reranker,
            True,
        ),
    ]

    all_records: list[QueryEvalRecord] = []
    summaries: list[ConfigBenchmarkSummary] = []

    for label, base_config, target_store, active_reranker, use_pc in plan_specs:
        logger.info(
            "benchmark_parent_child_config_start",
            config=label,
            queries=len(queries),
            parent_child=use_pc,
        )
        cfg_records: list[QueryEvalRecord] = []
        for query in queries:
            try:
                record = await _evaluate_single_offline_ir(
                    query,
                    config_name=base_config,
                    store=target_store,
                    reranker=active_reranker,
                    strategies=strategies,
                    fixed_top_k=fixed_top_k,
                    enable_parent_child=use_pc,
                    record_config_name=label,
                )
            except Exception as exc:
                logger.exception(
                    "benchmark_parent_child_query_error", query_id=query.id, error=str(exc)
                )
                record = QueryEvalRecord(
                    query_id=query.id,
                    stratum=query.stratum,
                    language=query.language,
                    config_name=label,
                    error=str(exc),
                )
            cfg_records.append(record)

        all_records.extend(cfg_records)
        summaries.append(
            aggregate_config_summary(
                config_name=label,
                mode="offline-ir",
                queries=queries,
                records=cfg_records,
            )
        )

    return BenchmarkReport(
        run_id=f"pc_ablation_{uuid.uuid4().hex[:8]}",
        timestamp_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        mode="offline-ir",
        dataset_source=dataset_source,
        total_queries=len(queries),
        configs=[spec[0] for spec in plan_specs],
        budget_exceeded=False,
        summaries=summaries,
        records=all_records,
    )


def _final_answer_to_record(
    query: GoldenQuery,
    config_name: str,
    answer: FinalAnswer,
    *,
    ragas_scores: RagasScores | None = None,
) -> QueryEvalRecord:
    retrieved_doc_ids = [chunk.doc_id for chunk in answer.evidence_chunks if chunk.doc_id]
    ir_scores = (
        compute_ir_scores(retrieved_doc_ids, query.expected_doc_ids)
        if query.expected_doc_ids
        else None
    )
    fact_cov = (
        compute_fact_coverage([chunk.text for chunk in answer.evidence_chunks], query.key_facts)
        if query.key_facts
        else None
    )
    context_tokens = (
        answer.dynamic_k_decision.total_context_tokens
        if answer.dynamic_k_decision is not None
        else _estimate_context_tokens(answer.evidence_chunks)
    )
    citation = answer.citation_report

    return QueryEvalRecord(
        query_id=query.id,
        stratum=query.stratum,
        language=query.language,
        config_name=config_name,
        predicted_query_type=answer.query_type,
        routing_correct=answer.query_type == query.expected_query_type,
        retrieved_doc_ids=retrieved_doc_ids,
        selected_k=len(answer.evidence_chunks),
        context_tokens=context_tokens,
        ir_scores=ir_scores,
        fact_coverage=fact_cov,
        confidence_score=answer.confidence.final_score if answer.confidence is not None else None,
        confidence_level=answer.confidence.level.value if answer.confidence is not None else None,
        fallback_used=answer.fallback_used,
        abstained=answer.abstained,
        rewrite_attempts=answer.rewrite_attempts,
        regeneration_attempts=answer.regeneration_attempts,
        citation_precision=citation.citation_precision if citation is not None else None,
        claim_support_rate=citation.claim_support_rate if citation is not None else None,
        citation_passed=citation.passed if citation is not None else None,
        latency_ms=answer.total_latency_ms,
        total_tokens=answer.usage.total_tokens,
        estimated_cost_usd=answer.usage.estimated_cost_usd,
        cache_hits=answer.usage.cache_hits,
        ragas=ragas_scores,
        answer_preview=answer.answer[:240],
    )


async def run_live_ablation_benchmark(
    queries: Sequence[GoldenQuery],
    *,
    workflow: AdaptiveRAGWorkflow,
    configs: Sequence[str] | None = None,
    fixed_top_k: int = 5,
    max_cost_usd: float = 0.50,
    with_ragas: bool = False,
    ragas_dense_embedder: DenseEmbedder | None = None,
    dataset_source: str = "data/eval/golden_set.jsonl",
) -> BenchmarkReport:
    """Run the full 4-way Agentic ablation benchmark through AdaptiveRAGWorkflow.arun.

    Protected by a hard FinOps ``CostTracker(max_cost_usd=...)`` guardrail; if the budget
    cap is hit mid-run, completed records are preserved and summarized gracefully.
    """
    active_configs = list(configs) if configs else list(LIVE_CONFIG_NAMES)
    shared_tracker = CostTracker(max_cost_usd=max_cost_usd)
    all_records: list[QueryEvalRecord] = []
    summaries: list[ConfigBenchmarkSummary] = []
    budget_exceeded = False

    for config_name in active_configs:
        if budget_exceeded:
            break
        ablation = resolve_live_ablation(config_name, fixed_top_k=fixed_top_k)
        logger.info(
            "benchmark_live_config_start",
            config=config_name,
            queries=len(queries),
            spent_usd=shared_tracker.usage.estimated_cost_usd,
            max_cost_usd=shared_tracker.max_cost_usd,
        )

        cfg_queries: list[GoldenQuery] = []
        cfg_answers: list[FinalAnswer] = []
        cfg_error_records: list[QueryEvalRecord] = []

        for idx, query in enumerate(queries, start=1):
            try:
                answer = await workflow.arun(
                    query.query,
                    ablation=ablation,
                    cost_tracker=shared_tracker,
                    preset_name=config_name,
                    trace_tags=["mode:benchmark", f"stratum:{query.stratum.value}"],
                )
                cfg_queries.append(query)
                cfg_answers.append(answer)
                logger.info(
                    "benchmark_live_query_done",
                    config=config_name,
                    progress=f"{idx}/{len(queries)}",
                    query_id=query.id,
                    query_type=answer.query_type.value,
                    selected_k=len(answer.evidence_chunks),
                    confidence=(
                        round(answer.confidence.final_score, 3)
                        if answer.confidence is not None
                        else None
                    ),
                    fallback=answer.fallback_used,
                    abstained=answer.abstained,
                    latency_ms=answer.total_latency_ms,
                    cost_usd=round(answer.usage.estimated_cost_usd, 6),
                    cumulative_usd=round(shared_tracker.usage.estimated_cost_usd, 6),
                )
            except BudgetExceededError as exc:
                logger.warning(
                    "benchmark_budget_exceeded",
                    config=config_name,
                    query_id=query.id,
                    detail=str(exc),
                )
                budget_exceeded = True
                break
            except Exception as exc:
                logger.exception(
                    "benchmark_live_query_error",
                    config=config_name,
                    query_id=query.id,
                    error=str(exc),
                )
                cfg_error_records.append(
                    QueryEvalRecord(
                        query_id=query.id,
                        stratum=query.stratum,
                        language=query.language,
                        config_name=config_name,
                        error=str(exc),
                    )
                )

        ragas_by_id: dict[str, RagasScores] = {}
        if with_ragas and cfg_answers and not budget_exceeded:
            embedder = ragas_dense_embedder or workflow.runtime.store.dense_embedder
            ragas_by_id = await aevaluate_with_ragas(
                cfg_queries,
                cfg_answers,
                judge_llm=workflow.runtime.models.fast_model,
                dense_embedder=embedder,
                judge_model_name=workflow.runtime.models.fast_model_name,
            )

        cfg_records: list[QueryEvalRecord] = [
            _final_answer_to_record(
                q,
                config_name,
                ans,
                ragas_scores=ragas_by_id.get(q.id),
            )
            for q, ans in zip(cfg_queries, cfg_answers, strict=True)
        ]
        cfg_records.extend(cfg_error_records)
        all_records.extend(cfg_records)

        if cfg_records:
            summaries.append(
                aggregate_config_summary(
                    config_name=config_name,
                    mode="live",
                    queries=queries,
                    records=cfg_records,
                )
            )

    return BenchmarkReport(
        run_id=f"live_{uuid.uuid4().hex[:8]}",
        timestamp_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        mode="live",
        dataset_source=dataset_source,
        total_queries=len(queries),
        configs=active_configs,
        budget_exceeded=budget_exceeded,
        summaries=summaries,
        records=all_records,
    )


def save_benchmark_artifacts(
    report: BenchmarkReport,
    output_dir: Path,
) -> dict[str, Path]:
    """Persist benchmark JSON report, Markdown scorecard, and optional Plotly HTML chart."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}

    json_path = output_dir / "benchmark_report.json"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    artifacts["json"] = json_path

    md_path = output_dir / "benchmark_summary.md"
    md_path.write_text(render_markdown_report(report), encoding="utf-8")
    artifacts["markdown"] = md_path

    chart_html = build_ablation_chart_html(report)
    if chart_html is not None:
        chart_path = output_dir / "benchmark_chart.html"
        chart_path.write_text(chart_html, encoding="utf-8")
        artifacts["chart"] = chart_path

    return artifacts


async def _seed_in_memory_store(
    store: QdrantHybridStore,
    documents: Sequence[LoadedDocument],
    *,
    chunk_config: ChunkingConfig | None = None,
) -> int:
    await store.ensure_collection(recreate=True)
    chunks = chunk_documents(list(documents), config=chunk_config)
    return await store.upsert_chunks(chunks)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m eval.run_benchmark",
        description="Run AdaptiveRAG 4-way ablation benchmark (offline $0 IR or live Agentic + RAGAS).",
    )
    parser.add_argument(
        "--mode",
        choices=("offline-ir", "live"),
        default="offline-ir",
        help="Benchmark mode: 'offline-ir' ($0 LLM cost) or 'live' (full LangGraph ablation).",
    )
    parser.add_argument(
        "--chunk-strategy",
        choices=("flat", "parent_child", "compare"),
        default="flat",
        help=(
            "Chunking strategy for --in-memory indexing: 'flat' (512-tok structure-aware), "
            "'parent_child' (Small-to-Big), or 'compare' (offline-ir head-to-head ablation)."
        ),
    )
    parser.add_argument(
        "--parent-max-tokens",
        type=int,
        default=1000,
        help="Parent passage token cap when --chunk-strategy uses parent_child (default: 1000).",
    )
    parser.add_argument(
        "--child-max-tokens",
        type=int,
        default=256,
        help="Child chunk token cap when --chunk-strategy uses parent_child (default: 256).",
    )
    parser.add_argument(
        "--child-overlap-tokens",
        type=int,
        default=40,
        help="Overlap tokens between sibling child chunks (default: 40).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/eval/golden_set.jsonl"),
        help="Path to golden JSONL dataset (default: data/eval/golden_set.jsonl).",
    )
    parser.add_argument(
        "--external-jsonl",
        action="store_true",
        help="Parse --dataset with flexible field aliases (question/ground_truth/doc_ids).",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        help="Optional HuggingFace dataset identifier (e.g., hotpot_qa, rungalileo/ragbench).",
    )
    parser.add_argument(
        "--hf-subset", type=str, default=None, help="Optional HF dataset subset/config."
    )
    parser.add_argument("--hf-split", type=str, default="validation", help="HF split name.")
    parser.add_argument(
        "--hf-preset",
        choices=("auto", "hotpotqa", "ragbench", "generic"),
        default="auto",
        help="Adapter preset for --hf-dataset.",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/raw"),
        help="Local corpus directory used when --in-memory is enabled.",
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Use an ephemeral in-memory Qdrant store seeded from --corpus-dir or HF documents.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Subset of configs to run (defaults to all 4 configs for the chosen mode).",
    )
    parser.add_argument(
        "--stratum",
        action="append",
        default=None,
        help="Filter to specific stratum(s) (repeatable, e.g., --stratum factual --stratum analytical).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Filter queries by language code (e.g., en or tr).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of queries to evaluate after filtering.",
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="When --limit is set, draw a deterministic proportional slice across all strata & languages.",
    )
    parser.add_argument(
        "--fixed-top-k",
        type=int,
        default=5,
        help="Static top-k value used by baseline configs (default: 5).",
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=0.50,
        help="Hard FinOps USD budget cap for live benchmark runs (default: 0.50).",
    )
    parser.add_argument(
        "--with-ragas",
        action="store_true",
        help="Enable optional RAGAS LLM-as-a-Judge evaluation in live mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/reports"),
        help="Directory where JSON, Markdown, and Plotly HTML reports are saved.",
    )
    parser.add_argument(
        "--langfuse-sync",
        action="store_true",
        help=(
            "Sync golden dataset items and publish ablation + RAGAS results to Langfuse "
            "Datasets & Experiments (requires LANGFUSE_ENABLED=true and API keys)."
        ),
    )
    parser.add_argument(
        "--langfuse-dataset",
        type=str,
        default="adaptiverag-golden-set",
        help="Target Langfuse Dataset name when --langfuse-sync is enabled.",
    )
    parser.add_argument(
        "--langfuse-experiment",
        type=str,
        default=None,
        help="Optional experiment run name prefix in Langfuse (defaults to benchmark run_id).",
    )
    return parser.parse_args(argv)


async def _run_cli(args: argparse.Namespace) -> int:
    settings: Settings = get_settings()
    configure_logging(settings.log_level)
    strategies = load_strategies(settings.strategies_config_path)

    hf_docs: list[LoadedDocument] = []
    if args.hf_dataset:
        preset: Literal["auto", "hotpotqa", "ragbench", "generic"] = args.hf_preset
        queries, hf_docs = load_huggingface_dataset(
            args.hf_dataset,
            subset=args.hf_subset,
            split=args.hf_split,
            limit=args.limit or 50,
            preset=preset,
        )
        dataset_source = f"hf://{args.hf_dataset}/{args.hf_subset or 'default'}:{args.hf_split}"
    elif args.external_jsonl:
        queries = load_external_jsonl(args.dataset)
        dataset_source = str(args.dataset)
    else:
        queries = load_golden_jsonl(args.dataset)
        dataset_source = str(args.dataset)

    queries = filter_queries(
        queries,
        strata=args.stratum,
        language=args.language,
        limit=args.limit,
        stratified=args.stratified,
    )
    if not queries:
        print("No evaluation queries matched the provided filters.", file=sys.stderr)
        return 1

    if args.chunk_strategy == "compare" and (args.mode != "offline-ir" or not args.in_memory):
        print(
            "--chunk-strategy compare requires --mode offline-ir --in-memory.",
            file=sys.stderr,
        )
        return 2

    pc_chunk_cfg = ChunkingConfig(
        parent_child_enabled=True,
        parent_max_tokens=args.parent_max_tokens,
        child_max_tokens=args.child_max_tokens,
        child_overlap_tokens=args.child_overlap_tokens,
    )

    if args.chunk_strategy == "compare":
        seed_docs = hf_docs if hf_docs else load_directory(args.corpus_dir)
        shared_dense = FastEmbedDenseEmbedder(model_name=settings.dense_embedding_model)
        shared_sparse = FastEmbedSparseEmbedder(model_name=settings.sparse_embedding_model)
        flat_store = QdrantHybridStore.from_settings(
            settings,
            collection_name=f"{settings.qdrant_collection}_flat",
            dense_embedder=shared_dense,
            sparse_embedder=shared_sparse,
            in_memory=True,
        )
        pc_store = QdrantHybridStore.from_settings(
            settings,
            collection_name=f"{settings.qdrant_collection}_parent_child",
            dense_embedder=shared_dense,
            sparse_embedder=shared_sparse,
            in_memory=True,
        )
        try:
            flat_chunks = await _seed_in_memory_store(flat_store, seed_docs)
            pc_chunks = await _seed_in_memory_store(pc_store, seed_docs, chunk_config=pc_chunk_cfg)
            logger.info(
                "benchmark_parent_child_compare_seeded",
                docs=len(seed_docs),
                flat_chunks=flat_chunks,
                parent_child_chunks=pc_chunks,
            )
            reranker = FastEmbedCrossEncoderReranker(model_name=settings.reranker_model)
            report = await run_parent_child_ablation_benchmark(
                queries,
                flat_store=flat_store,
                parent_child_store=pc_store,
                reranker=reranker,
                strategies=strategies,
                fixed_top_k=args.fixed_top_k,
                dataset_source=dataset_source,
            )
        finally:
            await flat_store.close()
            await pc_store.close()
    else:
        use_pc = args.chunk_strategy == "parent_child"
        store = QdrantHybridStore.from_settings(settings, in_memory=args.in_memory)
        try:
            if args.in_memory:
                seed_docs = hf_docs if hf_docs else load_directory(args.corpus_dir)
                chunk_count = await _seed_in_memory_store(
                    store,
                    seed_docs,
                    chunk_config=pc_chunk_cfg if use_pc else None,
                )
                logger.info(
                    "benchmark_in_memory_seeded",
                    docs=len(seed_docs),
                    chunks=chunk_count,
                    chunk_strategy=args.chunk_strategy,
                )

            if args.mode == "offline-ir":
                reranker = FastEmbedCrossEncoderReranker(model_name=settings.reranker_model)
                report = await run_offline_ir_benchmark(
                    queries,
                    store=store,
                    reranker=reranker,
                    strategies=strategies,
                    configs=args.configs,
                    fixed_top_k=args.fixed_top_k,
                    enable_parent_child=use_pc,
                    dataset_source=dataset_source,
                )
            else:
                live_settings = (
                    settings.model_copy(update={"enable_parent_child_retrieval": True})
                    if use_pc
                    else settings
                )
                workflow = AdaptiveRAGWorkflow.from_settings(
                    live_settings,
                    strategies=strategies,
                    store=store,
                )
                report = await run_live_ablation_benchmark(
                    queries,
                    workflow=workflow,
                    configs=args.configs,
                    fixed_top_k=args.fixed_top_k,
                    max_cost_usd=args.max_cost_usd,
                    with_ragas=args.with_ragas,
                    dataset_source=dataset_source,
                )
        finally:
            await store.close()

    artifacts = save_benchmark_artifacts(report, args.output_dir)
    markdown = render_markdown_report(report)
    print(markdown)
    print(f"Saved artifacts: {', '.join(f'{k}={v}' for k, v in artifacts.items())}")

    if args.langfuse_sync:
        obs = LangfuseObservability.from_settings(settings)
        lf_client = obs.get_client()
        if lf_client is None:
            print(
                "Langfuse sync skipped: set LANGFUSE_ENABLED=true, LANGFUSE_PUBLIC_KEY, "
                "LANGFUSE_SECRET_KEY, and LANGFUSE_HOST in .env.",
                file=sys.stderr,
            )
            return 2
        summary = publish_benchmark_report_to_langfuse(
            lf_client,
            report,
            queries,
            dataset_name=args.langfuse_dataset,
            experiment_prefix=args.langfuse_experiment,
        )
        run_list = ", ".join(f"{r.config_name}({r.items_evaluated})" for r in summary.runs)
        print(
            f"Published to Langfuse dataset '{summary.dataset_name}' "
            f"(items={summary.items_synced}, prefix={summary.experiment_prefix}): {run_list}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_run_cli(args))


if __name__ == "__main__":
    raise SystemExit(main())
