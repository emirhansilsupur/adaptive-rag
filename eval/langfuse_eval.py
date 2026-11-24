"""Langfuse v5 Datasets & Experiments synchronization for AdaptiveRAG benchmarks.

Synchronizes ``GoldenQuery`` evaluation sets (e.g. ``data/eval/golden_set.jsonl``)
idempotently into a Langfuse Dataset and publishes multi-config 4-way ablation + RAGAS
``BenchmarkReport`` runs as comparable Langfuse Experiments at ``$0`` extra LLM cost.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid5

import structlog

from eval.schemas import (
    BenchmarkReport,
    ConfigBenchmarkSummary,
    GoldenQuery,
    QueryEvalRecord,
)

logger = structlog.get_logger(__name__)

_DATASET_ITEM_NAMESPACE = UUID("6f3c8e21-8b4d-4f6c-9d2a-01a2b3c4d5e6")


@dataclass(frozen=True)
class DatasetSyncSummary:
    """Summary of an idempotent GoldenQuery -> Langfuse Dataset synchronization."""

    dataset_name: str
    items_synced: int
    item_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExperimentRunSummary:
    """Summary of one ablation configuration published as a Langfuse Experiment."""

    config_name: str
    experiment_name: str
    run_name: str
    items_evaluated: int
    dataset_run_url: str | None = None


@dataclass(frozen=True)
class LangfuseBenchmarkPublishSummary:
    """Complete result of publishing a BenchmarkReport to Langfuse."""

    dataset_name: str
    items_synced: int
    experiment_prefix: str
    runs: list[ExperimentRunSummary] = field(default_factory=list)


def deterministic_dataset_item_id(dataset_name: str, query_id: str) -> str:
    """Return a deterministic UUIDv5 item ID so re-syncing golden_set.jsonl is idempotent."""
    return str(uuid5(_DATASET_ITEM_NAMESPACE, f"{dataset_name.strip()}::{query_id.strip()}"))


def _make_evaluation(
    *,
    name: str,
    value: float | str | bool,
    data_type: str = "NUMERIC",
    comment: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Construct a Langfuse v5 Evaluation instance (falling back to dict if SDK absent)."""
    try:
        from langfuse.experiment import Evaluation

        return Evaluation(
            name=name,
            value=value,
            data_type=data_type,  # type: ignore[arg-type]
            comment=comment,
            metadata=metadata,
        )
    except ImportError:
        payload: dict[str, Any] = {"name": name, "value": value, "data_type": data_type}
        if comment is not None:
            payload["comment"] = comment
        if metadata is not None:
            payload["metadata"] = metadata
        return payload


def build_record_evaluations(record: QueryEvalRecord) -> list[Any]:
    """Convert one per-query ``QueryEvalRecord`` into Langfuse item-level Evaluation scores."""
    evals: list[Any] = []

    if record.error:
        evals.append(
            _make_evaluation(
                name="execution_ok",
                value=False,
                data_type="BOOLEAN",
                comment=record.error[:240],
            )
        )
        return evals

    evals.append(_make_evaluation(name="execution_ok", value=True, data_type="BOOLEAN"))

    if record.ir_scores is not None:
        ir = record.ir_scores
        evals.extend(
            [
                _make_evaluation(name="ir_mrr", value=round(ir.mrr_at_k, 6)),
                _make_evaluation(name="ir_recall_at_k", value=round(ir.recall_at_k, 6)),
                _make_evaluation(name="ir_precision_at_k", value=round(ir.precision_at_k, 6)),
                _make_evaluation(name="ir_ndcg_at_k", value=round(ir.ndcg_at_k, 6)),
                _make_evaluation(name="ir_hit_at_k", value=bool(ir.hit_at_k), data_type="BOOLEAN"),
            ]
        )

    if record.fact_coverage is not None:
        evals.append(_make_evaluation(name="fact_coverage", value=round(record.fact_coverage, 6)))

    if record.routing_correct is not None:
        evals.append(
            _make_evaluation(
                name="routing_correct",
                value=bool(record.routing_correct),
                data_type="BOOLEAN",
                comment=(
                    f"predicted={record.predicted_query_type.value}"
                    if record.predicted_query_type is not None
                    else None
                ),
            )
        )

    if record.confidence_score is not None:
        evals.append(
            _make_evaluation(
                name="crag_confidence",
                value=round(record.confidence_score, 6),
                comment=f"level={record.confidence_level}" if record.confidence_level else None,
            )
        )

    if record.citation_precision is not None:
        evals.append(
            _make_evaluation(
                name="citation_precision",
                value=round(record.citation_precision, 6),
            )
        )
    if record.claim_support_rate is not None:
        evals.append(
            _make_evaluation(
                name="claim_support_rate",
                value=round(record.claim_support_rate, 6),
            )
        )
    if record.citation_passed is not None:
        evals.append(
            _make_evaluation(
                name="citation_passed",
                value=bool(record.citation_passed),
                data_type="BOOLEAN",
            )
        )

    if record.ragas is not None:
        ragas = record.ragas
        if ragas.faithfulness is not None:
            evals.append(
                _make_evaluation(name="ragas_faithfulness", value=round(ragas.faithfulness, 6))
            )
        if ragas.answer_relevancy is not None:
            evals.append(
                _make_evaluation(
                    name="ragas_answer_relevancy",
                    value=round(ragas.answer_relevancy, 6),
                )
            )
        if ragas.context_precision is not None:
            evals.append(
                _make_evaluation(
                    name="ragas_context_precision",
                    value=round(ragas.context_precision, 6),
                )
            )
        if ragas.context_recall is not None:
            evals.append(
                _make_evaluation(name="ragas_context_recall", value=round(ragas.context_recall, 6))
            )

    evals.extend(
        [
            _make_evaluation(name="selected_k", value=float(record.selected_k)),
            _make_evaluation(name="context_tokens", value=float(record.context_tokens)),
            _make_evaluation(
                name="fallback_used", value=bool(record.fallback_used), data_type="BOOLEAN"
            ),
            _make_evaluation(name="abstained", value=bool(record.abstained), data_type="BOOLEAN"),
            _make_evaluation(name="latency_ms", value=round(record.latency_ms, 2)),
            _make_evaluation(name="estimated_cost_usd", value=round(record.estimated_cost_usd, 8)),
        ]
    )
    return evals


def build_summary_evaluations(summary: ConfigBenchmarkSummary) -> list[Any]:
    """Convert one ``ConfigBenchmarkSummary`` into Langfuse run-level aggregate evaluations."""
    pairs: list[tuple[str, float | None]] = [
        ("avg_mrr", summary.mean_mrr),
        ("avg_recall_at_k", summary.mean_recall),
        ("avg_precision_at_k", summary.mean_precision),
        ("avg_ndcg_at_k", summary.mean_ndcg),
        ("hit_rate", summary.hit_rate),
        ("avg_fact_coverage", summary.mean_fact_coverage),
        ("routing_accuracy", summary.routing_accuracy),
        ("avg_citation_precision", summary.mean_citation_precision),
        ("avg_claim_support_rate", summary.mean_claim_support_rate),
        ("citation_pass_rate", summary.citation_pass_rate),
        ("avg_selected_k", summary.avg_selected_k),
        ("avg_context_tokens", summary.avg_context_tokens),
        ("mean_latency_ms", summary.mean_latency_ms),
        ("p95_latency_ms", summary.latency_p95_ms),
        ("total_cost_usd", summary.total_cost_usd),
        ("ragas_faithfulness", summary.ragas_faithfulness),
        ("ragas_answer_relevancy", summary.ragas_answer_relevancy),
        ("ragas_context_precision", summary.ragas_context_precision),
        ("ragas_context_recall", summary.ragas_context_recall),
    ]
    evals: list[Any] = []
    for name, value in pairs:
        if value is not None:
            evals.append(_make_evaluation(name=name, value=round(float(value), 6)))
    return evals


def sync_golden_dataset(
    client: Any,
    queries: Sequence[GoldenQuery],
    *,
    dataset_name: str = "adaptiverag-golden-set",
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> DatasetSyncSummary:
    """Idempotently create/update a Langfuse Dataset from ``GoldenQuery`` items."""
    dataset_desc = (
        description
        or "AdaptiveRAG bilingual (TR + EN) stratified golden evaluation set covering "
        "factual, analytical, conversational, web-fallback, and unanswerable queries."
    )
    dataset_meta = {
        "framework": "adaptiverag",
        "query_count": len(queries),
        "strata": sorted({q.stratum.value for q in queries}),
        "languages": sorted({q.language for q in queries}),
        **(metadata or {}),
    }

    try:
        client.create_dataset(
            name=dataset_name,
            description=dataset_desc,
            metadata=dataset_meta,
        )
    except Exception as exc:
        # Langfuse create_dataset is normally idempotent; tolerate conflict errors gracefully.
        logger.debug("langfuse_create_dataset_notice", dataset=dataset_name, detail=str(exc))

    synced_ids: list[str] = []
    for q in queries:
        item_id = deterministic_dataset_item_id(dataset_name, q.id)
        client.create_dataset_item(
            dataset_name=dataset_name,
            id=item_id,
            input={
                "query_id": q.id,
                "query": q.query,
                "language": q.language,
                "stratum": q.stratum.value,
            },
            expected_output={
                "reference_answer": q.reference_answer,
                "expected_query_type": q.expected_query_type.value,
                "expected_doc_ids": list(q.expected_doc_ids),
                "key_facts": list(q.key_facts),
                "needs_retrieval": q.needs_retrieval,
                "expect_web_fallback": q.expect_web_fallback,
                "expect_abstention": q.expect_abstention,
            },
            metadata={
                "query_id": q.id,
                "stratum": q.stratum.value,
                "language": q.language,
                "expected_query_type": q.expected_query_type.value,
            },
        )
        synced_ids.append(item_id)

    logger.info(
        "langfuse_dataset_synced",
        dataset=dataset_name,
        items_synced=len(synced_ids),
    )
    return DatasetSyncSummary(
        dataset_name=dataset_name,
        items_synced=len(synced_ids),
        item_ids=synced_ids,
    )


def _extract_item_query_id(item: Any) -> str:
    if isinstance(item, dict):
        inp = item.get("input") or {}
        meta = item.get("metadata") or {}
    else:
        inp = getattr(item, "input", None) or {}
        meta = getattr(item, "metadata", None) or {}

    if isinstance(inp, dict) and inp.get("query_id"):
        return str(inp["query_id"])
    if isinstance(meta, dict) and meta.get("query_id"):
        return str(meta["query_id"])
    return ""


def _record_to_experiment_output(record: QueryEvalRecord | None) -> dict[str, Any]:
    if record is None:
        return {"skipped": True}
    return {
        "query_id": record.query_id,
        "config_name": record.config_name,
        "predicted_query_type": (
            record.predicted_query_type.value if record.predicted_query_type else None
        ),
        "retrieved_doc_ids": list(record.retrieved_doc_ids),
        "selected_k": record.selected_k,
        "context_tokens": record.context_tokens,
        "confidence_score": record.confidence_score,
        "confidence_level": record.confidence_level,
        "fallback_used": record.fallback_used,
        "abstained": record.abstained,
        "citation_precision": record.citation_precision,
        "claim_support_rate": record.claim_support_rate,
        "citation_passed": record.citation_passed,
        "latency_ms": record.latency_ms,
        "estimated_cost_usd": record.estimated_cost_usd,
        "answer_preview": record.answer_preview,
        "error": record.error,
    }


def publish_benchmark_report_to_langfuse(
    client: Any,
    report: BenchmarkReport,
    queries: Sequence[GoldenQuery],
    *,
    dataset_name: str = "adaptiverag-golden-set",
    experiment_prefix: str | None = None,
) -> LangfuseBenchmarkPublishSummary:
    """Sync the benchmark queries to Langfuse and publish each ablation config as an Experiment."""
    prefix = (experiment_prefix or report.run_id).strip()
    sync_summary = sync_golden_dataset(
        client,
        queries,
        dataset_name=dataset_name,
        metadata={"dataset_source": report.dataset_source},
    )

    dataset = client.get_dataset(dataset_name)
    all_dataset_items = list(getattr(dataset, "items", []) or [])
    evaluated_query_ids = {q.id for q in queries}
    filtered_items = [
        item for item in all_dataset_items if _extract_item_query_id(item) in evaluated_query_ids
    ]

    summary_by_config: dict[str, ConfigBenchmarkSummary] = {
        s.config_name: s for s in report.summaries
    }
    runs: list[ExperimentRunSummary] = []

    for config_name in report.configs:
        records_by_qid: dict[str, QueryEvalRecord] = {
            r.query_id: r for r in report.records if r.config_name == config_name
        }
        cfg_summary = summary_by_config.get(config_name)
        experiment_name = f"{prefix}-{config_name}"
        run_name = f"{prefix}::{config_name}"
        run_metadata = {
            "benchmark_run_id": report.run_id,
            "mode": report.mode,
            "config_name": config_name,
            "dataset_source": report.dataset_source,
            "timestamp_utc": report.timestamp_utc,
            "budget_exceeded": str(report.budget_exceeded).lower(),
        }

        def _make_task(
            cfg_records: dict[str, QueryEvalRecord],
        ) -> Any:
            def _task(*, item: Any, **kwargs: Any) -> dict[str, Any]:
                qid = _extract_item_query_id(item)
                return _record_to_experiment_output(cfg_records.get(qid))

            return _task

        def _make_item_evaluator(
            cfg_records: dict[str, QueryEvalRecord],
        ) -> Any:
            def _evaluator(
                *,
                input: Any,
                output: Any,
                expected_output: Any,
                metadata: dict[str, Any] | None,
                **kwargs: Any,
            ) -> list[Any]:
                qid = ""
                if isinstance(input, dict) and input.get("query_id"):
                    qid = str(input["query_id"])
                elif isinstance(output, dict) and output.get("query_id"):
                    qid = str(output["query_id"])
                elif isinstance(metadata, dict) and metadata.get("query_id"):
                    qid = str(metadata["query_id"])
                record = cfg_records.get(qid)
                if record is None:
                    return []
                return build_record_evaluations(record)

            return _evaluator

        def _make_run_evaluator(
            summary_obj: ConfigBenchmarkSummary | None,
        ) -> Any:
            def _run_evaluator(*, item_results: Any, **kwargs: Any) -> list[Any]:
                if summary_obj is None:
                    return []
                return build_summary_evaluations(summary_obj)

            return _run_evaluator

        task_fn = _make_task(records_by_qid)
        item_eval_fn = _make_item_evaluator(records_by_qid)
        run_eval_fns = [_make_run_evaluator(cfg_summary)] if cfg_summary is not None else []

        # If every item in the fetched dataset belongs to this slice, run directly via
        # DatasetClient.run_experiment; otherwise pass the filtered subset to client.run_experiment.
        if filtered_items and len(filtered_items) == len(all_dataset_items):
            exp_result = dataset.run_experiment(
                name=experiment_name,
                run_name=run_name,
                description=f"AdaptiveRAG {report.mode} ablation ({config_name})",
                task=task_fn,
                evaluators=[item_eval_fn],
                run_evaluators=run_eval_fns,
                max_concurrency=8,
                metadata=run_metadata,
            )
            items_count = len(all_dataset_items)
        else:
            experiment_data: list[Any] = filtered_items or [
                {
                    "input": {
                        "query_id": q.id,
                        "query": q.query,
                        "language": q.language,
                        "stratum": q.stratum.value,
                    },
                    "expected_output": {
                        "reference_answer": q.reference_answer,
                        "expected_query_type": q.expected_query_type.value,
                        "expected_doc_ids": list(q.expected_doc_ids),
                    },
                    "metadata": {"query_id": q.id, "stratum": q.stratum.value},
                }
                for q in queries
            ]
            exp_result = client.run_experiment(
                name=experiment_name,
                run_name=run_name,
                description=f"AdaptiveRAG {report.mode} ablation ({config_name})",
                data=experiment_data,
                task=task_fn,
                evaluators=[item_eval_fn],
                run_evaluators=run_eval_fns,
                max_concurrency=8,
                metadata=run_metadata,
            )
            items_count = len(experiment_data)

        run_url = getattr(exp_result, "dataset_run_url", None) or getattr(
            exp_result, "experiment_url", None
        )
        runs.append(
            ExperimentRunSummary(
                config_name=config_name,
                experiment_name=experiment_name,
                run_name=run_name,
                items_evaluated=items_count,
                dataset_run_url=str(run_url) if run_url else None,
            )
        )

    flush_fn = getattr(client, "flush", None)
    if callable(flush_fn):
        flush_fn()

    logger.info(
        "langfuse_benchmark_published",
        dataset=dataset_name,
        experiment_prefix=prefix,
        runs=len(runs),
    )
    return LangfuseBenchmarkPublishSummary(
        dataset_name=dataset_name,
        items_synced=sync_summary.items_synced,
        experiment_prefix=prefix,
        runs=runs,
    )
