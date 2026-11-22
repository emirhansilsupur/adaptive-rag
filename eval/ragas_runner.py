"""Optional RAGAS (v0.4+) LLM-as-a-Judge evaluation wrapper using local $0 embeddings."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

import structlog
from langchain_core.language_models import BaseChatModel

from adaptiverag.retrieval.embeddings import DenseEmbedder
from adaptiverag.schemas.response import FinalAnswer
from eval.schemas import GoldenQuery, RagasScores

logger = structlog.get_logger(__name__)


class FastEmbedRagasEmbeddings:
    """Adapter exposing AdaptiveRAG's local ONNX DenseEmbedder to RAGAS metrics ($0 cost).

    Implements the ``BaseRagasEmbeddings`` interface (``embed_documents``, ``embed_query``,
    ``aembed_documents``, ``aembed_query``) so ``ResponseRelevancy`` runs without paid
    OpenAI embedding calls.
    """

    def __init__(self, embedder: DenseEmbedder) -> None:
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embedder.aembed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await self._embedder.aembed_query(text)


def _safe_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return round(min(1.0, max(0.0, numeric)), 6)


def build_ragas_samples(
    queries: Sequence[GoldenQuery],
    answers: Sequence[FinalAnswer],
) -> tuple[list[str], list[Any]]:
    """Build RAGAS SingleTurnSample objects for retrieval-backed queries with references.

    Returns ``(included_query_ids, samples)`` so callers can map RAGAS outputs back onto
    per-query ``QueryEvalRecord`` rows even when conversational or empty-context rows are
    skipped.
    """
    from ragas import SingleTurnSample

    included_ids: list[str] = []
    samples: list[Any] = []

    for golden, final_answer in zip(queries, answers, strict=True):
        if not golden.needs_retrieval or not final_answer.evidence_chunks:
            continue
        if not final_answer.answer.strip():
            continue

        contexts = [chunk.text for chunk in final_answer.evidence_chunks if chunk.text.strip()]
        if not contexts:
            continue

        sample = SingleTurnSample(
            user_input=golden.query,
            retrieved_contexts=contexts,
            response=final_answer.answer,
            reference=golden.reference_answer or None,
        )
        included_ids.append(golden.id)
        samples.append(sample)

    return included_ids, samples


def evaluate_with_ragas(
    queries: Sequence[GoldenQuery],
    answers: Sequence[FinalAnswer],
    *,
    judge_llm: BaseChatModel,
    dense_embedder: DenseEmbedder,
    batch_size: int = 8,
    max_workers: int = 4,
) -> dict[str, RagasScores]:
    """Run RAGAS Faithfulness, AnswerRelevancy, ContextPrecision, and ContextRecall.

    Returns a mapping ``{query_id: RagasScores}``. Individual metric failures are caught
    gracefully (``raise_exceptions=False``) so a single malformed judge reply never aborts
    the benchmark run.
    """
    included_ids, samples = build_ragas_samples(queries, answers)
    if not samples:
        logger.info("ragas_skipped_no_eligible_samples")
        return {}

    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.embeddings import BaseRagasEmbeddings
    from ragas.llms.base import LangchainLLMWrapper
    from ragas.metrics._answer_relevance import ResponseRelevancy
    from ragas.metrics._context_precision import LLMContextPrecisionWithReference
    from ragas.metrics._context_recall import LLMContextRecall
    from ragas.metrics._faithfulness import Faithfulness

    class _RagasEmbedBridge(BaseRagasEmbeddings):
        def __init__(self, inner: FastEmbedRagasEmbeddings) -> None:
            super().__init__()
            self._inner = inner

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._inner.embed_documents(texts)

        def embed_query(self, text: str) -> list[float]:
            return self._inner.embed_query(text)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return await self._inner.aembed_documents(texts)

        async def aembed_query(self, text: str) -> list[float]:
            return await self._inner.aembed_query(text)

    has_reference = all(bool(s.reference) for s in samples)
    metrics: list[Any] = [Faithfulness(), ResponseRelevancy()]
    if has_reference:
        metrics.extend([LLMContextPrecisionWithReference(), LLMContextRecall()])

    ragas_llm = LangchainLLMWrapper(judge_llm)
    ragas_embeddings = _RagasEmbedBridge(FastEmbedRagasEmbeddings(dense_embedder))
    dataset = EvaluationDataset(samples=samples)

    logger.info(
        "ragas_evaluation_starting",
        sample_count=len(samples),
        metric_names=[m.name for m in metrics],
    )
    result: Any = cast(
        Any,
        evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=RunConfig(max_workers=max_workers, timeout=90),
            batch_size=batch_size,
            raise_exceptions=False,
            show_progress=False,
        ),
    )

    df = result.to_pandas()
    scores_by_id: dict[str, RagasScores] = {}
    for idx, qid in enumerate(included_ids):
        if idx >= len(df):
            break
        row = df.iloc[idx]
        scores_by_id[qid] = RagasScores(
            faithfulness=_safe_score(row.get("faithfulness")),
            answer_relevancy=_safe_score(row.get("answer_relevancy")),
            context_precision=_safe_score(
                row.get("llm_context_precision_with_reference", row.get("context_precision"))
            ),
            context_recall=_safe_score(row.get("context_recall")),
        )

    return scores_by_id
