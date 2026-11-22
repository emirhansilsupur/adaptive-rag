"""Optional RAGAS (v0.4+) LLM-as-a-Judge evaluation wrapper using local $0 embeddings."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

import structlog
from langchain_core.language_models import BaseChatModel

from adaptiverag.llm.cost import estimate_cost_usd
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
    """Build RAGAS SingleTurnSample objects for corpus-grounded retrieval queries.

    Skips conversational, web-fallback, abstention-expected, and empty-context rows so
    RAGAS ``ContextPrecision`` / ``ContextRecall`` / ``Faithfulness`` comparisons remain
    apples-to-apples across all ablation configurations (never scoring meta-references
    against abstained or out-of-corpus queries).
    """
    from ragas import SingleTurnSample

    included_ids: list[str] = []
    samples: list[Any] = []

    for golden, final_answer in zip(queries, answers, strict=True):
        if not golden.needs_retrieval or golden.expect_web_fallback or golden.expect_abstention:
            continue
        if not final_answer.evidence_chunks or not final_answer.answer.strip():
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


def _build_ragas_components(
    samples: Sequence[Any],
    *,
    judge_llm: BaseChatModel,
    dense_embedder: DenseEmbedder,
    relevancy_strictness: int = 2,
) -> tuple[Any, list[Any], Any, Any]:
    """Assemble RAGAS EvaluationDataset, metrics, OpenRouter-safe LLM wrapper, and $0 embeddings."""
    from ragas import EvaluationDataset
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
    metrics: list[Any] = [
        Faithfulness(),
        ResponseRelevancy(strictness=max(1, relevancy_strictness)),
    ]
    if has_reference:
        metrics.extend([LLMContextPrecisionWithReference(), LLMContextRecall()])

    # bypass_n=True + bypass_temperature=True ensures OpenRouter compatibility (avoids n>1
    # rejections) and prevents concurrent in-place attribute mutation on shared ChatOpenAI.
    ragas_llm = LangchainLLMWrapper(
        judge_llm,
        bypass_temperature=True,
        bypass_n=True,
    )
    ragas_embeddings = _RagasEmbedBridge(FastEmbedRagasEmbeddings(dense_embedder))
    dataset = EvaluationDataset(samples=list(samples))
    return dataset, metrics, ragas_llm, ragas_embeddings


def _parse_ragas_result(
    result: Any,
    included_ids: Sequence[str],
    *,
    judge_model_name: str | None = None,
) -> dict[str, RagasScores]:
    try:
        token_usage = result.total_tokens()
        if isinstance(token_usage, list):
            in_tok = sum(getattr(t, "input_tokens", 0) for t in token_usage)
            out_tok = sum(getattr(t, "output_tokens", 0) for t in token_usage)
        else:
            in_tok = int(getattr(token_usage, "input_tokens", 0))
            out_tok = int(getattr(token_usage, "output_tokens", 0))
        est_usd = estimate_cost_usd(judge_model_name, in_tok, out_tok) if judge_model_name else 0.0
        logger.info(
            "ragas_evaluation_complete",
            samples=len(included_ids),
            judge_model=judge_model_name or "unknown",
            input_tokens=in_tok,
            output_tokens=out_tok,
            estimated_judge_cost_usd=round(est_usd, 6),
        )
    except Exception:  # pragma: no cover - token callback best-effort logging
        logger.debug("ragas_token_usage_unavailable")

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


async def aevaluate_with_ragas(
    queries: Sequence[GoldenQuery],
    answers: Sequence[FinalAnswer],
    *,
    judge_llm: BaseChatModel,
    dense_embedder: DenseEmbedder,
    judge_model_name: str | None = None,
    batch_size: int = 8,
    max_workers: int = 4,
    relevancy_strictness: int = 2,
) -> dict[str, RagasScores]:
    """Async RAGAS Faithfulness, AnswerRelevancy, ContextPrecision, and ContextRecall pass.

    Uses ``ragas.aevaluate`` directly on the active event loop (no nested-loop conflicts)
    with ``bypass_n=True`` for full OpenRouter compatibility and local ``$0`` FastEmbed
    embeddings.
    """
    included_ids, samples = build_ragas_samples(queries, answers)
    if not samples:
        logger.info("ragas_skipped_no_eligible_samples")
        return {}

    from ragas import RunConfig, aevaluate
    from ragas.cost import get_token_usage_for_openai

    dataset, metrics, ragas_llm, ragas_embeddings = _build_ragas_components(
        samples,
        judge_llm=judge_llm,
        dense_embedder=dense_embedder,
        relevancy_strictness=relevancy_strictness,
    )

    logger.info(
        "ragas_evaluation_starting",
        sample_count=len(samples),
        metric_names=[m.name for m in metrics],
    )
    result: Any = await cast(
        Any,
        aevaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=RunConfig(max_workers=max_workers, timeout=120, max_retries=2),
            token_usage_parser=get_token_usage_for_openai,
            batch_size=batch_size,
            raise_exceptions=False,
            show_progress=False,
        ),
    )
    return _parse_ragas_result(result, included_ids, judge_model_name=judge_model_name)


def evaluate_with_ragas(
    queries: Sequence[GoldenQuery],
    answers: Sequence[FinalAnswer],
    *,
    judge_llm: BaseChatModel,
    dense_embedder: DenseEmbedder,
    judge_model_name: str | None = None,
    batch_size: int = 8,
    max_workers: int = 4,
    relevancy_strictness: int = 2,
) -> dict[str, RagasScores]:
    """Synchronous RAGAS Faithfulness, AnswerRelevancy, ContextPrecision, and ContextRecall."""
    included_ids, samples = build_ragas_samples(queries, answers)
    if not samples:
        logger.info("ragas_skipped_no_eligible_samples")
        return {}

    from ragas import RunConfig, evaluate
    from ragas.cost import get_token_usage_for_openai

    dataset, metrics, ragas_llm, ragas_embeddings = _build_ragas_components(
        samples,
        judge_llm=judge_llm,
        dense_embedder=dense_embedder,
        relevancy_strictness=relevancy_strictness,
    )

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
            run_config=RunConfig(max_workers=max_workers, timeout=120, max_retries=2),
            token_usage_parser=get_token_usage_for_openai,
            batch_size=batch_size,
            raise_exceptions=False,
            show_progress=False,
        ),
    )
    return _parse_ragas_result(result, included_ids, judge_model_name=judge_model_name)
