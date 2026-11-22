"""Multilingual cross-encoder reranking with numerically stable sigmoid calibration."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from threading import Lock
from typing import Protocol, runtime_checkable

import structlog
from fastembed.rerank.cross_encoder import TextCrossEncoder

from adaptiverag.schemas.retrieval import EvidenceChunk

logger = structlog.get_logger(__name__)


def sigmoid(logit: float, *, temperature: float = 1.0) -> float:
    """Convert an unbounded cross-encoder logit into a calibrated probability in [0.0, 1.0]."""
    if temperature <= 0.0:
        raise ValueError(f"Sigmoid temperature must be > 0, got {temperature}")
    scaled = logit / temperature
    if scaled >= 0.0:
        prob = 1.0 / (1.0 + math.exp(-scaled))
    else:
        exp_x = math.exp(scaled)
        prob = exp_x / (1.0 + exp_x)
    return min(1.0, max(0.0, prob))


def format_passage_for_rerank(chunk: EvidenceChunk) -> str:
    """Attach title and section breadcrumb so the cross-encoder sees structural context."""
    header_parts = [
        part.strip() for part in (chunk.title, chunk.section_path) if part and part.strip()
    ]
    if not header_parts:
        return chunk.text
    prefix = f"[{' | '.join(header_parts)}]"
    if chunk.text.lstrip().startswith(prefix):
        return chunk.text
    return f"{prefix}\n{chunk.text}"


@runtime_checkable
class Reranker(Protocol):
    """Protocol for cross-encoder rerankers (FastEmbed ONNX, Cohere, Jina, etc.)."""

    @property
    def model_name(self) -> str: ...

    def rerank(
        self,
        query: str,
        chunks: Sequence[EvidenceChunk],
        *,
        top_n: int | None = None,
    ) -> list[EvidenceChunk]: ...

    async def arerank(
        self,
        query: str,
        chunks: Sequence[EvidenceChunk],
        *,
        top_n: int | None = None,
    ) -> list[EvidenceChunk]: ...


class FastEmbedCrossEncoderReranker:
    """Thread-safe lazy FastEmbed ONNX multilingual cross-encoder reranker ($0 API cost).

    Defaults to ``BAAI/bge-reranker-v2-m3-int8``, which supports both Turkish and English
    queries/passages cleanly on CPU. Raw ONNX logits are stored in ``rerank_raw_score`` and
    mapped into ``[0.0, 1.0]`` via ``sigmoid`` for downstream Dynamic Top-K and Confidence
    Gating thresholds.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3-int8",
        *,
        cache_dir: str | None = None,
        threads: int | None = None,
        batch_size: int = 32,
        temperature: float = 1.0,
    ) -> None:
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._threads = threads
        self._batch_size = batch_size
        self._temperature = temperature
        self._model: TextCrossEncoder | None = None
        self._lock = Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def _get_model(self) -> TextCrossEncoder:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info("fastembed_reranker_loading", model=self._model_name)
                    self._model = TextCrossEncoder(
                        model_name=self._model_name,
                        cache_dir=self._cache_dir,
                        threads=self._threads,
                    )
        return self._model

    def rerank(
        self,
        query: str,
        chunks: Sequence[EvidenceChunk],
        *,
        top_n: int | None = None,
    ) -> list[EvidenceChunk]:
        """Score, calibrate, and sort candidate chunks descending by cross-encoder relevance."""
        if not chunks:
            return []
        if top_n is not None and top_n <= 0:
            return []

        cleaned_query = query.strip()
        if not cleaned_query:
            return list(chunks[:top_n]) if top_n is not None else list(chunks)

        model = self._get_model()
        passages = [format_passage_for_rerank(chunk) for chunk in chunks]
        raw_logits = [
            float(score)
            for score in model.rerank(cleaned_query, passages, batch_size=self._batch_size)
        ]

        scored: list[EvidenceChunk] = []
        for chunk, raw_logit in zip(chunks, raw_logits, strict=True):
            calibrated = sigmoid(raw_logit, temperature=self._temperature)
            scored.append(
                chunk.model_copy(
                    update={
                        "rerank_raw_score": round(raw_logit, 6),
                        "rerank_score": round(calibrated, 6),
                    }
                )
            )

        scored.sort(
            key=lambda c: (
                c.rerank_score if c.rerank_score is not None else 0.0,
                c.rerank_raw_score if c.rerank_raw_score is not None else float("-inf"),
                c.fusion_score if c.fusion_score is not None else 0.0,
            ),
            reverse=True,
        )

        return scored[:top_n] if top_n is not None else scored

    async def arerank(
        self,
        query: str,
        chunks: Sequence[EvidenceChunk],
        *,
        top_n: int | None = None,
    ) -> list[EvidenceChunk]:
        """Async non-blocking wrapper offloading ONNX cross-encoder inference to a worker thread."""
        if not chunks or (top_n is not None and top_n <= 0):
            return []
        return await asyncio.to_thread(self.rerank, query, chunks, top_n=top_n)
