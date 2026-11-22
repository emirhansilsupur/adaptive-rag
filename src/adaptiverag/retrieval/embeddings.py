"""Local multilingual Dense + Sparse (BM25) embedding wrappers powered by FastEmbed ONNX."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Protocol, runtime_checkable

import structlog
from fastembed import SparseTextEmbedding, TextEmbedding

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class SparseVectorData:
    """Backend-agnostic sparse vector representation (token hash indices + weights)."""

    indices: list[int]
    values: list[float]

    @property
    def is_empty(self) -> bool:
        return len(self.indices) == 0


@runtime_checkable
class DenseEmbedder(Protocol):
    """Protocol for dense text embedding backends."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...

    async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def aembed_query(self, query: str) -> list[float]: ...


@runtime_checkable
class SparseEmbedder(Protocol):
    """Protocol for sparse lexical (BM25 / SPLADE) embedding backends."""

    @property
    def model_name(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVectorData]: ...

    def embed_query(self, query: str) -> SparseVectorData: ...

    async def aembed_documents(self, texts: Sequence[str]) -> list[SparseVectorData]: ...

    async def aembed_query(self, query: str) -> SparseVectorData: ...


def _lookup_fastembed_dimension(model_name: str) -> int | None:
    """Resolve embedding dimension from FastEmbed's model registry without downloading weights."""
    normalized = model_name.strip().lower()
    for entry in TextEmbedding.list_supported_models():
        candidate = str(entry.get("model", "")).strip().lower()
        if candidate == normalized:
            dim = entry.get("dim")
            if isinstance(dim, int) and dim > 0:
                return dim
    return None


class FastEmbedDenseEmbedder:
    """Thread-safe lazy FastEmbed ONNX dense embedder supporting TR+EN multilingual models.

    Uses ``passage_embed`` / ``query_embed`` so asymmetric models (such as
    ``intfloat/multilingual-e5-large``) automatically receive their required prefixes.
    All async methods offload ONNX CPU inference via ``asyncio.to_thread`` to keep the
    FastAPI / LangGraph event loop responsive.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        *,
        cache_dir: str | None = None,
        threads: int | None = None,
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._threads = threads
        self._batch_size = batch_size
        self._model: TextEmbedding | None = None
        self._dimension: int | None = _lookup_fastembed_dimension(model_name)
        self._lock = Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            probe = self.embed_query("dimension probe")
            self._dimension = len(probe)
        return self._dimension

    def _get_model(self) -> TextEmbedding:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info("fastembed_dense_loading", model=self._model_name)
                    self._model = TextEmbedding(
                        model_name=self._model_name,
                        cache_dir=self._cache_dir,
                        threads=self._threads,
                    )
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        vectors: list[list[float]] = [
            [float(x) for x in vec.tolist()]
            for vec in model.passage_embed(list(texts), batch_size=self._batch_size)
        ]
        if vectors and self._dimension is None:
            self._dimension = len(vectors[0])
        return vectors

    def embed_query(self, query: str) -> list[float]:
        model = self._get_model()
        vec: list[float] = [float(x) for x in next(iter(model.query_embed([query]))).tolist()]
        if self._dimension is None:
            self._dimension = len(vec)
        return vec

    async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, query: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, query)


class FastEmbedSparseEmbedder:
    """Thread-safe lazy FastEmbed BM25 sparse embedder paired with Qdrant ``Modifier.IDF``."""

    def __init__(
        self,
        model_name: str = "Qdrant/bm25",
        *,
        cache_dir: str | None = None,
        threads: int | None = None,
        batch_size: int = 64,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._threads = threads
        self._batch_size = batch_size
        self._model: SparseTextEmbedding | None = None
        self._lock = Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def _get_model(self) -> SparseTextEmbedding:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info("fastembed_sparse_loading", model=self._model_name)
                    self._model = SparseTextEmbedding(
                        model_name=self._model_name,
                        cache_dir=self._cache_dir,
                        threads=self._threads,
                    )
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVectorData]:
        if not texts:
            return []
        model = self._get_model()
        results: list[SparseVectorData] = []
        for emb in model.embed(list(texts), batch_size=self._batch_size):
            results.append(
                SparseVectorData(
                    indices=[int(i) for i in emb.indices.tolist()],
                    values=[float(v) for v in emb.values.tolist()],
                )
            )
        return results

    def embed_query(self, query: str) -> SparseVectorData:
        model = self._get_model()
        emb = next(iter(model.query_embed([query])))
        return SparseVectorData(
            indices=[int(i) for i in emb.indices.tolist()],
            values=[float(v) for v in emb.values.tolist()],
        )

    async def aembed_documents(self, texts: Sequence[str]) -> list[SparseVectorData]:
        if not texts:
            return []
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, query: str) -> SparseVectorData:
        return await asyncio.to_thread(self.embed_query, query)
