"""Async native Qdrant hybrid store using Named Vectors, Modifier.IDF, Prefetch & RRF."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from enum import StrEnum
from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient, models

from adaptiverag.config import Settings
from adaptiverag.ingestion.chunker import DocumentChunk
from adaptiverag.retrieval.embeddings import (
    DenseEmbedder,
    FastEmbedDenseEmbedder,
    FastEmbedSparseEmbedder,
    SparseEmbedder,
    SparseVectorData,
)
from adaptiverag.schemas.retrieval import EvidenceChunk, SourceType

logger = structlog.get_logger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
RRF_K_CONSTANT = 60


class RetrievalMode(StrEnum):
    """Retrieval execution mode supporting ablation comparisons (Naive vs Hybrid)."""

    HYBRID = "hybrid"
    DENSE_ONLY = "dense_only"
    SPARSE_ONLY = "sparse_only"


def _to_qdrant_sparse(vec: SparseVectorData) -> models.SparseVector:
    return models.SparseVector(indices=vec.indices, values=vec.values)


def _build_filter(
    *,
    doc_ids: Sequence[str] | None = None,
    language: str | None = None,
) -> models.Filter | None:
    conditions: list[models.Condition] = []
    if doc_ids:
        conditions.append(
            models.FieldCondition(
                key="doc_id",
                match=models.MatchAny(any=list(doc_ids)),
            )
        )
    if language:
        conditions.append(
            models.FieldCondition(
                key="language",
                match=models.MatchValue(value=language),
            )
        )
    return models.Filter(must=conditions) if conditions else None


def _point_to_evidence(
    point: models.ScoredPoint,
    *,
    query: str,
    mode: RetrievalMode,
) -> EvidenceChunk:
    payload: dict[str, Any] = dict(point.payload or {})
    score = float(point.score)

    dense_score = score if mode == RetrievalMode.DENSE_ONLY else None
    sparse_score = score if mode == RetrievalMode.SPARSE_ONLY else None
    fusion_score = score

    raw_meta = payload.get("metadata")
    metadata: dict[str, Any] = dict(raw_meta) if isinstance(raw_meta, dict) else {}

    return EvidenceChunk(
        chunk_id=str(payload.get("chunk_id") or point.id),
        text=str(payload.get("text", "")),
        source_type=SourceType.LOCAL,
        doc_id=str(payload.get("doc_id", "")),
        title=str(payload.get("title", "")),
        section_path=str(payload.get("section_path", "")),
        metadata=metadata,
        dense_score=dense_score,
        sparse_score=sparse_score,
        fusion_score=fusion_score,
        matched_sub_queries=[query],
        token_count=int(payload.get("token_count", 0)),
    )


class QdrantHybridStore:
    """Async Qdrant vector store with native Dense + BM25 Hybrid RRF and multi-query fan-out."""

    def __init__(
        self,
        *,
        client: AsyncQdrantClient,
        collection_name: str,
        dense_embedder: DenseEmbedder,
        sparse_embedder: SparseEmbedder,
        is_local_mode: bool = False,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._dense = dense_embedder
        self._sparse = sparse_embedder
        self._is_local_mode = is_local_mode

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        collection_name: str | None = None,
        dense_embedder: DenseEmbedder | None = None,
        sparse_embedder: SparseEmbedder | None = None,
        in_memory: bool = False,
    ) -> QdrantHybridStore:
        """Build a configured QdrantHybridStore from application Settings."""
        if in_memory:
            client = AsyncQdrantClient(location=":memory:")
            is_local = True
        else:
            api_key = (
                settings.qdrant_api_key.get_secret_value().strip() or None
                if settings.qdrant_api_key is not None
                else None
            )
            client = AsyncQdrantClient(url=settings.qdrant_url, api_key=api_key)
            is_local = False

        return cls(
            client=client,
            collection_name=collection_name or settings.qdrant_collection,
            dense_embedder=dense_embedder
            or FastEmbedDenseEmbedder(model_name=settings.dense_embedding_model),
            sparse_embedder=sparse_embedder
            or FastEmbedSparseEmbedder(model_name=settings.sparse_embedding_model),
            is_local_mode=is_local,
        )

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def client(self) -> AsyncQdrantClient:
        return self._client

    @property
    def dense_embedder(self) -> DenseEmbedder:
        return self._dense

    @property
    def sparse_embedder(self) -> SparseEmbedder:
        return self._sparse

    async def close(self) -> None:
        """Close the underlying async Qdrant client connection."""
        await self._client.close()

    async def ensure_collection(self, *, recreate: bool = False) -> bool:
        """Ensure the hybrid collection exists with dense COSINE + sparse IDF vectors.

        Returns True if the collection was newly created (or recreated), False if it
        already existed.
        """
        exists = await self._client.collection_exists(self._collection_name)
        if exists and recreate:
            logger.info("qdrant_recreating_collection", collection=self._collection_name)
            await self._client.delete_collection(self._collection_name)
            exists = False

        if exists:
            return False

        dim = self._dense.dimension
        await self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )

        # Keyword payload indexes accelerate Filterable HNSW on server Qdrant.
        if not self._is_local_mode:
            for field_name in ("doc_id", "language"):
                await self._client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

        logger.info(
            "qdrant_collection_created",
            collection=self._collection_name,
            dense_dim=dim,
            dense_model=self._dense.model_name,
            sparse_model=self._sparse.model_name,
        )
        return True

    async def delete_by_doc_ids(self, doc_ids: Sequence[str]) -> None:
        """Delete existing chunks belonging to specified doc_ids before clean re-upsert."""
        unique_ids = list(dict.fromkeys(d for d in doc_ids if d))
        if not unique_ids:
            return
        flt = _build_filter(doc_ids=unique_ids)
        if flt is None:
            return
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.FilterSelector(filter=flt),
        )

    async def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        batch_size: int = 64,
        replace_existing_docs: bool = True,
    ) -> int:
        """Embed and batch-upsert DocumentChunks into Qdrant (idempotent by UUID chunk_id)."""
        if not chunks:
            return 0

        if replace_existing_docs:
            doc_ids = [c.doc_id for c in chunks]
            await self.delete_by_doc_ids(doc_ids)

        total_upserted = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            embed_texts = [c.embedding_text for c in batch]

            dense_vecs, sparse_vecs = await asyncio.gather(
                self._dense.aembed_documents(embed_texts),
                self._sparse.aembed_documents(embed_texts),
            )

            points: list[models.PointStruct] = []
            for chunk, d_vec, s_vec in zip(batch, dense_vecs, sparse_vecs, strict=True):
                payload: dict[str, Any] = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "section_path": chunk.section_path,
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "language": chunk.metadata.get("language", "en"),
                    "metadata": chunk.metadata,
                }
                points.append(
                    models.PointStruct(
                        id=chunk.chunk_id,
                        vector={
                            DENSE_VECTOR_NAME: d_vec,
                            SPARSE_VECTOR_NAME: _to_qdrant_sparse(s_vec),
                        },
                        payload=payload,
                    )
                )

            await self._client.upsert(
                collection_name=self._collection_name,
                points=points,
                wait=True,
            )
            total_upserted += len(points)

        logger.info(
            "qdrant_chunks_upserted",
            collection=self._collection_name,
            count=total_upserted,
        )
        return total_upserted

    async def search_single(
        self,
        query: str,
        *,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        dense_limit: int = 20,
        sparse_limit: int = 25,
        fusion_limit: int = 20,
        doc_ids: Sequence[str] | None = None,
        language: str | None = None,
    ) -> list[EvidenceChunk]:
        """Execute a single-query retrieval in HYBRID, DENSE_ONLY, or SPARSE_ONLY mode."""
        cleaned = query.strip()
        if not cleaned or fusion_limit <= 0:
            return []

        q_filter = _build_filter(doc_ids=doc_ids, language=language)

        if mode == RetrievalMode.DENSE_ONLY:
            dense_vec = await self._dense.aembed_query(cleaned)
            response = await self._client.query_points(
                collection_name=self._collection_name,
                query=dense_vec,
                using=DENSE_VECTOR_NAME,
                query_filter=q_filter,
                limit=fusion_limit,
                with_payload=True,
            )
            return [
                _point_to_evidence(pt, query=cleaned, mode=RetrievalMode.DENSE_ONLY)
                for pt in response.points
            ]

        if mode == RetrievalMode.SPARSE_ONLY:
            sparse_vec = await self._sparse.aembed_query(cleaned)
            if sparse_vec.is_empty:
                return []
            response = await self._client.query_points(
                collection_name=self._collection_name,
                query=_to_qdrant_sparse(sparse_vec),
                using=SPARSE_VECTOR_NAME,
                query_filter=q_filter,
                limit=fusion_limit,
                with_payload=True,
            )
            return [
                _point_to_evidence(pt, query=cleaned, mode=RetrievalMode.SPARSE_ONLY)
                for pt in response.points
            ]

        # Native Hybrid Search: parallel dense + BM25 sparse embedding -> server-side Prefetch + RRF
        dense_vec, sparse_vec = await asyncio.gather(
            self._dense.aembed_query(cleaned),
            self._sparse.aembed_query(cleaned),
        )

        # Graceful fallback if query produced zero lexical tokens (e.g., symbols only)
        if sparse_vec.is_empty or sparse_limit <= 0:
            response = await self._client.query_points(
                collection_name=self._collection_name,
                query=dense_vec,
                using=DENSE_VECTOR_NAME,
                query_filter=q_filter,
                limit=fusion_limit,
                with_payload=True,
            )
            return [
                _point_to_evidence(pt, query=cleaned, mode=RetrievalMode.DENSE_ONLY)
                for pt in response.points
            ]

        response = await self._client.query_points(
            collection_name=self._collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vec,
                    using=DENSE_VECTOR_NAME,
                    filter=q_filter,
                    limit=max(1, dense_limit),
                ),
                models.Prefetch(
                    query=_to_qdrant_sparse(sparse_vec),
                    using=SPARSE_VECTOR_NAME,
                    filter=q_filter,
                    limit=max(1, sparse_limit),
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=fusion_limit,
            with_payload=True,
        )
        return [
            _point_to_evidence(pt, query=cleaned, mode=RetrievalMode.HYBRID)
            for pt in response.points
        ]

    async def search_multi_query(
        self,
        queries: Sequence[str],
        *,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        dense_limit: int = 20,
        sparse_limit: int = 25,
        fusion_limit: int = 20,
        max_total_candidates: int = 40,
        doc_ids: Sequence[str] | None = None,
        language: str | None = None,
    ) -> list[EvidenceChunk]:
        """Fan out multiple sub-queries concurrently and merge via cross-query RRF deduplication."""
        unique_queries = list(dict.fromkeys(q.strip() for q in queries if q and q.strip()))
        if not unique_queries:
            return []

        if len(unique_queries) == 1:
            results = await self.search_single(
                unique_queries[0],
                mode=mode,
                dense_limit=dense_limit,
                sparse_limit=sparse_limit,
                fusion_limit=min(fusion_limit, max_total_candidates),
                doc_ids=doc_ids,
                language=language,
            )
            return results[:max_total_candidates]

        per_query_lists = await asyncio.gather(
            *(
                self.search_single(
                    q,
                    mode=mode,
                    dense_limit=dense_limit,
                    sparse_limit=sparse_limit,
                    fusion_limit=fusion_limit,
                    doc_ids=doc_ids,
                    language=language,
                )
                for q in unique_queries
            )
        )

        merged_chunks: dict[str, EvidenceChunk] = {}
        cross_query_rrf: dict[str, float] = {}

        for query_text, hits in zip(unique_queries, per_query_lists, strict=True):
            for rank, chunk in enumerate(hits, start=1):
                cid = chunk.chunk_id
                cross_query_rrf[cid] = cross_query_rrf.get(cid, 0.0) + (
                    1.0 / (RRF_K_CONSTANT + rank)
                )
                if cid not in merged_chunks:
                    merged_chunks[cid] = chunk.model_copy(
                        update={"matched_sub_queries": [query_text]}
                    )
                else:
                    existing = merged_chunks[cid]
                    matched = list(dict.fromkeys([*existing.matched_sub_queries, query_text]))
                    best_dense = _max_optional(existing.dense_score, chunk.dense_score)
                    best_sparse = _max_optional(existing.sparse_score, chunk.sparse_score)
                    merged_chunks[cid] = existing.model_copy(
                        update={
                            "matched_sub_queries": matched,
                            "dense_score": best_dense,
                            "sparse_score": best_sparse,
                        }
                    )

        ranked_ids = sorted(
            merged_chunks.keys(),
            key=lambda cid: (
                cross_query_rrf[cid],
                merged_chunks[cid].fusion_score or 0.0,
            ),
            reverse=True,
        )[:max_total_candidates]

        return [
            merged_chunks[cid].model_copy(update={"fusion_score": round(cross_query_rrf[cid], 6)})
            for cid in ranked_ids
        ]

    async def count_points(self) -> int:
        """Return exact point count in the active collection."""
        if not await self._client.collection_exists(self._collection_name):
            return 0
        res = await self._client.count(collection_name=self._collection_name, exact=True)
        return int(res.count)

    async def health_check(self) -> dict[str, Any]:
        """Return lightweight Qdrant collection health metadata."""
        exists = await self._client.collection_exists(self._collection_name)
        points_count = await self.count_points() if exists else 0
        return {
            "status": "ok" if exists else "collection_missing",
            "collection": self._collection_name,
            "collection_exists": exists,
            "points_count": points_count,
            "dense_model": self._dense.model_name,
            "sparse_model": self._sparse.model_name,
        }


def _max_optional(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)
