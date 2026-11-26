"""Async native Qdrant hybrid store using Named Vectors, Modifier.IDF, Prefetch & RRF."""

from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
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
from adaptiverag.retrieval.write_lock import CollectionWriteLock, default_write_lock_path
from adaptiverag.schemas.retrieval import EvidenceChunk, SourceType

logger = structlog.get_logger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
RRF_K_CONSTANT = 60

# Payload keys on every point naming the models that embedded it. The v1.13 server this project
# used to pin accepted collection metadata but didn't keep it, so the points carry the record
# themselves, which works the same on every server version.
DENSE_MODEL_PAYLOAD_KEY = "dense_model"
SPARSE_MODEL_PAYLOAD_KEY = "sparse_model"

# The first chunk of each document also carries the document's token total, so listing
# documents reads one point per document instead of every chunk.
DOC_TOKENS_PAYLOAD_KEY = "doc_token_count"
# Payload fields describing a document; every chunk of it holds the same values.
_DOCUMENT_FIELDS = ["doc_id", "title", "language", "metadata", "parent_id"]
# Older documents are added up from their chunks, filtered by this many doc_ids at a time.
_SCAN_DOC_IDS_PER_FILTER = 500


class EmbeddingModelMismatchError(RuntimeError):
    """The collection was indexed with other embedding models than this store embeds with."""


@dataclass(frozen=True)
class IndexedEmbeddings:
    """What a collection was indexed with, as far as it records it."""

    # Size of the stored dense vectors, from the collection's own vector config.
    dense_dimension: int | None
    # Models recorded on the stored passages; None when no passage records them, either
    # because the collection is empty or because it was indexed before they were recorded.
    dense_model: str | None
    sparse_model: str | None


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


def _document_tokens(parts: Iterable[tuple[int, str | None, int | None]]) -> int:
    """Tokens a document holds, from (token_count, parent_id, parent_token_count) per chunk.

    Children of one parent section overlap, so a parent-child document counts each parent
    once instead of adding up its children.
    """
    seen_parents: set[str] = set()
    total = 0
    for token_count, parent_id, parent_token_count in parts:
        if not parent_id:
            total += token_count
        elif parent_id not in seen_parents:
            seen_parents.add(parent_id)
            total += parent_token_count or token_count
    return total


def _document_heads(chunks: Sequence[DocumentChunk]) -> dict[str, int]:
    """Map the first chunk of each document in ``chunks`` to the document's token total."""
    by_doc: dict[str, list[DocumentChunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)
    return {
        min(doc_chunks, key=lambda c: c.chunk_index).chunk_id: _document_tokens(
            (c.token_count, c.parent_id, c.parent_token_count) for c in doc_chunks
        )
        for doc_chunks in by_doc.values()
    }


def _document_summary(
    payload: dict[str, Any],
    *,
    chunk_count: int,
    total_tokens: int,
    parent_child: bool,
) -> dict[str, Any]:
    """One ``list_documents`` row from a chunk's payload and the document's totals."""
    doc_id = str(payload.get("doc_id") or "").strip()
    raw_meta = payload.get("metadata")
    meta: dict[str, Any] = dict(raw_meta) if isinstance(raw_meta, dict) else {}
    source_url = meta.get("source_url") or meta.get("url")
    source_path = meta.get("source_path")
    file_name = meta.get("file_name")
    return {
        "doc_id": doc_id,
        "title": str(payload.get("title") or doc_id),
        "source_origin": (
            "web" if source_url or meta.get("source") == "firecrawl_local" else "local"
        ),
        "doc_type": str(meta.get("doc_type") or "markdown"),
        "language": str(payload.get("language") or meta.get("language") or "en"),
        "chunk_strategy": (
            "parent_child"
            if parent_child or meta.get("chunk_strategy") == "parent_child"
            else "flat"
        ),
        "chunk_count": chunk_count,
        "total_tokens": total_tokens,
        "source_url": str(source_url) if source_url else None,
        "source_path": str(source_path) if source_path else None,
        "file_name": str(file_name) if file_name else None,
    }


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
    source_url = metadata.get("source_url") or metadata.get("url")

    raw_parent_id = payload.get("parent_id")
    raw_parent_text = payload.get("parent_text")
    raw_parent_section = payload.get("parent_section_path")
    raw_parent_tokens = payload.get("parent_token_count")

    return EvidenceChunk(
        chunk_id=str(payload.get("chunk_id") or point.id),
        text=str(payload.get("text", "")),
        source_type=SourceType.LOCAL,
        doc_id=str(payload.get("doc_id", "")),
        title=str(payload.get("title", "")),
        section_path=str(payload.get("section_path", "")),
        url=str(source_url) if source_url else None,
        metadata=metadata,
        dense_score=dense_score,
        sparse_score=sparse_score,
        fusion_score=fusion_score,
        matched_sub_queries=[query],
        token_count=int(payload.get("token_count", 0)),
        parent_id=str(raw_parent_id) if raw_parent_id else None,
        parent_text=str(raw_parent_text) if raw_parent_text else None,
        parent_section_path=str(raw_parent_section) if raw_parent_section else None,
        parent_token_count=int(raw_parent_tokens) if raw_parent_tokens is not None else None,
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
        write_lock_path: Path | None = None,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._dense = dense_embedder
        self._sparse = sparse_embedder
        self._is_local_mode = is_local_mode
        # Writes run one at a time, so an upload can't land in a collection that a running
        # rebuild is about to replace. With a lock file, that holds for every process that
        # uses the same file, such as a second API worker or the CLI indexer next to the API.
        self._write_lock = CollectionWriteLock(write_lock_path)
        # When check_embedding_models last passed (time.monotonic()), so searches can reuse it.
        self._models_checked_at: float | None = None

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
        """Build a configured QdrantHybridStore from application Settings.

        A store on a Qdrant server takes the collection's lock file for every write, so the
        processes this user runs on this machine write to the collection one at a time.
        """
        name = collection_name or settings.qdrant_collection
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
            collection_name=name,
            dense_embedder=dense_embedder
            or FastEmbedDenseEmbedder(model_name=settings.dense_embedding_model),
            sparse_embedder=sparse_embedder
            or FastEmbedSparseEmbedder(model_name=settings.sparse_embedding_model),
            is_local_mode=is_local,
            write_lock_path=None if in_memory else default_write_lock_path(name),
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

    @property
    def _backing_prefix(self) -> str:
        return f"{self._collection_name}__"

    def _new_backing_name(self) -> str:
        """Name a new backing collection, e.g. ``adaptiverag_docs__20261126190501_3f9a1c``."""
        stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        return f"{self._backing_prefix}{stamp}_{secrets.token_hex(3)}"

    def _alias_switch(self, backing: str) -> list[models.AliasOperations]:
        """Alias operations that point the collection name at ``backing`` in one update."""
        return [
            models.DeleteAliasOperation(
                delete_alias=models.DeleteAlias(alias_name=self._collection_name)
            ),
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=backing,
                    alias_name=self._collection_name,
                )
            ),
        ]

    async def ensure_collection(self, *, recreate: bool = False) -> bool:
        """Ensure the hybrid collection exists with dense COSINE + sparse IDF vectors.

        A new collection is stored under a dated backing name and the configured name becomes
        a Qdrant alias for it, so ``rebuild_collection`` can later switch to a fresh copy in
        one step. ``recreate=True`` swaps in an empty collection that way.

        Returns True if a collection was created (or recreated), False if it already existed.
        """
        if recreate:
            await self.rebuild_collection([])
            return True
        if await self._client.collection_exists(self._collection_name):
            return False

        async with self._write_lock:
            if await self._client.collection_exists(self._collection_name):
                return False
            backing = self._new_backing_name()
            try:
                await self._create_backing_collection(backing)
                await self._client.update_collection_aliases(
                    change_aliases_operations=self._alias_switch(backing)
                )
            except BaseException:
                await self._drop_collection_quietly(backing)
                raise
            return True

    async def rebuild_collection(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        batch_size: int = 64,
    ) -> int:
        """Replace everything in the collection with ``chunks``; return how many were stored.

        The chunks go into a new backing collection while searches keep using the current
        one. Only when every batch is stored does the collection name move to the new copy,
        in a single alias update, and the old copy get dropped. If embedding or upserting
        fails, the new copy is dropped and the current collection is left as it was.
        """
        async with self._write_lock:
            backing = self._new_backing_name()
            try:
                await self._create_backing_collection(backing)
                upserted = await self._upsert_into(
                    backing,
                    chunks,
                    batch_size=batch_size,
                    replace_existing_docs=False,
                )
            except BaseException:
                await self._drop_collection_quietly(backing)
                raise

            try:
                previous = await self._point_name_at(backing)
            except Exception:
                logger.error(
                    "qdrant_alias_switch_failed",
                    collection=self._collection_name,
                    backing_collection=backing,
                    detail="The new copy is complete, but the collection name doesn't point at "
                    "it. Point the alias at it in Qdrant, or rebuild again.",
                )
                raise

            if previous is not None and previous.startswith(self._backing_prefix):
                await self._drop_collection_quietly(previous)
            elif previous is not None:
                logger.warning(
                    "qdrant_previous_collection_kept",
                    collection=self._collection_name,
                    previous_collection=previous,
                    detail="The alias pointed at a collection AdaptiveRAG didn't create, "
                    "so it was left in place.",
                )
            logger.info(
                "qdrant_collection_rebuilt",
                collection=self._collection_name,
                backing_collection=backing,
                replaced_collection=previous,
                points=upserted,
            )
            return upserted

    async def _point_name_at(self, backing: str) -> str | None:
        """Make the collection name an alias of ``backing``; return the collection it named."""
        aliases = await self._client.get_aliases()
        previous = next(
            (a.collection_name for a in aliases.aliases if a.alias_name == self._collection_name),
            None,
        )
        if previous is None and await self._client.collection_exists(self._collection_name):
            # The name is still a plain collection from before aliases were used. Qdrant can't
            # give an alias a name that a collection holds, so drop it first: searches miss for
            # the moment between the two calls. This happens once, on the first rebuild.
            logger.warning(
                "qdrant_collection_migrating_to_alias",
                collection=self._collection_name,
                backing_collection=backing,
            )
            await self._client.delete_collection(self._collection_name)
        await self._client.update_collection_aliases(
            change_aliases_operations=self._alias_switch(backing)
        )
        return previous

    async def _create_backing_collection(self, name: str) -> None:
        """Create an empty collection with this store's dense + sparse vector layout."""
        dim = self._dense.dimension
        await self._client.create_collection(
            collection_name=name,
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

        # Keyword payload indexes accelerate Filterable HNSW on server Qdrant; the doc_id one
        # also counts chunks per document for list_documents.
        if not self._is_local_mode:
            for field_name in ("doc_id", "language", "parent_id"):
                await self._client.create_payload_index(
                    collection_name=name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            # Lets list_documents find each document's first chunk without a full scan.
            await self._client.create_payload_index(
                collection_name=name,
                field_name=DOC_TOKENS_PAYLOAD_KEY,
                field_schema=models.PayloadSchemaType.INTEGER,
            )

        logger.info(
            "qdrant_collection_created",
            collection=self._collection_name,
            backing_collection=name,
            dense_dim=dim,
            dense_model=self._dense.model_name,
            sparse_model=self._sparse.model_name,
        )

    async def _drop_collection_quietly(self, name: str) -> None:
        """Delete a backing collection; if that fails it's only left behind, so log it."""
        try:
            await self._client.delete_collection(name)
        except Exception as exc:
            logger.warning("qdrant_collection_drop_failed", backing_collection=name, error=str(exc))

    async def indexed_embeddings(self) -> IndexedEmbeddings | None:
        """Read what the collection was indexed with; None if the collection doesn't exist."""
        if not await self._client.collection_exists(self._collection_name):
            return None
        info = await self._client.get_collection(self._collection_name)
        vectors = info.config.params.vectors
        dense_params = vectors.get(DENSE_VECTOR_NAME) if isinstance(vectors, dict) else None
        # One passage that records its models is enough: every write is checked against
        # them first, so the passages that record models all record the same ones.
        records, _ = await self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=models.Filter(
                must_not=[
                    models.IsEmptyCondition(
                        is_empty=models.PayloadField(key=DENSE_MODEL_PAYLOAD_KEY)
                    )
                ]
            ),
            limit=1,
            with_payload=[DENSE_MODEL_PAYLOAD_KEY, SPARSE_MODEL_PAYLOAD_KEY],
            with_vectors=False,
        )
        recorded: dict[str, Any] = dict(records[0].payload or {}) if records else {}
        dense_model = recorded.get(DENSE_MODEL_PAYLOAD_KEY)
        sparse_model = recorded.get(SPARSE_MODEL_PAYLOAD_KEY)
        return IndexedEmbeddings(
            dense_dimension=dense_params.size if dense_params is not None else None,
            dense_model=str(dense_model) if dense_model else None,
            sparse_model=str(sparse_model) if sparse_model else None,
        )

    async def check_embedding_models(self, *, max_age_seconds: float = 30.0) -> None:
        """Raise EmbeddingModelMismatchError if the collection was indexed with other models.

        Vectors from different models aren't comparable, so searching such a collection
        returns unrelated passages, and writing to it mixes two kinds of vectors. A check that
        passed less than ``max_age_seconds`` ago is reused, so searches don't each pay for
        one; writes pass 0 to always look. A failed check is never reused.
        """
        checked_at = self._models_checked_at
        if checked_at is not None and time.monotonic() - checked_at < max_age_seconds:
            return
        self._models_checked_at = None
        indexed = await self.indexed_embeddings()
        problem = self._embedding_mismatch(indexed) if indexed is not None else None
        if problem is not None:
            logger.warning(
                "qdrant_embedding_models_mismatch",
                collection=self._collection_name,
                detail=problem,
            )
            raise EmbeddingModelMismatchError(problem)
        self._models_checked_at = time.monotonic()

    def _embedding_mismatch(self, indexed: IndexedEmbeddings) -> str | None:
        """Explain how the collection's embeddings differ from this store's; None if they don't."""
        was: list[str] = []
        now: list[str] = []
        if indexed.dense_model is not None:
            if indexed.dense_model != self._dense.model_name:
                was.append(f"dense model '{indexed.dense_model}'")
                now.append(f"DENSE_EMBEDDING_MODEL is '{self._dense.model_name}'")
        elif (
            indexed.dense_dimension is not None and indexed.dense_dimension != self._dense.dimension
        ):
            # Passages from before models were recorded: only the vector size can tell.
            was.append(f"{indexed.dense_dimension}-dimension dense vectors")
            now.append(
                f"DENSE_EMBEDDING_MODEL '{self._dense.model_name}' makes "
                f"{self._dense.dimension}-dimension ones"
            )
        if indexed.sparse_model is not None and indexed.sparse_model != self._sparse.model_name:
            was.append(f"sparse model '{indexed.sparse_model}'")
            now.append(f"SPARSE_EMBEDDING_MODEL is '{self._sparse.model_name}'")
        if not was:
            return None
        return (
            f"Collection '{self._collection_name}' was indexed with {' and '.join(was)}, but "
            f"{' and '.join(now)}. Searching it with other embedding models returns unrelated "
            "passages. Re-index it by replacing everything in the collection, or switch back "
            "to the models it was indexed with."
        )

    async def delete_by_doc_ids(self, doc_ids: Sequence[str]) -> None:
        """Delete existing chunks belonging to specified doc_ids before clean re-upsert."""
        async with self._write_lock:
            await self._delete_by_doc_ids(doc_ids)

    async def _delete_by_doc_ids(self, doc_ids: Sequence[str]) -> None:
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

    async def _list_point_ids_by_doc_ids(
        self,
        collection_name: str,
        doc_ids: Sequence[str],
    ) -> list[models.ExtendedPointId]:
        """Collect existing Qdrant point IDs belonging to ``doc_ids`` without loading vectors or text."""
        unique_ids = list(dict.fromkeys(d for d in doc_ids if d))
        flt = _build_filter(doc_ids=unique_ids)
        if flt is None or not await self._client.collection_exists(collection_name):
            return []

        found_ids: list[models.ExtendedPointId] = []
        offset: Any = None
        while True:
            records, next_offset = await self._client.scroll(
                collection_name=collection_name,
                scroll_filter=flt,
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            found_ids.extend(rec.id for rec in records)
            if next_offset is None or not records:
                break
            offset = next_offset
        return found_ids

    async def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        batch_size: int = 64,
        replace_existing_docs: bool = True,
    ) -> int:
        """Embed and batch-upsert DocumentChunks into Qdrant (crash-safe & zero-gap re-indexing).

        Raises EmbeddingModelMismatchError, before embedding anything, if the collection was
        indexed with other models. Replacing the whole collection is the way to switch models.
        """
        async with self._write_lock:
            if chunks:
                await self.check_embedding_models(max_age_seconds=0)
            return await self._upsert_into(
                self._collection_name,
                chunks,
                batch_size=batch_size,
                replace_existing_docs=replace_existing_docs,
            )

    async def _upsert_into(
        self,
        collection_name: str,
        chunks: Sequence[DocumentChunk],
        *,
        batch_size: int,
        replace_existing_docs: bool,
    ) -> int:
        """Store ``chunks``, which hold every chunk of each document they belong to.

        Each document is stored as exactly these chunks, and its first chunk records the
        document's token total for ``list_documents``.
        """
        if not chunks:
            return 0

        existing_ids: list[models.ExtendedPointId] = []
        if replace_existing_docs:
            doc_ids = [c.doc_id for c in chunks]
            existing_ids = await self._list_point_ids_by_doc_ids(collection_name, doc_ids)

        document_tokens_by_head = _document_heads(chunks)
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
                    DENSE_MODEL_PAYLOAD_KEY: self._dense.model_name,
                    SPARSE_MODEL_PAYLOAD_KEY: self._sparse.model_name,
                }
                if chunk.parent_id is not None:
                    payload["parent_id"] = chunk.parent_id
                    payload["parent_text"] = chunk.parent_text
                    payload["parent_section_path"] = chunk.parent_section_path
                    payload["parent_token_count"] = chunk.parent_token_count
                    payload["child_index"] = chunk.child_index
                    payload["child_count"] = chunk.child_count
                if chunk.chunk_id in document_tokens_by_head:
                    payload[DOC_TOKENS_PAYLOAD_KEY] = document_tokens_by_head[chunk.chunk_id]
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
                collection_name=collection_name,
                points=points,
                wait=True,
            )
            total_upserted += len(points)

        # Clean up stale leftover point IDs ONLY after all new batches have succeeded,
        # eliminating both data-loss risk on embedding errors and empty-document query gaps.
        if replace_existing_docs and existing_ids:
            new_id_set = {c.chunk_id for c in chunks}
            stale_ids: list[models.ExtendedPointId] = [
                pid for pid in existing_ids if str(pid) not in new_id_set
            ]
            if stale_ids:
                await self._client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(points=stale_ids),
                    wait=True,
                )

        logger.info(
            "qdrant_chunks_upserted",
            collection=collection_name,
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
        """Execute a single-query retrieval in HYBRID, DENSE_ONLY, or SPARSE_ONLY mode.

        Raises EmbeddingModelMismatchError if the collection was indexed with other models.
        """
        await self.check_embedding_models()
        return await self._search_single(
            query,
            mode=mode,
            dense_limit=dense_limit,
            sparse_limit=sparse_limit,
            fusion_limit=fusion_limit,
            doc_ids=doc_ids,
            language=language,
        )

    async def _search_single(
        self,
        query: str,
        *,
        mode: RetrievalMode,
        dense_limit: int,
        sparse_limit: int,
        fusion_limit: int,
        doc_ids: Sequence[str] | None,
        language: str | None,
    ) -> list[EvidenceChunk]:
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
        """Fan out multiple sub-queries concurrently and merge via cross-query RRF deduplication.

        Raises EmbeddingModelMismatchError if the collection was indexed with other models.
        """
        unique_queries = list(dict.fromkeys(q.strip() for q in queries if q and q.strip()))
        if not unique_queries:
            return []
        # Checked once here rather than in each concurrent sub-query.
        await self.check_embedding_models()

        if len(unique_queries) == 1:
            results = await self._search_single(
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
                self._search_single(
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

    async def count_by_doc_id(self, doc_id: str) -> int:
        """Return exact chunk count for a single doc_id."""
        cleaned = doc_id.strip()
        if not cleaned or not await self._client.collection_exists(self._collection_name):
            return 0
        flt = _build_filter(doc_ids=[cleaned])
        res = await self._client.count(
            collection_name=self._collection_name,
            count_filter=flt,
            exact=True,
        )
        return int(res.count)

    async def delete_document(self, doc_id: str) -> tuple[int, int]:
        """Delete all chunks for a single doc_id and return (deleted_chunks, remaining_points)."""
        cleaned = doc_id.strip()
        if not cleaned or not await self._client.collection_exists(self._collection_name):
            return 0, 0

        async with self._write_lock:
            existing_count = await self.count_by_doc_id(cleaned)
            if existing_count == 0:
                return 0, await self.count_points()
            await self._delete_by_doc_ids([cleaned])
            remaining = await self.count_points()
        logger.info(
            "qdrant_document_deleted",
            collection=self._collection_name,
            doc_id=cleaned,
            deleted_chunks=existing_count,
            remaining_points=remaining,
        )
        return existing_count, remaining

    async def list_documents(self) -> list[dict[str, Any]]:
        """Every indexed document with its chunk and token counts, in display order.

        Reads one point per document: Qdrant counts the chunks per doc_id from its index,
        and each document's first chunk holds the rest. Documents stored before first chunks
        recorded a token total are added up from their chunks, and so is every document when
        the collection has no doc_id index to count with.
        """
        if not await self._client.collection_exists(self._collection_name):
            return []
        try:
            chunk_counts = await self._chunk_counts_by_doc_id()
        except Exception as exc:
            logger.warning(
                "qdrant_document_counts_unavailable",
                collection=self._collection_name,
                error=str(exc),
                detail="Listing documents by reading every chunk instead.",
            )
            docs = await self._documents_by_scan()
        else:
            docs = await self._documents_from_heads(chunk_counts)
            older = [doc_id for doc_id in chunk_counts if doc_id not in docs]
            if len(older) == len(chunk_counts):
                docs = await self._documents_by_scan()
            else:
                for start in range(0, len(older), _SCAN_DOC_IDS_PER_FILTER):
                    batch = older[start : start + _SCAN_DOC_IDS_PER_FILTER]
                    docs.update(await self._documents_by_scan(batch))

        return sorted(
            docs.values(),
            key=lambda d: (str(d["source_origin"]), str(d["title"]).lower(), str(d["doc_id"])),
        )

    async def _chunk_counts_by_doc_id(self) -> dict[str, int]:
        """Chunks per document, counted by Qdrant from the doc_id payload index."""
        points = await self._client.count(collection_name=self._collection_name, exact=True)
        if points.count == 0:
            return {}
        facet = await self._client.facet(
            collection_name=self._collection_name,
            key="doc_id",
            limit=points.count,  # every document has at least one chunk
            exact=True,
        )
        return {str(hit.value): int(hit.count) for hit in facet.hits if str(hit.value).strip()}

    async def _documents_from_heads(
        self, chunk_counts: dict[str, int]
    ) -> dict[str, dict[str, Any]]:
        """Summaries of the documents whose first chunk records their token total."""
        heads = models.Filter(
            must=[models.FieldCondition(key=DOC_TOKENS_PAYLOAD_KEY, range=models.Range(gte=0))]
        )
        docs: dict[str, dict[str, Any]] = {}
        async for payload in self._scroll_payloads(
            heads, [*_DOCUMENT_FIELDS, DOC_TOKENS_PAYLOAD_KEY]
        ):
            doc_id = str(payload.get("doc_id") or "")
            if doc_id in chunk_counts and doc_id not in docs:
                docs[doc_id] = _document_summary(
                    payload,
                    chunk_count=chunk_counts[doc_id],
                    total_tokens=int(payload[DOC_TOKENS_PAYLOAD_KEY]),
                    parent_child=bool(payload.get("parent_id")),
                )
        return docs

    async def _documents_by_scan(
        self, doc_ids: Sequence[str] | None = None
    ) -> dict[str, dict[str, Any]]:
        """Summaries added up from every chunk of ``doc_ids`` (all documents when None)."""
        first_payloads: dict[str, dict[str, Any]] = {}
        parts: dict[str, list[tuple[int, str | None, int | None]]] = {}
        fields = [*_DOCUMENT_FIELDS, "token_count", "parent_token_count"]
        async for payload in self._scroll_payloads(_build_filter(doc_ids=doc_ids), fields):
            doc_id = str(payload.get("doc_id") or "")
            if not doc_id.strip():
                continue
            first_payloads.setdefault(doc_id, payload)
            raw_parent_id = payload.get("parent_id")
            raw_parent_tokens = payload.get("parent_token_count")
            parts.setdefault(doc_id, []).append(
                (
                    int(payload.get("token_count") or 0),
                    str(raw_parent_id).strip() if raw_parent_id else None,
                    int(raw_parent_tokens) if raw_parent_tokens is not None else None,
                )
            )
        return {
            doc_id: _document_summary(
                first_payloads[doc_id],
                chunk_count=len(doc_parts),
                total_tokens=_document_tokens(doc_parts),
                parent_child=any(parent_id for _, parent_id, _ in doc_parts),
            )
            for doc_id, doc_parts in parts.items()
        }

    async def _scroll_payloads(
        self, scroll_filter: models.Filter | None, fields: list[str]
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield the payload ``fields`` of every point matching ``scroll_filter``."""
        offset: Any = None
        while True:
            records, offset = await self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=scroll_filter,
                limit=256,
                offset=offset,
                # Only the fields asked for; never `text` or `parent_text`.
                with_payload=fields,
                with_vectors=False,
            )
            for record in records:
                yield dict(record.payload or {})
            if offset is None or not records:
                return

    async def health_check(self) -> dict[str, Any]:
        """Return collection health, including the models its passages were indexed with."""
        indexed = await self.indexed_embeddings()
        mismatch = self._embedding_mismatch(indexed) if indexed is not None else None
        if indexed is None:
            status = "collection_missing"
        elif mismatch is not None:
            status = "embedding_mismatch"
        else:
            status = "ok"
        return {
            "status": status,
            "collection": self._collection_name,
            "collection_exists": indexed is not None,
            "points_count": await self.count_points() if indexed is not None else 0,
            # The models questions and new passages are embedded with.
            "dense_model": self._dense.model_name,
            "sparse_model": self._sparse.model_name,
            # The models the stored passages were embedded with (None if none record them).
            "indexed_dense_model": indexed.dense_model if indexed else None,
            "indexed_sparse_model": indexed.sparse_model if indexed else None,
            "indexed_dense_dimension": indexed.dense_dimension if indexed else None,
            "embedding_mismatch": mismatch,
        }


def _max_optional(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)
