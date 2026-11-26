"""The collection records the embedding models that indexed it and refuses to mix in others."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from qdrant_client import AsyncQdrantClient, models

from adaptiverag.ingestion.indexer import index_documents
from adaptiverag.ingestion.loader import LoadedDocument
from adaptiverag.retrieval.embeddings import SparseVectorData
from adaptiverag.retrieval.qdrant_store import (
    DENSE_MODEL_PAYLOAD_KEY,
    SPARSE_MODEL_PAYLOAD_KEY,
    EmbeddingModelMismatchError,
    IndexedEmbeddings,
    QdrantHybridStore,
)

COLLECTION = "kb"


class _Dense:
    def __init__(self, model_name: str, dimension: int) -> None:
        self.model_name = model_name
        self.dimension = dimension

    def _vector(self) -> list[float]:
        return [1.0] + [0.0] * (self.dimension - 1)

    async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector() for _ in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self._vector()


class _Sparse:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    async def aembed_documents(self, texts: Sequence[str]) -> list[SparseVectorData]:
        return [SparseVectorData(indices=[1], values=[1.0]) for _ in texts]

    async def aembed_query(self, text: str) -> SparseVectorData:
        return SparseVectorData(indices=[1], values=[1.0])


def _store(
    client: AsyncQdrantClient,
    *,
    dense: str = "dense-a",
    sparse: str = "bm25-a",
    dimension: int = 4,
) -> QdrantHybridStore:
    """The store a server configured with these models would build over ``client``."""
    return QdrantHybridStore(
        client=client,
        collection_name=COLLECTION,
        dense_embedder=_Dense(dense, dimension),  # type: ignore[arg-type]
        sparse_embedder=_Sparse(sparse),  # type: ignore[arg-type]
        is_local_mode=True,
    )


def _doc(doc_id: str) -> LoadedDocument:
    return LoadedDocument(
        doc_id=doc_id,
        title=doc_id,
        content=f"Text of {doc_id}.",
        source_path=f"inline://{doc_id}",
        doc_type="markdown",
        metadata={"language": "en"},
    )


async def _index(store: QdrantHybridStore, *doc_ids: str, recreate: bool = False) -> Any:
    docs = [_doc(doc_id) for doc_id in doc_ids]
    return await index_documents(docs, store, source="test", recreate=recreate)


async def _doc_ids(store: QdrantHybridStore) -> set[str]:
    return {str(doc["doc_id"]) for doc in await store.list_documents()}


@pytest.mark.asyncio
async def test_every_passage_records_the_models_that_embedded_it() -> None:
    client = AsyncQdrantClient(location=":memory:")
    store = _store(client)
    try:
        await _index(store, "guide", "faq")

        records, _ = await client.scroll(COLLECTION, limit=10, with_payload=True)
        assert len(records) == 2
        assert {(r.payload or {}).get(DENSE_MODEL_PAYLOAD_KEY) for r in records} == {"dense-a"}
        assert {(r.payload or {}).get(SPARSE_MODEL_PAYLOAD_KEY) for r in records} == {"bm25-a"}
        assert await store.indexed_embeddings() == IndexedEmbeddings(
            dense_dimension=4, dense_model="dense-a", sparse_model="bm25-a"
        )

        health = await store.health_check()
        assert health["status"] == "ok"
        assert health["indexed_dense_model"] == "dense-a"
        assert health["indexed_sparse_model"] == "bm25-a"
        assert health["indexed_dense_dimension"] == 4
        assert health["embedding_mismatch"] is None
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("swap", "recorded"),
    [
        ({"dense": "dense-b"}, "dense model 'dense-a'"),
        ({"sparse": "bm25-b"}, "sparse model 'bm25-a'"),
    ],
)
async def test_a_same_size_model_swap_is_refused_instead_of_searched(
    swap: dict[str, str], recorded: str
) -> None:
    """A model with the same vector size used to be searched silently (backlog #5)."""
    client = AsyncQdrantClient(location=":memory:")
    try:
        await _index(_store(client), "guide")
        swapped = _store(client, **swap)

        with pytest.raises(EmbeddingModelMismatchError, match=recorded):
            await swapped.search_multi_query(["guide"])
        with pytest.raises(EmbeddingModelMismatchError, match=recorded):
            await swapped.search_single("guide")
        with pytest.raises(EmbeddingModelMismatchError, match=recorded):
            await _index(swapped, "notes")
        assert await _doc_ids(swapped) == {"guide"}

        health = await swapped.health_check()
        assert health["status"] == "embedding_mismatch"
        assert recorded in health["embedding_mismatch"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_replacing_the_collection_switches_it_to_the_new_model() -> None:
    client = AsyncQdrantClient(location=":memory:")
    try:
        old = _store(client)
        await _index(old, "guide")
        new = _store(client, dense="dense-b")
        with pytest.raises(EmbeddingModelMismatchError):
            await new.search_multi_query(["guide"])

        await _index(new, "guide", "notes", recreate=True)

        assert len(await new.search_multi_query(["guide"])) == 2
        health = await new.health_check()
        assert health["status"] == "ok"
        assert health["dense_model"] == "dense-b"
        assert health["indexed_dense_model"] == "dense-b"
        # A server still configured with the old model is the mismatched one now.
        with pytest.raises(EmbeddingModelMismatchError, match="dense model 'dense-b'"):
            await old.check_embedding_models(max_age_seconds=0)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_a_collection_indexed_before_models_were_recorded() -> None:
    client = AsyncQdrantClient(location=":memory:")
    try:
        await _index(_store(client), "guide")
        # Earlier versions stored no model names on the passages.
        await client.delete_payload(
            collection_name=COLLECTION,
            keys=[DENSE_MODEL_PAYLOAD_KEY, SPARSE_MODEL_PAYLOAD_KEY],
            points=models.FilterSelector(filter=models.Filter()),
        )
        assert await _store(client).indexed_embeddings() == IndexedEmbeddings(
            dense_dimension=4, dense_model=None, sparse_model=None
        )

        # Same vector size: nothing tells the models apart, so it's searched as before.
        same_size = _store(client, dense="dense-b")
        assert len(await same_size.search_multi_query(["guide"])) == 1
        assert (await same_size.health_check())["status"] == "ok"

        # Another vector size shows in the collection's own vector config.
        bigger = _store(client, dense="dense-c", dimension=8)
        with pytest.raises(EmbeddingModelMismatchError, match="4-dimension dense vectors"):
            await bigger.search_multi_query(["guide"])
        with pytest.raises(EmbeddingModelMismatchError, match="4-dimension dense vectors"):
            await _index(bigger, "notes")

        await _index(bigger, "notes", recreate=True)
        assert await _doc_ids(bigger) == {"notes"}
        assert (await bigger.health_check())["indexed_dense_dimension"] == 8
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_searches_reuse_a_recent_check_but_uploads_always_look(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = AsyncQdrantClient(location=":memory:")
    try:
        await _index(_store(client), "guide")
        restarted = _store(client)
        original = restarted.indexed_embeddings
        reads = 0

        async def counting_read() -> IndexedEmbeddings | None:
            nonlocal reads
            reads += 1
            return await original()

        monkeypatch.setattr(restarted, "indexed_embeddings", counting_read)

        await restarted.search_multi_query(["guide", "faq", "notes"])
        await restarted.search_single("guide")
        assert reads == 1  # one check for the whole fan-out, reused by the next search

        await _index(restarted, "faq")
        await _index(restarted, "notes")
        assert reads == 3  # every upload looks again
    finally:
        await client.close()
