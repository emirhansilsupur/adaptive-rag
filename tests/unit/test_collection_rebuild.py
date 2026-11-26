"""Replacing the whole collection: the new copy takes over only once it is complete."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest
from qdrant_client import AsyncQdrantClient, models

from adaptiverag.ingestion.indexer import index_documents
from adaptiverag.ingestion.loader import LoadedDocument
from adaptiverag.retrieval.embeddings import SparseVectorData
from adaptiverag.retrieval.qdrant_store import QdrantHybridStore

COLLECTION = "kb"


class _Dense:
    """Offline dense embedder that fails on ``BROKEN`` text and can be held on ``SLOW`` text."""

    model_name = "stub-dense"
    dimension = 4

    def __init__(self) -> None:
        self.slow_started = asyncio.Event()
        self.release_slow = asyncio.Event()

    async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if any("BROKEN" in text for text in texts):
            raise RuntimeError("embedding model crashed")
        if any("SLOW" in text for text in texts):
            self.slow_started.set()
            await self.release_slow.wait()
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


class _Sparse:
    model_name = "stub-bm25"

    async def aembed_documents(self, texts: Sequence[str]) -> list[SparseVectorData]:
        return [SparseVectorData(indices=[1], values=[1.0]) for _ in texts]

    async def aembed_query(self, text: str) -> SparseVectorData:
        return SparseVectorData(indices=[1], values=[1.0])


def _store(dense: _Dense | None = None) -> QdrantHybridStore:
    return QdrantHybridStore(
        client=AsyncQdrantClient(location=":memory:"),
        collection_name=COLLECTION,
        dense_embedder=dense or _Dense(),  # type: ignore[arg-type]
        sparse_embedder=_Sparse(),  # type: ignore[arg-type]
        is_local_mode=True,
    )


def _doc(doc_id: str, text: str) -> LoadedDocument:
    return LoadedDocument(
        doc_id=doc_id,
        title=doc_id,
        content=text,
        source_path=f"inline://{doc_id}",
        doc_type="markdown",
        metadata={"language": "en"},
    )


async def _index(store: QdrantHybridStore, *docs: LoadedDocument, **kwargs: Any) -> Any:
    return await index_documents(list(docs), store, source="test", **kwargs)


async def _doc_ids(store: QdrantHybridStore) -> set[str]:
    return {str(doc["doc_id"]) for doc in await store.list_documents()}


async def _alias_target(store: QdrantHybridStore) -> str | None:
    aliases = (await store.client.get_aliases()).aliases
    return next((a.collection_name for a in aliases if a.alias_name == COLLECTION), None)


async def _collections(store: QdrantHybridStore) -> set[str]:
    return {c.name for c in (await store.client.get_collections()).collections}


@pytest.mark.asyncio
async def test_replacing_the_collection_swaps_in_the_new_documents() -> None:
    store = _store()
    try:
        first = await _index(store, _doc("old-guide", "Old guide text."))
        assert first.collection_created is True
        old_copy = await _alias_target(store)
        assert old_copy is not None
        assert old_copy.startswith(f"{COLLECTION}__")

        report = await _index(store, _doc("new-guide", "New guide text."), recreate=True)

        assert report.collection_created is True
        assert await _doc_ids(store) == {"new-guide"}
        assert report.total_collection_points == await store.count_points()
        new_copy = await _alias_target(store)
        assert new_copy is not None
        assert new_copy != old_copy
        assert await _collections(store) == {new_copy}
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_a_failed_replace_leaves_the_current_collection_untouched() -> None:
    """Replacing used to delete the collection first, so a failure left it empty (backlog #4)."""
    store = _store()
    try:
        await _index(store, _doc("old-guide", "Old guide text."))
        live_copy = await _alias_target(store)
        points_before = await store.count_points()

        # The first batch is stored in the new copy before the second one fails.
        with pytest.raises(RuntimeError, match="embedding model crashed"):
            await _index(
                store,
                _doc("new-guide", "New guide text."),
                _doc("bad-scan", "BROKEN page"),
                recreate=True,
                batch_size=1,
            )

        assert await _doc_ids(store) == {"old-guide"}
        assert await store.count_points() == points_before
        assert await _alias_target(store) == live_copy
        assert await _collections(store) == {live_copy}
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_a_failed_alias_switch_keeps_the_current_collection_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store()
    try:
        await _index(store, _doc("old-guide", "Old guide text."))
        live_copy = await _alias_target(store)

        async def _refuse(*args: Any, **kwargs: Any) -> bool:
            raise RuntimeError("qdrant unavailable")

        monkeypatch.setattr(store.client, "update_collection_aliases", _refuse)
        with pytest.raises(RuntimeError, match="qdrant unavailable"):
            await _index(store, _doc("new-guide", "New guide text."), recreate=True)

        assert await _alias_target(store) == live_copy
        assert await _doc_ids(store) == {"old-guide"}
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_the_first_replace_migrates_a_collection_made_before_aliases() -> None:
    store = _store()
    try:
        # Earlier versions stored the collection under its own name, with no alias.
        await store._create_backing_collection(COLLECTION)
        await _index(store, _doc("old-guide", "Old guide text."))
        assert await _alias_target(store) is None

        with pytest.raises(RuntimeError, match="embedding model crashed"):
            await _index(store, _doc("bad-scan", "BROKEN page"), recreate=True)
        assert await _alias_target(store) is None
        assert await _doc_ids(store) == {"old-guide"}

        await _index(store, _doc("new-guide", "New guide text."), recreate=True)
        new_copy = await _alias_target(store)
        assert new_copy is not None
        assert new_copy.startswith(f"{COLLECTION}__")
        assert await _collections(store) == {new_copy}
        assert await _doc_ids(store) == {"new-guide"}
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_replace_keeps_a_collection_it_did_not_create() -> None:
    store = _store()
    try:
        await store._create_backing_collection("hand_made")
        await store.client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name="hand_made",
                        alias_name=COLLECTION,
                    )
                )
            ]
        )

        await _index(store, _doc("new-guide", "New guide text."), recreate=True)

        new_copy = await _alias_target(store)
        assert new_copy is not None
        assert new_copy.startswith(f"{COLLECTION}__")
        assert await _collections(store) == {new_copy, "hand_made"}
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_an_upload_during_a_replace_lands_in_the_new_collection() -> None:
    dense = _Dense()
    store = _store(dense)
    try:
        await _index(store, _doc("old-guide", "Old guide text."))
        replace = asyncio.create_task(
            _index(store, _doc("new-guide", "SLOW new guide text."), recreate=True)
        )
        await asyncio.wait_for(dense.slow_started.wait(), timeout=5)

        upload = asyncio.create_task(_index(store, _doc("late-note", "Late note text.")))
        # Without the write lock the upload would finish here, in the copy about to be dropped.
        await asyncio.sleep(0.05)
        assert not upload.done()

        dense.release_slow.set()
        await asyncio.wait_for(replace, timeout=5)
        await asyncio.wait_for(upload, timeout=5)
        assert await _doc_ids(store) == {"new-guide", "late-note"}
    finally:
        await store.close()
