"""Listing documents reads one point per document instead of every chunk in the collection."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from qdrant_client import AsyncQdrantClient, models

from adaptiverag.api.dependencies import AppContainer
from adaptiverag.api.main import create_app
from adaptiverag.config import Settings, load_strategies
from adaptiverag.ingestion.chunker import ChunkingConfig, DocumentChunk, chunk_documents
from adaptiverag.ingestion.indexer import index_documents
from adaptiverag.ingestion.loader import LoadedDocument
from adaptiverag.retrieval.embeddings import SparseVectorData
from adaptiverag.retrieval.qdrant_store import DOC_TOKENS_PAYLOAD_KEY, QdrantHybridStore

COLLECTION = "kb"
FLAT = ChunkingConfig(max_tokens=64, overlap_tokens=0, min_chunk_tokens=1)
PARENT_CHILD = ChunkingConfig(
    max_tokens=64,
    overlap_tokens=0,
    min_chunk_tokens=1,
    parent_child_enabled=True,
    parent_max_tokens=200,
    child_max_tokens=64,
    child_overlap_tokens=16,
)


class _Dense:
    model_name = "stub-dense"
    dimension = 4

    async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


class _Sparse:
    model_name = "stub-bm25"

    async def aembed_documents(self, texts: Sequence[str]) -> list[SparseVectorData]:
        return [SparseVectorData(indices=[1], values=[1.0]) for _ in texts]

    async def aembed_query(self, text: str) -> SparseVectorData:
        return SparseVectorData(indices=[1], values=[1.0])


def _store() -> QdrantHybridStore:
    return QdrantHybridStore(
        client=AsyncQdrantClient(location=":memory:"),
        collection_name=COLLECTION,
        dense_embedder=_Dense(),  # type: ignore[arg-type]
        sparse_embedder=_Sparse(),  # type: ignore[arg-type]
        is_local_mode=True,
    )


def _doc(doc_id: str, *, sections: int = 3, paragraphs: int = 1) -> LoadedDocument:
    body = "\n\n".join(
        f"## Part {s}\n\n"
        + "\n\n".join(
            f"{doc_id} part {s}.{p}: ef_construct, m and ef decide how densely HNSW links "
            "each vector and how many candidates a search visits before it stops."
            for p in range(paragraphs)
        )
        for s in range(sections)
    )
    return LoadedDocument(
        doc_id=doc_id,
        title=f"Guide {doc_id}",
        content=f"# Guide {doc_id}\n\n{body}\n",
        source_path=f"inline://{doc_id}",
        doc_type="markdown",
        metadata={"language": "en"},
    )


async def _index_mixed(store: QdrantHybridStore) -> list[DocumentChunk]:
    """Store two flat documents and a parent-child one; return every chunk stored."""
    flat = [_doc("flat_a"), _doc("flat_b", sections=4)]
    nested = [_doc("nested_c", sections=2, paragraphs=4)]
    await index_documents(flat, store, source="test", chunk_config=FLAT)
    await index_documents(nested, store, source="test", chunk_config=PARENT_CHILD)
    chunks = chunk_documents(flat, config=FLAT) + chunk_documents(nested, config=PARENT_CHILD)
    # Children of one parent overlap, which is what the token totals must not double count.
    assert max(Counter(c.parent_id for c in chunks if c.parent_id).values()) >= 2
    assert len(chunks) > 2 * 3
    return chunks


def _expected(chunks: Sequence[DocumentChunk]) -> dict[str, tuple[int, int, str]]:
    """Chunk count, token total and strategy per document, added up from its chunks."""
    by_doc: dict[str, list[DocumentChunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)
    expected: dict[str, tuple[int, int, str]] = {}
    for doc_id, doc_chunks in by_doc.items():
        if doc_chunks[0].parent_id:
            parents = {c.parent_id: c.parent_token_count or c.token_count for c in doc_chunks}
            expected[doc_id] = (len(doc_chunks), sum(parents.values()), "parent_child")
        else:
            tokens = sum(c.token_count for c in doc_chunks)
            expected[doc_id] = (len(doc_chunks), tokens, "flat")
    return expected


def _listed(docs: list[dict[str, Any]]) -> dict[str, tuple[int, int, str]]:
    return {d["doc_id"]: (d["chunk_count"], d["total_tokens"], d["chunk_strategy"]) for d in docs}


def _count_points_read(store: QdrantHybridStore, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record how many points each scroll returns from now on."""
    reads: list[int] = []
    scroll = store.client.scroll

    async def counting_scroll(*args: Any, **kwargs: Any) -> Any:
        records, offset = await scroll(*args, **kwargs)
        reads.append(len(records))
        return records, offset

    monkeypatch.setattr(store.client, "scroll", counting_scroll)
    return reads


async def test_listing_reads_one_point_per_document(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store()
    chunks = await _index_mixed(store)
    reads = _count_points_read(store, monkeypatch)

    listed = await store.list_documents()

    assert _listed(listed) == _expected(chunks)
    assert [d["doc_id"] for d in listed] == ["flat_a", "flat_b", "nested_c"]
    assert listed[0]["title"] == "Guide flat_a"
    assert listed[0]["source_path"] == "inline://flat_a"
    assert sum(reads) == 3  # instead of every chunk
    await store.close()


@pytest.mark.parametrize(
    "older_docs",
    [{"flat_b", "nested_c"}, {"flat_a", "flat_b", "nested_c"}],
    ids=["some-older", "all-older"],
)
async def test_documents_stored_before_token_totals_are_added_up_from_their_chunks(
    monkeypatch: pytest.MonkeyPatch, older_docs: set[str]
) -> None:
    store = _store()
    chunks = await _index_mixed(store)
    # As if an earlier version stored these documents: no chunk holds a token total.
    await store.client.delete_payload(
        collection_name=COLLECTION,
        keys=[DOC_TOKENS_PAYLOAD_KEY],
        points=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=[*older_docs]))]
            )
        ),
    )
    reads = _count_points_read(store, monkeypatch)

    listed = await store.list_documents()

    assert _listed(listed) == _expected(chunks)
    newer_heads = 3 - len(older_docs)
    older_chunks = sum(1 for c in chunks if c.doc_id in older_docs)
    assert sum(reads) == newer_heads + older_chunks
    await store.close()


async def test_listing_reads_every_chunk_when_qdrant_cannot_count_by_doc_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store()
    chunks = await _index_mixed(store)

    async def no_doc_id_index(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Wrong input: No appropriate index for faceting: `doc_id`")

    monkeypatch.setattr(store.client, "facet", no_doc_id_index)
    reads = _count_points_read(store, monkeypatch)

    listed = await store.list_documents()

    assert _listed(listed) == _expected(chunks)
    assert sum(reads) == len(chunks)
    await store.close()


async def test_replacing_a_document_updates_its_listing() -> None:
    store = _store()
    await index_documents([_doc("guide", sections=4)], store, source="test", chunk_config=FLAT)
    shorter = [_doc("guide", sections=2)]

    await index_documents(shorter, store, source="test", chunk_config=FLAT)

    assert _listed(await store.list_documents()) == _expected(chunk_documents(shorter, config=FLAT))
    await store.close()


async def test_the_api_lists_every_document_past_the_first_thousand(tmp_path: Path) -> None:
    store = _store()
    docs = [_doc(f"doc_{i:04d}", sections=1) for i in range(1001)]
    await index_documents(docs, store, source="test", chunk_config=FLAT)
    container = AppContainer(
        settings=Settings(_env_file=None, qdrant_collection=COLLECTION),  # type: ignore[call-arg]
        strategies=load_strategies(Path("configs/strategies.yaml")),
        store=store,
        fallback_retriever=None,  # type: ignore[arg-type]
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(container=container)),
        base_url="http://localhost",
    ) as client:
        response = await client.get("/v1/documents")

    assert response.status_code == 200
    body = response.json()
    assert body["total_documents"] == len(body["documents"]) == 1001
    assert body["total_chunks"] == await store.count_points()
    await store.close()
