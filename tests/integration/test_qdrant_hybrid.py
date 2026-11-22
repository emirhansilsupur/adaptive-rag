"""Tests for native Qdrant hybrid search (Dense + BM25 RRF), filtering, and multi-query fan-out."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from qdrant_client import AsyncQdrantClient

from adaptiverag.ingestion import ChunkingConfig, index_corpus
from adaptiverag.retrieval import QdrantHybridStore, RetrievalMode, SparseVectorData

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_çğıöşüÇĞİÖŞÜ]+")


class DeterministicDenseStub:
    """Offline dense embedder using character-trigram hashing into a unit-normalized vector."""

    def __init__(self, dim: int = 128) -> None:
        self._dim = dim

    @property
    def model_name(self) -> str:
        return "stub/char-trigram-dense"

    @property
    def dimension(self) -> int:
        return self._dim

    def _encode(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        normalized = " ".join(text.lower().split())
        if len(normalized) < 3:
            normalized = normalized.ljust(3)
        for i in range(len(normalized) - 2):
            gram = normalized[i : i + 3]
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=4).digest()
            idx = int.from_bytes(digest, "little") % self._dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._encode(query)

    async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, query: str) -> list[float]:
        return self.embed_query(query)


class DeterministicBM25Stub:
    """Offline BM25-like term-frequency sparse embedder paired with Qdrant Modifier.IDF."""

    @property
    def model_name(self) -> str:
        return "stub/bm25-lexical"

    def _encode(self, text: str) -> SparseVectorData:
        tokens = [tok.lower() for tok in _TOKEN_RE.findall(text) if len(tok) > 1]
        if not tokens:
            return SparseVectorData(indices=[], values=[])

        counts = Counter(tokens)
        index_weights: dict[int, float] = {}
        for tok, freq in counts.items():
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=4).digest()
            idx = int.from_bytes(digest, "little") % 2_000_000
            # Sublinear BM25-like TF saturation
            tf = (freq * 2.2) / (freq + 1.2)
            index_weights[idx] = index_weights.get(idx, 0.0) + tf

        sorted_items = sorted(index_weights.items())
        return SparseVectorData(
            indices=[k for k, _ in sorted_items],
            values=[v for _, v in sorted_items],
        )

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVectorData]:
        return [self._encode(t) for t in texts]

    def embed_query(self, query: str) -> SparseVectorData:
        return self._encode(query)

    async def aembed_documents(self, texts: Sequence[str]) -> list[SparseVectorData]:
        return self.embed_documents(texts)

    async def aembed_query(self, query: str) -> SparseVectorData:
        return self.embed_query(query)


class ConstantBlindDenseStub(DeterministicDenseStub):
    """Intentionally uninformative dense stub to verify sparse BM25 + RRF lexical recall."""

    def _encode(self, text: str) -> list[float]:
        return [1.0 / math.sqrt(self._dim)] * self._dim


async def _build_indexed_memory_store(
    *,
    dense: DeterministicDenseStub | None = None,
    sparse: DeterministicBM25Stub | None = None,
) -> QdrantHybridStore:
    store = QdrantHybridStore(
        client=AsyncQdrantClient(location=":memory:"),
        collection_name="test_adaptiverag_hybrid",
        dense_embedder=dense or DeterministicDenseStub(),
        sparse_embedder=sparse or DeterministicBM25Stub(),
        is_local_mode=True,
    )
    await index_corpus(
        Path("data/raw"),
        store,
        chunk_config=ChunkingConfig(max_tokens=512, overlap_tokens=64),
        recreate=True,
    )
    return store


async def test_idempotent_corpus_indexing_creates_no_duplicates() -> None:
    store = await _build_indexed_memory_store()
    try:
        initial_count = await store.count_points()
        assert initial_count >= 25

        # Re-indexing without recreate must preserve exact point count (zero duplicates)
        second_report = await index_corpus(
            Path("data/raw"),
            store,
            chunk_config=ChunkingConfig(max_tokens=512, overlap_tokens=64),
            recreate=False,
        )
        assert second_report.documents_loaded == 10
        assert second_report.collection_created is False
        assert second_report.total_collection_points == initial_count

        health = await store.health_check()
        assert health["status"] == "ok"
        assert health["points_count"] == initial_count
    finally:
        await store.close()


async def test_hybrid_rrf_recalls_exact_technical_identifiers_when_dense_is_blind() -> None:
    store = await _build_indexed_memory_store(dense=ConstantBlindDenseStub())
    try:
        # Sparse-only BM25 ranks exact PostgreSQL config key #1
        sparse_pg = await store.search_single(
            "autovacuum_vacuum_scale_factor",
            mode=RetrievalMode.SPARSE_ONLY,
            fusion_limit=3,
        )
        assert sparse_pg
        assert sparse_pg[0].doc_id == "06_postgresql_mvcc_and_indexing"
        assert "autovacuum_vacuum_scale_factor" in sparse_pg[0].text

        # Hybrid RRF pulls exact identifier into top-2 even when dense vectors are completely blind
        hybrid_pg = await store.search_single(
            "autovacuum_vacuum_scale_factor",
            mode=RetrievalMode.HYBRID,
            dense_limit=10,
            sparse_limit=10,
            fusion_limit=5,
        )
        assert hybrid_pg
        assert any(
            h.doc_id == "06_postgresql_mvcc_and_indexing"
            and "autovacuum_vacuum_scale_factor" in h.text
            for h in hybrid_pg[:2]
        )

        # Exact vLLM serving flags recalled into top-2 via BM25 branch of Hybrid RRF
        hybrid_vllm = await store.search_single(
            "enable_chunked_prefill max_num_batched_tokens",
            mode=RetrievalMode.HYBRID,
            dense_limit=10,
            sparse_limit=10,
            fusion_limit=5,
        )
        assert hybrid_vllm
        assert any(
            h.doc_id == "04_vllm_serving_and_paged_attention"
            and "max_num_batched_tokens" in h.text
            for h in hybrid_vllm[:2]
        )
    finally:
        await store.close()


async def test_metadata_filters_restrict_by_language_and_doc_id() -> None:
    store = await _build_indexed_memory_store()
    try:
        tr_hits = await store.search_single(
            "dinamik top-k eşik değeri ve maliyet optimizasyonu",
            mode=RetrievalMode.HYBRID,
            language="tr",
            fusion_limit=10,
        )
        assert tr_hits
        assert all(hit.metadata.get("language") == "tr" for hit in tr_hits)
        assert {hit.doc_id for hit in tr_hits}.issubset(
            {
                "09_tr_rag_uretim_mimarisi_ve_optimizasyon",
                "10_tr_vektor_veritabani_ve_hibrit_arama_notlari",
            }
        )

        scoped_hits = await store.search_single(
            "HNSW",
            mode=RetrievalMode.DENSE_ONLY,
            doc_ids=["01_qdrant_architecture_and_hnsw"],
            fusion_limit=5,
        )
        assert scoped_hits
        assert all(hit.doc_id == "01_qdrant_architecture_and_hnsw" for hit in scoped_hits)
    finally:
        await store.close()


async def test_multi_query_fanout_deduplicates_and_tracks_subquery_coverage() -> None:
    store = await _build_indexed_memory_store()
    try:
        sub_queries = [
            "Qdrant ef_construct full_scan_threshold HNSW parameters",
            "vLLM PagedAttention block_size chunked prefill TTFT",
            "PostgreSQL MVCC autovacuum_vacuum_scale_factor PgBouncer",
        ]
        results = await store.search_multi_query(
            sub_queries,
            mode=RetrievalMode.HYBRID,
            dense_limit=15,
            sparse_limit=15,
            fusion_limit=10,
            max_total_candidates=20,
        )

        assert results
        # Strictly unique chunk IDs after cross-query deduplication
        chunk_ids = [r.chunk_id for r in results]
        assert len(chunk_ids) == len(set(chunk_ids))

        # All three distinct domains should be represented in top candidates
        retrieved_docs = {r.doc_id for r in results}
        assert "01_qdrant_architecture_and_hnsw" in retrieved_docs
        assert "04_vllm_serving_and_paged_attention" in retrieved_docs
        assert "06_postgresql_mvcc_and_indexing" in retrieved_docs

        # Every sub-query should have at least one matched evidence chunk
        covered = {q for r in results for q in r.matched_sub_queries}
        assert covered == set(sub_queries)
    finally:
        await store.close()


async def test_punctuation_only_query_falls_back_cleanly() -> None:
    store = await _build_indexed_memory_store()
    try:
        hits = await store.search_single("???", mode=RetrievalMode.HYBRID, fusion_limit=3)
        assert len(hits) == 3
        empty_hits = await store.search_single("   ", mode=RetrievalMode.HYBRID, fusion_limit=3)
        assert empty_hits == []
    finally:
        await store.close()
