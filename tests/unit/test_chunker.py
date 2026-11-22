"""Unit tests for document loading and structure-aware Markdown chunking."""

from __future__ import annotations

import uuid
from pathlib import Path

from adaptiverag.ingestion import (
    ChunkingConfig,
    LoadedDocument,
    chunk_document,
    chunk_documents,
    detect_document_language,
    load_directory,
)


def test_load_curated_corpus_detects_languages_and_titles() -> None:
    docs = load_directory(Path("data/raw"))
    assert len(docs) == 10

    by_id = {d.doc_id: d for d in docs}
    assert "01_qdrant_architecture_and_hnsw" in by_id
    assert by_id["01_qdrant_architecture_and_hnsw"].language == "en"
    assert "Qdrant" in by_id["01_qdrant_architecture_and_hnsw"].title

    assert by_id["09_tr_rag_uretim_mimarisi_ve_optimizasyon"].language == "tr"
    assert by_id["10_tr_vektor_veritabani_ve_hibrit_arama_notlari"].language == "tr"


def test_detect_document_language_handles_tr_and_en() -> None:
    assert (
        detect_document_language("Bu doküman Türkçe karakterler içerir: ğ, ş, ı, ö, ü, ç.") == "tr"
    )
    assert detect_document_language("Qdrant uses HNSW graphs and BM25 sparse vectors.") == "en"


def test_chunker_preserves_heading_breadcrumbs_and_ignores_code_comments() -> None:
    md = """# System Architecture

Intro paragraph explaining the top-level design.

## Indexing Layer

### HNSW Parameters

```python
# This comment must NOT become a heading!
## Neither should this comment!
hnsw_m = 16
ef_construct = 100
```

Tuning `ef_construct` improves recall without adding query RAM overhead.
"""
    doc = LoadedDocument(
        doc_id="arch_doc",
        title="System Architecture",
        content=md,
        source_path="arch_doc.md",
        doc_type="markdown",
        language="en",
    )

    chunks = chunk_document(doc, ChunkingConfig(max_tokens=256, overlap_tokens=32))
    assert len(chunks) >= 2

    code_chunks = [c for c in chunks if c.metadata.get("has_code_block")]
    assert len(code_chunks) == 1
    code_chunk = code_chunks[0]
    assert code_chunk.section_path == "System Architecture > Indexing Layer > HNSW Parameters"
    assert "# This comment must NOT become a heading!" in code_chunk.text
    assert code_chunk.text.count("```") == 2
    assert "[System Architecture | System Architecture > Indexing Layer > HNSW Parameters]" in (
        code_chunk.embedding_text
    )


def test_chunker_keeps_tables_intact_and_repeats_header_when_oversized() -> None:
    rows = "\n".join(
        f"| `param_{i}` | `{i * 10}` | Detailed configuration explanation for parameter number {i} in production systems |"
        for i in range(1, 25)
    )
    md = f"""# Reference Table

## Parameters

| Parameter | Default | Notes |
| :--- | :--- | :--- |
{rows}
"""
    doc = LoadedDocument(
        doc_id="table_doc",
        title="Reference Table",
        content=md,
        source_path="table_doc.md",
        doc_type="markdown",
        language="en",
    )

    chunks = chunk_document(
        doc, ChunkingConfig(max_tokens=140, overlap_tokens=0, min_chunk_tokens=10)
    )
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.metadata.get("has_table") is True
        assert "| Parameter | Default | Notes |" in chunk.text
        assert "| :--- | :--- | :--- |" in chunk.text


def test_chunk_ids_are_valid_deterministic_uuids() -> None:
    docs = load_directory(Path("data/raw"))
    first_run = chunk_documents(docs, ChunkingConfig(max_tokens=512, overlap_tokens=64))
    second_run = chunk_documents(docs, ChunkingConfig(max_tokens=512, overlap_tokens=64))

    assert len(first_run) > 20
    assert [c.chunk_id for c in first_run] == [c.chunk_id for c in second_run]
    # All chunk IDs must be unique and valid UUIDs for Qdrant PointId compatibility
    assert len({c.chunk_id for c in first_run}) == len(first_run)
    for chunk in first_run:
        parsed = uuid.UUID(chunk.chunk_id)
        assert parsed.version == 5
        assert chunk.token_count > 0
