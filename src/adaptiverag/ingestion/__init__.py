"""Document loading, structure-aware chunking, and Qdrant indexing pipelines."""

from adaptiverag.ingestion.chunker import (
    ChunkingConfig,
    DocumentChunk,
    chunk_document,
    chunk_documents,
    count_tokens,
)
from adaptiverag.ingestion.indexer import IndexingReport, index_corpus
from adaptiverag.ingestion.loader import (
    LoadedDocument,
    detect_document_language,
    load_directory,
    load_document,
)

__all__ = [
    "ChunkingConfig",
    "DocumentChunk",
    "IndexingReport",
    "LoadedDocument",
    "chunk_document",
    "chunk_documents",
    "count_tokens",
    "detect_document_language",
    "index_corpus",
    "load_directory",
    "load_document",
]
