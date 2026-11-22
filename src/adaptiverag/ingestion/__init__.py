"""Document loading, Firecrawl web/docs scraping, structure-aware chunking, and Qdrant indexing."""

from adaptiverag.ingestion.chunker import (
    ChunkingConfig,
    DocumentChunk,
    chunk_document,
    chunk_documents,
    count_tokens,
)
from adaptiverag.ingestion.indexer import (
    IndexingReport,
    index_corpus,
    index_documents,
    index_web_crawl,
    index_web_urls,
)
from adaptiverag.ingestion.loader import (
    LoadedDocument,
    detect_document_language,
    load_directory,
    load_document,
)
from adaptiverag.ingestion.web_loader import (
    FirecrawlClient,
    FirecrawlError,
    FirecrawlUnavailableError,
    UnsafeURLError,
    make_web_doc_id,
    validate_public_url,
)

__all__ = [
    "ChunkingConfig",
    "DocumentChunk",
    "FirecrawlClient",
    "FirecrawlError",
    "FirecrawlUnavailableError",
    "IndexingReport",
    "LoadedDocument",
    "UnsafeURLError",
    "chunk_document",
    "chunk_documents",
    "count_tokens",
    "detect_document_language",
    "index_corpus",
    "index_documents",
    "index_web_crawl",
    "index_web_urls",
    "load_directory",
    "load_document",
    "make_web_doc_id",
    "validate_public_url",
]
