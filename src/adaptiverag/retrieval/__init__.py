"""Multilingual embeddings, Qdrant hybrid retrieval, reranking, dynamic top-k, and fallback."""

from adaptiverag.retrieval.dynamic_k import assign_citation_ids, select_dynamic_top_k
from adaptiverag.retrieval.embeddings import (
    DenseEmbedder,
    FastEmbedDenseEmbedder,
    FastEmbedSparseEmbedder,
    SparseEmbedder,
    SparseVectorData,
)
from adaptiverag.retrieval.fallback import (
    DuckDuckGoSearchProvider,
    FallbackRetriever,
    TavilySearchProvider,
    WebFallbackRetriever,
    WebSearchProvider,
    WebSearchResult,
)
from adaptiverag.retrieval.parent_child import expand_to_parent_chunks
from adaptiverag.retrieval.qdrant_store import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    QdrantHybridStore,
    RetrievalMode,
)
from adaptiverag.retrieval.reranker import (
    FastEmbedCrossEncoderReranker,
    Reranker,
    format_passage_for_rerank,
    sigmoid,
)

__all__ = [
    "DENSE_VECTOR_NAME",
    "SPARSE_VECTOR_NAME",
    "DenseEmbedder",
    "DuckDuckGoSearchProvider",
    "FallbackRetriever",
    "FastEmbedCrossEncoderReranker",
    "FastEmbedDenseEmbedder",
    "FastEmbedSparseEmbedder",
    "QdrantHybridStore",
    "Reranker",
    "RetrievalMode",
    "SparseEmbedder",
    "SparseVectorData",
    "TavilySearchProvider",
    "WebFallbackRetriever",
    "WebSearchProvider",
    "WebSearchResult",
    "assign_citation_ids",
    "expand_to_parent_chunks",
    "format_passage_for_rerank",
    "select_dynamic_top_k",
    "sigmoid",
]
