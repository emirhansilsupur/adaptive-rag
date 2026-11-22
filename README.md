# AdaptiveRAG

Query-aware **Agentic RAG** engine built with **LangGraph**, **Pydantic v2**, **Qdrant** (native Dense + BM25 Hybrid Search & RRF), local **FastEmbed** multilingual embeddings & cross-encoder reranking (TR + EN), multi-signal confidence gating, web search fallback, and two-tier citation validation.

> Status: Phase 2 complete (Core schemas & FinOps LLM layer + 10-doc TR/EN corpus, structure-aware chunking, FastEmbed multilingual embeddings, native async Qdrant Hybrid RRF store & idempotent indexer).

## Quickstart

```bash
cp .env.example .env
make install
make up
make test-unit
```
