# AdaptiveRAG

Query-aware **Agentic RAG** engine built with **LangGraph**, **Pydantic v2**, **Qdrant** (native Dense + BM25 Hybrid Search & RRF), local **FastEmbed** multilingual embeddings & cross-encoder reranking (TR + EN), multi-signal confidence gating, web search fallback, and two-tier citation validation.

> Status: Phase 1 (Core foundation, schemas, OpenRouter/multi-provider LLM layer, FinOps cache & budget guardrails) initialized.

## Quickstart

```bash
cp .env.example .env
make install
make up
make test-unit
```
