# AdaptiveRAG

> **Query-aware Agentic RAG engine** built with **LangGraph**, **Pydantic v2**, **Qdrant** (native Dense + BM25 Hybrid Search & RRF), local **FastEmbed ONNX** multilingual embeddings & cross-encoder reranking (**TR + EN**), 3-stage Dynamic Top-K selection, multi-signal CRAG confidence gating, web search fallback, two-tier citation validation, FastAPI SSE streaming, and a **Glass-Box Streamlit Inspector**.

![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-1C3C3C)
![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant%20Hybrid%20RRF-DC244C)
![FastAPI](https://img.shields.io/badge/API-FastAPI%20%2B%20SSE-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit%20Glass--Box-FF4B4B?logo=streamlit&logoColor=white)
![Pydantic v2](https://img.shields.io/badge/Schemas-Pydantic%20v2-E92063)

---

## Why AdaptiveRAG?

Static ("one-size-fits-all") RAG pipelines fail in production along three predictable failure modes:

1. **Over-Retrieval on Simple / Conversational Turns:** Greetings, meta-instructions, and single-parameter lookups pay the same retrieval, reranking, and context-window tax as complex synthesis questions.
2. **Under-Retrieval on Analytical Comparisons:** Single-query dense search with a fixed `top_k=5` drops exact technical identifiers (e.g., `ef_construct`, `Modifier.IDF`, `max_num_batched_tokens`) and truncates cross-document trade-off evidence.
3. **Silent Hallucination & Stale Knowledge:** When local retrieval returns low-relevance passages—or the user asks for fresh/out-of-corpus information—naive pipelines still emit confident, uncited, or fabricated answers.

**AdaptiveRAG** combines deterministic Information Retrieval engineering with a **bounded cyclic LangGraph state machine** so every query dynamically selects its optimal path—and exposes every internal decision transparently over SSE and the Glass-Box UI.

---

## Architecture & Bounded Cyclic LangGraph Workflow

```mermaid
flowchart TD
    START(["User Query + Chat History"]) --> CLASSIFY["1. classify_and_plan (Query Router & Planner)"]

    CLASSIFY -->|"conversational (needs_retrieval = false)"| CONV["conversational_respond (Zero-Retrieval Fast Path)"]
    CONV --> END_NODE(["FinalAnswer + Glass-Box Trace"])

    CLASSIFY -->|"factual (single-query)"| RETRIEVE["3. hybrid_retrieve (Qdrant Dense + BM25 + RRF)"]
    CLASSIFY -->|"analytical (multi-hop)"| DECOMPOSE["2. decompose_queries (2-4 Atomic Sub-Queries)"]
    DECOMPOSE --> RETRIEVE

    RETRIEVE --> RERANK["4. rerank_and_select_k (Cross-Encoder + 3-Stage Dynamic Top-K)"]
    RERANK --> CONF["5. score_confidence (Multi-Signal + Ambiguous-Band CRAG)"]

    CONF -->|"HIGH (C >= 0.72) or SUFFICIENT"| GEN["7. generate_answer (Grounded Inline Citations)"]
    CONF -->|"MEDIUM / PARTIAL and rewrites < 1"| REWRITE["6a. corrective_rewrite (Expand / Rephrase Query)"]
    REWRITE --> RETRIEVE

    CONF -->|"LOW (C < 0.38) or rewrites exhausted"| FALLBACK["6b. web_fallback_search (Tavily / DuckDuckGo)"]
    FALLBACK -->|"Web evidence found"| GEN
    FALLBACK -->|"Still insufficient / disabled"| ABSTAIN["abstain_response (Honest TR/EN Abstention)"]
    ABSTAIN --> END_NODE

    GEN --> CITE["8. validate_citations (Tier-1 Regex/ID + Tier-2 Entailment)"]
    CITE -->|"Passed or regens >= 1"| END_NODE
    CITE -->|"Unsupported claim / phantom ID"| REGEN["9. regenerate_strict (Strict Evidence Repair)"]
    REGEN --> CITE
```

---

## Core Engineering Capabilities

| Pillar | Implementation Highlights |
| :--- | :--- |
| **1. Query Classification & Strategy Matrix** | Structured `QueryPlan` routes into `conversational` (zero retrieval), `factual` (sparse-leaning single-query, strict `k=2..5`), or `analytical` (2–4 concurrent sub-queries, permissive `k=4..10`). Tunable without code edits in [`configs/strategies.yaml`](configs/strategies.yaml). |
| **2. Native Qdrant Hybrid Search** | Single collection with Named Vectors: local multilingual dense (`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, 768d) + sparse lexical (`Qdrant/bm25`) with server-side `Modifier.IDF`, parallel `Prefetch`, and `FusionQuery(Fusion.RRF)`. |
| **3. Cross-Encoder Reranking (`$0` ONNX)** | Local `BAAI/bge-reranker-v2-m3-int8` (TR + EN cross-lingual support) with numerically stable sigmoid calibration $\sigma(z) \in [0, 1]$ and `asyncio.to_thread` offloading so the FastAPI event loop never blocks. |
| **4. 3-Stage Dynamic Top-K Cutoff** | Replaces static `top_k` with: **(1)** Relevance floor $s_i \ge \tau_{\min}$, **(2)** Relative drop & elbow gap $s_i \ge \alpha \cdot s_1 \land (s_{i-1} - s_i) \le \delta_{\text{gap}}$, and **(3)** `[rerank_min_k, rerank_max_k]` + `max_context_tokens` guard. |
| **5. Multi-Signal Confidence + CRAG** | Computes $C_{\text{det}} = 0.50\,S_{\text{top1}} + 0.30\,S_{\text{mean}} + 0.20\,S_{\text{cov}}$. Auto-passes `HIGH` ($\ge 0.72$) and auto-routes `LOW` ($< 0.38$) at `$0` LLM cost; invokes the lightweight CRAG grader *only* inside the ambiguous band $[0.38, 0.72)$. |
| **6. Web Search Fallback & Honest Abstention** | `FallbackRetriever` protocol backed by **Tavily** (primary) and **DuckDuckGo** (zero-key fallback), mapping web snippets into `[W1]..[Wn]` `EvidenceChunk`s. If local + web evidence still cannot ground the query, returns an honest TR/EN abstention. |
| **7. Two-Tier Citation Validation** | **Tier-1 (`0 ms`, `$0`):** Deterministic regex check catching phantom IDs (`[C9]`) and uncited factual sentences. **Tier-2:** Claim-level entailment (`SUPPORTED` / `PARTIALLY_SUPPORTED` / `UNSUPPORTED` / `CONTRADICTED`). Triggers at most 1 `regenerate_strict` repair pass when `claim_support_rate < 0.85`. |
| **8. FinOps & Production Guardrails** | Tiered LLM routing (`qwen/qwen3-30b-a3b-instruct-2507` fast layer + `deepseek/deepseek-v3.2` synthesis layer via OpenRouter, easily swappable to OpenAI / Gemini / Anthropic / Ollama), local `SQLiteLLMCache`, hard `CostTracker(max_cost_usd=...)` cap (`HTTP 402`), and optional Langfuse tracing. |

---

## Quickstart

### 1. Prerequisites
- Python **3.12+** and [`uv`](https://docs.astral.sh/uv/)
- Docker & Docker Compose (for Qdrant and optional Langfuse observability)
- An LLM API key (defaults to [OpenRouter](https://openrouter.ai/) for low-cost tiered models; OpenAI, Gemini, Anthropic, and local Ollama are also supported)

### 2. Install Dependencies

```bash
git clone https://github.com/emirhansilsupur/Adaptive-RAG.git
cd Adaptive-RAG

# Installs core engine + Streamlit UI + RAGAS/eval + dev tools
make install
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and set your provider key (minimal configuration for OpenRouter):

```dotenv
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-...
LLM_FAST_MODEL=qwen/qwen3-30b-a3b-instruct-2507
LLM_GENERATION_MODEL=deepseek/deepseek-v3.2

# Optional: Tavily primary web fallback (DuckDuckGo works automatically without a key)
TAVILY_API_KEY=
WEB_FALLBACK_ENABLED=true

# FinOps budget & prompt cache guardrails
LLM_CACHE_ENABLED=true
MAX_RUN_COST_USD=0.50
```

### 4. Start Qdrant & Ingest the Multilingual Technical Corpus

The repository ships with **10 structure-aware technical Markdown documents** (8 EN + 2 TR covering Qdrant HNSW/Hybrid RRF, LangGraph, vLLM PagedAttention, FastAPI Async/SSE, PostgreSQL MVCC/`pgvector`, CRAG/Self-RAG, RAGAS/IR metrics, and TR+EN FinOps RAG patterns) in [`data/raw/`](data/raw/).

```bash
# Start Qdrant vector database (localhost:6333)
make up

# Optional: start Qdrant + Langfuse observability stack (localhost:3000)
# make up-full

# Chunk and idempotently index all documents (UUIDv5 chunk IDs -> zero duplicates on re-run)
make ingest
```

### 5. Launch FastAPI Backend & Streamlit Glass-Box Inspector

In terminal 1:
```bash
make api
# FastAPI + OpenAPI docs available at http://localhost:8000/docs
```

In terminal 2:
```bash
make ui
# Streamlit Glass-Box RAG Inspector available at http://localhost:8501
```

---

## API & Live SSE Streaming

AdaptiveRAG exposes four versioned REST/SSE endpoints:

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/v1/health` | `GET` | Non-crashing Qdrant status (`ok` / `collection_missing` / `unreachable`), LLM readiness, active models, and web fallback providers |
| `/v1/ingest` | `POST` | Idempotently index a local corpus directory (`corpus_dir`) or inline documents (`documents`) |
| `/v1/query` | `POST` | Synchronous full `FinalAnswer` JSON with ablation preset overrides and optional `max_cost_usd` (`HTTP 402` if exceeded) |
| `/v1/query/stream` | `POST` | Server-Sent Events (`sse-starlette`) streaming `event: node_update` per LangGraph node followed by `event: final_answer` |

### Example: Synchronous Query with Ablation Preset

```bash
curl -s http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare Qdrant Filterable HNSW and PostgreSQL pgvector for hybrid RAG.",
    "ablation_preset": "full",
    "max_cost_usd": 0.05
  }' | jq '{query_type, confidence: .confidence.level, k: (.evidence_chunks | length), citation_passed: .citation_report.passed, cost_usd: .usage.estimated_cost_usd}'
```

### Example: Live SSE Pipeline Trace Stream

```bash
curl -N http://localhost:8000/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Qdrant ta ef_construct varsayilan degeri nedir?",
    "ablation_preset": "full"
  }'
```

### Glass-Box Streamlit Inspector (`ui/app.py`)
The Streamlit UI streams live node transitions (`st.status`) from `/v1/query/stream`, supports switching between all 4 ablation presets (plus fine-grained feature toggles) on the fly, includes one-click TR/EN demo prompts and corpus re-indexing, and renders **5 inspection tabs**:
1. **Trace & Query Plan:** Node-by-node latency bar chart, routing rationale, rewritten/decomposed sub-queries.
2. **Dynamic Top-K & Rerank:** Interactive Plotly score waterfall showing calibrated cross-encoder scores, $\tau_{\min}$ relevance floor, $\alpha \cdot s_1$ cutoff line, and token budget utilization.
3. **Confidence & CRAG:** Gauge & multi-signal breakdown ($S_{\text{top1}}$, $S_{\text{mean}}$, $S_{\text{cov}}$, $C_{\text{det}}$, CRAG verdict, and control action).
4. **Evidence Passages:** Ranked `[C1]..[Ck]` / `[W1]..[Wn]` chunks with `section_path`, scores, and source links.
5. **Citation Audit:** Tier-1 valid/phantom IDs, uncited sentence alerts, and Tier-2 claim-level entailment verdicts.

---

## Golden Evaluation Dataset & 4-Way Ablation Benchmark

### Stratified Multilingual Golden Set (`data/eval/golden_set.jsonl`)
AdaptiveRAG includes a **66-query TR + EN golden evaluation dataset** grounded in all 10 corpus documents (with ground-truth `expected_doc_ids` verified automatically in CI unit tests):

| Stratum | Count | Languages | What It Isolates |
| :--- | :---: | :---: | :--- |
| **`factual`** | 28 | 18 EN + 10 TR *(incl. TR→EN cross-lingual)* | Exact parameters, formulas, API flags (BM25 + Cross-Encoder precision & Dynamic Top-K token savings) |
| **`analytical`** | 18 | 11 EN + 7 TR | Multi-document trade-offs spanning 2–3 docs (Multi-Query decomposition + Context Recall / NDCG@k) |
| **`conversational`** | 6 | 3 EN + 3 TR | Zero-retrieval fast path (`needs_retrieval=false`, latency & token savings) |
| **`web_fallback`** | 7 | 4 EN + 3 TR | Fresh/time-sensitive queries requiring low-confidence detection + web search fallback |
| **`unanswerable`** | 7 | 4 EN + 3 TR | Fictional/out-of-domain queries requiring honest abstention instead of hallucination |

### 4-Way Ablation Matrix

| Preset | Retrieval | Routing & Multi-Query | Cross-Encoder Rerank | Dynamic Top-K | Confidence / CRAG / Fallback / Citation Validator |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`naive_dense`** | Dense only | Off | Off | Static (`k=5`) | Off *(Tier-1 citation telemetry only)* |
| **`hybrid_rerank`** | Dense + BM25 RRF | Off | On | Static (`k=5`) | Off *(Tier-1 citation telemetry only)* |
| **`adaptive_routing`** | Dense + BM25 RRF | On | On | 3-Stage Adaptive | Off *(Tier-1 citation telemetry only)* |
| **`full`** | Dense + BM25 RRF | On | On | 3-Stage Adaptive | **All Enabled** |

### Running Benchmarks

```bash
# 1. Zero-API-cost offline IR ablation (in-memory Qdrant seeded from data/raw, local ONNX models)
uv run python -m eval.run_benchmark --mode offline-ir --in-memory

# 2. Full 4-way live Agentic ablation (strict FinOps budget cap + SQLite cache)
uv run python -m eval.run_benchmark --mode live --in-memory --max-cost-usd 0.50

# 3. Live ablation + optional RAGAS LLM-as-a-Judge (Faithfulness, AnswerRelevancy, Context Precision/Recall)
uv run python -m eval.run_benchmark --mode live --in-memory --limit 16 --with-ragas --max-cost-usd 0.50

# 4. Run against external JSONL or HuggingFace QA datasets (HotpotQA / RAGBench adapter)
uv run python -m eval.run_benchmark --mode offline-ir --in-memory \
  --hf-dataset hotpot_qa --hf-subset distractor --limit 25
```

Every run writes machine-readable JSON (`benchmark_report.json`), a GitHub-flavored Markdown scorecard (`benchmark_summary.md`), and an interactive Plotly bar chart (`benchmark_chart.html`) to `eval/reports/`.

---

## Project Structure

```text
adaptiverag/
├── configs/
│   └── strategies.yaml              # Declarative routing, dynamic-k, confidence & citation thresholds
├── data/
│   ├── raw/                         # 10 multilingual AI/ML & Systems Engineering Markdown docs (8 EN + 2 TR)
│   └── eval/
│       └── golden_set.jsonl         # 66-query stratified TR+EN golden benchmark dataset (5 strata)
├── src/adaptiverag/
│   ├── config.py                    # Pydantic Settings + YAML strategy loader
│   ├── schemas/                     # Pure Pydantic v2 contracts (query, retrieval, citation, response)
│   ├── llm/                         # Multi-provider factory, resilient structured output, SQLite cache, CostTracker
│   ├── ingestion/                   # Markdown/PDF loader, structure-aware chunker, idempotent Qdrant indexer
│   ├── retrieval/                   # FastEmbed Dense/BM25, Async Qdrant Hybrid RRF, Cross-Encoder, Dynamic Top-K, Web Fallback
│   ├── verification/                # Multi-signal Confidence + CRAG grader, Two-Tier Citation Validator
│   ├── graph/                       # LangGraph AgentState, AblationConfig, prompts, 12 async nodes, conditional edges, workflow
│   ├── observability/               # Structlog JSON/console logging + Langfuse callback
│   └── api/                         # FastAPI app, lifespan container, REST + SSE streaming routes
├── ui/
│   └── app.py                       # Streamlit Glass-Box RAG Inspector (live SSE + 5 Plotly/audit tabs)
├── eval/
│   ├── schemas.py                   # GoldenQuery, IRScores, RagasScores, BenchmarkReport schemas
│   ├── datasets.py                  # Golden JSONL loader + flexible external JSONL & HuggingFace (HotpotQA/RAGBench) adapters
│   ├── metrics.py                   # Recall@k, Precision@k, MRR, NDCG@k, percentiles, Markdown/Plotly report generators
│   ├── ragas_runner.py              # RAGAS v0.4+ LLM-as-a-Judge wrapper with local $0 FastEmbed embeddings
│   └── run_benchmark.py             # CLI & async 4-way ablation runner (offline-ir + live + budget guardrails)
├── tests/unit/                      # Deterministic $0 unit & graph test suite (zero paid API calls)
├── docker-compose.yml               # Qdrant + optional Langfuse observability profile
├── Makefile                         # install, up, ingest, api, ui, test, lint, typecheck, eval
└── pyproject.toml                   # uv project config, ruff, mypy, pytest settings
```

---

## Testing & Quality Gates

All automated tests run **100% offline at `$0` API cost** using deterministic embeddings/reranker stubs, in-memory Qdrant, and scripted fake LLMs:

```bash
# Run full unit & end-to-end graph/API/eval test suite
make test

# Static lint & formatting verification (src, tests, eval, ui)
make lint

# Static type checking (strict untyped-defs enforcement on src/)
make typecheck
```
