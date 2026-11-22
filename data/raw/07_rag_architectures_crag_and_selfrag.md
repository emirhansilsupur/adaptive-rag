# Evolution of RAG Architectures: Adaptive-RAG, CRAG & Self-RAG

Standard ("Naive") Retrieval-Augmented Generation executes an identical pipeline for every user input: embed the raw query, fetch a fixed `top_k` (typically `k=5`) using cosine similarity, concatenate chunks into a prompt, and generate an answer. In production, this one-size-fits-all pipeline fails along four predictable axes: unnecessary retrieval latency on conversational queries, missed exact keywords, insufficient context breadth on multi-hop comparisons, and confident hallucinations when local retrieval fails.

## Query-Aware Adaptive Routing (Adaptive-RAG)

Inspired by Jeong et al. (NAACL 2024, *Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity*), an adaptive router classifies incoming queries by complexity and retrieval need before executing any vector search:

1. **Conversational / No-Retrieval (`conversational`)**: Greetings, clarifications about prior assistant outputs, or pure meta-instructions bypass vector search completely (`needs_retrieval=False`), saving embedding latency, Qdrant I/O, and prompt tokens.
2. **Single-Step Factual (`factual`)**: Specific definitions, numeric defaults, CLI flags, or single-entity lookups execute a focused **Hybrid Search (Dense + BM25)** with high lexical weight, a strict **Dynamic Top-K** (`k=2..5`, steep `alpha_relative=0.65`), and Cross-Encoder reranking.
3. **Multi-Step Analytical (`analytical`)**: Comparative, architectural trade-off, or root-cause questions are decomposed into **2 to 4 distinct sub-queries**, fanned out concurrently across Qdrant, deduplicated, reranked, and granted a broader **Dynamic Top-K** window (`k=4..10`, permissive `alpha_relative=0.45`, higher token budget).

## Corrective RAG (CRAG)

Proposed by Yan et al. (2024, *Corrective Retrieval Augmented Generation*), CRAG recognizes that **bad retrieval poisons even the strongest LLM**. Instead of blindly stuffing retrieved chunks into the generator, CRAG inserts a lightweight **Retrieval Evaluator** that categorizes evidence quality into three bands:

| CRAG Verdict | Condition | Control-Flow Action |
| :--- | :--- | :--- |
| **Correct (`HIGH`)** | Retrieved local chunks clearly answer the query with high relevance | Proceed directly to grounded generation with inline citations |
| **Ambiguous (`MEDIUM`)** | Evidence is partial, noisy, or covers only some sub-queries | Run a **Corrective Query Rewrite** (max 1 retry) targeting missing aspects; if still insufficient, augment with web search |
| **Incorrect (`LOW`)** | Top reranked evidence falls below relevance floor (out-of-corpus or stale info) | Discard noisy local chunks and trigger **Web Search Fallback** (e.g., Tavily or DuckDuckGo) or safe **Abstention** if web fallback is disabled/empty |

### Combining Deterministic Signals with LLM CRAG Grading
Running an LLM grader on *every* query wastes latency and API budget. Production systems first compute a deterministic multi-signal confidence score from calibrated Cross-Encoder probabilities (`top1_score`, `mean_topk_score`, and `subquery_coverage`):
- Scores `>= 0.72` auto-pass as `HIGH` (`$0` grader cost).
- Scores `< 0.38` auto-route as `LOW` directly to corrective rewrite or web fallback (`$0` grader cost).
- Only the ambiguous band `[0.38, 0.72)` invokes a fast lightweight LLM CRAG grader to disambiguate `SUFFICIENT` vs `PARTIAL` vs `INSUFFICIENT`.

## Self-RAG & Grounded Citation Verification

Asai et al. (ICLR 2024, *Self-RAG*) introduces reflection checkpoints evaluating retrieval relevance (`ISREL`), factual support (`ISSUP`), and overall utility (`ISUSE`). In modular systems built on general-purpose LLMs (such as DeepSeek V3.2 or Qwen3), the `ISSUP` check is implemented as a **Two-Tier Citation Validator**:

1. **Tier-1 Deterministic Checks (0 ms LLM cost)**:
   - Parse every inline citation marker (`[C1]`, `[W2]`) via regex and verify every cited ID exists in the supplied context set (catches fabricated citation IDs immediately).
   - Detect factual sentences carrying numbers, configuration keys, or technical claims that lack any citation marker.
2. **Tier-2 Semantic Entailment Verification**:
   - Decompose the draft answer into atomic claims and verify whether each cited passage strictly entails its claim (`SUPPORTED`, `PARTIAL`, `UNSUPPORTED`).
   - If `claim_support_rate` drops below `0.85` or hallucinated IDs exist, trigger at most **one** strict regeneration pass instructing the LLM to remove or qualify unsupported assertions.

## Bi-Encoder vs Cross-Encoder & Query Transformations

- **Bi-Encoders** encode query and passage independently into fixed vectors (`O(1)` ANN lookup via HNSW), making them fast for first-stage recall over millions of chunks, but unable to model fine-grained token-to-token interactions.
- **Cross-Encoders** concatenate `[CLS] query [SEP] passage [SEP]` through every transformer attention layer, delivering dramatically higher precision and calibrated relevance probabilities at the cost of `O(N)` scoring time—making them ideal as a second-stage reranker over `N = 15..30` candidates.
- **Query Rewriting & Multi-Query Decomposition** resolve conversational pronouns ("it", "that parameter", "bunun varsayılanı") into standalone queries and split compound comparisons ("Compare Qdrant vs pgvector and explain how vLLM Prefix Caching affects TTFT") into focused atomic retrieval targets.
