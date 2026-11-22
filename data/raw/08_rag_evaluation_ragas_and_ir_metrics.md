# Evaluating RAG Systems: IR Metrics, RAGAS, Citation Auditing & Ablation Studies

Deploying a RAG pipeline without quantitative evaluation invites silent regressions—a prompt tweak or chunk-size change can quietly halve recall while demo responses still look fluent. Rigorous RAG evaluation decomposes quality into **Retrieval (IR) Metrics**, **Generation & Grounding Metrics**, **Agentic Control Metrics**, and **Operational Efficiency (Latency & Cost)**.

## Information Retrieval (IR) Metrics

When evaluating retrieval against a labeled golden dataset containing relevant `doc_id` or `chunk_id` targets:

- **Recall@k**: Fraction of all ground-truth relevant chunks present in the top-`k` retrieved set. Critical for RAG because the generator cannot cite evidence it never saw.
- **Precision@k**: Fraction of top-`k` retrieved chunks that are actually relevant. High precision prevents context pollution ("lost in the middle") and saves prompt tokens.
- **MRR (Mean Reciprocal Rank)**: Average of $\frac{1}{\text{rank}_i}$ for the first relevant document per query. Ideal for **factual single-answer** queries where placing the exact passage at rank `#1` maximizes accuracy.
- **NDCG@k (Normalized Discounted Cumulative Gain)**: Measures graded ranking quality with logarithmic position discounting ($\frac{1}{\log_2(i + 1)}$). Ideal for **analytical multi-document** queries where multiple passages matter and top ordering is critical.

## Generation & Grounding Metrics (RAGAS Framework)

The **RAGAS** evaluation framework (Es et al., EACL 2024) evaluates RAG outputs along the RAG triad without requiring brittle string-matching metrics like BLEU or ROUGE:

| Metric | Inputs Required | What It Measures |
| :--- | :--- | :--- |
| **Faithfulness** | Answer + Retrieved Contexts | Fraction of atomic claims in the answer that are strictly entailed by the retrieved contexts (hallucination detector). |
| **Answer Relevancy** | Query + Answer | Semantic alignment between the generated answer and the user's actual question (catches evasive or off-topic replies). |
| **Context Precision** | Query + Retrieved Contexts + Ground Truth | Whether relevant chunks are ranked above irrelevant chunks inside the retrieved context window. |
| **Context Recall** | Ground Truth Answer + Retrieved Contexts | Fraction of ground-truth statements attributable to the retrieved context (measures retriever completeness). |

### LLM-as-a-Judge Biases & Mitigation
Using an LLM evaluator introduces well-documented systematic biases:
- **Self-preference bias**: Models score outputs written by their own model family higher. Mitigate by using a distinct judge model or reporting deterministic non-LLM metrics alongside LLM judges.
- **Verbosity bias**: Judges reward longer answers even when extra sentences add fluff. Mitigate via explicit rubric instructions penalizing unsupported or redundant prose.
- **Position bias**: In pairwise comparisons, judges favor whichever candidate appears first. Mitigate by swapping order and averaging.

## Agentic & Citation-Specific Metrics

Beyond standard RAGAS, an Agentic RAG system requires custom engineering metrics:
1. **Routing Accuracy**: Exact match accuracy of the query classifier across `factual`, `analytical`, and `conversational` classes (target `>= 90%`).
2. **Citation Precision**: $\frac{\text{Valid Supported Citations}}{\text{Total Citations Emitted}}$. Catches phantom IDs and misattributed sources.
3. **Claim Groundedness Rate**: Fraction of factual claims backed by an entailing citation.
4. **Fallback Trigger Precision & Recall**: On out-of-corpus queries, how reliably confidence gating triggers Web Fallback (Recall) without falsely firing on well-covered local queries (Precision).
5. **Context Token Efficiency**: Average context tokens passed to generation—demonstrating how Dynamic Top-K preserves or improves Faithfulness while cutting prompt tokens by 25–40% on factual queries.

## Ablation Study Design

To prove that architectural complexity pays for itself, benchmark **four controlled configurations** over the exact same stratified golden test set (same embeddings, same generator LLM, deterministic `temperature=0.0`):

1. **Config A — Naive RAG**: Dense-only cosine retrieval, fixed `k=5`, no routing, no reranking, no fallback, no citation validator.
2. **Config B — Hybrid RAG**: Native Qdrant Dense + BM25 RRF, fixed `k=5`, no routing, no reranking.
3. **Config C — Hybrid + Cross-Encoder Rerank**: Hybrid RRF + multilingual Cross-Encoder (`bge-reranker-v2-m3-int8`), fixed `k=5`.
4. **Config D — Full AdaptiveRAG**: Query classification + adaptive strategy + multi-query decomposition + Hybrid RRF + Cross-Encoder + 3-stage Dynamic Top-K + Multi-signal Confidence/CRAG + Web Fallback + Two-Tier Citation Validation.

Comparing Config A → B → C → D isolates the exact marginal gain of every component in terms of Faithfulness, Context Recall, Citation Precision, p95 Latency, and USD Cost per Query.
