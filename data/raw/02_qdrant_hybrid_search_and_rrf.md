# Qdrant Native Hybrid Search, BM25 Sparse Vectors & Score Fusion

Dense embeddings capture semantic equivalence and cross-lingual similarity, but they routinely underperform on exact identifiers, version numbers, acronyms, API flag names (such as `ef_construct` or `max_num_batched_tokens`), and rare domain terms. Native hybrid search combines dense semantic retrieval with sparse lexical retrieval inside a single vector database round-trip.

## Named Vectors & Server-Side IDF

Qdrant collections support **named dense vectors** and **named sparse vectors** on the same point. For BM25-style lexical retrieval, the sparse vector space is configured with `Modifier.IDF`:

```python
from qdrant_client import AsyncQdrantClient, models

await client.create_collection(
    collection_name="adaptiverag_docs",
    vectors_config={
        "dense": models.VectorParams(size=768, distance=models.Distance.COSINE),
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
    },
)
```

With `Modifier.IDF` enabled:
- Client-side tokenizers (such as FastEmbed's `Qdrant/bm25` model) compute term frequency (`TF`) weights and hashed token indices per chunk without needing global corpus statistics at ingestion time.
- Qdrant maintains **live document frequency statistics** across the collection and multiplies query sparse weights by the **Inverse Document Frequency (IDF)** dynamically at search time.
- As new documents are upserted or deleted, IDF weights automatically stay accurate without re-indexing the entire corpus.

## Multi-Stage Retrieval with `Prefetch` & `FusionQuery`

Qdrant's Universal Query API (`query_points`) executes multiple candidate retrieval branches in parallel using `Prefetch` and merges them server-side:

```python
response = await client.query_points(
    collection_name="adaptiverag_docs",
    prefetch=[
        models.Prefetch(query=dense_vec, using="dense", limit=20),
        models.Prefetch(query=sparse_vec, using="sparse", limit=25),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=20,
    with_payload=True,
)
```

### Tuning Prefetch Limits by Query Intent
- **Factual queries** benefit from a slightly deeper `sparse` prefetch (`sparse_prefetch_limit=25`, `dense_prefetch_limit=20`) because exact parameter names, error codes, and numeric thresholds strongly discriminate the single right passage.
- **Analytical queries** require broader recall (`dense_prefetch_limit=30`, `sparse_prefetch_limit=30`, `fusion_limit=25`) across multiple decomposed sub-queries so cross-document comparisons are not truncated early.

## Reciprocal Rank Fusion (RRF) vs Distribution-Based Score Fusion (DBSF)

Dense cosine similarity lies in `[-1, 1]`, whereas BM25 dot-product scores are unbounded positive floats (e.g., `0.0` to `45.0`). Directly summing raw scores fails because whichever modality has larger scale dominates the ranking.

### Reciprocal Rank Fusion (`Fusion.RRF`)
RRF ignores raw score magnitudes and fuses solely by rank position across retrieval lists $R$:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

- Standard smoothing constant is $k = 60$ (Qdrant's default).
- A document ranked `#1` in both dense and sparse lists receives $\frac{1}{61} + \frac{1}{61} \approx 0.0328$ (or `1/(2+rank)` in simplified normalized implementations).
- **Strengths:** Completely scale-invariant, parameter-free, and robust to score distribution shifts across languages or query lengths.
- **Limitation:** RRF discards **score confidence margins**. A query whose top dense hit has cosine similarity `0.92` and a completely irrelevant query whose top dense hit has cosine similarity `0.18` can receive the **exact same RRF score** if both appear at rank `#1`.

### Distribution-Based Score Fusion (`Fusion.DBSF`)
DBSF normalizes each branch's scores using its sample mean $\mu$ and standard deviation $\sigma$ mapped into $[\mu - 3\sigma, \mu + 3\sigma]$ before summing. DBSF preserves score magnitudes better than RRF when candidate distributions are well-behaved, but it is more sensitive to small prefetch sizes or skewed sparse score outliers.

## Why Cross-Encoder Reranking Is Essential After RRF

Because RRF scores are **rank-based rather than relevance-calibrated**, they cannot reliably power:
1. **Dynamic top-k cutoffs** (you cannot set an absolute relevance floor `tau_min` on a rank score).
2. **Retrieval confidence gating** (you cannot detect out-of-corpus queries from rank `#1` RRF alone).

Running a **Cross-Encoder reranker** (such as `BAAI/bge-reranker-v2-m3-int8`) over the top 15–30 fused candidates jointly encodes `(query, passage)` with full cross-attention and outputs a calibrated relevance logit that, after sigmoid transformation $\sigma(z) = \frac{1}{1 + e^{-z}}$, enables both adaptive `top_k` selection and out-of-corpus fallback detection.
