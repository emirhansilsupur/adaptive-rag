# Qdrant Vector Database Architecture & HNSW Internals

Qdrant is an open-source vector similarity search engine written in Rust, purpose-built for production retrieval workloads that combine high-dimensional dense embeddings, sparse lexical vectors, and structured metadata filtering.

## Storage Engine & Segment Lifecycle

A Qdrant collection is partitioned into **shards**, and each shard manages multiple independent **segments**. Segments are the fundamental unit of storage and indexing:

- **Appendable segments** absorb real-time writes (upserts and deletes) with low latency without rebuilding the HNSW graph on every point insertion.
- **Sealed (non-appendable) segments** have finalized HNSW graphs and payload indexes optimized for fast read performance.
- **Write-Ahead Log (WAL)** persists every mutation sequentially on disk before acknowledging the client, guaranteeing crash recovery without forcing immediate segment flushes.
- A background **optimizer** merges small segments, removes soft-deleted points (`deleted_threshold`), and triggers HNSW indexing once a segment crosses `indexing_threshold` (default: `20000` KB of vectors).

## HNSW Graph Indexing Parameters

Qdrant uses Hierarchical Navigable Small World (HNSW) graphs for approximate nearest neighbor (ANN) search over dense vectors. Tuning HNSW controls the classic triangle between **recall**, **query latency**, and **RAM usage**:

| Parameter | Default Value | Scope | Engineering Impact |
| :--- | :--- | :--- | :--- |
| `m` | `16` | Index build | Number of bidirectional edges per node. Higher `m` (e.g., `32`) improves recall on high-dimensional vectors but increases RAM and build time. |
| `ef_construct` | `100` | Index build | Size of the dynamic candidate list during graph construction. Raising `ef_construct` to `200` improves graph quality with zero impact on query RAM. |
| `full_scan_threshold` | `10000` | Index/Query | Segments with fewer than `10000` KB (or small point counts) use exact brute-force scan instead of HNSW, guaranteeing 100% recall with minimal overhead. |
| `hnsw_ef` | `128` (auto) | Query time | Search-time beam width (`SearchParams(hnsw_ef=...)`). Raising `hnsw_ef` increases recall linearly with search latency without rebuilding the index. |
| `on_disk` | `false` | Storage | Stores the HNSW graph via memory-mapped files (`mmap`) instead of locking it in RAM, trading tail latency for lower memory cost. |

```python
from qdrant_client import models

hnsw_config = models.HnswConfigDiff(
    m=16,
    ef_construct=128,
    full_scan_threshold=10_000,
    on_disk=False,
)
```

## Filterable HNSW & Payload Indexes

Naive pre-filtering degrades HNSW recall because removing filtered-out nodes disconnects graph paths; naive post-filtering wastes compute when filter selectivity is high. Qdrant solves this with **Filterable HNSW**:

1. During index construction, Qdrant augments the HNSW graph with **additional edges connecting points that share the same payload values** for indexed fields (`keyword`, `integer`, `float`, `bool`, `geo`, `datetime`, `text`).
2. At query time, Qdrant inspects the **cardinality estimate** of the filter:
   - If matching points are sparse (below `full_scan_threshold`), Qdrant bypasses HNSW and runs an exact scan directly on the filtered subset.
   - If matching points are dense, Qdrant traverses the augmented HNSW graph while evaluating the filter inline at every hop.
3. Creating explicit payload indexes (`create_payload_index`) is mandatory for high-throughput filtered queries; unindexed payload filters force full payload scans.

## Vector Quantization & Memory Optimization

Storing 768-dimensional `float32` vectors requires `768 * 4 = 3,072` bytes per vector before HNSW graph overhead. Qdrant provides three quantization strategies to cut RAM usage by 4x to 32x:

### Scalar Quantization (`int8`)
Maps each `float32` dimension to an 8-bit integer (`4x` memory compression) using a quantile range such as `quantile=0.99` to ignore outliers. Setting `always_ram=True` keeps the quantized vectors hot in RAM while original `float32` vectors remain on disk via `mmap`. Recall loss is typically under `1%` when paired with `rescore=True` and `oversampling=1.5` to `2.0`.

### Binary Quantization (`1-bit`)
Compresses each dimension to a single bit (`32x` memory compression) using sign thresholding. Works best on high-dimensional normalized embeddings (`>= 1024` dimensions such as OpenAI `text-embedding-3-large` or Cohere `embed-v3`) where Hamming distance approximates cosine similarity. Requires higher oversampling (`oversampling=3.0` to `4.0`) plus original-vector rescoring (`rescore=True`) to recover top-k accuracy.

### Product Quantization (PQ)
Splits vectors into sub-vectors and replaces each sub-vector with a centroid ID (`x8` to `x64` compression). PQ is slower to build and typically yields lower recall than scalar quantization on modern RAG workloads, so Scalar Quantization is generally the first choice for corpora under 50 million vectors.
