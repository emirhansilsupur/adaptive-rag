# PostgreSQL Internals: MVCC, Indexing, Memory Tuning & `pgvector` Trade-offs

PostgreSQL underpins metadata stores, LangGraph durable checkpoints, observability telemetry (such as Langfuse), and small-to-medium vector workloads via `pgvector`. Understanding its storage engine prevents silent performance collapse under write-heavy AI workloads.

## MVCC, Dead Tuples & `VACUUM`

PostgreSQL implements Multi-Version Concurrency Control (**MVCC**) using immutable heap tuples stamped with transaction IDs:
- `xmin`: Transaction ID that inserted the tuple version.
- `xmax`: Transaction ID that deleted or superseded the tuple (0 if still live).

An `UPDATE` in PostgreSQL does **not** overwrite a row in place; it marks the old tuple with `xmax` and inserts a brand-new tuple version. Because readers never block writers and writers never block readers, concurrency stays high—at the cost of accumulating **dead tuples** (table bloat).

### Autovacuum Trigger Formula
Autovacuum runs on a table when dead tuples exceed:

$$\text{vacuum\_threshold} = \text{autovacuum\_vacuum\_threshold} + \text{autovacuum\_vacuum\_scale\_factor} \times \text{reltuples}$$

- Default `autovacuum_vacuum_threshold` = `50`.
- Default `autovacuum_vacuum_scale_factor` = `0.2` (`20%` of the table!).
- On a 50-million-row trace or checkpoint table, a `20%` scale factor waits until **10 million dead rows** accumulate before vacuuming! Production hot tables lower `autovacuum_vacuum_scale_factor` to `0.01`–`0.02` via per-table storage parameters.
- Failing to freeze old transaction IDs risks **Transaction ID (XID) Wraparound**, which forces PostgreSQL into read-only emergency shutdown once age approaches `2^31` transactions (`autovacuum_freeze_max_age`, default `200,000,000`).

## Memory & Connection Pooling Parameters

| Parameter | Typical Default | Production Guidance |
| :--- | :--- | :--- |
| `shared_buffers` | `128MB` | Set to ~`25%` of total server RAM on dedicated DB hosts for shared page caching. |
| `work_mem` | `4MB` | Memory per sort/hash operation **per query node** (not per connection!). Setting `work_mem` too high globally can cause OOM under 200 concurrent complex queries. |
| `effective_cache_size` | `4GB` | Planner hint estimating total OS page cache + `shared_buffers` (~`50%–75%` of RAM); does not allocate memory directly. |
| `max_connections` | `100` | Keep moderate (`100–300`) and place **PgBouncer** in front for high-concurrency microservices. |

### PgBouncer Pooling Modes
- **Session pooling**: Assigns a backend server connection for the entire client session lifetime (compatible with prepared statements and `LISTEN/NOTIFY`, lower multiplexing ratio).
- **Transaction pooling**: Returns the server connection to the pool immediately after `COMMIT`/`ROLLBACK`, allowing thousands of FastAPI workers to share 40 PostgreSQL backends cleanly.

## Indexing Families & `pgvector` vs Dedicated Vector Databases

PostgreSQL offers specialized index access methods:
- **B-Tree**: Default for equality and range queries (`=`, `<`, `>`, `BETWEEN`, `ORDER BY`).
- **GIN (Generalized Inverted Index)**: Excels at JSONB containment (`@>`), array membership, and full-text `tsvector` lexical search (`to_tsvector` + `ts_rank_cd`), though write amplification is higher than B-Tree.
- **`pgvector` (`ivfflat` and `hnsw`)**: Brings vector similarity inside ACID SQL transactions—outstanding when corpora are `< 1–5M` vectors and strong transactional joins with relational tables are required.

### Where Dedicated Engines (Qdrant) Diverge from `pgvector`
1. **Filtered ANN Recall**: Traditional `pgvector` post-filtering over HNSW can return fewer than `k` results when `WHERE` clauses are selective (partially mitigated by `hnsw.iterative_scan` in `pgvector >= 0.8.0`), whereas Qdrant's **Filterable HNSW** embeds payload links directly into the graph and dynamically switches to exact scan below `full_scan_threshold`.
2. **Native Hybrid Sparse + Dense**: Qdrant natively indexes BM25 sparse vectors with live `Modifier.IDF` alongside dense vectors and runs server-side `Prefetch + RRF` in one API call; replicating this in PostgreSQL requires maintaining separate `tsvector` + `GIN` + `pgvector` indexes and manually fusing via SQL CTEs.
3. **Quantization & Horizontal Sharding**: Qdrant offers built-in `int8` scalar and `1-bit` binary quantization with disk-backed `mmap` + oversampled rescoring and Raft-coordinated horizontal sharding out of the box.
