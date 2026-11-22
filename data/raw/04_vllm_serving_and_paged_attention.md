# vLLM Serving Internals: PagedAttention, Continuous Batching & Latency Tuning

Serving large language models in production is fundamentally **memory-bandwidth bound** during token generation (decode) and **compute bound** during prompt processing (prefill). vLLM addresses GPU memory waste and scheduling bottlenecks through PagedAttention and continuous iteration-level batching.

## KV-Cache Fragmentation & PagedAttention

During autoregressive decoding, transformers cache Key (`K`) and Value (`V`) tensors for every prior token in every attention layer. For a 70B parameter model with GQA or standard MHA, a single 8K-context request can consume several gigabytes of KV-cache.

Traditional serving engines pre-allocate contiguous VRAM buffers sized to `max_model_len` for each request, wasting **60% to 80% of KV-cache memory** due to:
- **Internal fragmentation**: Request finishes after 200 tokens even though 4,096 token slots were reserved.
- **External fragmentation**: Free VRAM blocks scattered between active allocations cannot fit a new contiguous request.

**PagedAttention** borrows virtual memory paging from operating systems:
1. It partitions each sequence's KV-cache into fixed-size **KV blocks** (default `block_size=16` tokens).
2. Blocks do **not** need to be contiguous in physical GPU HBM; a per-sequence **block table** maps logical token blocks to physical VRAM blocks on demand.
3. Physical blocks are allocated dynamically as tokens are generated, reducing KV-cache waste to under `4%` (only the partially filled final block).
4. **Copy-on-Write (CoW) block sharing** allows parallel beam search, best-of-N sampling, and shared system prompts to reference identical physical KV blocks with zero memory duplication.

## Continuous Batching & Chunked Prefill

Static batching waits until the longest sequence in a batch finishes before admitting new requests, leaving GPUs idle whenever response lengths vary.

- **Continuous (iteration-level) batching** reschedules at **every single forward pass**. As soon as one request emits `<EOS>`, its KV blocks are freed and a waiting request enters the batch on the very next step.
- **Chunked Prefill** (`--enable-chunked-prefill` / `enable_chunked_prefill=True`) splits long prompt prefills (e.g., an 8,000-token RAG context) into smaller token chunks governed by `--max-num-batched-tokens` (`max_num_batched_tokens`, commonly tuned between `2048` and `8192`) and interleaves them with decode tokens from active requests. Without chunked prefill, a single long RAG prompt causes severe **decode stalls** (latency spikes) for every concurrent user.
- **Automatic Prefix Caching** (`--enable-prefix-caching` / `enable_prefix_caching=True`) hashes KV blocks of shared prompt prefixes (such as a fixed RAG system prompt or shared company handbook chunks) so repeated calls skip prefill computation entirely, cutting **TTFT** by 50–85%.

## Key Serving Flags & Tuning Guide

```bash
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 128
```

| Configuration Flag | Default | Purpose & Trade-off |
| :--- | :--- | :--- |
| `gpu_memory_utilization` | `0.90` | Fraction of GPU VRAM allocated for model weights + KV-cache pool. Raising to `0.95` increases concurrency capacity but risks OOM if another process shares the GPU. |
| `tensor_parallel_size` | `1` | Splits layer weight matrices across `N` GPUs over NVLink. Reduces latency (`TTFT` and `TPOT`) for large models, while single-GPU replicas with data parallelism maximize total throughput per dollar. |
| `max_model_len` | Model max | Caps total prompt + output tokens. Lowering from `131072` to `32768` dramatically increases available KV-cache blocks for concurrent requests. |
| `max_num_seqs` | `256` | Maximum concurrent sequences scheduled in a single iteration. Lower values reduce tail latency (`TPOT`); higher values boost batch throughput. |

## Core LLM Latency Metrics

Evaluating inference backends requires separating three distinct latency components:
- **TTFT (Time To First Token)**: Time from HTTP request arrival to the first streamed token. Dominated by queue wait time and prompt prefill length (directly impacted by RAG `top_k` context tokens!).
- **TPOT (Time Per Output Token)** / **ITL (Inter-Token Latency)**: Average (or p95) interval between consecutive generated tokens. Humans comfortably read around `5–8` tokens/sec (`~125–200 ms` TPOT), while interactive apps target `< 50 ms` TPOT.
- **End-to-End Latency (E2EL)**: `TTFT + TPOT * output_tokens`. Reducing RAG context from 6,000 tokens to 2,200 tokens via **Dynamic Top-K** cuts both TTFT and prompt token spend simultaneously.
