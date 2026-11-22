# FastAPI Async Concurrency, ONNX Offloading & SSE Streaming for AI Backends

FastAPI is widely adopted as the serving gateway for RAG and agentic backends because of its native ASGI async foundation, Pydantic v2 validation speed (powered by Rust `pydantic-core`), and automatic OpenAPI generation. However, mixing async I/O with CPU-bound ML inference introduces subtle concurrency traps if not architected carefully.

## Async Event Loop vs Threadpool Execution

FastAPI executes endpoint functions differently depending on how they are declared:
- **`async def` endpoints** execute directly on the single-threaded `asyncio` event loop. They excel at high-concurrency network I/O (awaiting Qdrant HTTP/gRPC calls, OpenRouter LLM streaming, Tavily web search, PostgreSQL queries).
- **Synchronous `def` endpoints** are automatically dispatched by Starlette/AnyIO to an external worker threadpool (default size: **40 threads** via `anyio.to_thread.current_default_thread_limiter()`).

### The Event-Loop Blocking Pitfall in RAG Backends
A classic production bug in local-embedding RAG services is calling synchronous CPU-bound inference directly inside an `async def` route or LangGraph node:

```python
# ANTI-PATTERN: Blocks the entire asyncio event loop for 80-300ms!
async def retrieve(query: str):
    vector = list(fastembed_model.query_embed(query))[0]  # Synchronous ONNX!
    return await qdrant.query_points(...)
```

While ONNX Runtime computes dense embeddings or Cross-Encoder reranking on CPU, **every other concurrent user's SSE stream, health check, and Qdrant callback freezes**.

**Correct Pattern:** Offload CPU-bound embedding and reranking calls via `asyncio.to_thread(...)` (or a bounded `ProcessPoolExecutor` / dedicated `ThreadPoolExecutor` when ONNX releases the GIL, which ONNX Runtime C++ kernels do):

```python
import asyncio

async def aembed_query(model, query: str) -> list[float]:
    def _run() -> list[float]:
        return next(iter(model.query_embed([query]))).tolist()

    return await asyncio.to_thread(_run)
```

## Lifespan Resource Management

Heavy clients—such as `AsyncQdrantClient`, `httpx.AsyncClient` connection pools, SQLite LLM caches, and pre-warmed FastEmbed ONNX sessions—should be initialized **once** during FastAPI's `lifespan` async context manager and closed cleanly on shutdown, never recreated per request:

- Creating a new `AsyncQdrantClient` per request leaks TCP sockets and skips HTTP/2 keep-alive reuse.
- Lazy-loading a 1GB multilingual ONNX model on the first user request creates a 5–15 second cold-start latency spike unless warmed during startup or guarded with an async lock.

## Server-Sent Events (SSE) vs WebSockets for Agentic Streaming

For Agentic RAG UIs that stream intermediate reasoning steps (`route`, `retrieve`, `rerank`, `confidence`, `fallback`, `token`, `citations`, `done`), **Server-Sent Events (SSE)** via `sse-starlette` (`EventSourceResponse`) is generally superior to WebSockets:

| Dimension | Server-Sent Events (SSE) | WebSockets |
| :--- | :--- | :--- |
| Communication Pattern | Unidirectional server → client over standard HTTP | Bidirectional full-duplex TCP upgrade |
| Proxy / Load Balancer Support | Works seamlessly through HTTP/2, CDNs, NGINX, API gateways | Requires sticky sessions & explicit upgrade rules |
| Automatic Reconnection | Built into browser `EventSource` spec (`Last-Event-ID`) | Must be manually implemented |
| Best AI Use Case | RAG pipeline telemetry + LLM token streaming | Real-time bidirectional voice/audio agents |

### Production SSE Hygiene
1. Set `headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}` so reverse proxies (like NGINX) flush each event immediately rather than buffering 4 KB chunks.
2. Listen to `await request.is_disconnected()` inside the async generator so client aborts immediately cancel upstream LLM calls and stop burning tokens.
3. Enforce explicit per-call timeouts (`llm_request_timeout=45.0`) and send structured `event: error` frames on upstream failure instead of dropping the HTTP stream abruptly.
