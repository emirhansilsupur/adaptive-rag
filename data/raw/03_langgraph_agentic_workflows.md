# LangGraph Agentic Workflows: StateGraph, Cycles & Production Guardrails

LangGraph models LLM applications as explicit **stateful directed graphs** (`StateGraph`) rather than opaque autonomous agent loops or rigid linear chains. For Agentic RAG systems, LangGraph provides deterministic control flow, bounded self-correction cycles, typed state transitions, and streaming visibility into every node.

## Graph State Design & Reducers

Every node in a `StateGraph` receives the current state object and returns a partial dictionary of state updates. State schemas can be declared as Python `TypedDict` classes or Pydantic `BaseModel` classes.

- **Overwrite fields** (default behavior): returning `{"confidence": assessment}` replaces the previous value cleanly—ideal for single-slot outputs like `classification`, `confidence`, or `draft_answer`.
- **Append-only reducer fields**: annotating a list with `Annotated[list[NodeTrace], operator.add]` ensures each node appends telemetry or messages without clobbering prior entries.

```python
import operator
from typing import Annotated, TypedDict

class AgentState(TypedDict, total=False):
    query: str
    sub_queries: list[str]
    rewrite_count: int
    regeneration_count: int
    node_traces: Annotated[list[dict], operator.add]
```

## Conditional Edges & Bounded Self-Correction Loops

The defining difference between a DAG pipeline and an Agentic RAG graph is **controlled cyclicity**. Examples include:
1. **Corrective Retrieval Loop**: When retrieval confidence falls into the ambiguous/low band and `rewrite_count < max_corrective_rewrites` (typically `1`), a conditional edge routes from `assess_confidence` back to `corrective_rewrite -> hybrid_retrieve`.
2. **Citation Repair Loop**: When Tier-1 or Tier-2 citation validation detects hallucinated source IDs or unsupported claims and `regeneration_count < max_regenerations` (typically `1`), the graph routes back to `generate_answer` with explicit corrective feedback.

### Preventing Infinite Agent Loops
Production graphs MUST enforce **two independent safety nets**:
- **Domain counters in state**: Explicit `rewrite_count` and `regeneration_count` checked directly inside conditional edge functions, transitioning to `web_fallback` or safe `abstain` once exhausted.
- **Graph-level `recursion_limit`**: Passed via `config={"recursion_limit": 20}` during `graph.ainvoke(...)` or `graph.astream(...)` as a hard upper bound against unexpected routing bugs.

## Checkpointing & Multi-Turn Persistence

LangGraph decouples graph execution from persistence through the `BaseCheckpointSaver` interface:

| Checkpointer | Backing Store | Intended Usage |
| :--- | :--- | :--- |
| `MemorySaver` | In-process Python dict | Unit tests, local demos, ephemeral sessions |
| `SqliteSaver` / `AsyncSqliteSaver` | Local SQLite file | Single-node desktop apps, lightweight CLI tools |
| `PostgresSaver` / `AsyncPostgresSaver` | PostgreSQL | Multi-replica production APIs, durable thread history |

Passing `config={"configurable": {"thread_id": session_id}}` automatically persists state snapshots after every super-step, enabling:
- **Multi-turn conversational memory** and coreference resolution across user turns.
- **Time-travel debugging** to replay a failed RAG trace from any intermediate node.
- **Human-in-the-loop interrupts** (`interrupt()`) when high-stakes tool actions require operator approval.

## Streaming Modes for Glass-Box RAG Interfaces

Rather than making users wait 5–10 seconds for an entire agentic RAG graph to finish before showing output, LangGraph supports granular async streaming via `graph.astream(..., stream_mode=...)`:

- `stream_mode="updates"`: Yields `{node_name: state_delta}` immediately as each node completes—ideal for streaming routing decisions, reranker scores, dynamic `top_k` cutoffs, and confidence gauges over Server-Sent Events (SSE).
- `stream_mode="messages"`: Streams individual LLM tokens alongside metadata identifying which graph node produced them, enabling live token streaming only for the final `generate_answer` node while suppressing raw JSON tokens from internal router/grader nodes.
- `stream_mode="values"`: Yields the full merged state snapshot after every node (heavier payload, useful for debugging).

Combining `stream_mode=["updates", "messages"]` delivers both live pipeline observability and responsive token-by-token UX.
