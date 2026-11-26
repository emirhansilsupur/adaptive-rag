import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import {
  createThreadStore,
  LEGACY_THREADS_KEY,
  openThreadStore,
  type ThreadBackend,
  type ThreadStore,
} from "../lib/threadStore";
import type { ChatSession, ChatTurn, FinalAnswer } from "../types/api";

function makeAnswer(query: string, answer: string, passage: string): FinalAnswer {
  return {
    query,
    answer,
    query_type: "factual",
    language: "en",
    query_plan: null,
    evidence_chunks: [
      {
        citation_id: "C1",
        chunk_id: `chunk_${query}`,
        text: passage,
        source_type: "local",
        doc_id: "qdrant_guide",
        title: "Qdrant guide",
        section_path: "HNSW",
        url: null,
        metadata: {},
        dense_score: 0.8,
        sparse_score: null,
        fusion_score: 0.03,
        rerank_raw_score: 3.1,
        rerank_score: 0.9,
        matched_sub_queries: [],
        token_count: 40,
      },
    ],
    dynamic_k_decision: null,
    confidence: null,
    citation_report: null,
    fallback_used: false,
    abstained: false,
    rewrite_attempts: 0,
    regeneration_attempts: 0,
    total_latency_ms: 1200,
    usage: {
      prompt_tokens: 400,
      completion_tokens: 40,
      total_tokens: 440,
      estimated_cost_usd: 0,
      cache_hits: 0,
    },
    trace: [],
    trace_id: null,
    trace_url: null,
  };
}

function makeTurn(query: string, passage = "HNSW builds its graph with ef_construct = 100."): ChatTurn {
  return {
    id: `turn_${query}`,
    query,
    createdAt: 0,
    presetLabel: "Full",
    result: makeAnswer(query, `Answer to ${query} [C1].`, passage),
  };
}

function makeThread(id: string, createdAt: number, turns: ChatTurn[]): ChatSession {
  return { id, title: turns[0]?.query ?? "New thread", createdAt, updatedAt: createdAt, turns };
}

/** IndexedDB stand-in: stores copies, writes asynchronously, and can be told to fail. */
function memoryBackend(initial: ChatSession[] = []) {
  const copy = (thread: ChatSession): ChatSession => JSON.parse(JSON.stringify(thread));
  const records = new Map(initial.map((thread) => [thread.id, copy(thread)]));
  const writes: { put: string[]; remove: string[] }[] = [];
  const failures: unknown[] = [];
  const backend: ThreadBackend = {
    loadAll: async () => [...records.values()].map(copy),
    write: async ({ put, remove }) => {
      await Promise.resolve();
      const failure = failures.shift();
      if (failure) throw failure;
      writes.push({ put: put.map((thread) => thread.id), remove: [...remove] });
      for (const thread of put) records.set(thread.id, copy(thread));
      for (const id of remove) records.delete(id);
    },
  };
  return {
    backend,
    records,
    writes,
    failNextWrite: (err: unknown) => failures.push(err),
    savedQueries: () => [...records.values()].flatMap((t) => t.turns.map((turn) => turn.query)),
  };
}

const quotaExceeded = () => new DOMException("The quota has been exceeded.", "QuotaExceededError");

afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
});

describe("thread store", () => {
  it("keeps threads across a reload, newest first", async () => {
    const mem = memoryBackend();
    const store = await createThreadStore(mem.backend, localStorage);
    expect(store.threads).toEqual([]);

    const older = makeThread("t_old", 1_000, [makeTurn("First question")]);
    const newer = makeThread("t_new", 2_000, []);
    expect(await store.save([newer, older])).toBeNull();

    const reloaded = await createThreadStore(mem.backend, localStorage);
    expect(reloaded.threads.map((t) => t.id)).toEqual(["t_new", "t_old"]);
    expect(reloaded.threads[1].turns[0].query).toBe("First question");
  });

  it("writes only the threads that changed and removes deleted ones", async () => {
    const mem = memoryBackend([makeThread("a", 1, []), makeThread("b", 2, []), makeThread("c", 3, [])]);
    const store = await createThreadStore(mem.backend, localStorage);
    const [c, b] = store.threads;
    const bAnswered = { ...b, turns: [makeTurn("New question")] };

    expect(await store.save([c, bAnswered])).toBeNull();
    expect(mem.writes).toEqual([{ put: ["b"], remove: ["a"] }]);

    expect(await store.save([c, bAnswered])).toBeNull();
    expect(mem.writes).toHaveLength(1);
  });

  it("keeps a thread deleted while an earlier save was still running deleted", async () => {
    const mem = memoryBackend();
    const store = await createThreadStore(mem.backend, localStorage);
    const kept = makeThread("kept", 1, []);
    const shortLived = makeThread("short_lived", 2, []);

    const first = store.save([shortLived, kept]);
    const second = store.save([kept]);

    expect(await first).toBeNull();
    expect(await second).toBeNull();
    expect([...mem.records.keys()]).toEqual(["kept"]);
  });

  it("moves threads that an older Studio kept in localStorage", async () => {
    const inDb = makeThread("in_db", 3_000, []);
    const mem = memoryBackend([inDb]);
    localStorage.setItem(
      LEGACY_THREADS_KEY,
      JSON.stringify([
        makeThread("legacy_new", 2_000, [makeTurn("Asked in the old Studio")]),
        { ...inDb, title: "Stale copy" },
        makeThread("legacy_old", 1_000, []),
      ])
    );

    const store = await createThreadStore(mem.backend, localStorage);

    expect(store.problem).toBeNull();
    expect(store.threads.map((t) => t.id)).toEqual(["in_db", "legacy_new", "legacy_old"]);
    expect(store.threads[0].title).toBe("New thread");
    expect(mem.writes).toEqual([{ put: ["legacy_new", "legacy_old"], remove: [] }]);
    expect(localStorage.getItem(LEGACY_THREADS_KEY)).toBeNull();

    const reloaded = await createThreadStore(mem.backend, localStorage);
    expect(reloaded.threads.map((t) => t.id)).toEqual(["in_db", "legacy_new", "legacy_old"]);
  });

  it("keeps the localStorage copy when moving it fails, and moves it with the first save", async () => {
    const mem = memoryBackend();
    localStorage.setItem(
      LEGACY_THREADS_KEY,
      JSON.stringify([makeThread("legacy", 1_000, [makeTurn("Old question")])])
    );
    mem.failNextWrite(quotaExceeded());

    const store = await createThreadStore(mem.backend, localStorage);
    expect(store.problem).toEqual({ kind: "full" });
    expect(store.threads.map((t) => t.id)).toEqual(["legacy"]);
    expect(localStorage.getItem(LEGACY_THREADS_KEY)).not.toBeNull();

    expect(await store.save(store.threads)).toBeNull();
    expect(mem.savedQueries()).toEqual(["Old question"]);
  });

  it("reports a failed save and stores everything once a later save goes through", async () => {
    const mem = memoryBackend([makeThread("t", 1, [])]);
    const store = await createThreadStore(mem.backend, localStorage);
    const answered = { ...store.threads[0], turns: [makeTurn("Unsaved question")] };

    mem.failNextWrite(quotaExceeded());
    expect(await store.save([answered])).toEqual({ kind: "full" });
    mem.failNextWrite(new Error("Disk I/O error."));
    expect(await store.save([answered])).toEqual({ kind: "failed", detail: "Disk I/O error" });
    expect(mem.savedQueries()).toEqual([]);

    // Nothing changed since, and the retry still writes what the failed saves missed.
    expect(await store.save([answered])).toBeNull();
    expect(mem.savedQueries()).toEqual(["Unsaved question"]);
  });

  it("says why threads can't be saved when the browser has no IndexedDB", async () => {
    localStorage.setItem(LEGACY_THREADS_KEY, JSON.stringify([makeThread("legacy", 1, [])]));
    const store = await openThreadStore({ indexedDB: null, storage: localStorage });
    const problem = { kind: "unavailable", detail: "this browser has no IndexedDB" };

    expect(store.problem).toEqual(problem);
    expect(store.threads.map((t) => t.id)).toEqual(["legacy"]);
    expect(await store.save([])).toEqual(problem);
    expect(localStorage.getItem(LEGACY_THREADS_KEY)).not.toBeNull();
  });

  it("stops waiting for an IndexedDB that never opens", async () => {
    const neverOpens = { open: () => ({}) } as unknown as IDBFactory;
    const store = await openThreadStore({ indexedDB: neverOpens, storage: localStorage, timeoutMs: 10 });
    expect(store.problem).toEqual({
      kind: "unavailable",
      detail: "IndexedDB didn't open within 0.01 s",
    });
  });
});

describe("Studio threads", () => {
  const QUESTION = "What does ef_construct default to?";
  const ANSWER = /It defaults to 100/;

  /** Serves one streamed answer; every other API call fails as if the API were down. */
  function fakeApi(answer: FinalAnswer) {
    return vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).endsWith("/v1/query/stream")) {
        return new Response(`event: final_answer\ndata: ${JSON.stringify(answer)}\n\n`, {
          headers: { "Content-Type": "text/event-stream" },
        });
      }
      return new Response(JSON.stringify({ detail: "No API in tests." }), {
        status: 503,
        headers: { "Content-Type": "application/json" },
      });
    });
  }

  /** Renders the Studio and waits until its startup calls to the (absent) API have settled. */
  async function renderStudio(store: Promise<ThreadStore>) {
    const view = render(<App threadStore={await store} />);
    await screen.findByText("Offline");
    return view;
  }

  async function ask(question: string) {
    fireEvent.change(screen.getByPlaceholderText("Ask a question about your documents"), {
      target: { value: question },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send question" }));
    await screen.findByText(ANSWER);
  }

  /**
   * 24 threads and 164 answers saved by an older Studio, filling localStorage to within 10 KB of
   * its quota. The 20 newest hold almost all of it; that Studio only ever kept the newest 20.
   */
  function fillLocalStorageWithOldThreads() {
    const passage = "x".repeat(28_000); // a real answer carries ~29 KB of passages and scores
    const threads = Array.from({ length: 24 }, (_, i) =>
      makeThread(
        `legacy_${i}`,
        1_000 + i,
        Array.from({ length: i < 4 ? 1 : 8 }, (_, j) =>
          makeTurn(`Older question ${i}.${j}`, i < 4 ? "Short passage." : passage)
        )
      )
    ).reverse();
    const used = LEGACY_THREADS_KEY.length + JSON.stringify(threads).length;
    threads[0].turns[0].result.evidence_chunks[0].text += "y".repeat(4_990_000 - used);
    localStorage.setItem(LEGACY_THREADS_KEY, JSON.stringify(threads));
  }

  it("keeps a new answer after a reload when older threads fill localStorage", async () => {
    fillLocalStorageWithOldThreads();
    const mem = memoryBackend();
    vi.stubGlobal("fetch", fakeApi(makeAnswer(QUESTION, "It defaults to 100 [C1].", "z".repeat(28_000))));

    const first = await renderStudio(createThreadStore(mem.backend, localStorage));
    await ask(QUESTION);
    await waitFor(() => expect(mem.savedQueries()).toContain(QUESTION));
    first.unmount();

    await renderStudio(createThreadStore(mem.backend, localStorage));
    expect(screen.getByRole("heading", { name: "Older question 23.0" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: QUESTION })).toBeInTheDocument();
    expect(screen.getByText(ANSWER)).toBeInTheDocument();
    // Threads past the newest 20 are kept too.
    expect(screen.getByTitle("Older question 0.0")).toBeInTheDocument();
  });

  it("warns while an answer isn't saved and stores it once deleting a thread frees space", async () => {
    const mem = memoryBackend([
      makeThread("older", 1_000, [makeTurn("Older question")]),
      makeThread("current", 2_000, []),
    ]);
    vi.stubGlobal("fetch", fakeApi(makeAnswer(QUESTION, "It defaults to 100 [C1].", "Passage.")));
    await renderStudio(createThreadStore(mem.backend, localStorage));

    mem.failNextWrite(quotaExceeded());
    await ask(QUESTION);
    expect(await screen.findByText("Threads aren't being saved")).toBeInTheDocument();
    expect(screen.getByText(/Browser storage for this site is full/)).toBeInTheDocument();
    expect(mem.savedQueries()).not.toContain(QUESTION);

    fireEvent.click(screen.getByRole("button", { name: 'Delete thread "Older question"' }));
    fireEvent.click(screen.getByRole("button", { name: "Delete" }));

    await waitFor(() => expect(screen.queryByText("Threads aren't being saved")).toBeNull());
    expect(mem.savedQueries()).toEqual([QUESTION]);
  });

  it("tells the user when this browser can't store threads at all", async () => {
    vi.stubGlobal("fetch", fakeApi(makeAnswer(QUESTION, "It defaults to 100 [C1].", "Passage.")));
    await renderStudio(openThreadStore({ indexedDB: null, storage: localStorage }));

    expect(screen.getByText("Threads aren't being saved")).toBeInTheDocument();
    expect(screen.getByText(/\(this browser has no IndexedDB\)/)).toBeInTheDocument();
  });
});
