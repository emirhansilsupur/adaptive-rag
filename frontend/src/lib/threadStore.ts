import type { ChatSession } from "../types/api";

/*
 * Threads live in IndexedDB. They used to live in localStorage, whose ~5 MB quota fills after
 * roughly 150 answers; from then on every save failed silently and new answers vanished on reload.
 */

/** Where older Studio versions kept threads. Read at startup, moved, then removed. */
export const LEGACY_THREADS_KEY = "adaptiverag_studio_sessions_v1";

const DB_NAME = "adaptiverag-studio";
const DB_VERSION = 1;
const THREADS = "threads";
const OPEN_TIMEOUT_MS = 3000;

/** Why threads aren't being saved. */
export type ThreadSaveProblem =
  | { kind: "full" }
  | { kind: "unavailable"; detail: string }
  | { kind: "failed"; detail: string };

/** Where threads are written: IndexedDB in the browser, a Map in tests. */
export interface ThreadBackend {
  loadAll(): Promise<ChatSession[]>;
  /** Puts and removes in one transaction: all of it is stored, or none of it. */
  write(changes: { put: ChatSession[]; remove: string[] }): Promise<void>;
}

export interface ThreadStore {
  /** Threads saved earlier, newest first. */
  readonly threads: ChatSession[];
  /** A problem found while opening the store, or null. */
  readonly problem: ThreadSaveProblem | null;
  /**
   * Writes the threads that changed since the last successful save and removes deleted ones.
   * Saves run one at a time; a call made during a save is written right after it. Resolves to
   * null once `threads` is stored, or to what went wrong. Never rejects.
   */
  save(threads: ChatSession[]): Promise<ThreadSaveProblem | null>;
}

/**
 * Opens the browser's thread store. Never rejects: when the browser can't store anything, the
 * store says why and its saves do nothing.
 */
export async function openThreadStore(
  options: { indexedDB?: IDBFactory | null; storage?: Storage | null; timeoutMs?: number } = {}
): Promise<ThreadStore> {
  const storage = options.storage !== undefined ? options.storage : browserLocalStorage();
  const factory = options.indexedDB !== undefined ? options.indexedDB : browserIndexedDB();
  if (!factory) {
    return unsavedStore(storage, "this browser has no IndexedDB");
  }
  let backend: ThreadBackend;
  try {
    backend = await openIndexedDb(factory, options.timeoutMs ?? OPEN_TIMEOUT_MS);
  } catch (err) {
    return unsavedStore(storage, errorDetail(err));
  }
  return createThreadStore(backend, storage);
}

/** Loads the saved threads and moves any that an older Studio left in localStorage. */
export async function createThreadStore(
  backend: ThreadBackend,
  storage: Storage | null
): Promise<ThreadStore> {
  let saved: ChatSession[];
  try {
    saved = (await backend.loadAll()).filter(isThread);
  } catch (err) {
    return unsavedStore(storage, errorDetail(err));
  }
  // What the backend holds. Each save writes the difference between this and the new list.
  let stored = new Map(saved.map((thread) => [thread.id, thread]));
  let problem: ThreadSaveProblem | null = null;

  const legacy = readLegacyThreads(storage);
  if (legacy) {
    const missing = legacy.filter((thread) => !stored.has(thread.id));
    saved = [...saved, ...missing];
    try {
      if (missing.length > 0) {
        await backend.write({ put: missing, remove: [] });
        for (const thread of missing) stored.set(thread.id, thread);
      }
      removeLegacyThreads(storage);
    } catch (err) {
      // The old copy stays, and the first save tries to store these threads again.
      problem = toProblem(err);
    }
  }

  let pending: ChatSession[] | null = null;
  let draining: Promise<ThreadSaveProblem | null> | null = null;

  const writeChanges = async (threads: ChatSession[]): Promise<ThreadSaveProblem | null> => {
    try {
      const ids = new Set(threads.map((thread) => thread.id));
      // React replaces a thread object whenever it changes, so identity marks what to write.
      const put = threads.filter((thread) => stored.get(thread.id) !== thread);
      const remove = [...stored.keys()].filter((id) => !ids.has(id));
      if (put.length > 0 || remove.length > 0) {
        await backend.write({ put, remove });
        stored = new Map(threads.map((thread) => [thread.id, thread]));
      }
      return null;
    } catch (err) {
      return toProblem(err);
    }
  };

  // Each pass diffs against what is actually stored, so a failed save is retried by the next one.
  const drain = async (): Promise<ThreadSaveProblem | null> => {
    let outcome: ThreadSaveProblem | null = null;
    while (pending) {
      const next = pending;
      pending = null;
      outcome = await writeChanges(next);
    }
    draining = null;
    return outcome;
  };

  return {
    threads: newestFirst(saved),
    problem,
    save(threads) {
      pending = threads;
      draining ??= drain();
      return draining;
    },
  };
}

/** A store for a browser that can't store threads: it shows what an older Studio left behind. */
function unsavedStore(storage: Storage | null, detail: string): ThreadStore {
  const problem: ThreadSaveProblem = { kind: "unavailable", detail };
  return {
    threads: newestFirst(readLegacyThreads(storage) ?? []),
    problem,
    save: () => Promise.resolve(problem),
  };
}

async function openIndexedDb(factory: IDBFactory, timeoutMs: number): Promise<ThreadBackend> {
  const db = await openDatabase(factory, timeoutMs);
  // Let a newer Studio in another tab upgrade the database instead of waiting for this tab.
  db.onversionchange = () => db.close();
  return {
    loadAll: () =>
      new Promise((resolve, reject) => {
        const request = db.transaction(THREADS, "readonly").objectStore(THREADS).getAll();
        request.onsuccess = () => resolve(request.result as ChatSession[]);
        request.onerror = () => reject(request.error);
      }),
    write: ({ put, remove }) =>
      new Promise((resolve, reject) => {
        const tx = db.transaction(THREADS, "readwrite");
        tx.oncomplete = () => resolve();
        // A failed request aborts the whole transaction; a full disk ends up here too.
        tx.onabort = () => reject(tx.error ?? new DOMException("The save was cancelled.", "AbortError"));
        try {
          const store = tx.objectStore(THREADS);
          for (const thread of put) store.put(thread);
          for (const id of remove) store.delete(id);
        } catch (err) {
          reject(err);
          try {
            tx.abort();
          } catch {
            // already finished
          }
        }
      }),
  };
}

function openDatabase(factory: IDBFactory, timeoutMs: number): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    let settled = false;
    const settle = (finish: () => void) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      finish();
    };
    // Some browsers have left open() pending forever; don't keep the Studio from starting.
    const timer = setTimeout(
      () => settle(() => reject(new Error(`IndexedDB didn't open within ${timeoutMs / 1000} s`))),
      timeoutMs
    );
    try {
      const request = factory.open(DB_NAME, DB_VERSION);
      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(THREADS)) {
          db.createObjectStore(THREADS, { keyPath: "id" });
        }
      };
      request.onsuccess = () => {
        if (settled) {
          request.result.close();
          return;
        }
        settle(() => resolve(request.result));
      };
      request.onerror = () =>
        settle(() => reject(request.error ?? new Error("IndexedDB refused to open")));
    } catch (err) {
      settle(() => reject(err));
    }
  });
}

function readLegacyThreads(storage: Storage | null): ChatSession[] | null {
  try {
    const raw = storage?.getItem(LEGACY_THREADS_KEY);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter(isThread) : null;
  } catch {
    return null;
  }
}

function removeLegacyThreads(storage: Storage | null): void {
  try {
    storage?.removeItem(LEGACY_THREADS_KEY);
  } catch {
    // The next start finds these threads already moved and removes the key then.
  }
}

function isThread(value: unknown): value is ChatSession {
  if (!value || typeof value !== "object") return false;
  const thread = value as Partial<ChatSession>;
  return typeof thread.id === "string" && Array.isArray(thread.turns);
}

/** New threads go first in the sidebar, so restore that order. */
function newestFirst(threads: ChatSession[]): ChatSession[] {
  return [...threads].sort((a, b) => (b.createdAt ?? 0) - (a.createdAt ?? 0));
}

function toProblem(err: unknown): ThreadSaveProblem {
  return errorName(err) === "QuotaExceededError"
    ? { kind: "full" }
    : { kind: "failed", detail: errorDetail(err) };
}

function errorName(err: unknown): string {
  return typeof err === "object" && err !== null && "name" in err ? String(err.name) : "";
}

function errorDetail(err: unknown): string {
  const message =
    typeof err === "object" && err !== null && "message" in err ? String(err.message) : String(err);
  return message.trim().replace(/\.$/, "") || errorName(err) || "unknown error";
}

function browserIndexedDB(): IDBFactory | null {
  try {
    return typeof indexedDB === "undefined" ? null : indexedDB;
  } catch {
    return null; // sandboxed frames can throw on access
  }
}

function browserLocalStorage(): Storage | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}
