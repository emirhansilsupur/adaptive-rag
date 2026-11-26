import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Menu, Moon, PanelLeftOpen, PanelRightClose, PanelRightOpen, Sun } from "lucide-react";
import { ChatWorkspace } from "./components/ChatWorkspace";
import { CompareArena } from "./components/CompareArena";
import { GlassBoxInspector } from "./components/GlassBoxInspector";
import { KnowledgeBaseView } from "./components/KnowledgeBaseView";
import { ModelSelectorPopover } from "./components/ModelSelectorPopover";
import { Sidebar } from "./components/Sidebar";
import { cn, IconButton, Kbd } from "./components/ui/primitives";
import {
  DEFAULT_API_URL,
  DEFAULT_CUSTOM_ABLATION,
  DEFAULT_THREAD_CONTEXT,
  PRESET_OPTIONS,
  resolveEffectiveInstructions,
  RUN_STOPPED_MESSAGE,
} from "./constants/pipeline";
import {
  buildQueryPayload,
  fetchDocuments,
  fetchHealth,
  fetchModelsCatalog,
  runQueryBatch,
  setStudioApiKey,
  streamQuery,
  submitFeedback,
} from "./lib/api";
import { createCompareInspectorTurns, getPresetShortLabel } from "./lib/compare";
import type { ThreadSaveProblem, ThreadStore } from "./lib/threadStore";
import type {
  AblationConfig,
  AblationPreset,
  AppView,
  ChatMessage,
  ChatSession,
  ChatTurn,
  CompareRun,
  CompareSideKey,
  DocumentListResponse,
  FinalAnswer,
  HealthResponse,
  InspectorTab,
  LiveStreamStep,
  LLMOverride,
  LLMProvider,
  ModelsCatalogResponse,
  PresetSelection,
  ProviderApiKeys,
  ThreadContextConfig,
  TokenUsage,
} from "./types/api";

const STORAGE_COMPARE_KEY = "adaptiverag_studio_compare_v1";
const STORAGE_THEME_KEY = "adaptiverag_studio_theme_v1";
const STORAGE_API_URL_KEY = "adaptiverag_studio_api_url_v1";
const STORAGE_LLM_OVERRIDE_KEY = "adaptiverag_studio_llm_override_v1";
const SESSION_API_KEY_STORAGE_KEY = "adaptiverag_studio_api_key_v1";
const SESSION_PROVIDER_KEYS_STORAGE_KEY = "adaptiverag_studio_provider_keys_v1";

/** Sidebar docks from 1024px; the inspector docks only when there's room for three columns. */
const DESKTOP_QUERY = "(min-width: 1024px)";
const WIDE_QUERY = "(min-width: 1280px)";
const INSPECTOR_MIN = 360;
const INSPECTOR_MAX = 680;

function loadInitialLlmOverride(): LLMOverride | null {
  try {
    const raw = localStorage.getItem(STORAGE_LLM_OVERRIDE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as LLMOverride | null;
    if (parsed && typeof parsed === "object") {
      return parsed;
    }
  } catch {
    // ignore storage errors
  }
  return null;
}

function loadInitialProviderApiKeys(): ProviderApiKeys {
  try {
    const raw = sessionStorage.getItem(SESSION_PROVIDER_KEYS_STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as ProviderApiKeys | null;
    if (parsed && typeof parsed === "object") {
      return parsed;
    }
  } catch {
    // ignore storage errors
  }
  return {};
}

function loadInitialCompareRuns(): CompareRun[] {
  try {
    const raw = localStorage.getItem(STORAGE_COMPARE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as CompareRun[];
    if (Array.isArray(parsed)) {
      return parsed.filter((r) => r && typeof r.id === "string" && r.left && r.right);
    }
  } catch {
    // ignore storage errors
  }
  return [];
}

function createEmptySession(): ChatSession {
  const now = Date.now();
  return {
    id: `sess_${now}_${Math.random().toString(36).slice(2, 7)}`,
    title: "New thread",
    createdAt: now,
    updatedAt: now,
    turns: [],
  };
}

function loadInitialTheme(): "dark" | "light" {
  try {
    const saved = localStorage.getItem(STORAGE_THEME_KEY);
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    // ignore storage errors
  }
  if (typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)").matches) {
    return "dark";
  }
  return "light";
}

function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() =>
    typeof window !== "undefined" && window.matchMedia ? window.matchMedia(query).matches : true
  );
  useEffect(() => {
    if (!window.matchMedia) return;
    const mql = window.matchMedia(query);
    const onChange = () => setMatches(mql.matches);
    onChange();
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, [query]);
  return matches;
}

interface AppProps {
  /** Opened before the first render, so a saved list is never replaced by an empty one. */
  threadStore: ThreadStore;
}

export default function App({ threadStore }: AppProps) {
  /* ------------------------------------------------------------------------ */
  /* Theme: saved choice, otherwise the operating system's preference          */
  /* ------------------------------------------------------------------------ */
  const [theme, setTheme] = useState<"dark" | "light">(loadInitialTheme);
  const themeChosenRef = useRef(false);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    if (!themeChosenRef.current) return;
    try {
      localStorage.setItem(STORAGE_THEME_KEY, theme);
    } catch {
      // ignore
    }
  }, [theme]);

  const toggleTheme = () => {
    themeChosenRef.current = true;
    setTheme((t) => (t === "dark" ? "light" : "dark"));
  };

  /* ------------------------------------------------------------------------ */
  /* Layout                                                                    */
  /* ------------------------------------------------------------------------ */
  const isDesktop = useMediaQuery(DESKTOP_QUERY);
  const isWide = useMediaQuery(WIDE_QUERY);
  const [activeView, setActiveView] = useState<AppView>("workbench");
  const [sidebarOpen, setSidebarOpen] = useState(
    () => typeof window === "undefined" || !window.matchMedia || window.matchMedia(DESKTOP_QUERY).matches
  );
  const [inspectorOpen, setInspectorOpen] = useState(
    () => typeof window === "undefined" || !window.matchMedia || window.matchMedia(WIDE_QUERY).matches
  );
  const [inspectorWidth, setInspectorWidth] = useState(440);
  const resizingRef = useRef(false);

  // Crossing a breakpoint resets the panels to that layout's defaults.
  const prevDesktopRef = useRef(isDesktop);
  const prevWideRef = useRef(isWide);
  useEffect(() => {
    if (prevDesktopRef.current !== isDesktop) {
      prevDesktopRef.current = isDesktop;
      setSidebarOpen(isDesktop);
    }
    if (prevWideRef.current !== isWide) {
      prevWideRef.current = isWide;
      setInspectorOpen(isWide && activeView !== "knowledge");
    }
  }, [isDesktop, isWide, activeView]);

  /* ------------------------------------------------------------------------ */
  /* API, health, knowledge base                                               */
  /* ------------------------------------------------------------------------ */
  const [apiUrl, setApiUrl] = useState<string>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_API_URL_KEY);
      return saved !== null ? saved : DEFAULT_API_URL;
    } catch {
      return DEFAULT_API_URL;
    }
  });
  const [apiKey, setApiKey] = useState<string>(() => {
    try {
      const saved = sessionStorage.getItem(SESSION_API_KEY_STORAGE_KEY) ?? "";
      setStudioApiKey(saved);
      return saved;
    } catch {
      return "";
    }
  });
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [healthLoading, setHealthLoading] = useState(false);

  const [documentsData, setDocumentsData] = useState<DocumentListResponse | null>(null);
  const [documentsLoading, setDocumentsLoading] = useState(false);
  const [documentsError, setDocumentsError] = useState<string | null>(null);

  const [modelsCatalog, setModelsCatalog] = useState<ModelsCatalogResponse | null>(null);
  const [modelsCatalogLoading, setModelsCatalogLoading] = useState(false);
  const [llmOverride, setLlmOverride] = useState<LLMOverride | null>(loadInitialLlmOverride);
  const [providerApiKeys, setProviderApiKeys] = useState<ProviderApiKeys>(loadInitialProviderApiKeys);

  /* ------------------------------------------------------------------------ */
  /* Threads                                                                   */
  /* ------------------------------------------------------------------------ */
  const [sessions, setSessions] = useState<ChatSession[]>(() =>
    threadStore.threads.length > 0 ? threadStore.threads : [createEmptySession()]
  );
  const [threadSaveProblem, setThreadSaveProblem] = useState<ThreadSaveProblem | null>(
    threadStore.problem
  );
  const [activeSessionId, setActiveSessionId] = useState<string>(() => sessions[0].id);
  const [selectedTurnIndex, setSelectedTurnIndex] = useState<number>(() =>
    Math.max(0, (sessions[0]?.turns.length ?? 1) - 1)
  );
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>("trace");
  const [highlightedCitationId, setHighlightedCitationId] = useState<string | null>(null);
  const [highlightToken, setHighlightToken] = useState<number>(0);

  /* ------------------------------------------------------------------------ */
  /* Pipeline controls                                                         */
  /* ------------------------------------------------------------------------ */
  const [preset, setPreset] = useState<PresetSelection>("full");
  const [customAblation, setCustomAblation] = useState<AblationConfig>(DEFAULT_CUSTOM_ABLATION);
  const [useStreaming, setUseStreaming] = useState(true);
  const [enableBudgetCap, setEnableBudgetCap] = useState(false);
  const [maxCostUsd, setMaxCostUsd] = useState(0.05);
  // The server caps every question at MAX_RUN_COST_USD; a cap set here can only go lower.
  const budgetCap = enableBudgetCap
    ? Math.min(maxCostUsd, health?.default_max_run_cost_usd ?? maxCostUsd)
    : null;

  /* ------------------------------------------------------------------------ */
  /* Active run                                                                */
  /* ------------------------------------------------------------------------ */
  const [isRunning, setIsRunning] = useState(false);
  const [pendingQuery, setPendingQuery] = useState<string | null>(null);
  const [liveSteps, setLiveSteps] = useState<LiveStreamStep[]>([]);
  const [liveUsage, setLiveUsage] = useState<Partial<TokenUsage> | null>(null);
  const [streamError, setStreamError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  /* ------------------------------------------------------------------------ */
  /* Arena                                                                     */
  /* ------------------------------------------------------------------------ */
  const [compareRuns, setCompareRuns] = useState<CompareRun[]>(loadInitialCompareRuns);
  const [activeCompareRunId, setActiveCompareRunId] = useState<string | null>(
    () => loadInitialCompareRuns()[0]?.id ?? null
  );
  const [leftPreset, setLeftPreset] = useState<AblationPreset>("naive_dense");
  const [rightPreset, setRightPreset] = useState<AblationPreset>("full");
  const [compareRunning, setCompareRunning] = useState(false);
  const [inspectedCompareSide, setInspectedCompareSide] = useState<CompareSideKey>("right");
  const compareAbortRef = useRef<AbortController | null>(null);

  const activeCompareRun = useMemo(
    () =>
      activeCompareRunId ? (compareRuns.find((r) => r.id === activeCompareRunId) ?? null) : null,
    [compareRuns, activeCompareRunId]
  );

  const compareInspector = useMemo(
    () => createCompareInspectorTurns(activeCompareRun),
    [activeCompareRun]
  );

  // Persist completed Arena runs
  useEffect(() => {
    try {
      const serializable = compareRuns
        .filter((r) => r.left.status !== "running" && r.right.status !== "running")
        .slice(0, 10)
        .map((r) => ({
          ...r,
          left: { ...r.left, liveSteps: [] },
          right: { ...r.right, liveSteps: [] },
        }));
      localStorage.setItem(STORAGE_COMPARE_KEY, JSON.stringify(serializable));
    } catch {
      // ignore storage errors
    }
  }, [compareRuns]);

  const activeSession = useMemo(
    () => sessions.find((s) => s.id === activeSessionId) ?? sessions[0],
    [sessions, activeSessionId]
  );

  // Every thread is kept, and a failed save stays on screen until a later one goes through.
  useEffect(() => {
    let current = true;
    void threadStore.save(sessions).then((problem) => {
      if (current) setThreadSaveProblem(problem);
    });
    return () => {
      current = false;
    };
  }, [sessions, threadStore]);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_API_URL_KEY, apiUrl);
    } catch {
      // ignore
    }
  }, [apiUrl]);

  useEffect(() => {
    try {
      if (llmOverride) {
        localStorage.setItem(STORAGE_LLM_OVERRIDE_KEY, JSON.stringify(llmOverride));
      } else {
        localStorage.removeItem(STORAGE_LLM_OVERRIDE_KEY);
      }
    } catch {
      // ignore
    }
  }, [llmOverride]);

  const handleProviderApiKeyChange = useCallback((provider: LLMProvider, nextKey: string) => {
    setProviderApiKeys((prev) => {
      const next: ProviderApiKeys = { ...prev };
      const cleaned = nextKey.trim();
      if (cleaned) {
        next[provider] = cleaned;
      } else {
        delete next[provider];
      }
      try {
        if (Object.keys(next).length > 0) {
          sessionStorage.setItem(SESSION_PROVIDER_KEYS_STORAGE_KEY, JSON.stringify(next));
        } else {
          sessionStorage.removeItem(SESSION_PROVIDER_KEYS_STORAGE_KEY);
        }
      } catch {
        // ignore storage errors
      }
      return next;
    });
  }, []);

  const activeEffectiveProvider = (llmOverride?.provider ??
    modelsCatalog?.default_provider ??
    health?.llm_provider ??
    "openrouter") as LLMProvider;
  const activeProviderApiKey = providerApiKeys[activeEffectiveProvider]?.trim() || undefined;

  const handleApiKeyChange = useCallback((nextKey: string) => {
    const cleaned = nextKey.trim();
    setStudioApiKey(cleaned);
    setApiKey(cleaned);
    try {
      if (cleaned) {
        sessionStorage.setItem(SESSION_API_KEY_STORAGE_KEY, cleaned);
      } else {
        sessionStorage.removeItem(SESSION_API_KEY_STORAGE_KEY);
      }
    } catch {
      // ignore storage errors
    }
  }, []);

  const refreshHealth = useCallback(async () => {
    setHealthLoading(true);
    try {
      const data = await fetchHealth(apiUrl);
      setHealth(data);
      setHealthError(null);
    } catch (err) {
      setHealth(null);
      setHealthError(err instanceof Error ? err.message : String(err));
    } finally {
      setHealthLoading(false);
    }
    // apiKey is a dependency on purpose: it's sent as a header, so a new key re-probes.
  }, [apiUrl, apiKey]);

  const refreshDocuments = useCallback(async () => {
    setDocumentsLoading(true);
    try {
      const data = await fetchDocuments(apiUrl);
      setDocumentsData(data);
      setDocumentsError(null);
    } catch (err) {
      setDocumentsError(err instanceof Error ? err.message : String(err));
    } finally {
      setDocumentsLoading(false);
    }
  }, [apiUrl, apiKey]);

  const refreshModelsCatalog = useCallback(async () => {
    setModelsCatalogLoading(true);
    try {
      const data = await fetchModelsCatalog(apiUrl);
      setModelsCatalog(data);
    } catch {
      // keep previous catalog or null gracefully
    } finally {
      setModelsCatalogLoading(false);
    }
  }, [apiUrl, apiKey]);

  const refreshAll = useCallback(async () => {
    await Promise.all([refreshHealth(), refreshDocuments(), refreshModelsCatalog()]);
  }, [refreshHealth, refreshDocuments, refreshModelsCatalog]);

  useEffect(() => {
    void refreshAll();
  }, [refreshAll]);

  // Keyboard: ⌘I / Ctrl+I toggles the inspector, Escape closes mobile overlays.
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "i") {
        e.preventDefault();
        setInspectorOpen((prev) => !prev);
        return;
      }
      if (e.key === "Escape" && !e.defaultPrevented) {
        if (!isDesktop) setSidebarOpen(false);
        if (!isWide) setInspectorOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isDesktop, isWide]);

  // Drag to resize the docked inspector
  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (!resizingRef.current) return;
      const nextWidth = Math.min(INSPECTOR_MAX, Math.max(INSPECTOR_MIN, window.innerWidth - e.clientX));
      setInspectorWidth(nextWidth);
    };
    const onMouseUp = () => {
      if (!resizingRef.current) return;
      resizingRef.current = false;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }, []);

  const closeSidebarOnMobile = () => {
    if (!isDesktop) setSidebarOpen(false);
  };

  const handleNewSession = () => {
    const fresh = createEmptySession();
    setSessions((prev) => [fresh, ...prev]);
    setActiveSessionId(fresh.id);
    setSelectedTurnIndex(0);
    setHighlightedCitationId(null);
    setStreamError(null);
    closeSidebarOnMobile();
  };

  const handleDeleteSession = (id: string) => {
    setSessions((prev) => {
      const remaining = prev.filter((s) => s.id !== id);
      if (remaining.length === 0) {
        const fresh = createEmptySession();
        setActiveSessionId(fresh.id);
        return [fresh];
      }
      if (id === activeSessionId) {
        setActiveSessionId(remaining[0].id);
        setSelectedTurnIndex(Math.max(0, remaining[0].turns.length - 1));
      }
      return remaining;
    });
  };

  const handleSelectSession = (id: string) => {
    setActiveSessionId(id);
    const target = sessions.find((s) => s.id === id);
    setSelectedTurnIndex(target ? Math.max(0, target.turns.length - 1) : 0);
    setHighlightedCitationId(null);
    setStreamError(null);
    closeSidebarOnMobile();
  };

  const handleUpdateActiveThreadContext = (nextContext: ThreadContextConfig) => {
    const targetId = activeSession.id;
    setSessions((prev) =>
      prev.map((sess) =>
        sess.id === targetId ? { ...sess, context: nextContext, updatedAt: Date.now() } : sess
      )
    );
  };

  /** Selecting an answer updates the inspector; asking for a tab also opens it. */
  const handleInspectTurn = (turnIdx: number, tab?: InspectorTab) => {
    setSelectedTurnIndex(turnIdx);
    if (tab) {
      setInspectorTab(tab);
      setInspectorOpen(true);
    }
  };

  const handleCitationClick = (turnIdx: number, citationId: string) => {
    setSelectedTurnIndex(turnIdx);
    setInspectorTab("evidence");
    setHighlightedCitationId(citationId);
    setHighlightToken((t) => t + 1);
    setInspectorOpen(true);
  };

  const handleJumpWithinInspector = (citationId: string) => {
    setInspectorTab("evidence");
    setHighlightedCitationId(citationId);
    setHighlightToken((t) => t + 1);
  };

  const handleStopStream = () => {
    abortControllerRef.current?.abort();
  };

  const handleChangeView = (nextView: AppView) => {
    setActiveView(nextView);
    if (isWide) {
      if (nextView === "compare" && window.innerWidth < 1540) {
        setInspectorOpen(false);
      } else if (nextView === "workbench") {
        setInspectorOpen(true);
      }
    } else {
      setInspectorOpen(false);
    }
    if (!isDesktop) setSidebarOpen(false);
  };

  const handleInspectCompareSide = (side: CompareSideKey, tab?: InspectorTab) => {
    setInspectedCompareSide(side);
    if (tab) {
      setInspectorTab(tab);
      setInspectorOpen(true);
    }
  };

  const handleCompareCitationClick = (side: CompareSideKey, citationId: string) => {
    setInspectedCompareSide(side);
    setInspectorTab("evidence");
    setHighlightedCitationId(citationId);
    setHighlightToken((t) => t + 1);
    setInspectorOpen(true);
  };

  const handleStopCompare = () => {
    compareAbortRef.current?.abort();
  };

  const handleSwapPresets = () => {
    setLeftPreset(rightPreset);
    setRightPreset(leftPreset);
  };

  const handleDeleteCompareRun = (runId: string) => {
    setCompareRuns((prev) => {
      const remaining = prev.filter((r) => r.id !== runId);
      if (activeCompareRunId === runId) {
        setActiveCompareRunId(remaining[0]?.id ?? null);
      }
      return remaining;
    });
  };

  const handleSubmitCompareFeedback = async (
    runId: string,
    side: CompareSideKey,
    score: 0 | 1,
    comment?: string
  ) => {
    const run = compareRuns.find((r) => r.id === runId);
    const sideState = run?.[side];
    const traceId = sideState?.result?.trace_id;
    if (!run || !sideState || !traceId) return;

    const resp = await submitFeedback(apiUrl, {
      trace_id: traceId,
      score,
      comment,
      session_id: `${run.id}_${side}`,
      query: run.query,
    });

    setCompareRuns((prev) =>
      prev.map((item) => {
        if (item.id !== runId) return item;
        return {
          ...item,
          [side]: {
            ...item[side],
            feedback: {
              score,
              comment: resp.comment ?? comment,
              persistedToLangfuse: resp.persisted_to_langfuse,
              submittedAt: Date.now(),
            },
          },
        };
      })
    );
  };

  const handleSubmitCompare = async (query: string) => {
    if (compareRunning) return;

    const runId = `cmp_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
    const leftLabel = getPresetShortLabel(leftPreset);
    const rightLabel = getPresetShortLabel(rightPreset);

    const initialRun: CompareRun = {
      id: runId,
      query,
      createdAt: Date.now(),
      left: {
        preset: leftPreset,
        presetLabel: leftLabel,
        status: "running",
        liveSteps: [],
        liveUsage: null,
        result: null,
        error: null,
      },
      right: {
        preset: rightPreset,
        presetLabel: rightLabel,
        status: "running",
        liveSteps: [],
        liveUsage: null,
        result: null,
        error: null,
      },
    };

    const controller = new AbortController();
    compareAbortRef.current = controller;

    setCompareRuns((prev) => [initialRun, ...prev.slice(0, 9)]);
    setActiveCompareRunId(runId);
    setCompareRunning(true);
    setHighlightedCitationId(null);

    const updateSide = (
      side: CompareSideKey,
      updater: (prev: CompareRun["left"]) => CompareRun["left"]
    ) => {
      setCompareRuns((prev) => prev.map((r) => (r.id === runId ? { ...r, [side]: updater(r[side]) } : r)));
    };

    const executeSide = async (side: CompareSideKey, sidePreset: AblationPreset) => {
      const payload = buildQueryPayload({
        query,
        chatHistory: [],
        preset: sidePreset,
        customAblation,
        maxCostUsd: budgetCap,
        sessionId: `${runId}_${side}`,
        llmOverride,
      });

      try {
        let answer: FinalAnswer;
        if (useStreaming) {
          answer = await streamQuery(
            apiUrl,
            payload,
            {
              onNodeUpdate: (update) => {
                updateSide(side, (prev) => {
                  const traceEvents = update.trace || [];
                  const appended: LiveStreamStep[] =
                    traceEvents.length > 0
                      ? traceEvents.map((evt, i) => ({
                          id: `${side}_${evt.node}_${prev.liveSteps.length + i}_${Date.now()}`,
                          node: evt.node,
                          latency_ms: evt.latency_ms,
                          summary: evt.summary,
                          metadata: evt.metadata || {},
                        }))
                      : update.node
                        ? [
                            {
                              id: `${side}_${update.node}_${prev.liveSteps.length}_${Date.now()}`,
                              node: update.node,
                              latency_ms: 0,
                              summary: "",
                              metadata: {},
                            },
                          ]
                        : [];
                  return {
                    ...prev,
                    liveUsage: update.usage ?? prev.liveUsage,
                    liveSteps: [...prev.liveSteps, ...appended],
                  };
                });
              },
            },
            controller.signal,
            activeProviderApiKey
          );
        } else {
          answer = await runQueryBatch(apiUrl, payload, controller.signal, activeProviderApiKey);
        }

        updateSide(side, (prev) => ({ ...prev, status: "done", result: answer, error: null }));
      } catch (err) {
        const message =
          (err as Error)?.name === "AbortError"
            ? "You stopped this comparison before it finished."
            : err instanceof Error
              ? err.message
              : String(err);
        updateSide(side, (prev) => ({ ...prev, status: "error", error: message }));
      }
    };

    try {
      await Promise.allSettled([executeSide("left", leftPreset), executeSide("right", rightPreset)]);
    } finally {
      setCompareRunning(false);
      compareAbortRef.current = null;
    }
  };

  const handleSubmitFeedback = async (turnId: string, score: 0 | 1, comment?: string) => {
    if (!activeSession) return;
    const targetTurn = activeSession.turns.find((t) => t.id === turnId);
    const traceId = targetTurn?.result.trace_id;
    if (!targetTurn || !traceId) return;

    const resp = await submitFeedback(apiUrl, {
      trace_id: traceId,
      score,
      comment,
      session_id: activeSession.id,
      query: targetTurn.query,
    });

    setSessions((prev) =>
      prev.map((sess) => {
        if (sess.id !== activeSession.id) return sess;
        return {
          ...sess,
          turns: sess.turns.map((turn) =>
            turn.id === turnId
              ? {
                  ...turn,
                  feedback: {
                    score,
                    comment: resp.comment ?? comment,
                    persistedToLangfuse: resp.persisted_to_langfuse,
                    submittedAt: Date.now(),
                  },
                }
              : turn
          ),
        };
      })
    );
  };

  const presetMeta = PRESET_OPTIONS.find((p) => p.value === preset) ?? PRESET_OPTIONS[0];

  const handleSubmitQuery = async (query: string) => {
    if (!activeSession || isRunning) return;

    const targetSessionId = activeSession.id;
    const chatHistory: ChatMessage[] = activeSession.turns.flatMap((t) => [
      { role: "user", content: t.query },
      { role: "assistant", content: t.result.answer },
    ]);

    const effectiveInstructions = resolveEffectiveInstructions(activeSession.context);
    const scopedDocIds = activeSession.context?.scopedDocIds ?? [];

    const payload = buildQueryPayload({
      query,
      chatHistory,
      preset,
      customAblation,
      maxCostUsd: budgetCap,
      sessionId: targetSessionId,
      customInstructions: effectiveInstructions,
      docIds: scopedDocIds,
      llmOverride,
    });

    const controller = new AbortController();
    abortControllerRef.current = controller;

    setIsRunning(true);
    setPendingQuery(query);
    setLiveSteps([]);
    setLiveUsage(null);
    setStreamError(null);

    try {
      let answer: FinalAnswer;
      if (useStreaming) {
        answer = await streamQuery(
          apiUrl,
          payload,
          {
            onNodeUpdate: (update) => {
              if (update.usage) {
                setLiveUsage(update.usage);
              }
              const traceEvents = update.trace || [];
              if (traceEvents.length > 0) {
                setLiveSteps((prev) => [
                  ...prev,
                  ...traceEvents.map((evt, i) => ({
                    id: `${evt.node}_${prev.length + i}_${Date.now()}`,
                    node: evt.node,
                    latency_ms: evt.latency_ms,
                    summary: evt.summary,
                    metadata: evt.metadata || {},
                  })),
                ]);
              } else if (update.node) {
                setLiveSteps((prev) => [
                  ...prev,
                  {
                    id: `${update.node}_${prev.length}_${Date.now()}`,
                    node: update.node,
                    latency_ms: 0,
                    summary: "",
                    metadata: {},
                  },
                ]);
              }
            },
          },
          controller.signal,
          activeProviderApiKey
        );
      } else {
        answer = await runQueryBatch(apiUrl, payload, controller.signal, activeProviderApiKey);
      }

      const newTurn: ChatTurn = {
        id: `turn_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
        query,
        createdAt: Date.now(),
        presetLabel: presetMeta.shortLabel,
        result: answer,
      };

      let nextTurnIdx = 0;
      setSessions((prev) =>
        prev.map((sess) => {
          if (sess.id !== targetSessionId) return sess;
          const updatedTurns = [...sess.turns, newTurn];
          nextTurnIdx = updatedTurns.length - 1;
          const updatedTitle =
            sess.turns.length === 0 ? (query.length > 48 ? `${query.slice(0, 48)}…` : query) : sess.title;
          return { ...sess, title: updatedTitle, updatedAt: Date.now(), turns: updatedTurns };
        })
      );
      setSelectedTurnIndex(nextTurnIdx);
      setHighlightedCitationId(null);
    } catch (err) {
      if ((err as Error)?.name === "AbortError") {
        setStreamError(RUN_STOPPED_MESSAGE);
      } else {
        setStreamError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      setIsRunning(false);
      setPendingQuery(null);
      abortControllerRef.current = null;
    }
  };

  /* ------------------------------------------------------------------------ */
  /* Render                                                                    */
  /* ------------------------------------------------------------------------ */
  const viewName =
    activeView === "workbench" ? "Workbench" : activeView === "compare" ? "Arena" : "Knowledge base";
  const viewSubtitle =
    activeView === "workbench"
      ? activeSession.title
      : activeView === "compare"
        ? `${getPresetShortLabel(leftPreset)} vs ${getPresetShortLabel(rightPreset)}`
        : null;

  const showInspectorToggle = activeView === "workbench" || activeView === "compare";
  const inspectorVisible = showInspectorToggle && inspectorOpen;
  const budgetCapLabel = budgetCap !== null ? `$${budgetCap.toFixed(2)}` : null;

  const sidebar = (
    <Sidebar
      activeView={activeView}
      onChangeView={handleChangeView}
      sessions={sessions}
      activeSessionId={activeSession.id}
      onSelectSession={handleSelectSession}
      onNewSession={handleNewSession}
      onDeleteSession={handleDeleteSession}
      apiUrl={apiUrl}
      onApiUrlChange={(url) => setApiUrl(url)}
      apiKey={apiKey}
      onApiKeyChange={handleApiKeyChange}
      health={health}
      healthError={healthError}
      healthLoading={healthLoading}
      onRefreshHealth={() => void refreshAll()}
      documentCount={documentsData?.total_documents}
      onCollapse={() => setSidebarOpen(false)}
      variant={isDesktop ? "docked" : "drawer"}
    />
  );

  const inspector =
    activeView === "compare" ? (
      <GlassBoxInspector
        turns={compareInspector.turns}
        selectedTurnIndex={
          compareInspector.sideIndexMap[inspectedCompareSide] >= 0
            ? compareInspector.sideIndexMap[inspectedCompareSide]
            : 0
        }
        onSelectTurnIndex={(idx) => {
          if (idx === compareInspector.sideIndexMap.left) setInspectedCompareSide("left");
          else if (idx === compareInspector.sideIndexMap.right) setInspectedCompareSide("right");
        }}
        activeTab={inspectorTab}
        onTabChange={setInspectorTab}
        highlightedCitationId={highlightedCitationId}
        highlightToken={highlightToken}
        onJumpToCitation={handleJumpWithinInspector}
        onClose={() => setInspectorOpen(false)}
        selectorLabel="Side to inspect"
      />
    ) : (
      <GlassBoxInspector
        turns={activeSession.turns}
        selectedTurnIndex={selectedTurnIndex}
        onSelectTurnIndex={setSelectedTurnIndex}
        activeTab={inspectorTab}
        onTabChange={setInspectorTab}
        highlightedCitationId={highlightedCitationId}
        highlightToken={highlightToken}
        onJumpToCitation={handleJumpWithinInspector}
        onClose={() => setInspectorOpen(false)}
      />
    );

  return (
    <div className="flex h-dvh w-screen overflow-hidden bg-desk text-ink">
      {/* Sidebar: docked on desktop, drawer on small screens */}
      {isDesktop
        ? sidebarOpen && sidebar
        : sidebarOpen && (
            <div className="fixed inset-0 z-40 flex">
              <div
                aria-hidden
                className="absolute inset-0 bg-[#0B0F18]/45 backdrop-blur-[1px]"
                onClick={() => setSidebarOpen(false)}
              />
              <div className="anim-sheet-left relative h-full shadow-[var(--shadow-float)]">{sidebar}</div>
            </div>
          )}

      {/* Main column */}
      <div className="flex min-w-0 flex-1 flex-col bg-paper">
        <header className="flex h-12 shrink-0 items-center justify-between gap-2 border-b border-line bg-paper px-2.5 sm:px-3.5">
          <div className="flex min-w-0 items-center gap-1.5">
            {(!sidebarOpen || !isDesktop) && (
              <IconButton
                icon={isDesktop ? PanelLeftOpen : Menu}
                label={isDesktop ? "Open sidebar" : "Open menu"}
                onClick={() => setSidebarOpen(true)}
              />
            )}
            <nav aria-label="Location" className="flex min-w-0 items-center gap-2 pl-1 text-ui">
              <span className={cn("shrink-0", viewSubtitle ? "text-ink-3" : "font-medium text-ink")}>
                {viewName}
              </span>
              {viewSubtitle && (
                <>
                  <span aria-hidden className="hidden shrink-0 text-ink-4 sm:inline">
                    /
                  </span>
                  <span className="hidden truncate font-medium text-ink sm:inline">{viewSubtitle}</span>
                </>
              )}
            </nav>
          </div>

          <div className="flex shrink-0 items-center gap-1">
            <ModelSelectorPopover
              catalog={modelsCatalog}
              catalogLoading={modelsCatalogLoading}
              health={health}
              override={llmOverride}
              providerApiKeys={providerApiKeys}
              onOverrideChange={setLlmOverride}
              onProviderApiKeyChange={handleProviderApiKeyChange}
              onRefreshCatalog={() => void refreshModelsCatalog()}
              useStreaming={useStreaming}
              onUseStreamingChange={setUseStreaming}
              enableBudgetCap={enableBudgetCap}
              onEnableBudgetCapChange={setEnableBudgetCap}
              maxCostUsd={maxCostUsd}
              onMaxCostUsdChange={setMaxCostUsd}
            />
            <IconButton
              icon={theme === "dark" ? Sun : Moon}
              label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
              onClick={toggleTheme}
            />
            {showInspectorToggle && (
              <button
                type="button"
                onClick={() => setInspectorOpen((o) => !o)}
                aria-pressed={inspectorOpen}
                title="Toggle the inspector (⌘I)"
                className={cn(
                  "inline-flex h-8 cursor-pointer items-center gap-1.5 rounded-md px-2 text-ui transition-colors",
                  inspectorOpen ? "bg-selected text-ink" : "text-ink-2 hover:bg-hover hover:text-ink"
                )}
              >
                {inspectorOpen ? (
                  <PanelRightClose className="h-4 w-4" aria-hidden />
                ) : (
                  <PanelRightOpen className="h-4 w-4" aria-hidden />
                )}
                <span className="hidden md:inline">Inspector</span>
                {isDesktop && <Kbd>⌘I</Kbd>}
              </button>
            )}
          </div>
        </header>

        <main className="flex min-h-0 flex-1 overflow-hidden">
          {activeView === "knowledge" ? (
            <KnowledgeBaseView
              apiUrl={apiUrl}
              health={health}
              documentsData={documentsData}
              documentsLoading={documentsLoading}
              documentsError={documentsError}
              onRefreshAll={refreshAll}
            />
          ) : activeView === "compare" ? (
            <CompareArena
              runs={compareRuns}
              activeRunId={activeCompareRunId}
              leftPreset={leftPreset}
              rightPreset={rightPreset}
              inspectedSide={inspectedCompareSide}
              isRunning={compareRunning}
              useStreaming={useStreaming}
              budgetCapLabel={budgetCapLabel}
              onLeftPresetChange={setLeftPreset}
              onRightPresetChange={setRightPreset}
              onSwapPresets={handleSwapPresets}
              onSelectRun={(id) => {
                setActiveCompareRunId(id);
                setHighlightedCitationId(null);
              }}
              onDeleteRun={handleDeleteCompareRun}
              onNewComparison={() => {
                setActiveCompareRunId(null);
                setHighlightedCitationId(null);
              }}
              onSubmitCompare={(q) => void handleSubmitCompare(q)}
              onStopCompare={handleStopCompare}
              onInspectSide={handleInspectCompareSide}
              onCitationClick={handleCompareCitationClick}
              onSubmitSideFeedback={handleSubmitCompareFeedback}
            />
          ) : (
            <ChatWorkspace
              turns={activeSession.turns}
              selectedTurnIndex={selectedTurnIndex}
              isRunning={isRunning}
              pendingQuery={pendingQuery}
              liveSteps={liveSteps}
              liveUsage={liveUsage}
              streamError={streamError}
              presetShortLabel={presetMeta.shortLabel}
              useStreaming={useStreaming}
              budgetCapLabel={budgetCapLabel}
              threadContext={activeSession.context ?? DEFAULT_THREAD_CONTEXT}
              availableDocuments={documentsData?.documents ?? []}
              onThreadContextChange={handleUpdateActiveThreadContext}
              onSubmitQuery={(q) => void handleSubmitQuery(q)}
              onStopStream={handleStopStream}
              onInspectTurn={handleInspectTurn}
              onCitationClick={handleCitationClick}
              onSubmitFeedback={handleSubmitFeedback}
              preset={preset}
              onPresetChange={setPreset}
              customAblation={customAblation}
              onCustomAblationChange={setCustomAblation}
              onOpenKnowledgeBase={() => handleChangeView("knowledge")}
              threadSaveProblem={threadSaveProblem}
            />
          )}
        </main>
      </div>

      {/* Inspector: docked and resizable on wide screens, a sheet otherwise */}
      {inspectorVisible &&
        (isWide ? (
          <div className="relative flex h-full shrink-0 border-l border-line" style={{ width: inspectorWidth }}>
            <div
              role="separator"
              aria-orientation="vertical"
              aria-label="Resize inspector"
              aria-valuemin={INSPECTOR_MIN}
              aria-valuemax={INSPECTOR_MAX}
              aria-valuenow={inspectorWidth}
              tabIndex={0}
              onMouseDown={(e) => {
                e.preventDefault();
                resizingRef.current = true;
                document.body.style.cursor = "col-resize";
                document.body.style.userSelect = "none";
              }}
              onKeyDown={(e) => {
                if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
                  e.preventDefault();
                  const step = e.shiftKey ? 48 : 16;
                  setInspectorWidth((w) =>
                    Math.min(INSPECTOR_MAX, Math.max(INSPECTOR_MIN, w + (e.key === "ArrowLeft" ? step : -step)))
                  );
                }
              }}
              className="group absolute inset-y-0 -left-[3px] z-10 w-[6px] cursor-col-resize focus-visible:outline-none"
            >
              <span className="absolute inset-y-0 left-[2px] w-[2px] bg-transparent transition-colors group-hover:bg-accent group-focus-visible:bg-accent" />
            </div>
            <div className="min-w-0 flex-1">{inspector}</div>
          </div>
        ) : (
          <div className="fixed inset-0 z-40 flex justify-end">
            <div
              aria-hidden
              className="absolute inset-0 bg-[#0B0F18]/45 backdrop-blur-[1px]"
              onClick={() => setInspectorOpen(false)}
            />
            <div className="anim-sheet-right relative h-full w-full max-w-[480px] shadow-[var(--shadow-float)]">
              {inspector}
            </div>
          </div>
        ))}
    </div>
  );
}
