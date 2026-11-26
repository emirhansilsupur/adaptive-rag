import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Columns2,
  Database,
  MessageSquare,
  Moon,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  Sun,
} from "lucide-react";
import { ChatWorkspace } from "./components/ChatWorkspace";
import { CompareArena } from "./components/CompareArena";
import { GlassBoxInspector } from "./components/GlassBoxInspector";
import { KnowledgeBaseView } from "./components/KnowledgeBaseView";
import { Sidebar } from "./components/Sidebar";
import {
  DEFAULT_API_URL,
  DEFAULT_CUSTOM_ABLATION,
  DEFAULT_THREAD_CONTEXT,
  getInstructionPresetLabel,
  PRESET_OPTIONS,
  resolveEffectiveInstructions,
} from "./constants/pipeline";
import {
  buildQueryPayload,
  fetchDocuments,
  fetchHealth,
  runQueryBatch,
  setStudioApiKey,
  streamQuery,
  submitFeedback,
} from "./lib/api";
import {
  createCompareInspectorTurns,
  getPresetShortLabel,
} from "./lib/compare";
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
  PresetSelection,
  ThreadContextConfig,
  TokenUsage,
} from "./types/api";

const STORAGE_SESSIONS_KEY = "adaptiverag_studio_sessions_v1";
const STORAGE_COMPARE_KEY = "adaptiverag_studio_compare_v1";
const STORAGE_THEME_KEY = "adaptiverag_studio_theme_v1";
const STORAGE_API_URL_KEY = "adaptiverag_studio_api_url_v1";
const SESSION_API_KEY_STORAGE_KEY = "adaptiverag_studio_api_key_v1";

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
    title: "Untitled Thread",
    createdAt: now,
    updatedAt: now,
    turns: [],
  };
}

function loadInitialSessions(): ChatSession[] {
  try {
    const raw = localStorage.getItem(STORAGE_SESSIONS_KEY);
    if (!raw) return [createEmptySession()];
    const parsed = JSON.parse(raw) as ChatSession[];
    if (Array.isArray(parsed) && parsed.length > 0) {
      return parsed;
    }
  } catch {
    // fallback to clean session
  }
  return [createEmptySession()];
}

export default function App() {
  // Theme (dark-first with persistent toggle)
  const [theme, setTheme] = useState<"dark" | "light">(() => {
    try {
      const saved = localStorage.getItem(STORAGE_THEME_KEY);
      if (saved === "light" || saved === "dark") return saved;
    } catch {
      // ignore storage errors
    }
    return "dark";
  });

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "dark") {
      root.classList.add("dark");
    } else {
      root.classList.remove("dark");
    }
    try {
      localStorage.setItem(STORAGE_THEME_KEY, theme);
    } catch {
      // ignore
    }
  }, [theme]);

  // Layout state
  const [activeView, setActiveView] = useState<AppView>("workbench");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [inspectorWidth, setInspectorWidth] = useState(432);
  const resizingRef = useRef(false);

  // API & Health & Knowledge Base state
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

  // Sessions & turns
  const [sessions, setSessions] = useState<ChatSession[]>(loadInitialSessions);
  const [activeSessionId, setActiveSessionId] = useState<string>(
    () => sessions[0].id
  );
  const [selectedTurnIndex, setSelectedTurnIndex] = useState<number>(0);
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>("trace");
  const [highlightedCitationId, setHighlightedCitationId] = useState<string | null>(null);
  const [highlightToken, setHighlightToken] = useState<number>(0);

  // Pipeline controls
  const [preset, setPreset] = useState<PresetSelection>("full");
  const [customAblation, setCustomAblation] = useState<AblationConfig>(
    DEFAULT_CUSTOM_ABLATION
  );
  const [useStreaming, setUseStreaming] = useState(true);
  const [enableBudgetCap, setEnableBudgetCap] = useState(false);
  const [maxCostUsd, setMaxCostUsd] = useState(0.05);

  // Active streaming execution state
  const [isRunning, setIsRunning] = useState(false);
  const [pendingQuery, setPendingQuery] = useState<string | null>(null);
  const [liveSteps, setLiveSteps] = useState<LiveStreamStep[]>([]);
  const [liveUsage, setLiveUsage] = useState<Partial<TokenUsage> | null>(null);
  const [streamError, setStreamError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Side-by-Side Ablation Arena (Compare Mode) state
  const [compareRuns, setCompareRuns] = useState<CompareRun[]>(loadInitialCompareRuns);
  const [activeCompareRunId, setActiveCompareRunId] = useState<string | null>(
    () => loadInitialCompareRuns()[0]?.id ?? null
  );
  const [leftPreset, setLeftPreset] = useState<AblationPreset>("naive_dense");
  const [rightPreset, setRightPreset] = useState<AblationPreset>("full");
  const [compareRunning, setCompareRunning] = useState(false);
  const [inspectedCompareSide, setInspectedCompareSide] =
    useState<CompareSideKey>("right");
  const compareAbortRef = useRef<AbortController | null>(null);

  const activeCompareRun = useMemo(
    () =>
      activeCompareRunId
        ? compareRuns.find((r) => r.id === activeCompareRunId) ?? null
        : null,
    [compareRuns, activeCompareRunId]
  );

  const compareInspector = useMemo(
    () => createCompareInspectorTurns(activeCompareRun),
    [activeCompareRun]
  );

  // Persist completed Compare Arena runs
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

  // Ensure activeSessionId always resolves to an existing session
  const activeSession = useMemo(
    () => sessions.find((s) => s.id === activeSessionId) ?? sessions[0],
    [sessions, activeSessionId]
  );

  // Persist sessions in localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_SESSIONS_KEY, JSON.stringify(sessions.slice(0, 20)));
    } catch {
      // ignore quota errors
    }
  }, [sessions]);

  // Persist API URL & session-scoped API Key
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_API_URL_KEY, apiUrl);
    } catch {
      // ignore
    }
  }, [apiUrl]);

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

  const refreshAll = useCallback(async () => {
    await Promise.all([refreshHealth(), refreshDocuments()]);
  }, [refreshHealth, refreshDocuments]);

  useEffect(() => {
    void refreshAll();
  }, [refreshAll]);

  // Keyboard shortcut: Cmd+I / Ctrl+I toggles Inspector
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "i") {
        e.preventDefault();
        setInspectorOpen((prev) => !prev);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  // Drag resize for right inspector pane
  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (!resizingRef.current) return;
      const nextWidth = Math.min(640, Math.max(340, window.innerWidth - e.clientX));
      setInspectorWidth(nextWidth);
    };
    const onMouseUp = () => {
      resizingRef.current = false;
      document.body.style.cursor = "";
    };
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }, []);

  const handleNewSession = () => {
    const fresh = createEmptySession();
    setSessions((prev) => [fresh, ...prev]);
    setActiveSessionId(fresh.id);
    setSelectedTurnIndex(0);
    setHighlightedCitationId(null);
    setStreamError(null);
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
  };

  const handleUpdateActiveThreadContext = (nextContext: ThreadContextConfig) => {
    const targetId = activeSession.id;
    setSessions((prev) =>
      prev.map((sess) =>
        sess.id === targetId
          ? {
              ...sess,
              context: nextContext,
              updatedAt: Date.now(),
            }
          : sess
      )
    );
  };

  const handleInspectTurn = (turnIdx: number, tab?: InspectorTab) => {
    setSelectedTurnIndex(turnIdx);
    if (tab) {
      setInspectorTab(tab);
    }
    setInspectorOpen(true);
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
    if (nextView === "compare" && typeof window !== "undefined" && window.innerWidth < 1540) {
      setInspectorOpen(false);
    } else if (nextView === "workbench") {
      setInspectorOpen(true);
    }
  };

  const handleInspectCompareSide = (side: CompareSideKey, tab?: InspectorTab) => {
    setInspectedCompareSide(side);
    if (tab) {
      setInspectorTab(tab);
    }
    setInspectorOpen(true);
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
      setCompareRuns((prev) =>
        prev.map((r) => (r.id === runId ? { ...r, [side]: updater(r[side]) } : r))
      );
    };

    const executeSide = async (side: CompareSideKey, sidePreset: AblationPreset) => {
      const payload = buildQueryPayload({
        query,
        chatHistory: [],
        preset: sidePreset,
        customAblation,
        maxCostUsd: enableBudgetCap ? maxCostUsd : null,
        sessionId: `${runId}_${side}`,
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
            controller.signal
          );
        } else {
          answer = await runQueryBatch(apiUrl, payload, controller.signal);
        }

        updateSide(side, (prev) => ({
          ...prev,
          status: "done",
          result: answer,
          error: null,
        }));
      } catch (err) {
        const message =
          (err as Error)?.name === "AbortError"
            ? "Comparison cancelled by user."
            : err instanceof Error
              ? err.message
              : String(err);
        updateSide(side, (prev) => ({
          ...prev,
          status: "error",
          error: message,
        }));
      }
    };

    try {
      await Promise.allSettled([
        executeSide("left", leftPreset),
        executeSide("right", rightPreset),
      ]);
    } finally {
      setCompareRunning(false);
      compareAbortRef.current = null;
    }
  };

  const handleSubmitFeedback = async (
    turnId: string,
    score: 0 | 1,
    comment?: string
  ) => {
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

  const presetMeta =
    PRESET_OPTIONS.find((p) => p.value === preset) ?? PRESET_OPTIONS[0];

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
      maxCostUsd: enableBudgetCap ? maxCostUsd : null,
      sessionId: targetSessionId,
      customInstructions: effectiveInstructions,
      docIds: scopedDocIds,
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
          controller.signal
        );
      } else {
        answer = await runQueryBatch(apiUrl, payload, controller.signal);
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
            sess.turns.length === 0
              ? query.length > 44
                ? `${query.slice(0, 44)}…`
                : query
              : sess.title;
          return {
            ...sess,
            title: updatedTitle,
            updatedAt: Date.now(),
            turns: updatedTurns,
          };
        })
      );
      setSelectedTurnIndex(nextTurnIdx);
      setHighlightedCitationId(null);
    } catch (err) {
      if ((err as Error)?.name === "AbortError") {
        setStreamError("Pipeline stream cancelled by user.");
      } else {
        setStreamError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      setIsRunning(false);
      setPendingQuery(null);
      abortControllerRef.current = null;
    }
  };

  const viewBreadcrumb =
    activeView === "workbench"
      ? "Workbench"
      : activeView === "compare"
        ? "Ablation Arena"
        : "Knowledge Base";

  const viewSubtitle =
    activeView === "workbench"
      ? activeSession.title
      : activeView === "compare"
        ? `${getPresetShortLabel(leftPreset)} vs ${getPresetShortLabel(rightPreset)}`
        : `${documentsData?.total_documents ?? 0} docs · ${health?.qdrant?.points_count ?? 0} chunks`;

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[var(--bg-canvas)] text-[var(--text-primary)]">
      {/* Left Navigation & Controls Sidebar */}
      {sidebarOpen && (
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
          preset={preset}
          onPresetChange={setPreset}
          customAblation={customAblation}
          onCustomAblationChange={setCustomAblation}
          useStreaming={useStreaming}
          onUseStreamingChange={setUseStreaming}
          enableBudgetCap={enableBudgetCap}
          onEnableBudgetCapChange={setEnableBudgetCap}
          maxCostUsd={maxCostUsd}
          onMaxCostUsdChange={setMaxCostUsd}
          onCollapse={() => setSidebarOpen(false)}
        />
      )}

      {/* Center + Right Split Container */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* Global Top Header Bar */}
        <header className="flex h-12 shrink-0 items-center justify-between gap-3 border-b border-[var(--border-subtle)] bg-[var(--bg-panel)] px-3.5">
          <div className="flex min-w-0 items-center gap-2">
            {!sidebarOpen && (
              <button
                type="button"
                onClick={() => setSidebarOpen(true)}
                title="Open sidebar"
                className="rounded-md p-1.5 text-[var(--text-muted)] transition hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)] cursor-pointer"
              >
                <PanelLeftOpen className="h-3.5 w-3.5" />
              </button>
            )}

            <nav aria-label="Workspace breadcrumb" className="flex min-w-0 items-center gap-1.5 text-xs">
              <span className="shrink-0 font-medium text-[var(--text-muted)]">
                {viewBreadcrumb}
              </span>
              <span className="shrink-0 text-[var(--text-muted)]/50">/</span>
              <span className="truncate font-medium text-[var(--text-primary)]">
                {viewSubtitle}
              </span>
            </nav>

            {activeView === "workbench" && (
              <>
                <span className="hidden shrink-0 rounded border border-[var(--border-subtle)] bg-[var(--bg-surface)] px-1.5 py-0.5 font-mono text-[10px] font-medium text-[var(--text-secondary)] sm:inline">
                  {presetMeta.shortLabel}
                </span>
                {resolveEffectiveInstructions(activeSession.context) && (
                  <span className="hidden shrink-0 rounded border border-emerald-500/35 bg-emerald-500/10 px-1.5 py-0.5 font-mono text-[10px] font-medium text-emerald-700 dark:text-emerald-300 md:inline">
                    style:{getInstructionPresetLabel(activeSession.context)}
                  </span>
                )}
                {(activeSession.context?.scopedDocIds?.length ?? 0) > 0 && (
                  <span className="hidden shrink-0 rounded border border-sky-500/35 bg-sky-500/10 px-1.5 py-0.5 font-mono text-[10px] font-medium text-sky-700 dark:text-sky-300 md:inline">
                    {activeSession.context?.scopedDocIds?.length} doc(s)
                  </span>
                )}
              </>
            )}
          </div>

          <div className="flex items-center gap-1.5">
            {/* Compact mode switcher shown only when sidebar is collapsed */}
            {!sidebarOpen && (
              <div className="mr-1 inline-flex items-center rounded-md border border-[var(--border-subtle)] bg-[var(--bg-surface)] p-0.5">
                {(
                  [
                    ["workbench", "Chat", MessageSquare],
                    ["compare", "Arena", Columns2],
                    ["knowledge", "KB", Database],
                  ] as const
                ).map(([id, lbl, Icon]) => (
                  <button
                    key={id}
                    type="button"
                    onClick={() => handleChangeView(id)}
                    className={`inline-flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium transition cursor-pointer ${
                      activeView === id
                        ? "bg-[var(--bg-elevated)] text-[var(--text-primary)]"
                        : "text-[var(--text-muted)] hover:text-[var(--text-primary)]"
                    }`}
                  >
                    <Icon className="h-3 w-3" />
                    <span className="hidden sm:inline">{lbl}</span>
                  </button>
                ))}
              </div>
            )}

            <button
              type="button"
              onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
              title={theme === "dark" ? "Switch to Light theme" : "Switch to Dark theme"}
              aria-label="Toggle color theme"
              className="flex h-7.5 w-7.5 items-center justify-center rounded-md border border-[var(--border-subtle)] bg-[var(--bg-surface)] text-[var(--text-secondary)] transition hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)] cursor-pointer"
            >
              {theme === "dark" ? (
                <Sun className="h-3.5 w-3.5" />
              ) : (
                <Moon className="h-3.5 w-3.5" />
              )}
            </button>

            {(activeView === "workbench" || activeView === "compare") && (
              <button
                type="button"
                onClick={() => setInspectorOpen((o) => !o)}
                title="Toggle Inspector (⌘I)"
                className={`inline-flex h-7.5 items-center gap-1.5 rounded-md border px-2 text-xs font-medium transition cursor-pointer ${
                  inspectorOpen
                    ? "border-[var(--border-strong)] bg-[var(--bg-elevated)] text-[var(--text-primary)]"
                    : "border-[var(--border-subtle)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                }`}
              >
                {inspectorOpen ? (
                  <PanelRightClose className="h-3.5 w-3.5" />
                ) : (
                  <PanelRightOpen className="h-3.5 w-3.5" />
                )}
                <span className="hidden md:inline">Inspector</span>
                <kbd className="ar-kbd hidden md:inline-flex">⌘I</kbd>
              </button>
            )}
          </div>
        </header>

        {/* Workspace Content */}
        <div className="flex min-h-0 flex-1 overflow-hidden">
          {activeView === "knowledge" ? (
            <KnowledgeBaseView
              apiUrl={apiUrl}
              health={health}
              documentsData={documentsData}
              documentsLoading={documentsLoading}
              documentsError={documentsError}
              onRefreshAll={refreshAll}
            />
          ) : (
            <>
              {activeView === "compare" ? (
                <CompareArena
                  runs={compareRuns}
                  activeRunId={activeCompareRunId}
                  leftPreset={leftPreset}
                  rightPreset={rightPreset}
                  inspectedSide={inspectedCompareSide}
                  isRunning={compareRunning}
                  useStreaming={useStreaming}
                  budgetCapLabel={enableBudgetCap ? `$${maxCostUsd.toFixed(2)}` : null}
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
                  budgetCapLabel={enableBudgetCap ? `$${maxCostUsd.toFixed(2)}` : null}
                  threadContext={activeSession.context ?? DEFAULT_THREAD_CONTEXT}
                  availableDocuments={documentsData?.documents ?? []}
                  onThreadContextChange={handleUpdateActiveThreadContext}
                  onSubmitQuery={(q) => void handleSubmitQuery(q)}
                  onStopStream={handleStopStream}
                  onInspectTurn={handleInspectTurn}
                  onCitationClick={handleCitationClick}
                  onSubmitFeedback={handleSubmitFeedback}
                />
              )}

              {/* Resizable Inspector */}
              {inspectorOpen && (
                <>
                  <div
                    role="separator"
                    aria-orientation="vertical"
                    onMouseDown={() => {
                      resizingRef.current = true;
                      document.body.style.cursor = "col-resize";
                    }}
                    className="hidden w-1 shrink-0 cursor-col-resize bg-transparent transition hover:bg-[var(--border-strong)] lg:block"
                  />
                  <div
                    style={{ width: `${inspectorWidth}px` }}
                    className="hidden h-full shrink-0 lg:block"
                  >
                    {activeView === "compare" ? (
                      <GlassBoxInspector
                        turns={compareInspector.turns}
                        selectedTurnIndex={
                          compareInspector.sideIndexMap[inspectedCompareSide] >= 0
                            ? compareInspector.sideIndexMap[inspectedCompareSide]
                            : 0
                        }
                        onSelectTurnIndex={(idx) => {
                          if (idx === compareInspector.sideIndexMap.left) {
                            setInspectedCompareSide("left");
                          } else if (idx === compareInspector.sideIndexMap.right) {
                            setInspectedCompareSide("right");
                          }
                        }}
                        activeTab={inspectorTab}
                        onTabChange={setInspectorTab}
                        highlightedCitationId={highlightedCitationId}
                        highlightToken={highlightToken}
                        onJumpToCitation={handleJumpWithinInspector}
                        onClose={() => setInspectorOpen(false)}
                        selectorLabel="Inspecting Side"
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
                    )}
                  </div>
                </>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
