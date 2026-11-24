import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Database,
  Moon,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  Sun,
} from "lucide-react";
import { ChatWorkspace } from "./components/ChatWorkspace";
import { GlassBoxInspector } from "./components/GlassBoxInspector";
import { KnowledgeBaseView } from "./components/KnowledgeBaseView";
import { Sidebar } from "./components/Sidebar";
import {
  DEFAULT_API_URL,
  DEFAULT_CUSTOM_ABLATION,
  PRESET_OPTIONS,
} from "./constants/pipeline";
import {
  buildQueryPayload,
  fetchDocuments,
  fetchHealth,
  runQueryBatch,
  streamQuery,
  submitFeedback,
} from "./lib/api";
import type {
  AblationConfig,
  AppView,
  ChatMessage,
  ChatSession,
  ChatTurn,
  DocumentListResponse,
  FinalAnswer,
  HealthResponse,
  InspectorTab,
  LiveStreamStep,
  PresetSelection,
  TokenUsage,
} from "./types/api";

const STORAGE_SESSIONS_KEY = "adaptiverag_studio_sessions_v1";
const STORAGE_THEME_KEY = "adaptiverag_studio_theme_v1";
const STORAGE_API_URL_KEY = "adaptiverag_studio_api_url_v1";

function createEmptySession(): ChatSession {
  const now = Date.now();
  return {
    id: `sess_${now}_${Math.random().toString(36).slice(2, 7)}`,
    title: "New Research Session",
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
  const [inspectorWidth, setInspectorWidth] = useState(440);
  const resizingRef = useRef(false);

  // API & Health & Knowledge Base state
  const [apiUrl, setApiUrl] = useState<string>(() => {
    try {
      return localStorage.getItem(STORAGE_API_URL_KEY) || DEFAULT_API_URL;
    } catch {
      return DEFAULT_API_URL;
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
    () => loadInitialSessions()[0].id
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

  // Persist API URL
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_API_URL_KEY, apiUrl);
    } catch {
      // ignore
    }
  }, [apiUrl]);

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
  }, [apiUrl]);

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
  }, [apiUrl]);

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

    const payload = buildQueryPayload({
      query,
      chatHistory,
      preset,
      customAblation,
      maxCostUsd: enableBudgetCap ? maxCostUsd : null,
      sessionId: targetSessionId,
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

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[var(--bg-canvas)] text-[var(--text-primary)]">
      {/* Left Navigation & Controls Sidebar */}
      {sidebarOpen && (
        <Sidebar
          activeView={activeView}
          onChangeView={setActiveView}
          sessions={sessions}
          activeSessionId={activeSession.id}
          onSelectSession={handleSelectSession}
          onNewSession={handleNewSession}
          onDeleteSession={handleDeleteSession}
          apiUrl={apiUrl}
          onApiUrlChange={(url) => setApiUrl(url)}
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
        <header className="flex h-13 shrink-0 items-center justify-between gap-3 border-b border-[var(--border-subtle)] bg-[var(--bg-panel)] px-4">
          <div className="flex min-w-0 items-center gap-2.5">
            {!sidebarOpen && (
              <button
                type="button"
                onClick={() => setSidebarOpen(true)}
                title="Open sidebar"
                className="rounded-lg p-1.5 text-[var(--text-muted)] transition hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)] cursor-pointer"
              >
                <PanelLeftOpen className="h-4 w-4" />
              </button>
            )}
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="truncate text-xs font-semibold text-[var(--text-primary)]">
                  {activeView === "workbench"
                    ? activeSession.title
                    : "Knowledge Base & Ingestion Studio"}
                </span>
                {activeView === "workbench" && (
                  <span className="hidden rounded-full border border-blue-500/30 bg-blue-500/10 px-2 py-0.5 font-mono text-[10px] font-medium text-blue-600 sm:inline dark:text-blue-300">
                    {presetMeta.shortLabel}
                  </span>
                )}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={() =>
                setActiveView((v) => (v === "workbench" ? "knowledge" : "workbench"))
              }
              className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition cursor-pointer ${
                activeView === "knowledge"
                  ? "border-blue-500/40 bg-blue-500/12 text-blue-600 dark:text-blue-300"
                  : "border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60 text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
              }`}
            >
              <Database className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">
                {activeView === "knowledge" ? "Back to Workbench" : "Knowledge Base"}
              </span>
            </button>

            <button
              type="button"
              onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
              title={theme === "dark" ? "Switch to Light theme" : "Switch to Dark theme"}
              aria-label="Toggle color theme"
              className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60 p-1.5 text-[var(--text-secondary)] transition hover:text-[var(--text-primary)] cursor-pointer"
            >
              {theme === "dark" ? (
                <Sun className="h-4 w-4" />
              ) : (
                <Moon className="h-4 w-4" />
              )}
            </button>

            {activeView === "workbench" && (
              <button
                type="button"
                onClick={() => setInspectorOpen((o) => !o)}
                title="Toggle Glass-Box Inspector (⌘I)"
                className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition cursor-pointer ${
                  inspectorOpen
                    ? "border-blue-500/40 bg-blue-500/12 text-blue-600 dark:text-blue-300"
                    : "border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60 text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                }`}
              >
                {inspectorOpen ? (
                  <PanelRightClose className="h-3.5 w-3.5" />
                ) : (
                  <PanelRightOpen className="h-3.5 w-3.5" />
                )}
                <span className="hidden md:inline">Inspector</span>
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
                onSubmitQuery={(q) => void handleSubmitQuery(q)}
                onStopStream={handleStopStream}
                onInspectTurn={handleInspectTurn}
                onCitationClick={handleCitationClick}
                onSubmitFeedback={handleSubmitFeedback}
              />

              {/* Resizable Glass-Box Inspector */}
              {inspectorOpen && (
                <>
                  <div
                    role="separator"
                    aria-orientation="vertical"
                    onMouseDown={() => {
                      resizingRef.current = true;
                      document.body.style.cursor = "col-resize";
                    }}
                    className="hidden w-1 shrink-0 cursor-col-resize bg-transparent transition hover:bg-blue-500/40 lg:block"
                  />
                  <div
                    style={{ width: `${inspectorWidth}px` }}
                    className="hidden h-full shrink-0 lg:block"
                  >
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
