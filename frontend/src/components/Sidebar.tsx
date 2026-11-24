import React, { useState } from "react";
import {
  Activity,
  ChevronDown,
  ChevronRight,
  Columns2,
  Database,
  Flame,
  MessageSquare,
  PanelLeftClose,
  Plus,
  RefreshCw,
  Server,
  SlidersHorizontal,
  Trash2,
} from "lucide-react";
import { PRESET_OPTIONS } from "../constants/pipeline";
import type {
  AblationConfig,
  AppView,
  ChatSession,
  HealthResponse,
  PresetSelection,
} from "../types/api";

interface SidebarProps {
  activeView: AppView;
  onChangeView: (view: AppView) => void;
  sessions: ChatSession[];
  activeSessionId: string;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  onDeleteSession: (id: string) => void;
  apiUrl: string;
  onApiUrlChange: (url: string) => void;
  health: HealthResponse | null;
  healthError: string | null;
  healthLoading: boolean;
  onRefreshHealth: () => void;
  preset: PresetSelection;
  onPresetChange: (preset: PresetSelection) => void;
  customAblation: AblationConfig;
  onCustomAblationChange: (cfg: AblationConfig) => void;
  useStreaming: boolean;
  onUseStreamingChange: (val: boolean) => void;
  enableBudgetCap: boolean;
  onEnableBudgetCapChange: (val: boolean) => void;
  maxCostUsd: number;
  onMaxCostUsdChange: (val: number) => void;
  onCollapse: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeView,
  onChangeView,
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  apiUrl,
  onApiUrlChange,
  health,
  healthError,
  healthLoading,
  onRefreshHealth,
  preset,
  onPresetChange,
  customAblation,
  onCustomAblationChange,
  useStreaming,
  onUseStreamingChange,
  enableBudgetCap,
  onEnableBudgetCapChange,
  maxCostUsd,
  onMaxCostUsdChange,
  onCollapse,
}) => {
  const [controlsOpen, setControlsOpen] = useState(true);
  const [healthExpanded, setHealthExpanded] = useState(false);
  const [apiInput, setApiInput] = useState(apiUrl);

  const qdrantPoints = health?.qdrant?.points_count ?? 0;
  const fcStatus = health?.firecrawl?.status;
  const lfMode = health?.langfuse?.mode ?? (health?.langfuse_enabled ? "cloud" : "disabled");
  const lfLabel =
    lfMode === "cloud"
      ? "Cloud"
      : lfMode === "self_hosted"
        ? "Self-Hosted"
        : lfMode === "misconfigured"
          ? "Keys Missing"
          : "Off";
  const statusTone = !health
    ? "bg-rose-500"
    : health.status === "ok"
      ? "bg-emerald-500"
      : "bg-amber-500";

  const updateCustom = <K extends keyof AblationConfig>(
    key: K,
    val: AblationConfig[K]
  ) => {
    onCustomAblationChange({ ...customAblation, [key]: val });
  };

  return (
    <aside className="flex h-full w-[280px] shrink-0 flex-col border-r border-[var(--border-subtle)] bg-[var(--bg-panel)]">
      {/* Brand Header */}
      <div className="flex h-13 shrink-0 items-center justify-between border-b border-[var(--border-subtle)] px-3.5">
        <div className="flex items-center gap-2.5">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-blue-600 to-indigo-600 text-white shadow-sm">
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2.2}>
              <path d="M12 3L4 8v8l8 5 8-5V8l-8-5z" strokeLinejoin="round" />
              <path d="M12 12l8-4M12 12v9M12 12L4 8" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
          <div>
            <div className="text-xs font-bold tracking-tight text-[var(--text-primary)]">
              AdaptiveRAG
            </div>
            <div className="text-[10px] font-medium text-[var(--text-muted)]">
              Glass-Box Studio
            </div>
          </div>
        </div>
        <button
          type="button"
          onClick={onCollapse}
          title="Collapse sidebar"
          className="rounded-lg p-1 text-[var(--text-muted)] transition hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)] cursor-pointer"
        >
          <PanelLeftClose className="h-4 w-4" />
        </button>
      </div>

      {/* Primary Action & View Navigation */}
      <div className="space-y-1.5 border-b border-[var(--border-subtle)] p-3">
        <button
          type="button"
          onClick={() => {
            onNewSession();
            onChangeView("workbench");
          }}
          className="flex w-full items-center justify-center gap-1.5 rounded-xl bg-blue-600 px-3 py-2 text-xs font-semibold text-white shadow-xs transition hover:bg-blue-500 cursor-pointer"
        >
          <Plus className="h-3.5 w-3.5" />
          <span>New Session</span>
        </button>

        <div className="grid grid-cols-3 gap-1 pt-1">
          <button
            type="button"
            onClick={() => onChangeView("workbench")}
            data-nav="workbench"
            className={`flex items-center justify-center gap-1 rounded-lg px-1.5 py-1.5 text-[11.5px] font-medium transition cursor-pointer ${
              activeView === "workbench"
                ? "bg-[var(--bg-elevated)] text-[var(--text-primary)] shadow-2xs ring-1 ring-[var(--border-subtle)]"
                : "text-[var(--text-secondary)] hover:bg-[var(--bg-elevated)]/60"
            }`}
          >
            <MessageSquare className="h-3.5 w-3.5 shrink-0" />
            <span>Chat</span>
          </button>

          <button
            type="button"
            onClick={() => onChangeView("compare")}
            data-nav="compare"
            className={`flex items-center justify-center gap-1 rounded-lg px-1.5 py-1.5 text-[11.5px] font-medium transition cursor-pointer ${
              activeView === "compare"
                ? "bg-[var(--bg-elevated)] text-indigo-600 shadow-2xs ring-1 ring-indigo-500/35 dark:text-indigo-300"
                : "text-[var(--text-secondary)] hover:bg-[var(--bg-elevated)]/60"
            }`}
          >
            <Columns2 className="h-3.5 w-3.5 shrink-0" />
            <span>Arena</span>
          </button>

          <button
            type="button"
            onClick={() => onChangeView("knowledge")}
            data-nav="knowledge"
            className={`flex items-center justify-center gap-1 rounded-lg px-1.5 py-1.5 text-[11.5px] font-medium transition cursor-pointer ${
              activeView === "knowledge"
                ? "bg-[var(--bg-elevated)] text-[var(--text-primary)] shadow-2xs ring-1 ring-[var(--border-subtle)]"
                : "text-[var(--text-secondary)] hover:bg-[var(--bg-elevated)]/60"
            }`}
          >
            <Database className="h-3.5 w-3.5 shrink-0" />
            <span>KB</span>
            <span className="rounded bg-blue-500/15 px-1 font-mono text-[9.5px] font-semibold text-blue-500 dark:text-blue-400 tabular-nums">
              {qdrantPoints}
            </span>
          </button>
        </div>
      </div>

      {/* Scrollable Middle: Sessions + Pipeline Controls */}
      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-4">
        {/* Sessions */}
        <div>
          <div className="mb-1.5 px-1 text-[10.5px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
            Sessions ({sessions.length})
          </div>
          <div className="space-y-0.5">
            {sessions.map((sess) => {
              const isActive = sess.id === activeSessionId && activeView === "workbench";
              return (
                <div
                  key={sess.id}
                  className={`group flex items-center justify-between gap-1 rounded-lg px-2.5 py-1.5 text-xs transition ${
                    isActive
                      ? "bg-blue-500/12 text-blue-600 font-medium dark:text-blue-300"
                      : "text-[var(--text-secondary)] hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)]"
                  }`}
                >
                  <button
                    type="button"
                    onClick={() => {
                      onSelectSession(sess.id);
                      onChangeView("workbench");
                    }}
                    className="flex min-w-0 flex-1 items-center gap-2 text-left cursor-pointer"
                  >
                    <MessageSquare className="h-3.5 w-3.5 shrink-0 opacity-65" />
                    <span className="truncate">{sess.title}</span>
                  </button>
                  <span className="shrink-0 font-mono text-[10px] opacity-60 tabular-nums">
                    {sess.turns.length}
                  </span>
                  {sessions.length > 1 && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteSession(sess.id);
                      }}
                      title="Delete session"
                      className="hidden shrink-0 rounded p-0.5 text-[var(--text-muted)] hover:text-rose-500 group-hover:inline-flex cursor-pointer"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Pipeline & Ablation Controls */}
        <div className="rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/45">
          <button
            type="button"
            onClick={() => setControlsOpen((p) => !p)}
            className="flex w-full items-center justify-between px-3 py-2 text-[11px] font-semibold uppercase tracking-wider text-[var(--text-secondary)] cursor-pointer"
          >
            <span className="inline-flex items-center gap-1.5">
              <SlidersHorizontal className="h-3.5 w-3.5 text-blue-400" />
              Pipeline Controls
            </span>
            {controlsOpen ? (
              <ChevronDown className="h-3.5 w-3.5" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5" />
            )}
          </button>

          {controlsOpen && (
            <div className="space-y-3 border-t border-[var(--border-subtle)] px-3 py-2.5 text-xs">
              {activeView === "compare" && (
                <div className="rounded-lg border border-indigo-500/30 bg-indigo-500/10 px-2.5 py-2 text-[10.5px] leading-snug text-indigo-700 dark:text-indigo-300">
                  Arena A/B presets are selected in the top bar. Streaming &amp; USD
                  cap below apply to both sides.
                </div>
              )}

              <div>
                <label className="block text-[10.5px] font-medium text-[var(--text-muted)]">
                  Ablation Preset
                </label>
                <select
                  value={preset}
                  onChange={(e) => onPresetChange(e.target.value as PresetSelection)}
                  className="mt-1 w-full rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-panel)] px-2 py-1.5 text-xs text-[var(--text-primary)] focus:border-blue-500 focus:outline-none"
                >
                  {PRESET_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
                <p className="mt-1 text-[10.5px] leading-snug text-[var(--text-muted)]">
                  {PRESET_OPTIONS.find((o) => o.value === preset)?.description}
                </p>
              </div>

              {preset === "custom" && (
                <div className="space-y-2 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-panel)] p-2.5 text-[11px]">
                  <div>
                    <label className="text-[10px] text-[var(--text-muted)]">
                      Retrieval Mode
                    </label>
                    <select
                      value={customAblation.retrieval_mode}
                      onChange={(e) =>
                        updateCustom(
                          "retrieval_mode",
                          e.target.value as AblationConfig["retrieval_mode"]
                        )
                      }
                      className="mt-0.5 w-full rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)] px-2 py-1 text-[11px]"
                    >
                      <option value="hybrid">Hybrid (Dense + BM25 RRF)</option>
                      <option value="dense_only">Dense Only</option>
                      <option value="sparse_only">BM25 Sparse Only</option>
                    </select>
                  </div>

                  {(
                    [
                      ["enable_query_routing", "Query Routing"],
                      ["enable_multi_query", "Multi-Query Decomp."],
                      ["enable_rerank", "Cross-Encoder Rerank"],
                      ["enable_dynamic_k", "3-Stage Dynamic Top-K"],
                      ["enable_confidence_gating", "Confidence + CRAG"],
                      ["enable_web_fallback", "Web Search Fallback"],
                      ["enable_citation_validation", "Two-Tier Cite Audit"],
                    ] as const
                  ).map(([key, lbl]) => (
                    <label
                      key={key}
                      className="flex items-center justify-between gap-2 cursor-pointer"
                    >
                      <span className="text-[var(--text-secondary)]">{lbl}</span>
                      <input
                        type="checkbox"
                        checked={customAblation[key]}
                        onChange={(e) => updateCustom(key, e.target.checked)}
                        className="h-3.5 w-3.5 accent-blue-600"
                      />
                    </label>
                  ))}

                  {!customAblation.enable_dynamic_k && (
                    <div>
                      <div className="flex justify-between text-[10.5px]">
                        <span className="text-[var(--text-muted)]">Static Top-K</span>
                        <span className="font-mono font-semibold">
                          {customAblation.fixed_top_k}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={1}
                        max={15}
                        value={customAblation.fixed_top_k}
                        onChange={(e) =>
                          updateCustom("fixed_top_k", Number(e.target.value))
                        }
                        className="w-full accent-blue-600"
                      />
                    </div>
                  )}

                  <div>
                    <label className="text-[10px] text-[var(--text-muted)]">
                      Corpus Language Filter
                    </label>
                    <select
                      value={customAblation.language_filter ?? "all"}
                      onChange={(e) =>
                        updateCustom(
                          "language_filter",
                          e.target.value === "all"
                            ? null
                            : (e.target.value as "en" | "tr")
                        )
                      }
                      className="mt-0.5 w-full rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)] px-2 py-1 text-[11px]"
                    >
                      <option value="all">All (TR + EN)</option>
                      <option value="en">English Only</option>
                      <option value="tr">Turkish Only</option>
                    </select>
                  </div>
                </div>
              )}

              <div className="space-y-2 border-t border-[var(--border-subtle)] pt-2">
                <label className="flex items-center justify-between gap-2 cursor-pointer">
                  <span className="text-[11.5px] text-[var(--text-secondary)]">
                    Live SSE Trace Streaming
                  </span>
                  <input
                    type="checkbox"
                    checked={useStreaming}
                    onChange={(e) => onUseStreamingChange(e.target.checked)}
                    className="h-3.5 w-3.5 accent-blue-600"
                  />
                </label>

                <label className="flex items-center justify-between gap-2 cursor-pointer">
                  <span className="text-[11.5px] text-[var(--text-secondary)]">
                    Per-Request USD Cap
                  </span>
                  <input
                    type="checkbox"
                    checked={enableBudgetCap}
                    onChange={(e) => onEnableBudgetCapChange(e.target.checked)}
                    className="h-3.5 w-3.5 accent-blue-600"
                  />
                </label>

                {enableBudgetCap && (
                  <div>
                    <div className="flex items-center justify-between text-[10.5px]">
                      <span className="text-[var(--text-muted)]">Max USD</span>
                      <span className="font-mono font-semibold text-blue-400">
                        ${maxCostUsd.toFixed(2)}
                      </span>
                    </div>
                    <input
                      type="range"
                      min={0.01}
                      max={0.5}
                      step={0.01}
                      value={maxCostUsd}
                      onChange={(e) => onMaxCostUsdChange(Number(e.target.value))}
                      className="w-full accent-blue-600"
                    />
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Bottom Live Backend Health Footer */}
      <div className="shrink-0 border-t border-[var(--border-subtle)] bg-[var(--bg-elevated)]/40 p-2.5">
        <div className="flex items-center justify-between gap-2">
          <button
            type="button"
            onClick={() => setHealthExpanded((p) => !p)}
            className="flex min-w-0 flex-1 items-center gap-2 rounded-lg px-2 py-1.5 text-left transition hover:bg-[var(--bg-elevated)] cursor-pointer"
          >
            <span className={`h-2 w-2 shrink-0 rounded-full ${statusTone}`} />
            <div className="min-w-0 flex-1">
              <div className="truncate text-[11.5px] font-semibold text-[var(--text-primary)]">
                {health
                  ? health.status === "ok"
                    ? `API Online · ${qdrantPoints} chunks`
                    : `API Degraded · ${qdrantPoints} chunks`
                  : "API Offline"}
              </div>
              <div className="flex items-center gap-1.5 truncate font-mono text-[10px] text-[var(--text-muted)]">
                <Flame
                  className={`h-2.5 w-2.5 shrink-0 ${
                    fcStatus === "ok" ? "text-emerald-400" : "text-slate-500"
                  }`}
                />
                <span>FC: {fcStatus || "off"}</span>
                <span className="opacity-40">·</span>
                <Activity
                  className={`h-2.5 w-2.5 shrink-0 ${
                    lfMode === "cloud" || lfMode === "self_hosted"
                      ? "text-indigo-400"
                      : lfMode === "misconfigured"
                        ? "text-amber-400"
                        : "text-slate-500"
                  }`}
                />
                <span>LF: {lfLabel}</span>
              </div>
            </div>
          </button>
          <button
            type="button"
            onClick={onRefreshHealth}
            disabled={healthLoading}
            title="Refresh backend health"
            className="rounded-lg p-1.5 text-[var(--text-muted)] transition hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)] cursor-pointer"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${healthLoading ? "animate-spin" : ""}`} />
          </button>
        </div>

        {healthExpanded && (
          <div className="mt-2 space-y-2 rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-panel)] p-2.5 text-[10.5px]">
            <div>
              <label className="flex items-center gap-1 font-medium text-[var(--text-muted)]">
                <Server className="h-3 w-3" /> FastAPI URL
              </label>
              <div className="mt-1 flex gap-1">
                <input
                  type="url"
                  value={apiInput}
                  onChange={(e) => setApiInput(e.target.value)}
                  className="min-w-0 flex-1 rounded-md border border-[var(--border-subtle)] bg-[var(--bg-elevated)] px-2 py-1 font-mono text-[10.5px] text-[var(--text-primary)] focus:border-blue-500 focus:outline-none"
                />
                <button
                  type="button"
                  onClick={() => onApiUrlChange(apiInput.trim().replace(/\/+$/, ""))}
                  className="rounded-md bg-blue-600 px-2 py-1 text-[10.5px] font-medium text-white hover:bg-blue-500 cursor-pointer"
                >
                  Set
                </button>
              </div>
            </div>

            {health ? (
              <dl className="space-y-1 border-t border-[var(--border-subtle)] pt-2 font-mono text-[10px]">
                <div className="flex justify-between gap-2">
                  <dt className="text-[var(--text-muted)]">Fast LLM</dt>
                  <dd className="truncate text-[var(--text-primary)]">{health.fast_model}</dd>
                </div>
                <div className="flex justify-between gap-2">
                  <dt className="text-[var(--text-muted)]">Gen LLM</dt>
                  <dd className="truncate text-[var(--text-primary)]">
                    {health.generation_model}
                  </dd>
                </div>
                <div className="flex justify-between gap-2">
                  <dt className="text-[var(--text-muted)]">Reranker</dt>
                  <dd className="truncate text-[var(--text-primary)]">
                    {health.reranker_model.split("/").pop()}
                  </dd>
                </div>
                <div className="flex justify-between gap-2">
                  <dt className="text-[var(--text-muted)]">Web Fallback</dt>
                  <dd className="truncate text-[var(--text-primary)]">
                    {health.web_fallback_providers.join(", ") || "none"}
                  </dd>
                </div>
                <div className="flex justify-between gap-2">
                  <dt className="text-[var(--text-muted)]">Firecrawl</dt>
                  <dd className="truncate text-[var(--text-primary)]">
                    {health.firecrawl?.status || "disabled"}
                  </dd>
                </div>
                <div className="flex justify-between gap-2">
                  <dt className="text-[var(--text-muted)]">Langfuse</dt>
                  <dd className="truncate text-[var(--text-primary)]">
                    {lfLabel}
                    {health.langfuse?.environment && lfMode !== "disabled"
                      ? ` (${health.langfuse.environment})`
                      : ""}
                  </dd>
                </div>
                <div className="flex justify-between gap-2">
                  <dt className="text-[var(--text-muted)]">Prompts</dt>
                  <dd className="truncate text-[var(--text-primary)]">
                    {health.langfuse?.prompt_management?.active
                      ? `Langfuse (${health.langfuse.prompt_management.label ?? "production"})`
                      : `Local (${health.langfuse?.prompt_management?.managed_prompts ?? 8})`}
                  </dd>
                </div>
              </dl>
            ) : (
              <p className="text-[10.5px] text-rose-400">
                {healthError || "Start backend with `make api`."}
              </p>
            )}
          </div>
        )}
      </div>
    </aside>
  );
};
