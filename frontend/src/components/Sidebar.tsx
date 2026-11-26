import React, { useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  Columns2,
  Database,
  KeyRound,
  MessageSquare,
  PanelLeftClose,
  Plus,
  RefreshCw,
  Server,
  SlidersHorizontal,
  Trash2,
} from "lucide-react";
import { PRESET_OPTIONS, resolveEffectiveInstructions } from "../constants/pipeline";
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
  apiKey?: string;
  onApiKeyChange?: (key: string) => void;
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

function CompactSwitch({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (next: boolean) => void;
}) {
  return (
    <label className="group flex items-center justify-between gap-2 py-0.5 cursor-pointer select-none">
      <span className="text-[11.5px] text-[var(--text-secondary)] transition group-hover:text-[var(--text-primary)]">
        {label}
      </span>
      <span className="relative inline-flex h-4 w-7 shrink-0 items-center">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="peer sr-only"
        />
        <span className="h-4 w-7 rounded-full border border-[var(--border-strong)] bg-[var(--bg-elevated)] transition-colors peer-checked:border-transparent peer-checked:bg-[var(--cta-bg)] peer-focus-visible:ring-2 peer-focus-visible:ring-zinc-400/50" />
        <span className="pointer-events-none absolute left-0.5 h-3 w-3 rounded-full bg-[var(--text-muted)] transition-transform peer-checked:translate-x-3 peer-checked:bg-[var(--cta-fg)]" />
      </span>
    </label>
  );
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
  apiKey = "",
  onApiKeyChange,
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
  const [apiKeyInput, setApiKeyInput] = useState(apiKey);

  const authRequiredMissingKey =
    Boolean(health?.security?.api_key_auth_enabled) && !apiKey.trim();

  const qdrantPoints = health?.qdrant?.points_count ?? 0;
  const fcStatus = health?.firecrawl?.status;
  const lfMode =
    health?.langfuse?.mode ?? (health?.langfuse_enabled ? "cloud" : "disabled");
  const lfShort =
    lfMode === "cloud"
      ? "cloud"
      : lfMode === "self_hosted"
        ? "local"
        : lfMode === "misconfigured"
          ? "no-key"
          : "off";
  const fcShort =
    fcStatus === "ok" ? "ready" : fcStatus === "unreachable" ? "down" : "off";

  const statusTone = !health
    ? "bg-rose-500"
    : authRequiredMissingKey
      ? "bg-amber-500"
      : health.status === "ok"
        ? "bg-emerald-500"
        : "bg-amber-500";

  const updateCustom = <K extends keyof AblationConfig>(
    key: K,
    val: AblationConfig[K]
  ) => {
    onCustomAblationChange({ ...customAblation, [key]: val });
  };

  const navItems: {
    id: AppView;
    label: string;
    icon: React.FC<{ className?: string }>;
    badge?: string;
  }[] = [
    { id: "workbench", label: "Workbench", icon: MessageSquare },
    { id: "compare", label: "Ablation Arena", icon: Columns2 },
    {
      id: "knowledge",
      label: "Knowledge Base",
      icon: Database,
      badge: String(qdrantPoints),
    },
  ];

  return (
    <aside className="flex h-full w-[264px] shrink-0 flex-col border-r border-[var(--border-subtle)] bg-[var(--bg-panel)]">
      {/* Brand Bar */}
      <div className="flex h-12 shrink-0 items-center justify-between border-b border-[var(--border-subtle)] px-3">
        <div className="flex items-center gap-2">
          <div className="flex h-6 w-6 items-center justify-center rounded-[5px] bg-[var(--cta-bg)] text-[var(--cta-fg)]">
            <svg
              viewBox="0 0 24 24"
              className="h-3.5 w-3.5"
              fill="none"
              stroke="currentColor"
              strokeWidth={2.25}
            >
              <path d="M12 3L4 7.5v9L12 21l8-4.5v-9L12 3z" strokeLinejoin="round" />
              <path
                d="M12 12l8-4.5M12 12v9M12 12L4 7.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
          <span className="text-[13px] font-semibold tracking-tight text-[var(--text-primary)]">
            AdaptiveRAG
          </span>
          <span className="rounded border border-[var(--border-subtle)] bg-[var(--bg-surface)] px-1.5 py-0.5 font-mono text-[9.5px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
            Studio
          </span>
        </div>

        <div className="flex items-center gap-0.5">
          <button
            type="button"
            onClick={() => {
              onNewSession();
              onChangeView("workbench");
            }}
            title="New research thread"
            className="rounded-md p-1.5 text-[var(--text-muted)] transition hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)] cursor-pointer"
          >
            <Plus className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            onClick={onCollapse}
            title="Collapse sidebar"
            className="rounded-md p-1.5 text-[var(--text-muted)] transition hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)] cursor-pointer"
          >
            <PanelLeftClose className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Primary Workspace Switcher */}
      <div className="space-y-1 border-b border-[var(--border-subtle)] p-2.5">
        <button
          type="button"
          onClick={() => {
            onNewSession();
            onChangeView("workbench");
          }}
          className="mb-1.5 flex h-8 w-full items-center justify-between rounded-md border border-[var(--border-strong)] bg-[var(--bg-surface)] px-2.5 text-xs font-medium text-[var(--text-primary)] shadow-2xs transition hover:bg-[var(--bg-hover)] cursor-pointer"
        >
          <span className="inline-flex items-center gap-1.5">
            <Plus className="h-3.5 w-3.5 text-[var(--text-muted)]" />
            <span>New Thread</span>
          </span>
          <span className="font-mono text-[10px] text-[var(--text-muted)]">EN/TR</span>
        </button>

        {navItems.map(({ id, label, icon: Icon, badge }) => {
          const active = activeView === id;
          return (
            <button
              key={id}
              type="button"
              onClick={() => onChangeView(id)}
              data-nav={id}
              className={`flex h-7.5 w-full items-center justify-between rounded-md px-2 text-xs transition cursor-pointer ${
                active
                  ? "bg-[var(--bg-elevated)] font-medium text-[var(--text-primary)]"
                  : "text-[var(--text-secondary)] hover:bg-[var(--bg-elevated)]/60 hover:text-[var(--text-primary)]"
              }`}
            >
              <span className="inline-flex items-center gap-2">
                <Icon
                  className={`h-3.5 w-3.5 shrink-0 ${
                    active ? "text-[var(--text-primary)]" : "text-[var(--text-muted)]"
                  }`}
                />
                <span>{label}</span>
              </span>
              {badge !== undefined && (
                <span className="rounded border border-[var(--border-subtle)] bg-[var(--bg-surface)] px-1.5 py-0.2 font-mono text-[10px] text-[var(--text-muted)] tabular-nums">
                  {badge}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Scrollable Middle: Threads + Pipeline Configuration */}
      <div className="flex-1 space-y-4 overflow-y-auto px-2.5 py-3">
        {/* Threads */}
        <div>
          <div className="mb-1 flex items-center justify-between px-1.5">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
              Threads
            </span>
            <span className="font-mono text-[10px] text-[var(--text-muted)] tabular-nums">
              {sessions.length}
            </span>
          </div>
          <div className="space-y-0.5">
            {sessions.map((sess) => {
              const isActive =
                sess.id === activeSessionId && activeView === "workbench";
              const hasCustomStyle = Boolean(resolveEffectiveInstructions(sess.context));
              const scopedCount = sess.context?.scopedDocIds?.length ?? 0;
              const hasCustomContext = hasCustomStyle || scopedCount > 0;
              return (
                <div
                  key={sess.id}
                  title={
                    hasCustomContext
                      ? `${sess.title} (${[
                          hasCustomStyle ? "custom style" : null,
                          scopedCount > 0 ? `${scopedCount} scoped doc(s)` : null,
                        ]
                          .filter(Boolean)
                          .join(" · ")})`
                      : sess.title
                  }
                  className={`group flex h-7.5 items-center justify-between gap-1 rounded-md px-2 text-xs transition ${
                    isActive
                      ? "bg-[var(--bg-elevated)] font-medium text-[var(--text-primary)]"
                      : "text-[var(--text-secondary)] hover:bg-[var(--bg-elevated)]/55 hover:text-[var(--text-primary)]"
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
                    <span
                      className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                        hasCustomContext
                          ? "bg-emerald-500"
                          : isActive
                            ? "bg-[var(--text-primary)]"
                            : "bg-[var(--border-strong)]"
                      }`}
                    />
                    <span className="truncate">{sess.title}</span>
                  </button>
                  <span className="shrink-0 font-mono text-[10px] text-[var(--text-muted)] tabular-nums group-hover:hidden">
                    {sess.turns.length}
                  </span>
                  {sessions.length > 1 ? (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteSession(sess.id);
                      }}
                      title="Delete thread"
                      className="hidden shrink-0 rounded p-0.5 text-[var(--text-muted)] hover:text-rose-500 group-hover:inline-flex cursor-pointer"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  ) : (
                    <span className="hidden shrink-0 font-mono text-[10px] text-[var(--text-muted)] tabular-nums group-hover:inline">
                      {sess.turns.length}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Pipeline Configuration */}
        <div className="border-t border-[var(--border-subtle)] pt-3">
          <button
            type="button"
            onClick={() => setControlsOpen((p) => !p)}
            className="flex w-full items-center justify-between px-1.5 py-1 text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)] transition hover:text-[var(--text-primary)] cursor-pointer"
          >
            <span className="inline-flex items-center gap-1.5">
              <SlidersHorizontal className="h-3 w-3" />
              <span>Pipeline Config</span>
            </span>
            {controlsOpen ? (
              <ChevronDown className="h-3 w-3" />
            ) : (
              <ChevronRight className="h-3 w-3" />
            )}
          </button>

          {controlsOpen && (
            <div className="mt-2 space-y-3 px-1.5 text-xs">
              {activeView === "compare" && (
                <div className="rounded-md border border-[var(--border-subtle)] bg-[var(--bg-surface)] px-2.5 py-2 text-[11px] leading-snug text-[var(--text-secondary)]">
                  A/B presets are configured in the Arena toolbar. Streaming and
                  cost cap below apply to both sides.
                </div>
              )}

              <div>
                <label className="block text-[10.5px] font-medium text-[var(--text-muted)]">
                  Execution Preset
                </label>
                <select
                  value={preset}
                  onChange={(e) => onPresetChange(e.target.value as PresetSelection)}
                  className="ar-select mt-1 h-8 w-full rounded-md border border-[var(--border-strong)] bg-[var(--bg-surface)] px-2.5 text-xs font-medium text-[var(--text-primary)] focus:border-[var(--text-muted)] focus:outline-none"
                >
                  {PRESET_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
                <p className="mt-1.5 text-[10.5px] leading-relaxed text-[var(--text-muted)]">
                  {PRESET_OPTIONS.find((o) => o.value === preset)?.description}
                </p>
              </div>

              {preset === "custom" && (
                <div className="space-y-2.5 rounded-md border border-[var(--border-subtle)] bg-[var(--bg-surface)] p-2.5 text-[11px]">
                  <div>
                    <label className="text-[10px] font-medium text-[var(--text-muted)]">
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
                      className="ar-select mt-1 h-7.5 w-full rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)] px-2 text-[11px] text-[var(--text-primary)] focus:outline-none"
                    >
                      <option value="hybrid">Hybrid (Dense + BM25 RRF)</option>
                      <option value="dense_only">Dense Only</option>
                      <option value="sparse_only">BM25 Sparse Only</option>
                    </select>
                  </div>

                  <div className="space-y-1.5 border-t border-[var(--border-subtle)] pt-2">
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
                      <CompactSwitch
                        key={key}
                        label={lbl}
                        checked={customAblation[key]}
                        onChange={(val) => updateCustom(key, val)}
                      />
                    ))}
                  </div>

                  {!customAblation.enable_dynamic_k && (
                    <div className="border-t border-[var(--border-subtle)] pt-2">
                      <div className="mb-1 flex justify-between text-[10.5px]">
                        <span className="text-[var(--text-muted)]">Static Top-K</span>
                        <span className="font-mono font-semibold text-[var(--text-primary)]">
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
                        className="ar-range w-full"
                      />
                    </div>
                  )}

                  <div className="space-y-2 border-t border-[var(--border-subtle)] pt-2">
                    <div>
                      <label className="text-[10px] font-medium text-[var(--text-muted)]">
                        Small-to-Big Parent-Child
                      </label>
                      <select
                        value={
                          customAblation.enable_parent_child === true
                            ? "enabled"
                            : customAblation.enable_parent_child === false
                              ? "disabled"
                              : "default"
                        }
                        onChange={(e) =>
                          updateCustom(
                            "enable_parent_child",
                            e.target.value === "enabled"
                              ? true
                              : e.target.value === "disabled"
                                ? false
                                : null
                          )
                        }
                        className="ar-select mt-1 h-7.5 w-full rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)] px-2 text-[11px] text-[var(--text-primary)] focus:outline-none"
                      >
                        <option value="default">Server Default</option>
                        <option value="enabled">Force Enable (Expand Parents)</option>
                        <option value="disabled">Force Disable (Child Only)</option>
                      </select>
                    </div>

                    <div>
                      <label className="text-[10px] font-medium text-[var(--text-muted)]">
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
                        className="ar-select mt-1 h-7.5 w-full rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)] px-2 text-[11px] text-[var(--text-primary)] focus:outline-none"
                      >
                        <option value="all">All (TR + EN)</option>
                        <option value="en">English Only</option>
                        <option value="tr">Turkish Only</option>
                      </select>
                    </div>
                  </div>
                </div>
              )}

              <div className="space-y-2 border-t border-[var(--border-subtle)] pt-2.5">
                <CompactSwitch
                  label="Live SSE Trace Stream"
                  checked={useStreaming}
                  onChange={onUseStreamingChange}
                />
                <CompactSwitch
                  label="Per-Request Cost Cap"
                  checked={enableBudgetCap}
                  onChange={onEnableBudgetCapChange}
                />

                {enableBudgetCap && (
                  <div className="pt-0.5">
                    <div className="mb-1 flex items-center justify-between text-[10.5px]">
                      <span className="text-[var(--text-muted)]">Max Spend</span>
                      <span className="font-mono font-medium text-[var(--text-primary)] tabular-nums">
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
                      className="ar-range w-full"
                    />
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Runtime Health Footer */}
      <div className="shrink-0 border-t border-[var(--border-subtle)] p-2">
        <div className="flex items-center justify-between gap-1">
          <button
            type="button"
            onClick={() => setHealthExpanded((p) => !p)}
            className="flex min-w-0 flex-1 items-center gap-2 rounded-md px-2 py-1.5 text-left transition hover:bg-[var(--bg-elevated)] cursor-pointer"
          >
            <span className={`h-2 w-2 shrink-0 rounded-full ${statusTone}`} />
            <div className="min-w-0 flex-1">
              <div className="flex items-center justify-between gap-2">
                <span className="truncate text-[11.5px] font-medium text-[var(--text-primary)]">
                  {health
                    ? authRequiredMissingKey
                      ? "API Key Required"
                      : health.status === "ok"
                        ? "Runtime Online"
                        : "Runtime Degraded"
                    : "Runtime Offline"}
                </span>
                <span className="shrink-0 font-mono text-[10px] text-[var(--text-muted)] tabular-nums">
                  {qdrantPoints} pts
                </span>
              </div>
              <div className="mt-0.5 flex items-center gap-2 font-mono text-[10px] text-[var(--text-muted)]">
                <span>fc:{fcShort}</span>
                <span className="opacity-40">/</span>
                <span>lf:{lfShort}</span>
              </div>
            </div>
          </button>
          <button
            type="button"
            onClick={onRefreshHealth}
            disabled={healthLoading}
            title="Refresh backend status"
            className="rounded-md p-1.5 text-[var(--text-muted)] transition hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)] cursor-pointer"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${healthLoading ? "animate-spin" : ""}`} />
          </button>
        </div>

        {(healthExpanded || authRequiredMissingKey) && (
          <div className="mt-1.5 space-y-2.5 rounded-md border border-[var(--border-subtle)] bg-[var(--bg-surface)] p-2.5 text-[10.5px]">
            <div>
              <label className="flex items-center gap-1 font-medium text-[var(--text-muted)]">
                <Server className="h-3 w-3" /> Endpoint URL
              </label>
              <div className="mt-1 flex gap-1">
                <input
                  type="text"
                  value={apiInput}
                  placeholder="Same-origin or http://localhost:8000"
                  onChange={(e) => setApiInput(e.target.value)}
                  className="min-w-0 flex-1 rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)] px-2 py-1 font-mono text-[10.5px] text-[var(--text-primary)] focus:border-[var(--text-muted)] focus:outline-none"
                />
                <button
                  type="button"
                  onClick={() => onApiUrlChange(apiInput.trim().replace(/\/+$/, ""))}
                  className="rounded bg-[var(--cta-bg)] px-2 py-1 text-[10.5px] font-medium text-[var(--cta-fg)] hover:bg-[var(--cta-hover)] cursor-pointer"
                >
                  Set
                </button>
              </div>
            </div>

            {onApiKeyChange && (
              <div>
                <label className="flex items-center justify-between font-medium text-[var(--text-muted)]">
                  <span className="inline-flex items-center gap-1">
                    <KeyRound className="h-3 w-3" /> API Key (X-API-Key)
                  </span>
                  {apiKey.trim() && (
                    <span className="font-mono text-[9.5px] text-emerald-500">
                      active
                    </span>
                  )}
                </label>
                <div className="mt-1 flex gap-1">
                  <input
                    type="password"
                    value={apiKeyInput}
                    placeholder={
                      health?.security?.api_key_auth_enabled
                        ? "Required by server"
                        : "Optional API_AUTH_KEY"
                    }
                    onChange={(e) => setApiKeyInput(e.target.value)}
                    className="min-w-0 flex-1 rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)] px-2 py-1 font-mono text-[10.5px] text-[var(--text-primary)] focus:border-[var(--text-muted)] focus:outline-none"
                  />
                  <button
                    type="button"
                    onClick={() => onApiKeyChange(apiKeyInput)}
                    className="rounded bg-[var(--cta-bg)] px-2 py-1 text-[10.5px] font-medium text-[var(--cta-fg)] hover:bg-[var(--cta-hover)] cursor-pointer"
                  >
                    Save
                  </button>
                </div>
              </div>
            )}

            {health ? (
              <dl className="space-y-1 border-t border-[var(--border-subtle)] pt-2 font-mono text-[10px]">
                <div className="flex justify-between gap-2">
                  <dt className="text-[var(--text-muted)]">fast_llm</dt>
                  <dd className="truncate text-[var(--text-primary)]">{health.fast_model}</dd>
                </div>
                <div className="flex justify-between gap-2">
                  <dt className="text-[var(--text-muted)]">gen_llm</dt>
                  <dd className="truncate text-[var(--text-primary)]">
                    {health.generation_model}
                  </dd>
                </div>
                <div className="flex justify-between gap-2">
                  <dt className="text-[var(--text-muted)]">reranker</dt>
                  <dd className="truncate text-[var(--text-primary)]">
                    {health.reranker_model.split("/").pop()}
                  </dd>
                </div>
                <div className="flex justify-between gap-2">
                  <dt className="text-[var(--text-muted)]">fallback</dt>
                  <dd className="truncate text-[var(--text-primary)]">
                    {health.web_fallback_providers.join(", ") || "none"}
                  </dd>
                </div>
                {health.parent_child_retrieval && (
                  <div className="flex justify-between gap-2">
                    <dt className="text-[var(--text-muted)]">chunking</dt>
                    <dd className="truncate text-[var(--text-primary)]">
                      {health.parent_child_retrieval.enabled
                        ? `${health.parent_child_retrieval.child_max_tokens}→${health.parent_child_retrieval.parent_max_tokens}`
                        : "flat-512"}
                    </dd>
                  </div>
                )}
                {health.security && (
                  <div className="flex justify-between gap-2">
                    <dt className="text-[var(--text-muted)]">guardrails</dt>
                    <dd className="truncate text-[var(--text-primary)]">
                      {health.security.guardrails_enabled
                        ? `on (${health.security.pii_mode ?? "redact"})`
                        : "off"}
                      {health.security.api_key_auth_enabled ? " · auth" : ""}
                      {health.security.rate_limit_enabled ? " · rl" : ""}
                    </dd>
                  </div>
                )}
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
