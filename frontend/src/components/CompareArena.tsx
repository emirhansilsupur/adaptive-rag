import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  ArrowLeftRight,
  ArrowUp,
  Check,
  CheckCircle2,
  Columns2,
  ExternalLink,
  FileText,
  Globe,
  Microscope,
  Plus,
  ShieldAlert,
  Sparkles,
  Square,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  TrendingDown,
  TrendingUp,
} from "lucide-react";
import {
  ARENA_PRESET_OPTIONS,
  COMPARE_DEMO_QUERIES,
  COMPARE_PAIR_PRESETS,
} from "../constants/pipeline";
import {
  computeCompareSummary,
  formatSignedPct,
  getContextTokens,
  getPresetShortLabel,
} from "../lib/compare";
import type {
  AblationPreset,
  CompareRun,
  CompareSideKey,
  CompareSideState,
  InspectorTab,
} from "../types/api";
import { AnswerTelemetryBar } from "./AnswerTelemetryBar";
import { LivePipelineTracker } from "./LivePipelineTracker";
import { MarkdownWithCitations } from "./MarkdownWithCitations";

interface CompareArenaProps {
  runs: CompareRun[];
  activeRunId: string | null;
  leftPreset: AblationPreset;
  rightPreset: AblationPreset;
  inspectedSide: CompareSideKey;
  isRunning: boolean;
  useStreaming: boolean;
  budgetCapLabel: string | null;
  onLeftPresetChange: (preset: AblationPreset) => void;
  onRightPresetChange: (preset: AblationPreset) => void;
  onSwapPresets: () => void;
  onSelectRun: (runId: string) => void;
  onDeleteRun: (runId: string) => void;
  onNewComparison: () => void;
  onSubmitCompare: (query: string) => void;
  onStopCompare: () => void;
  onInspectSide: (side: CompareSideKey, tab?: InspectorTab) => void;
  onCitationClick: (side: CompareSideKey, citationId: string) => void;
  onSubmitSideFeedback?: (
    runId: string,
    side: CompareSideKey,
    score: 0 | 1,
    comment?: string
  ) => Promise<void>;
}

function DeltaTile({
  label,
  value,
  deltaBadge,
  tone = "neutral",
}: {
  label: string;
  value: string;
  deltaBadge?: string;
  tone?: "emerald" | "amber" | "rose" | "blue" | "neutral";
}) {
  const badgeTone =
    tone === "emerald"
      ? "border-emerald-500/35 bg-emerald-500/12 text-emerald-600 dark:text-emerald-300"
      : tone === "amber"
        ? "border-amber-500/35 bg-amber-500/12 text-amber-600 dark:text-amber-300"
        : tone === "rose"
          ? "border-rose-500/35 bg-rose-500/12 text-rose-600 dark:text-rose-300"
          : tone === "blue"
            ? "border-blue-500/35 bg-blue-500/12 text-blue-600 dark:text-blue-300"
            : "border-[var(--border-subtle)] bg-[var(--bg-elevated)] text-[var(--text-secondary)]";

  return (
    <div className="rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-panel)] px-3 py-2.5">
      <div className="flex items-center justify-between gap-1.5">
        <span className="truncate text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
          {label}
        </span>
        {deltaBadge && (
          <span
            className={`shrink-0 whitespace-nowrap rounded-full border px-1.5 py-0.5 font-mono text-[10px] font-semibold tabular-nums ${badgeTone}`}
          >
            {deltaBadge}
          </span>
        )}
      </div>
      <div className="mt-1 font-mono text-xs font-semibold text-[var(--text-primary)] tabular-nums">
        {value}
      </div>
    </div>
  );
}

interface ArenaColumnCardProps {
  sideKey: CompareSideKey;
  runId: string;
  side: CompareSideState;
  isInspected: boolean;
  useStreaming: boolean;
  onInspect: (tab?: InspectorTab) => void;
  onCitationClick: (citationId: string) => void;
  onSubmitFeedback?: (score: 0 | 1, comment?: string) => Promise<void>;
}

const ArenaColumnCard: React.FC<ArenaColumnCardProps> = ({
  sideKey,
  runId,
  side,
  isInspected,
  useStreaming,
  onInspect,
  onCitationClick,
  onSubmitFeedback,
}) => {
  const [commentOpen, setCommentOpen] = useState(false);
  const [commentDraft, setCommentDraft] = useState("");
  const [savingFeedback, setSavingFeedback] = useState(false);
  const [feedbackError, setFeedbackError] = useState<string | null>(null);

  useEffect(() => {
    setCommentOpen(false);
    setCommentDraft("");
    setFeedbackError(null);
  }, [runId]);

  const isBaseline = sideKey === "left";
  const result = side.result;
  const chunks = result?.evidence_chunks ?? [];
  const ctxTokens = result ? getContextTokens(result) : null;

  const submitSideRating = async (score: 0 | 1, comment?: string) => {
    if (!onSubmitFeedback || savingFeedback) return;
    setSavingFeedback(true);
    setFeedbackError(null);
    try {
      await onSubmitFeedback(score, comment);
      setCommentOpen(false);
      setCommentDraft("");
    } catch (err) {
      setFeedbackError(err instanceof Error ? err.message : String(err));
    } finally {
      setSavingFeedback(false);
    }
  };

  return (
    <section
      data-compare-side={sideKey}
      onClick={() => result && onInspect()}
      className={`flex min-w-0 flex-col rounded-2xl border p-4 transition-colors ${
        isInspected && result
          ? isBaseline
            ? "border-amber-500/45 bg-[var(--bg-panel)] shadow-xs"
            : "border-blue-500/50 bg-[var(--bg-panel)] shadow-xs"
          : "border-[var(--border-subtle)] bg-[var(--bg-panel)]/90 hover:border-[var(--border-strong)]"
      }`}
    >
      {/* Column Header */}
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-[var(--border-subtle)] pb-2.5">
        <div className="flex min-w-0 items-center gap-2">
          <span
            className={`inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-md font-mono text-[10px] font-bold ${
              isBaseline
                ? "bg-amber-500/15 text-amber-600 dark:text-amber-300"
                : "bg-blue-500/15 text-blue-600 dark:text-blue-300"
            }`}
          >
            {isBaseline ? "A" : "B"}
          </span>
          <div className="min-w-0">
            <div className="flex items-center gap-1.5">
              <span className="truncate text-xs font-semibold text-[var(--text-primary)]">
                {side.presetLabel}
              </span>
              <span className="rounded bg-[var(--bg-elevated)] px-1.5 py-0.5 font-mono text-[9.5px] text-[var(--text-muted)]">
                {isBaseline ? "Baseline" : "Challenger"}
              </span>
            </div>
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-1.5">
          {side.status === "running" && (
            <span className="inline-flex items-center gap-1.5 rounded-full border border-blue-500/30 bg-blue-500/10 px-2 py-0.5 text-[10.5px] font-medium text-blue-600 dark:text-blue-300">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-blue-500" />
              Running
            </span>
          )}
          {result && (
            <>
              <span className="rounded-md border border-[var(--border-subtle)] bg-[var(--bg-elevated)] px-1.5 py-0.5 font-mono text-[10px] text-[var(--text-secondary)] tabular-nums">
                k={chunks.length} · {ctxTokens?.toLocaleString("en-US")} ctx tok
              </span>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onInspect("trace");
                }}
                className={`inline-flex items-center gap-1 rounded-lg px-2 py-1 text-[10.5px] font-medium transition cursor-pointer ${
                  isInspected
                    ? "bg-blue-500/15 text-blue-600 dark:text-blue-300"
                    : "text-[var(--text-muted)] hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)]"
                }`}
              >
                <Microscope className="h-3 w-3" />
                <span>{isInspected ? "Inspecting" : "Inspect"}</span>
              </button>
            </>
          )}
        </div>
      </div>

      {/* Running State */}
      {side.status === "running" && (
        <div className="space-y-3">
          <LivePipelineTracker
            mode="streaming"
            steps={side.liveSteps}
            liveUsage={side.liveUsage}
          />
          <div className="flex items-center gap-2 text-xs text-[var(--text-muted)]">
            <span className="h-2 w-2 animate-pulse rounded-full bg-blue-500" />
            <span>
              {useStreaming
                ? `Streaming ${side.presetLabel} over SSE...`
                : `Running ${side.presetLabel}...`}
            </span>
          </div>
        </div>
      )}

      {/* Error State */}
      {side.status === "error" && (
        <div className="flex items-start gap-2 rounded-xl border border-rose-500/35 bg-rose-500/10 p-3 text-xs text-rose-600 dark:text-rose-300">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
          <div className="min-w-0">
            <div className="font-semibold">{side.presetLabel} Failed</div>
            <p className="mt-0.5 break-words opacity-90">{side.error}</p>
          </div>
        </div>
      )}

      {/* Completed Result */}
      {result && (
        <div className="flex flex-1 flex-col">
          <LivePipelineTracker
            mode="completed"
            steps={result.trace || []}
            totalLatencyMs={result.total_latency_ms}
            onOpenInspector={() => onInspect("trace")}
          />

          <div className="mb-3">
            <AnswerTelemetryBar
              result={result}
              onBadgeClick={(tab) => onInspect(tab)}
            />
          </div>

          <div className="flex-1">
            <MarkdownWithCitations
              content={result.answer}
              evidenceChunks={chunks}
              onCitationClick={(cid) => onCitationClick(cid)}
            />
          </div>

          {/* Sources + Feedback Footer */}
          <div className="mt-4 flex flex-wrap items-center justify-between gap-2 border-t border-[var(--border-subtle)] pt-3">
            <div className="flex flex-wrap items-center gap-1.5">
              {chunks.slice(0, 4).map((ch, cIdx) => {
                const cid = ch.citation_id || `#${cIdx + 1}`;
                const isWeb = ch.source_type === "web";
                return (
                  <button
                    key={ch.chunk_id || cIdx}
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onCitationClick(cid);
                    }}
                    title={`${ch.title || ch.doc_id}${ch.url ? ` (${ch.url})` : ""}`}
                    className="inline-flex max-w-[165px] items-center gap-1 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/70 px-2 py-0.5 text-[10.5px] text-[var(--text-secondary)] transition hover:border-blue-500/40 hover:text-[var(--text-primary)] cursor-pointer"
                  >
                    <span
                      className={`shrink-0 font-mono text-[10px] font-bold ${
                        isWeb ? "text-purple-400" : "text-teal-400"
                      }`}
                    >
                      [{cid}]
                    </span>
                    {isWeb ? (
                      <Globe className="h-2.5 w-2.5 shrink-0 text-purple-400" />
                    ) : (
                      <FileText className="h-2.5 w-2.5 shrink-0 text-teal-400" />
                    )}
                    <span className="truncate">{ch.title || ch.doc_id}</span>
                    {ch.url && <ExternalLink className="h-2.5 w-2.5 shrink-0 opacity-60" />}
                  </button>
                );
              })}
              {chunks.length > 4 && (
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onInspect("evidence");
                  }}
                  className="text-[10.5px] font-medium text-blue-500 hover:underline cursor-pointer"
                >
                  +{chunks.length - 4} more
                </button>
              )}
              {chunks.length === 0 && (
                <span className="font-mono text-[10.5px] text-[var(--text-muted)]">
                  No passages retrieved
                </span>
              )}
            </div>

            {onSubmitFeedback && result.trace_id && (
              <div
                className="inline-flex items-center gap-1"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="inline-flex items-center gap-0.5 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60 p-0.5">
                  <button
                    type="button"
                    aria-label={`Mark ${side.presetLabel} answer helpful`}
                    disabled={savingFeedback}
                    onClick={() => void submitSideRating(1)}
                    className={`rounded-md p-1 transition cursor-pointer disabled:opacity-50 ${
                      side.feedback?.score === 1
                        ? "bg-emerald-500/20 text-emerald-500 dark:text-emerald-300"
                        : "text-[var(--text-muted)] hover:bg-[var(--bg-panel)] hover:text-emerald-500"
                    }`}
                  >
                    <ThumbsUp className="h-3 w-3" />
                  </button>
                  <button
                    type="button"
                    aria-label={`Mark ${side.presetLabel} answer unhelpful`}
                    disabled={savingFeedback}
                    onClick={() => {
                      setCommentOpen((prev) => !prev);
                      setCommentDraft(side.feedback?.comment ?? "");
                    }}
                    className={`rounded-md p-1 transition cursor-pointer disabled:opacity-50 ${
                      side.feedback?.score === 0 || commentOpen
                        ? "bg-rose-500/20 text-rose-500 dark:text-rose-300"
                        : "text-[var(--text-muted)] hover:bg-[var(--bg-panel)] hover:text-rose-500"
                    }`}
                  >
                    <ThumbsDown className="h-3 w-3" />
                  </button>
                </div>
                {side.feedback && (
                  <span className="inline-flex items-center gap-0.5 rounded bg-[var(--bg-elevated)] px-1.5 py-0.5 text-[10px] text-[var(--text-muted)]">
                    <Check className="h-2.5 w-2.5 text-emerald-500" />
                    {side.feedback.persistedToLangfuse ? "LF" : "Saved"}
                  </span>
                )}
              </div>
            )}
          </div>

          {commentOpen && (
            <div
              onClick={(e) => e.stopPropagation()}
              className="mt-2 flex flex-wrap items-center gap-1.5 rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/65 p-2"
            >
              <input
                type="text"
                value={commentDraft}
                onChange={(e) => setCommentDraft(e.target.value)}
                placeholder="Optional note for this preset..."
                className="min-w-[140px] flex-1 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-panel)] px-2 py-1 text-xs text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:border-blue-500 focus:outline-none"
              />
              <button
                type="button"
                disabled={savingFeedback}
                onClick={() =>
                  void submitSideRating(0, commentDraft.trim() || undefined)
                }
                className="rounded-lg bg-rose-600 px-2 py-1 text-[10.5px] font-medium text-white hover:bg-rose-500 disabled:opacity-50 cursor-pointer"
              >
                Save
              </button>
            </div>
          )}

          {feedbackError && (
            <p className="mt-1.5 text-[10.5px] text-rose-500">{feedbackError}</p>
          )}
        </div>
      )}
    </section>
  );
};

export const CompareArena: React.FC<CompareArenaProps> = ({
  runs,
  activeRunId,
  leftPreset,
  rightPreset,
  inspectedSide,
  isRunning,
  useStreaming,
  budgetCapLabel,
  onLeftPresetChange,
  onRightPresetChange,
  onSwapPresets,
  onSelectRun,
  onDeleteRun,
  onNewComparison,
  onSubmitCompare,
  onStopCompare,
  onInspectSide,
  onCitationClick,
  onSubmitSideFeedback,
}) => {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const activeRun = useMemo(
    () => (activeRunId ? runs.find((r) => r.id === activeRunId) ?? null : null),
    [runs, activeRunId]
  );

  const delta = useMemo(() => {
    if (!activeRun?.left.result || !activeRun?.right.result) return null;
    return computeCompareSummary(activeRun.left.result, activeRun.right.result);
  }, [activeRun]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
  }, [input]);

  const handleSend = () => {
    const cleaned = input.trim();
    if (!cleaned || isRunning) return;
    setInput("");
    onSubmitCompare(cleaned);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const leftShort = getPresetShortLabel(leftPreset);
  const rightShort = getPresetShortLabel(rightPreset);

  return (
    <div className="flex min-w-0 flex-1 flex-col bg-[var(--bg-canvas)]">
      {/* Top Arena Configuration Bar */}
      <div className="shrink-0 border-b border-[var(--border-subtle)] bg-[var(--bg-panel)]/75 px-4 py-2.5 sm:px-6">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-2.5">
          {/* A vs B Preset Selectors */}
          <div className="flex flex-wrap items-center gap-2">
            <div className="flex items-center gap-1.5">
              <span className="inline-flex h-5 w-5 items-center justify-center rounded-md bg-amber-500/15 font-mono text-[10px] font-bold text-amber-600 dark:text-amber-300">
                A
              </span>
              <select
                aria-label="Baseline preset (Left)"
                value={leftPreset}
                disabled={isRunning}
                onChange={(e) => onLeftPresetChange(e.target.value as AblationPreset)}
                className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-elevated)] px-2.5 py-1 text-xs font-medium text-[var(--text-primary)] focus:border-blue-500 focus:outline-none disabled:opacity-50"
              >
                {ARENA_PRESET_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>

            <button
              type="button"
              onClick={onSwapPresets}
              disabled={isRunning}
              title="Swap A and B presets"
              aria-label="Swap A and B presets"
              className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-elevated)] p-1.5 text-[var(--text-secondary)] transition hover:border-blue-500/40 hover:text-[var(--text-primary)] disabled:opacity-40 cursor-pointer"
            >
              <ArrowLeftRight className="h-3.5 w-3.5" />
            </button>

            <div className="flex items-center gap-1.5">
              <span className="inline-flex h-5 w-5 items-center justify-center rounded-md bg-blue-500/15 font-mono text-[10px] font-bold text-blue-600 dark:text-blue-300">
                B
              </span>
              <select
                aria-label="Challenger preset (Right)"
                value={rightPreset}
                disabled={isRunning}
                onChange={(e) => onRightPresetChange(e.target.value as AblationPreset)}
                className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-elevated)] px-2.5 py-1 text-xs font-medium text-[var(--text-primary)] focus:border-blue-500 focus:outline-none disabled:opacity-50"
              >
                {ARENA_PRESET_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Quick Pairs + History Controls */}
          <div className="flex flex-wrap items-center gap-1.5">
            {COMPARE_PAIR_PRESETS.map((pair) => {
              const active = leftPreset === pair.left && rightPreset === pair.right;
              return (
                <button
                  key={pair.id}
                  type="button"
                  disabled={isRunning}
                  onClick={() => {
                    onLeftPresetChange(pair.left);
                    onRightPresetChange(pair.right);
                  }}
                  className={`rounded-full border px-2.5 py-0.5 text-[10.5px] font-medium transition cursor-pointer disabled:opacity-40 ${
                    active
                      ? "border-blue-500/40 bg-blue-500/12 text-blue-600 dark:text-blue-300"
                      : "border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60 text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                  }`}
                >
                  {pair.label}
                </button>
              );
            })}

            {runs.length > 0 && (
              <div className="ml-1 flex items-center gap-1 border-l border-[var(--border-subtle)] pl-2">
                <select
                  aria-label="Past Arena Comparisons"
                  value={activeRun?.id ?? ""}
                  onChange={(e) => {
                    if (e.target.value) onSelectRun(e.target.value);
                  }}
                  className="max-w-[180px] truncate rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-elevated)] px-2 py-1 text-[11px] text-[var(--text-primary)] focus:border-blue-500 focus:outline-none"
                >
                  <option value="" disabled>
                    Select run ({runs.length})
                  </option>
                  {runs.map((r, idx) => (
                    <option key={r.id} value={r.id}>
                      #{runs.length - idx} · {r.query.slice(0, 34)}
                    </option>
                  ))}
                </select>
                {activeRun && !isRunning && (
                  <button
                    type="button"
                    onClick={() => onDeleteRun(activeRun.id)}
                    title="Delete this comparison"
                    className="rounded-lg p-1 text-[var(--text-muted)] transition hover:bg-[var(--bg-elevated)] hover:text-rose-500 cursor-pointer"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                )}
                <button
                  type="button"
                  onClick={onNewComparison}
                  disabled={isRunning}
                  title="New comparison prompt"
                  className="inline-flex items-center gap-1 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-elevated)] px-2 py-1 text-[11px] font-medium text-[var(--text-secondary)] transition hover:text-[var(--text-primary)] disabled:opacity-40 cursor-pointer"
                >
                  <Plus className="h-3 w-3" />
                  <span className="hidden sm:inline">New</span>
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main Scrollable Arena Body */}
      <div className="flex-1 overflow-y-auto px-4 py-5 sm:px-6">
        <div className="mx-auto max-w-6xl space-y-4">
          {!activeRun ? (
            /* Empty Arena Welcome */
            <div className="mx-auto max-w-3xl py-6 sm:py-10">
              <div className="inline-flex items-center gap-2 rounded-full border border-indigo-500/30 bg-indigo-500/10 px-3 py-1 text-[11px] font-medium text-indigo-600 dark:text-indigo-300">
                <Columns2 className="h-3.5 w-3.5" />
                <span>Live Side-by-Side Ablation Arena · Parallel SSE Execution</span>
              </div>

              <h1 className="mt-4 text-2xl font-bold tracking-tight text-[var(--text-primary)] sm:text-3xl">
                Compare {leftShort} vs. {rightShort} on the exact same query.
              </h1>
              <p className="mt-2 max-w-2xl text-sm leading-relaxed text-[var(--text-secondary)]">
                Fire both RAG configurations simultaneously against your Qdrant corpus.
                Measure real-time <strong>context token savings</strong>,{" "}
                <strong>Dynamic Top-K</strong> cutoffs,{" "}
                <strong>CRAG confidence</strong>,{" "}
                <strong>Two-Tier Citation Audit</strong>, and hallucination protection
                side by side.
              </p>

              <div className="mt-7">
                <div className="mb-2.5 text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                  High-Contrast Ablation Prompts
                </div>
                <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
                  {COMPARE_DEMO_QUERIES.map((demo) => (
                    <button
                      key={demo.id}
                      type="button"
                      onClick={() => onSubmitCompare(demo.query)}
                      className="group flex flex-col justify-between rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-panel)] p-3.5 text-left transition hover:border-blue-500/45 hover:bg-[var(--bg-elevated)]/60 cursor-pointer"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-xs font-semibold text-[var(--text-primary)] group-hover:text-blue-500 dark:group-hover:text-blue-400">
                          {demo.title}
                        </span>
                        <div className="flex shrink-0 items-center gap-1">
                          <span className="rounded border border-emerald-500/30 bg-emerald-500/10 px-1.5 py-0.5 font-mono text-[10px] font-medium text-emerald-600 dark:text-emerald-300">
                            {demo.highlight}
                          </span>
                          <span className="rounded border border-blue-500/30 bg-blue-500/10 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-blue-600 dark:text-blue-300">
                            {demo.lang}
                          </span>
                        </div>
                      </div>
                      <p className="mt-2 line-clamp-2 text-xs leading-relaxed text-[var(--text-secondary)]">
                        {demo.query}
                      </p>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <>
              {/* Evaluated Query Banner */}
              <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-blue-500/25 bg-blue-600/[0.07] px-4 py-3">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-1.5 text-[10.5px] font-semibold uppercase tracking-wider text-blue-600 dark:text-blue-300">
                    <Sparkles className="h-3 w-3" />
                    <span>Evaluated Prompt</span>
                    <span className="opacity-50">·</span>
                    <span className="font-mono">
                      {activeRun.left.presetLabel} vs {activeRun.right.presetLabel}
                    </span>
                  </div>
                  <p className="mt-1 text-sm font-medium leading-relaxed text-[var(--text-primary)]">
                    {activeRun.query}
                  </p>
                </div>
              </div>

              {/* Delta Scorecard Ribbon (once both sides finish) */}
              {delta && (
                <div
                  data-testid="compare-delta-scorecard"
                  className="space-y-2.5 rounded-2xl border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/45 p-3.5"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-[var(--text-secondary)]">
                      {delta.contextTokensDelta <= 0 ? (
                        <TrendingDown className="h-3.5 w-3.5 text-emerald-500" />
                      ) : (
                        <TrendingUp className="h-3.5 w-3.5 text-blue-500" />
                      )}
                      <span>A → B Ablation Delta Summary</span>
                    </div>

                    {delta.highlights.length > 0 && (
                      <div className="flex flex-wrap items-center gap-1.5">
                        {delta.highlights.map((hl) => {
                          const cls =
                            hl.tone === "emerald"
                              ? "border-emerald-500/35 bg-emerald-500/12 text-emerald-600 dark:text-emerald-300"
                              : hl.tone === "purple"
                                ? "border-purple-500/35 bg-purple-500/12 text-purple-600 dark:text-purple-300"
                                : hl.tone === "amber"
                                  ? "border-amber-500/35 bg-amber-500/12 text-amber-600 dark:text-amber-300"
                                  : hl.tone === "rose"
                                    ? "border-rose-500/35 bg-rose-500/12 text-rose-600 dark:text-rose-300"
                                    : "border-blue-500/35 bg-blue-500/12 text-blue-600 dark:text-blue-300";
                          return (
                            <span
                              key={hl.label}
                              className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10.5px] font-semibold ${cls}`}
                            >
                              {hl.tone === "emerald" ? (
                                <CheckCircle2 className="h-3 w-3" />
                              ) : (
                                <ShieldAlert className="h-3 w-3" />
                              )}
                              {hl.label}
                            </span>
                          );
                        })}
                      </div>
                    )}
                  </div>

                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
                    <DeltaTile
                      label="Context Tokens"
                      value={`${delta.leftContextTokens.toLocaleString("en-US")} → ${delta.rightContextTokens.toLocaleString("en-US")}`}
                      deltaBadge={formatSignedPct(delta.contextTokensPct)}
                      tone={
                        delta.contextTokensDelta < 0
                          ? "emerald"
                          : delta.contextTokensDelta > 0
                            ? "amber"
                            : "neutral"
                      }
                    />
                    <DeltaTile
                      label="Selected (k)"
                      value={`${delta.leftK} → ${delta.rightK}`}
                      deltaBadge={
                        delta.kDelta === 0
                          ? "same"
                          : `${delta.kDelta > 0 ? "+" : ""}${delta.kDelta} k`
                      }
                      tone={
                        delta.kDelta < 0
                          ? "emerald"
                          : delta.kDelta > 0
                            ? "blue"
                            : "neutral"
                      }
                    />
                    <DeltaTile
                      label="CRAG Confidence"
                      value={`${delta.leftConfidence !== null ? delta.leftConfidence.toFixed(2) : "Off"} → ${delta.rightConfidence !== null ? delta.rightConfidence.toFixed(2) : "Off"}`}
                      deltaBadge={
                        delta.rightConfidenceLevel
                          ? delta.rightConfidenceLevel.toUpperCase()
                          : "N/A"
                      }
                      tone={
                        delta.rightConfidenceLevel === "high"
                          ? "emerald"
                          : delta.rightConfidenceLevel === "medium"
                            ? "amber"
                            : delta.rightConfidenceLevel === "low"
                              ? "rose"
                              : "neutral"
                      }
                    />
                    <DeltaTile
                      label="Citation Audit"
                      value={`${delta.leftClaimSupport !== null ? `${Math.round(delta.leftClaimSupport * 100)}%` : "Off"} → ${delta.rightClaimSupport !== null ? `${Math.round(delta.rightClaimSupport * 100)}%` : "Off"}`}
                      deltaBadge={
                        delta.rightAuditPassed === true
                          ? "PASS"
                          : delta.rightAuditPassed === false
                            ? "FLAG"
                            : "N/A"
                      }
                      tone={
                        delta.rightAuditPassed === true
                          ? "emerald"
                          : delta.rightAuditPassed === false
                            ? "rose"
                            : "neutral"
                      }
                    />
                    <DeltaTile
                      label="Latency / Cost"
                      value={`${(delta.leftLatencyMs / 1000).toFixed(1)}s → ${(delta.rightLatencyMs / 1000).toFixed(1)}s`}
                      deltaBadge={`${delta.costDeltaUsd >= 0 ? "+" : "-"}$${Math.abs(delta.costDeltaUsd).toFixed(5)}`}
                      tone={delta.costDeltaUsd <= 0 ? "emerald" : "neutral"}
                    />
                  </div>
                </div>
              )}

              {/* Side-by-Side Pipeline Columns */}
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                <ArenaColumnCard
                  sideKey="left"
                  runId={activeRun.id}
                  side={activeRun.left}
                  isInspected={inspectedSide === "left"}
                  useStreaming={useStreaming}
                  onInspect={(tab) => onInspectSide("left", tab)}
                  onCitationClick={(cid) => onCitationClick("left", cid)}
                  onSubmitFeedback={
                    onSubmitSideFeedback
                      ? (score, comment) =>
                          onSubmitSideFeedback(activeRun.id, "left", score, comment)
                      : undefined
                  }
                />
                <ArenaColumnCard
                  sideKey="right"
                  runId={activeRun.id}
                  side={activeRun.right}
                  isInspected={inspectedSide === "right"}
                  useStreaming={useStreaming}
                  onInspect={(tab) => onInspectSide("right", tab)}
                  onCitationClick={(cid) => onCitationClick("right", cid)}
                  onSubmitFeedback={
                    onSubmitSideFeedback
                      ? (score, comment) =>
                          onSubmitSideFeedback(activeRun.id, "right", score, comment)
                      : undefined
                  }
                />
              </div>
            </>
          )}
        </div>
      </div>

      {/* Bottom Sticky Arena Composer */}
      <div className="shrink-0 border-t border-[var(--border-subtle)] bg-[var(--bg-panel)]/90 px-4 py-3 backdrop-blur sm:px-6">
        <div className="mx-auto max-w-4xl">
          <div className="relative flex items-end gap-2 rounded-2xl border border-[var(--border-strong)] bg-[var(--bg-canvas)] px-3.5 py-2.5 shadow-xs focus-within:border-blue-500 focus-within:ring-2 focus-within:ring-blue-500/15">
            <textarea
              ref={textareaRef}
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isRunning}
              placeholder={`Compare ${leftShort} vs ${rightShort} on any EN/TR query... (Enter to run both in parallel)`}
              className="max-h-36 flex-1 resize-none bg-transparent py-1 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none disabled:opacity-60"
            />
            {isRunning ? (
              <button
                type="button"
                onClick={onStopCompare}
                title="Stop both streams"
                className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-rose-600 text-white transition hover:bg-rose-500 cursor-pointer"
              >
                <Square className="h-3.5 w-3.5 fill-current" />
              </button>
            ) : (
              <button
                type="button"
                onClick={handleSend}
                disabled={!input.trim()}
                title="Run side-by-side comparison"
                className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-blue-600 text-white transition hover:bg-blue-500 disabled:opacity-35 cursor-pointer"
              >
                <ArrowUp className="h-4 w-4" />
              </button>
            )}
          </div>

          <div className="mt-1.5 flex items-center justify-between px-1 text-[10.5px] text-[var(--text-muted)]">
            <div className="flex items-center gap-2">
              <span>
                Arena:{" "}
                <strong className="text-[var(--text-secondary)]">
                  {leftShort} vs {rightShort}
                </strong>
              </span>
              <span>·</span>
              <span>Parallel {useStreaming ? "SSE" : "Batch"}</span>
              {budgetCapLabel && (
                <>
                  <span>·</span>
                  <span className="font-mono text-blue-400">
                    Cap {budgetCapLabel}/side
                  </span>
                </>
              )}
            </div>
            <span className="hidden sm:inline">
              Click either card or <code className="font-mono">[C1]</code> to focus the
              Glass-Box Inspector
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
