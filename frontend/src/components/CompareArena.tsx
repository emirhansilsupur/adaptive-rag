import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  ArrowLeftRight,
  ArrowRight,
  ArrowUp,
  Check,
  CheckCircle2,
  ExternalLink,
  FileText,
  Globe,
  PanelRightOpen,
  Plus,
  ShieldAlert,
  Square,
  ThumbsDown,
  ThumbsUp,
  Trash2,
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

function DeltaLedgerCell({
  label,
  value,
  deltaBadge,
  tone = "neutral",
}: {
  label: string;
  value: string;
  deltaBadge?: string;
  tone?: "emerald" | "amber" | "rose" | "sky" | "neutral";
}) {
  const badgeTone =
    tone === "emerald"
      ? "text-emerald-600 dark:text-emerald-400"
      : tone === "amber"
        ? "text-amber-600 dark:text-amber-400"
        : tone === "rose"
          ? "text-rose-600 dark:text-rose-400"
          : tone === "sky"
            ? "text-sky-600 dark:text-sky-400"
            : "text-[var(--text-muted)]";

  return (
    <div className="bg-[var(--bg-surface)] px-3 py-2.5">
      <div className="flex items-center justify-between gap-1.5">
        <span className="truncate font-mono text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
          {label}
        </span>
        {deltaBadge && (
          <span
            className={`shrink-0 whitespace-nowrap font-mono text-[10px] font-semibold tabular-nums ${badgeTone}`}
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
      className={`flex min-w-0 flex-col rounded-lg border p-4 transition-colors ${
        isInspected && result
          ? "border-[var(--border-strong)] bg-[var(--bg-surface)]"
          : "border-[var(--border-subtle)] bg-[var(--bg-surface)]/80 hover:border-[var(--border-strong)]"
      }`}
    >
      {/* Column Header */}
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-[var(--border-subtle)] pb-2.5">
        <div className="flex min-w-0 items-center gap-2">
          <span
            className={`inline-flex h-5 w-5 shrink-0 items-center justify-center rounded border font-mono text-[10px] font-semibold ${
              isBaseline
                ? "border-[var(--border-strong)] bg-[var(--bg-elevated)] text-[var(--text-secondary)]"
                : "border-[var(--border-strong)] bg-[var(--cta-bg)] text-[var(--cta-fg)]"
            }`}
          >
            {isBaseline ? "A" : "B"}
          </span>
          <span className="truncate text-xs font-semibold text-[var(--text-primary)]">
            {side.presetLabel}
          </span>
          <span className="font-mono text-[10px] text-[var(--text-muted)]">
            {isBaseline ? "baseline" : "challenger"}
          </span>
        </div>

        <div className="flex shrink-0 items-center gap-1.5">
          {side.status === "running" && (
            <span className="inline-flex items-center gap-1.5 rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)] px-2 py-0.5 font-mono text-[10px] text-[var(--text-secondary)]">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[var(--text-primary)]" />
              running
            </span>
          )}
          {result && (
            <>
              <span className="rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60 px-1.5 py-0.5 font-mono text-[10px] text-[var(--text-secondary)] tabular-nums">
                k={chunks.length} · {ctxTokens?.toLocaleString("en-US")} tok
              </span>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onInspect("trace");
                }}
                className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 font-mono text-[10px] font-medium transition cursor-pointer ${
                  isInspected
                    ? "border-[var(--border-strong)] bg-[var(--bg-elevated)] text-[var(--text-primary)]"
                    : "border-transparent text-[var(--text-muted)] hover:border-[var(--border-subtle)] hover:text-[var(--text-primary)]"
                }`}
              >
                <PanelRightOpen className="h-3 w-3" />
                <span>{isInspected ? "Inspecting" : "Inspect"}</span>
              </button>
            </>
          )}
        </div>
      </div>

      {/* Running State */}
      {side.status === "running" && (
        <div className="space-y-2.5">
          <LivePipelineTracker
            mode="streaming"
            steps={side.liveSteps}
            liveUsage={side.liveUsage}
          />
          <div className="flex items-center gap-2 font-mono text-[11px] text-[var(--text-muted)]">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[var(--text-primary)]" />
            <span>
              {useStreaming
                ? `Streaming ${side.presetLabel}...`
                : `Running ${side.presetLabel}...`}
            </span>
          </div>
        </div>
      )}

      {/* Error State */}
      {side.status === "error" && (
        <div className="flex items-start gap-2 rounded-lg border border-rose-500/35 bg-rose-500/10 p-3 text-xs text-rose-600 dark:text-rose-300">
          <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <div className="min-w-0">
            <div className="font-medium">{side.presetLabel} Failed</div>
            <p className="mt-0.5 break-words opacity-90">{side.error}</p>
          </div>
        </div>
      )}

      {/* Completed Result */}
      {result && (
        <div className="flex flex-1 flex-col">
          <div className="mb-3.5 border-b border-[var(--border-subtle)] pb-3">
            <LivePipelineTracker
              mode="completed"
              steps={result.trace || []}
              totalLatencyMs={result.total_latency_ms}
              onOpenInspector={() => onInspect("trace")}
            />
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
                    className="inline-flex max-w-[160px] items-center gap-1 rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60 px-1.5 py-0.5 text-[10.5px] text-[var(--text-secondary)] transition hover:border-[var(--border-strong)] hover:text-[var(--text-primary)] cursor-pointer"
                  >
                    <span
                      className={`shrink-0 font-mono text-[10px] font-semibold ${
                        isWeb
                          ? "text-amber-600 dark:text-amber-300"
                          : "text-sky-600 dark:text-sky-300"
                      }`}
                    >
                      {cid}
                    </span>
                    {isWeb ? (
                      <Globe className="h-2.5 w-2.5 shrink-0 text-[var(--text-muted)]" />
                    ) : (
                      <FileText className="h-2.5 w-2.5 shrink-0 text-[var(--text-muted)]" />
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
                  className="font-mono text-[10px] text-[var(--text-muted)] hover:text-[var(--text-primary)] cursor-pointer"
                >
                  +{chunks.length - 4} more
                </button>
              )}
              {chunks.length === 0 && (
                <span className="font-mono text-[10px] text-[var(--text-muted)]">
                  0 passages
                </span>
              )}
            </div>

            {onSubmitFeedback && result.trace_id && (
              <div
                className="inline-flex items-center gap-1"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="inline-flex items-center gap-0.5 rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60 p-0.5">
                  <button
                    type="button"
                    aria-label={`Mark ${side.presetLabel} answer helpful`}
                    disabled={savingFeedback}
                    onClick={() => void submitSideRating(1)}
                    className={`rounded p-1 transition cursor-pointer disabled:opacity-50 ${
                      side.feedback?.score === 1
                        ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400"
                        : "text-[var(--text-muted)] hover:bg-[var(--bg-surface)] hover:text-[var(--text-primary)]"
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
                    className={`rounded p-1 transition cursor-pointer disabled:opacity-50 ${
                      side.feedback?.score === 0 || commentOpen
                        ? "bg-rose-500/15 text-rose-600 dark:text-rose-400"
                        : "text-[var(--text-muted)] hover:bg-[var(--bg-surface)] hover:text-[var(--text-primary)]"
                    }`}
                  >
                    <ThumbsDown className="h-3 w-3" />
                  </button>
                </div>
                {side.feedback && (
                  <span className="inline-flex items-center gap-0.5 rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60 px-1.5 py-0.5 font-mono text-[9.5px] text-[var(--text-muted)]">
                    <Check className="h-2.5 w-2.5 text-emerald-500" />
                    {side.feedback.persistedToLangfuse ? "lf" : "ok"}
                  </span>
                )}
              </div>
            )}
          </div>

          {commentOpen && (
            <div
              onClick={(e) => e.stopPropagation()}
              className="mt-2 flex flex-wrap items-center gap-1.5 rounded-md border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60 p-2"
            >
              <input
                type="text"
                value={commentDraft}
                onChange={(e) => setCommentDraft(e.target.value)}
                placeholder="Optional note for this preset..."
                className="min-w-[140px] flex-1 rounded border border-[var(--border-subtle)] bg-[var(--bg-surface)] px-2 py-1 text-xs text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:border-[var(--text-muted)] focus:outline-none"
              />
              <button
                type="button"
                disabled={savingFeedback}
                onClick={() =>
                  void submitSideRating(0, commentDraft.trim() || undefined)
                }
                className="rounded bg-[var(--cta-bg)] px-2 py-1 text-[10.5px] font-medium text-[var(--cta-fg)] hover:bg-[var(--cta-hover)] disabled:opacity-50 cursor-pointer"
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
      {/* Arena Configuration Toolbar */}
      <div className="shrink-0 border-b border-[var(--border-subtle)] bg-[var(--bg-panel)] px-4 py-2 sm:px-6">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-2">
          {/* A vs B Selectors */}
          <div className="flex flex-wrap items-center gap-1.5">
            <div className="flex items-center gap-1.5">
              <span className="inline-flex h-5 w-5 items-center justify-center rounded border border-[var(--border-strong)] bg-[var(--bg-surface)] font-mono text-[10px] font-semibold text-[var(--text-secondary)]">
                A
              </span>
              <select
                aria-label="Baseline preset (Left)"
                value={leftPreset}
                disabled={isRunning}
                onChange={(e) => onLeftPresetChange(e.target.value as AblationPreset)}
                className="ar-select h-7.5 rounded-md border border-[var(--border-subtle)] bg-[var(--bg-surface)] px-2.5 text-xs font-medium text-[var(--text-primary)] focus:border-[var(--text-muted)] focus:outline-none disabled:opacity-50"
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
              className="flex h-7.5 w-7.5 items-center justify-center rounded-md border border-[var(--border-subtle)] bg-[var(--bg-surface)] text-[var(--text-secondary)] transition hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)] disabled:opacity-40 cursor-pointer"
            >
              <ArrowLeftRight className="h-3 w-3" />
            </button>

            <div className="flex items-center gap-1.5">
              <span className="inline-flex h-5 w-5 items-center justify-center rounded border border-[var(--border-strong)] bg-[var(--cta-bg)] font-mono text-[10px] font-semibold text-[var(--cta-fg)]">
                B
              </span>
              <select
                aria-label="Challenger preset (Right)"
                value={rightPreset}
                disabled={isRunning}
                onChange={(e) => onRightPresetChange(e.target.value as AblationPreset)}
                className="ar-select h-7.5 rounded-md border border-[var(--border-subtle)] bg-[var(--bg-surface)] px-2.5 text-xs font-medium text-[var(--text-primary)] focus:border-[var(--text-muted)] focus:outline-none disabled:opacity-50"
              >
                {ARENA_PRESET_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Quick Pairs + Past Comparisons */}
          <div className="flex flex-wrap items-center gap-1.5">
            <div className="hidden items-center rounded-md border border-[var(--border-subtle)] bg-[var(--bg-surface)] p-0.5 xl:inline-flex">
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
                    className={`rounded px-2 py-0.5 text-[10.5px] font-medium transition cursor-pointer disabled:opacity-40 ${
                      active
                        ? "bg-[var(--bg-elevated)] text-[var(--text-primary)]"
                        : "text-[var(--text-muted)] hover:text-[var(--text-primary)]"
                    }`}
                  >
                    {pair.label}
                  </button>
                );
              })}
            </div>

            {runs.length > 0 && (
              <div className="flex items-center gap-1">
                <select
                  aria-label="Past Arena Comparisons"
                  value={activeRun?.id ?? ""}
                  onChange={(e) => {
                    if (e.target.value) onSelectRun(e.target.value);
                  }}
                  className="ar-select h-7.5 max-w-[180px] truncate rounded-md border border-[var(--border-subtle)] bg-[var(--bg-surface)] px-2 text-[11px] text-[var(--text-primary)] focus:outline-none"
                >
                  <option value="" disabled>
                    History ({runs.length})
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
                    title="Delete comparison"
                    className="rounded-md p-1.5 text-[var(--text-muted)] transition hover:bg-[var(--bg-elevated)] hover:text-rose-500 cursor-pointer"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                )}
                <button
                  type="button"
                  onClick={onNewComparison}
                  disabled={isRunning}
                  title="New comparison"
                  className="inline-flex h-7.5 items-center gap-1 rounded-md border border-[var(--border-subtle)] bg-[var(--bg-surface)] px-2 text-[11px] font-medium text-[var(--text-secondary)] transition hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)] disabled:opacity-40 cursor-pointer"
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
            /* Empty Arena Console */
            <div className="mx-auto max-w-3xl py-6 sm:py-12">
              <div className="font-mono text-[10.5px] font-medium uppercase tracking-widest text-[var(--text-muted)]">
                Parallel SSE Ablation Arena
              </div>

              <h1 className="mt-2.5 text-xl font-semibold tracking-tight text-[var(--text-primary)] sm:text-2xl">
                Compare {leftShort} vs. {rightShort} on identical queries.
              </h1>
              <p className="mt-2 max-w-2xl text-[13.5px] leading-relaxed text-[var(--text-secondary)]">
                Execute baseline and challenger pipelines concurrently against the same
                Qdrant index to measure context token deltas, Dynamic Top-K selection,
                CRAG confidence, and citation entailment side by side.
              </p>

              <div className="mt-8">
                <div className="mb-2 font-mono text-[10.5px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
                  Controlled Ablation Prompts
                </div>
                <div className="divide-y divide-[var(--border-subtle)] overflow-hidden rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-surface)]">
                  {COMPARE_DEMO_QUERIES.map((demo) => (
                    <button
                      key={demo.id}
                      type="button"
                      onClick={() => onSubmitCompare(demo.query)}
                      className="group flex w-full items-start justify-between gap-4 px-4 py-3 text-left transition hover:bg-[var(--bg-elevated)]/65 cursor-pointer"
                    >
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-medium text-[var(--text-primary)]">
                            {demo.title}
                          </span>
                          <span className="rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)] px-1.5 py-0.2 font-mono text-[9.5px] text-[var(--text-secondary)]">
                            {demo.highlight}
                          </span>
                          <span className="font-mono text-[9.5px] text-[var(--text-muted)]">
                            {demo.lang}
                          </span>
                        </div>
                        <p className="mt-1 line-clamp-1 text-xs text-[var(--text-secondary)]">
                          {demo.query}
                        </p>
                      </div>
                      <ArrowRight className="mt-1 h-3.5 w-3.5 shrink-0 text-[var(--text-muted)] opacity-0 transition group-hover:translate-x-0.5 group-hover:text-[var(--text-primary)] group-hover:opacity-100" />
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <>
              {/* Evaluated Query Header */}
              <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-surface)] px-4 py-3">
                <div className="flex items-center gap-2 font-mono text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
                  <span>Query Under Evaluation</span>
                  <span>·</span>
                  <span className="text-[var(--text-secondary)]">
                    {activeRun.left.presetLabel} vs {activeRun.right.presetLabel}
                  </span>
                </div>
                <p className="mt-1 text-sm font-medium leading-snug text-[var(--text-primary)]">
                  {activeRun.query}
                </p>
              </div>

              {/* Hairline Ledger Delta Scorecard */}
              {delta && (
                <div
                  data-testid="compare-delta-scorecard"
                  className="overflow-hidden rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-surface)]"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[var(--border-subtle)] px-3.5 py-2">
                    <span className="font-mono text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                      A → B Ablation Delta
                    </span>

                    {delta.highlights.length > 0 && (
                      <div className="flex flex-wrap items-center gap-1.5">
                        {delta.highlights.map((hl) => {
                          const cls =
                            hl.tone === "emerald"
                              ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
                              : hl.tone === "amber"
                                ? "border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-300"
                                : hl.tone === "rose"
                                  ? "border-rose-500/30 bg-rose-500/10 text-rose-700 dark:text-rose-300"
                                  : "border-[var(--border-subtle)] bg-[var(--bg-elevated)] text-[var(--text-secondary)]";
                          return (
                            <span
                              key={hl.label}
                              className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 font-mono text-[10px] font-medium ${cls}`}
                            >
                              {hl.tone === "emerald" ? (
                                <CheckCircle2 className="h-2.5 w-2.5" />
                              ) : (
                                <ShieldAlert className="h-2.5 w-2.5" />
                              )}
                              {hl.label}
                            </span>
                          );
                        })}
                      </div>
                    )}
                  </div>

                  <div className="grid grid-cols-2 gap-px bg-[var(--border-subtle)] sm:grid-cols-3 lg:grid-cols-5">
                    <DeltaLedgerCell
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
                    <DeltaLedgerCell
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
                            ? "sky"
                            : "neutral"
                      }
                    />
                    <DeltaLedgerCell
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
                    <DeltaLedgerCell
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
                    <DeltaLedgerCell
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
      <div className="shrink-0 border-t border-[var(--border-subtle)] bg-[var(--bg-panel)]/95 px-4 py-3 backdrop-blur sm:px-6">
        <div className="mx-auto max-w-4xl">
          <div className="relative flex items-end gap-2 rounded-lg border border-[var(--border-strong)] bg-[var(--bg-surface)] px-3 py-2 transition focus-within:border-[var(--text-muted)]">
            <textarea
              ref={textareaRef}
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isRunning}
              placeholder={`Run ${leftShort} vs ${rightShort} on an EN/TR query...`}
              className={`max-h-36 flex-1 resize-none bg-transparent py-1 text-[13.5px] leading-5 text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none disabled:opacity-60 ${
                input.includes("\n") || input.length > 90
                  ? "overflow-y-auto"
                  : "overflow-hidden"
              }`}
            />
            {isRunning ? (
              <button
                type="button"
                onClick={onStopCompare}
                title="Stop both streams"
                className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-rose-600 text-white transition hover:bg-rose-500 cursor-pointer"
              >
                <Square className="h-3 w-3 fill-current" />
              </button>
            ) : (
              <button
                type="button"
                onClick={handleSend}
                disabled={!input.trim()}
                title="Run side-by-side comparison"
                className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[var(--cta-bg)] text-[var(--cta-fg)] transition hover:bg-[var(--cta-hover)] disabled:opacity-25 cursor-pointer"
              >
                <ArrowUp className="h-3.5 w-3.5" />
              </button>
            )}
          </div>

          <div className="mt-1.5 flex items-center justify-between px-0.5 font-mono text-[10px] text-[var(--text-muted)]">
            <div className="flex items-center gap-2">
              <span>
                arena:{leftShort.toLowerCase().replace(/\s+/g, "-")} vs{" "}
                {rightShort.toLowerCase().replace(/\s+/g, "-")}
              </span>
              <span>·</span>
              <span>parallel-{useStreaming ? "sse" : "batch"}</span>
              {budgetCapLabel && (
                <>
                  <span>·</span>
                  <span>cap:{budgetCapLabel}/side</span>
                </>
              )}
            </div>
            <div className="hidden items-center gap-1.5 sm:flex">
              <span>Run parallel</span>
              <kbd className="ar-kbd">↵</kbd>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
