import React, { useEffect, useMemo, useState } from "react";
import {
  ArrowLeftRight,
  CheckCircle2,
  PanelRightOpen,
  Plus,
  ShieldAlert,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  Check,
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
import { AUDIT_LABEL, type AuditState } from "../lib/audit";
import { formatDuration, formatInt, formatUsd } from "../lib/format";
import type {
  AblationPreset,
  CompareRun,
  CompareSideKey,
  CompareSideState,
  InspectorTab,
} from "../types/api";
import { AnswerTelemetryBar } from "./AnswerTelemetryBar";
import { SourcesRow } from "./ChatWorkspace";
import { Composer } from "./Composer";
import { LivePipelineTracker } from "./LivePipelineTracker";
import { MarkdownWithCitations } from "./MarkdownWithCitations";
import {
  Button,
  Callout,
  cn,
  IconButton,
  INPUT_CLASS,
  SectionHeading,
  Segmented,
  SELECT_CLASS,
  Tag,
  type Tone,
} from "./ui/primitives";

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

const TONE_TEXT: Record<Tone, string> = {
  neutral: "text-ink-3",
  ok: "text-ok",
  warn: "text-warn",
  bad: "text-bad",
  accent: "text-accent",
};

/** A or B lane marker. B is filled so the challenger is always the darker lane. */
function LaneTile({ lane, className }: { lane: "A" | "B"; className?: string }) {
  return (
    <span
      aria-hidden
      className={cn(
        "inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-small font-semibold",
        lane === "A" ? "border border-line-strong bg-paper text-ink-2" : "bg-ink text-paper",
        className
      )}
    >
      {lane}
    </span>
  );
}

const AUDIT_DELTA_TONE: Record<AuditState, Tone> = {
  passed: "ok",
  flagged: "bad",
  unverified: "warn",
  structural: "neutral",
};

/** One side of the audit delta; an unverified side has no claim-support figure to show. */
function auditSideValue(state: AuditState | null, claimSupport: number | null): string {
  if (state === null || claimSupport === null) return "Off";
  if (state === "unverified") return "—";
  return `${Math.round(claimSupport * 100)}%`;
}

function DeltaCell({
  label,
  value,
  delta,
  tone = "neutral",
}: {
  label: string;
  value: string;
  delta?: string;
  tone?: Tone;
}) {
  return (
    <div className="min-w-0 bg-paper px-3.5 py-3">
      <div className="truncate text-caption text-ink-3">{label}</div>
      <div className="mt-1 truncate text-body font-medium tabular-nums text-ink">{value}</div>
      {delta && (
        <div className={cn("mt-0.5 truncate text-small font-medium tabular-nums", TONE_TEXT[tone])}>
          {delta}
        </div>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* One side of the comparison                                                 */
/* -------------------------------------------------------------------------- */

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

  const lane = sideKey === "left" ? "A" : "B";
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
      aria-label={`${lane}: ${side.presetLabel}`}
      onClick={() => result && onInspect()}
      className={cn(
        "flex min-w-0 flex-col rounded-xl border bg-paper transition-shadow",
        isInspected && result
          ? "border-accent/50 shadow-[0_0_0_3px_var(--accent-soft)]"
          : "border-line"
      )}
    >
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-3">
        <div className="flex min-w-0 items-center gap-2.5">
          <LaneTile lane={lane} />
          <div className="min-w-0">
            <div className="truncate text-ui font-semibold text-ink">{side.presetLabel}</div>
            <div className="text-caption text-ink-3">{lane === "A" ? "Baseline" : "Challenger"}</div>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {side.status === "running" && <Tag tone="accent">Running</Tag>}
          {result && (
            <>
              <span className="hidden text-small tabular-nums text-ink-3 sm:inline">
                {chunks.length} passages, {formatInt(ctxTokens ?? 0)} tokens
              </span>
              <Button
                variant="ghost"
                size="xs"
                icon={PanelRightOpen}
                onClick={(e) => {
                  e.stopPropagation();
                  onInspect("trace");
                }}
                className={isInspected ? "text-ink" : undefined}
              >
                {isInspected ? "Inspecting" : "Inspect"}
              </Button>
            </>
          )}
        </div>
      </header>

      <div className="flex flex-1 flex-col p-4">
        {side.status === "running" && (
          <LivePipelineTracker
            mode="streaming"
            steps={side.liveSteps}
            liveUsage={side.liveUsage}
            runningLabel={
              useStreaming ? `Running ${side.presetLabel}` : `Running ${side.presetLabel} in batch mode`
            }
          />
        )}

        {side.status === "error" && (
          <Callout live tone="bad" title={`${side.presetLabel} failed`}>
            {side.error}
          </Callout>
        )}

        {result && (
          <>
            <div className="rounded-lg border border-line bg-sunken px-3 py-2.5">
              <AnswerTelemetryBar result={result} onBadgeClick={(tab) => onInspect(tab)} />
              {(result.trace?.length ?? 0) > 0 && (
                <div className="mt-2">
                  <LivePipelineTracker
                    mode="completed"
                    steps={result.trace || []}
                    totalLatencyMs={result.total_latency_ms}
                    onOpenInspector={() => onInspect("trace")}
                  />
                </div>
              )}
            </div>

            <div className="mt-4 flex-1">
              <MarkdownWithCitations
                content={result.answer}
                evidenceChunks={chunks}
                onCitationClick={(cid) => onCitationClick(cid)}
                density="compact"
              />
            </div>

            <div className="mt-4">
              <SourcesRow
                chunks={chunks}
                limit={4}
                onCitationClick={onCitationClick}
                onShowAll={() => onInspect("evidence")}
              />
              {chunks.length === 0 && (
                <p className="text-small text-ink-3">No passages were used for this answer.</p>
              )}
            </div>

            {onSubmitFeedback && result.trace_id && (
              <div
                className="mt-3 flex flex-wrap items-center gap-1"
                onClick={(e) => e.stopPropagation()}
              >
                <IconButton
                  icon={ThumbsUp}
                  label={`Mark ${side.presetLabel} answer helpful`}
                  title="Helpful"
                  disabled={savingFeedback}
                  pressed={side.feedback?.score === 1}
                  onClick={() => void submitSideRating(1)}
                  className={side.feedback?.score === 1 ? "!text-ok" : undefined}
                />
                <IconButton
                  icon={ThumbsDown}
                  label={`Mark ${side.presetLabel} answer unhelpful`}
                  title="Needs work: add an optional note"
                  disabled={savingFeedback}
                  pressed={side.feedback?.score === 0 || commentOpen}
                  onClick={() => {
                    setCommentOpen((prev) => !prev);
                    setCommentDraft(side.feedback?.comment ?? "");
                  }}
                  className={side.feedback?.score === 0 || commentOpen ? "!text-bad" : undefined}
                />
                {side.feedback && (
                  <span className="ml-1 inline-flex items-center gap-1 text-small text-ink-3">
                    <Check className="h-3.5 w-3.5 text-ok" aria-hidden />
                    {side.feedback.persistedToLangfuse ? "Saved to Langfuse" : "Feedback saved"}
                  </span>
                )}
              </div>
            )}

            {commentOpen && (
              <div
                onClick={(e) => e.stopPropagation()}
                className="mt-2 flex flex-wrap items-center gap-2 rounded-lg border border-line bg-sunken p-2"
              >
                <input
                  type="text"
                  value={commentDraft}
                  autoFocus
                  onChange={(e) => setCommentDraft(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      void submitSideRating(0, commentDraft.trim() || undefined);
                    }
                    if (e.key === "Escape") setCommentOpen(false);
                  }}
                  placeholder="Optional note about this preset's answer"
                  className={cn(INPUT_CLASS, "h-8 min-w-[160px] flex-1 px-2.5")}
                />
                <Button
                  variant="primary"
                  size="sm"
                  loading={savingFeedback}
                  onClick={() => void submitSideRating(0, commentDraft.trim() || undefined)}
                >
                  Send feedback
                </Button>
              </div>
            )}

            {feedbackError && (
              <p className="mt-1.5 text-small text-bad">Feedback wasn't saved: {feedbackError}</p>
            )}
          </>
        )}
      </div>
    </section>
  );
};

/* -------------------------------------------------------------------------- */
/* Arena                                                                      */
/* -------------------------------------------------------------------------- */

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

  const activeRun = useMemo(
    () => (activeRunId ? (runs.find((r) => r.id === activeRunId) ?? null) : null),
    [runs, activeRunId]
  );

  const delta = useMemo(() => {
    if (!activeRun?.left.result || !activeRun?.right.result) return null;
    return computeCompareSummary(activeRun.left.result, activeRun.right.result);
  }, [activeRun]);

  const handleSend = () => {
    const cleaned = input.trim();
    if (!cleaned || isRunning) return;
    setInput("");
    onSubmitCompare(cleaned);
  };

  const leftShort = getPresetShortLabel(leftPreset);
  const rightShort = getPresetShortLabel(rightPreset);
  const leftMeta = ARENA_PRESET_OPTIONS.find((o) => o.value === leftPreset);
  const rightMeta = ARENA_PRESET_OPTIONS.find((o) => o.value === rightPreset);
  const activePair =
    COMPARE_PAIR_PRESETS.find((p) => p.left === leftPreset && p.right === rightPreset)?.id ?? "";

  return (
    <div className="flex min-w-0 flex-1 flex-col bg-paper">
      {/* Toolbar */}
      <div className="shrink-0 border-b border-line px-4 py-2.5 sm:px-6">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-x-4 gap-y-2">
          <div className="flex min-w-0 flex-wrap items-center gap-2">
            <div className="flex items-center gap-2">
              <LaneTile lane="A" />
              <select
                aria-label="Baseline preset (A)"
                value={leftPreset}
                disabled={isRunning}
                onChange={(e) => onLeftPresetChange(e.target.value as AblationPreset)}
                className={cn(SELECT_CLASS, "w-auto max-w-[220px] font-medium")}
              >
                {ARENA_PRESET_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>

            <IconButton
              icon={ArrowLeftRight}
              label="Swap A and B presets"
              onClick={onSwapPresets}
              disabled={isRunning}
            />

            <div className="flex items-center gap-2">
              <LaneTile lane="B" />
              <select
                aria-label="Challenger preset (B)"
                value={rightPreset}
                disabled={isRunning}
                onChange={(e) => onRightPresetChange(e.target.value as AblationPreset)}
                className={cn(SELECT_CLASS, "w-auto max-w-[220px] font-medium")}
              >
                {ARENA_PRESET_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {runs.length > 0 && (
              <div className="flex items-center gap-1">
                <select
                  aria-label="Past comparisons"
                  value={activeRun?.id ?? ""}
                  onChange={(e) => {
                    if (e.target.value) onSelectRun(e.target.value);
                  }}
                  className={cn(SELECT_CLASS, "h-7 w-auto max-w-[200px] text-small")}
                >
                  <option value="" disabled>
                    History ({runs.length})
                  </option>
                  {runs.map((r, idx) => (
                    <option key={r.id} value={r.id}>
                      {runs.length - idx}. {r.query.slice(0, 40)}
                    </option>
                  ))}
                </select>
                {activeRun && !isRunning && (
                  <IconButton
                    icon={Trash2}
                    tone="bad"
                    label="Delete this comparison"
                    onClick={() => onDeleteRun(activeRun.id)}
                  />
                )}
                <Button
                  size="sm"
                  icon={Plus}
                  onClick={onNewComparison}
                  disabled={isRunning}
                  title="Start a new comparison"
                >
                  New
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto">
        {!activeRun ? (
          <div className="mx-auto max-w-[46rem] px-4 pt-[8vh] pb-12 sm:px-8">
            <h1 className="text-display font-semibold tracking-[-0.015em] text-ink">
              Compare {leftShort} vs. {rightShort}
            </h1>
            <p className="mt-3 max-w-[36rem] text-body text-ink-2">
              Both presets answer the same question at the same time, against the same index. The
              scorecard then shows what changed: context tokens, passages kept, confidence, citation
              audit, time and cost.
            </p>

            <div className="mt-8 flex flex-wrap items-center gap-x-3 gap-y-2">
              <span className="text-small text-ink-3">Common pairs</span>
              <Segmented<string>
                ariaLabel="Common pairs"
                value={activePair}
                onChange={(id) => {
                  const pair = COMPARE_PAIR_PRESETS.find((p) => p.id === id);
                  if (!pair || isRunning) return;
                  onLeftPresetChange(pair.left);
                  onRightPresetChange(pair.right);
                }}
                options={COMPARE_PAIR_PRESETS.map((pair) => ({
                  value: pair.id,
                  label: pair.label,
                  disabled: isRunning,
                }))}
              />
            </div>

            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              {(
                [
                  ["A", leftMeta, "Baseline"],
                  ["B", rightMeta, "Challenger"],
                ] as const
              ).map(([lane, meta, role]) => (
                <div key={lane} className="rounded-lg border border-line p-3.5">
                  <div className="flex items-center gap-2.5">
                    <LaneTile lane={lane} />
                    <div className="min-w-0">
                      <div className="truncate text-ui font-semibold text-ink">{meta?.label}</div>
                      <div className="text-caption text-ink-3">{role}</div>
                    </div>
                  </div>
                  <p className="mt-2.5 text-small text-ink-2">{meta?.description}</p>
                </div>
              ))}
            </div>

            <section className="mt-10" aria-labelledby="arena-samples-heading">
              <SectionHeading>
                <span id="arena-samples-heading">Questions that show the difference</span>
              </SectionHeading>
              <ul className="mt-2.5 divide-y divide-line rounded-lg border border-line">
                {COMPARE_DEMO_QUERIES.map((demo) => (
                  <li key={demo.id}>
                    <button
                      type="button"
                      onClick={() => onSubmitCompare(demo.query)}
                      className="w-full cursor-pointer px-3.5 py-3 text-left transition-colors hover:bg-hover"
                    >
                      <span className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
                        <span className="text-ui font-medium text-ink">{demo.title}</span>
                        <Tag>{demo.highlight}</Tag>
                        <span className="ml-auto text-caption text-ink-3">{demo.lang}</span>
                      </span>
                      <span className="mt-0.5 line-clamp-1 text-small text-ink-2">
                        {demo.query}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          </div>
        ) : (
          <div className="mx-auto max-w-6xl space-y-5 px-4 py-6 sm:px-6">
            <section>
              <div className="text-caption text-ink-3">Question</div>
              <p className="mt-1 text-title font-semibold text-balance text-ink">{activeRun.query}</p>
            </section>

            {delta && (
              <section
                data-testid="compare-delta-scorecard"
                className="overflow-hidden rounded-xl border border-line"
              >
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line bg-sunken px-3.5 py-2.5">
                  <h2 className="flex items-center gap-2 text-ui font-semibold text-ink">
                    <LaneTile lane="A" className="h-5 w-5 text-caption" />
                    <span className="text-ink-3">to</span>
                    <LaneTile lane="B" className="h-5 w-5 text-caption" />
                    <span>What changed</span>
                  </h2>
                  {delta.highlights.length > 0 && (
                    <div className="flex flex-wrap items-center gap-1.5">
                      {delta.highlights.map((hl) => (
                        <Tag
                          key={hl.label}
                          tone={hl.tone === "emerald" ? "ok" : hl.tone === "rose" ? "bad" : hl.tone === "amber" ? "warn" : "accent"}
                          icon={hl.tone === "emerald" ? CheckCircle2 : ShieldAlert}
                        >
                          {hl.label}
                        </Tag>
                      ))}
                    </div>
                  )}
                </div>
                <div className="grid grid-cols-2 gap-px bg-line sm:grid-cols-3 lg:grid-cols-6">
                  <DeltaCell
                    label="Context tokens"
                    value={`${formatInt(delta.leftContextTokens)} → ${formatInt(delta.rightContextTokens)}`}
                    delta={formatSignedPct(delta.contextTokensPct)}
                    tone={
                      delta.contextTokensDelta < 0 ? "ok" : delta.contextTokensDelta > 0 ? "warn" : "neutral"
                    }
                  />
                  <DeltaCell
                    label="Passages kept"
                    value={`${delta.leftK} → ${delta.rightK}`}
                    delta={delta.kDelta === 0 ? "Same" : `${delta.kDelta > 0 ? "+" : ""}${delta.kDelta}`}
                    tone={delta.kDelta < 0 ? "ok" : "neutral"}
                  />
                  <DeltaCell
                    label="Confidence"
                    value={`${delta.leftConfidence !== null ? delta.leftConfidence.toFixed(2) : "Off"} → ${delta.rightConfidence !== null ? delta.rightConfidence.toFixed(2) : "Off"}`}
                    delta={
                      delta.rightConfidenceLevel
                        ? `B: ${delta.rightConfidenceLevel} confidence`
                        : "Not scored"
                    }
                    tone={
                      delta.rightConfidenceLevel === "high"
                        ? "ok"
                        : delta.rightConfidenceLevel === "medium"
                          ? "warn"
                          : delta.rightConfidenceLevel === "low"
                            ? "bad"
                            : "neutral"
                    }
                  />
                  <DeltaCell
                    label="Citation audit"
                    value={`${auditSideValue(delta.leftAudit, delta.leftClaimSupport)} → ${auditSideValue(delta.rightAudit, delta.rightClaimSupport)}`}
                    delta={delta.rightAudit ? `B: ${AUDIT_LABEL[delta.rightAudit].toLowerCase()}` : "Not run"}
                    tone={delta.rightAudit ? AUDIT_DELTA_TONE[delta.rightAudit] : "neutral"}
                  />
                  <DeltaCell
                    label="Time"
                    value={`${formatDuration(delta.leftLatencyMs)} → ${formatDuration(delta.rightLatencyMs)}`}
                    delta={`${delta.latencyDeltaMs > 0 ? "+" : delta.latencyDeltaMs < 0 ? "-" : ""}${formatDuration(Math.abs(delta.latencyDeltaMs))}`}
                    tone={delta.latencyDeltaMs > 0 ? "warn" : delta.latencyDeltaMs < 0 ? "ok" : "neutral"}
                  />
                  <DeltaCell
                    label="Cost"
                    value={`${formatUsd(delta.leftCostUsd)} → ${formatUsd(delta.rightCostUsd)}`}
                    delta={`${delta.costDeltaUsd > 0 ? "+" : ""}${formatUsd(delta.costDeltaUsd)}`}
                    tone={delta.costDeltaUsd < 0 ? "ok" : "neutral"}
                  />
                </div>
              </section>
            )}

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
                    ? (score, comment) => onSubmitSideFeedback(activeRun.id, "left", score, comment)
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
                    ? (score, comment) => onSubmitSideFeedback(activeRun.id, "right", score, comment)
                    : undefined
                }
              />
            </div>
          </div>
        )}
      </div>

      {/* Composer */}
      <div className="relative shrink-0 bg-paper px-4 pb-4 sm:px-6">
        <div
          aria-hidden
          className="pointer-events-none absolute inset-x-0 -top-8 h-8 bg-gradient-to-b from-transparent to-paper"
        />
        <div className="mx-auto w-full max-w-[52rem]">
          <Composer
            value={input}
            onChange={setInput}
            onSubmit={handleSend}
            onStop={onStopCompare}
            isRunning={isRunning}
            placeholder={`Ask a question to run ${leftShort} and ${rightShort} side by side`}
            submitLabel="Run comparison"
            stopLabel="Stop both runs"
            maxHeight={150}
            toolbar={
              <span className="flex items-center gap-1.5 px-1.5 text-small text-ink-3">
                <LaneTile lane="A" className="h-5 w-5 text-caption" />
                <span className="max-w-[9rem] truncate">{leftShort}</span>
                <span>and</span>
                <LaneTile lane="B" className="h-5 w-5 text-caption" />
                <span className="max-w-[9rem] truncate">{rightShort}</span>
              </span>
            }
            footer={
              <>
                <span>{useStreaming ? "Both runs stream in parallel" : "Both runs in batch mode"}</span>
                {budgetCapLabel && <span>Spend cap {budgetCapLabel} per side</span>}
              </>
            }
          />
        </div>
      </div>
    </div>
  );
};
