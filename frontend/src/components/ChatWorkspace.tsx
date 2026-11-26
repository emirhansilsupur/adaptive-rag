import React, { useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  ArrowRight,
  ArrowUp,
  Check,
  ExternalLink,
  FileText,
  Globe,
  Lock,
  PanelRightOpen,
  RotateCcw,
  SlidersHorizontal,
  Square,
  ThumbsDown,
  ThumbsUp,
  X,
} from "lucide-react";
import {
  DEFAULT_THREAD_CONTEXT,
  DEMO_QUERIES,
  getInstructionPresetLabel,
  INSTRUCTION_PRESETS,
  MAX_CUSTOM_INSTRUCTIONS_CHARS,
  resolveEffectiveInstructions,
} from "../constants/pipeline";
import type {
  ChatTurn,
  IndexedDocumentInfo,
  InspectorTab,
  InstructionPresetId,
  LiveStreamStep,
  ThreadContextConfig,
  TokenUsage,
} from "../types/api";
import { AnswerTelemetryBar } from "./AnswerTelemetryBar";
import { LivePipelineTracker } from "./LivePipelineTracker";
import { MarkdownWithCitations } from "./MarkdownWithCitations";

interface ChatWorkspaceProps {
  turns: ChatTurn[];
  selectedTurnIndex: number;
  isRunning: boolean;
  pendingQuery: string | null;
  liveSteps: LiveStreamStep[];
  liveUsage: Partial<TokenUsage> | null;
  streamError: string | null;
  presetShortLabel: string;
  useStreaming: boolean;
  budgetCapLabel: string | null;
  threadContext?: ThreadContextConfig;
  availableDocuments?: IndexedDocumentInfo[];
  onThreadContextChange?: (next: ThreadContextConfig) => void;
  onSubmitQuery: (query: string) => void;
  onStopStream: () => void;
  onInspectTurn: (turnIndex: number, tab?: InspectorTab) => void;
  onCitationClick: (turnIndex: number, citationId: string) => void;
  onSubmitFeedback?: (
    turnId: string,
    score: 0 | 1,
    comment?: string
  ) => Promise<void>;
}

export const ChatWorkspace: React.FC<ChatWorkspaceProps> = ({
  turns,
  selectedTurnIndex,
  isRunning,
  pendingQuery,
  liveSteps,
  liveUsage,
  streamError,
  presetShortLabel,
  useStreaming,
  budgetCapLabel,
  threadContext,
  availableDocuments = [],
  onThreadContextChange,
  onSubmitQuery,
  onStopStream,
  onInspectTurn,
  onCitationClick,
  onSubmitFeedback,
}) => {
  const [input, setInput] = useState("");
  const [contextPanelOpen, setContextPanelOpen] = useState(false);
  const [commentingTurnId, setCommentingTurnId] = useState<string | null>(null);
  const [pendingScore, setPendingScore] = useState<0 | 1>(0);
  const [commentDraft, setCommentDraft] = useState("");
  const [submittingFeedbackTurnId, setSubmittingFeedbackTurnId] =
    useState<string | null>(null);
  const [feedbackErrorByTurn, setFeedbackErrorByTurn] = useState<
    Record<string, string>
  >({});
  const scrollEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleFeedbackSubmit = async (
    turnId: string,
    score: 0 | 1,
    comment?: string
  ) => {
    if (!onSubmitFeedback || submittingFeedbackTurnId) return;
    setSubmittingFeedbackTurnId(turnId);
    setFeedbackErrorByTurn((prev) => {
      const next = { ...prev };
      delete next[turnId];
      return next;
    });
    try {
      await onSubmitFeedback(turnId, score, comment);
      setCommentingTurnId(null);
      setCommentDraft("");
    } catch (err) {
      setFeedbackErrorByTurn((prev) => ({
        ...prev,
        [turnId]: err instanceof Error ? err.message : String(err),
      }));
    } finally {
      setSubmittingFeedbackTurnId(null);
    }
  };

  useEffect(() => {
    scrollEndRef.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [turns.length, isRunning, liveSteps.length]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [input]);

  const handleSend = () => {
    const cleaned = input.trim();
    if (!cleaned || isRunning) return;
    setInput("");
    onSubmitQuery(cleaned);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const activeContext = threadContext ?? DEFAULT_THREAD_CONTEXT;
  const effectiveInstructions = resolveEffectiveInstructions(activeContext);
  const styleLabel = getInstructionPresetLabel(activeContext);
  const scopedDocIds = activeContext.scopedDocIds;
  const hasActiveThreadContext =
    Boolean(effectiveInstructions) || scopedDocIds.length > 0;

  const handleSelectInstructionPreset = (presetId: InstructionPresetId) => {
    if (!onThreadContextChange) return;
    const matched = INSTRUCTION_PRESETS.find((p) => p.id === presetId);
    if (presetId === "default") {
      onThreadContextChange({
        ...activeContext,
        instructionPreset: "default",
        customInstructions: "",
      });
      return;
    }
    if (presetId === "custom") {
      onThreadContextChange({
        ...activeContext,
        instructionPreset: "custom",
      });
      return;
    }
    onThreadContextChange({
      ...activeContext,
      instructionPreset: presetId,
      customInstructions: matched?.instructions ?? "",
    });
  };

  const handleCustomInstructionsInput = (value: string) => {
    if (!onThreadContextChange) return;
    const clipped = value.slice(0, MAX_CUSTOM_INSTRUCTIONS_CHARS);
    const matchingPreset = INSTRUCTION_PRESETS.find(
      (p) => p.id !== "default" && p.id !== "custom" && p.instructions === clipped.trim()
    );
    onThreadContextChange({
      ...activeContext,
      instructionPreset: matchingPreset ? matchingPreset.id : "custom",
      customInstructions: clipped,
    });
  };

  const handleToggleScopedDoc = (docId: string) => {
    if (!onThreadContextChange) return;
    const exists = scopedDocIds.includes(docId);
    const nextScoped = exists
      ? scopedDocIds.filter((id) => id !== docId)
      : [...scopedDocIds, docId];
    onThreadContextChange({
      ...activeContext,
      scopedDocIds: nextScoped,
    });
  };

  const handleResetThreadContext = () => {
    if (!onThreadContextChange) return;
    onThreadContextChange({ ...DEFAULT_THREAD_CONTEXT });
  };

  const hasContent = turns.length > 0 || isRunning;

  return (
    <div className="flex min-w-0 flex-1 flex-col bg-[var(--bg-canvas)]">
      {/* Scrollable Conversation Area */}
      <div className="flex-1 overflow-y-auto px-4 py-6 sm:px-8">
        <div className="mx-auto max-w-3xl">
          {!hasContent ? (
            /* Understated Engineering Empty State */
            <div className="py-6 sm:py-12">
              <div className="font-mono text-[10.5px] font-medium uppercase tracking-widest text-[var(--text-muted)]">
                AdaptiveRAG · Query-Aware Retrieval Engine
              </div>

              <h1 className="mt-2.5 text-xl font-semibold tracking-tight text-[var(--text-primary)] sm:text-2xl">
                Traceable retrieval and citation-audited synthesis.
              </h1>
              <p className="mt-2 max-w-2xl text-[13.5px] leading-relaxed text-[var(--text-secondary)]">
                Run queries in English or Turkish against Qdrant Hybrid Search (Dense +
                BM25 RRF). Inspect node latencies, Dynamic Top-K cutoffs, CRAG confidence
                gates, and{" "}
                <code className="rounded border border-sky-500/30 bg-sky-500/10 px-1 py-0.2 font-mono text-[11px] font-semibold text-sky-600 dark:text-sky-300">
                  C1
                </code>{" "}
                evidence entailment in the right pane.
              </p>

              {/* Hairline Ledger Demo Prompts */}
              <div className="mt-8">
                <div className="mb-2 flex items-center justify-between">
                  <span className="font-mono text-[10.5px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
                    Sample Evaluation Queries
                  </span>
                  <span className="font-mono text-[10.5px] text-[var(--text-muted)]">
                    Preset: {presetShortLabel}
                  </span>
                </div>

                <div className="divide-y divide-[var(--border-subtle)] overflow-hidden rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-surface)]">
                  {DEMO_QUERIES.map((demo) => (
                    <button
                      key={demo.id}
                      type="button"
                      onClick={() => onSubmitQuery(demo.query)}
                      className="group flex w-full items-start justify-between gap-4 px-4 py-3 text-left transition hover:bg-[var(--bg-elevated)]/65 cursor-pointer"
                    >
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-medium text-[var(--text-primary)]">
                            {demo.title}
                          </span>
                          <span className="rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)] px-1.5 py-0.2 font-mono text-[9.5px] text-[var(--text-secondary)]">
                            {demo.category}
                          </span>
                          <span className="font-mono text-[9.5px] font-medium text-[var(--text-muted)]">
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
            /* Editorial Research Log Thread */
            <div className="space-y-8 pb-4">
              {turns.map((turn, idx) => {
                const isSelected = idx === selectedTurnIndex;
                const chunks = turn.result.evidence_chunks || [];
                return (
                  <article key={turn.id} className="space-y-3">
                    {/* User Query Header (no speech bubble) */}
                    <div className="flex items-baseline gap-2.5">
                      <span className="shrink-0 rounded border border-[var(--border-subtle)] bg-[var(--bg-surface)] px-1.5 py-0.5 font-mono text-[10px] font-medium text-[var(--text-muted)] tabular-nums">
                        #{String(idx + 1).padStart(2, "0")}
                      </span>
                      <h2 className="text-[15px] font-medium leading-snug tracking-tight text-[var(--text-primary)]">
                        {turn.query}
                      </h2>
                    </div>

                    {/* Response Surface */}
                    <div
                      onClick={() => onInspectTurn(idx)}
                      className={`rounded-lg border p-4 sm:p-5 transition-colors ${
                        isSelected
                          ? "border-[var(--border-strong)] bg-[var(--bg-surface)]"
                          : "border-[var(--border-subtle)] bg-[var(--bg-surface)]/75 hover:border-[var(--border-strong)]"
                      }`}
                    >
                      {/* Execution Trace + Telemetry Header */}
                      <div className="mb-4 border-b border-[var(--border-subtle)] pb-3">
                        <LivePipelineTracker
                          mode="completed"
                          steps={turn.result.trace || []}
                          totalLatencyMs={turn.result.total_latency_ms}
                          onOpenInspector={() => onInspectTurn(idx, "trace")}
                        />
                        <AnswerTelemetryBar
                          result={turn.result}
                          onBadgeClick={(tab) => onInspectTurn(idx, tab)}
                        />
                      </div>

                      {/* Grounded Markdown Answer */}
                      <MarkdownWithCitations
                        content={turn.result.answer}
                        evidenceChunks={chunks}
                        onCitationClick={(cid) => onCitationClick(idx, cid)}
                      />

                      {/* Sources & Actions Footer */}
                      <div className="mt-4 flex flex-wrap items-center justify-between gap-2 border-t border-[var(--border-subtle)] pt-3">
                        <div className="flex flex-wrap items-center gap-1.5">
                          {chunks.slice(0, 6).map((ch, cIdx) => {
                            const cid = ch.citation_id || `#${cIdx + 1}`;
                            const isWeb = ch.source_type === "web";
                            return (
                              <button
                                key={ch.chunk_id || cIdx}
                                type="button"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  onCitationClick(idx, cid);
                                }}
                                title={`${ch.title || ch.doc_id}${ch.url ? ` (${ch.url})` : ""}`}
                                className="inline-flex max-w-[190px] items-center gap-1.5 rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60 px-2 py-0.5 text-[11px] text-[var(--text-secondary)] transition hover:border-[var(--border-strong)] hover:text-[var(--text-primary)] cursor-pointer"
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
                                <span className="truncate">
                                  {ch.title || ch.doc_id || "Source"}
                                </span>
                                {ch.url && (
                                  <ExternalLink className="h-2.5 w-2.5 shrink-0 opacity-60" />
                                )}
                              </button>
                            );
                          })}
                          {chunks.length > 6 && (
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                onInspectTurn(idx, "evidence");
                              }}
                              className="rounded px-1.5 py-0.5 font-mono text-[10.5px] text-[var(--text-muted)] hover:text-[var(--text-primary)] cursor-pointer"
                            >
                              +{chunks.length - 6} more
                            </button>
                          )}
                        </div>

                        <div className="flex shrink-0 items-center gap-1.5">
                          {onSubmitFeedback && turn.result.trace_id && (
                            <div
                              className="inline-flex items-center gap-0.5 rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60 p-0.5"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <button
                                type="button"
                                aria-label="Mark answer helpful"
                                title="Helpful (+1 score)"
                                disabled={submittingFeedbackTurnId === turn.id}
                                onClick={() => void handleFeedbackSubmit(turn.id, 1)}
                                className={`inline-flex items-center justify-center rounded p-1 transition cursor-pointer disabled:opacity-50 ${
                                  turn.feedback?.score === 1
                                    ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400"
                                    : "text-[var(--text-muted)] hover:bg-[var(--bg-surface)] hover:text-[var(--text-primary)]"
                                }`}
                              >
                                <ThumbsUp className="h-3 w-3" />
                              </button>
                              <button
                                type="button"
                                aria-label="Mark answer unhelpful"
                                title="Needs improvement (0 score + optional note)"
                                disabled={submittingFeedbackTurnId === turn.id}
                                onClick={() => {
                                  setCommentingTurnId((curr) =>
                                    curr === turn.id ? null : turn.id
                                  );
                                  setPendingScore(0);
                                  setCommentDraft(turn.feedback?.comment ?? "");
                                }}
                                className={`inline-flex items-center justify-center rounded p-1 transition cursor-pointer disabled:opacity-50 ${
                                  turn.feedback?.score === 0 ||
                                  commentingTurnId === turn.id
                                    ? "bg-rose-500/15 text-rose-600 dark:text-rose-400"
                                    : "text-[var(--text-muted)] hover:bg-[var(--bg-surface)] hover:text-[var(--text-primary)]"
                                }`}
                              >
                                <ThumbsDown className="h-3 w-3" />
                              </button>
                            </div>
                          )}

                          {turn.feedback && (
                            <span
                              className="hidden items-center gap-1 rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60 px-1.5 py-0.5 font-mono text-[10px] text-[var(--text-muted)] sm:inline-flex"
                              title={
                                turn.feedback.comment
                                  ? `Feedback note: ${turn.feedback.comment}`
                                  : undefined
                              }
                            >
                              <Check className="h-2.5 w-2.5 text-emerald-500" />
                              <span>
                                {turn.feedback.persistedToLangfuse ? "langfuse" : "saved"}
                              </span>
                            </span>
                          )}

                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              onInspectTurn(idx, "trace");
                            }}
                            className={`inline-flex shrink-0 items-center gap-1 rounded border px-2 py-0.5 font-mono text-[10.5px] font-medium transition cursor-pointer ${
                              isSelected
                                ? "border-[var(--border-strong)] bg-[var(--bg-elevated)] text-[var(--text-primary)]"
                                : "border-transparent text-[var(--text-muted)] hover:border-[var(--border-subtle)] hover:text-[var(--text-primary)]"
                            }`}
                          >
                            <PanelRightOpen className="h-3 w-3" />
                            <span>{isSelected ? "Inspecting" : `Inspect #${idx + 1}`}</span>
                          </button>
                        </div>
                      </div>

                      {commentingTurnId === turn.id && (
                        <div
                          onClick={(e) => e.stopPropagation()}
                          className="mt-2.5 flex flex-wrap items-center gap-2 rounded-md border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60 p-2"
                        >
                          <input
                            type="text"
                            value={commentDraft}
                            onChange={(e) => setCommentDraft(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") {
                                e.preventDefault();
                                void handleFeedbackSubmit(
                                  turn.id,
                                  pendingScore,
                                  commentDraft.trim() || undefined
                                );
                              }
                            }}
                            placeholder="Optional note: missing citation, stale source, wrong route..."
                            className="min-w-[200px] flex-1 rounded border border-[var(--border-subtle)] bg-[var(--bg-surface)] px-2.5 py-1 text-xs text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:border-[var(--text-muted)] focus:outline-none"
                          />
                          <button
                            type="button"
                            disabled={submittingFeedbackTurnId === turn.id}
                            onClick={() =>
                              void handleFeedbackSubmit(
                                turn.id,
                                pendingScore,
                                commentDraft.trim() || undefined
                              )
                            }
                            className="rounded bg-[var(--cta-bg)] px-2.5 py-1 text-[11px] font-medium text-[var(--cta-fg)] transition hover:bg-[var(--cta-hover)] disabled:opacity-50 cursor-pointer"
                          >
                            {submittingFeedbackTurnId === turn.id
                              ? "Saving..."
                              : "Send Feedback"}
                          </button>
                          <button
                            type="button"
                            onClick={() => {
                              setCommentingTurnId(null);
                              setCommentDraft("");
                            }}
                            className="rounded px-2 py-1 text-[11px] text-[var(--text-muted)] hover:text-[var(--text-primary)] cursor-pointer"
                          >
                            Cancel
                          </button>
                        </div>
                      )}

                      {feedbackErrorByTurn[turn.id] && (
                        <p className="mt-2 text-[11px] text-rose-500">
                          Feedback error: {feedbackErrorByTurn[turn.id]}
                        </p>
                      )}
                    </div>
                  </article>
                );
              })}

              {/* Active Streaming Turn */}
              {isRunning && pendingQuery && (
                <article className="space-y-3">
                  <div className="flex items-baseline gap-2.5">
                    <span className="shrink-0 rounded border border-[var(--border-subtle)] bg-[var(--bg-surface)] px-1.5 py-0.5 font-mono text-[10px] font-medium text-[var(--text-muted)] tabular-nums">
                      #{String(turns.length + 1).padStart(2, "0")}
                    </span>
                    <h2 className="text-[15px] font-medium leading-snug tracking-tight text-[var(--text-primary)]">
                      {pendingQuery}
                    </h2>
                  </div>
                  <div className="rounded-lg border border-[var(--border-strong)] bg-[var(--bg-surface)] p-4 sm:p-5">
                    <LivePipelineTracker
                      mode="streaming"
                      steps={liveSteps}
                      liveUsage={liveUsage}
                    />
                    <div className="flex items-center gap-2 font-mono text-[11px] text-[var(--text-muted)]">
                      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[var(--text-primary)]" />
                      <span>
                        {useStreaming
                          ? "Streaming LangGraph state events over SSE..."
                          : "Executing AdaptiveRAG batch pipeline..."}
                      </span>
                    </div>
                  </div>
                </article>
              )}

              {/* Stream Error Banner */}
              {streamError && !isRunning && (
                <div className="flex items-start gap-2.5 rounded-lg border border-rose-500/35 bg-rose-500/10 px-3.5 py-2.5 text-xs text-rose-600 dark:text-rose-300">
                  <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  <div>
                    <div className="font-medium">Execution Failed</div>
                    <p className="mt-0.5 opacity-90">{streamError}</p>
                  </div>
                </div>
              )}

              <div ref={scrollEndRef} />
            </div>
          )}
        </div>
      </div>

      {/* Bottom Sticky Composer */}
      <div className="shrink-0 border-t border-[var(--border-subtle)] bg-[var(--bg-panel)]/95 px-4 py-3 backdrop-blur sm:px-8">
        <div className="mx-auto max-w-3xl">
          {/* Optional Collapsible Thread Context Drawer (Layer-2 Style + Document Scope) */}
          {onThreadContextChange && contextPanelOpen && (
            <div
              data-testid="thread-context-panel"
              className="mb-2.5 rounded-lg border border-[var(--border-strong)] bg-[var(--bg-surface)] p-3 text-xs"
            >
              <div className="flex items-center justify-between gap-2 border-b border-[var(--border-subtle)] pb-2">
                <div className="flex items-center gap-1.5">
                  <SlidersHorizontal className="h-3.5 w-3.5 text-[var(--text-muted)]" />
                  <span className="font-medium text-[var(--text-primary)]">
                    Thread Context
                  </span>
                  <span className="font-mono text-[10px] text-[var(--text-muted)]">
                    optional · per-thread
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  {hasActiveThreadContext && (
                    <button
                      type="button"
                      onClick={handleResetThreadContext}
                      className="inline-flex items-center gap-1 rounded border border-[var(--border-subtle)] px-1.5 py-0.5 font-mono text-[10px] text-[var(--text-muted)] transition hover:border-[var(--border-strong)] hover:text-[var(--text-primary)] cursor-pointer"
                    >
                      <RotateCcw className="h-2.5 w-2.5" />
                      <span>Reset</span>
                    </button>
                  )}
                  <button
                    type="button"
                    aria-label="Close thread context panel"
                    onClick={() => setContextPanelOpen(false)}
                    className="rounded p-0.5 text-[var(--text-muted)] transition hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)] cursor-pointer"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>

              {/* Layer-2 Response Style Presets & Custom Instructions */}
              <div className="mt-2.5">
                <div className="mb-1.5 flex items-center justify-between gap-2">
                  <span className="font-mono text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
                    Response Style
                  </span>
                  <span className="inline-flex items-center gap-1 text-[10.5px] text-[var(--text-muted)]">
                    <Lock className="h-2.5 w-2.5 shrink-0" />
                    <span>Grounding &amp; [C#] citations stay locked</span>
                  </span>
                </div>

                <div className="flex flex-wrap gap-1.5">
                  {INSTRUCTION_PRESETS.map((opt) => {
                    const selected = activeContext.instructionPreset === opt.id;
                    return (
                      <button
                        key={opt.id}
                        type="button"
                        title={opt.description}
                        onClick={() => handleSelectInstructionPreset(opt.id)}
                        className={`rounded-md border px-2 py-1 text-[11px] font-medium transition cursor-pointer ${
                          selected
                            ? "border-emerald-500/45 bg-emerald-500/12 text-emerald-700 dark:text-emerald-300"
                            : "border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60 text-[var(--text-secondary)] hover:border-[var(--border-strong)] hover:text-[var(--text-primary)]"
                        }`}
                      >
                        {opt.label}
                      </button>
                    );
                  })}
                </div>

                {activeContext.instructionPreset !== "default" && (
                  <div className="mt-2">
                    <textarea
                      aria-label="Thread custom instructions"
                      rows={2}
                      maxLength={MAX_CUSTOM_INSTRUCTIONS_CHARS}
                      value={activeContext.customInstructions}
                      onChange={(e) => handleCustomInstructionsInput(e.target.value)}
                      placeholder="e.g. Summarize obligations and risks in a concise table using plain language..."
                      className="w-full resize-none rounded-md border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/65 px-2.5 py-1.5 text-xs leading-relaxed text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:border-[var(--text-muted)] focus:outline-none"
                    />
                    <div className="mt-1 flex items-center justify-between font-mono text-[10px] text-[var(--text-muted)]">
                      <span>Layer-2 synthesis preference (screened by guardrails)</span>
                      <span className="tabular-nums">
                        {activeContext.customInstructions.length}/{MAX_CUSTOM_INSTRUCTIONS_CHARS}
                      </span>
                    </div>
                  </div>
                )}
              </div>

              {/* Per-Thread Document Scope (doc_ids) */}
              <div className="mt-3 border-t border-[var(--border-subtle)] pt-2.5">
                <div className="mb-1.5 flex items-center justify-between gap-2">
                  <span className="font-mono text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
                    Document Scope{" "}
                    {scopedDocIds.length > 0 ? `(${scopedDocIds.length} selected)` : "(All indexed docs)"}
                  </span>
                  {scopedDocIds.length > 0 && (
                    <button
                      type="button"
                      onClick={() =>
                        onThreadContextChange({ ...activeContext, scopedDocIds: [] })
                      }
                      className="font-mono text-[10px] text-[var(--text-muted)] hover:text-[var(--text-primary)] cursor-pointer"
                    >
                      Use all docs
                    </button>
                  )}
                </div>

                {availableDocuments.length === 0 ? (
                  <p className="text-[11px] text-[var(--text-muted)]">
                    No indexed documents loaded yet. All collection documents will be searched.
                  </p>
                ) : (
                  <div className="flex max-h-24 flex-wrap gap-1.5 overflow-y-auto pr-1">
                    {availableDocuments.map((doc) => {
                      const isScoped = scopedDocIds.includes(doc.doc_id);
                      return (
                        <button
                          key={doc.doc_id}
                          type="button"
                          onClick={() => handleToggleScopedDoc(doc.doc_id)}
                          title={`${doc.title} (${doc.doc_id}) · ${doc.chunk_count} chunks`}
                          className={`inline-flex max-w-[220px] items-center gap-1.5 rounded-md border px-2 py-1 text-[11px] transition cursor-pointer ${
                            isScoped
                              ? "border-sky-500/45 bg-sky-500/12 text-sky-700 dark:text-sky-300"
                              : "border-[var(--border-subtle)] bg-[var(--bg-elevated)]/55 text-[var(--text-secondary)] hover:border-[var(--border-strong)] hover:text-[var(--text-primary)]"
                          }`}
                        >
                          {isScoped ? (
                            <Check className="h-3 w-3 shrink-0" />
                          ) : doc.source_origin === "web" ? (
                            <Globe className="h-3 w-3 shrink-0 opacity-65" />
                          ) : (
                            <FileText className="h-3 w-3 shrink-0 opacity-65" />
                          )}
                          <span className="truncate">{doc.title || doc.doc_id}</span>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          )}

          <div className="relative flex items-end gap-2 rounded-lg border border-[var(--border-strong)] bg-[var(--bg-surface)] px-3 py-2 transition focus-within:border-[var(--text-muted)]">
            <textarea
              ref={textareaRef}
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isRunning}
              placeholder="Ask a question about your documents in English or Turkish..."
              className={`max-h-40 flex-1 resize-none bg-transparent py-1 text-[13.5px] leading-5 text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none disabled:opacity-60 ${
                input.includes("\n") || input.length > 90
                  ? "overflow-y-auto"
                  : "overflow-hidden"
              }`}
            />
            {isRunning ? (
              <button
                type="button"
                onClick={onStopStream}
                title="Stop generation"
                className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-rose-600 text-white transition hover:bg-rose-500 cursor-pointer"
              >
                <Square className="h-3 w-3 fill-current" />
              </button>
            ) : (
              <button
                type="button"
                onClick={handleSend}
                disabled={!input.trim()}
                title="Submit query"
                className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[var(--cta-bg)] text-[var(--cta-fg)] transition hover:bg-[var(--cta-hover)] disabled:opacity-25 cursor-pointer"
              >
                <ArrowUp className="h-3.5 w-3.5" />
              </button>
            )}
          </div>

          <div className="mt-1.5 flex flex-wrap items-center justify-between gap-2 px-0.5 font-mono text-[10px] text-[var(--text-muted)]">
            <div className="flex flex-wrap items-center gap-1.5">
              <span>preset:{presetShortLabel.toLowerCase().replace(/\s+/g, "-")}</span>
              <span>·</span>
              <span>{useStreaming ? "sse" : "batch"}</span>
              {budgetCapLabel && (
                <>
                  <span>·</span>
                  <span>cap:{budgetCapLabel}</span>
                </>
              )}
              {onThreadContextChange && (
                <>
                  <span>·</span>
                  <button
                    type="button"
                    onClick={() => setContextPanelOpen((prev) => !prev)}
                    aria-expanded={contextPanelOpen}
                    title="Configure optional thread response style and document scope"
                    className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 transition cursor-pointer ${
                      effectiveInstructions
                        ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
                        : "border-[var(--border-subtle)] bg-[var(--bg-surface)] text-[var(--text-muted)] hover:border-[var(--border-strong)] hover:text-[var(--text-primary)]"
                    }`}
                  >
                    <SlidersHorizontal className="h-2.5 w-2.5" />
                    <span>style:{styleLabel}</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setContextPanelOpen((prev) => !prev)}
                    aria-expanded={contextPanelOpen}
                    title="Scope retrieval to specific indexed documents for this thread"
                    className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 transition cursor-pointer ${
                      scopedDocIds.length > 0
                        ? "border-sky-500/40 bg-sky-500/10 text-sky-700 dark:text-sky-300"
                        : "border-[var(--border-subtle)] bg-[var(--bg-surface)] text-[var(--text-muted)] hover:border-[var(--border-strong)] hover:text-[var(--text-primary)]"
                    }`}
                  >
                    <FileText className="h-2.5 w-2.5" />
                    <span>
                      sources:
                      {scopedDocIds.length > 0
                        ? `${scopedDocIds.length} doc${scopedDocIds.length > 1 ? "s" : ""}`
                        : "all"}
                    </span>
                  </button>
                </>
              )}
            </div>
            <div className="hidden items-center gap-1.5 sm:flex">
              <span>Submit</span>
              <kbd className="ar-kbd">↵</kbd>
              <span className="ml-1">Newline</span>
              <kbd className="ar-kbd">⇧↵</kbd>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
