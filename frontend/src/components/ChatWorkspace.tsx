import React, { useEffect, useRef, useState } from "react";
import {
  Check,
  Copy,
  FileText,
  Files,
  Globe,
  Lock,
  PanelRightOpen,
  PenLine,
  RotateCcw,
  ThumbsDown,
  ThumbsUp,
  X,
} from "lucide-react";
import {
  DEFAULT_THREAD_CONTEXT,
  DEMO_QUERIES,
  INSTRUCTION_PRESETS,
  MAX_CUSTOM_INSTRUCTIONS_CHARS,
  RUN_STOPPED_MESSAGE,
  resolveEffectiveInstructions,
} from "../constants/pipeline";
import { formatInt, languageName } from "../lib/format";
import type { ThreadSaveProblem } from "../lib/threadStore";
import type {
  AblationConfig,
  ChatTurn,
  EvidenceChunk,
  IndexedDocumentInfo,
  InspectorTab,
  InstructionPresetId,
  LiveStreamStep,
  PresetSelection,
  ThreadContextConfig,
  TokenUsage,
} from "../types/api";
import { AnswerTelemetryBar } from "./AnswerTelemetryBar";
import { Composer, ComposerChip } from "./Composer";
import { LivePipelineTracker } from "./LivePipelineTracker";
import { MarkdownWithCitations } from "./MarkdownWithCitations";
import { PresetPicker } from "./PresetPicker";
import { CitationMark, isWebCitation } from "./ui/CitationMark";
import { Button, Callout, cn, IconButton, INPUT_CLASS, SectionHeading } from "./ui/primitives";

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
  onSubmitFeedback?: (turnId: string, score: 0 | 1, comment?: string) => Promise<void>;
  preset?: PresetSelection;
  onPresetChange?: (preset: PresetSelection) => void;
  customAblation?: AblationConfig;
  onCustomAblationChange?: (cfg: AblationConfig) => void;
  onOpenKnowledgeBase?: () => void;
  /** Set while threads can't be saved in this browser. */
  threadSaveProblem?: ThreadSaveProblem | null;
}

/* -------------------------------------------------------------------------- */
/* Unsaved threads                                                            */
/* -------------------------------------------------------------------------- */

function ThreadSaveNotice({ problem }: { problem: ThreadSaveProblem }) {
  return (
    <Callout live tone="warn" title="Threads aren't being saved" className="mb-2">
      {problem.kind === "full"
        ? "Browser storage for this site is full. Delete threads you no longer need; until a save goes through, reloading loses new answers."
        : problem.kind === "unavailable"
          ? `This browser won't let the Studio store data (${problem.detail}), so reloading loses the answers in this tab.`
          : `The last save failed (${problem.detail}). The Studio tries again with your next change; until then, reloading loses new answers.`}
    </Callout>
  );
}

/* -------------------------------------------------------------------------- */
/* Sources row                                                                */
/* -------------------------------------------------------------------------- */

function sourceLabel(chunk: EvidenceChunk): string {
  const section = chunk.section_path?.split(/\s*[>›/]\s*/).filter(Boolean).pop();
  return section || chunk.title || chunk.doc_id || "Source";
}

export function SourcesRow({
  chunks,
  limit,
  onCitationClick,
  onShowAll,
}: {
  chunks: EvidenceChunk[];
  limit: number;
  onCitationClick: (citationId: string) => void;
  onShowAll: () => void;
}) {
  if (chunks.length === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="mr-1 text-small text-ink-3">Sources</span>
      {chunks.slice(0, limit).map((ch, cIdx) => {
        const cid = ch.citation_id || `#${cIdx + 1}`;
        const web = isWebCitation(cid, ch.source_type);
        return (
          <button
            key={ch.chunk_id || cIdx}
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onCitationClick(cid);
            }}
            title={`${ch.title || ch.doc_id}${ch.section_path ? ` › ${ch.section_path}` : ""}${ch.url ? `\n${ch.url}` : ""}`}
            className="inline-flex h-7 max-w-[240px] cursor-pointer items-center gap-1.5 rounded-md border border-line bg-paper pr-2 pl-1 text-small text-ink-2 transition-colors hover:border-line-strong hover:text-ink"
          >
            <CitationMark citationId={cid} web={web} static />
            <span className="truncate">{sourceLabel(ch)}</span>
          </button>
        );
      })}
      {chunks.length > limit && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onShowAll();
          }}
          className="h-7 cursor-pointer rounded-md px-2 text-small font-medium text-ink-2 transition-colors hover:bg-hover hover:text-ink"
        >
          {chunks.length - limit} more
        </button>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Workbench                                                                  */
/* -------------------------------------------------------------------------- */

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
  preset,
  onPresetChange,
  customAblation,
  onCustomAblationChange,
  onOpenKnowledgeBase,
  threadSaveProblem = null,
}) => {
  const [input, setInput] = useState("");
  const [contextPanelOpen, setContextPanelOpen] = useState(false);
  const [commentingTurnId, setCommentingTurnId] = useState<string | null>(null);
  const [commentDraft, setCommentDraft] = useState("");
  const [submittingFeedbackTurnId, setSubmittingFeedbackTurnId] = useState<string | null>(null);
  const [feedbackErrorByTurn, setFeedbackErrorByTurn] = useState<Record<string, string>>({});
  const [copiedTurnId, setCopiedTurnId] = useState<string | null>(null);
  const scrollEndRef = useRef<HTMLDivElement>(null);

  const handleFeedbackSubmit = async (turnId: string, score: 0 | 1, comment?: string) => {
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

  const handleCopy = (turn: ChatTurn) => {
    void navigator.clipboard?.writeText(turn.result.answer);
    setCopiedTurnId(turn.id);
    window.setTimeout(() => setCopiedTurnId((curr) => (curr === turn.id ? null : curr)), 1600);
  };

  useEffect(() => {
    scrollEndRef.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [turns.length, isRunning, liveSteps.length]);

  const handleSend = () => {
    const cleaned = input.trim();
    if (!cleaned || isRunning) return;
    setInput("");
    onSubmitQuery(cleaned);
  };

  const activeContext = threadContext ?? DEFAULT_THREAD_CONTEXT;
  const effectiveInstructions = resolveEffectiveInstructions(activeContext);
  const scopedDocIds = activeContext.scopedDocIds;
  const hasActiveThreadContext = Boolean(effectiveInstructions) || scopedDocIds.length > 0;
  const stylePreset = INSTRUCTION_PRESETS.find((p) => p.id === activeContext.instructionPreset);
  const styleChipLabel = !effectiveInstructions
    ? "Default style"
    : activeContext.instructionPreset === "custom"
      ? "Custom style"
      : (stylePreset?.label ?? "Custom style");
  const scopeChipLabel =
    scopedDocIds.length === 0
      ? "All documents"
      : `${scopedDocIds.length} ${scopedDocIds.length === 1 ? "document" : "documents"}`;

  const handleSelectInstructionPreset = (presetId: InstructionPresetId) => {
    if (!onThreadContextChange) return;
    const matched = INSTRUCTION_PRESETS.find((p) => p.id === presetId);
    if (presetId === "default") {
      onThreadContextChange({ ...activeContext, instructionPreset: "default", customInstructions: "" });
      return;
    }
    if (presetId === "custom") {
      onThreadContextChange({ ...activeContext, instructionPreset: "custom" });
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
    onThreadContextChange({
      ...activeContext,
      scopedDocIds: exists ? scopedDocIds.filter((id) => id !== docId) : [...scopedDocIds, docId],
    });
  };

  const hasContent = turns.length > 0 || isRunning;
  const isStopNotice = streamError === RUN_STOPPED_MESSAGE;

  return (
    <div className="flex min-w-0 flex-1 flex-col bg-paper">
      {/* Conversation */}
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-[47rem] px-4 sm:px-8">
          {!hasContent ? (
            <EmptyWorkbench
              documents={availableDocuments}
              presetShortLabel={presetShortLabel}
              onSubmitQuery={onSubmitQuery}
              onOpenKnowledgeBase={onOpenKnowledgeBase}
            />
          ) : (
            <div className="space-y-12 pt-8 pb-10">
              {turns.map((turn, idx) => {
                const isSelected = idx === selectedTurnIndex;
                const chunks = turn.result.evidence_chunks || [];
                const isCommenting = commentingTurnId === turn.id;
                const isSaving = submittingFeedbackTurnId === turn.id;
                return (
                  <article
                    key={turn.id}
                    aria-label={`Question ${idx + 1}`}
                    onClick={() => onInspectTurn(idx)}
                    className="grid grid-cols-[1.5rem_minmax(0,1fr)] gap-x-2 sm:grid-cols-[2.25rem_minmax(0,1fr)] sm:gap-x-3"
                  >
                    <span
                      aria-hidden
                      className={cn(
                        "pt-0.5 text-right text-ui tabular-nums",
                        isSelected ? "font-semibold text-accent" : "text-ink-3"
                      )}
                    >
                      {idx + 1}
                    </span>
                    <div className="min-w-0">
                      <h2 className="text-title font-semibold text-balance text-ink">{turn.query}</h2>

                      <div className="mt-3 rounded-lg border border-line bg-sunken px-3 py-2.5">
                        <AnswerTelemetryBar
                          result={turn.result}
                          onBadgeClick={(tab) => onInspectTurn(idx, tab)}
                        />
                        {(turn.result.trace?.length ?? 0) > 0 && (
                          <div className="mt-2">
                            <LivePipelineTracker
                              mode="completed"
                              steps={turn.result.trace || []}
                              totalLatencyMs={turn.result.total_latency_ms}
                              onOpenInspector={() => onInspectTurn(idx, "trace")}
                            />
                          </div>
                        )}
                      </div>

                      <div className="mt-5">
                        <MarkdownWithCitations
                          content={turn.result.answer}
                          evidenceChunks={chunks}
                          onCitationClick={(cid) => onCitationClick(idx, cid)}
                        />
                      </div>

                      <div className="mt-5">
                        <SourcesRow
                          chunks={chunks}
                          limit={6}
                          onCitationClick={(cid) => onCitationClick(idx, cid)}
                          onShowAll={() => onInspectTurn(idx, "evidence")}
                        />
                      </div>

                      <div
                        className="mt-3 flex flex-wrap items-center gap-1"
                        onClick={(e) => e.stopPropagation()}
                      >
                        {onSubmitFeedback && turn.result.trace_id && (
                          <>
                            <IconButton
                              icon={ThumbsUp}
                              label="Mark answer helpful"
                              title="Helpful"
                              disabled={isSaving}
                              pressed={turn.feedback?.score === 1}
                              onClick={() => void handleFeedbackSubmit(turn.id, 1)}
                              className={turn.feedback?.score === 1 ? "!text-ok" : undefined}
                            />
                            <IconButton
                              icon={ThumbsDown}
                              label="Mark answer unhelpful"
                              title="Needs work: add an optional note"
                              disabled={isSaving}
                              pressed={turn.feedback?.score === 0 || isCommenting}
                              onClick={() => {
                                setCommentingTurnId((curr) => (curr === turn.id ? null : turn.id));
                                setCommentDraft(turn.feedback?.comment ?? "");
                              }}
                              className={
                                turn.feedback?.score === 0 || isCommenting ? "!text-bad" : undefined
                              }
                            />
                          </>
                        )}
                        <IconButton
                          icon={copiedTurnId === turn.id ? Check : Copy}
                          label={copiedTurnId === turn.id ? "Answer copied" : "Copy answer"}
                          onClick={() => handleCopy(turn)}
                        />
                        <Button
                          variant="ghost"
                          size="sm"
                          icon={PanelRightOpen}
                          onClick={() => onInspectTurn(idx, "trace")}
                          className={cn(isSelected && "text-ink")}
                        >
                          {isSelected ? "Inspecting" : "Inspect"}
                        </Button>
                        {turn.feedback && (
                          <span
                            className="ml-1 inline-flex items-center gap-1 text-small text-ink-3"
                            title={turn.feedback.comment ? `Note: ${turn.feedback.comment}` : undefined}
                          >
                            <Check className="h-3.5 w-3.5 text-ok" aria-hidden />
                            {turn.feedback.persistedToLangfuse
                              ? "Feedback saved to Langfuse"
                              : "Feedback saved"}
                          </span>
                        )}
                      </div>

                      {isCommenting && (
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
                                void handleFeedbackSubmit(turn.id, 0, commentDraft.trim() || undefined);
                              }
                              if (e.key === "Escape") setCommentingTurnId(null);
                            }}
                            placeholder="Optional note: missing citation, outdated source, wrong route…"
                            className={cn(INPUT_CLASS, "h-8 min-w-[200px] flex-1 px-2.5")}
                          />
                          <Button
                            variant="primary"
                            size="sm"
                            loading={isSaving}
                            onClick={() =>
                              void handleFeedbackSubmit(turn.id, 0, commentDraft.trim() || undefined)
                            }
                          >
                            Send feedback
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => {
                              setCommentingTurnId(null);
                              setCommentDraft("");
                            }}
                          >
                            Cancel
                          </Button>
                        </div>
                      )}

                      {feedbackErrorByTurn[turn.id] && (
                        <p className="mt-2 text-small text-bad">
                          Feedback wasn't saved: {feedbackErrorByTurn[turn.id]}
                        </p>
                      )}
                    </div>
                  </article>
                );
              })}

              {/* Running question */}
              {isRunning && pendingQuery && (
                <article className="grid grid-cols-[1.5rem_minmax(0,1fr)] gap-x-2 sm:grid-cols-[2.25rem_minmax(0,1fr)] sm:gap-x-3">
                  <span aria-hidden className="pt-0.5 text-right text-ui tabular-nums text-ink-3">
                    {turns.length + 1}
                  </span>
                  <div className="min-w-0">
                    <h2 className="text-title font-semibold text-balance text-ink">{pendingQuery}</h2>
                    <div className="mt-3">
                      <LivePipelineTracker
                        mode="streaming"
                        steps={liveSteps}
                        liveUsage={liveUsage}
                        runningLabel={
                          useStreaming
                            ? `Running ${presetShortLabel}`
                            : `Running ${presetShortLabel} (batch mode, steps arrive at the end)`
                        }
                      />
                    </div>
                  </div>
                </article>
              )}

              {streamError && !isRunning && (
                <div className="sm:pl-[3rem]">
                  <Callout
                    live
                    tone={isStopNotice ? "info" : "bad"}
                    title={isStopNotice ? "Run stopped" : "The run failed"}
                  >
                    {isStopNotice
                      ? "No answer was saved. Ask again when you're ready."
                      : streamError}
                  </Callout>
                </div>
              )}

              <div ref={scrollEndRef} />
            </div>
          )}
        </div>
      </div>

      {/* Composer */}
      <div className="relative shrink-0 bg-paper px-4 pb-4 sm:px-8">
        <div
          aria-hidden
          className="pointer-events-none absolute inset-x-0 -top-8 h-8 bg-gradient-to-b from-transparent to-paper"
        />
        <div className="mx-auto w-full max-w-[47rem] sm:pl-[3rem]">
          {threadSaveProblem && <ThreadSaveNotice problem={threadSaveProblem} />}

          {onThreadContextChange && contextPanelOpen && (
            <ThreadContextPanel
              context={activeContext}
              documents={availableDocuments}
              hasActiveContext={hasActiveThreadContext}
              onClose={() => setContextPanelOpen(false)}
              onReset={() => onThreadContextChange({ ...DEFAULT_THREAD_CONTEXT })}
              onSelectPreset={handleSelectInstructionPreset}
              onInstructionsChange={handleCustomInstructionsInput}
              onToggleDoc={handleToggleScopedDoc}
              onUseAllDocs={() => onThreadContextChange({ ...activeContext, scopedDocIds: [] })}
            />
          )}

          <Composer
            value={input}
            onChange={setInput}
            onSubmit={handleSend}
            onStop={onStopStream}
            isRunning={isRunning}
            placeholder="Ask a question about your documents"
            submitLabel="Send question"
            stopLabel="Stop the run"
            toolbar={
              <>
                {preset && onPresetChange && customAblation && onCustomAblationChange ? (
                  <PresetPicker
                    preset={preset}
                    onPresetChange={onPresetChange}
                    customAblation={customAblation}
                    onCustomAblationChange={onCustomAblationChange}
                    disabled={isRunning}
                  />
                ) : (
                  <span className="px-2 text-small text-ink-3">{presetShortLabel}</span>
                )}
                {onThreadContextChange && (
                  <>
                    <ComposerChip
                      icon={PenLine}
                      active={contextPanelOpen || Boolean(effectiveInstructions)}
                      aria-expanded={contextPanelOpen}
                      aria-label={`Response style: ${styleChipLabel}`}
                      onClick={() => setContextPanelOpen((prev) => !prev)}
                    >
                      <span className="truncate">{styleChipLabel}</span>
                    </ComposerChip>
                    <ComposerChip
                      icon={Files}
                      active={contextPanelOpen || scopedDocIds.length > 0}
                      aria-expanded={contextPanelOpen}
                      aria-label={`Document scope: ${scopeChipLabel}`}
                      onClick={() => setContextPanelOpen((prev) => !prev)}
                    >
                      <span className="truncate">{scopeChipLabel}</span>
                    </ComposerChip>
                  </>
                )}
              </>
            }
            footer={
              <>
                {!useStreaming && <span>Batch mode: steps appear when the run ends</span>}
                {budgetCapLabel && <span>Spend cap {budgetCapLabel} per question</span>}
              </>
            }
          />
        </div>
      </div>
    </div>
  );
};

/* -------------------------------------------------------------------------- */
/* Empty state                                                                */
/* -------------------------------------------------------------------------- */

function EmptyWorkbench({
  documents,
  presetShortLabel,
  onSubmitQuery,
  onOpenKnowledgeBase,
}: {
  documents: IndexedDocumentInfo[];
  presetShortLabel: string;
  onSubmitQuery: (query: string) => void;
  onOpenKnowledgeBase?: () => void;
}) {
  const visibleDocs = documents.slice(0, 4);
  return (
    <div className="pt-[9vh] pb-12 sm:pl-[3rem]">
      <h1 className="text-display font-semibold tracking-[-0.015em] text-ink">Ask your documents</h1>
      <p className="mt-3 max-w-[34rem] text-body text-ink-2">
        Every answer cites the passages it relies on. Hover a mark like{" "}
        <CitationMark citationId="C1" static className="mx-0.5 align-[1px]" /> to read the
        source, or open the inspector to see how the answer was routed, retrieved and checked.
      </p>

      <section className="mt-10" aria-labelledby="scope-heading">
        <SectionHeading
          aside={
            onOpenKnowledgeBase && (
              <Button variant="ghost" size="xs" onClick={onOpenKnowledgeBase}>
                Manage documents
              </Button>
            )
          }
        >
          <span id="scope-heading">
            {documents.length === 0
              ? "No documents indexed yet"
              : `Searching ${documents.length} ${documents.length === 1 ? "document" : "documents"}`}
          </span>
        </SectionHeading>
        {documents.length === 0 ? (
          <p className="mt-2 text-small text-ink-3">
            Add files or web pages in the knowledge base. Until then, questions can only be answered
            from the web fallback, if it's enabled.
          </p>
        ) : (
          <ul className="mt-2.5 divide-y divide-line rounded-lg border border-line">
            {visibleDocs.map((doc) => (
              <li key={doc.doc_id} className="flex items-center gap-3 px-3.5 py-2.5">
                {doc.source_origin === "web" ? (
                  <Globe className="h-4 w-4 shrink-0 text-ink-3" aria-hidden />
                ) : (
                  <FileText className="h-4 w-4 shrink-0 text-ink-3" aria-hidden />
                )}
                <span className="min-w-0 flex-1 truncate text-ui font-medium text-ink">
                  {doc.title || doc.doc_id}
                </span>
                <span className="hidden shrink-0 gap-3 text-small text-ink-3 sm:flex">
                  <span>{doc.source_origin === "web" ? "Web page" : doc.doc_type.toUpperCase()}</span>
                  <span>{languageName(doc.language)}</span>
                  <span className="tabular-nums">{formatInt(doc.chunk_count)} passages</span>
                </span>
              </li>
            ))}
            {documents.length > visibleDocs.length && (
              <li className="px-3.5 py-2 text-small text-ink-3">
                {documents.length - visibleDocs.length} more in the knowledge base
              </li>
            )}
          </ul>
        )}
      </section>

      <section className="mt-10" aria-labelledby="samples-heading">
        <SectionHeading
          aside={<span className="text-small text-ink-3">Runs with {presetShortLabel}</span>}
        >
          <span id="samples-heading">Sample questions</span>
        </SectionHeading>
        <ul className="mt-2.5 divide-y divide-line rounded-lg border border-line">
          {DEMO_QUERIES.map((demo) => (
            <li key={demo.id}>
              <button
                type="button"
                onClick={() => onSubmitQuery(demo.query)}
                className="grid w-full cursor-pointer grid-cols-[6.5rem_minmax(0,1fr)] items-baseline gap-x-3 px-3.5 py-3 text-left transition-colors hover:bg-hover"
              >
                <span className="text-small text-ink-3">{demo.category}</span>
                <span className="min-w-0">
                  <span className="flex items-baseline justify-between gap-3">
                    <span className="truncate text-ui font-medium text-ink">{demo.title}</span>
                    <span className="shrink-0 text-caption text-ink-3">{demo.lang}</span>
                  </span>
                  <span className="mt-0.5 line-clamp-1 text-small text-ink-2">{demo.query}</span>
                </span>
              </button>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Thread settings (response style + document scope)                          */
/* -------------------------------------------------------------------------- */

function ThreadContextPanel({
  context,
  documents,
  hasActiveContext,
  onClose,
  onReset,
  onSelectPreset,
  onInstructionsChange,
  onToggleDoc,
  onUseAllDocs,
}: {
  context: ThreadContextConfig;
  documents: IndexedDocumentInfo[];
  hasActiveContext: boolean;
  onClose: () => void;
  onReset: () => void;
  onSelectPreset: (id: InstructionPresetId) => void;
  onInstructionsChange: (value: string) => void;
  onToggleDoc: (docId: string) => void;
  onUseAllDocs: () => void;
}) {
  const scoped = context.scopedDocIds;
  return (
    <div
      data-testid="thread-context-panel"
      className="anim-pop mb-2.5 rounded-xl border border-line-strong bg-raised p-4 shadow-[var(--shadow-float)]"
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-ui font-semibold text-ink">Thread settings</div>
          <p className="mt-0.5 text-small text-ink-3">Apply to every question in this thread.</p>
        </div>
        <div className="flex items-center gap-1">
          {hasActiveContext && (
            <Button variant="ghost" size="xs" icon={RotateCcw} onClick={onReset}>
              Reset
            </Button>
          )}
          <IconButton icon={X} size="sm" label="Close thread settings" onClick={onClose} />
        </div>
      </div>

      <div className="mt-4">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <span className="text-small font-medium text-ink">Response style</span>
          <span className="inline-flex items-center gap-1 text-caption text-ink-3">
            <Lock className="h-3 w-3" aria-hidden />
            Citations and grounding rules always apply
          </span>
        </div>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {INSTRUCTION_PRESETS.map((opt) => {
            const selected = context.instructionPreset === opt.id;
            return (
              <button
                key={opt.id}
                type="button"
                title={opt.description}
                aria-pressed={selected}
                onClick={() => onSelectPreset(opt.id)}
                className={cn(
                  "h-7 cursor-pointer rounded-md border px-2.5 text-small font-medium transition-colors",
                  selected
                    ? "border-ink bg-ink text-paper"
                    : "border-line-strong bg-paper text-ink-2 hover:text-ink hover:tint"
                )}
              >
                {opt.label}
              </button>
            );
          })}
        </div>
        {context.instructionPreset !== "default" && (
          <div className="mt-2.5">
            <textarea
              aria-label="Thread custom instructions"
              rows={2}
              maxLength={MAX_CUSTOM_INSTRUCTIONS_CHARS}
              value={context.customInstructions}
              onChange={(e) => onInstructionsChange(e.target.value)}
              placeholder="For example: summarize obligations and risks in a short table, in plain language."
              className={cn(INPUT_CLASS, "resize-none px-3 py-2 leading-relaxed")}
            />
            <div className="mt-1 flex items-center justify-between gap-2 text-caption text-ink-3">
              <span>Guardrails screen these instructions before they reach the model.</span>
              <span className="tabular-nums">
                {formatInt(context.customInstructions.length)} / {formatInt(MAX_CUSTOM_INSTRUCTIONS_CHARS)}
              </span>
            </div>
          </div>
        )}
      </div>

      <div className="mt-4 border-t border-line pt-3.5">
        <div className="flex items-baseline justify-between gap-2">
          <span className="text-small font-medium text-ink">
            Documents{" "}
            <span className="font-normal text-ink-3">
              {scoped.length > 0 ? `${scoped.length} selected` : "All indexed documents"}
            </span>
          </span>
          {scoped.length > 0 && (
            <Button variant="ghost" size="xs" onClick={onUseAllDocs}>
              Use all documents
            </Button>
          )}
        </div>
        {documents.length === 0 ? (
          <p className="mt-2 text-small text-ink-3">
            No indexed documents yet, so the whole collection is searched.
          </p>
        ) : (
          <div className="mt-2 flex max-h-28 flex-wrap gap-1.5 overflow-y-auto pr-1">
            {documents.map((doc) => {
              const isScoped = scoped.includes(doc.doc_id);
              return (
                <button
                  key={doc.doc_id}
                  type="button"
                  aria-pressed={isScoped}
                  onClick={() => onToggleDoc(doc.doc_id)}
                  title={`${doc.title} (${doc.doc_id}), ${doc.chunk_count} passages`}
                  className={cn(
                    "inline-flex h-7 max-w-[240px] cursor-pointer items-center gap-1.5 rounded-md border px-2 text-small transition-colors",
                    isScoped
                      ? "border-accent bg-accent-soft text-ink"
                      : "border-line-strong bg-paper text-ink-2 hover:text-ink hover:tint"
                  )}
                >
                  {isScoped ? (
                    <Check className="h-3.5 w-3.5 shrink-0 text-accent" aria-hidden />
                  ) : doc.source_origin === "web" ? (
                    <Globe className="h-3.5 w-3.5 shrink-0 text-ink-3" aria-hidden />
                  ) : (
                    <FileText className="h-3.5 w-3.5 shrink-0 text-ink-3" aria-hidden />
                  )}
                  <span className="truncate">{doc.title || doc.doc_id}</span>
                </button>
              );
            })}
          </div>
        )}
      </div>
      <div className="sr-only" aria-live="polite">
        {scoped.length > 0 ? `${scoped.length} documents in scope` : "All documents in scope"}
      </div>
    </div>
  );
}
