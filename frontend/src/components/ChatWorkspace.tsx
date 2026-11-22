import React, { useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  ArrowUp,
  ExternalLink,
  FileText,
  Globe,
  Microscope,
  Sparkles,
  Square,
} from "lucide-react";
import { DEMO_QUERIES } from "../constants/pipeline";
import type {
  ChatTurn,
  InspectorTab,
  LiveStreamStep,
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
  onSubmitQuery: (query: string) => void;
  onStopStream: () => void;
  onInspectTurn: (turnIndex: number, tab?: InspectorTab) => void;
  onCitationClick: (turnIndex: number, citationId: string) => void;
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
  onSubmitQuery,
  onStopStream,
  onInspectTurn,
  onCitationClick,
}) => {
  const [input, setInput] = useState("");
  const scrollEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll smoothly when new turns or live steps arrive
  useEffect(() => {
    scrollEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns.length, isRunning, liveSteps.length]);

  // Auto-resize composer textarea
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

  const hasContent = turns.length > 0 || isRunning;

  return (
    <div className="flex min-w-0 flex-1 flex-col bg-[var(--bg-canvas)]">
      {/* Scrollable Conversation Area */}
      <div className="flex-1 overflow-y-auto px-4 py-6 sm:px-7">
        <div className="mx-auto max-w-3xl">
          {!hasContent ? (
            /* Empty Welcome Canvas */
            <div className="py-8 sm:py-12">
              <div className="inline-flex items-center gap-2 rounded-full border border-blue-500/30 bg-blue-500/10 px-3 py-1 text-[11px] font-medium text-blue-600 dark:text-blue-300">
                <Sparkles className="h-3.5 w-3.5" />
                <span>
                  Query-Aware Agentic RAG · Hybrid RRF · Dynamic Top-K · CRAG · Citation Audit
                </span>
              </div>

              <h1 className="mt-4 text-2xl font-bold tracking-tight text-[var(--text-primary)] sm:text-3xl">
                Inspect every retrieval & grounding decision in real time.
              </h1>
              <p className="mt-2 max-w-2xl text-sm leading-relaxed text-[var(--text-secondary)]">
                Ask technical questions in <strong>English</strong> or <strong>Turkish</strong>.
                Watch LangGraph nodes execute live over SSE, click inline{" "}
                <code className="rounded border border-blue-500/30 bg-blue-500/10 px-1.5 py-0.5 font-mono text-xs text-blue-500 dark:text-blue-300">
                  [C1]
                </code>{" "}
                citations to highlight exact evidence chunks, and audit every claim.
              </p>

              {/* Quick Demo Query Cards */}
              <div className="mt-7">
                <div className="mb-2.5 text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                  Quick Bilingual Demo Queries
                </div>
                <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
                  {DEMO_QUERIES.map((demo) => (
                    <button
                      key={demo.id}
                      type="button"
                      onClick={() => onSubmitQuery(demo.query)}
                      className="group flex flex-col justify-between rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-panel)] p-3.5 text-left transition hover:border-blue-500/45 hover:bg-[var(--bg-elevated)]/60 cursor-pointer"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-xs font-semibold text-[var(--text-primary)] group-hover:text-blue-500 dark:group-hover:text-blue-400">
                          {demo.title}
                        </span>
                        <div className="flex shrink-0 items-center gap-1">
                          <span className="rounded bg-[var(--bg-elevated)] px-1.5 py-0.5 font-mono text-[10px] font-medium text-[var(--text-secondary)]">
                            {demo.category}
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
            /* Conversation Thread */
            <div className="space-y-7 pb-4">
              {turns.map((turn, idx) => {
                const isSelected = idx === selectedTurnIndex;
                const chunks = turn.result.evidence_chunks || [];
                return (
                  <article key={turn.id} className="space-y-4">
                    {/* User Query Bubble */}
                    <div className="flex justify-end">
                      <div className="max-w-[85%] rounded-2xl rounded-tr-sm border border-blue-500/25 bg-blue-600/12 px-4 py-2.5 text-sm leading-relaxed text-[var(--text-primary)]">
                        {turn.query}
                      </div>
                    </div>

                    {/* Assistant Response Card */}
                    <div
                      onClick={() => onInspectTurn(idx)}
                      className={`rounded-2xl border p-4 sm:p-5 transition-colors ${
                        isSelected
                          ? "border-blue-500/40 bg-[var(--bg-panel)] shadow-xs"
                          : "border-[var(--border-subtle)] bg-[var(--bg-panel)]/85 hover:border-[var(--border-strong)]"
                      }`}
                    >
                      {/* Collapsible Completed LangGraph Pipeline Summary */}
                      <LivePipelineTracker
                        mode="completed"
                        steps={turn.result.trace || []}
                        totalLatencyMs={turn.result.total_latency_ms}
                        onOpenInspector={() => onInspectTurn(idx, "trace")}
                      />

                      {/* Telemetry Pill Ribbon */}
                      <div className="mb-3.5">
                        <AnswerTelemetryBar
                          result={turn.result}
                          onBadgeClick={(tab) => onInspectTurn(idx, tab)}
                        />
                      </div>

                      {/* Markdown Answer with Clickable Inline Citations */}
                      <MarkdownWithCitations
                        content={turn.result.answer}
                        evidenceChunks={chunks}
                        onCitationClick={(cid) => onCitationClick(idx, cid)}
                      />

                      {/* Bottom Sources & Inspector Bar */}
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
                                className="inline-flex max-w-[200px] items-center gap-1.5 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/70 px-2 py-1 text-[11px] text-[var(--text-secondary)] transition hover:border-blue-500/40 hover:text-[var(--text-primary)] cursor-pointer"
                              >
                                <span
                                  className={`shrink-0 font-mono text-[10px] font-bold ${
                                    isWeb ? "text-purple-400" : "text-teal-400"
                                  }`}
                                >
                                  [{cid}]
                                </span>
                                {isWeb ? (
                                  <Globe className="h-3 w-3 shrink-0 text-purple-400" />
                                ) : (
                                  <FileText className="h-3 w-3 shrink-0 text-teal-400" />
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
                              className="rounded-lg px-2 py-1 text-[11px] font-medium text-blue-500 hover:underline cursor-pointer"
                            >
                              +{chunks.length - 6} more
                            </button>
                          )}
                        </div>

                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            onInspectTurn(idx, "trace");
                          }}
                          className={`inline-flex shrink-0 items-center gap-1.5 rounded-lg px-2.5 py-1 text-[11px] font-medium transition cursor-pointer ${
                            isSelected
                              ? "bg-blue-500/15 text-blue-500 dark:text-blue-300"
                              : "text-[var(--text-muted)] hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)]"
                          }`}
                        >
                          <Microscope className="h-3.5 w-3.5" />
                          <span>{isSelected ? "Inspecting" : `Inspect #${idx + 1}`}</span>
                        </button>
                      </div>
                    </div>
                  </article>
                );
              })}

              {/* Active Streaming Turn */}
              {isRunning && pendingQuery && (
                <article className="space-y-4">
                  <div className="flex justify-end">
                    <div className="max-w-[85%] rounded-2xl rounded-tr-sm border border-blue-500/25 bg-blue-600/12 px-4 py-2.5 text-sm leading-relaxed text-[var(--text-primary)]">
                      {pendingQuery}
                    </div>
                  </div>
                  <div className="rounded-2xl border border-blue-500/35 bg-[var(--bg-panel)] p-4 sm:p-5">
                    <LivePipelineTracker
                      mode="streaming"
                      steps={liveSteps}
                      liveUsage={liveUsage}
                    />
                    <div className="flex items-center gap-2 text-xs text-[var(--text-muted)]">
                      <span className="h-2 w-2 animate-pulse rounded-full bg-blue-500" />
                      <span>
                        {useStreaming
                          ? "Streaming live LangGraph state transitions over SSE..."
                          : "Running full AdaptiveRAG batch pipeline..."}
                      </span>
                    </div>
                  </div>
                </article>
              )}

              {/* Stream Error Banner */}
              {streamError && !isRunning && (
                <div className="flex items-start gap-2.5 rounded-xl border border-rose-500/35 bg-rose-500/10 px-4 py-3 text-xs text-rose-600 dark:text-rose-300">
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                  <div>
                    <div className="font-semibold">Query Execution Failed</div>
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
      <div className="shrink-0 border-t border-[var(--border-subtle)] bg-[var(--bg-panel)]/90 px-4 py-3 backdrop-blur sm:px-7">
        <div className="mx-auto max-w-3xl">
          <div className="relative flex items-end gap-2 rounded-2xl border border-[var(--border-strong)] bg-[var(--bg-canvas)] px-3.5 py-2.5 shadow-xs focus-within:border-blue-500 focus-within:ring-2 focus-within:ring-blue-500/15">
            <textarea
              ref={textareaRef}
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isRunning}
              placeholder="Ask a technical question in English or Turkish... (Enter to send, Shift+Enter for newline)"
              className="max-h-40 flex-1 resize-none bg-transparent py-1 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none disabled:opacity-60"
            />
            {isRunning ? (
              <button
                type="button"
                onClick={onStopStream}
                title="Stop generation"
                className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-rose-600 text-white transition hover:bg-rose-500 cursor-pointer"
              >
                <Square className="h-3.5 w-3.5 fill-current" />
              </button>
            ) : (
              <button
                type="button"
                onClick={handleSend}
                disabled={!input.trim()}
                title="Send query"
                className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-blue-600 text-white transition hover:bg-blue-500 disabled:opacity-35 cursor-pointer"
              >
                <ArrowUp className="h-4 w-4" />
              </button>
            )}
          </div>

          <div className="mt-1.5 flex items-center justify-between px-1 text-[10.5px] text-[var(--text-muted)]">
            <div className="flex items-center gap-2">
              <span>
                Mode: <strong className="text-[var(--text-secondary)]">{presetShortLabel}</strong>
              </span>
              <span>·</span>
              <span>{useStreaming ? "SSE Stream" : "Batch JSON"}</span>
              {budgetCapLabel && (
                <>
                  <span>·</span>
                  <span className="font-mono text-blue-400">Cap {budgetCapLabel}</span>
                </>
              )}
            </div>
            <span className="hidden sm:inline">
              Click any <code className="font-mono">[C1]</code> badge to inspect source chunk
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
