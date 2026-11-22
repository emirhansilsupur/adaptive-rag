import React, { useState } from "react";
import {
  Activity,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Loader2,
  Sparkles,
} from "lucide-react";
import { getNodeMeta } from "../constants/pipeline";
import type { LiveStreamStep, NodeTraceEvent, TokenUsage } from "../types/api";

interface LivePipelineTrackerProps {
  mode: "streaming" | "completed";
  steps: (LiveStreamStep | NodeTraceEvent)[];
  totalLatencyMs?: number;
  liveUsage?: Partial<TokenUsage> | null;
  onOpenInspector?: () => void;
}

export const LivePipelineTracker: React.FC<LivePipelineTrackerProps> = ({
  mode,
  steps,
  totalLatencyMs,
  liveUsage,
  onOpenInspector,
}) => {
  const [expanded, setExpanded] = useState(mode === "streaming");
  const isStreaming = mode === "streaming";
  const computedMs =
    totalLatencyMs ?? steps.reduce((sum, s) => sum + (s.latency_ms || 0), 0);

  if (!isStreaming && steps.length === 0) {
    return null;
  }

  return (
    <div
      className={`mb-3 overflow-hidden rounded-xl border transition-colors ${
        isStreaming
          ? "border-blue-500/35 bg-blue-500/[0.04]"
          : "border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60"
      }`}
    >
      {/* Header / Summary Row */}
      <div className="flex items-center justify-between gap-2 px-3 py-2">
        <button
          type="button"
          onClick={() => setExpanded((prev) => !prev)}
          className="flex min-w-0 flex-1 items-center gap-2 text-left text-xs font-medium text-[var(--text-primary)] cursor-pointer"
        >
          {isStreaming ? (
            <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-blue-400" />
          ) : (
            <Activity className="h-3.5 w-3.5 shrink-0 text-emerald-400" />
          )}

          <span className="truncate">
            {isStreaming
              ? steps.length > 0
                ? `Executing LangGraph · Last completed: ${getNodeMeta(steps[steps.length - 1].node).label}`
                : "Starting AdaptiveRAG agentic pipeline..."
              : `Executed ${steps.length} LangGraph nodes`}
          </span>

          <span className="shrink-0 font-mono text-[11px] text-[var(--text-muted)] tabular-nums">
            {Math.round(computedMs).toLocaleString("en-US")} ms
          </span>

          {isStreaming && liveUsage && (liveUsage.total_tokens ?? 0) > 0 && (
            <span className="hidden shrink-0 font-mono text-[11px] text-blue-400 tabular-nums sm:inline">
              {(liveUsage.total_tokens ?? 0).toLocaleString("en-US")} tok · $
              {(liveUsage.estimated_cost_usd ?? 0).toFixed(5)}
            </span>
          )}

          {expanded ? (
            <ChevronDown className="ml-auto h-3.5 w-3.5 shrink-0 text-[var(--text-muted)]" />
          ) : (
            <ChevronRight className="ml-auto h-3.5 w-3.5 shrink-0 text-[var(--text-muted)]" />
          )}
        </button>

        {!isStreaming && onOpenInspector && (
          <button
            type="button"
            onClick={onOpenInspector}
            className="inline-flex shrink-0 items-center gap-1 rounded-md border border-[var(--border-subtle)] bg-[var(--bg-panel)] px-2 py-0.5 text-[11px] font-medium text-[var(--text-secondary)] transition hover:border-blue-500/40 hover:text-blue-400 cursor-pointer"
          >
            <Sparkles className="h-3 w-3" />
            <span>Trace</span>
          </button>
        )}
      </div>

      {/* Horizontal pill track (always visible when streaming or expanded) */}
      {steps.length > 0 && (
        <div className="flex flex-wrap items-center gap-1 border-t border-[var(--border-subtle)]/70 px-3 py-1.5">
          {steps.map((step, idx) => {
            const meta = getNodeMeta(step.node);
            return (
              <React.Fragment key={`${step.node}-${idx}`}>
                <span
                  className={`inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[10.5px] font-medium ${meta.bgClass} ${meta.borderClass} ${meta.textClass}`}
                  title={`${meta.label} (${step.latency_ms.toFixed(1)} ms) — ${step.summary}`}
                >
                  <span
                    className="h-1.5 w-1.5 rounded-full"
                    style={{ backgroundColor: meta.color }}
                  />
                  <span>{meta.shortLabel}</span>
                  <span className="font-mono text-[10px] opacity-75 tabular-nums">
                    {step.latency_ms >= 1000
                      ? `${(step.latency_ms / 1000).toFixed(1)}s`
                      : `${Math.round(step.latency_ms)}ms`}
                  </span>
                </span>
                {idx < steps.length - 1 && (
                  <span className="text-[10px] text-[var(--text-muted)]">→</span>
                )}
              </React.Fragment>
            );
          })}
          {isStreaming && (
            <span className="inline-flex items-center gap-1 rounded-md border border-dashed border-blue-500/40 bg-blue-500/10 px-1.5 py-0.5 text-[10.5px] text-blue-400">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-blue-400" />
              <span>next</span>
            </span>
          )}
        </div>
      )}

      {/* Detailed step list when expanded */}
      {expanded && steps.length > 0 && (
        <div className="divide-y divide-[var(--border-subtle)]/60 border-t border-[var(--border-subtle)] bg-[var(--bg-panel)]/70">
          {steps.map((step, idx) => {
            const meta = getNodeMeta(step.node);
            return (
              <div
                key={`detail-${step.node}-${idx}`}
                className="flex items-start gap-2.5 px-3 py-1.5 text-xs"
              >
                <CheckCircle2
                  className="mt-0.5 h-3.5 w-3.5 shrink-0"
                  style={{ color: meta.color }}
                />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-medium text-[var(--text-primary)]">
                      {idx + 1}. {meta.label}
                    </span>
                    <span className="shrink-0 font-mono text-[11px] text-[var(--text-muted)] tabular-nums">
                      {step.latency_ms.toFixed(1)} ms
                    </span>
                  </div>
                  {step.summary && (
                    <p className="mt-0.5 truncate text-[11.5px] text-[var(--text-secondary)]">
                      {step.summary}
                    </p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
