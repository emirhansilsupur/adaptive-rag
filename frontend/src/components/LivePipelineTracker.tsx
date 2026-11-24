import React, { useState } from "react";
import { ArrowUpRight, ChevronDown, ChevronRight, Loader2 } from "lucide-react";
import { getNodeMeta } from "../constants/pipeline";
import type { LiveStreamStep, NodeTraceEvent, TokenUsage } from "../types/api";

interface LivePipelineTrackerProps {
  mode: "streaming" | "completed";
  steps: (LiveStreamStep | NodeTraceEvent)[];
  totalLatencyMs?: number;
  liveUsage?: Partial<TokenUsage> | null;
  onOpenInspector?: () => void;
}

function formatDuration(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.round(ms)}ms`;
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
  const stepSumMs = Math.max(
    steps.reduce((sum, s) => sum + Math.max(s.latency_ms || 0, 1), 0),
    1
  );

  if (!isStreaming && steps.length === 0) {
    return null;
  }

  return (
    <div className="mb-2.5 overflow-hidden rounded-md border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/45">
      {/* Compact Single-Line Header */}
      <div className="flex h-8 items-center justify-between gap-2 px-2.5">
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            setExpanded((prev) => !prev);
          }}
          className="flex min-w-0 flex-1 items-center gap-2 text-left text-[11.5px] text-[var(--text-secondary)] transition hover:text-[var(--text-primary)] cursor-pointer"
        >
          {expanded ? (
            <ChevronDown className="h-3 w-3 shrink-0 text-[var(--text-muted)]" />
          ) : (
            <ChevronRight className="h-3 w-3 shrink-0 text-[var(--text-muted)]" />
          )}

          {isStreaming ? (
            <Loader2 className="h-3 w-3 shrink-0 animate-spin text-[var(--text-primary)]" />
          ) : (
            <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-500" />
          )}

          <span className="shrink-0 font-mono text-[11px] font-medium text-[var(--text-primary)] tabular-nums">
            {isStreaming
              ? steps.length > 0
                ? `Running · ${getNodeMeta(steps[steps.length - 1].node).shortLabel}`
                : "Initializing graph"
              : `${steps.length} steps`}
          </span>

          {!expanded && steps.length > 0 && (
            <span className="hidden truncate font-mono text-[10.5px] text-[var(--text-muted)] sm:inline">
              {steps.map((s) => getNodeMeta(s.node).shortLabel).join(" → ")}
            </span>
          )}

          <span className="ml-auto shrink-0 font-mono text-[10.5px] text-[var(--text-muted)] tabular-nums">
            {formatDuration(computedMs)}
          </span>

          {isStreaming && liveUsage && (liveUsage.total_tokens ?? 0) > 0 && (
            <span className="hidden shrink-0 font-mono text-[10.5px] text-[var(--text-secondary)] tabular-nums sm:inline">
              · {(liveUsage.total_tokens ?? 0).toLocaleString("en-US")} tok
            </span>
          )}
        </button>

        {!isStreaming && onOpenInspector && (
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onOpenInspector();
            }}
            className="inline-flex shrink-0 items-center gap-0.5 rounded border border-[var(--border-subtle)] bg-[var(--bg-surface)] px-1.5 py-0.5 font-mono text-[10px] font-medium text-[var(--text-secondary)] transition hover:border-[var(--border-strong)] hover:text-[var(--text-primary)] cursor-pointer"
          >
            <span>Trace</span>
            <ArrowUpRight className="h-2.5 w-2.5" />
          </button>
        )}
      </div>

      {/* Proportional 2px Segmented Latency Bar */}
      {steps.length > 0 && (
        <div className="flex h-[2px] w-full overflow-hidden bg-[var(--bg-hover)]">
          {steps.map((step, idx) => {
            const meta = getNodeMeta(step.node);
            const widthPct = Math.max(2, (Math.max(step.latency_ms, 1) / stepSumMs) * 100);
            return (
              <div
                key={`seg-${step.node}-${idx}`}
                style={{ width: `${widthPct}%`, backgroundColor: meta.color }}
                className="h-full border-r border-[var(--bg-canvas)]/60 last:border-r-0"
                title={`${meta.label}: ${step.latency_ms.toFixed(1)} ms`}
              />
            );
          })}
        </div>
      )}

      {/* Expanded Step Ledger */}
      {expanded && steps.length > 0 && (
        <div
          onClick={(e) => e.stopPropagation()}
          className="divide-y divide-[var(--border-subtle)] border-t border-[var(--border-subtle)] bg-[var(--bg-surface)]/80"
        >
          {steps.map((step, idx) => {
            const meta = getNodeMeta(step.node);
            return (
              <div
                key={`detail-${step.node}-${idx}`}
                className="flex items-start gap-2.5 px-2.5 py-1.5 text-[11.5px]"
              >
                <span className="mt-1 flex items-center gap-1.5 shrink-0">
                  <span className="w-3 font-mono text-[10px] text-[var(--text-muted)] tabular-nums">
                    {idx + 1}.
                  </span>
                  <span
                    className="h-1.5 w-1.5 rounded-full"
                    style={{ backgroundColor: meta.color }}
                  />
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-medium text-[var(--text-primary)]">
                      {meta.label}
                    </span>
                    <span className="shrink-0 font-mono text-[10.5px] text-[var(--text-muted)] tabular-nums">
                      {step.latency_ms.toFixed(1)} ms
                    </span>
                  </div>
                  {step.summary && (
                    <p className="mt-0.5 truncate text-[11px] text-[var(--text-secondary)]">
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
