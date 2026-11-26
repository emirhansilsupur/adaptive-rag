import React, { useState } from "react";
import { ChevronDown } from "lucide-react";
import { getNodeMeta } from "../constants/pipeline";
import { formatDuration, formatInt } from "../lib/format";
import type { LiveStreamStep, NodeTraceEvent, TokenUsage } from "../types/api";
import { cn, Spinner } from "./ui/primitives";

type Step = LiveStreamStep | NodeTraceEvent;

interface LivePipelineTrackerProps {
  mode: "streaming" | "completed";
  steps: Step[];
  totalLatencyMs?: number;
  liveUsage?: Partial<TokenUsage> | null;
  onOpenInspector?: () => void;
  /** Heading shown while streaming, e.g. "Running Naive Dense". */
  runningLabel?: string;
}

/** Proportional strip: one segment per graph node, coloured by pipeline stage. */
export function StageStrip({
  steps,
  className,
  live = false,
}: {
  steps: Step[];
  className?: string;
  live?: boolean;
}) {
  return (
    <div
      aria-hidden
      className={cn("flex h-1.5 w-full gap-px overflow-hidden rounded-full bg-line", className)}
    >
      {steps.map((step, idx) => {
        const meta = getNodeMeta(step.node);
        return (
          <span
            key={`${step.node}-${idx}`}
            title={`${meta.label}: ${formatDuration(step.latency_ms)}`}
            className="h-full min-w-[3px] first:rounded-l-full last:rounded-r-full"
            style={{
              flexGrow: Math.max(step.latency_ms, 1),
              flexBasis: 0,
              backgroundColor: meta.color,
            }}
          />
        );
      })}
      {live && <span className="live-dot h-full min-w-[24px] grow-0 bg-ink-4/50" />}
    </div>
  );
}

function StepRows({ steps, pending }: { steps: Step[]; pending?: boolean }) {
  return (
    <ol className="divide-y divide-line overflow-hidden rounded-lg border border-line bg-paper">
      {steps.map((step, idx) => {
        const meta = getNodeMeta(step.node);
        return (
          <li
            key={`${step.node}-${idx}`}
            className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-start gap-x-2.5 px-3 py-2"
          >
            <span
              aria-hidden
              className="mt-[6px] h-2 w-2 rounded-full"
              style={{ backgroundColor: meta.color }}
            />
            <div className="min-w-0">
              <div className="text-ui text-ink">{meta.label}</div>
              {step.summary && (
                <div className="truncate text-small text-ink-3" title={step.summary}>
                  {step.summary}
                </div>
              )}
            </div>
            <span className="text-small tabular-nums text-ink-3">
              {step.latency_ms > 0 ? formatDuration(step.latency_ms) : "…"}
            </span>
          </li>
        );
      })}
      {pending && (
        <li className="flex items-center gap-2.5 px-3 py-2 text-small text-ink-3">
          <Spinner className="h-3 w-3" />
          <span>Next step running</span>
        </li>
      )}
    </ol>
  );
}

export const LivePipelineTracker: React.FC<LivePipelineTrackerProps> = ({
  mode,
  steps,
  totalLatencyMs,
  liveUsage,
  onOpenInspector,
  runningLabel,
}) => {
  const [expanded, setExpanded] = useState(false);
  const elapsedMs = totalLatencyMs ?? steps.reduce((sum, s) => sum + (s.latency_ms || 0), 0);

  if (mode === "streaming") {
    const lastStep = steps.length > 0 ? getNodeMeta(steps[steps.length - 1].node) : null;
    const tokens = liveUsage?.total_tokens ?? 0;
    return (
      <div className="rounded-lg border border-line bg-sunken p-3" aria-live="polite">
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1">
          <span className="inline-flex min-w-0 items-center gap-2 text-ui font-medium text-ink">
            <Spinner className="text-accent" />
            <span className="truncate">{runningLabel ?? "Running the pipeline"}</span>
          </span>
          <span className="flex items-center gap-3 text-small tabular-nums text-ink-3">
            <span>
              {steps.length} {steps.length === 1 ? "step" : "steps"} done
            </span>
            <span>{formatDuration(elapsedMs)}</span>
            {tokens > 0 && <span>{formatInt(tokens)} tokens</span>}
          </span>
        </div>
        <StageStrip steps={steps} live className="mt-2.5" />
        {steps.length === 0 ? (
          <p className="mt-2.5 text-small text-ink-3">Routing the question…</p>
        ) : (
          <div className="mt-2.5">
            <StepRows steps={steps} pending />
            {lastStep && <span className="sr-only">Last finished step: {lastStep.label}</span>}
          </div>
        )}
      </div>
    );
  }

  if (steps.length === 0) return null;

  return (
    <div>
      <div className="flex items-center gap-3">
        <StageStrip steps={steps} className="flex-1" />
        <button
          type="button"
          aria-expanded={expanded}
          onClick={(e) => {
            e.stopPropagation();
            setExpanded((prev) => !prev);
          }}
          className="inline-flex shrink-0 cursor-pointer items-center gap-1 rounded-md px-1.5 py-0.5 text-small text-ink-3 transition-colors hover:bg-hover hover:text-ink"
        >
          <span className="tabular-nums">
            {steps.length} {steps.length === 1 ? "step" : "steps"}
          </span>
          <ChevronDown
            className={cn("h-3.5 w-3.5 transition-transform", expanded && "rotate-180")}
            aria-hidden
          />
        </button>
      </div>
      {expanded && (
        <div className="mt-2.5" onClick={(e) => e.stopPropagation()}>
          <StepRows steps={steps} />
          {onOpenInspector && (
            <button
              type="button"
              onClick={onOpenInspector}
              className="mt-1.5 cursor-pointer text-small font-medium text-accent hover:underline"
            >
              Open the timeline in the inspector
            </button>
          )}
        </div>
      )}
    </div>
  );
};
