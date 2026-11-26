import React from "react";
import { ArrowUpRight, Globe, RefreshCw, ShieldAlert, Wand2, Zap } from "lucide-react";
import { capitalize, formatDuration, formatInt, formatUsd } from "../lib/format";
import type { FinalAnswer, InspectorTab } from "../types/api";
import { cn, Tag, type Tone } from "./ui/primitives";

interface AnswerTelemetryBarProps {
  result: FinalAnswer;
  onBadgeClick?: (tab: InspectorTab) => void;
  className?: string;
}

const LEVEL_TONE: Record<string, Tone> = {
  high: "ok",
  medium: "warn",
  low: "bad",
};

const TONE_TEXT: Record<Tone, string> = {
  neutral: "text-ink-2",
  ok: "text-ok",
  warn: "text-warn",
  bad: "text-bad",
  accent: "text-accent",
};

function ReceiptCell({
  label,
  children,
  onClick,
  title,
}: {
  label: string;
  children: React.ReactNode;
  onClick?: () => void;
  title?: string;
}) {
  const body = (
    <>
      <span className="block text-caption text-ink-3">{label}</span>
      <span className="mt-px flex items-baseline gap-1.5 whitespace-nowrap text-ui font-medium text-ink tabular-nums">
        {children}
      </span>
    </>
  );
  if (!onClick) {
    return (
      <div className="px-1.5 py-1" title={title}>
        {body}
      </div>
    );
  }
  return (
    <button
      type="button"
      title={title}
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
      className="cursor-pointer rounded-md px-1.5 py-1 text-left transition-colors hover:bg-hover"
    >
      {body}
    </button>
  );
}

/**
 * The run receipt: route, retrieval, confidence, audit, time and cost in a
 * fixed order. Each cell opens the matching inspector tab.
 */
export const AnswerTelemetryBar: React.FC<AnswerTelemetryBarProps> = ({
  result,
  onBadgeClick,
  className,
}) => {
  const conf = result.confidence;
  const cite = result.citation_report;
  const usage = result.usage;
  const passages = result.evidence_chunks?.length ?? 0;
  const retrievalSkipped = result.query_plan ? !result.query_plan.needs_retrieval : passages === 0;
  const open = (tab: InspectorTab) => (onBadgeClick ? () => onBadgeClick(tab) : undefined);

  const confTone = conf ? (LEVEL_TONE[conf.level] ?? "neutral") : "neutral";
  const auditTone: Tone = cite ? (cite.passed ? "ok" : "bad") : "neutral";

  const notes: { key: string; tone: Tone; icon: typeof Globe; label: string }[] = [];
  if (result.fallback_used) {
    notes.push({ key: "fallback", tone: "warn", icon: Globe, label: "Web fallback used" });
  }
  if (result.abstained) {
    notes.push({ key: "abstained", tone: "bad", icon: ShieldAlert, label: "Abstained" });
  }
  if (result.rewrite_attempts > 0) {
    notes.push({
      key: "rewrite",
      tone: "neutral",
      icon: RefreshCw,
      label: `Query rewritten ×${result.rewrite_attempts}`,
    });
  }
  if (result.regeneration_attempts > 0) {
    notes.push({
      key: "regen",
      tone: "neutral",
      icon: Wand2,
      label: `Regenerated ×${result.regeneration_attempts}`,
    });
  }
  if (usage.cache_hits > 0) {
    notes.push({ key: "cache", tone: "ok", icon: Zap, label: `Cache hits ×${usage.cache_hits}` });
  }

  return (
    <div className={cn("-mx-1.5", className)}>
      <div className="flex flex-wrap items-start gap-x-3 gap-y-1">
        <ReceiptCell label="Route" onClick={open("trace")} title="Open the query plan">
          <span>{capitalize(result.query_type)}</span>
          <span className="text-small font-normal text-ink-3">
            {(result.language || "").toUpperCase()}
          </span>
        </ReceiptCell>

        <ReceiptCell
          label="Passages"
          onClick={open("dynamic_k")}
          title="Open retrieval scores"
        >
          {retrievalSkipped && passages === 0 ? (
            <span className="font-normal text-ink-3">Skipped</span>
          ) : (
            <span>{passages}</span>
          )}
        </ReceiptCell>

        <ReceiptCell label="Confidence" onClick={open("confidence")} title="Open the CRAG gate">
          {conf ? (
            <>
              <span>{conf.final_score.toFixed(2)}</span>
              <span className={cn("text-small", TONE_TEXT[confTone])}>{capitalize(conf.level)}</span>
            </>
          ) : (
            <span className="font-normal text-ink-3">Not scored</span>
          )}
        </ReceiptCell>

        <ReceiptCell label="Citation audit" onClick={open("citations")} title="Open the audit">
          {cite ? (
            <>
              <span>{Math.round(cite.claim_support_rate * 100)}%</span>
              <span className={cn("text-small", TONE_TEXT[auditTone])}>
                {cite.passed ? "Passed" : "Flagged"}
              </span>
            </>
          ) : (
            <span className="font-normal text-ink-3">Skipped</span>
          )}
        </ReceiptCell>

        <ReceiptCell label="Time">
          <span>{formatDuration(result.total_latency_ms)}</span>
        </ReceiptCell>

        <ReceiptCell label="Usage">
          <span>{formatInt(usage.total_tokens)} tok</span>
          <span className="text-small font-normal text-ink-3">
            {formatUsd(usage.estimated_cost_usd)}
          </span>
        </ReceiptCell>

        {result.trace_url && (
          <a
            href={result.trace_url}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
            title={result.trace_id ? `Open Langfuse trace ${result.trace_id}` : "Open in Langfuse"}
            className="self-end rounded-md px-1.5 py-1 text-small font-medium text-accent transition-colors hover:bg-hover"
          >
            <span className="inline-flex items-center gap-0.5">
              Langfuse
              <ArrowUpRight className="h-3.5 w-3.5" aria-hidden />
            </span>
          </a>
        )}
      </div>

      {notes.length > 0 && (
        <div className="mt-1.5 flex flex-wrap gap-1.5 px-1.5">
          {notes.map((note) => (
            <Tag key={note.key} tone={note.tone} icon={note.icon}>
              {note.label}
            </Tag>
          ))}
        </div>
      )}
    </div>
  );
};
