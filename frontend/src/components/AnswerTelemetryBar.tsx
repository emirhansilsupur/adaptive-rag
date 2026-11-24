import React from "react";
import {
  AlertTriangle,
  ArrowUpRight,
  Check,
  Globe,
  RefreshCw,
  ShieldAlert,
  Zap,
} from "lucide-react";
import type { FinalAnswer } from "../types/api";

interface AnswerTelemetryBarProps {
  result: FinalAnswer;
  onBadgeClick?: (tab: "trace" | "confidence" | "citations" | "evidence") => void;
}

export const AnswerTelemetryBar: React.FC<AnswerTelemetryBarProps> = ({
  result,
  onBadgeClick,
}) => {
  const conf = result.confidence;
  const cite = result.citation_report;
  const usage = result.usage;

  const confDot =
    conf?.level === "high"
      ? "bg-emerald-500"
      : conf?.level === "medium"
        ? "bg-amber-500"
        : "bg-rose-500";

  return (
    <div className="flex flex-wrap items-center gap-1.5 font-mono text-[10.5px] text-[var(--text-secondary)] tabular-nums">
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onBadgeClick?.("trace");
        }}
        className="inline-flex items-center gap-1 rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60 px-1.5 py-0.5 font-medium text-[var(--text-primary)] transition hover:border-[var(--border-strong)] cursor-pointer"
      >
        <span className="uppercase">{result.query_type}</span>
        <span className="text-[var(--text-muted)]">/</span>
        <span className="uppercase text-[var(--text-secondary)]">{result.language}</span>
      </button>

      {conf && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onBadgeClick?.("confidence");
          }}
          className="inline-flex items-center gap-1.5 rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60 px-1.5 py-0.5 transition hover:border-[var(--border-strong)] hover:text-[var(--text-primary)] cursor-pointer"
        >
          <span className={`h-1.5 w-1.5 rounded-full ${confDot}`} />
          <span>
            CRAG {conf.final_score.toFixed(2)} ({conf.level.toUpperCase()})
          </span>
        </button>
      )}

      {cite && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onBadgeClick?.("citations");
          }}
          className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 transition cursor-pointer ${
            cite.passed
              ? "border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60 hover:border-[var(--border-strong)] hover:text-[var(--text-primary)]"
              : "border-rose-500/35 bg-rose-500/10 text-rose-600 dark:text-rose-300"
          }`}
        >
          {cite.passed ? (
            <Check className="h-2.5 w-2.5 text-emerald-500" />
          ) : (
            <AlertTriangle className="h-2.5 w-2.5 text-rose-500" />
          )}
          <span>Audit {Math.round(cite.claim_support_rate * 100)}%</span>
        </button>
      )}

      {result.fallback_used && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onBadgeClick?.("evidence");
          }}
          className="inline-flex items-center gap-1 rounded border border-amber-500/30 bg-amber-500/10 px-1.5 py-0.5 text-amber-600 dark:text-amber-300 cursor-pointer"
        >
          <Globe className="h-2.5 w-2.5" />
          <span>Web Fallback</span>
        </button>
      )}

      {result.abstained && (
        <span className="inline-flex items-center gap-1 rounded border border-rose-500/30 bg-rose-500/10 px-1.5 py-0.5 text-rose-600 dark:text-rose-300">
          <ShieldAlert className="h-2.5 w-2.5" />
          <span>Abstained</span>
        </span>
      )}

      {result.rewrite_attempts > 0 && (
        <span className="inline-flex items-center gap-1 rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60 px-1.5 py-0.5">
          <RefreshCw className="h-2.5 w-2.5 text-amber-500" />
          <span>Rewrite ×{result.rewrite_attempts}</span>
        </span>
      )}

      {result.regeneration_attempts > 0 && (
        <span className="inline-flex items-center gap-1 rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60 px-1.5 py-0.5">
          <span>Regen ×{result.regeneration_attempts}</span>
        </span>
      )}

      <span className="inline-flex items-center gap-1.5 px-1 text-[var(--text-muted)]">
        <span>{Math.round(result.total_latency_ms).toLocaleString("en-US")} ms</span>
        <span>·</span>
        <span>{usage.total_tokens.toLocaleString("en-US")} tok</span>
        <span>·</span>
        <span>${usage.estimated_cost_usd.toFixed(5)}</span>
      </span>

      {usage.cache_hits > 0 && (
        <span className="inline-flex items-center gap-0.5 rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60 px-1.5 py-0.5 text-emerald-600 dark:text-emerald-400">
          <Zap className="h-2.5 w-2.5" />
          <span>cache ×{usage.cache_hits}</span>
        </span>
      )}

      {result.trace_url && (
        <a
          href={result.trace_url}
          target="_blank"
          rel="noopener noreferrer"
          onClick={(e) => e.stopPropagation()}
          title={
            result.trace_id
              ? `Open Langfuse trace ${result.trace_id}`
              : "Open in Langfuse"
          }
          className="inline-flex items-center gap-0.5 rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60 px-1.5 py-0.5 text-[var(--text-secondary)] transition hover:border-[var(--border-strong)] hover:text-[var(--text-primary)]"
        >
          <span>Langfuse</span>
          <ArrowUpRight className="h-2.5 w-2.5 opacity-75" />
        </a>
      )}
    </div>
  );
};
