import React from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  Coins,
  Compass,
  Globe,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Target,
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

  const confTone =
    conf?.level === "high"
      ? "border-emerald-500/35 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
      : conf?.level === "medium"
        ? "border-amber-500/35 bg-amber-500/10 text-amber-700 dark:text-amber-300"
        : "border-rose-500/35 bg-rose-500/10 text-rose-700 dark:text-rose-300";

  return (
    <div className="flex flex-wrap items-center gap-1.5 text-[11px] font-medium tabular-nums">
      <button
        type="button"
        onClick={() => onBadgeClick?.("trace")}
        className="inline-flex items-center gap-1 rounded-full border border-blue-500/30 bg-blue-500/10 px-2.5 py-0.5 text-blue-700 transition hover:bg-blue-500/20 dark:text-blue-300 cursor-pointer"
      >
        <Compass className="h-3 w-3" />
        <span className="uppercase">{result.query_type}</span>
        <span className="opacity-60">·</span>
        <span className="uppercase">{result.language}</span>
      </button>

      {conf && (
        <button
          type="button"
          onClick={() => onBadgeClick?.("confidence")}
          className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 transition hover:brightness-110 cursor-pointer ${confTone}`}
        >
          <Target className="h-3 w-3" />
          <span>
            Conf {conf.final_score.toFixed(2)} ({conf.level.toUpperCase()})
          </span>
        </button>
      )}

      {cite && (
        <button
          type="button"
          onClick={() => onBadgeClick?.("citations")}
          className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 transition hover:brightness-110 cursor-pointer ${
            cite.passed
              ? "border-emerald-500/35 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
              : "border-rose-500/35 bg-rose-500/10 text-rose-700 dark:text-rose-300"
          }`}
        >
          {cite.passed ? (
            <CheckCircle2 className="h-3 w-3" />
          ) : (
            <AlertTriangle className="h-3 w-3" />
          )}
          <span>Audit {Math.round(cite.claim_support_rate * 100)}%</span>
        </button>
      )}

      {result.fallback_used && (
        <button
          type="button"
          onClick={() => onBadgeClick?.("evidence")}
          className="inline-flex items-center gap-1 rounded-full border border-purple-500/35 bg-purple-500/10 px-2.5 py-0.5 text-purple-700 dark:text-purple-300 cursor-pointer"
        >
          <Globe className="h-3 w-3" />
          <span>Web Fallback</span>
        </button>
      )}

      {result.abstained && (
        <span className="inline-flex items-center gap-1 rounded-full border border-rose-500/35 bg-rose-500/10 px-2.5 py-0.5 text-rose-700 dark:text-rose-300">
          <ShieldAlert className="h-3 w-3" />
          <span>Abstained</span>
        </span>
      )}

      {result.rewrite_attempts > 0 && (
        <span className="inline-flex items-center gap-1 rounded-full border border-amber-500/35 bg-amber-500/10 px-2.5 py-0.5 text-amber-700 dark:text-amber-300">
          <RefreshCw className="h-3 w-3" />
          <span>Rewrites: {result.rewrite_attempts}</span>
        </span>
      )}

      {result.regeneration_attempts > 0 && (
        <span className="inline-flex items-center gap-1 rounded-full border border-orange-500/35 bg-orange-500/10 px-2.5 py-0.5 text-orange-700 dark:text-orange-300">
          <ShieldCheck className="h-3 w-3" />
          <span>Strict Regen: {result.regeneration_attempts}</span>
        </span>
      )}

      <span className="inline-flex items-center gap-1 rounded-full border border-[var(--border-subtle)] bg-[var(--bg-elevated)] px-2.5 py-0.5 text-[var(--text-secondary)]">
        <Clock className="h-3 w-3 opacity-75" />
        <span>{Math.round(result.total_latency_ms).toLocaleString("en-US")} ms</span>
      </span>

      <span className="inline-flex items-center gap-1 rounded-full border border-[var(--border-subtle)] bg-[var(--bg-elevated)] px-2.5 py-0.5 font-mono text-[var(--text-secondary)]">
        <Coins className="h-3 w-3 opacity-75" />
        <span>
          {usage.total_tokens.toLocaleString("en-US")} tok · $
          {usage.estimated_cost_usd.toFixed(5)}
        </span>
      </span>

      {usage.cache_hits > 0 && (
        <span className="inline-flex items-center gap-1 rounded-full border border-emerald-500/35 bg-emerald-500/10 px-2 py-0.5 text-emerald-700 dark:text-emerald-300">
          <Zap className="h-3 w-3" />
          <span>Cache ×{usage.cache_hits}</span>
        </span>
      )}
    </div>
  );
};
