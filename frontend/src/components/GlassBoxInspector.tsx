import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  BarChart3,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Code2,
  ExternalLink,
  FileText,
  Gauge,
  Globe,
  Layers,
  ShieldCheck,
  X,
  XCircle,
} from "lucide-react";
import { getNodeMeta } from "../constants/pipeline";
import type {
  ChatTurn,
  EvidenceChunk,
  FinalAnswer,
  InspectorTab,
} from "../types/api";

interface GlassBoxInspectorProps {
  turns: ChatTurn[];
  selectedTurnIndex: number;
  onSelectTurnIndex: (idx: number) => void;
  activeTab: InspectorTab;
  onTabChange: (tab: InspectorTab) => void;
  highlightedCitationId: string | null;
  highlightToken: number;
  onJumpToCitation: (citationId: string) => void;
  onClose: () => void;
  selectorLabel?: string;
}

const TABS: { id: InspectorTab; label: string; icon: React.FC<{ className?: string }> }[] = [
  { id: "trace", label: "Trace", icon: Activity },
  { id: "dynamic_k", label: "Top-K", icon: BarChart3 },
  { id: "confidence", label: "CRAG", icon: Gauge },
  { id: "evidence", label: "Evidence", icon: BookOpen },
  { id: "citations", label: "Audit", icon: ShieldCheck },
];

function KpiTile({
  label,
  value,
  subtext,
  tone = "neutral",
}: {
  label: string;
  value: string;
  subtext?: string;
  tone?: "neutral" | "emerald" | "amber" | "rose" | "blue";
}) {
  const valueTone =
    tone === "emerald"
      ? "text-emerald-500 dark:text-emerald-400"
      : tone === "amber"
        ? "text-amber-500 dark:text-amber-400"
        : tone === "rose"
          ? "text-rose-500 dark:text-rose-400"
          : tone === "blue"
            ? "text-blue-500 dark:text-blue-400"
            : "text-[var(--text-primary)]";

  return (
    <div className="rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/70 px-3 py-2.5">
      <div className="text-[10.5px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
        {label}
      </div>
      <div className={`mt-1 font-mono text-base font-semibold tabular-nums ${valueTone}`}>
        {value}
      </div>
      {subtext && (
        <div className="mt-0.5 truncate text-[10.5px] text-[var(--text-secondary)]">
          {subtext}
        </div>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Tab 1: Trace & Query Plan                                                  */
/* -------------------------------------------------------------------------- */

function TraceAndPlanTab({ result }: { result: FinalAnswer }) {
  const [showStrategy, setShowStrategy] = useState(false);
  const plan = result.query_plan;
  const trace = result.trace || [];
  const totalMs = Math.max(
    result.total_latency_ms || 0,
    trace.reduce((acc, e) => acc + e.latency_ms, 0),
    1
  );
  const maxStepMs = Math.max(...trace.map((e) => e.latency_ms), 1);

  return (
    <div className="space-y-4">
      {plan && (
        <section className="space-y-2.5">
          <h4 className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
            Query Plan & Active Strategy
          </h4>
          <div className="grid grid-cols-2 gap-2">
            <KpiTile
              label="Query Type"
              value={plan.query_type.toUpperCase()}
              tone="blue"
            />
            <KpiTile
              label="Language"
              value={plan.language.toUpperCase()}
              subtext={plan.needs_retrieval ? "Retrieval ON" : "Direct Chat"}
            />
            <KpiTile
              label="Retrieval"
              value={plan.needs_retrieval ? "Active" : "Skipped"}
              tone={plan.needs_retrieval ? "emerald" : "neutral"}
            />
            <KpiTile
              label="Fresh Web Flag"
              value={plan.requires_fresh_web_info ? "True" : "False"}
              tone={plan.requires_fresh_web_info ? "amber" : "neutral"}
            />
          </div>

          <div className="rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60 p-3 space-y-2">
            <div>
              <div className="text-[10.5px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
                Standalone Query
              </div>
              <p className="mt-1 font-mono text-xs text-[var(--text-primary)] leading-relaxed">
                {plan.standalone_query}
              </p>
            </div>
            {plan.reasoning && (
              <div className="border-t border-[var(--border-subtle)] pt-2">
                <div className="text-[10.5px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
                  Router Reasoning
                </div>
                <p className="mt-1 text-xs text-[var(--text-secondary)] leading-relaxed">
                  {plan.reasoning}
                </p>
              </div>
            )}
            {plan.sub_queries && plan.sub_queries.length > 0 && (
              <div className="border-t border-[var(--border-subtle)] pt-2">
                <div className="text-[10.5px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
                  Sub-Queries Executed ({plan.sub_queries.length})
                </div>
                <ol className="mt-1.5 space-y-1">
                  {plan.sub_queries.map((sq, i) => (
                    <li
                      key={i}
                      className="flex items-start gap-2 rounded-lg bg-[var(--bg-panel)] px-2.5 py-1.5 font-mono text-[11px] text-[var(--text-primary)]"
                    >
                      <span className="shrink-0 text-blue-500">{i + 1}.</span>
                      <span>{sq}</span>
                    </li>
                  ))}
                </ol>
              </div>
            )}
          </div>

          {plan.strategy && (
            <div className="overflow-hidden rounded-xl border border-[var(--border-subtle)]">
              <button
                type="button"
                onClick={() => setShowStrategy((prev) => !prev)}
                className="flex w-full items-center justify-between bg-[var(--bg-elevated)]/60 px-3 py-2 text-xs font-medium text-[var(--text-secondary)] transition hover:text-[var(--text-primary)] cursor-pointer"
              >
                <span className="inline-flex items-center gap-1.5">
                  <Code2 className="h-3.5 w-3.5 text-blue-400" />
                  Strategy Hyperparameters
                </span>
                {showStrategy ? (
                  <ChevronDown className="h-3.5 w-3.5" />
                ) : (
                  <ChevronRight className="h-3.5 w-3.5" />
                )}
              </button>
              {showStrategy && (
                <pre className="overflow-x-auto border-t border-[var(--border-subtle)] bg-[#0b0f17] p-3 font-mono text-[11px] text-slate-300">
                  {JSON.stringify(plan.strategy, null, 2)}
                </pre>
              )}
            </div>
          )}
        </section>
      )}

      {/* Latency Waterfall */}
      <section className="space-y-2.5">
        <div className="flex items-center justify-between">
          <h4 className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
            LangGraph Latency Waterfall
          </h4>
          <span className="font-mono text-[11px] text-[var(--text-secondary)] tabular-nums">
            Total {Math.round(totalMs).toLocaleString("en-US")} ms
          </span>
        </div>

        {trace.length === 0 ? (
          <div className="rounded-xl border border-dashed border-[var(--border-subtle)] p-4 text-center text-xs text-[var(--text-muted)]">
            No node trace events recorded for this turn.
          </div>
        ) : (
          <div className="space-y-2">
            {trace.map((evt, idx) => {
              const meta = getNodeMeta(evt.node);
              const pct = Math.min(100, Math.max(3, (evt.latency_ms / maxStepMs) * 100));
              const share = Math.round((evt.latency_ms / totalMs) * 100);
              return (
                <div
                  key={`${evt.node}-${idx}`}
                  className="rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/55 p-2.5"
                >
                  <div className="flex items-center justify-between gap-2 text-xs">
                    <div className="flex min-w-0 items-center gap-2">
                      <span
                        className="flex h-5 w-5 shrink-0 items-center justify-center rounded-md font-mono text-[10px] font-semibold text-white"
                        style={{ backgroundColor: meta.color }}
                      >
                        {idx + 1}
                      </span>
                      <span className="truncate font-medium text-[var(--text-primary)]">
                        {meta.label}
                      </span>
                    </div>
                    <div className="shrink-0 font-mono text-[11px] tabular-nums">
                      <span className="font-semibold text-[var(--text-primary)]">
                        {evt.latency_ms.toFixed(1)} ms
                      </span>
                      <span className="ml-1.5 text-[var(--text-muted)]">({share}%)</span>
                    </div>
                  </div>

                  {/* Proportional waterfall bar */}
                  <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-[var(--bg-hover)]">
                    <div
                      className="h-full rounded-full transition-all duration-300"
                      style={{ width: `${pct}%`, backgroundColor: meta.color }}
                    />
                  </div>

                  {evt.summary && (
                    <p className="mt-1.5 text-[11.5px] leading-snug text-[var(--text-secondary)]">
                      {evt.summary}
                    </p>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Tab 2: Dynamic Top-K & Rerank Chart                                        */
/* -------------------------------------------------------------------------- */

function DynamicKTab({
  result,
  onSelectChunk,
}: {
  result: FinalAnswer;
  onSelectChunk: (citationId: string) => void;
}) {
  const dk = result.dynamic_k_decision;
  const chunks = result.evidence_chunks || [];
  const tauMin = result.query_plan?.strategy?.tau_min;

  if (!dk && chunks.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-[var(--border-subtle)] p-6 text-center text-xs text-[var(--text-muted)]">
        Retrieval was skipped for this turn (conversational route).
      </div>
    );
  }

  const useRerank = chunks.some((c) => c.rerank_score !== null && c.rerank_score !== undefined);
  const scores = chunks.map((ch) =>
    useRerank && ch.rerank_score !== null
      ? ch.rerank_score
      : (ch.fusion_score ?? ch.dense_score ?? 0)
  );
  const maxVal = useRerank ? 1.0 : Math.max(...scores, 0.05) * 1.15;

  const chartHeight = 190;
  const topPad = 24;
  const bottomPad = 28;
  const plotHeight = chartHeight - topPad - bottomPad;
  const tauY =
    useRerank && tauMin !== undefined
      ? topPad + plotHeight * (1 - Math.min(1, Math.max(0, tauMin)))
      : null;

  return (
    <div className="space-y-4">
      {dk && (
        <>
          <div className="grid grid-cols-2 gap-2">
            <KpiTile
              label="Selected / Cand."
              value={`${dk.selected_count} / ${dk.candidate_count}`}
              tone="blue"
            />
            <KpiTile
              label="Top Score"
              value={dk.top_score.toFixed(3)}
              tone={dk.top_score >= 0.72 ? "emerald" : dk.top_score >= 0.38 ? "amber" : "rose"}
            />
            <KpiTile label="Cutoff Score" value={dk.cutoff_score.toFixed(3)} />
            <KpiTile
              label="Context Tokens"
              value={dk.total_context_tokens.toLocaleString("en-US")}
            />
          </div>
          <div className="flex items-center gap-2 rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60 px-3 py-2 text-xs">
            <Layers className="h-3.5 w-3.5 shrink-0 text-teal-400" />
            <span className="text-[var(--text-muted)]">Cutoff Reason:</span>
            <code className="font-mono text-[11px] font-medium text-[var(--text-primary)]">
              {dk.cutoff_reason}
            </code>
          </div>
        </>
      )}

      {chunks.length === 0 ? (
        <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 text-xs text-amber-400">
          No evidence chunks survived the dynamic top-k filter.
        </div>
      ) : (
        <div className="rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/55 p-3">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-[11px]">
            <span className="font-semibold text-[var(--text-primary)]">
              {useRerank ? "Calibrated Rerank Score (Sigmoid)" : "RRF / Retrieval Score"}
            </span>
            <div className="flex items-center gap-2.5 text-[10.5px] text-[var(--text-muted)]">
              <span className="inline-flex items-center gap-1">
                <span className="h-2 w-2 rounded-sm bg-teal-500" /> Local
              </span>
              <span className="inline-flex items-center gap-1">
                <span className="h-2 w-2 rounded-sm bg-purple-500" /> Web
              </span>
              {useRerank && tauMin !== undefined && (
                <span className="inline-flex items-center gap-1 font-mono font-semibold text-rose-500">
                  <span className="h-0.5 w-3 border-t-2 border-dashed border-rose-500" />
                  tau_min {tauMin.toFixed(2)}
                </span>
              )}
            </div>
          </div>

          <svg
            viewBox={`0 0 ${Math.max(320, chunks.length * 56 + 44)} ${chartHeight}`}
            className="h-52 w-full select-none"
            role="img"
            aria-label="Dynamic Top-K Passage Scores"
          >
            {/* Horizontal grid lines */}
            {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
              const y = topPad + plotHeight * (1 - tick);
              return (
                <g key={tick}>
                  <line
                    x1={28}
                    x2={Math.max(312, chunks.length * 56 + 34)}
                    y1={y}
                    y2={y}
                    stroke="currentColor"
                    strokeOpacity={0.09}
                    strokeWidth={1}
                  />
                  <text
                    x={24}
                    y={y + 3}
                    textAnchor="end"
                    fontSize={9}
                    className="fill-[var(--text-muted)] font-mono"
                  >
                    {(tick * maxVal).toFixed(useRerank ? 2 : 3)}
                  </text>
                </g>
              );
            })}

            {/* Bars */}
            {chunks.map((ch, i) => {
              const score = scores[i];
              const ratio = Math.min(1, Math.max(0, score / maxVal));
              const barHeight = Math.max(4, ratio * plotHeight);
              const barW = 32;
              const x = 42 + i * 56;
              const y = topPad + plotHeight - barHeight;
              const isWeb = ch.source_type === "web";
              const cid = ch.citation_id || `#${i + 1}`;

              return (
                <g
                  key={ch.chunk_id || i}
                  onClick={() => onSelectChunk(cid)}
                  className="cursor-pointer group"
                >
                  <title>{`[${cid}] ${ch.title || ch.doc_id} — Score: ${score.toFixed(4)} (click to inspect)`}</title>
                  <rect
                    x={x}
                    y={y}
                    width={barW}
                    height={barHeight}
                    rx={5}
                    fill={isWeb ? "#A855F7" : "#14B8A6"}
                    className="transition-opacity group-hover:opacity-80"
                  />
                  <text
                    x={x + barW / 2}
                    y={y - 6}
                    textAnchor="middle"
                    fontSize={9.5}
                    fontWeight={600}
                    className="fill-[var(--text-primary)] font-mono"
                  >
                    {score.toFixed(useRerank ? 2 : 3)}
                  </text>
                  <text
                    x={x + barW / 2}
                    y={chartHeight - 10}
                    textAnchor="middle"
                    fontSize={10}
                    fontWeight={600}
                    className="fill-[var(--text-secondary)] font-mono"
                  >
                    [{cid}]
                  </text>
                </g>
              );
            })}

            {/* tau_min threshold line (legend sits cleanly in header without colliding with bars) */}
            {tauY !== null && tauMin !== undefined && (
              <line
                x1={28}
                x2={Math.max(312, chunks.length * 56 + 34)}
                y1={tauY}
                y2={tauY}
                stroke="#F43F5E"
                strokeWidth={1.75}
                strokeDasharray="5 3"
              />
            )}
          </svg>
          <p className="mt-1 text-center text-[10.5px] text-[var(--text-muted)]">
            Click any passage bar to jump directly to its text in Evidence
          </p>
        </div>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Tab 3: Confidence & CRAG Gauge                                             */
/* -------------------------------------------------------------------------- */

function ConfidenceTab({ result }: { result: FinalAnswer }) {
  const conf = result.confidence;
  if (!conf) {
    return (
      <div className="rounded-xl border border-dashed border-[var(--border-subtle)] p-6 text-center text-xs text-[var(--text-muted)]">
        Confidence assessment was not run for this turn.
      </div>
    );
  }

  const score = Math.min(1, Math.max(0, conf.final_score));
  const arcColor =
    conf.level === "high" ? "#10B981" : conf.level === "medium" ? "#F59E0B" : "#F43F5E";
  const radius = 58;
  const circumference = Math.PI * radius;
  const strokeDashoffset = circumference * (1 - score);

  const signals = [
    { label: "Top-1 Rerank (w=0.50)", value: conf.top1_score, color: "#3B82F6" },
    { label: "Mean Top-K (w=0.30)", value: conf.mean_topk_score, color: "#14B8A6" },
    { label: "Sub-Query Coverage (w=0.20)", value: conf.subquery_coverage, color: "#6366F1" },
    { label: "Deterministic Composite", value: conf.deterministic_score, color: "#F59E0B" },
    { label: "Final Calibrated Score", value: conf.final_score, color: arcColor },
  ];

  return (
    <div className="space-y-4">
      {/* Semicircular Confidence Gauge Card */}
      <div className="rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60 p-4">
        <div className="flex flex-col items-center">
          <svg viewBox="0 0 160 108" className="h-30 w-44">
            <path
              d="M 22 78 A 58 58 0 0 1 138 78"
              fill="none"
              stroke="currentColor"
              strokeOpacity={0.14}
              strokeWidth={11}
              strokeLinecap="round"
            />
            <path
              d="M 22 78 A 58 58 0 0 1 138 78"
              fill="none"
              stroke={arcColor}
              strokeWidth={11}
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              className="transition-all duration-500"
            />
            <text
              x="80"
              y="66"
              textAnchor="middle"
              fontSize="24"
              fontWeight="700"
              className="fill-[var(--text-primary)] font-mono tabular-nums"
            >
              {conf.final_score.toFixed(2)}
            </text>
            <text
              x="80"
              y="98"
              textAnchor="middle"
              fontSize="9.5"
              fontWeight="600"
              fill={arcColor}
              className="uppercase tracking-widest"
            >
              {conf.level} CONFIDENCE
            </text>
          </svg>
        </div>

        <div className="mt-3 grid grid-cols-2 gap-2">
          <KpiTile
            label="CRAG Grade"
            value={(conf.crag_grade || "Not Needed").toUpperCase()}
            tone={
              conf.crag_grade === "sufficient"
                ? "emerald"
                : conf.crag_grade === "partial"
                  ? "amber"
                  : conf.crag_grade === "insufficient"
                    ? "rose"
                    : "neutral"
            }
          />
          <KpiTile
            label="Action"
            value={conf.recommended_action.replace(/_/g, " ").toUpperCase()}
            tone={conf.recommended_action === "proceed" ? "emerald" : "amber"}
          />
        </div>
      </div>

      {conf.reasoning && (
        <div className="rounded-xl border border-blue-500/25 bg-blue-500/[0.06] p-3 text-xs leading-relaxed text-[var(--text-secondary)]">
          <div className="mb-1 text-[10.5px] font-semibold uppercase tracking-wider text-blue-500 dark:text-blue-400">
            Gate Reasoning
          </div>
          {conf.reasoning}
        </div>
      )}

      {/* Multi-Signal Breakdown */}
      <div className="rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/55 p-3.5 space-y-3">
        <div className="flex items-center justify-between text-[11px]">
          <span className="font-semibold text-[var(--text-primary)]">
            Multi-Signal Breakdown
          </span>
          <span className="font-mono text-[10px] text-[var(--text-muted)]">
            low 0.38 · high 0.72
          </span>
        </div>

        <div className="relative space-y-2.5">
          {signals.map((sig) => (
            <div key={sig.label} className="space-y-1">
              <div className="flex items-center justify-between text-[11.5px]">
                <span className="text-[var(--text-secondary)]">{sig.label}</span>
                <span className="font-mono font-semibold text-[var(--text-primary)] tabular-nums">
                  {sig.value.toFixed(3)}
                </span>
              </div>
              <div className="relative h-2 w-full overflow-hidden rounded-full bg-[var(--bg-hover)]">
                {/* low (0.38) & high (0.72) threshold ticks */}
                <div
                  className="pointer-events-none absolute top-0 bottom-0 z-10 w-[1.5px] bg-rose-500/70"
                  style={{ left: "38%" }}
                  title="Low confidence threshold (0.38)"
                />
                <div
                  className="pointer-events-none absolute top-0 bottom-0 z-10 w-[1.5px] bg-emerald-500/70"
                  style={{ left: "72%" }}
                  title="High confidence threshold (0.72)"
                />
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${Math.min(100, Math.max(0, sig.value * 100))}%`,
                    backgroundColor: sig.color,
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Tab 4: Evidence Passages (with Click-to-Highlight)                         */
/* -------------------------------------------------------------------------- */

function EvidenceTab({
  result,
  highlightedCitationId,
  highlightToken,
}: {
  result: FinalAnswer;
  highlightedCitationId: string | null;
  highlightToken: number;
}) {
  const chunks = result.evidence_chunks || [];
  const [filter, setFilter] = useState<"all" | "local" | "web">("all");
  const [expandedIds, setExpandedIds] = useState<Set<string>>(() => {
    const initial = new Set<string>();
    chunks.slice(0, 2).forEach((ch, idx) => initial.add(ch.citation_id || `#${idx + 1}`));
    return initial;
  });
  const cardRefs = useRef<Map<string, HTMLDivElement>>(new Map());

  // Reset initial expanded cards when switching to a different turn
  useEffect(() => {
    const next = new Set<string>();
    chunks.slice(0, 2).forEach((ch, idx) => next.add(ch.citation_id || `#${idx + 1}`));
    if (highlightedCitationId) {
      next.add(highlightedCitationId);
    }
    setExpandedIds(next);
  }, [result, highlightedCitationId]);

  // Expand, reset filter if needed, and scroll smoothly to clicked citation
  useEffect(() => {
    if (!highlightedCitationId) return;
    setFilter("all");
    setExpandedIds((prev) => {
      const next = new Set(prev);
      next.add(highlightedCitationId);
      return next;
    });
    const timer = window.setTimeout(() => {
      const el = cardRefs.current.get(highlightedCitationId);
      el?.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 60);
    return () => window.clearTimeout(timer);
  }, [highlightedCitationId, highlightToken]);

  const filtered = useMemo(
    () =>
      chunks.filter((ch) =>
        filter === "all" ? true : ch.source_type.toLowerCase() === filter
      ),
    [chunks, filter]
  );

  if (chunks.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-[var(--border-subtle)] p-6 text-center text-xs text-[var(--text-muted)]">
        No evidence passages attached to this response.
      </div>
    );
  }

  const toggleExpand = (key: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const localCount = chunks.filter((c) => c.source_type === "local").length;
  const webCount = chunks.filter((c) => c.source_type === "web").length;

  return (
    <div className="space-y-3">
      {/* Filter pill bar */}
      <div className="flex items-center justify-between gap-2">
        <div className="inline-flex rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-elevated)] p-0.5 text-[11px]">
          {(
            [
              ["all", `All (${chunks.length})`],
              ["local", `Local (${localCount})`],
              ["web", `Web (${webCount})`],
            ] as const
          ).map(([val, label]) => (
            <button
              key={val}
              type="button"
              onClick={() => setFilter(val)}
              className={`rounded-md px-2.5 py-1 font-medium transition cursor-pointer ${
                filter === val
                  ? "bg-[var(--bg-panel)] text-[var(--text-primary)] shadow-xs"
                  : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {filtered.length === 0 ? (
        <div className="rounded-xl border border-dashed border-[var(--border-subtle)] p-5 text-center text-xs text-[var(--text-muted)]">
          No {filter} evidence passages in this turn.
        </div>
      ) : (
        <div className="space-y-2.5">
          {filtered.map((ch: EvidenceChunk, idx: number) => {
            const cid = ch.citation_id || `#${idx + 1}`;
            const isWeb = ch.source_type === "web";
            const isExpanded = expandedIds.has(cid);
            const isHighlighted = highlightedCitationId === cid;
            const scoreBadge =
              ch.rerank_score !== null && ch.rerank_score !== undefined
                ? `rerank ${ch.rerank_score.toFixed(3)}`
                : ch.fusion_score !== null && ch.fusion_score !== undefined
                  ? `fusion ${ch.fusion_score.toFixed(4)}`
                  : null;

            return (
              <div
                key={`${ch.chunk_id}-${highlightToken && isHighlighted ? highlightToken : "idle"}`}
                ref={(node) => {
                  if (node) cardRefs.current.set(cid, node);
                  else cardRefs.current.delete(cid);
                }}
                data-evidence-id={cid}
                className={`overflow-hidden rounded-xl border transition-all ${
                  isHighlighted
                    ? "citation-highlight border-blue-500 bg-blue-500/[0.06]"
                    : "border-[var(--border-subtle)] bg-[var(--bg-elevated)]/55"
                }`}
              >
                <button
                  type="button"
                  onClick={() => toggleExpand(cid)}
                  className="flex w-full items-start gap-2.5 px-3 py-2.5 text-left transition hover:bg-[var(--bg-hover)]/40 cursor-pointer"
                >
                  <span
                    className={`mt-0.5 shrink-0 rounded-md border px-1.5 py-0.5 font-mono text-[11px] font-semibold ${
                      isWeb
                        ? "border-purple-500/40 bg-purple-500/12 text-purple-600 dark:text-purple-300"
                        : "border-teal-500/40 bg-teal-500/12 text-teal-600 dark:text-teal-300"
                    }`}
                  >
                    {cid}
                  </span>

                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-2">
                      <span className="truncate text-xs font-semibold text-[var(--text-primary)]">
                        {ch.title || ch.doc_id || "Untitled Source"}
                      </span>
                      {scoreBadge && (
                        <span className="shrink-0 rounded bg-[var(--bg-panel)] px-1.5 py-0.5 font-mono text-[10px] text-[var(--text-secondary)] tabular-nums">
                          {scoreBadge}
                        </span>
                      )}
                    </div>
                    <div className="mt-0.5 flex items-center gap-1.5 text-[10.5px] text-[var(--text-muted)]">
                      {isWeb ? (
                        <Globe className="h-3 w-3 text-purple-400" />
                      ) : (
                        <FileText className="h-3 w-3 text-teal-400" />
                      )}
                      <span className="uppercase">{ch.source_type}</span>
                      {ch.section_path && (
                        <>
                          <span>·</span>
                          <span className="truncate">{ch.section_path}</span>
                        </>
                      )}
                    </div>
                  </div>

                  {isExpanded ? (
                    <ChevronDown className="mt-1 h-3.5 w-3.5 shrink-0 text-[var(--text-muted)]" />
                  ) : (
                    <ChevronRight className="mt-1 h-3.5 w-3.5 shrink-0 text-[var(--text-muted)]" />
                  )}
                </button>

                {isExpanded && (
                  <div className="border-t border-[var(--border-subtle)] bg-[var(--bg-panel)]/80 px-3 py-2.5 text-xs space-y-2">
                    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[10.5px] text-[var(--text-muted)]">
                      {ch.doc_id && (
                        <span>
                          Doc:{" "}
                          <code className="font-mono text-[var(--text-secondary)]">
                            {ch.doc_id}
                          </code>
                        </span>
                      )}
                      {ch.token_count > 0 && (
                        <span className="font-mono tabular-nums">
                          {ch.token_count} tokens
                        </span>
                      )}
                    </div>

                    {ch.url && (
                      <a
                        href={ch.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex max-w-full items-center gap-1 rounded-md border border-blue-500/25 bg-blue-500/8 px-2 py-1 text-[11px] font-medium text-blue-500 hover:bg-blue-500/15 dark:text-blue-400"
                      >
                        <ExternalLink className="h-3 w-3 shrink-0" />
                        <span className="truncate">{ch.url}</span>
                      </a>
                    )}

                    {ch.matched_sub_queries && ch.matched_sub_queries.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {ch.matched_sub_queries.map((mq, mIdx) => (
                          <span
                            key={mIdx}
                            className="rounded bg-[var(--bg-elevated)] px-1.5 py-0.5 font-mono text-[10px] text-[var(--text-secondary)]"
                          >
                            {mq}
                          </span>
                        ))}
                      </div>
                    )}

                    <div className="max-h-72 overflow-y-auto whitespace-pre-wrap rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-canvas)]/70 p-2.5 font-sans text-[12px] leading-relaxed text-[var(--text-primary)]">
                      {ch.text}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Tab 5: Two-Tier Citation Audit                                             */
/* -------------------------------------------------------------------------- */

function CitationAuditTab({
  result,
  onJumpToCitation,
}: {
  result: FinalAnswer;
  onJumpToCitation: (citationId: string) => void;
}) {
  const report = result.citation_report;
  if (!report) {
    return (
      <div className="rounded-xl border border-dashed border-[var(--border-subtle)] p-6 text-center text-xs text-[var(--text-muted)]">
        Citation validation was skipped for this turn.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Status Banner */}
      <div
        className={`flex items-center gap-2.5 rounded-xl border px-3.5 py-2.5 text-xs font-medium ${
          report.passed
            ? "border-emerald-500/35 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
            : "border-rose-500/35 bg-rose-500/10 text-rose-700 dark:text-rose-300"
        }`}
      >
        {report.passed ? (
          <CheckCircle2 className="h-4 w-4 shrink-0" />
        ) : (
          <AlertTriangle className="h-4 w-4 shrink-0" />
        )}
        <span>
          {report.passed
            ? "Two-Tier Grounding Audit Passed"
            : "Grounding Audit Flagged Unverified Claims"}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <KpiTile
          label="Citation Precision"
          value={`${(report.citation_precision * 100).toFixed(1)}%`}
          tone={report.citation_precision >= 0.9 ? "emerald" : "amber"}
        />
        <KpiTile
          label="Claim Support"
          value={`${(report.claim_support_rate * 100).toFixed(1)}%`}
          tone={report.claim_support_rate >= 0.8 ? "emerald" : "rose"}
        />
        <KpiTile
          label="Citations Found"
          value={String(report.total_citations_found)}
        />
        <KpiTile
          label="Fabricated IDs"
          value={String(report.invalid_citation_ids.length)}
          tone={report.invalid_citation_ids.length === 0 ? "emerald" : "rose"}
        />
      </div>

      {/* Tier-1 deterministic valid/invalid citation lists */}
      <div className="rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60 p-3 space-y-2.5">
        <div>
          <div className="text-[10.5px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
            Tier-1 Valid Citations
          </div>
          {report.valid_citation_ids.length > 0 ? (
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {report.valid_citation_ids.map((cid) => (
                <button
                  key={cid}
                  type="button"
                  onClick={() => onJumpToCitation(cid)}
                  className="rounded-md border border-emerald-500/35 bg-emerald-500/10 px-2 py-0.5 font-mono text-[11px] font-semibold text-emerald-600 transition hover:bg-emerald-500/20 dark:text-emerald-300 cursor-pointer"
                >
                  [{cid}]
                </button>
              ))}
            </div>
          ) : (
            <p className="mt-1 text-xs text-[var(--text-muted)]">None</p>
          )}
        </div>

        {report.invalid_citation_ids.length > 0 && (
          <div className="border-t border-[var(--border-subtle)] pt-2">
            <div className="text-[10.5px] font-semibold uppercase tracking-wider text-rose-500">
              Invalid / Fabricated Citation IDs
            </div>
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {report.invalid_citation_ids.map((cid) => (
                <span
                  key={cid}
                  className="rounded-md border border-rose-500/40 bg-rose-500/12 px-2 py-0.5 font-mono text-[11px] font-semibold text-rose-500"
                >
                  [{cid}]
                </span>
              ))}
            </div>
          </div>
        )}

        {report.uncited_factual_sentences.length > 0 && (
          <div className="border-t border-[var(--border-subtle)] pt-2">
            <div className="text-[10.5px] font-semibold uppercase tracking-wider text-amber-500">
              Uncited Factual Sentences ({report.uncited_factual_sentences.length})
            </div>
            <ul className="mt-1.5 space-y-1 text-xs text-[var(--text-secondary)]">
              {report.uncited_factual_sentences.map((sent, i) => (
                <li key={i} className="flex items-start gap-1.5">
                  <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-500" />
                  <span>{sent}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Tier-2 Semantic Entailment Claims */}
      <div className="space-y-2">
        <h4 className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
          Tier-2 Semantic Claim Entailment ({report.claim_verifications.length})
        </h4>
        {report.claim_verifications.length === 0 ? (
          <p className="rounded-xl border border-dashed border-[var(--border-subtle)] p-4 text-center text-xs text-[var(--text-muted)]">
            No Tier-2 entailment claims evaluated for this turn.
          </p>
        ) : (
          <div className="space-y-2">
            {report.claim_verifications.map((cv, idx) => {
              const verdictConfig =
                cv.verdict === "supported"
                  ? {
                      label: "Supported",
                      icon: CheckCircle2,
                      cls: "border-emerald-500/35 bg-emerald-500/12 text-emerald-600 dark:text-emerald-300",
                    }
                  : cv.verdict === "partially_supported"
                    ? {
                        label: "Partial",
                        icon: AlertTriangle,
                        cls: "border-amber-500/35 bg-amber-500/12 text-amber-600 dark:text-amber-300",
                      }
                    : cv.verdict === "contradicted"
                      ? {
                          label: "Contradicted",
                          icon: XCircle,
                          cls: "border-rose-500/35 bg-rose-500/12 text-rose-600 dark:text-rose-300",
                        }
                      : {
                          label: "Unsupported",
                          icon: AlertCircle,
                          cls: "border-rose-500/35 bg-rose-500/12 text-rose-600 dark:text-rose-300",
                        };
              const Icon = verdictConfig.icon;
              return (
                <div
                  key={idx}
                  className="rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/55 p-3 space-y-1.5"
                >
                  <div className="flex flex-wrap items-center justify-between gap-1.5">
                    <span
                      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10.5px] font-semibold ${verdictConfig.cls}`}
                    >
                      <Icon className="h-3 w-3" />
                      {verdictConfig.label}
                    </span>
                    <div className="flex flex-wrap gap-1">
                      {cv.cited_ids.map((cid) => (
                        <button
                          key={cid}
                          type="button"
                          onClick={() => onJumpToCitation(cid)}
                          className="rounded border border-blue-500/35 bg-blue-500/10 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-blue-500 hover:bg-blue-500/20 dark:text-blue-300 cursor-pointer"
                        >
                          [{cid}]
                        </button>
                      ))}
                    </div>
                  </div>
                  <p className="text-xs font-medium text-[var(--text-primary)] leading-snug">
                    {cv.claim}
                  </p>
                  {cv.explanation && (
                    <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">
                      {cv.explanation}
                    </p>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Root Inspector Shell                                                       */
/* -------------------------------------------------------------------------- */

export const GlassBoxInspector: React.FC<GlassBoxInspectorProps> = ({
  turns,
  selectedTurnIndex,
  onSelectTurnIndex,
  activeTab,
  onTabChange,
  highlightedCitationId,
  highlightToken,
  onJumpToCitation,
  onClose,
  selectorLabel = "Inspecting Turn",
}) => {
  const safeIdx =
    selectedTurnIndex >= 0 && selectedTurnIndex < turns.length
      ? selectedTurnIndex
      : turns.length - 1;
  const activeTurn = safeIdx >= 0 ? turns[safeIdx] : null;

  return (
    <aside className="flex h-full w-full flex-col border-l border-[var(--border-subtle)] bg-[var(--bg-panel)]">
      {/* Header */}
      <div className="flex h-13 shrink-0 items-center justify-between gap-2 border-b border-[var(--border-subtle)] px-4">
        <div className="flex min-w-0 items-center gap-2">
          <div className="flex h-6 w-6 items-center justify-center rounded-lg bg-blue-500/15 text-blue-500">
            <Activity className="h-3.5 w-3.5" />
          </div>
          <span className="truncate text-xs font-semibold tracking-tight text-[var(--text-primary)]">
            Glass-Box Inspector
          </span>
        </div>
        <button
          type="button"
          onClick={onClose}
          title="Close Inspector (⌘I)"
          className="rounded-lg p-1 text-[var(--text-muted)] transition hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)] cursor-pointer"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Turn Selector when > 1 turn */}
      {turns.length > 0 && (
        <div className="border-b border-[var(--border-subtle)] bg-[var(--bg-elevated)]/40 px-3 py-2">
          <div className="flex items-center justify-between gap-2">
            <span className="text-[10.5px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
              {selectorLabel}
            </span>
            {activeTurn && (
              <span className="rounded bg-[var(--bg-panel)] px-1.5 py-0.5 font-mono text-[10px] text-[var(--text-secondary)]">
                {activeTurn.presetLabel}
              </span>
            )}
          </div>
          <select
            value={safeIdx}
            onChange={(e) => onSelectTurnIndex(Number(e.target.value))}
            aria-label="Inspecting Conversation Turn"
            className="mt-1.5 w-full truncate rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-panel)] px-2.5 py-1.5 text-xs text-[var(--text-primary)] focus:border-blue-500 focus:outline-none"
          >
            {turns.map((turn, idx) => (
              <option key={turn.id} value={idx}>
                #{idx + 1} · {turn.query.slice(0, 62)}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* 5-Tab Bar */}
      <div className="grid shrink-0 grid-cols-5 border-b border-[var(--border-subtle)] bg-[var(--bg-panel)] px-1.5 pt-1.5">
        {TABS.map(({ id, label, icon: Icon }) => {
          const isActive = activeTab === id;
          const evidenceCount = activeTurn?.result.evidence_chunks.length ?? 0;
          const citePassed = activeTurn?.result.citation_report?.passed;
          return (
            <button
              key={id}
              type="button"
              onClick={() => onTabChange(id)}
              data-inspector-tab={id}
              className={`relative flex flex-col items-center gap-0.5 rounded-t-lg px-1 py-2 text-[11px] font-medium transition cursor-pointer ${
                isActive
                  ? "text-blue-600 dark:text-blue-400"
                  : "text-[var(--text-muted)] hover:text-[var(--text-primary)]"
              }`}
            >
              <div className="relative flex items-center justify-center">
                <Icon className="h-3.5 w-3.5" />
                {id === "evidence" && evidenceCount > 0 && (
                  <span className="ml-1 font-mono text-[9.5px] opacity-80">
                    {evidenceCount}
                  </span>
                )}
                {id === "citations" && citePassed !== undefined && (
                  <span
                    className={`ml-1 h-1.5 w-1.5 rounded-full ${
                      citePassed ? "bg-emerald-500" : "bg-rose-500"
                    }`}
                  />
                )}
              </div>
              <span>{label}</span>
              {isActive && (
                <span className="absolute right-2 bottom-0 left-2 h-0.5 rounded-full bg-blue-500" />
              )}
            </button>
          );
        })}
      </div>

      {/* Tab Body */}
      <div className="flex-1 overflow-y-auto p-3.5">
        {!activeTurn ? (
          <div className="flex h-full flex-col items-center justify-center px-6 text-center">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-[var(--border-subtle)] bg-[var(--bg-elevated)] text-[var(--text-muted)]">
              <Activity className="h-5 w-5" />
            </div>
            <p className="mt-3 text-xs font-medium text-[var(--text-primary)]">
              No Pipeline Trace Yet
            </p>
            <p className="mt-1 text-[11.5px] leading-relaxed text-[var(--text-muted)]">
              Send a query or click one of the bilingual demo prompts to inspect
              LangGraph node latencies, dynamic top-k cutoffs, CRAG confidence signals,
              and two-tier citation verification.
            </p>
          </div>
        ) : (
          <>
            {activeTab === "trace" && <TraceAndPlanTab result={activeTurn.result} />}
            {activeTab === "dynamic_k" && (
              <DynamicKTab
                result={activeTurn.result}
                onSelectChunk={onJumpToCitation}
              />
            )}
            {activeTab === "confidence" && <ConfidenceTab result={activeTurn.result} />}
            {activeTab === "evidence" && (
              <EvidenceTab
                result={activeTurn.result}
                highlightedCitationId={highlightedCitationId}
                highlightToken={highlightToken}
              />
            )}
            {activeTab === "citations" && (
              <CitationAuditTab
                result={activeTurn.result}
                onJumpToCitation={onJumpToCitation}
              />
            )}
          </>
        )}
      </div>
    </aside>
  );
};
