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

const TABS: {
  id: InspectorTab;
  label: string;
  icon: React.FC<{ className?: string }>;
}[] = [
  { id: "trace", label: "Trace", icon: Activity },
  { id: "dynamic_k", label: "Top-K", icon: BarChart3 },
  { id: "confidence", label: "CRAG", icon: Gauge },
  { id: "evidence", label: "Evidence", icon: BookOpen },
  { id: "citations", label: "Audit", icon: ShieldCheck },
];

function LedgerCell({
  label,
  value,
  subtext,
  tone = "neutral",
}: {
  label: string;
  value: string;
  subtext?: string;
  tone?: "neutral" | "emerald" | "amber" | "rose" | "sky";
}) {
  const valueTone =
    tone === "emerald"
      ? "text-emerald-600 dark:text-emerald-400"
      : tone === "amber"
        ? "text-amber-600 dark:text-amber-400"
        : tone === "rose"
          ? "text-rose-600 dark:text-rose-400"
          : tone === "sky"
            ? "text-sky-600 dark:text-sky-400"
            : "text-[var(--text-primary)]";

  return (
    <div className="bg-[var(--bg-surface)] px-3 py-2.5">
      <div className="font-mono text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
        {label}
      </div>
      <div className={`mt-1 font-mono text-sm font-semibold tabular-nums ${valueTone}`}>
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
          <div className="font-mono text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
            Query Plan
          </div>

          <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-[var(--border-subtle)] bg-[var(--border-subtle)]">
            <LedgerCell label="Route" value={plan.query_type.toUpperCase()} />
            <LedgerCell
              label="Language"
              value={plan.language.toUpperCase()}
              subtext={plan.needs_retrieval ? "Retrieval active" : "Direct response"}
            />
            <LedgerCell
              label="Retrieval"
              value={plan.needs_retrieval ? "Enabled" : "Skipped"}
              tone={plan.needs_retrieval ? "emerald" : "neutral"}
            />
            <LedgerCell
              label="Fresh Web"
              value={plan.requires_fresh_web_info ? "Required" : "No"}
              tone={plan.requires_fresh_web_info ? "amber" : "neutral"}
            />
          </div>

          <div className="divide-y divide-[var(--border-subtle)] overflow-hidden rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-surface)]">
            <div className="p-3">
              <div className="font-mono text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
                Standalone Query
              </div>
              <p className="mt-1 font-mono text-[11.5px] leading-relaxed text-[var(--text-primary)]">
                {plan.standalone_query}
              </p>
            </div>

            {plan.reasoning && (
              <div className="p-3">
                <div className="font-mono text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
                  Router Rationale
                </div>
                <p className="mt-1 text-xs leading-relaxed text-[var(--text-secondary)]">
                  {plan.reasoning}
                </p>
              </div>
            )}

            {plan.sub_queries && plan.sub_queries.length > 0 && (
              <div className="p-3">
                <div className="font-mono text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
                  Sub-Queries ({plan.sub_queries.length})
                </div>
                <ol className="mt-1.5 space-y-1">
                  {plan.sub_queries.map((sq, i) => (
                    <li
                      key={i}
                      className="flex items-start gap-2 rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/50 px-2 py-1 font-mono text-[11px] text-[var(--text-primary)]"
                    >
                      <span className="shrink-0 text-[var(--text-muted)]">
                        {i + 1}.
                      </span>
                      <span>{sq}</span>
                    </li>
                  ))}
                </ol>
              </div>
            )}
          </div>

          {plan.strategy && (
            <div className="overflow-hidden rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-surface)]">
              <button
                type="button"
                onClick={() => setShowStrategy((prev) => !prev)}
                className="flex w-full items-center justify-between px-3 py-2 text-xs font-medium text-[var(--text-secondary)] transition hover:text-[var(--text-primary)] cursor-pointer"
              >
                <span className="inline-flex items-center gap-1.5 font-mono text-[11px]">
                  <Code2 className="h-3.5 w-3.5 text-[var(--text-muted)]" />
                  <span>strategy_params.json</span>
                </span>
                {showStrategy ? (
                  <ChevronDown className="h-3.5 w-3.5" />
                ) : (
                  <ChevronRight className="h-3.5 w-3.5" />
                )}
              </button>
              {showStrategy && (
                <pre className="overflow-x-auto border-t border-[var(--border-subtle)] bg-[#0c0c0f] p-3 font-mono text-[11px] text-zinc-300">
                  {JSON.stringify(plan.strategy, null, 2)}
                </pre>
              )}
            </div>
          )}
        </section>
      )}

      {/* Latency Waterfall */}
      <section className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="font-mono text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
            Node Waterfall
          </span>
          <span className="font-mono text-[10.5px] text-[var(--text-secondary)] tabular-nums">
            {Math.round(totalMs).toLocaleString("en-US")} ms
          </span>
        </div>

        {trace.length === 0 ? (
          <div className="rounded-lg border border-dashed border-[var(--border-subtle)] p-4 text-center text-xs text-[var(--text-muted)]">
            No node trace events recorded for this turn.
          </div>
        ) : (
          <div className="divide-y divide-[var(--border-subtle)] overflow-hidden rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-surface)]">
            {trace.map((evt, idx) => {
              const meta = getNodeMeta(evt.node);
              const pct = Math.min(
                100,
                Math.max(2, (evt.latency_ms / maxStepMs) * 100)
              );
              const share = Math.round((evt.latency_ms / totalMs) * 100);
              return (
                <div key={`${evt.node}-${idx}`} className="p-2.5">
                  <div className="flex items-center justify-between gap-2 text-xs">
                    <div className="flex min-w-0 items-center gap-2">
                      <span className="w-4 shrink-0 font-mono text-[10px] text-[var(--text-muted)] tabular-nums">
                        {String(idx + 1).padStart(2, "0")}
                      </span>
                      <span
                        className="h-1.5 w-1.5 shrink-0 rounded-full"
                        style={{ backgroundColor: meta.color }}
                      />
                      <span className="truncate font-medium text-[var(--text-primary)]">
                        {meta.label}
                      </span>
                    </div>
                    <div className="shrink-0 font-mono text-[10.5px] tabular-nums">
                      <span className="font-medium text-[var(--text-primary)]">
                        {evt.latency_ms.toFixed(1)}ms
                      </span>
                      <span className="ml-1.5 text-[var(--text-muted)]">
                        {share}%
                      </span>
                    </div>
                  </div>

                  <div className="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-[var(--bg-elevated)]">
                    <div
                      className="h-full rounded-full transition-all duration-300"
                      style={{ width: `${pct}%`, backgroundColor: meta.color }}
                    />
                  </div>

                  {evt.summary && (
                    <p className="mt-1.5 text-[11px] leading-snug text-[var(--text-secondary)]">
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
      <div className="rounded-lg border border-dashed border-[var(--border-subtle)] p-6 text-center text-xs text-[var(--text-muted)]">
        Retrieval was skipped for this turn (conversational route).
      </div>
    );
  }

  const useRerank = chunks.some(
    (c) => c.rerank_score !== null && c.rerank_score !== undefined
  );
  const scores = chunks.map((ch) =>
    useRerank && ch.rerank_score !== null
      ? ch.rerank_score
      : (ch.fusion_score ?? ch.dense_score ?? 0)
  );
  const maxVal = useRerank ? 1.0 : Math.max(...scores, 0.05) * 1.15;

  const chartHeight = 184;
  const topPad = 22;
  const bottomPad = 26;
  const plotHeight = chartHeight - topPad - bottomPad;
  const tauY =
    useRerank && tauMin !== undefined
      ? topPad + plotHeight * (1 - Math.min(1, Math.max(0, tauMin)))
      : null;

  return (
    <div className="space-y-3.5">
      {dk && (
        <>
          <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-[var(--border-subtle)] bg-[var(--border-subtle)]">
            <LedgerCell
              label="Selected / Pool"
              value={`${dk.selected_count} / ${dk.candidate_count}`}
            />
            <LedgerCell
              label="Top Score"
              value={dk.top_score.toFixed(3)}
              tone={
                dk.top_score >= 0.72
                  ? "emerald"
                  : dk.top_score >= 0.38
                    ? "amber"
                    : "rose"
              }
            />
            <LedgerCell label="Cutoff Score" value={dk.cutoff_score.toFixed(3)} />
            <LedgerCell
              label="Context Budget"
              value={`${dk.total_context_tokens.toLocaleString("en-US")} tok`}
            />
          </div>

          <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-surface)] px-3 py-2 text-xs">
            <div className="flex items-center gap-2">
              <Layers className="h-3.5 w-3.5 shrink-0 text-[var(--text-muted)]" />
              <span className="text-[var(--text-muted)]">Cutoff:</span>
              <code className="font-mono text-[11px] font-medium text-[var(--text-primary)]">
                {dk.cutoff_reason}
              </code>
            </div>
            {dk.parent_expansion_applied && (
              <span className="inline-flex items-center gap-1 rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)] px-1.5 py-0.5 font-mono text-[10px] font-medium text-[var(--text-secondary)]">
                Small→Big: {dk.collapsed_child_count ?? 0}→{dk.expanded_parent_count ?? 0}
              </span>
            )}
          </div>
        </>
      )}

      {chunks.length === 0 ? (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3.5 text-xs text-amber-600 dark:text-amber-300">
          No evidence chunks survived the dynamic top-k filter.
        </div>
      ) : (
        <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-surface)] p-3">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-[11px]">
            <span className="font-medium text-[var(--text-primary)]">
              {useRerank ? "Cross-Encoder Score" : "RRF Fusion Score"}
            </span>
            <div className="flex items-center gap-2.5 font-mono text-[10px] text-[var(--text-muted)]">
              <span className="inline-flex items-center gap-1">
                <span className="h-2 w-2 rounded-[2px] bg-sky-500" /> local
              </span>
              <span className="inline-flex items-center gap-1">
                <span className="h-2 w-2 rounded-[2px] bg-amber-500" /> web
              </span>
              {useRerank && tauMin !== undefined && (
                <span className="inline-flex items-center gap-1 text-rose-500">
                  <span className="h-px w-2.5 border-t border-dashed border-rose-500" />
                  τ={tauMin.toFixed(2)}
                </span>
              )}
            </div>
          </div>

          <svg
            viewBox={`0 0 ${Math.max(320, chunks.length * 52 + 44)} ${chartHeight}`}
            className="h-48 w-full select-none"
            role="img"
            aria-label="Dynamic Top-K Passage Scores"
          >
            {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
              const y = topPad + plotHeight * (1 - tick);
              return (
                <g key={tick}>
                  <line
                    x1={28}
                    x2={Math.max(312, chunks.length * 52 + 34)}
                    y1={y}
                    y2={y}
                    stroke="currentColor"
                    strokeOpacity={0.08}
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

            {chunks.map((ch, i) => {
              const score = scores[i];
              const ratio = Math.min(1, Math.max(0, score / maxVal));
              const barHeight = Math.max(4, ratio * plotHeight);
              const barW = 26;
              const x = 42 + i * 52;
              const y = topPad + plotHeight - barHeight;
              const isWeb = ch.source_type === "web";
              const cid = ch.citation_id || `#${i + 1}`;

              return (
                <g
                  key={ch.chunk_id || i}
                  onClick={() => onSelectChunk(cid)}
                  className="group cursor-pointer"
                >
                  <title>{`[${cid}] ${ch.title || ch.doc_id} — ${score.toFixed(4)}`}</title>
                  <rect
                    x={x}
                    y={y}
                    width={barW}
                    height={barHeight}
                    rx={3}
                    fill={isWeb ? "#f59e0b" : "#38bdf8"}
                    fillOpacity={0.85}
                    className="transition-opacity group-hover:opacity-100"
                  />
                  <text
                    x={x + barW / 2}
                    y={y - 5}
                    textAnchor="middle"
                    fontSize={9}
                    fontWeight={500}
                    className="fill-[var(--text-secondary)] font-mono"
                  >
                    {score.toFixed(useRerank ? 2 : 3)}
                  </text>
                  <text
                    x={x + barW / 2}
                    y={chartHeight - 9}
                    textAnchor="middle"
                    fontSize={9.5}
                    fontWeight={600}
                    className="fill-[var(--text-muted)] font-mono"
                  >
                    {cid}
                  </text>
                </g>
              );
            })}

            {tauY !== null && tauMin !== undefined && (
              <line
                x1={28}
                x2={Math.max(312, chunks.length * 52 + 34)}
                y1={tauY}
                y2={tauY}
                stroke="#f43f5e"
                strokeWidth={1.25}
                strokeDasharray="4 3"
              />
            )}
          </svg>
        </div>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Tab 3: Confidence & CRAG Calibration                                       */
/* -------------------------------------------------------------------------- */

function ConfidenceTab({ result }: { result: FinalAnswer }) {
  const conf = result.confidence;
  if (!conf) {
    return (
      <div className="rounded-lg border border-dashed border-[var(--border-subtle)] p-6 text-center text-xs text-[var(--text-muted)]">
        Confidence assessment was not run for this turn.
      </div>
    );
  }

  const score = Math.min(1, Math.max(0, conf.final_score));
  const barColor =
    conf.level === "high"
      ? "#10b981"
      : conf.level === "medium"
        ? "#f59e0b"
        : "#f43f5e";

  const signals = [
    { label: "Top-1 Rerank (0.50)", value: conf.top1_score },
    { label: "Mean Top-K (0.30)", value: conf.mean_topk_score },
    { label: "Sub-Query Coverage (0.20)", value: conf.subquery_coverage },
    { label: "Deterministic Composite", value: conf.deterministic_score },
    { label: "Final Calibrated Score", value: conf.final_score, highlight: true },
  ];

  return (
    <div className="space-y-3.5">
      {/* Linear Confidence Readout Card */}
      <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-surface)] p-3.5">
        <div className="flex items-baseline justify-between gap-2">
          <div>
            <div className="font-mono text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
              Calibrated Confidence
            </div>
            <div className="mt-1 flex items-baseline gap-2">
              <span className="font-mono text-2xl font-semibold tracking-tight text-[var(--text-primary)] tabular-nums">
                {conf.final_score.toFixed(2)}
              </span>
              <span
                className="font-mono text-[10.5px] font-semibold uppercase tracking-wider"
                style={{ color: barColor }}
              >
                {conf.level}
              </span>
            </div>
          </div>
          <span className="font-mono text-[10px] text-[var(--text-muted)]">
            gates: 0.38 / 0.72
          </span>
        </div>

        {/* Threshold Meter */}
        <div className="relative mt-3 h-2 w-full overflow-hidden rounded-full bg-[var(--bg-elevated)]">
          <div
            className="pointer-events-none absolute top-0 bottom-0 z-10 w-px bg-rose-500/75"
            style={{ left: "38%" }}
            title="Low threshold (0.38)"
          />
          <div
            className="pointer-events-none absolute top-0 bottom-0 z-10 w-px bg-emerald-500/75"
            style={{ left: "72%" }}
            title="High threshold (0.72)"
          />
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{ width: `${score * 100}%`, backgroundColor: barColor }}
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-[var(--border-subtle)] bg-[var(--border-subtle)]">
        <LedgerCell
          label="CRAG Evaluator"
          value={(conf.crag_grade || "Bypassed").toUpperCase()}
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
        <LedgerCell
          label="Gate Action"
          value={conf.recommended_action.replace(/_/g, " ").toUpperCase()}
          tone={conf.recommended_action === "proceed" ? "emerald" : "amber"}
        />
      </div>

      {conf.reasoning && (
        <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-surface)] p-3 text-xs leading-relaxed text-[var(--text-secondary)]">
          <div className="mb-1 font-mono text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
            Gate Reasoning
          </div>
          {conf.reasoning}
        </div>
      )}

      {/* Signal Breakdown */}
      <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-surface)] p-3 space-y-2.5">
        <div className="font-mono text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
          Signal Decomposition
        </div>

        <div className="space-y-2">
          {signals.map((sig) => (
            <div key={sig.label} className="space-y-1">
              <div className="flex items-center justify-between text-[11px]">
                <span
                  className={
                    sig.highlight
                      ? "font-medium text-[var(--text-primary)]"
                      : "text-[var(--text-secondary)]"
                  }
                >
                  {sig.label}
                </span>
                <span className="font-mono text-[11px] font-medium text-[var(--text-primary)] tabular-nums">
                  {sig.value.toFixed(3)}
                </span>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-[var(--bg-elevated)]">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${Math.min(100, Math.max(0, sig.value * 100))}%`,
                    backgroundColor: sig.highlight ? barColor : "#71717a",
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
/* Tab 4: Evidence Passages                                                   */
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

  useEffect(() => {
    const next = new Set<string>();
    chunks.slice(0, 2).forEach((ch, idx) => next.add(ch.citation_id || `#${idx + 1}`));
    if (highlightedCitationId) {
      next.add(highlightedCitationId);
    }
    setExpandedIds(next);
  }, [result, highlightedCitationId]);

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
      <div className="rounded-lg border border-dashed border-[var(--border-subtle)] p-6 text-center text-xs text-[var(--text-muted)]">
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
      <div className="inline-flex rounded-md border border-[var(--border-subtle)] bg-[var(--bg-surface)] p-0.5 text-[11px]">
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
            className={`rounded px-2.5 py-1 font-medium transition cursor-pointer ${
              filter === val
                ? "bg-[var(--bg-elevated)] text-[var(--text-primary)]"
                : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {filtered.length === 0 ? (
        <div className="rounded-lg border border-dashed border-[var(--border-subtle)] p-5 text-center text-xs text-[var(--text-muted)]">
          No {filter} evidence passages in this turn.
        </div>
      ) : (
        <div className="space-y-2">
          {filtered.map((ch: EvidenceChunk, idx: number) => {
            const cid = ch.citation_id || `#${idx + 1}`;
            const isWeb = ch.source_type === "web";
            const isExpanded = expandedIds.has(cid);
            const isHighlighted = highlightedCitationId === cid;
            const scoreBadge =
              ch.rerank_score !== null && ch.rerank_score !== undefined
                ? `rerank ${ch.rerank_score.toFixed(3)}`
                : ch.fusion_score !== null && ch.fusion_score !== undefined
                  ? `rrf ${ch.fusion_score.toFixed(4)}`
                  : null;

            return (
              <div
                key={`${ch.chunk_id}-${highlightToken && isHighlighted ? highlightToken : "idle"}`}
                ref={(node) => {
                  if (node) cardRefs.current.set(cid, node);
                  else cardRefs.current.delete(cid);
                }}
                data-evidence-id={cid}
                className={`overflow-hidden rounded-lg border transition-all ${
                  isHighlighted
                    ? "citation-highlight border-sky-500/70 bg-[var(--bg-surface)]"
                    : "border-[var(--border-subtle)] bg-[var(--bg-surface)]"
                }`}
              >
                <button
                  type="button"
                  onClick={() => toggleExpand(cid)}
                  className="flex w-full items-start gap-2.5 px-3 py-2.5 text-left transition hover:bg-[var(--bg-elevated)]/40 cursor-pointer"
                >
                  <span
                    className={`mt-0.5 shrink-0 rounded border px-1.5 py-0.5 font-mono text-[10.5px] font-semibold ${
                      isWeb
                        ? "border-amber-500/35 bg-amber-500/10 text-amber-700 dark:text-amber-300"
                        : "border-sky-500/35 bg-sky-500/10 text-sky-700 dark:text-sky-300"
                    }`}
                  >
                    {cid}
                  </span>

                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-2">
                      <span className="truncate text-xs font-medium text-[var(--text-primary)]">
                        {ch.title || ch.doc_id || "Untitled Source"}
                      </span>
                      {scoreBadge && (
                        <span className="shrink-0 font-mono text-[10px] text-[var(--text-muted)] tabular-nums">
                          {scoreBadge}
                        </span>
                      )}
                    </div>
                    <div className="mt-0.5 flex items-center gap-1.5 font-mono text-[10px] text-[var(--text-muted)]">
                      {isWeb ? (
                        <Globe className="h-2.5 w-2.5" />
                      ) : (
                        <FileText className="h-2.5 w-2.5" />
                      )}
                      <span className="uppercase">{ch.source_type}</span>
                      {ch.section_path && (
                        <>
                          <span>·</span>
                          <span className="truncate font-sans">{ch.section_path}</span>
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
                  <div className="space-y-2 border-t border-[var(--border-subtle)] bg-[var(--bg-canvas)]/50 px-3 py-2.5 text-xs">
                    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[10px] text-[var(--text-muted)]">
                      {ch.doc_id && <span>doc:{ch.doc_id}</span>}
                      {ch.token_count > 0 && (
                        <span className="tabular-nums">{ch.token_count} tok</span>
                      )}
                      {ch.expanded_to_parent && (
                        <span className="rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)] px-1.5 py-0.5 text-[10px] font-medium text-[var(--text-secondary)]">
                          Parent Section ({ch.matched_child_ids?.length || 1} child)
                        </span>
                      )}
                    </div>

                    {ch.url && (
                      <a
                        href={ch.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex max-w-full items-center gap-1 rounded border border-[var(--border-subtle)] bg-[var(--bg-surface)] px-2 py-1 font-mono text-[10.5px] text-[var(--text-secondary)] transition hover:border-[var(--border-strong)] hover:text-[var(--text-primary)]"
                      >
                        <ExternalLink className="h-2.5 w-2.5 shrink-0" />
                        <span className="truncate">{ch.url}</span>
                      </a>
                    )}

                    {ch.matched_sub_queries && ch.matched_sub_queries.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {ch.matched_sub_queries.map((mq, mIdx) => (
                          <span
                            key={mIdx}
                            className="rounded border border-[var(--border-subtle)] bg-[var(--bg-surface)] px-1.5 py-0.5 font-mono text-[10px] text-[var(--text-secondary)]"
                          >
                            {mq}
                          </span>
                        ))}
                      </div>
                    )}

                    <div className="max-h-72 overflow-y-auto whitespace-pre-wrap rounded border border-[var(--border-subtle)] bg-[var(--bg-surface)] p-2.5 font-sans text-[12px] leading-relaxed text-[var(--text-primary)]">
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
      <div className="rounded-lg border border-dashed border-[var(--border-subtle)] p-6 text-center text-xs text-[var(--text-muted)]">
        Citation validation was skipped for this turn.
      </div>
    );
  }

  return (
    <div className="space-y-3.5">
      <div
        className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-xs font-medium ${
          report.passed
            ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
            : "border-rose-500/30 bg-rose-500/10 text-rose-700 dark:text-rose-300"
        }`}
      >
        {report.passed ? (
          <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
        ) : (
          <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
        )}
        <span>
          {report.passed
            ? "Two-Tier Grounding Audit Passed"
            : "Grounding Audit Flagged Unverified Claims"}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-[var(--border-subtle)] bg-[var(--border-subtle)]">
        <LedgerCell
          label="Precision"
          value={`${(report.citation_precision * 100).toFixed(1)}%`}
          tone={report.citation_precision >= 0.9 ? "emerald" : "amber"}
        />
        <LedgerCell
          label="Claim Support"
          value={`${(report.claim_support_rate * 100).toFixed(1)}%`}
          tone={report.claim_support_rate >= 0.8 ? "emerald" : "rose"}
        />
        <LedgerCell
          label="Citations Found"
          value={String(report.total_citations_found)}
        />
        <LedgerCell
          label="Fabricated IDs"
          value={String(report.invalid_citation_ids.length)}
          tone={report.invalid_citation_ids.length === 0 ? "emerald" : "rose"}
        />
      </div>

      {/* Tier-1 Deterministic Check */}
      <div className="divide-y divide-[var(--border-subtle)] rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-surface)]">
        <div className="p-3">
          <div className="font-mono text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
            Tier-1 Verified IDs
          </div>
          {report.valid_citation_ids.length > 0 ? (
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {report.valid_citation_ids.map((cid) => (
                <button
                  key={cid}
                  type="button"
                  onClick={() => onJumpToCitation(cid)}
                  className="rounded border border-sky-500/35 bg-sky-500/10 px-1.5 py-0.5 font-mono text-[10.5px] font-semibold text-sky-700 transition hover:bg-sky-500/20 dark:text-sky-300 cursor-pointer"
                >
                  {cid}
                </button>
              ))}
            </div>
          ) : (
            <p className="mt-1 text-xs text-[var(--text-muted)]">None</p>
          )}
        </div>

        {report.invalid_citation_ids.length > 0 && (
          <div className="p-3">
            <div className="font-mono text-[10px] font-semibold uppercase tracking-wider text-rose-500">
              Fabricated Citation IDs
            </div>
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {report.invalid_citation_ids.map((cid) => (
                <span
                  key={cid}
                  className="rounded border border-rose-500/35 bg-rose-500/10 px-1.5 py-0.5 font-mono text-[10.5px] font-semibold text-rose-500"
                >
                  {cid}
                </span>
              ))}
            </div>
          </div>
        )}

        {report.uncited_factual_sentences.length > 0 && (
          <div className="p-3">
            <div className="font-mono text-[10px] font-semibold uppercase tracking-wider text-amber-500">
              Uncited Sentences ({report.uncited_factual_sentences.length})
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
        <div className="font-mono text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
          Tier-2 Claim Entailment ({report.claim_verifications.length})
        </div>
        {report.claim_verifications.length === 0 ? (
          <p className="rounded-lg border border-dashed border-[var(--border-subtle)] p-4 text-center text-xs text-[var(--text-muted)]">
            No Tier-2 entailment claims evaluated for this turn.
          </p>
        ) : (
          <div className="divide-y divide-[var(--border-subtle)] overflow-hidden rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-surface)]">
            {report.claim_verifications.map((cv, idx) => {
              const verdictConfig =
                cv.verdict === "supported"
                  ? {
                      label: "Supported",
                      icon: CheckCircle2,
                      cls: "text-emerald-600 dark:text-emerald-400",
                    }
                  : cv.verdict === "partially_supported"
                    ? {
                        label: "Partial",
                        icon: AlertTriangle,
                        cls: "text-amber-600 dark:text-amber-400",
                      }
                    : cv.verdict === "contradicted"
                      ? {
                          label: "Contradicted",
                          icon: XCircle,
                          cls: "text-rose-600 dark:text-rose-400",
                        }
                      : {
                          label: "Unsupported",
                          icon: AlertCircle,
                          cls: "text-rose-600 dark:text-rose-400",
                        };
              const Icon = verdictConfig.icon;
              return (
                <div key={idx} className="space-y-1.5 p-3">
                  <div className="flex flex-wrap items-center justify-between gap-1.5">
                    <span
                      className={`inline-flex items-center gap-1 font-mono text-[10.5px] font-semibold uppercase ${verdictConfig.cls}`}
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
                          className="rounded border border-sky-500/35 bg-sky-500/10 px-1.25 py-0.5 font-mono text-[10px] font-semibold text-sky-700 hover:bg-sky-500/20 dark:text-sky-300 cursor-pointer"
                        >
                          {cid}
                        </button>
                      ))}
                    </div>
                  </div>
                  <p className="text-xs font-medium leading-snug text-[var(--text-primary)]">
                    {cv.claim}
                  </p>
                  {cv.explanation && (
                    <p className="text-[11px] leading-relaxed text-[var(--text-secondary)]">
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
      <div className="flex h-12 shrink-0 items-center justify-between gap-2 border-b border-[var(--border-subtle)] px-3.5">
        <div className="flex min-w-0 items-center gap-2">
          <span className="truncate text-xs font-semibold tracking-tight text-[var(--text-primary)]">
            Inspector
          </span>
          {activeTurn && (
            <span className="rounded border border-[var(--border-subtle)] bg-[var(--bg-surface)] px-1.5 py-0.5 font-mono text-[10px] text-[var(--text-muted)]">
              {activeTurn.presetLabel}
            </span>
          )}
        </div>
        <button
          type="button"
          onClick={onClose}
          title="Close Inspector (⌘I)"
          className="rounded-md p-1 text-[var(--text-muted)] transition hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)] cursor-pointer"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* Turn Selector */}
      {turns.length > 0 && (
        <div className="border-b border-[var(--border-subtle)] px-3 py-2">
          <div className="mb-1 font-mono text-[9.5px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
            {selectorLabel}
          </div>
          <select
            value={safeIdx}
            onChange={(e) => onSelectTurnIndex(Number(e.target.value))}
            aria-label="Inspecting Conversation Turn"
            className="ar-select h-7.5 w-full truncate rounded-md border border-[var(--border-subtle)] bg-[var(--bg-surface)] px-2 text-xs text-[var(--text-primary)] focus:border-[var(--text-muted)] focus:outline-none"
          >
            {turns.map((turn, idx) => (
              <option key={turn.id} value={idx}>
                #{idx + 1} · {turn.query.slice(0, 62)}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Horizontal DevTools 5-Tab Strip */}
      <div className="grid h-9 shrink-0 grid-cols-5 border-b border-[var(--border-subtle)] bg-[var(--bg-panel)] px-1">
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
              className={`relative flex items-center justify-center gap-1 px-1 text-[11px] font-medium transition cursor-pointer ${
                isActive
                  ? "text-[var(--text-primary)]"
                  : "text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
              }`}
            >
              <Icon className="h-3 w-3 shrink-0" />
              <span className="truncate">{label}</span>
              {id === "evidence" && evidenceCount > 0 && (
                <span className="font-mono text-[9.5px] text-[var(--text-muted)] tabular-nums">
                  {evidenceCount}
                </span>
              )}
              {id === "citations" && citePassed !== undefined && (
                <span
                  className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                    citePassed ? "bg-emerald-500" : "bg-rose-500"
                  }`}
                />
              )}
              {isActive && (
                <span className="absolute right-1.5 bottom-0 left-1.5 h-[1.5px] bg-[var(--text-primary)]" />
              )}
            </button>
          );
        })}
      </div>

      {/* Tab Body */}
      <div className="flex-1 overflow-y-auto p-3.5">
        {!activeTurn ? (
          <div className="flex h-full flex-col items-center justify-center px-6 text-center">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-surface)] text-[var(--text-muted)]">
              <Activity className="h-4 w-4" />
            </div>
            <p className="mt-3 text-xs font-medium text-[var(--text-primary)]">
              No Trace Selected
            </p>
            <p className="mt-1 text-[11.5px] leading-relaxed text-[var(--text-muted)]">
              Execute a query to inspect node waterfall timings, Dynamic Top-K
              thresholds, CRAG calibration, and evidence entailment.
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
