import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  BarChart3,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  ExternalLink,
  FileText,
  Gauge,
  Globe,
  ShieldCheck,
  X,
  XCircle,
  type LucideIcon,
} from "lucide-react";
import { getNodeMeta, STAGE_COLORS, STAGE_LABELS } from "../constants/pipeline";
import {
  capitalize,
  formatDuration,
  formatInt,
  formatPct,
  humanize,
  languageName,
} from "../lib/format";
import type {
  ChatTurn,
  EvidenceChunk,
  FinalAnswer,
  InspectorTab,
  StrategyProfile,
} from "../types/api";
import { CitationMark, isWebCitation } from "./ui/CitationMark";
import {
  Callout,
  cn,
  EmptyState,
  IconButton,
  SectionHeading,
  SELECT_CLASS,
  Segmented,
  Tag,
  type Tone,
} from "./ui/primitives";

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

const TABS: { id: InspectorTab; label: string; icon: LucideIcon }[] = [
  { id: "trace", label: "Trace", icon: Activity },
  { id: "dynamic_k", label: "Retrieval", icon: BarChart3 },
  { id: "confidence", label: "Confidence", icon: Gauge },
  { id: "evidence", label: "Evidence", icon: BookOpen },
  { id: "citations", label: "Audit", icon: ShieldCheck },
];

/** CRAG gate thresholds: below LOW rewrites or falls back, at or above HIGH proceeds. */
const GATE_LOW = 0.38;
const GATE_HIGH = 0.72;

const TONE_TEXT: Record<Tone, string> = {
  neutral: "text-ink",
  ok: "text-ok",
  warn: "text-warn",
  bad: "text-bad",
  accent: "text-accent",
};

/* -------------------------------------------------------------------------- */
/* Shared bits                                                                */
/* -------------------------------------------------------------------------- */

function LedgerGrid({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line">
      {children}
    </div>
  );
}

function LedgerCell({
  label,
  value,
  sub,
  tone = "neutral",
}: {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  tone?: Tone;
}) {
  return (
    <div className="min-w-0 bg-paper px-3 py-2.5">
      <div className="truncate text-caption text-ink-3">{label}</div>
      <div className={cn("mt-0.5 truncate text-body font-medium tabular-nums", TONE_TEXT[tone])}>
        {value}
      </div>
      {sub && <div className="mt-0.5 truncate text-caption text-ink-3">{sub}</div>}
    </div>
  );
}

function Block({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-caption text-ink-3">{label}</div>
      <div className="mt-1">{children}</div>
    </div>
  );
}

function chunkLabel(chunk: EvidenceChunk): string {
  const section = chunk.section_path?.split(/\s*[>›/]\s*/).filter(Boolean).pop();
  return section || chunk.title || chunk.doc_id || "Source";
}

/* -------------------------------------------------------------------------- */
/* Trace                                                                      */
/* -------------------------------------------------------------------------- */

const STRATEGY_LABELS: { key: keyof StrategyProfile; label: string; format?: (v: number) => string }[] =
  [
    { key: "max_sub_queries", label: "Max sub-queries" },
    { key: "dense_prefetch_limit", label: "Dense prefetch" },
    { key: "sparse_prefetch_limit", label: "Sparse prefetch" },
    { key: "fusion_limit", label: "Fusion limit" },
    { key: "rerank_min_k", label: "Min k" },
    { key: "rerank_max_k", label: "Max k" },
    { key: "tau_min", label: "τ min (relevance floor)", format: (v) => v.toFixed(2) },
    { key: "alpha_relative", label: "α (relative drop)", format: (v) => v.toFixed(2) },
    { key: "delta_gap", label: "δ (score gap)", format: (v) => v.toFixed(2) },
    { key: "max_context_tokens", label: "Context budget", format: (v) => `${formatInt(v)} tokens` },
  ];

function TraceTab({ result }: { result: FinalAnswer }) {
  const plan = result.query_plan;
  const trace = result.trace || [];
  const stepSum = trace.reduce((acc, e) => acc + Math.max(e.latency_ms, 0), 0);
  const totalMs = Math.max(result.total_latency_ms || 0, stepSum, 1);
  const starts = trace.reduce<number[]>((acc, _evt, i) => {
    acc.push(i === 0 ? 0 : acc[i - 1] + Math.max(trace[i - 1].latency_ms, 0));
    return acc;
  }, []);
  const stagesPresent = Array.from(new Set(trace.map((e) => getNodeMeta(e.node).stage)));

  return (
    <div className="space-y-7">
      {plan && (
        <section className="space-y-3">
          <SectionHeading>Query plan</SectionHeading>
          <LedgerGrid>
            <LedgerCell label="Route" value={capitalize(plan.query_type)} />
            <LedgerCell label="Language" value={languageName(plan.language)} />
            <LedgerCell
              label="Retrieval"
              value={plan.needs_retrieval ? "Used" : "Skipped"}
              tone={plan.needs_retrieval ? "neutral" : "warn"}
            />
            <LedgerCell
              label="Fresh web info"
              value={plan.requires_fresh_web_info ? "Needed" : "Not needed"}
              tone={plan.requires_fresh_web_info ? "warn" : "neutral"}
            />
          </LedgerGrid>

          <div className="space-y-3.5 rounded-lg border border-line bg-paper p-3.5">
            <Block label="Standalone question">
              <p className="text-ui text-ink">{plan.standalone_query}</p>
            </Block>
            {plan.reasoning && (
              <Block label="Why this route">
                <p className="text-ui text-ink-2">{plan.reasoning}</p>
              </Block>
            )}
            {plan.sub_queries && plan.sub_queries.length > 0 && (
              <Block label={`Sub-queries (${plan.sub_queries.length})`}>
                <ol className="list-decimal space-y-1 pl-5 text-ui text-ink marker:text-ink-3">
                  {plan.sub_queries.map((sq, i) => (
                    <li key={i} className="pl-0.5">
                      {sq}
                    </li>
                  ))}
                </ol>
              </Block>
            )}
          </div>

          {plan.strategy && (
            <details className="group rounded-lg border border-line bg-paper">
              <summary className="flex cursor-pointer list-none items-center justify-between gap-2 rounded-lg px-3.5 py-2.5 text-ui font-medium text-ink hover:tint [&::-webkit-details-marker]:hidden">
                Strategy parameters
                <ChevronDown
                  className="h-4 w-4 text-ink-3 transition-transform group-open:rotate-180"
                  aria-hidden
                />
              </summary>
              <dl className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1.5 border-t border-line px-3.5 py-3 text-small">
                {STRATEGY_LABELS.map(({ key, label, format }) => {
                  const raw = plan.strategy[key];
                  const value =
                    typeof raw === "number" ? (format ? format(raw) : formatInt(raw)) : String(raw);
                  return (
                    <React.Fragment key={key}>
                      <dt className="truncate text-ink-3" title={key}>
                        {label}
                      </dt>
                      <dd className="text-right font-medium tabular-nums text-ink">{value}</dd>
                    </React.Fragment>
                  );
                })}
              </dl>
            </details>
          )}
        </section>
      )}

      <section className="space-y-3">
        <SectionHeading
          aside={
            <span className="text-small tabular-nums text-ink-3">{formatDuration(totalMs)}</span>
          }
        >
          Timeline
        </SectionHeading>

        {trace.length === 0 ? (
          <EmptyState icon={Activity} title="No steps were recorded for this answer" />
        ) : (
          <>
            <ol className="rounded-lg border border-line bg-paper py-1">
              {trace.map((evt, idx) => {
                const meta = getNodeMeta(evt.node);
                const left = (starts[idx] / totalMs) * 100;
                const width = Math.max(0.8, (Math.max(evt.latency_ms, 0) / totalMs) * 100);
                return (
                  <li key={`${evt.node}-${idx}`} className="px-3.5 py-2">
                    <div className="flex items-baseline justify-between gap-3">
                      <span className="flex min-w-0 items-center gap-2 text-ui text-ink">
                        <span
                          aria-hidden
                          className="h-2 w-2 shrink-0 rounded-full"
                          style={{ backgroundColor: meta.color }}
                        />
                        <span className="truncate">{meta.label}</span>
                      </span>
                      <span className="shrink-0 text-small tabular-nums text-ink-2">
                        {formatDuration(evt.latency_ms)}
                      </span>
                    </div>
                    <div className="relative mt-1.5 h-1.5 rounded-full bg-sunken">
                      <span
                        className="absolute inset-y-0 rounded-full"
                        style={{
                          left: `${Math.min(left, 99.2)}%`,
                          width: `${Math.min(width, 100 - Math.min(left, 99.2))}%`,
                          backgroundColor: meta.color,
                        }}
                      />
                    </div>
                    {evt.summary && (
                      <p className="mt-1 text-small text-ink-3">{evt.summary}</p>
                    )}
                  </li>
                );
              })}
            </ol>
            <div className="flex flex-wrap gap-x-4 gap-y-1 text-caption text-ink-3">
              {stagesPresent.map((stage) => (
                <span key={stage} className="inline-flex items-center gap-1.5">
                  <span
                    aria-hidden
                    className="h-2 w-2 rounded-full"
                    style={{ backgroundColor: STAGE_COLORS[stage] }}
                  />
                  {STAGE_LABELS[stage]}
                </span>
              ))}
            </div>
          </>
        )}
      </section>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Retrieval (dynamic top-k)                                                  */
/* -------------------------------------------------------------------------- */

function RetrievalTab({
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
      <EmptyState icon={BarChart3} title="Retrieval was skipped">
        This question took the conversational route, so no passages were searched.
      </EmptyState>
    );
  }

  const useRerank = chunks.some((c) => c.rerank_score !== null && c.rerank_score !== undefined);
  const scores = chunks.map((ch) =>
    useRerank && ch.rerank_score !== null && ch.rerank_score !== undefined
      ? ch.rerank_score
      : (ch.fusion_score ?? ch.dense_score ?? 0)
  );
  const maxVal = useRerank ? 1 : Math.max(...scores, 0.05) * 1.15;
  const tauPct =
    useRerank && tauMin !== undefined ? Math.min(100, Math.max(0, tauMin * 100)) : null;

  return (
    <div className="space-y-7">
      {dk && (
        <section className="space-y-3">
          <SectionHeading>Dynamic top-k</SectionHeading>
          <LedgerGrid>
            <LedgerCell
              label="Kept"
              value={`${dk.selected_count} of ${dk.candidate_count}`}
              sub="passages"
            />
            <LedgerCell
              label="Top score"
              value={dk.top_score.toFixed(3)}
              tone={dk.top_score >= GATE_HIGH ? "ok" : dk.top_score >= GATE_LOW ? "warn" : "bad"}
            />
            <LedgerCell label="Cutoff score" value={dk.cutoff_score.toFixed(3)} />
            <LedgerCell
              label="Context"
              value={formatInt(dk.total_context_tokens)}
              sub="tokens sent to the model"
            />
          </LedgerGrid>
          <p className="text-small text-ink-2">
            Cut off by the{" "}
            <code className="rounded-[4px] border border-line bg-paper px-1 font-mono text-[11.5px] text-ink">
              {dk.cutoff_reason}
            </code>{" "}
            rule.
            {dk.parent_expansion_applied && (
              <>
                {" "}
                {dk.collapsed_child_count ?? 0} child chunks were expanded into{" "}
                {dk.expanded_parent_count ?? 0} parent sections.
              </>
            )}
          </p>
        </section>
      )}

      {chunks.length === 0 ? (
        <Callout tone="warn" title="No passages passed the cutoff">
          The answer was written without local evidence.
        </Callout>
      ) : (
        <section className="space-y-3">
          <SectionHeading
            aside={
              <span className="text-caption text-ink-3">
                {useRerank ? "Calibrated 0 to 1" : "Reciprocal rank fusion"}
              </span>
            }
          >
            {useRerank ? "Cross-encoder scores" : "Fusion scores"}
          </SectionHeading>
          <ul className="rounded-lg border border-line bg-paper py-1">
            {chunks.map((ch, i) => {
              const cid = ch.citation_id || `#${i + 1}`;
              const ratio = Math.min(1, Math.max(0, scores[i] / maxVal));
              return (
                <li key={ch.chunk_id || i}>
                  <button
                    type="button"
                    onClick={() => onSelectChunk(cid)}
                    title={`${ch.title || ch.doc_id}${ch.section_path ? ` › ${ch.section_path}` : ""}`}
                    className="grid w-full cursor-pointer grid-cols-[2.5rem_minmax(0,1fr)_3rem] items-center gap-x-2.5 px-3 py-2 text-left transition-colors hover:bg-hover"
                  >
                    <CitationMark citationId={cid} web={isWebCitation(cid, ch.source_type)} static />
                    <span className="min-w-0">
                      <span className="block truncate text-small text-ink-2">{chunkLabel(ch)}</span>
                      <span className="relative mt-1 block h-1.5 rounded-full bg-sunken">
                        <span
                          className="absolute inset-y-0 left-0 rounded-full bg-ink-3"
                          style={{ width: `${Math.max(2, ratio * 100)}%` }}
                        />
                        {tauPct !== null && (
                          <span
                            aria-hidden
                            className="absolute -top-1 -bottom-1 w-0 border-l border-dashed border-bad"
                            style={{ left: `${tauPct}%` }}
                          />
                        )}
                      </span>
                    </span>
                    <span className="text-right text-small font-medium tabular-nums text-ink">
                      {scores[i].toFixed(useRerank ? 2 : 3)}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
          {tauPct !== null && tauMin !== undefined && (
            <p className="flex items-center gap-2 text-caption text-ink-3">
              <span aria-hidden className="h-3 w-0 border-l border-dashed border-bad" />
              Relevance floor τ = {tauMin.toFixed(2)}. Passages below it are dropped.
            </p>
          )}
        </section>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Confidence (CRAG gate)                                                     */
/* -------------------------------------------------------------------------- */

const LEVEL_TONE: Record<string, Tone> = { high: "ok", medium: "warn", low: "bad" };
const GRADE_TONE: Record<string, Tone> = { sufficient: "ok", partial: "warn", insufficient: "bad" };

function GateScale({ score }: { score: number }) {
  const pct = Math.min(100, Math.max(0, score * 100));
  return (
    <div className="mt-5">
      <div className="relative h-2.5">
        <div className="absolute inset-0 flex gap-px overflow-hidden rounded-full">
          <span className="h-full bg-bad/25" style={{ width: `${GATE_LOW * 100}%` }} />
          <span
            className="h-full bg-warn/30"
            style={{ width: `${(GATE_HIGH - GATE_LOW) * 100}%` }}
          />
          <span className="h-full flex-1 bg-ok/30" />
        </div>
        <span
          aria-hidden
          className="absolute -top-1.5 -bottom-1.5 w-[3px] -translate-x-1/2 rounded-full bg-ink ring-2 ring-paper"
          style={{ left: `${pct}%` }}
        />
      </div>
      <div className="relative mt-2 h-4 text-caption tabular-nums text-ink-3">
        <span className="absolute left-0">0</span>
        <span className="absolute -translate-x-1/2" style={{ left: `${GATE_LOW * 100}%` }}>
          {GATE_LOW}
        </span>
        <span className="absolute -translate-x-1/2" style={{ left: `${GATE_HIGH * 100}%` }}>
          {GATE_HIGH}
        </span>
        <span className="absolute right-0">1</span>
      </div>
      <div className="mt-1 grid grid-cols-[38fr_34fr_28fr] gap-2 text-caption text-ink-3">
        <span>Rewrite or fall back</span>
        <span>Grader decides</span>
        <span>Proceed</span>
      </div>
    </div>
  );
}

function ConfidenceTab({ result }: { result: FinalAnswer }) {
  const conf = result.confidence;
  if (!conf) {
    return (
      <EmptyState icon={Gauge} title="Confidence wasn't scored">
        The preset used for this answer skips the confidence gate.
      </EmptyState>
    );
  }

  const tone = LEVEL_TONE[conf.level] ?? "neutral";
  const toneVar =
    tone === "ok" ? "var(--ok)" : tone === "warn" ? "var(--warn)" : tone === "bad" ? "var(--bad)" : "var(--ink-3)";

  const signals = [
    { label: "Top-1 rerank score", weight: "× 0.50", value: conf.top1_score },
    { label: "Mean top-k score", weight: "× 0.30", value: conf.mean_topk_score },
    { label: "Sub-query coverage", weight: "× 0.20", value: conf.subquery_coverage },
    { label: "Deterministic composite", weight: "", value: conf.deterministic_score },
    { label: "Final calibrated score", weight: "", value: conf.final_score, highlight: true },
  ];

  return (
    <div className="space-y-7">
      <section className="rounded-lg border border-line bg-paper p-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="text-caption text-ink-3">Calibrated confidence</div>
            <div className="mt-1 flex items-center gap-2.5">
              <span className="text-display font-semibold tabular-nums text-ink">
                {conf.final_score.toFixed(2)}
              </span>
              <Tag tone={tone}>{capitalize(conf.level)}</Tag>
            </div>
          </div>
          <div className="text-right">
            <div className="text-caption text-ink-3">Gate decision</div>
            <div className="mt-1 text-body font-medium text-ink">
              {humanize(conf.recommended_action)}
            </div>
          </div>
        </div>
        <GateScale score={conf.final_score} />
      </section>

      <LedgerGrid>
        <LedgerCell
          label="CRAG grader"
          value={conf.crag_grade ? capitalize(conf.crag_grade) : "Not needed"}
          sub={conf.crag_grade ? "LLM verdict" : "Score was outside the grey zone"}
          tone={conf.crag_grade ? (GRADE_TONE[conf.crag_grade] ?? "neutral") : "neutral"}
        />
        <LedgerCell
          label="Next step"
          value={humanize(conf.recommended_action)}
          tone={conf.recommended_action === "proceed" ? "ok" : "warn"}
        />
      </LedgerGrid>

      {conf.reasoning && (
        <Block label="Reasoning">
          <p className="text-ui text-ink-2">{conf.reasoning}</p>
        </Block>
      )}

      <section className="space-y-3">
        <SectionHeading>Signals</SectionHeading>
        <ul className="space-y-3 rounded-lg border border-line bg-paper p-3.5">
          {signals.map((sig) => (
            <li key={sig.label}>
              <div className="flex items-baseline justify-between gap-3 text-small">
                <span className={sig.highlight ? "font-medium text-ink" : "text-ink-2"}>
                  {sig.label}
                  {sig.weight && <span className="ml-1.5 text-ink-3">{sig.weight}</span>}
                </span>
                <span className="font-medium tabular-nums text-ink">{sig.value.toFixed(3)}</span>
              </div>
              <div className="mt-1 h-1.5 rounded-full bg-sunken">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${Math.min(100, Math.max(0, sig.value * 100))}%`,
                    backgroundColor: sig.highlight ? toneVar : "var(--ink-3)",
                  }}
                />
              </div>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Evidence                                                                   */
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
  const cardRefs = useRef<Map<string, HTMLLIElement>>(new Map());

  useEffect(() => {
    const next = new Set<string>();
    chunks.slice(0, 2).forEach((ch, idx) => next.add(ch.citation_id || `#${idx + 1}`));
    if (highlightedCitationId) next.add(highlightedCitationId);
    setExpandedIds(next);
  }, [result, highlightedCitationId]);

  useEffect(() => {
    if (!highlightedCitationId) return;
    setFilter("all");
    setExpandedIds((prev) => new Set(prev).add(highlightedCitationId));
    const timer = window.setTimeout(() => {
      cardRefs.current
        .get(highlightedCitationId)
        ?.scrollIntoView?.({ behavior: "smooth", block: "start" });
    }, 60);
    return () => window.clearTimeout(timer);
  }, [highlightedCitationId, highlightToken]);

  const filtered = useMemo(
    () => chunks.filter((ch) => (filter === "all" ? true : ch.source_type.toLowerCase() === filter)),
    [chunks, filter]
  );

  if (chunks.length === 0) {
    return (
      <EmptyState icon={BookOpen} title="No passages are attached to this answer">
        Either retrieval was skipped or nothing passed the relevance cutoff.
      </EmptyState>
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
      <Segmented
        ariaLabel="Filter passages by source"
        value={filter}
        onChange={setFilter}
        options={[
          { value: "all", label: "All", count: chunks.length },
          { value: "local", label: "Documents", count: localCount },
          { value: "web", label: "Web", count: webCount },
        ]}
      />

      {filtered.length === 0 ? (
        <EmptyState
          title={filter === "web" ? "No web passages in this answer" : "No document passages in this answer"}
        />
      ) : (
        <ul className="space-y-2">
          {filtered.map((ch: EvidenceChunk, idx: number) => {
            const cid = ch.citation_id || `#${idx + 1}`;
            const web = isWebCitation(cid, ch.source_type);
            const isExpanded = expandedIds.has(cid);
            const isHighlighted = highlightedCitationId === cid;
            const score =
              ch.rerank_score !== null && ch.rerank_score !== undefined
                ? `Relevance ${ch.rerank_score.toFixed(2)}`
                : ch.fusion_score !== null && ch.fusion_score !== undefined
                  ? `RRF ${ch.fusion_score.toFixed(4)}`
                  : null;

            return (
              <li
                key={`${ch.chunk_id}-${isHighlighted && highlightToken ? highlightToken : "idle"}`}
                ref={(el) => {
                  if (el) cardRefs.current.set(cid, el);
                  else cardRefs.current.delete(cid);
                }}
                data-evidence-id={cid}
                className={cn(
                  "scroll-mt-3 overflow-hidden rounded-lg border bg-paper",
                  isHighlighted ? "evidence-flash border-mark-strong" : "border-line"
                )}
              >
                <button
                  type="button"
                  aria-expanded={isExpanded}
                  onClick={() => toggleExpand(cid)}
                  className="flex w-full cursor-pointer items-start gap-2.5 px-3 py-2.5 text-left transition-colors hover:bg-hover"
                >
                  <CitationMark citationId={cid} web={web} static size="md" />
                  <span className="min-w-0 flex-1">
                    <span className="flex items-baseline justify-between gap-2">
                      <span className="truncate text-ui font-medium text-ink">
                        {ch.title || ch.doc_id || "Untitled source"}
                      </span>
                      {score && (
                        <span className="shrink-0 text-caption tabular-nums text-ink-3">{score}</span>
                      )}
                    </span>
                    <span className="mt-0.5 flex items-center gap-1.5 text-small text-ink-3">
                      {web ? (
                        <Globe className="h-3 w-3 shrink-0" aria-hidden />
                      ) : (
                        <FileText className="h-3 w-3 shrink-0" aria-hidden />
                      )}
                      <span className="truncate">
                        {ch.section_path || (web ? "Web result" : "Document passage")}
                      </span>
                    </span>
                  </span>
                  <ChevronDown
                    className={cn(
                      "mt-0.5 h-4 w-4 shrink-0 text-ink-3 transition-transform",
                      isExpanded && "rotate-180"
                    )}
                    aria-hidden
                  />
                </button>

                {isExpanded && (
                  <div className="border-t border-line px-3 pt-2.5 pb-3">
                    <div className="flex flex-wrap gap-1.5">
                      {ch.doc_id && (
                        <Tag mono title="Document ID">
                          {ch.doc_id}
                        </Tag>
                      )}
                      {ch.token_count > 0 && <Tag>{formatInt(ch.token_count)} tokens</Tag>}
                      {ch.expanded_to_parent && (
                        <Tag tone="accent">
                          Expanded to parent section from {ch.matched_child_ids?.length || 1} child
                          chunks
                        </Tag>
                      )}
                    </div>

                    {ch.url && (
                      <a
                        href={ch.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="mt-2 inline-flex max-w-full items-center gap-1 text-small text-accent hover:underline"
                      >
                        <ExternalLink className="h-3 w-3 shrink-0" aria-hidden />
                        <span className="truncate">{ch.url}</span>
                      </a>
                    )}

                    {ch.matched_sub_queries && ch.matched_sub_queries.length > 0 && (
                      <div className="mt-2.5">
                        <div className="text-caption text-ink-3">Matched sub-queries</div>
                        <ul className="mt-1 space-y-0.5 text-small text-ink-2">
                          {ch.matched_sub_queries.map((mq, mIdx) => (
                            <li key={mIdx} className="truncate" title={mq}>
                              {mq}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    <div className="prose-passage mt-3 max-h-80 overflow-y-auto border-l-2 border-mark-strong pl-3">
                      {ch.text}
                    </div>
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Citation audit                                                             */
/* -------------------------------------------------------------------------- */

const VERDICTS: Record<string, { label: string; tone: Tone; icon: LucideIcon }> = {
  supported: { label: "Supported", tone: "ok", icon: CheckCircle2 },
  partially_supported: { label: "Partly supported", tone: "warn", icon: AlertTriangle },
  contradicted: { label: "Contradicted", tone: "bad", icon: XCircle },
  unsupported: { label: "Unsupported", tone: "bad", icon: AlertCircle },
};

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
      <EmptyState icon={ShieldCheck} title="The citation audit didn't run">
        The preset used for this answer skips citation checks.
      </EmptyState>
    );
  }

  return (
    <div className="space-y-7">
      <Callout
        tone={report.passed ? "ok" : "bad"}
        title={report.passed ? "Citation audit passed" : "Citation audit flagged problems"}
      >
        {report.passed
          ? "Every cited ID exists and the checked claims are backed by their passages."
          : "Some claims aren't backed by the passages they cite, or cite passages that don't exist."}
      </Callout>

      <LedgerGrid>
        <LedgerCell
          label="Citation precision"
          value={formatPct(report.citation_precision, 1)}
          tone={report.citation_precision >= 0.9 ? "ok" : "warn"}
        />
        <LedgerCell
          label="Claim support"
          value={formatPct(report.claim_support_rate, 1)}
          tone={report.claim_support_rate >= 0.8 ? "ok" : "bad"}
        />
        <LedgerCell label="Citations found" value={report.total_citations_found} />
        <LedgerCell
          label="Fabricated IDs"
          value={report.invalid_citation_ids.length}
          tone={report.invalid_citation_ids.length === 0 ? "neutral" : "bad"}
        />
      </LedgerGrid>

      <section className="space-y-2.5">
        <SectionHeading>Verified citations</SectionHeading>
        {report.valid_citation_ids.length > 0 ? (
          <div className="flex flex-wrap gap-1.5">
            {report.valid_citation_ids.map((cid) => (
              <CitationMark
                key={cid}
                citationId={cid}
                web={isWebCitation(cid)}
                size="md"
                onClick={() => onJumpToCitation(cid)}
                title={`Show passage ${cid}`}
              />
            ))}
          </div>
        ) : (
          <p className="text-small text-ink-3">None of the citations could be matched to a passage.</p>
        )}
      </section>

      {report.invalid_citation_ids.length > 0 && (
        <section className="space-y-2.5">
          <SectionHeading>Citations that don't exist</SectionHeading>
          <div className="flex flex-wrap gap-1.5">
            {report.invalid_citation_ids.map((cid) => (
              <Tag key={cid} tone="bad">
                {cid}
              </Tag>
            ))}
          </div>
        </section>
      )}

      {report.uncited_factual_sentences.length > 0 && (
        <section className="space-y-2.5">
          <SectionHeading>
            Factual sentences without a citation ({report.uncited_factual_sentences.length})
          </SectionHeading>
          <ul className="space-y-2 rounded-lg border border-line bg-paper p-3.5">
            {report.uncited_factual_sentences.map((sent, i) => (
              <li key={i} className="flex items-start gap-2">
                <AlertTriangle className="mt-1 h-3.5 w-3.5 shrink-0 text-warn" aria-hidden />
                <span className="font-serif text-ui leading-relaxed text-ink">{sent}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="space-y-2.5">
        <SectionHeading>Claim checks ({report.claim_verifications.length})</SectionHeading>
        {report.claim_verifications.length === 0 ? (
          <EmptyState title="No claims were checked for this answer" />
        ) : (
          <ul className="divide-y divide-line overflow-hidden rounded-lg border border-line bg-paper">
            {report.claim_verifications.map((cv, idx) => {
              const verdict = VERDICTS[cv.verdict] ?? VERDICTS.unsupported;
              return (
                <li key={idx} className="space-y-1.5 p-3.5">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <Tag tone={verdict.tone} icon={verdict.icon}>
                      {verdict.label}
                    </Tag>
                    <div className="flex flex-wrap gap-1">
                      {cv.cited_ids.map((cid) => (
                        <CitationMark
                          key={cid}
                          citationId={cid}
                          web={isWebCitation(cid)}
                          onClick={() => onJumpToCitation(cid)}
                          title={`Show passage ${cid}`}
                        />
                      ))}
                    </div>
                  </div>
                  <p className="font-serif text-body leading-relaxed text-ink">{cv.claim}</p>
                  {cv.explanation && <p className="text-small text-ink-3">{cv.explanation}</p>}
                </li>
              );
            })}
          </ul>
        )}
      </section>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Inspector shell                                                            */
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
  selectorLabel = "Question to inspect",
}) => {
  const safeIdx =
    selectedTurnIndex >= 0 && selectedTurnIndex < turns.length ? selectedTurnIndex : turns.length - 1;
  const activeTurn = safeIdx >= 0 ? turns[safeIdx] : null;
  const tabRefs = useRef<Map<InspectorTab, HTMLButtonElement>>(new Map());

  const onTabKeyDown = (e: React.KeyboardEvent<HTMLButtonElement>, idx: number) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    e.preventDefault();
    const nextIdx = (idx + (e.key === "ArrowRight" ? 1 : -1) + TABS.length) % TABS.length;
    const nextTab = TABS[nextIdx].id;
    onTabChange(nextTab);
    tabRefs.current.get(nextTab)?.focus();
  };

  return (
    <aside aria-label="Inspector" className="flex h-full w-full flex-col bg-desk">
      <div className="flex h-12 shrink-0 items-center justify-between gap-2 border-b border-line px-3.5">
        <div className="flex min-w-0 items-center gap-2">
          <h2 className="text-ui font-semibold text-ink">Inspector</h2>
          {activeTurn && <span className="truncate text-small text-ink-3">{activeTurn.presetLabel}</span>}
        </div>
        <IconButton icon={X} label="Close inspector" title="Close inspector (⌘I)" onClick={onClose} />
      </div>

      {turns.length > 0 && (
        <div className="border-b border-line px-3.5 py-2.5">
          <select
            value={safeIdx}
            onChange={(e) => onSelectTurnIndex(Number(e.target.value))}
            aria-label={selectorLabel}
            className={cn(SELECT_CLASS, "bg-raised")}
          >
            {turns.map((turn, idx) => (
              <option key={turn.id} value={idx}>
                {idx + 1}. {turn.query.slice(0, 72)}
              </option>
            ))}
          </select>
        </div>
      )}

      <div
        role="tablist"
        aria-label="Inspector sections"
        className="flex h-10 shrink-0 items-stretch gap-0.5 overflow-x-auto border-b border-line px-2"
      >
        {TABS.map(({ id, label, icon: Icon }, idx) => {
          const isActive = activeTab === id;
          const evidenceCount = activeTurn?.result.evidence_chunks.length ?? 0;
          const citePassed = activeTurn?.result.citation_report?.passed;
          return (
            <button
              key={id}
              ref={(el) => {
                if (el) tabRefs.current.set(id, el);
              }}
              type="button"
              role="tab"
              id={`inspector-tab-${id}`}
              aria-selected={isActive}
              aria-controls="inspector-panel"
              tabIndex={isActive ? 0 : -1}
              data-inspector-tab={id}
              onClick={() => onTabChange(id)}
              onKeyDown={(e) => onTabKeyDown(e, idx)}
              className={cn(
                "relative inline-flex shrink-0 cursor-pointer items-center gap-1.5 px-2 text-small font-medium transition-colors",
                isActive ? "text-ink" : "text-ink-3 hover:text-ink"
              )}
            >
              <Icon className="h-3.5 w-3.5 shrink-0" aria-hidden />
              <span>{label}</span>
              {id === "evidence" && evidenceCount > 0 && (
                <span className="tabular-nums text-ink-3">{evidenceCount}</span>
              )}
              {id === "citations" && citePassed !== undefined && (
                <span
                  aria-label={citePassed ? "passed" : "flagged"}
                  className={cn("h-1.5 w-1.5 shrink-0 rounded-full", citePassed ? "bg-ok" : "bg-bad")}
                />
              )}
              {isActive && (
                <span aria-hidden className="absolute inset-x-1.5 bottom-0 h-0.5 rounded-full bg-accent" />
              )}
            </button>
          );
        })}
      </div>

      <div
        id="inspector-panel"
        role="tabpanel"
        aria-labelledby={`inspector-tab-${activeTab}`}
        className="flex-1 overflow-y-auto p-4"
      >
        {!activeTurn ? (
          <div className="flex h-full flex-col items-center justify-center px-6 text-center">
            <Activity className="h-5 w-5 text-ink-4" aria-hidden />
            <p className="mt-3 text-ui font-medium text-ink">Nothing to inspect yet</p>
            <p className="mt-1 max-w-[17rem] text-small text-ink-3">
              Ask a question to see its route, retrieval scores, confidence gate and citation audit
              here.
            </p>
          </div>
        ) : (
          <>
            {activeTab === "trace" && <TraceTab result={activeTurn.result} />}
            {activeTab === "dynamic_k" && (
              <RetrievalTab result={activeTurn.result} onSelectChunk={onJumpToCitation} />
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
              <CitationAuditTab result={activeTurn.result} onJumpToCitation={onJumpToCitation} />
            )}
          </>
        )}
      </div>
    </aside>
  );
};
