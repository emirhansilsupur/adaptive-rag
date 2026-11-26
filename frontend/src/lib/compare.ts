import { ARENA_PRESET_OPTIONS } from "../constants/pipeline";
import { getAuditState, isAuditClear, type AuditState } from "./audit";
import type {
  AblationPreset,
  ChatTurn,
  CompareRun,
  ConfidenceLevel,
  FinalAnswer,
} from "../types/api";

export interface CompareDeltaSummary {
  leftContextTokens: number;
  rightContextTokens: number;
  contextTokensDelta: number;
  contextTokensPct: number | null;
  leftK: number;
  rightK: number;
  kDelta: number;
  leftConfidence: number | null;
  rightConfidence: number | null;
  leftConfidenceLevel: ConfidenceLevel | null;
  rightConfidenceLevel: ConfidenceLevel | null;
  leftAudit: AuditState | null;
  rightAudit: AuditState | null;
  leftClaimSupport: number | null;
  rightClaimSupport: number | null;
  leftInvalidCitations: number;
  rightInvalidCitations: number;
  leftLatencyMs: number;
  rightLatencyMs: number;
  latencyDeltaMs: number;
  leftCostUsd: number;
  rightCostUsd: number;
  costDeltaUsd: number;
  costPct: number | null;
  highlights: { label: string; tone: "emerald" | "blue" | "purple" | "amber" | "rose" }[];
}

export function getPresetShortLabel(preset: AblationPreset): string {
  return (
    ARENA_PRESET_OPTIONS.find((opt) => opt.value === preset)?.shortLabel ?? preset
  );
}

export function getPresetLabel(preset: AblationPreset): string {
  return ARENA_PRESET_OPTIONS.find((opt) => opt.value === preset)?.label ?? preset;
}

export function getContextTokens(result: FinalAnswer): number {
  if (
    result.dynamic_k_decision &&
    typeof result.dynamic_k_decision.total_context_tokens === "number"
  ) {
    return result.dynamic_k_decision.total_context_tokens;
  }
  return (result.evidence_chunks || []).reduce(
    (sum, ch) => sum + (ch.token_count || 0),
    0
  );
}

export function formatSignedPct(pct: number | null): string {
  if (pct === null || !Number.isFinite(pct)) return "—";
  const rounded = Math.round(pct);
  if (rounded === 0) return "0%";
  return `${rounded > 0 ? "+" : ""}${rounded}%`;
}

export function computeCompareSummary(
  left: FinalAnswer,
  right: FinalAnswer
): CompareDeltaSummary {
  const leftContextTokens = getContextTokens(left);
  const rightContextTokens = getContextTokens(right);
  const contextTokensDelta = rightContextTokens - leftContextTokens;
  const contextTokensPct =
    leftContextTokens > 0
      ? (contextTokensDelta / leftContextTokens) * 100
      : null;

  const leftK = (left.evidence_chunks || []).length;
  const rightK = (right.evidence_chunks || []).length;
  const kDelta = rightK - leftK;

  const leftConfidence = left.confidence?.final_score ?? null;
  const rightConfidence = right.confidence?.final_score ?? null;
  const leftConfidenceLevel = left.confidence?.level ?? null;
  const rightConfidenceLevel = right.confidence?.level ?? null;

  const leftAudit = left.citation_report ? getAuditState(left.citation_report) : null;
  const rightAudit = right.citation_report ? getAuditState(right.citation_report) : null;
  const leftClaimSupport = left.citation_report?.claim_support_rate ?? null;
  const rightClaimSupport = right.citation_report?.claim_support_rate ?? null;
  const leftInvalidCitations = left.citation_report?.invalid_citation_ids?.length ?? 0;
  const rightInvalidCitations = right.citation_report?.invalid_citation_ids?.length ?? 0;

  const leftLatencyMs = left.total_latency_ms || 0;
  const rightLatencyMs = right.total_latency_ms || 0;
  const latencyDeltaMs = rightLatencyMs - leftLatencyMs;

  const leftCostUsd = left.usage?.estimated_cost_usd ?? 0;
  const rightCostUsd = right.usage?.estimated_cost_usd ?? 0;
  const costDeltaUsd = rightCostUsd - leftCostUsd;
  const costPct = leftCostUsd > 0 ? (costDeltaUsd / leftCostUsd) * 100 : null;

  const highlights: CompareDeltaSummary["highlights"] = [];

  if (contextTokensPct !== null && contextTokensPct <= -15) {
    highlights.push({
      label: `${Math.abs(Math.round(contextTokensPct))}% fewer context tokens`,
      tone: "emerald",
    });
  }
  if (leftK > 0 && rightK === 0 && right.query_type === "conversational") {
    highlights.push({
      label: `Zero-retrieval fast path (k: ${leftK} → 0)`,
      tone: "blue",
    });
  }
  if (right.abstained && !left.abstained) {
    highlights.push({
      label: "Hallucination prevented (Honest Abstention)",
      tone: "amber",
    });
  }
  if (right.fallback_used && !left.fallback_used) {
    highlights.push({
      label: "Live Web Fallback triggered",
      tone: "purple",
    });
  }
  if (right.rewrite_attempts > 0) {
    highlights.push({
      label: `Corrective rewrite ×${right.rewrite_attempts}`,
      tone: "amber",
    });
  }
  if (right.regeneration_attempts > 0) {
    highlights.push({
      label: `Strict citation repair ×${right.regeneration_attempts}`,
      tone: "rose",
    });
  }
  if (isAuditClear(rightAudit) && leftAudit === "flagged") {
    highlights.push({
      label: "Citation audit upgraded (Fail → Pass)",
      tone: "emerald",
    });
  }

  return {
    leftContextTokens,
    rightContextTokens,
    contextTokensDelta,
    contextTokensPct,
    leftK,
    rightK,
    kDelta,
    leftConfidence,
    rightConfidence,
    leftConfidenceLevel,
    rightConfidenceLevel,
    leftAudit,
    rightAudit,
    leftClaimSupport,
    rightClaimSupport,
    leftInvalidCitations,
    rightInvalidCitations,
    leftLatencyMs,
    rightLatencyMs,
    latencyDeltaMs,
    leftCostUsd,
    rightCostUsd,
    costDeltaUsd,
    costPct,
    highlights,
  };
}

export function createCompareInspectorTurns(run: CompareRun | null): {
  turns: ChatTurn[];
  sideIndexMap: { left: number; right: number };
} {
  if (!run) {
    return { turns: [], sideIndexMap: { left: -1, right: -1 } };
  }
  const turns: ChatTurn[] = [];
  let leftIdx = -1;
  let rightIdx = -1;

  if (run.left.result) {
    leftIdx = turns.length;
    turns.push({
      id: `${run.id}_left`,
      query: `[A · ${run.left.presetLabel}] ${run.query}`,
      createdAt: run.createdAt,
      presetLabel: `Baseline · ${run.left.presetLabel}`,
      result: run.left.result,
      feedback: run.left.feedback,
    });
  }
  if (run.right.result) {
    rightIdx = turns.length;
    turns.push({
      id: `${run.id}_right`,
      query: `[B · ${run.right.presetLabel}] ${run.query}`,
      createdAt: run.createdAt,
      presetLabel: `Challenger · ${run.right.presetLabel}`,
      result: run.right.result,
      feedback: run.right.feedback,
    });
  }

  return { turns, sideIndexMap: { left: leftIdx, right: rightIdx } };
}
