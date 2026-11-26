import type { CitationReport, SemanticAuditStatus } from "../types/api";

/**
 * One reading of a citation report, shared by the receipt, the inspector and the arena.
 *
 * - `passed`: claims were checked against their passages and hold up.
 * - `flagged`: the answer has broken citations or unsupported claims.
 * - `unverified`: the claim check failed to run, but the citations themselves are fine.
 * - `structural`: only citation IDs and coverage were checked, by design.
 */
export type AuditState = "passed" | "flagged" | "unverified" | "structural";

export const AUDIT_LABEL: Record<AuditState, string> = {
  passed: "Passed",
  flagged: "Flagged",
  unverified: "Unverified",
  structural: "Cited",
};

/** The claim-check status, inferred for answers saved before the API reported it. */
export function semanticAuditOf(report: CitationReport): SemanticAuditStatus {
  return report.semantic_audit ?? (report.claim_verifications.length > 0 ? "completed" : "skipped");
}

export function getAuditState(report: CitationReport): AuditState {
  const semantic = semanticAuditOf(report);
  if (semantic === "failed") {
    return report.structural_passed === false ? "flagged" : "unverified";
  }
  if (!report.passed) return "flagged";
  return semantic === "completed" ? "passed" : "structural";
}

/** True for states where the answer cleared every check that ran. */
export function isAuditClear(state: AuditState | null): boolean {
  return state === "passed" || state === "structural";
}
