import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { AnswerTelemetryBar } from "../components/AnswerTelemetryBar";
import { GlassBoxInspector } from "../components/GlassBoxInspector";
import { getAuditState } from "../lib/audit";
import { computeCompareSummary } from "../lib/compare";
import type { CitationReport, FinalAnswer } from "../types/api";

function makeReport(overrides: Partial<CitationReport> = {}): CitationReport {
  return {
    total_citations_found: 1,
    valid_citation_ids: ["C1"],
    invalid_citation_ids: [],
    uncited_factual_sentences: [],
    claim_verifications: [],
    citation_precision: 1,
    claim_support_rate: 1,
    passed: true,
    regeneration_triggered: false,
    ...overrides,
  };
}

const SUPPORTED_CLAIM = {
  claim: "The default ef_construct is 100.",
  cited_ids: ["C1"],
  verdict: "supported" as const,
  explanation: "Stated in C1.",
};

// The judge failed while the citations were fine: the API reports passed=false.
const UNVERIFIED = makeReport({
  passed: false,
  structural_passed: true,
  semantic_audit: "failed",
  semantic_audit_error: "The judge model timed out.",
});

function makeAnswer(citation_report: CitationReport | null): FinalAnswer {
  return {
    query: "What is the default ef_construct?",
    answer: "The default value of ef_construct is 100 [C1].",
    query_type: "factual",
    language: "en",
    query_plan: null,
    evidence_chunks: [],
    dynamic_k_decision: null,
    confidence: null,
    citation_report,
    fallback_used: false,
    abstained: false,
    rewrite_attempts: 0,
    regeneration_attempts: 0,
    total_latency_ms: 1200,
    usage: {
      prompt_tokens: 100,
      completion_tokens: 20,
      total_tokens: 120,
      estimated_cost_usd: 0,
      cache_hits: 0,
    },
    trace: [],
  };
}

describe("getAuditState", () => {
  it("separates a broken judge from a broken answer", () => {
    expect(
      getAuditState(
        makeReport({ semantic_audit: "completed", claim_verifications: [SUPPORTED_CLAIM] })
      )
    ).toBe("passed");
    expect(getAuditState(makeReport({ passed: false, semantic_audit: "completed" }))).toBe(
      "flagged"
    );
    expect(getAuditState(UNVERIFIED)).toBe("unverified");
    expect(
      getAuditState(
        makeReport({
          passed: false,
          structural_passed: false,
          invalid_citation_ids: ["C9"],
          semantic_audit: "failed",
        })
      )
    ).toBe("flagged");
    expect(getAuditState(makeReport({ semantic_audit: "skipped" }))).toBe("structural");
  });

  it("reads answers saved before the API reported the claim check", () => {
    // Older answers have no semantic_audit field; verdicts mean the judge ran.
    expect(getAuditState(makeReport({ claim_verifications: [SUPPORTED_CLAIM] }))).toBe("passed");
    expect(getAuditState(makeReport())).toBe("structural");
    expect(getAuditState(makeReport({ passed: false }))).toBe("flagged");
  });
});

describe("Unverified citation audit", () => {
  it("shows a warning in the receipt instead of Passed or a support percentage", () => {
    render(<AnswerTelemetryBar result={makeAnswer(UNVERIFIED)} />);

    expect(screen.getByText("Unverified")).toBeInTheDocument();
    expect(screen.queryByText("Passed")).not.toBeInTheDocument();
    expect(screen.queryByText("Flagged")).not.toBeInTheDocument();
    expect(screen.queryByText("100%")).not.toBeInTheDocument();
  });

  it("explains the failed claim check in the Audit tab", () => {
    render(
      <GlassBoxInspector
        turns={[
          {
            id: "turn_1",
            query: "What is the default ef_construct?",
            createdAt: Date.now(),
            presetLabel: "Full Agentic",
            result: makeAnswer(UNVERIFIED),
          },
        ]}
        selectedTurnIndex={0}
        onSelectTurnIndex={vi.fn()}
        activeTab="citations"
        onTabChange={vi.fn()}
        highlightedCitationId={null}
        highlightToken={0}
        onJumpToCitation={vi.fn()}
        onClose={vi.fn()}
      />
    );

    expect(screen.getByText("Claims weren't checked")).toBeInTheDocument();
    expect(screen.getByText(/The judge model timed out\./)).toBeInTheDocument();
    expect(screen.getByText("Sentences cited")).toBeInTheDocument();
    expect(screen.queryByText("Claim support")).not.toBeInTheDocument();
    expect(screen.getByLabelText("unverified")).toBeInTheDocument();
  });

  it("says the model returned no answer instead of listing citation problems", () => {
    const emptyReport = makeReport({
      total_citations_found: 0,
      valid_citation_ids: [],
      citation_precision: 0,
      claim_support_rate: 0,
      passed: false,
      structural_passed: false,
      empty_answer: true,
    });
    render(
      <GlassBoxInspector
        turns={[
          {
            id: "turn_1",
            query: "What is the default ef_construct?",
            createdAt: Date.now(),
            presetLabel: "Full Agentic",
            result: {
              ...makeAnswer(emptyReport),
              answer: "The model returned an empty answer to this question.",
              regeneration_attempts: 1,
            },
          },
        ]}
        selectedTurnIndex={0}
        onSelectTurnIndex={vi.fn()}
        activeTab="citations"
        onTabChange={vi.fn()}
        highlightedCitationId={null}
        highlightToken={0}
        onJumpToCitation={vi.fn()}
        onClose={vi.fn()}
      />
    );

    expect(screen.getByText("The model returned no answer")).toBeInTheDocument();
    expect(screen.getByText(/the retry came back empty too/)).toBeInTheDocument();
    expect(screen.queryByText("Citation audit flagged problems")).not.toBeInTheDocument();
    expect(screen.queryByText("Citation precision")).not.toBeInTheDocument();
    expect(getAuditState(emptyReport)).toBe("flagged");
  });

  it("marks an empty answer as flagged on the receipt without a support percentage", () => {
    const emptyReport = makeReport({
      total_citations_found: 0,
      valid_citation_ids: [],
      citation_precision: 0,
      claim_support_rate: 0,
      passed: false,
      structural_passed: false,
      empty_answer: true,
    });
    render(<AnswerTelemetryBar result={makeAnswer(emptyReport)} />);

    expect(screen.getByText("Flagged")).toBeInTheDocument();
    expect(screen.queryByText("0%")).not.toBeInTheDocument();
    expect(screen.getByTitle("The model returned no answer. Open the audit")).toBeInTheDocument();
  });

  it("does not count an unverified answer as a failed or an upgraded audit in the arena", () => {
    const flagged = makeAnswer(
      makeReport({ passed: false, structural_passed: false, invalid_citation_ids: ["C9"] })
    );
    const unverified = makeAnswer(UNVERIFIED);

    const summary = computeCompareSummary(flagged, unverified);
    expect(summary.leftAudit).toBe("flagged");
    expect(summary.rightAudit).toBe("unverified");
    expect(summary.highlights.map((h) => h.label)).not.toContain(
      "Citation audit upgraded (Fail → Pass)"
    );

    const reversed = computeCompareSummary(unverified, makeAnswer(makeReport()));
    expect(reversed.highlights.map((h) => h.label)).not.toContain(
      "Citation audit upgraded (Fail → Pass)"
    );
  });
});
