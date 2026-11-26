import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { CompareArena } from "../components/CompareArena";
import {
  computeCompareSummary,
  createCompareInspectorTurns,
  formatSignedPct,
  getContextTokens,
} from "../lib/compare";
import type { CompareRun, EvidenceChunk, FinalAnswer } from "../types/api";

function makeChunk(id: string, tokens: number, score = 0.92): EvidenceChunk {
  return {
    citation_id: id,
    chunk_id: `chunk_${id}`,
    text: `Technical content for ${id}.`,
    source_type: "local",
    doc_id: "01_qdrant_hybrid",
    title: "Qdrant HNSW & Hybrid Guide",
    section_path: "HNSW Parameters",
    url: null,
    metadata: {},
    dense_score: 0.78,
    sparse_score: null,
    fusion_score: 0.032,
    rerank_raw_score: 3.1,
    rerank_score: score,
    matched_sub_queries: ["ef_construct default"],
    token_count: tokens,
  };
}

function makeAnswer(overrides: Partial<FinalAnswer> = {}): FinalAnswer {
  return {
    query: "What is the default ef_construct in Qdrant?",
    answer: "The default value of ef_construct in Qdrant is 100 [C1].",
    query_type: "factual",
    language: "en",
    query_plan: null,
    evidence_chunks: [makeChunk("C1", 220)],
    dynamic_k_decision: {
      candidate_count: 20,
      selected_count: 1,
      cutoff_reason: "threshold_tau_min",
      top_score: 0.96,
      cutoff_score: 0.96,
      total_context_tokens: 220,
      dropped_chunk_ids: [],
    },
    confidence: {
      top1_score: 0.96,
      mean_topk_score: 0.96,
      subquery_coverage: 1.0,
      deterministic_score: 0.968,
      crag_grade: null,
      final_score: 0.968,
      level: "high",
      recommended_action: "proceed",
      reasoning: "High deterministic confidence.",
    },
    citation_report: {
      total_citations_found: 1,
      valid_citation_ids: ["C1"],
      invalid_citation_ids: [],
      uncited_factual_sentences: [],
      claim_verifications: [],
      citation_precision: 1.0,
      claim_support_rate: 1.0,
      passed: true,
      regeneration_triggered: false,
    },
    fallback_used: false,
    abstained: false,
    rewrite_attempts: 0,
    regeneration_attempts: 0,
    total_latency_ms: 4200,
    usage: {
      prompt_tokens: 520,
      completion_tokens: 60,
      total_tokens: 580,
      estimated_cost_usd: 0.00054,
      cache_hits: 0,
    },
    trace: [
      {
        node: "classify_and_plan",
        latency_ms: 150,
        summary: "factual",
        metadata: {},
      },
    ],
    trace_id: "000000000000000000000000000000aa",
    trace_url: "http://localhost:3000/traces/000000000000000000000000000000aa",
    ...overrides,
  };
}

describe("Compare Mode delta computation helpers", () => {
  it("computes context token reduction, k delta, and safety highlights accurately", () => {
    const naiveLeft = makeAnswer({
      evidence_chunks: [
        makeChunk("C1", 200),
        makeChunk("C2", 200),
        makeChunk("C3", 200),
        makeChunk("C4", 200),
        makeChunk("C5", 200),
      ],
      dynamic_k_decision: null,
      confidence: null,
      citation_report: {
        total_citations_found: 2,
        valid_citation_ids: ["C1"],
        invalid_citation_ids: ["C9"],
        uncited_factual_sentences: [],
        claim_verifications: [],
        citation_precision: 0.5,
        claim_support_rate: 0.5,
        passed: false,
        regeneration_triggered: false,
      },
      total_latency_ms: 1400,
      usage: {
        prompt_tokens: 1300,
        completion_tokens: 80,
        total_tokens: 1380,
        estimated_cost_usd: 0.00049,
        cache_hits: 0,
      },
    });

    const adaptiveRight = makeAnswer({
      evidence_chunks: [makeChunk("C1", 250)],
      dynamic_k_decision: {
        candidate_count: 20,
        selected_count: 1,
        cutoff_reason: "threshold_tau_min",
        top_score: 0.97,
        cutoff_score: 0.97,
        total_context_tokens: 250,
        dropped_chunk_ids: [],
      },
      rewrite_attempts: 1,
      total_latency_ms: 4800,
      usage: {
        prompt_tokens: 550,
        completion_tokens: 65,
        total_tokens: 615,
        estimated_cost_usd: 0.00058,
        cache_hits: 0,
      },
    });

    expect(getContextTokens(naiveLeft)).toBe(1000);
    expect(getContextTokens(adaptiveRight)).toBe(250);

    const summary = computeCompareSummary(naiveLeft, adaptiveRight);
    expect(summary.contextTokensDelta).toBe(-750);
    expect(summary.contextTokensPct).toBeCloseTo(-75, 5);
    expect(formatSignedPct(summary.contextTokensPct)).toBe("-75%");
    expect(summary.leftK).toBe(5);
    expect(summary.rightK).toBe(1);
    expect(summary.kDelta).toBe(-4);
    expect(summary.leftInvalidCitations).toBe(1);
    expect(summary.rightInvalidCitations).toBe(0);
    expect(summary.highlights.map((h) => h.label)).toEqual(
      expect.arrayContaining([
        "75% fewer context tokens",
        "Corrective rewrite ×1",
        "Citation audit upgraded (Fail → Pass)",
      ])
    );
  });

  it("detects zero-retrieval conversational skip and honest abstention highlights", () => {
    const left = makeAnswer({
      evidence_chunks: [makeChunk("C1", 300), makeChunk("C2", 300)],
      dynamic_k_decision: null,
      abstained: false,
    });
    const convRight = makeAnswer({
      query_type: "conversational",
      evidence_chunks: [],
      dynamic_k_decision: null,
    });
    const convSummary = computeCompareSummary(left, convRight);
    expect(convSummary.highlights.map((h) => h.label)).toContain(
      "Zero-retrieval fast path (k: 2 → 0)"
    );

    const abstainRight = makeAnswer({
      evidence_chunks: [],
      dynamic_k_decision: null,
      abstained: true,
    });
    const abstainSummary = computeCompareSummary(left, abstainRight);
    expect(abstainSummary.highlights.map((h) => h.label)).toContain(
      "Hallucination prevented (Honest Abstention)"
    );
  });

  it("maps completed compare sides to virtual GlassBoxInspector turns", () => {
    const run: CompareRun = {
      id: "cmp_1",
      query: "What is ef_construct?",
      createdAt: 1000,
      left: {
        preset: "naive_dense",
        presetLabel: "Naive Dense",
        status: "done",
        liveSteps: [],
        liveUsage: null,
        result: makeAnswer(),
        error: null,
      },
      right: {
        preset: "full",
        presetLabel: "Full Agentic",
        status: "done",
        liveSteps: [],
        liveUsage: null,
        result: makeAnswer(),
        error: null,
      },
    };
    const mapped = createCompareInspectorTurns(run);
    expect(mapped.turns).toHaveLength(2);
    expect(mapped.sideIndexMap).toEqual({ left: 0, right: 1 });
    expect(mapped.turns[0].presetLabel).toBe("Baseline · Naive Dense");
    expect(mapped.turns[1].presetLabel).toBe("Challenger · Full Agentic");
  });
});

describe("CompareArena component", () => {
  it("renders high-contrast demo prompts in empty state and fires submission", () => {
    const onSubmitCompare = vi.fn();
    const onSwapPresets = vi.fn();
    render(
      <CompareArena
        runs={[]}
        activeRunId={null}
        leftPreset="naive_dense"
        rightPreset="full"
        inspectedSide="right"
        isRunning={false}
        useStreaming={true}
        budgetCapLabel={null}
        onLeftPresetChange={vi.fn()}
        onRightPresetChange={vi.fn()}
        onSwapPresets={onSwapPresets}
        onSelectRun={vi.fn()}
        onDeleteRun={vi.fn()}
        onNewComparison={vi.fn()}
        onSubmitCompare={onSubmitCompare}
        onStopCompare={vi.fn()}
        onInspectSide={vi.fn()}
        onCitationClick={vi.fn()}
      />
    );

    expect(
      screen.getByText(/Compare Naive Dense vs\. Full Agentic/i)
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Swap A and B presets" }));
    expect(onSwapPresets).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByText("Factual dynamic top-k savings"));
    expect(onSubmitCompare).toHaveBeenCalledWith(
      expect.stringContaining("ef_construct")
    );
  });

  it("renders side-by-side results, delta scorecard, citation clicks, and independent feedback", async () => {
    const onCitationClick = vi.fn();
    const onInspectSide = vi.fn();
    const onSubmitSideFeedback = vi.fn(async () => {});

    const run: CompareRun = {
      id: "cmp_live_1",
      query: "What is the default ef_construct in Qdrant?",
      createdAt: Date.now(),
      left: {
        preset: "naive_dense",
        presetLabel: "Naive Dense",
        status: "done",
        liveSteps: [],
        liveUsage: null,
        result: makeAnswer({
          answer: "Naive baseline answer cites [C1] from 5 static chunks.",
          evidence_chunks: [
            makeChunk("C1", 200),
            makeChunk("C2", 200),
            makeChunk("C3", 200),
            makeChunk("C4", 200),
            makeChunk("C5", 200),
          ],
          dynamic_k_decision: null,
          confidence: null,
          trace_id: "00000000000000000000000000000001",
        }),
        error: null,
      },
      right: {
        preset: "full",
        presetLabel: "Full Agentic",
        status: "done",
        liveSteps: [],
        liveUsage: null,
        result: makeAnswer({
          answer: "AdaptiveRAG pinpoints ef_construct=100 [C1].",
          evidence_chunks: [makeChunk("C1", 250)],
          dynamic_k_decision: {
            candidate_count: 20,
            selected_count: 1,
            cutoff_reason: "threshold_tau_min",
            top_score: 0.98,
            cutoff_score: 0.98,
            total_context_tokens: 250,
            dropped_chunk_ids: [],
          },
          trace_id: "00000000000000000000000000000002",
        }),
        error: null,
      },
    };

    render(
      <CompareArena
        runs={[run]}
        activeRunId={run.id}
        leftPreset="naive_dense"
        rightPreset="full"
        inspectedSide="right"
        isRunning={false}
        useStreaming={true}
        budgetCapLabel="$0.05"
        onLeftPresetChange={vi.fn()}
        onRightPresetChange={vi.fn()}
        onSwapPresets={vi.fn()}
        onSelectRun={vi.fn()}
        onDeleteRun={vi.fn()}
        onNewComparison={vi.fn()}
        onSubmitCompare={vi.fn()}
        onStopCompare={vi.fn()}
        onInspectSide={onInspectSide}
        onCitationClick={onCitationClick}
        onSubmitSideFeedback={onSubmitSideFeedback}
      />
    );

    expect(screen.getByTestId("compare-delta-scorecard")).toBeInTheDocument();
    expect(screen.getByText("-75%")).toBeInTheDocument();
    expect(screen.getByText("5 → 1")).toBeInTheDocument();
    expect(screen.getByText("75% fewer context tokens")).toBeInTheDocument();

    // Clicking an inline citation on the left card targets "left" side
    const citeButtons = screen.getAllByRole("button", { name: "C1" });
    expect(citeButtons.length).toBeGreaterThanOrEqual(2);
    fireEvent.click(citeButtons[0]);
    expect(onCitationClick).toHaveBeenCalledWith("left", "C1");

    // Clicking thumbs-up on right side submits feedback for "right"
    fireEvent.click(
      screen.getByRole("button", { name: "Mark Full Agentic answer helpful" })
    );
    await waitFor(() => {
      expect(onSubmitSideFeedback).toHaveBeenCalledWith(
        "cmp_live_1",
        "right",
        1,
        undefined
      );
    });
  });

  it("isolates a failure on one side while preserving the healthy sibling result", () => {
    const run: CompareRun = {
      id: "cmp_partial",
      query: "Compare HNSW and pgvector",
      createdAt: Date.now(),
      left: {
        preset: "naive_dense",
        presetLabel: "Naive Dense",
        status: "error",
        liveSteps: [],
        liveUsage: null,
        result: null,
        error: "HTTP 402: Budget cap exceeded",
      },
      right: {
        preset: "full",
        presetLabel: "Full Agentic",
        status: "done",
        liveSteps: [],
        liveUsage: null,
        result: makeAnswer({ answer: "AdaptiveRAG succeeded cleanly [C1]." }),
        error: null,
      },
    };

    render(
      <CompareArena
        runs={[run]}
        activeRunId={run.id}
        leftPreset="naive_dense"
        rightPreset="full"
        inspectedSide="right"
        isRunning={false}
        useStreaming={true}
        budgetCapLabel={null}
        onLeftPresetChange={vi.fn()}
        onRightPresetChange={vi.fn()}
        onSwapPresets={vi.fn()}
        onSelectRun={vi.fn()}
        onDeleteRun={vi.fn()}
        onNewComparison={vi.fn()}
        onSubmitCompare={vi.fn()}
        onStopCompare={vi.fn()}
        onInspectSide={vi.fn()}
        onCitationClick={vi.fn()}
      />
    );

    expect(screen.getByText("Naive Dense failed")).toBeInTheDocument();
    expect(screen.getByText(/HTTP 402: Budget cap exceeded/)).toBeInTheDocument();
    expect(
      screen.getByText(/AdaptiveRAG succeeded cleanly/i)
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("compare-delta-scorecard")
    ).not.toBeInTheDocument();
  });
});
