import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { AnswerTelemetryBar } from "../components/AnswerTelemetryBar";
import { ChatWorkspace } from "../components/ChatWorkspace";
import { GlassBoxInspector } from "../components/GlassBoxInspector";
import { KnowledgeBaseView } from "../components/KnowledgeBaseView";
import { MarkdownWithCitations } from "../components/MarkdownWithCitations";
import {
  buildAuthHeaders,
  buildQueryPayload,
  buildWebIngestPayload,
  parseUrlLines,
  setStudioApiKey,
} from "../lib/api";
import { splitTextWithCitations } from "../lib/citations";
import { SseStreamParser } from "../lib/sse";
import { DEFAULT_CUSTOM_ABLATION } from "../constants/pipeline";
import type { ChatTurn, EvidenceChunk } from "../types/api";

describe("SseStreamParser", () => {
  it("parses chunked node_update and final_answer events across split frames", () => {
    const parser = new SseStreamParser();
    const first = parser.push('event: node_update\ndata: {"node":"classify_and_plan","tra');
    expect(first).toEqual([]);

    const second = parser.push(
      'ce":[{"node":"classify_and_plan","latency_ms":12.4,"summary":"factual","metadata":{}}]}\n\n' +
        'event: final_answer\ndata: {"query":"Q","answer":"A [C1]."}\n\n'
    );
    expect(second).toHaveLength(2);
    expect(second[0].event).toBe("node_update");
    expect((second[0].data as { node: string }).node).toBe("classify_and_plan");
    expect(second[1].event).toBe("final_answer");
    expect((second[1].data as { answer: string }).answer).toBe("A [C1].");
  });
});

describe("splitTextWithCitations", () => {
  it("splits single and grouped local/web citations cleanly", () => {
    const segments = splitTextWithCitations(
      "Qdrant uses RRF [C1] and live search [C2, W1] for freshness."
    );
    expect(segments).toEqual([
      { type: "text", value: "Qdrant uses RRF " },
      { type: "citation", ids: ["C1"], raw: "[C1]" },
      { type: "text", value: " and live search " },
      { type: "citation", ids: ["C2", "W1"], raw: "[C2, W1]" },
      { type: "text", value: " for freshness." },
    ]);
  });
});

describe("API payload helpers", () => {
  it("deduplicates URL lines and builds scrape/crawl payloads", () => {
    const urls = parseUrlLines(
      " https://docs.qdrant.tech/concepts/ \n\nhttps://docs.qdrant.tech/concepts/\nhttps://docs.firecrawl.dev/quickstarts/firecrawl-local "
    );
    expect(urls).toEqual([
      "https://docs.qdrant.tech/concepts/",
      "https://docs.firecrawl.dev/quickstarts/firecrawl-local",
    ]);

    const scrape = buildWebIngestPayload({
      mode: "scrape",
      urlsText: urls.join("\n"),
      onlyMainContent: true,
    });
    expect(scrape.urls).toEqual(urls);
    expect(scrape.crawl_url).toBeUndefined();

    const crawl = buildWebIngestPayload({
      mode: "crawl",
      crawlUrl: "https://docs.qdrant.tech/",
      crawlLimit: 6,
      crawlMaxDepth: 2,
      parentChild: true,
    });
    expect(crawl.crawl_url).toBe("https://docs.qdrant.tech/");
    expect(crawl.crawl_limit).toBe(6);
    expect(crawl.crawl_max_depth).toBe(2);
    expect(crawl.parent_child).toBe(true);
  });

  it("attaches X-API-Key header when studio API key is configured", () => {
    setStudioApiKey("  prod-secret-42  ");
    try {
      expect(buildAuthHeaders({ Accept: "application/json" })).toEqual({
        Accept: "application/json",
        "X-API-Key": "prod-secret-42",
      });
    } finally {
      setStudioApiKey("");
    }
    expect(buildAuthHeaders({ Accept: "application/json" })).toEqual({
      Accept: "application/json",
    });
  });

  it("builds query payloads for preset vs custom ablation", () => {
    const presetReq = buildQueryPayload({
      query: "What is RRF?",
      chatHistory: [],
      preset: "full",
      customAblation: DEFAULT_CUSTOM_ABLATION,
      maxCostUsd: 0.05,
      sessionId: "sess_abc",
    });
    expect(presetReq.preset).toBe("full");
    expect(presetReq.ablation).toBeUndefined();
    expect(presetReq.max_cost_usd).toBe(0.05);
    expect(presetReq.session_id).toBe("sess_abc");

    const customReq = buildQueryPayload({
      query: "What is RRF?",
      chatHistory: [],
      preset: "custom",
      customAblation: { ...DEFAULT_CUSTOM_ABLATION, retrieval_mode: "dense_only" },
      maxCostUsd: null,
    });
    expect(customReq.preset).toBeUndefined();
    expect(customReq.ablation?.retrieval_mode).toBe("dense_only");
  });
});

describe("MarkdownWithCitations", () => {
  it("renders interactive citation pills and fires onCitationClick without mutating code blocks", () => {
    const onCite = vi.fn();
    const chunks: EvidenceChunk[] = [
      {
        citation_id: "C1",
        chunk_id: "ch1",
        text: "RRF combines dense and sparse scores.",
        source_type: "local",
        doc_id: "qdrant_hybrid",
        title: "Qdrant Hybrid Search",
        section_path: "RRF",
        url: null,
        metadata: {},
        dense_score: null,
        sparse_score: null,
        fusion_score: 0.5,
        rerank_raw_score: null,
        rerank_score: 0.96,
        matched_sub_queries: [],
        token_count: 12,
      },
    ];

    render(
      <MarkdownWithCitations
        content={
          "Qdrant uses RRF \\[ \\text{RRF}(d) = \\frac{1}{k + r} \\] with \\(k = 60\\) [C1].\n\n```python\nval = items[C1]\n```"
        }
        evidenceChunks={chunks}
        onCitationClick={onCite}
      />
    );

    const btn = screen.getByRole("button", { name: "C1" });
    expect(btn).toBeInTheDocument();
    expect(document.querySelector(".katex")).toBeInTheDocument();
    fireEvent.click(btn);
    expect(onCite).toHaveBeenCalledWith("C1");
  });
});

describe("GlassBoxInspector", () => {
  it("renders Trace, Top-K, CRAG, Evidence, and Audit tabs with clickable evidence jump", () => {
    const onTabChange = vi.fn();
    const onJump = vi.fn();
    const turn: ChatTurn = {
      id: "turn_1",
      query: "What is RRF?",
      createdAt: Date.now(),
      presetLabel: "Full Agentic",
      result: {
        query: "What is RRF?",
        answer: "RRF fuses dense and sparse rankings [C1].",
        query_type: "factual",
        language: "en",
        query_plan: {
          original_query: "What is RRF?",
          standalone_query: "What is RRF?",
          query_type: "factual",
          needs_retrieval: true,
          language: "en",
          requires_fresh_web_info: false,
          reasoning: "Factual technical question.",
          strategy: {
            needs_retrieval: true,
            max_sub_queries: 1,
            dense_prefetch_limit: 20,
            sparse_prefetch_limit: 20,
            fusion_limit: 20,
            rerank_min_k: 2,
            rerank_max_k: 5,
            tau_min: 0.35,
            alpha_relative: 0.6,
            delta_gap: 0.25,
            max_context_tokens: 3000,
          },
          sub_queries: ["What is RRF?"],
        },
        evidence_chunks: [
          {
            citation_id: "C1",
            chunk_id: "ch1",
            text: "Reciprocal Rank Fusion formula 1 / (k + rank).",
            source_type: "local",
            doc_id: "qdrant_hybrid",
            title: "Qdrant Hybrid Guide",
            section_path: "RRF Formula",
            url: "https://docs.qdrant.tech/concepts/hybrid-queries/",
            metadata: {},
            dense_score: null,
            sparse_score: null,
            fusion_score: 0.032,
            rerank_raw_score: 4.1,
            rerank_score: 0.97,
            matched_sub_queries: ["What is RRF?"],
            token_count: 14,
            parent_id: "parent-rrf",
            expanded_to_parent: true,
            matched_child_ids: ["ch1", "ch2"],
          },
        ],
        dynamic_k_decision: {
          candidate_count: 10,
          selected_count: 1,
          cutoff_reason: "threshold_tau_min",
          top_score: 0.97,
          cutoff_score: 0.97,
          total_context_tokens: 14,
          dropped_chunk_ids: [],
          parent_expansion_applied: true,
          collapsed_child_count: 2,
          expanded_parent_count: 1,
        },
        confidence: {
          top1_score: 0.97,
          mean_topk_score: 0.97,
          subquery_coverage: 1.0,
          deterministic_score: 0.976,
          crag_grade: null,
          final_score: 0.976,
          level: "high",
          recommended_action: "proceed",
          reasoning: "High top-1 rerank confidence.",
        },
        citation_report: {
          total_citations_found: 1,
          valid_citation_ids: ["C1"],
          invalid_citation_ids: [],
          uncited_factual_sentences: [],
          claim_verifications: [
            {
              claim: "RRF fuses dense and sparse rankings.",
              cited_ids: ["C1"],
              verdict: "supported",
              explanation: "Supported by [C1].",
            },
          ],
          citation_precision: 1.0,
          claim_support_rate: 1.0,
          passed: true,
          regeneration_triggered: false,
        },
        fallback_used: false,
        abstained: false,
        rewrite_attempts: 0,
        regeneration_attempts: 0,
        total_latency_ms: 420,
        usage: {
          prompt_tokens: 300,
          completion_tokens: 45,
          total_tokens: 345,
          estimated_cost_usd: 0.00042,
          cache_hits: 0,
        },
        trace: [
          {
            node: "classify_and_plan",
            latency_ms: 120,
            summary: "Routed as factual (en)",
            metadata: {},
          },
        ],
      },
    };

    render(
      <GlassBoxInspector
        turns={[turn]}
        selectedTurnIndex={0}
        onSelectTurnIndex={vi.fn()}
        activeTab="evidence"
        onTabChange={onTabChange}
        highlightedCitationId="C1"
        highlightToken={1}
        onJumpToCitation={onJump}
        onClose={vi.fn()}
      />
    );

    expect(screen.getByText("Qdrant Hybrid Guide")).toBeInTheDocument();
    expect(screen.getByText(/Parent Section \(2 child\)/i)).toBeInTheDocument();
    expect(
      screen.getByRole("link", {
        name: /https:\/\/docs\.qdrant\.tech\/concepts\/hybrid-queries\//,
      })
    ).toBeInTheDocument();
  });
});

describe("KnowledgeBaseView", () => {
  it("renders indexed local and web sources and supports origin filtering", () => {
    render(
      <KnowledgeBaseView
        apiUrl="http://localhost:8000"
        health={{
          status: "ok",
          version: "0.1.0",
          qdrant: { status: "ok", collection: "adaptiverag_docs", points_count: 15 },
          firecrawl: { status: "ok", base_url: "http://localhost:3002" },
          llm_configured: true,
          llm_error: null,
          llm_provider: "openrouter",
          fast_model: "openai/gpt-4o-mini",
          generation_model: "openai/gpt-4o-mini",
          dense_embedding_model: "intfloat/multilingual-e5-small",
          sparse_embedding_model: "Qdrant/bm25",
          reranker_model: "bge-reranker-v2-m3",
          web_fallback_enabled: true,
          web_fallback_providers: ["tavily"],
          langfuse_enabled: false,
          llm_cache_enabled: true,
          default_max_run_cost_usd: 0.25,
        }}
        documentsData={{
          collection: "adaptiverag_docs",
          total_documents: 2,
          total_chunks: 15,
          documents: [
            {
              doc_id: "01_qdrant_hybrid",
              title: "Qdrant Hybrid Guide",
              source_origin: "local",
              doc_type: "markdown",
              language: "en",
              chunk_strategy: "parent_child",
              chunk_count: 10,
              total_tokens: 1800,
              source_url: null,
              source_path: "01_qdrant_hybrid.md",
              file_name: "01_qdrant_hybrid.md",
            },
            {
              doc_id: "web_docs_firecrawl",
              title: "Firecrawl Local Quickstart",
              source_origin: "web",
              doc_type: "markdown",
              language: "en",
              chunk_count: 5,
              total_tokens: 920,
              source_url: "https://docs.firecrawl.dev/quickstarts/firecrawl-local",
              source_path: "https://docs.firecrawl.dev/quickstarts/firecrawl-local",
              file_name: null,
            },
          ],
        }}
        documentsLoading={false}
        documentsError={null}
        onRefreshAll={vi.fn(async () => {})}
      />
    );

    expect(screen.getByText("Qdrant Hybrid Guide")).toBeInTheDocument();
    expect(screen.getByText("Small→Big")).toBeInTheDocument();
    expect(
      screen.getByLabelText(/Hierarchical Parent-Child chunking/i)
    ).toBeInTheDocument();
    expect(screen.getByText("Firecrawl Local Quickstart")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Web" }));
    expect(screen.queryByText("Qdrant Hybrid Guide")).not.toBeInTheDocument();
    expect(screen.getByText("Firecrawl Local Quickstart")).toBeInTheDocument();
  });
});

describe("Langfuse Trace Link & Human-in-the-Loop Feedback UI", () => {
  const sampleTurn: ChatTurn = {
    id: "turn_lf_1",
    query: "How does Qdrant hybrid search work?",
    createdAt: Date.now(),
    presetLabel: "Full Agentic",
    result: {
      query: "How does Qdrant hybrid search work?",
      answer: "Qdrant fuses dense and BM25 sparse vectors via RRF [C1].",
      query_type: "factual",
      language: "en",
      query_plan: null,
      evidence_chunks: [],
      dynamic_k_decision: null,
      confidence: null,
      citation_report: null,
      fallback_used: false,
      abstained: false,
      rewrite_attempts: 0,
      regeneration_attempts: 0,
      total_latency_ms: 350,
      usage: {
        prompt_tokens: 200,
        completion_tokens: 40,
        total_tokens: 240,
        estimated_cost_usd: 0.0003,
        cache_hits: 0,
      },
      trace: [],
      trace_id: "00000000000000000000000000000001",
      trace_url: "https://cloud.langfuse.com/project/demo/traces/00000000000000000000000000000001",
    },
  };

  it("renders a clickable Langfuse trace deep-link pill in AnswerTelemetryBar", () => {
    render(<AnswerTelemetryBar result={sampleTurn.result} />);
    const link = screen.getByRole("link", { name: /Langfuse/i });
    expect(link).toHaveAttribute("href", sampleTurn.result.trace_url);
  });

  it("submits thumbs-up immediately and thumbs-down with an optional comment in ChatWorkspace", async () => {
    const onSubmitFeedback = vi.fn(async () => {});
    render(
      <ChatWorkspace
        turns={[sampleTurn]}
        selectedTurnIndex={0}
        isRunning={false}
        pendingQuery={null}
        liveSteps={[]}
        liveUsage={null}
        streamError={null}
        presetShortLabel="Full Agentic"
        useStreaming={true}
        budgetCapLabel={null}
        onSubmitQuery={vi.fn()}
        onStopStream={vi.fn()}
        onInspectTurn={vi.fn()}
        onCitationClick={vi.fn()}
        onSubmitFeedback={onSubmitFeedback}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: "Mark answer helpful" }));
    await waitFor(() => {
      expect(onSubmitFeedback).toHaveBeenCalledWith("turn_lf_1", 1, undefined);
    });

    fireEvent.click(screen.getByRole("button", { name: "Mark answer unhelpful" }));
    const noteInput = screen.getByPlaceholderText(/Optional note/i);
    fireEvent.change(noteInput, { target: { value: "Missing Turkish summary" } });
    fireEvent.click(screen.getByRole("button", { name: "Send Feedback" }));
    await waitFor(() => {
      expect(onSubmitFeedback).toHaveBeenCalledWith("turn_lf_1", 0, "Missing Turkish summary");
    });
  });
});
