import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { AnswerTelemetryBar } from "../components/AnswerTelemetryBar";
import { ChatWorkspace } from "../components/ChatWorkspace";
import { GlassBoxInspector } from "../components/GlassBoxInspector";
import { KnowledgeBaseView } from "../components/KnowledgeBaseView";
import { MarkdownWithCitations } from "../components/MarkdownWithCitations";
import { ModelSelectorPopover } from "../components/ModelSelectorPopover";
import {
  buildAuthHeaders,
  buildQueryPayload,
  buildWebIngestPayload,
  parseUrlLines,
  setStudioApiKey,
} from "../lib/api";
import { splitTextWithCitations } from "../lib/citations";
import { SseStreamParser } from "../lib/sse";
import {
  DEFAULT_CUSTOM_ABLATION,
  DEFAULT_THREAD_CONTEXT,
  resolveEffectiveInstructions,
} from "../constants/pipeline";
import type {
  ChatTurn,
  EvidenceChunk,
  LLMOverride,
  ModelsCatalogResponse,
  ThreadContextConfig,
} from "../types/api";

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

  it("attaches X-API-Key and X-Provider-API-Key headers cleanly when configured", () => {
    setStudioApiKey("  prod-secret-42  ");
    try {
      expect(
        buildAuthHeaders({ Accept: "application/json" }, undefined, "  sk-byok-99  ")
      ).toEqual({
        Accept: "application/json",
        "X-API-Key": "prod-secret-42",
        "X-Provider-API-Key": "sk-byok-99",
      });
    } finally {
      setStudioApiKey("");
    }
    expect(buildAuthHeaders({ Accept: "application/json" }, undefined, "   ")).toEqual({
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
      customInstructions: "  Summarize in 3 bullets.  ",
      docIds: [" doc_a ", "doc_a", "doc_b"],
      llmOverride: {
        provider: "openrouter",
        fast_model: " google/gemini-2.5-flash ",
        generation_model: " deepseek/deepseek-v3.2 ",
        temperature: 0.2,
      },
    });
    expect(customReq.preset).toBeUndefined();
    expect(customReq.ablation?.retrieval_mode).toBe("dense_only");
    expect(customReq.custom_instructions).toBe("Summarize in 3 bullets.");
    expect(customReq.doc_ids).toEqual(["doc_a", "doc_b"]);
    expect(customReq.llm_override).toEqual({
      provider: "openrouter",
      fast_model: "google/gemini-2.5-flash",
      generation_model: "deepseek/deepseek-v3.2",
      temperature: 0.2,
    });

    const emptyCtxReq = buildQueryPayload({
      query: "What is RRF?",
      chatHistory: [],
      preset: "full",
      customAblation: DEFAULT_CUSTOM_ABLATION,
      maxCostUsd: null,
      customInstructions: "   ",
      docIds: [],
      llmOverride: { fast_model: "  " },
    });
    expect(emptyCtxReq.custom_instructions).toBeUndefined();
    expect(emptyCtxReq.doc_ids).toBeUndefined();
    expect(emptyCtxReq.llm_override).toBeUndefined();
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
    expect(screen.getByText(/Expanded to parent section from 2 child chunks/i)).toBeInTheDocument();
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
          pdf_ocr: {
            parser: "liteparse",
            default_mode: "auto",
            ocria_nano_installed: true,
          },
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
    expect(screen.getByText("OCRia-nano ready")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Auto" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "OCRia-nano" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Vision LLM" })).toBeInTheDocument();
    expect(
      screen.getByLabelText(/Hierarchical Parent-Child chunking/i)
    ).toBeInTheDocument();
    expect(screen.getByText("Firecrawl Local Quickstart")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Web" }));
    expect(screen.queryByText("Qdrant Hybrid Guide")).not.toBeInTheDocument();
    expect(screen.getByText("Firecrawl Local Quickstart")).toBeInTheDocument();
  });

  it("sends selected PDF OCR mode when re-indexing local corpus and displays OCR telemetry", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          collection: "adaptiverag_docs",
          source: "data/raw",
          documents_processed: 3,
          chunks_created: 14,
          points_upserted: 14,
          total_collection_points: 73,
          collection_created: false,
          elapsed_ms: 412,
          source_urls: [],
          parent_child_enabled: false,
          unique_parents_created: 0,
          ocr_pages_processed: 2,
          ocr_backends_used: ["ocria-nano"],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    );

    try {
      render(
        <KnowledgeBaseView
          apiUrl="http://localhost:8000"
          health={null}
          documentsData={null}
          documentsLoading={false}
          documentsError={null}
          onRefreshAll={vi.fn(async () => {})}
        />
      );

      fireEvent.click(screen.getByRole("button", { name: "OCRia-nano" }));
      fireEvent.click(screen.getByRole("button", { name: /Index data\/raw/i }));

      await waitFor(() => {
        expect(fetchSpy).toHaveBeenCalledTimes(1);
      });
      const [, reqInit] = fetchSpy.mock.calls[0];
      const sentBody = JSON.parse(String(reqInit?.body));
      expect(sentBody.ocr_mode).toBe("ocria_nano");

      expect(
        await screen.findByText(/2 scanned pages read with ocria-nano/i)
      ).toBeInTheDocument();
    } finally {
      fetchSpy.mockRestore();
    }
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
    fireEvent.click(screen.getByRole("button", { name: "Send feedback" }));
    await waitFor(() => {
      expect(onSubmitFeedback).toHaveBeenCalledWith("turn_lf_1", 0, "Missing Turkish summary");
    });
  });

  it("configures optional Layer-2 style presets and document scope in ChatWorkspace", () => {
    const onThreadContextChange = vi.fn();
    let currentCtx: ThreadContextConfig = { ...DEFAULT_THREAD_CONTEXT };

    const { rerender } = render(
      <ChatWorkspace
        turns={[]}
        selectedTurnIndex={0}
        isRunning={false}
        pendingQuery={null}
        liveSteps={[]}
        liveUsage={null}
        streamError={null}
        presetShortLabel="Full Agentic"
        useStreaming={true}
        budgetCapLabel={null}
        threadContext={currentCtx}
        availableDocuments={[
          {
            doc_id: "msa_contract",
            title: "Vendor MSA Contract",
            source_origin: "local",
            doc_type: "pdf",
            language: "en",
            chunk_count: 12,
            total_tokens: 2400,
            source_url: null,
            source_path: "msa_contract.pdf",
            file_name: "msa_contract.pdf",
          },
        ]}
        onThreadContextChange={(next) => {
          currentCtx = next;
          onThreadContextChange(next);
        }}
        onSubmitQuery={vi.fn()}
        onStopStream={vi.fn()}
        onInspectTurn={vi.fn()}
        onCitationClick={vi.fn()}
      />
    );

    // Drawer is collapsed by default
    expect(screen.queryByTestId("thread-context-panel")).not.toBeInTheDocument();

    // Open drawer via the response style chip
    fireEvent.click(screen.getByRole("button", { name: /Response style/i }));
    expect(screen.getByTestId("thread-context-panel")).toBeInTheDocument();
    expect(screen.getByText(/Citations and grounding rules always apply/i)).toBeInTheDocument();

    // Select the Executive brief preset
    fireEvent.click(screen.getByRole("button", { name: "Executive brief" }));
    expect(onThreadContextChange).toHaveBeenLastCalledWith(
      expect.objectContaining({
        instructionPreset: "executive",
      })
    );
    expect(resolveEffectiveInstructions(currentCtx)).toContain("executive summary");

    // Re-render with updated state and verify textarea + doc scope toggle
    rerender(
      <ChatWorkspace
        turns={[]}
        selectedTurnIndex={0}
        isRunning={false}
        pendingQuery={null}
        liveSteps={[]}
        liveUsage={null}
        streamError={null}
        presetShortLabel="Full Agentic"
        useStreaming={true}
        budgetCapLabel={null}
        threadContext={currentCtx}
        availableDocuments={[
          {
            doc_id: "msa_contract",
            title: "Vendor MSA Contract",
            source_origin: "local",
            doc_type: "pdf",
            language: "en",
            chunk_count: 12,
            total_tokens: 2400,
            source_url: null,
            source_path: "msa_contract.pdf",
            file_name: "msa_contract.pdf",
          },
        ]}
        onThreadContextChange={(next) => {
          currentCtx = next;
          onThreadContextChange(next);
        }}
        onSubmitQuery={vi.fn()}
        onStopStream={vi.fn()}
        onInspectTurn={vi.fn()}
        onCitationClick={vi.fn()}
      />
    );

    const textarea = screen.getByLabelText("Thread custom instructions");
    fireEvent.change(textarea, { target: { value: "Focus on termination risks." } });
    expect(currentCtx.instructionPreset).toBe("custom");
    expect(currentCtx.customInstructions).toBe("Focus on termination risks.");

    fireEvent.click(screen.getByRole("button", { name: /Vendor MSA Contract/i }));
    expect(currentCtx.scopedDocIds).toEqual(["msa_contract"]);
  });
});

describe("ModelSelectorPopover", () => {
  const sampleCatalog: ModelsCatalogResponse = {
    default_provider: "openrouter",
    default_fast_model: "qwen/qwen3-30b-a3b-instruct-2507",
    default_generation_model: "deepseek/deepseek-v3.2",
    default_temperature: 0.0,
    ollama_base_url: "http://localhost:8080/v1",
    allow_client_provider_keys: true,
    providers: [
      {
        id: "openrouter",
        label: "OpenRouter",
        is_local: false,
        requires_api_key: true,
        sdk_installed: true,
        configured: true,
        reachable: null,
        status_detail: null,
        default_fast_model: "qwen/qwen3-30b-a3b-instruct-2507",
        default_generation_model: "deepseek/deepseek-v3.2",
        recommended_fast_models: ["qwen/qwen3-30b-a3b-instruct-2507", "google/gemini-2.5-flash"],
        recommended_generation_models: ["deepseek/deepseek-v3.2", "anthropic/claude-sonnet-4"],
        detected_models: [],
      },
      {
        id: "ollama",
        label: "Local (Ollama / llama-server)",
        is_local: true,
        requires_api_key: false,
        sdk_installed: true,
        configured: true,
        reachable: true,
        status_detail: null,
        default_fast_model: "inkling-2-mini",
        default_generation_model: "inkling-2-mini",
        recommended_fast_models: ["inkling-2-mini"],
        recommended_generation_models: ["inkling-2-mini"],
        detected_models: ["inkling-2-mini"],
      },
      {
        id: "openai",
        label: "OpenAI",
        is_local: false,
        requires_api_key: true,
        sdk_installed: true,
        configured: false,
        reachable: null,
        status_detail: "Set OPENAI_API_KEY in .env or enter a session API key to use OpenAI.",
        default_fast_model: "gpt-4o-mini",
        default_generation_model: "gpt-4.1",
        recommended_fast_models: ["gpt-4o-mini"],
        recommended_generation_models: ["gpt-4.1"],
        detected_models: [],
      },
      {
        id: "gemini",
        label: "Google Gemini",
        is_local: false,
        requires_api_key: true,
        sdk_installed: false,
        configured: false,
        reachable: null,
        status_detail:
          "Install optional extra 'providers' (`uv sync --extra providers`) to use Google Gemini.",
        default_fast_model: "gemini-2.5-flash",
        default_generation_model: "gemini-2.5-pro",
        recommended_fast_models: ["gemini-2.5-flash"],
        recommended_generation_models: ["gemini-2.5-pro"],
        detected_models: [],
      },
    ],
  };

  it("opens from top header trigger, supports BYOK session API keys, and switches to detected local model", () => {
    let activeOverride: LLMOverride | null = null;
    let sessionKeys: Partial<Record<"openrouter" | "ollama" | "openai" | "gemini" | "anthropic", string>> = {};
    const onOverrideChange = vi.fn((next: LLMOverride | null) => {
      activeOverride = next;
    });
    const onProviderApiKeyChange = vi.fn((prov: "openrouter" | "ollama" | "openai" | "gemini" | "anthropic", key: string) => {
      sessionKeys = { ...sessionKeys, [prov]: key };
    });

    const { rerender } = render(
      <ModelSelectorPopover
        catalog={sampleCatalog}
        catalogLoading={false}
        health={null}
        override={activeOverride}
        providerApiKeys={sessionKeys}
        onOverrideChange={onOverrideChange}
        onProviderApiKeyChange={onProviderApiKeyChange}
        onRefreshCatalog={vi.fn()}
      />
    );

    const trigger = screen.getByRole("button", {
      name: "Configure LLM provider and models",
    });
    expect(trigger).toHaveTextContent("deepseek-v3.2");
    fireEvent.click(trigger);

    expect(screen.getByTestId("model-selector-popover")).toBeInTheDocument();
    expect(screen.getByText(/sent via X-Provider-API-Key header only/i)).toBeInTheDocument();

    // OpenAI is selectable when BYOK is enabled and prompts for a session-only API key
    const openAiBtn = screen.getByRole("button", { name: "OpenAI" });
    expect(openAiBtn).not.toBeDisabled();
    fireEvent.click(openAiBtn);
    expect(onOverrideChange).toHaveBeenLastCalledWith({
      provider: "openai",
      fast_model: "gpt-4o-mini",
      generation_model: "gpt-4.1",
      temperature: 0,
    });

    rerender(
      <ModelSelectorPopover
        catalog={sampleCatalog}
        catalogLoading={false}
        health={null}
        override={activeOverride}
        providerApiKeys={sessionKeys}
        onOverrideChange={onOverrideChange}
        onProviderApiKeyChange={onProviderApiKeyChange}
        onRefreshCatalog={vi.fn()}
      />
    );
    expect(screen.getByText("Key required")).toBeInTheDocument();
    const keyInput = screen.getByLabelText(/OpenAI API Key/i);
    fireEvent.change(keyInput, { target: { value: "sk-session-openai-123" } });
    expect(onProviderApiKeyChange).toHaveBeenLastCalledWith("openai", "sk-session-openai-123");

    // Re-render with active session key -> shows the session key badge
    rerender(
      <ModelSelectorPopover
        catalog={sampleCatalog}
        catalogLoading={false}
        health={null}
        override={activeOverride}
        providerApiKeys={sessionKeys}
        onOverrideChange={onOverrideChange}
        onProviderApiKeyChange={onProviderApiKeyChange}
        onRefreshCatalog={vi.fn()}
      />
    );
    expect(screen.getByText("Session key")).toBeInTheDocument();
    expect(screen.getByText("Session key active")).toBeInTheDocument();

    // Switch to Local runtime with detected inkling-2-mini model
    fireEvent.click(screen.getByRole("button", { name: "Local" }));
    expect(onOverrideChange).toHaveBeenLastCalledWith({
      provider: "ollama",
      fast_model: "inkling-2-mini",
      generation_model: "inkling-2-mini",
      temperature: 1.0,
    });

    // Re-render and test Reset button
    rerender(
      <ModelSelectorPopover
        catalog={sampleCatalog}
        catalogLoading={false}
        health={null}
        override={activeOverride}
        providerApiKeys={sessionKeys}
        onOverrideChange={onOverrideChange}
        onProviderApiKeyChange={onProviderApiKeyChange}
        onRefreshCatalog={vi.fn()}
      />
    );
    expect(screen.getByText("Custom")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Reset/i }));
    expect(onOverrideChange).toHaveBeenLastCalledWith(null);

    // When server disables BYOK (allow_client_provider_keys=false), unconfigured OpenAI is disabled
    rerender(
      <ModelSelectorPopover
        catalog={{ ...sampleCatalog, allow_client_provider_keys: false }}
        catalogLoading={false}
        health={null}
        override={null}
        providerApiKeys={{}}
        onOverrideChange={onOverrideChange}
        onProviderApiKeyChange={onProviderApiKeyChange}
        onRefreshCatalog={vi.fn()}
      />
    );
    expect(screen.getByRole("button", { name: "OpenAI" })).toBeDisabled();
    expect(screen.getByText(/BYOK disabled by server policy/i)).toBeInTheDocument();
  });
});
