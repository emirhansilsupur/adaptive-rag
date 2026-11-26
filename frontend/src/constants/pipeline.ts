import type {
  AblationConfig,
  AblationPreset,
  InstructionPresetId,
  PresetSelection,
  ThreadContextConfig,
} from "../types/api";

export type PipelineStage = "routing" | "retrieval" | "verification" | "generation";

export interface NodeVisualMeta {
  label: string;
  shortLabel: string;
  /** CSS colour (theme-aware custom property) for the node's pipeline stage. */
  color: string;
  stage: PipelineStage;
}

export const STAGE_COLORS: Record<PipelineStage, string> = {
  routing: "var(--stage-route)",
  retrieval: "var(--stage-retrieve)",
  verification: "var(--stage-verify)",
  generation: "var(--stage-generate)",
};

export const STAGE_LABELS: Record<PipelineStage, string> = {
  routing: "Route",
  retrieval: "Retrieve",
  verification: "Verify",
  generation: "Write",
};

function node(
  label: string,
  shortLabel: string,
  stage: PipelineStage,
  color: string = STAGE_COLORS[stage]
): NodeVisualMeta {
  return { label, shortLabel, stage, color };
}

export const NODE_META: Record<string, NodeVisualMeta> = {
  classify_and_plan: node("Classify and plan", "Router", "routing"),
  conversational_respond: node("Conversational reply", "Direct reply", "generation"),
  decompose_queries: node("Decompose into sub-queries", "Decompose", "routing"),
  hybrid_retrieve: node("Hybrid retrieval (dense + BM25)", "Hybrid search", "retrieval"),
  rerank_and_select_k: node("Rerank and dynamic top-k", "Rerank + k", "retrieval"),
  score_confidence: node("Confidence and CRAG gate", "CRAG gate", "verification"),
  corrective_rewrite: node("Corrective query rewrite", "Rewrite", "verification"),
  web_fallback_search: node("Web fallback search", "Web search", "retrieval"),
  generate_answer: node("Grounded generation", "Generate", "generation"),
  validate_citations: node("Two-tier citation audit", "Citation audit", "verification"),
  regenerate_strict: node(
    "Strict grounded regeneration",
    "Strict regen",
    "generation",
    "var(--stage-repair)"
  ),
  abstain_response: node("Honest abstention", "Abstain", "generation", "var(--stage-route)"),
};

export function getNodeMeta(nodeName: string): NodeVisualMeta {
  return (
    NODE_META[nodeName] ?? node(nodeName.replace(/_/g, " "), nodeName, "retrieval")
  );
}

export interface PresetOption {
  value: PresetSelection;
  label: string;
  shortLabel: string;
  description: string;
}

export const PRESET_OPTIONS: PresetOption[] = [
  {
    value: "full",
    label: "Full AdaptiveRAG",
    shortLabel: "Full Agentic",
    description: "Routing, hybrid search, reranking, dynamic top-k, CRAG gate, web fallback and citation audit.",
  },
  {
    value: "adaptive_routing",
    label: "Adaptive routing + dynamic k",
    shortLabel: "Adaptive K",
    description: "Query-aware routing and dynamic top-k, without corrective rewrite or web fallback.",
  },
  {
    value: "hybrid_rerank",
    label: "Hybrid BM25 + dense + rerank",
    shortLabel: "Hybrid Static",
    description: "Qdrant hybrid search with cross-encoder reranking and a fixed top-k.",
  },
  {
    value: "naive_dense",
    label: "Naive dense baseline",
    shortLabel: "Naive Dense",
    description: "Dense-only cosine retrieval with a fixed top-k. Use it as the baseline.",
  },
  {
    value: "custom",
    label: "Custom switches",
    shortLabel: "Custom",
    description: "Switch individual pipeline stages on or off.",
  },
];

export const DEFAULT_CUSTOM_ABLATION: AblationConfig = {
  retrieval_mode: "hybrid",
  enable_query_routing: true,
  enable_multi_query: true,
  enable_rerank: true,
  enable_dynamic_k: true,
  fixed_top_k: 5,
  enable_confidence_gating: true,
  enable_web_fallback: true,
  enable_citation_validation: true,
  language_filter: null,
};

export interface DemoQueryItem {
  id: string;
  title: string;
  category: string;
  lang: "EN" | "TR";
  query: string;
}

export const DEMO_QUERIES: DemoQueryItem[] = [
  {
    id: "en-rrf",
    title: "Qdrant hybrid RRF formula",
    category: "Factual",
    lang: "EN",
    query:
      "What is the Reciprocal Rank Fusion (RRF) formula and how does Qdrant combine dense and sparse prefetch results?",
  },
  {
    id: "tr-hybrid",
    title: "Hibrit arama ve cross-encoder rerank",
    category: "Analytical",
    lang: "TR",
    query:
      "RAG sistemlerinde hibrit arama ve cross-encoder reranking neden birlikte kullanılır; gecikme ve doğruluk dengesini karşılaştır?",
  },
  {
    id: "en-langgraph",
    title: "Cyclic LangGraph vs linear RAG",
    category: "Analytical",
    lang: "EN",
    query:
      "Compare cyclic LangGraph state graphs against linear RAG chains for corrective retrieval and citation verification.",
  },
  {
    id: "tr-conv",
    title: "Doğrudan sohbet rotası",
    category: "Conversational",
    lang: "TR",
    query: "Merhaba! Bugün nasılsın, hangi konularda yardımcı olabilirsin?",
  },
  {
    id: "en-web",
    title: "Live web fallback trigger",
    category: "Web fallback",
    lang: "EN",
    query: "What is the latest stable release version of Qdrant available today?",
  },
];

export interface ArenaPresetOption {
  value: AblationPreset;
  label: string;
  shortLabel: string;
  description: string;
}

export const ARENA_PRESET_OPTIONS: ArenaPresetOption[] = PRESET_OPTIONS.filter(
  (opt): opt is ArenaPresetOption => opt.value !== "custom"
);

export interface ComparePairPreset {
  id: string;
  label: string;
  left: AblationPreset;
  right: AblationPreset;
}

export const COMPARE_PAIR_PRESETS: ComparePairPreset[] = [
  {
    id: "naive-vs-full",
    label: "Naive vs Full",
    left: "naive_dense",
    right: "full",
  },
  {
    id: "hybrid-vs-adaptive",
    label: "Hybrid vs Adaptive K",
    left: "hybrid_rerank",
    right: "adaptive_routing",
  },
  {
    id: "adaptive-vs-full",
    label: "Adaptive K vs Full",
    left: "adaptive_routing",
    right: "full",
  },
];

export interface CompareDemoItem {
  id: string;
  title: string;
  highlight: string;
  lang: "EN" | "TR";
  query: string;
}

export const COMPARE_DEMO_QUERIES: CompareDemoItem[] = [
  {
    id: "cmp-factual-tokens",
    title: "Factual dynamic top-k savings",
    highlight: "Fewer context tokens",
    lang: "EN",
    query:
      "What is the default value of ef_construct in Qdrant HNSW and how does it trade off index build time versus recall?",
  },
  {
    id: "cmp-analytical-multi",
    title: "Multi-document trade-off",
    highlight: "Multi-query and audit",
    lang: "EN",
    query:
      "Compare Qdrant Filterable HNSW and PostgreSQL pgvector for hybrid RAG.",
  },
  {
    id: "cmp-tr-conv",
    title: "Zero-retrieval fast path",
    highlight: "k=5 → k=0",
    lang: "TR",
    query: "Merhaba! Bana kendini kısaca tanıtabilir misin?",
  },
  {
    id: "cmp-unanswerable",
    title: "Out-of-corpus hallucination trap",
    highlight: "Abstention guard",
    lang: "EN",
    query:
      "What are the exact RAM and HNSW defaults of the fictional QuantumVectorX-9000 engine released in 2029?",
  },
];

export interface InstructionPresetOption {
  id: InstructionPresetId;
  label: string;
  shortLabel: string;
  description: string;
  instructions: string;
}

export const MAX_CUSTOM_INSTRUCTIONS_CHARS = 1500;

export const INSTRUCTION_PRESETS: InstructionPresetOption[] = [
  {
    id: "default",
    label: "Default",
    shortLabel: "default",
    description: "The standard balanced, grounded response style.",
    instructions: "",
  },
  {
    id: "executive",
    label: "Executive brief",
    shortLabel: "executive",
    description: "Two or three summary bullets up front, with key numbers, decisions and risks.",
    instructions:
      "Lead with a concise 2-3 bullet executive summary, highlight key decisions, numbers, and risks clearly, and keep prose crisp.",
  },
  {
    id: "technical",
    label: "Deep technical",
    shortLabel: "technical",
    description: "Keeps exact formulas and parameters, and explains architectural trade-offs.",
    instructions:
      "Use precise engineering terminology, preserve exact formulas, parameters, and code details, and explain underlying architectural trade-offs.",
  },
  {
    id: "analyst",
    label: "Policy and legal",
    shortLabel: "policy",
    description: "Breaks down clauses, obligations, thresholds and exceptions.",
    instructions:
      "Break down clauses, obligations, thresholds, exceptions, and potential risks in a structured table or bullet list using plain, unambiguous language.",
  },
  {
    id: "tutor",
    label: "Study tutor",
    shortLabel: "tutor",
    description: "Step-by-step explanations with clear takeaways and examples.",
    instructions:
      "Explain concepts step-by-step in clear, accessible language with structured takeaways and concrete examples grounded in the sources.",
  },
  {
    id: "custom",
    label: "Custom",
    shortLabel: "custom",
    description: "Write your own tone, format or audience instructions for this thread.",
    instructions: "",
  },
];

export const DEFAULT_THREAD_CONTEXT: ThreadContextConfig = {
  instructionPreset: "default",
  customInstructions: "",
  scopedDocIds: [],
};

export function resolveEffectiveInstructions(
  ctx?: ThreadContextConfig | null
): string | undefined {
  if (!ctx || ctx.instructionPreset === "default") {
    return undefined;
  }
  const direct = ctx.customInstructions.trim();
  if (direct) {
    return direct.slice(0, MAX_CUSTOM_INSTRUCTIONS_CHARS);
  }
  const preset = INSTRUCTION_PRESETS.find((p) => p.id === ctx.instructionPreset);
  const fallback = preset?.instructions.trim() ?? "";
  return fallback || undefined;
}

const rawViteApiUrl = import.meta.env.VITE_API_URL as string | undefined;

export const DEFAULT_API_URL = (
  rawViteApiUrl !== undefined ? rawViteApiUrl : "http://localhost:8000"
).replace(/\/+$/, "");

/** Shown when the user stops a run; rendered as a neutral notice, not an error. */
export const RUN_STOPPED_MESSAGE = "You stopped this run before it finished.";
