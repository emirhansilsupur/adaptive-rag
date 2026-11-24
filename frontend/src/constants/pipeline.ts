import type { AblationConfig, AblationPreset, PresetSelection } from "../types/api";

export interface NodeVisualMeta {
  label: string;
  shortLabel: string;
  /** Restrained 4-stage indicator dot color (routing, retrieval, verification, generation) */
  color: string;
  bgClass: string;
  textClass: string;
  borderClass: string;
  stage: "routing" | "retrieval" | "verification" | "generation";
}

const STAGE_SURFACE = {
  routing: {
    color: "#a1a1aa",
    bgClass: "bg-[var(--bg-surface)]",
    textClass: "text-[var(--text-secondary)]",
    borderClass: "border-[var(--border-subtle)]",
  },
  retrieval: {
    color: "#38bdf8",
    bgClass: "bg-[var(--bg-surface)]",
    textClass: "text-[var(--text-secondary)]",
    borderClass: "border-[var(--border-subtle)]",
  },
  verification: {
    color: "#f59e0b",
    bgClass: "bg-[var(--bg-surface)]",
    textClass: "text-[var(--text-secondary)]",
    borderClass: "border-[var(--border-subtle)]",
  },
  generation: {
    color: "#10b981",
    bgClass: "bg-[var(--bg-surface)]",
    textClass: "text-[var(--text-secondary)]",
    borderClass: "border-[var(--border-subtle)]",
  },
} as const;

export const NODE_META: Record<string, NodeVisualMeta> = {
  classify_and_plan: {
    label: "Classify & Plan",
    shortLabel: "Router",
    ...STAGE_SURFACE.routing,
    stage: "routing",
  },
  conversational_respond: {
    label: "Conversational Respond",
    shortLabel: "Direct Chat",
    ...STAGE_SURFACE.generation,
    stage: "generation",
  },
  decompose_queries: {
    label: "Decompose Sub-Queries",
    shortLabel: "Decompose",
    ...STAGE_SURFACE.routing,
    stage: "routing",
  },
  hybrid_retrieve: {
    label: "Hybrid Retrieve (Dense + BM25)",
    shortLabel: "Hybrid RRF",
    ...STAGE_SURFACE.retrieval,
    stage: "retrieval",
  },
  rerank_and_select_k: {
    label: "Rerank & Dynamic Top-K",
    shortLabel: "Rerank + K",
    ...STAGE_SURFACE.retrieval,
    stage: "retrieval",
  },
  score_confidence: {
    label: "Confidence & CRAG Gate",
    shortLabel: "CRAG Gate",
    ...STAGE_SURFACE.verification,
    stage: "verification",
  },
  corrective_rewrite: {
    label: "Corrective Query Rewrite",
    shortLabel: "Rewrite",
    ...STAGE_SURFACE.verification,
    stage: "verification",
  },
  web_fallback_search: {
    label: "Web Fallback Search",
    shortLabel: "Web Search",
    ...STAGE_SURFACE.retrieval,
    stage: "retrieval",
  },
  generate_answer: {
    label: "Grounded Generation",
    shortLabel: "Generate",
    ...STAGE_SURFACE.generation,
    stage: "generation",
  },
  validate_citations: {
    label: "Two-Tier Citation Audit",
    shortLabel: "Cite Audit",
    ...STAGE_SURFACE.verification,
    stage: "verification",
  },
  regenerate_strict: {
    label: "Strict Grounded Regeneration",
    shortLabel: "Strict Regen",
    color: "#f43f5e",
    bgClass: "bg-[var(--bg-surface)]",
    textClass: "text-[var(--text-secondary)]",
    borderClass: "border-[var(--border-subtle)]",
    stage: "generation",
  },
  abstain_response: {
    label: "Honest Abstention",
    shortLabel: "Abstain",
    color: "#71717a",
    bgClass: "bg-[var(--bg-surface)]",
    textClass: "text-[var(--text-secondary)]",
    borderClass: "border-[var(--border-subtle)]",
    stage: "generation",
  },
};

export function getNodeMeta(node: string): NodeVisualMeta {
  return (
    NODE_META[node] ?? {
      label: node.replace(/_/g, " "),
      shortLabel: node,
      ...STAGE_SURFACE.routing,
      stage: "retrieval",
    }
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
    description: "Query routing, hybrid RRF, rerank, dynamic K, CRAG, web fallback, and citation audit.",
  },
  {
    value: "adaptive_routing",
    label: "Adaptive Routing + Dynamic K",
    shortLabel: "Adaptive K",
    description: "Query-aware routing and dynamic top-k without corrective rewrite or web fallback.",
  },
  {
    value: "hybrid_rerank",
    label: "Hybrid BM25+Dense + Rerank",
    shortLabel: "Hybrid Static",
    description: "Native Qdrant hybrid search + cross-encoder reranking with static top-k.",
  },
  {
    value: "naive_dense",
    label: "Naive Dense Baseline",
    shortLabel: "Naive Dense",
    description: "Dense-only cosine retrieval with static top-k for baseline comparison.",
  },
  {
    value: "custom",
    label: "Custom Feature Switches",
    shortLabel: "Custom",
    description: "Granular ablation toggles for every pipeline stage.",
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
    title: "Qdrant Hybrid RRF Formula",
    category: "Factual",
    lang: "EN",
    query:
      "What is the Reciprocal Rank Fusion (RRF) formula and how does Qdrant combine dense and sparse prefetch results?",
  },
  {
    id: "tr-hybrid",
    title: "Hibrit Arama & Cross-Encoder Rerank",
    category: "Analytical",
    lang: "TR",
    query:
      "RAG sistemlerinde hibrit arama ve cross-encoder reranking neden birlikte kullanılır; gecikme ve doğruluk dengesini karşılaştır?",
  },
  {
    id: "en-langgraph",
    title: "Cyclic LangGraph vs Linear RAG",
    category: "Analytical",
    lang: "EN",
    query:
      "Compare cyclic LangGraph state graphs against linear RAG chains for corrective retrieval and citation verification.",
  },
  {
    id: "tr-conv",
    title: "Doğrudan Sohbet Rotası",
    category: "Conversational",
    lang: "TR",
    query: "Merhaba! Bugün nasılsın, hangi konularda yardımcı olabilirsin?",
  },
  {
    id: "en-web",
    title: "Live Web Fallback Trigger",
    category: "Web Fallback",
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
    label: "Naive Dense vs Full",
    left: "naive_dense",
    right: "full",
  },
  {
    id: "hybrid-vs-adaptive",
    label: "Hybrid Static vs Adaptive K",
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
    title: "Factual Dynamic Top-K Savings",
    highlight: "Context Tokens ↓",
    lang: "EN",
    query:
      "What is the default value of ef_construct in Qdrant HNSW and how does it trade off index build time versus recall?",
  },
  {
    id: "cmp-analytical-multi",
    title: "Multi-Doc Architectural Trade-Off",
    highlight: "Multi-Query + Audit",
    lang: "EN",
    query:
      "Compare Qdrant Filterable HNSW and PostgreSQL pgvector for hybrid RAG.",
  },
  {
    id: "cmp-tr-conv",
    title: "Zero-Retrieval Fast Path",
    highlight: "k=5 → k=0",
    lang: "TR",
    query: "Merhaba! Bana kendini kısaca tanıtabilir misin?",
  },
  {
    id: "cmp-unanswerable",
    title: "Out-of-Corpus Hallucination Trap",
    highlight: "Abstention Guard",
    lang: "EN",
    query:
      "What are the exact RAM and HNSW defaults of the fictional QuantumVectorX-9000 engine released in 2029?",
  },
];

const rawViteApiUrl = import.meta.env.VITE_API_URL as string | undefined;

export const DEFAULT_API_URL = (
  rawViteApiUrl !== undefined ? rawViteApiUrl : "http://localhost:8000"
).replace(/\/+$/, "");
