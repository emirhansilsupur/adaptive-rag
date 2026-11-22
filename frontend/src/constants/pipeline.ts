import type { AblationConfig, PresetSelection } from "../types/api";

export interface NodeVisualMeta {
  label: string;
  shortLabel: string;
  color: string;
  bgClass: string;
  textClass: string;
  borderClass: string;
  stage: "routing" | "retrieval" | "verification" | "generation";
}

export const NODE_META: Record<string, NodeVisualMeta> = {
  classify_and_plan: {
    label: "Classify & Plan",
    shortLabel: "Router",
    color: "#3B82F6",
    bgClass: "bg-blue-500/12",
    textClass: "text-blue-400",
    borderClass: "border-blue-500/30",
    stage: "routing",
  },
  conversational_respond: {
    label: "Conversational Respond",
    shortLabel: "Direct Chat",
    color: "#0EA5E9",
    bgClass: "bg-sky-500/12",
    textClass: "text-sky-400",
    borderClass: "border-sky-500/30",
    stage: "generation",
  },
  decompose_queries: {
    label: "Decompose Sub-Queries",
    shortLabel: "Decompose",
    color: "#6366F1",
    bgClass: "bg-indigo-500/12",
    textClass: "text-indigo-400",
    borderClass: "border-indigo-500/30",
    stage: "routing",
  },
  hybrid_retrieve: {
    label: "Hybrid Retrieve (Dense + BM25)",
    shortLabel: "Hybrid RRF",
    color: "#14B8A6",
    bgClass: "bg-teal-500/12",
    textClass: "text-teal-400",
    borderClass: "border-teal-500/30",
    stage: "retrieval",
  },
  rerank_and_select_k: {
    label: "Rerank & Dynamic Top-K",
    shortLabel: "Rerank + K",
    color: "#06B6D4",
    bgClass: "bg-cyan-500/12",
    textClass: "text-cyan-400",
    borderClass: "border-cyan-500/30",
    stage: "retrieval",
  },
  score_confidence: {
    label: "Confidence & CRAG Gate",
    shortLabel: "CRAG Gate",
    color: "#F59E0B",
    bgClass: "bg-amber-500/12",
    textClass: "text-amber-400",
    borderClass: "border-amber-500/30",
    stage: "verification",
  },
  corrective_rewrite: {
    label: "Corrective Query Rewrite",
    shortLabel: "Rewrite",
    color: "#F97316",
    bgClass: "bg-orange-500/12",
    textClass: "text-orange-400",
    borderClass: "border-orange-500/30",
    stage: "verification",
  },
  web_fallback_search: {
    label: "Web Fallback Search",
    shortLabel: "Web Search",
    color: "#A855F7",
    bgClass: "bg-purple-500/12",
    textClass: "text-purple-400",
    borderClass: "border-purple-500/30",
    stage: "retrieval",
  },
  generate_answer: {
    label: "Grounded Generation",
    shortLabel: "Generate",
    color: "#10B981",
    bgClass: "bg-emerald-500/12",
    textClass: "text-emerald-400",
    borderClass: "border-emerald-500/30",
    stage: "generation",
  },
  validate_citations: {
    label: "Two-Tier Citation Audit",
    shortLabel: "Cite Audit",
    color: "#22C55E",
    bgClass: "bg-green-500/12",
    textClass: "text-green-400",
    borderClass: "border-green-500/30",
    stage: "verification",
  },
  regenerate_strict: {
    label: "Strict Grounded Regeneration",
    shortLabel: "Strict Regen",
    color: "#F43F5E",
    bgClass: "bg-rose-500/12",
    textClass: "text-rose-400",
    borderClass: "border-rose-500/30",
    stage: "generation",
  },
  abstain_response: {
    label: "Honest Abstention",
    shortLabel: "Abstain",
    color: "#64748B",
    bgClass: "bg-slate-500/15",
    textClass: "text-slate-400",
    borderClass: "border-slate-500/30",
    stage: "generation",
  },
};

export function getNodeMeta(node: string): NodeVisualMeta {
  return (
    NODE_META[node] ?? {
      label: node.replace(/_/g, " "),
      shortLabel: node,
      color: "#64748B",
      bgClass: "bg-slate-500/15",
      textClass: "text-slate-400",
      borderClass: "border-slate-500/30",
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
    description: "All routing, hybrid RRF, rerank, dynamic K, CRAG, web fallback & citation audit loops.",
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

export const DEFAULT_API_URL = (
  import.meta.env.VITE_API_URL || "http://localhost:8000"
).replace(/\/+$/, "");
