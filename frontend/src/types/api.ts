export type QueryType = "conversational" | "factual" | "analytical";
export type SourceType = "local" | "web";
export type ConfidenceLevel = "high" | "medium" | "low";
export type CRAGGrade = "sufficient" | "partial" | "insufficient";
export type ConfidenceAction = "proceed" | "rewrite" | "web_fallback" | "abstain";
export type ClaimVerdict =
  | "supported"
  | "partially_supported"
  | "unsupported"
  | "contradicted";

export type AblationPreset =
  | "full"
  | "adaptive_routing"
  | "hybrid_rerank"
  | "naive_dense";

export type PresetSelection = AblationPreset | "custom";

export interface AblationConfig {
  retrieval_mode: "hybrid" | "dense_only" | "sparse_only";
  enable_query_routing: boolean;
  enable_multi_query: boolean;
  enable_rerank: boolean;
  enable_dynamic_k: boolean;
  fixed_top_k: number;
  enable_confidence_gating: boolean;
  enable_web_fallback: boolean;
  enable_citation_validation: boolean;
  enable_parent_child?: boolean | null;
  language_filter: "en" | "tr" | null;
}

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface QueryRequest {
  query: string;
  chat_history: ChatMessage[];
  preset?: AblationPreset;
  ablation?: AblationConfig;
  max_cost_usd?: number;
  session_id?: string;
  user_id?: string;
}

export interface StrategyProfile {
  needs_retrieval: boolean;
  max_sub_queries: number;
  dense_prefetch_limit: number;
  sparse_prefetch_limit: number;
  fusion_limit: number;
  rerank_min_k: number;
  rerank_max_k: number;
  tau_min: number;
  alpha_relative: number;
  delta_gap: number;
  max_context_tokens: number;
}

export interface QueryPlan {
  original_query: string;
  standalone_query: string;
  query_type: QueryType;
  needs_retrieval: boolean;
  language: "tr" | "en" | "other";
  requires_fresh_web_info: boolean;
  reasoning: string;
  strategy: StrategyProfile;
  sub_queries: string[];
}

export interface EvidenceChunk {
  citation_id: string;
  chunk_id: string;
  text: string;
  source_type: SourceType;
  doc_id: string;
  title: string;
  section_path: string;
  url: string | null;
  metadata: Record<string, unknown>;
  dense_score: number | null;
  sparse_score: number | null;
  fusion_score: number | null;
  rerank_raw_score: number | null;
  rerank_score: number | null;
  matched_sub_queries: string[];
  token_count: number;
  parent_id?: string | null;
  parent_section_path?: string | null;
  parent_token_count?: number | null;
  expanded_to_parent?: boolean;
  matched_child_ids?: string[];
}

export interface DynamicKDecision {
  candidate_count: number;
  selected_count: number;
  cutoff_reason: string;
  top_score: number;
  cutoff_score: number;
  total_context_tokens: number;
  dropped_chunk_ids: string[];
  parent_expansion_applied?: boolean;
  collapsed_child_count?: number;
  expanded_parent_count?: number;
}

export interface ConfidenceAssessment {
  top1_score: number;
  mean_topk_score: number;
  subquery_coverage: number;
  deterministic_score: number;
  crag_grade: CRAGGrade | null;
  final_score: number;
  level: ConfidenceLevel;
  recommended_action: ConfidenceAction;
  reasoning: string;
}

export interface ClaimVerification {
  claim: string;
  cited_ids: string[];
  verdict: ClaimVerdict;
  explanation: string;
}

export interface CitationReport {
  total_citations_found: number;
  valid_citation_ids: string[];
  invalid_citation_ids: string[];
  uncited_factual_sentences: string[];
  claim_verifications: ClaimVerification[];
  citation_precision: number;
  claim_support_rate: number;
  passed: boolean;
  regeneration_triggered: boolean;
}

export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  estimated_cost_usd: number;
  cache_hits: number;
}

export interface NodeTraceEvent {
  node: string;
  latency_ms: number;
  summary: string;
  metadata: Record<string, unknown>;
}

export interface FinalAnswer {
  query: string;
  answer: string;
  query_type: QueryType;
  language: string;
  query_plan: QueryPlan | null;
  evidence_chunks: EvidenceChunk[];
  dynamic_k_decision: DynamicKDecision | null;
  confidence: ConfidenceAssessment | null;
  citation_report: CitationReport | null;
  fallback_used: boolean;
  abstained: boolean;
  rewrite_attempts: number;
  regeneration_attempts: number;
  total_latency_ms: number;
  usage: TokenUsage;
  trace: NodeTraceEvent[];
  trace_id?: string | null;
  trace_url?: string | null;
}

export interface FeedbackRequest {
  trace_id: string;
  score: 0 | 1;
  comment?: string;
  session_id?: string;
  query?: string;
}

export interface FeedbackResponse {
  status: "recorded";
  trace_id: string;
  score: number;
  comment: string | null;
  persisted_to_langfuse: boolean;
  langfuse_error: string | null;
}

export interface LangfusePromptManagementHealthInfo {
  enabled?: boolean;
  active?: boolean;
  mode?: "langfuse" | "local_fallback";
  label?: string;
  cache_ttl_seconds?: number;
  managed_prompts?: number;
}

export interface LangfuseHealthInfo {
  enabled?: boolean;
  configured?: boolean;
  mode?: "cloud" | "self_hosted" | "misconfigured" | "disabled";
  host?: string;
  environment?: string;
  prompt_management?: LangfusePromptManagementHealthInfo;
}

export interface QdrantHealthInfo {
  status: string;
  collection?: string;
  collection_exists?: boolean;
  points_count?: number;
  dense_model?: string;
  sparse_model?: string;
  error?: string;
}

export interface FirecrawlHealthInfo {
  status?: string;
  base_url?: string | null;
  error?: string;
  http_status?: number;
}

export interface SecurityHealthInfo {
  api_key_auth_enabled?: boolean;
  rate_limit_enabled?: boolean;
  rate_limit_requests_per_minute?: number;
  guardrails_enabled?: boolean;
  prompt_injection_protection?: boolean;
  pii_mode?: "off" | "redact" | "block";
}

export interface ParentChildHealthInfo {
  enabled?: boolean;
  parent_max_tokens?: number;
  child_max_tokens?: number;
  child_overlap_tokens?: number;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  qdrant: QdrantHealthInfo;
  firecrawl: FirecrawlHealthInfo;
  langfuse?: LangfuseHealthInfo;
  security?: SecurityHealthInfo;
  parent_child_retrieval?: ParentChildHealthInfo;
  llm_configured: boolean;
  llm_error: string | null;
  llm_provider: string;
  fast_model: string;
  generation_model: string;
  dense_embedding_model: string;
  sparse_embedding_model: string;
  reranker_model: string;
  web_fallback_enabled: boolean;
  web_fallback_providers: string[];
  langfuse_enabled: boolean;
  llm_cache_enabled: boolean;
  default_max_run_cost_usd: number;
}

export interface IngestRequest {
  corpus_dir?: string;
  urls?: string[];
  crawl_url?: string;
  crawl_limit?: number;
  crawl_max_depth?: number;
  only_main_content?: boolean;
  recreate_collection?: boolean;
  max_tokens?: number;
  overlap_tokens?: number;
  parent_child?: boolean | null;
  parent_max_tokens?: number;
  child_max_tokens?: number;
  child_overlap_tokens?: number;
  batch_size?: number;
}

export interface IngestResponse {
  collection: string;
  source: string;
  documents_processed: number;
  chunks_created: number;
  points_upserted: number;
  total_collection_points: number;
  collection_created: boolean;
  elapsed_ms: number;
  source_urls: string[];
  parent_child_enabled?: boolean;
  unique_parents_created?: number;
}

export interface IndexedDocumentInfo {
  doc_id: string;
  title: string;
  source_origin: "local" | "web";
  doc_type: string;
  language: string;
  chunk_strategy?: "flat" | "parent_child";
  chunk_count: number;
  total_tokens: number;
  source_url: string | null;
  source_path: string | null;
  file_name: string | null;
}

export interface DocumentListResponse {
  collection: string;
  total_documents: number;
  total_chunks: number;
  documents: IndexedDocumentInfo[];
}

export interface DeleteDocumentResponse {
  collection: string;
  doc_id: string;
  deleted_chunks: number;
  remaining_collection_points: number;
}

export interface SseNodeUpdatePayload {
  node: string;
  trace: NodeTraceEvent[];
  usage: Partial<TokenUsage>;
}

export interface SseErrorPayload {
  code?: string;
  message?: string;
}

export interface LiveStreamStep {
  id: string;
  node: string;
  latency_ms: number;
  summary: string;
  metadata: Record<string, unknown>;
}

export interface TurnFeedback {
  score: 0 | 1;
  comment?: string;
  persistedToLangfuse: boolean;
  submittedAt: number;
}

export interface ChatTurn {
  id: string;
  query: string;
  createdAt: number;
  presetLabel: string;
  result: FinalAnswer;
  feedback?: TurnFeedback;
}

export interface ChatSession {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  turns: ChatTurn[];
}

export type InspectorTab = "trace" | "dynamic_k" | "confidence" | "evidence" | "citations";
export type AppView = "workbench" | "compare" | "knowledge";

export type CompareSideKey = "left" | "right";

export interface CompareSideState {
  preset: AblationPreset;
  presetLabel: string;
  status: "idle" | "running" | "done" | "error";
  liveSteps: LiveStreamStep[];
  liveUsage: Partial<TokenUsage> | null;
  result: FinalAnswer | null;
  error: string | null;
  feedback?: TurnFeedback;
}

export interface CompareRun {
  id: string;
  query: string;
  createdAt: number;
  left: CompareSideState;
  right: CompareSideState;
}
