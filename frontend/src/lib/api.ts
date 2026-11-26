import type {
  AblationConfig,
  ChatMessage,
  DeleteDocumentResponse,
  DocumentListResponse,
  FeedbackRequest,
  FeedbackResponse,
  FinalAnswer,
  HealthResponse,
  IngestRequest,
  IngestResponse,
  PdfOcrMode,
  PresetSelection,
  QueryRequest,
  SseErrorPayload,
  SseNodeUpdatePayload,
} from "../types/api";
import { SseStreamParser } from "./sse";

let activeStudioApiKey = "";

export function setStudioApiKey(apiKey: string): void {
  activeStudioApiKey = apiKey.trim();
}

export function getStudioApiKey(): string {
  return activeStudioApiKey;
}

export function buildAuthHeaders(
  base: Record<string, string> = {},
  explicitApiKey?: string
): Record<string, string> {
  const resolved = (explicitApiKey ?? activeStudioApiKey).trim();
  if (!resolved) {
    return base;
  }
  return { ...base, "X-API-Key": resolved };
}

function normalizeBase(apiBaseUrl: string): string {
  return apiBaseUrl.trim().replace(/\/+$/, "");
}

async function parseErrorDetail(resp: Response): Promise<string> {
  const raw = await resp.text();
  if (!raw) {
    return `HTTP ${resp.status} ${resp.statusText}`;
  }
  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    if (typeof parsed.detail === "string" && parsed.detail.trim()) {
      return `HTTP ${resp.status}: ${parsed.detail}`;
    }
    if (Array.isArray(parsed.detail)) {
      const first = parsed.detail[0] as Record<string, unknown> | undefined;
      if (first && typeof first.msg === "string") {
        return `HTTP ${resp.status}: ${first.msg}`;
      }
    }
  } catch {
    // keep raw string
  }
  return `HTTP ${resp.status}: ${raw.slice(0, 300)}`;
}

export function buildQueryPayload(params: {
  query: string;
  chatHistory: ChatMessage[];
  preset: PresetSelection;
  customAblation: AblationConfig;
  maxCostUsd: number | null;
  sessionId?: string;
  customInstructions?: string | null;
  docIds?: string[] | null;
}): QueryRequest {
  const payload: QueryRequest = {
    query: params.query.trim(),
    chat_history: params.chatHistory,
  };
  if (params.preset === "custom") {
    payload.ablation = params.customAblation;
  } else {
    payload.preset = params.preset;
  }
  if (params.maxCostUsd !== null && Number.isFinite(params.maxCostUsd)) {
    payload.max_cost_usd = params.maxCostUsd;
  }
  if (params.customInstructions && params.customInstructions.trim()) {
    payload.custom_instructions = params.customInstructions.trim();
  }
  if (params.docIds && params.docIds.length > 0) {
    const uniqueDocIds = Array.from(
      new Set(params.docIds.map((d) => d.trim()).filter(Boolean))
    );
    if (uniqueDocIds.length > 0) {
      payload.doc_ids = uniqueDocIds;
    }
  }
  if (params.sessionId && params.sessionId.trim()) {
    payload.session_id = params.sessionId.trim();
  }
  return payload;
}

export function parseUrlLines(rawText: string): string[] {
  const seen = new Set<string>();
  const urls: string[] = [];
  for (const line of rawText.split(/\r?\n/)) {
    const cleaned = line.trim();
    if (cleaned && !seen.has(cleaned)) {
      seen.add(cleaned);
      urls.push(cleaned);
    }
  }
  return urls;
}

export function buildWebIngestPayload(params: {
  mode: "scrape" | "crawl";
  urlsText?: string;
  crawlUrl?: string;
  crawlLimit?: number;
  crawlMaxDepth?: number;
  onlyMainContent?: boolean;
  recreateCollection?: boolean;
  parentChild?: boolean;
}): IngestRequest {
  const payload: IngestRequest = {
    only_main_content: params.onlyMainContent ?? true,
    recreate_collection: params.recreateCollection ?? false,
  };
  if (params.parentChild !== undefined) {
    payload.parent_child = params.parentChild;
  }
  if (params.mode === "crawl") {
    const cleaned = (params.crawlUrl ?? "").trim();
    if (!cleaned) {
      throw new Error("Enter a public root documentation URL to crawl.");
    }
    payload.crawl_url = cleaned;
    payload.crawl_limit = params.crawlLimit ?? 8;
    if (params.crawlMaxDepth !== undefined) {
      payload.crawl_max_depth = params.crawlMaxDepth;
    }
  } else {
    const urls = parseUrlLines(params.urlsText ?? "");
    if (urls.length === 0) {
      throw new Error("Enter at least one public http(s) URL to scrape.");
    }
    payload.urls = urls;
  }
  return payload;
}

export async function fetchHealth(
  apiBaseUrl: string,
  signal?: AbortSignal
): Promise<HealthResponse> {
  const resp = await fetch(`${normalizeBase(apiBaseUrl)}/v1/health`, {
    method: "GET",
    headers: buildAuthHeaders({ Accept: "application/json" }),
    signal,
  });
  if (!resp.ok) {
    throw new Error(await parseErrorDetail(resp));
  }
  return (await resp.json()) as HealthResponse;
}

export async function runQueryBatch(
  apiBaseUrl: string,
  payload: QueryRequest,
  signal?: AbortSignal
): Promise<FinalAnswer> {
  const resp = await fetch(`${normalizeBase(apiBaseUrl)}/v1/query`, {
    method: "POST",
    headers: buildAuthHeaders({
      "Content-Type": "application/json",
      Accept: "application/json",
    }),
    body: JSON.stringify(payload),
    signal,
  });
  if (!resp.ok) {
    throw new Error(await parseErrorDetail(resp));
  }
  return (await resp.json()) as FinalAnswer;
}

export interface StreamCallbacks {
  onNodeUpdate?: (payload: SseNodeUpdatePayload) => void;
  onFinalAnswer?: (answer: FinalAnswer) => void;
  onErrorEvent?: (err: SseErrorPayload) => void;
}

export async function streamQuery(
  apiBaseUrl: string,
  payload: QueryRequest,
  callbacks: StreamCallbacks,
  signal?: AbortSignal
): Promise<FinalAnswer> {
  const resp = await fetch(`${normalizeBase(apiBaseUrl)}/v1/query/stream`, {
    method: "POST",
    headers: buildAuthHeaders({
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    }),
    body: JSON.stringify(payload),
    signal,
  });

  if (!resp.ok) {
    throw new Error(await parseErrorDetail(resp));
  }
  if (!resp.body) {
    throw new Error("Streaming response body is empty.");
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder("utf-8");
  const parser = new SseStreamParser();
  let finalAnswer: FinalAnswer | null = null;
  let streamError: string | null = null;

  const handleEvents = (events: ReturnType<SseStreamParser["push"]>) => {
    for (const evt of events) {
      if (evt.event === "node_update" && evt.data && typeof evt.data === "object") {
        callbacks.onNodeUpdate?.(evt.data as SseNodeUpdatePayload);
      } else if (evt.event === "final_answer" && evt.data && typeof evt.data === "object") {
        finalAnswer = evt.data as FinalAnswer;
        callbacks.onFinalAnswer?.(finalAnswer);
      } else if (evt.event === "error") {
        const errPayload = (evt.data ?? {}) as SseErrorPayload;
        callbacks.onErrorEvent?.(errPayload);
        streamError = errPayload.message || "Unknown streaming pipeline error.";
      }
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      handleEvents(parser.flush());
      break;
    }
    const text = decoder.decode(value, { stream: true });
    handleEvents(parser.push(text));
    if (streamError) {
      await reader.cancel();
      throw new Error(streamError);
    }
  }

  if (streamError) {
    throw new Error(streamError);
  }
  if (!finalAnswer) {
    throw new Error("Stream ended without a final_answer payload.");
  }
  return finalAnswer;
}

export async function fetchDocuments(
  apiBaseUrl: string,
  signal?: AbortSignal
): Promise<DocumentListResponse> {
  const resp = await fetch(`${normalizeBase(apiBaseUrl)}/v1/documents`, {
    method: "GET",
    headers: buildAuthHeaders({ Accept: "application/json" }),
    signal,
  });
  if (!resp.ok) {
    throw new Error(await parseErrorDetail(resp));
  }
  return (await resp.json()) as DocumentListResponse;
}

export async function deleteDocument(
  apiBaseUrl: string,
  docId: string
): Promise<DeleteDocumentResponse> {
  const encodedId = docId
    .split("/")
    .map((seg) => encodeURIComponent(seg))
    .join("/");
  const resp = await fetch(`${normalizeBase(apiBaseUrl)}/v1/documents/${encodedId}`, {
    method: "DELETE",
    headers: buildAuthHeaders({ Accept: "application/json" }),
  });
  if (!resp.ok) {
    throw new Error(await parseErrorDetail(resp));
  }
  return (await resp.json()) as DeleteDocumentResponse;
}

export async function ingestPayload(
  apiBaseUrl: string,
  payload: IngestRequest
): Promise<IngestResponse> {
  const resp = await fetch(`${normalizeBase(apiBaseUrl)}/v1/ingest`, {
    method: "POST",
    headers: buildAuthHeaders({
      "Content-Type": "application/json",
      Accept: "application/json",
    }),
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    throw new Error(await parseErrorDetail(resp));
  }
  return (await resp.json()) as IngestResponse;
}

export async function uploadFiles(
  apiBaseUrl: string,
  files: File[],
  options?: {
    recreateCollection?: boolean;
    maxTokens?: number;
    overlapTokens?: number;
    parentChild?: boolean;
    ocrMode?: PdfOcrMode;
  }
): Promise<IngestResponse> {
  const form = new FormData();
  for (const file of files) {
    form.append("files", file, file.name);
  }
  form.append("recreate_collection", String(options?.recreateCollection ?? false));
  if (options?.maxTokens !== undefined) {
    form.append("max_tokens", String(options.maxTokens));
  }
  if (options?.overlapTokens !== undefined) {
    form.append("overlap_tokens", String(options.overlapTokens));
  }
  if (options?.parentChild !== undefined) {
    form.append("parent_child", String(options.parentChild));
  }
  if (options?.ocrMode !== undefined) {
    form.append("ocr_mode", options.ocrMode);
  }

  const resp = await fetch(`${normalizeBase(apiBaseUrl)}/v1/ingest/upload`, {
    method: "POST",
    headers: buildAuthHeaders(),
    body: form,
  });
  if (!resp.ok) {
    throw new Error(await parseErrorDetail(resp));
  }
  return (await resp.json()) as IngestResponse;
}

export async function submitFeedback(
  apiBaseUrl: string,
  payload: FeedbackRequest
): Promise<FeedbackResponse> {
  const resp = await fetch(`${normalizeBase(apiBaseUrl)}/v1/feedback`, {
    method: "POST",
    headers: buildAuthHeaders({
      "Content-Type": "application/json",
      Accept: "application/json",
    }),
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    throw new Error(await parseErrorDetail(resp));
  }
  return (await resp.json()) as FeedbackResponse;
}
