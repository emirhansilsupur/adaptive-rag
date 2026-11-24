import React, { useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  ExternalLink,
  FileText,
  FolderSync,
  Globe,
  Loader2,
  RefreshCw,
  Search,
  Trash2,
  UploadCloud,
  X,
} from "lucide-react";
import {
  buildWebIngestPayload,
  deleteDocument,
  ingestPayload,
  uploadFiles,
} from "../lib/api";
import type {
  DocumentListResponse,
  HealthResponse,
  IndexedDocumentInfo,
  IngestResponse,
} from "../types/api";

interface KnowledgeBaseViewProps {
  apiUrl: string;
  health: HealthResponse | null;
  documentsData: DocumentListResponse | null;
  documentsLoading: boolean;
  documentsError: string | null;
  onRefreshAll: () => Promise<void>;
}

const ACCEPTED_EXTS = [".md", ".markdown", ".txt", ".pdf"];

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function InlineSwitch({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (next: boolean) => void;
}) {
  return (
    <label className="group flex items-center justify-between gap-3 py-0.5 cursor-pointer select-none">
      <span className="text-xs text-[var(--text-secondary)] transition group-hover:text-[var(--text-primary)]">
        {label}
      </span>
      <span className="relative inline-flex h-4 w-7 shrink-0 items-center">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="peer sr-only"
        />
        <span className="h-4 w-7 rounded-full border border-[var(--border-strong)] bg-[var(--bg-elevated)] transition-colors peer-checked:border-transparent peer-checked:bg-[var(--cta-bg)]" />
        <span className="pointer-events-none absolute left-0.5 h-3 w-3 rounded-full bg-[var(--text-muted)] transition-transform peer-checked:translate-x-3 peer-checked:bg-[var(--cta-fg)]" />
      </span>
    </label>
  );
}

export const KnowledgeBaseView: React.FC<KnowledgeBaseViewProps> = ({
  apiUrl,
  health,
  documentsData,
  documentsLoading,
  documentsError,
  onRefreshAll,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [queuedFiles, setQueuedFiles] = useState<File[]>([]);
  const [recreateCollection, setRecreateCollection] = useState(false);
  const [parentChildChunking, setParentChildChunking] = useState<boolean>(
    () => Boolean(health?.parent_child_retrieval?.enabled)
  );
  const [uploading, setUploading] = useState(false);
  const [indexingLocal, setIndexingLocal] = useState(false);
  const [uploadFeedback, setUploadFeedback] = useState<{
    kind: "success" | "error";
    message: string;
  } | null>(null);

  const [webMode, setWebMode] = useState<"scrape" | "crawl">("scrape");
  const [urlsText, setUrlsText] = useState("");
  const [crawlUrl, setCrawlUrl] = useState("");
  const [crawlLimit, setCrawlLimit] = useState(8);
  const [crawlDepth, setCrawlDepth] = useState(2);
  const [onlyMainContent, setOnlyMainContent] = useState(true);
  const [webLoading, setWebLoading] = useState(false);
  const [webFeedback, setWebFeedback] = useState<{
    kind: "success" | "error";
    message: string;
    urls?: string[];
  } | null>(null);

  const [searchQuery, setSearchQuery] = useState("");
  const [originFilter, setOriginFilter] = useState<"all" | "local" | "web">("all");
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const docs = documentsData?.documents ?? [];
  const localCount = docs.filter((d) => d.source_origin === "local").length;
  const webCount = docs.filter((d) => d.source_origin === "web").length;
  const fcStatus = health?.firecrawl?.status || "unknown";
  const fcBaseUrl = health?.firecrawl?.base_url || "http://localhost:3002";

  const filteredDocs = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    return docs.filter((doc) => {
      if (originFilter !== "all" && doc.source_origin !== originFilter) {
        return false;
      }
      if (!q) return true;
      return (
        doc.title.toLowerCase().includes(q) ||
        doc.doc_id.toLowerCase().includes(q) ||
        (doc.source_url || "").toLowerCase().includes(q) ||
        (doc.file_name || "").toLowerCase().includes(q)
      );
    });
  }, [docs, originFilter, searchQuery]);

  const addFiles = (incoming: FileList | File[]) => {
    const arr = Array.from(incoming);
    const valid: File[] = [];
    const rejected: string[] = [];
    for (const file of arr) {
      const lower = file.name.toLowerCase();
      if (ACCEPTED_EXTS.some((ext) => lower.endsWith(ext))) {
        valid.push(file);
      } else {
        rejected.push(file.name);
      }
    }
    if (rejected.length > 0) {
      setUploadFeedback({
        kind: "error",
        message: `Skipped unsupported file(s): ${rejected.join(", ")}. Supported: .md, .txt, .pdf`,
      });
    } else {
      setUploadFeedback(null);
    }
    if (valid.length > 0) {
      setQueuedFiles((prev) => {
        const names = new Set(prev.map((f) => f.name));
        return [...prev, ...valid.filter((f) => !names.has(f.name))];
      });
    }
  };

  const handleUploadSubmit = async () => {
    if (queuedFiles.length === 0) return;
    setUploading(true);
    setUploadFeedback(null);
    try {
      const res: IngestResponse = await uploadFiles(apiUrl, queuedFiles, {
        recreateCollection,
        parentChild: parentChildChunking,
      });
      setQueuedFiles([]);
      const parentSuffix = res.parent_child_enabled
        ? ` · ${res.unique_parents_created ?? 0} parent sections`
        : "";
      setUploadFeedback({
        kind: "success",
        message: `Indexed ${res.documents_processed} uploaded file(s) → ${res.chunks_created} chunks${parentSuffix} (${res.total_collection_points} total points in ${Math.round(res.elapsed_ms)} ms).`,
      });
      await onRefreshAll();
    } catch (err) {
      setUploadFeedback({
        kind: "error",
        message: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setUploading(false);
    }
  };

  const handleReindexLocalCorpus = async () => {
    setIndexingLocal(true);
    setUploadFeedback(null);
    try {
      const res = await ingestPayload(apiUrl, {
        recreate_collection: recreateCollection,
        parent_child: parentChildChunking,
      });
      const parentSuffix = res.parent_child_enabled
        ? ` · ${res.unique_parents_created ?? 0} parent sections`
        : "";
      setUploadFeedback({
        kind: "success",
        message: `Indexed local corpus (${res.source}): ${res.documents_processed} docs → ${res.chunks_created} chunks${parentSuffix} (${res.total_collection_points} total points in ${Math.round(res.elapsed_ms)} ms).`,
      });
      await onRefreshAll();
    } catch (err) {
      setUploadFeedback({
        kind: "error",
        message: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setIndexingLocal(false);
    }
  };

  const handleWebIngest = async (e: React.FormEvent) => {
    e.preventDefault();
    setWebFeedback(null);
    let payload;
    try {
      payload = buildWebIngestPayload({
        mode: webMode,
        urlsText,
        crawlUrl,
        crawlLimit,
        crawlMaxDepth: crawlDepth,
        onlyMainContent,
        recreateCollection,
        parentChild: parentChildChunking,
      });
    } catch (err) {
      setWebFeedback({
        kind: "error",
        message: err instanceof Error ? err.message : String(err),
      });
      return;
    }

    setWebLoading(true);
    try {
      const res = await ingestPayload(apiUrl, payload);
      setWebFeedback({
        kind: "success",
        message: `Indexed ${res.documents_processed} web page(s) → ${res.chunks_created} chunks (${res.total_collection_points} total points in ${Math.round(res.elapsed_ms)} ms).`,
        urls: res.source_urls,
      });
      await onRefreshAll();
    } catch (err) {
      setWebFeedback({
        kind: "error",
        message: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setWebLoading(false);
    }
  };

  const handleDeleteDoc = async (doc: IndexedDocumentInfo) => {
    setDeletingId(doc.doc_id);
    setDeleteError(null);
    try {
      await deleteDocument(apiUrl, doc.doc_id);
      setConfirmDeleteId(null);
      await onRefreshAll();
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : String(err));
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto bg-[var(--bg-canvas)]">
      <div className="mx-auto max-w-6xl space-y-6 px-5 py-6 sm:px-8">
        {/* Header */}
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-base font-semibold tracking-tight text-[var(--text-primary)]">
              Corpus &amp; Ingestion
            </h2>
            <p className="mt-0.5 text-xs text-[var(--text-secondary)]">
              Inspect Qdrant hybrid vectors, index Markdown/PDF/Text files, or crawl
              documentation via Firecrawl Local.
            </p>
          </div>
          <button
            type="button"
            onClick={() => void onRefreshAll()}
            disabled={documentsLoading}
            className="inline-flex h-8 items-center gap-1.5 rounded-md border border-[var(--border-subtle)] bg-[var(--bg-surface)] px-3 text-xs font-medium text-[var(--text-primary)] transition hover:bg-[var(--bg-elevated)] cursor-pointer disabled:opacity-50"
          >
            <RefreshCw
              className={`h-3.5 w-3.5 ${documentsLoading ? "animate-spin" : ""}`}
            />
            <span>Refresh</span>
          </button>
        </div>

        {/* 4-Column Hairline KPI Ledger */}
        <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-[var(--border-subtle)] bg-[var(--border-subtle)] lg:grid-cols-4">
          <div className="bg-[var(--bg-surface)] p-4">
            <div className="font-mono text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
              Documents
            </div>
            <div className="mt-1.5 font-mono text-xl font-semibold text-[var(--text-primary)] tabular-nums">
              {documentsData?.total_documents ?? 0}
            </div>
            <div className="mt-1 font-mono text-[10.5px] text-[var(--text-muted)]">
              {localCount} local · {webCount} web
            </div>
          </div>

          <div className="bg-[var(--bg-surface)] p-4">
            <div className="font-mono text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
              Qdrant Points
            </div>
            <div className="mt-1.5 font-mono text-xl font-semibold text-[var(--text-primary)] tabular-nums">
              {(
                health?.qdrant?.points_count ??
                documentsData?.total_chunks ??
                0
              ).toLocaleString("en-US")}
            </div>
            <div className="mt-1 truncate font-mono text-[10.5px] text-[var(--text-muted)]">
              {health?.qdrant?.collection ||
                documentsData?.collection ||
                "adaptiverag_docs"}
            </div>
          </div>

          <div className="bg-[var(--bg-surface)] p-4">
            <div className="font-mono text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
              Vector Index
            </div>
            <div className="mt-1.5 font-mono text-sm font-semibold text-[var(--text-primary)]">
              Dense + BM25 RRF
            </div>
            <div className="mt-1 truncate font-mono text-[10.5px] text-[var(--text-muted)]">
              {health?.dense_embedding_model ||
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"}
            </div>
          </div>

          <div className="bg-[var(--bg-surface)] p-4">
            <div className="font-mono text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
              Firecrawl Local
            </div>
            <div className="mt-1.5 flex items-center gap-2">
              <span
                className={`h-2 w-2 rounded-full ${
                  fcStatus === "ok"
                    ? "bg-emerald-500"
                    : fcStatus === "unreachable"
                      ? "bg-amber-500"
                      : "bg-zinc-500"
                }`}
              />
              <span className="font-mono text-sm font-semibold uppercase text-[var(--text-primary)]">
                {fcStatus}
              </span>
            </div>
            <div className="mt-1 truncate font-mono text-[10.5px] text-[var(--text-muted)]">
              {fcBaseUrl}
            </div>
          </div>
        </div>

        {/* Two-Column Ingestion Panels */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {/* Panel 1: File Upload & Local Corpus */}
          <div className="flex flex-col rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-surface)] p-5">
            <div>
              <h3 className="text-sm font-semibold text-[var(--text-primary)]">
                Local Files &amp; Corpus Directory
              </h3>
              <p className="mt-0.5 text-xs text-[var(--text-muted)]">
                Structure-aware chunking with idempotent Qdrant point IDs (.md, .txt,
                .pdf)
              </p>
            </div>

            {/* Minimal Dropzone */}
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={() => setDragActive(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragActive(false);
                if (e.dataTransfer.files.length > 0) {
                  addFiles(e.dataTransfer.files);
                }
              }}
              onClick={() => fileInputRef.current?.click()}
              className={`mt-4 flex flex-col items-center justify-center rounded-lg border border-dashed px-4 py-6 text-center transition cursor-pointer ${
                dragActive
                  ? "border-[var(--text-primary)] bg-[var(--bg-elevated)]"
                  : "border-[var(--border-strong)] bg-[var(--bg-canvas)]/50 hover:bg-[var(--bg-elevated)]/50"
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".md,.markdown,.txt,.pdf"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files) {
                    addFiles(e.target.files);
                    e.target.value = "";
                  }
                }}
              />
              <UploadCloud className="h-5 w-5 text-[var(--text-muted)]" />
              <p className="mt-2 text-xs font-medium text-[var(--text-primary)]">
                Drop Markdown, Text, or PDF files, or click to select
              </p>
              <p className="mt-0.5 font-mono text-[10.5px] text-[var(--text-muted)]">
                max 20 files · 15 MB/file
              </p>
            </div>

            {queuedFiles.length > 0 && (
              <div className="mt-3 space-y-1">
                {queuedFiles.map((file) => (
                  <div
                    key={file.name}
                    className="flex items-center justify-between gap-2 rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60 px-2.5 py-1.5 text-xs"
                  >
                    <div className="flex min-w-0 items-center gap-2">
                      <FileText className="h-3.5 w-3.5 shrink-0 text-[var(--text-muted)]" />
                      <span className="truncate font-medium text-[var(--text-primary)]">
                        {file.name}
                      </span>
                      <span className="shrink-0 font-mono text-[10px] text-[var(--text-muted)]">
                        {formatBytes(file.size)}
                      </span>
                    </div>
                    <button
                      type="button"
                      onClick={() =>
                        setQueuedFiles((prev) =>
                          prev.filter((f) => f.name !== file.name)
                        )
                      }
                      className="rounded p-0.5 text-[var(--text-muted)] hover:text-rose-500 cursor-pointer"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                ))}
              </div>
            )}

            <div className="mt-4 space-y-2 border-t border-[var(--border-subtle)] pt-3">
              <InlineSwitch
                label="Hierarchical Parent-Child chunking (256 → 1000 tok)"
                checked={parentChildChunking}
                onChange={setParentChildChunking}
              />
              <InlineSwitch
                label="Recreate Qdrant collection before indexing"
                checked={recreateCollection}
                onChange={setRecreateCollection}
              />
            </div>

            <div className="mt-4 flex flex-wrap gap-2">
              <button
                type="button"
                disabled={queuedFiles.length === 0 || uploading}
                onClick={() => void handleUploadSubmit()}
                className="inline-flex h-8 flex-1 items-center justify-center gap-1.5 rounded-md bg-[var(--cta-bg)] px-3 text-xs font-medium text-[var(--cta-fg)] transition hover:bg-[var(--cta-hover)] disabled:opacity-35 cursor-pointer"
              >
                {uploading ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <UploadCloud className="h-3.5 w-3.5" />
                )}
                <span>
                  {uploading
                    ? "Indexing..."
                    : `Upload & Index${queuedFiles.length > 0 ? ` (${queuedFiles.length})` : ""}`}
                </span>
              </button>

              <button
                type="button"
                disabled={indexingLocal}
                onClick={() => void handleReindexLocalCorpus()}
                className="inline-flex h-8 items-center justify-center gap-1.5 rounded-md border border-[var(--border-strong)] bg-[var(--bg-elevated)] px-3 text-xs font-medium text-[var(--text-primary)] transition hover:bg-[var(--bg-hover)] disabled:opacity-45 cursor-pointer"
              >
                {indexingLocal ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <FolderSync className="h-3.5 w-3.5 text-[var(--text-muted)]" />
                )}
                <span>Index data/raw</span>
              </button>
            </div>

            {uploadFeedback && (
              <div
                className={`mt-3 flex items-start gap-2 rounded-md border px-3 py-2 text-xs ${
                  uploadFeedback.kind === "success"
                    ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
                    : "border-rose-500/30 bg-rose-500/10 text-rose-700 dark:text-rose-300"
                }`}
              >
                {uploadFeedback.kind === "success" ? (
                  <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                ) : (
                  <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                )}
                <span>{uploadFeedback.message}</span>
              </div>
            )}
          </div>

          {/* Panel 2: Firecrawl Local Web Ingestion */}
          <form
            onSubmit={(e) => void handleWebIngest(e)}
            className="flex flex-col rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-surface)] p-5"
          >
            <div className="flex items-start justify-between gap-2">
              <div>
                <h3 className="text-sm font-semibold text-[var(--text-primary)]">
                  Web URL &amp; Docs Crawl
                </h3>
                <p className="mt-0.5 text-xs text-[var(--text-muted)]">
                  Powered by <code className="font-mono">firecrawl-local</code> · SSRF
                  validated
                </p>
              </div>

              <div className="inline-flex shrink-0 whitespace-nowrap rounded-md border border-[var(--border-subtle)] bg-[var(--bg-canvas)] p-0.5 text-[11px]">
                <button
                  type="button"
                  onClick={() => setWebMode("scrape")}
                  className={`rounded px-2.5 py-1 font-medium transition cursor-pointer ${
                    webMode === "scrape"
                      ? "bg-[var(--bg-elevated)] text-[var(--text-primary)]"
                      : "text-[var(--text-muted)] hover:text-[var(--text-primary)]"
                  }`}
                >
                  Scrape URL(s)
                </button>
                <button
                  type="button"
                  onClick={() => setWebMode("crawl")}
                  className={`rounded px-2.5 py-1 font-medium transition cursor-pointer ${
                    webMode === "crawl"
                      ? "bg-[var(--bg-elevated)] text-[var(--text-primary)]"
                      : "text-[var(--text-muted)] hover:text-[var(--text-primary)]"
                  }`}
                >
                  Crawl Docs
                </button>
              </div>
            </div>

            {fcStatus !== "ok" && (
              <div className="mt-3 rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-[11.5px] text-amber-700 dark:text-amber-300">
                Firecrawl Local is <strong>{fcStatus}</strong> at{" "}
                <code className="font-mono">{fcBaseUrl}</code>. Start with{" "}
                <code className="font-mono">FIRECRAWL_CLOUD=off firecrawl-local serve</code>.
              </div>
            )}

            <div className="mt-4 flex-1 space-y-3">
              {webMode === "scrape" ? (
                <div>
                  <label className="block text-[11px] font-medium text-[var(--text-secondary)]">
                    Public HTTP/HTTPS URLs (one per line, max 25)
                  </label>
                  <textarea
                    rows={4}
                    value={urlsText}
                    onChange={(e) => setUrlsText(e.target.value)}
                    placeholder={
                      "https://docs.qdrant.tech/concepts/hybrid-queries/\nhttps://docs.firecrawl.dev/quickstarts/firecrawl-local"
                    }
                    className="mt-1.5 w-full rounded-md border border-[var(--border-subtle)] bg-[var(--bg-canvas)]/70 px-3 py-2 font-mono text-xs text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:border-[var(--text-muted)] focus:outline-none"
                  />
                </div>
              ) : (
                <div className="space-y-3">
                  <div>
                    <label className="block text-[11px] font-medium text-[var(--text-secondary)]">
                      Root Documentation URL
                    </label>
                    <input
                      type="url"
                      value={crawlUrl}
                      onChange={(e) => setCrawlUrl(e.target.value)}
                      placeholder="https://docs.qdrant.tech/concepts/"
                      className="mt-1.5 h-8 w-full rounded-md border border-[var(--border-subtle)] bg-[var(--bg-canvas)]/70 px-3 font-mono text-xs text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:border-[var(--text-muted)] focus:outline-none"
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <div className="mb-1 flex items-center justify-between text-[11px]">
                        <span className="text-[var(--text-secondary)]">Max Pages</span>
                        <span className="font-mono font-medium text-[var(--text-primary)]">
                          {crawlLimit}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={1}
                        max={25}
                        value={crawlLimit}
                        onChange={(e) => setCrawlLimit(Number(e.target.value))}
                        className="ar-range w-full"
                      />
                    </div>
                    <div>
                      <div className="mb-1 flex items-center justify-between text-[11px]">
                        <span className="text-[var(--text-secondary)]">Link Depth</span>
                        <span className="font-mono font-medium text-[var(--text-primary)]">
                          {crawlDepth}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={1}
                        max={4}
                        value={crawlDepth}
                        onChange={(e) => setCrawlDepth(Number(e.target.value))}
                        className="ar-range w-full"
                      />
                    </div>
                  </div>
                </div>
              )}

              <div className="border-t border-[var(--border-subtle)] pt-2.5">
                <InlineSwitch
                  label="Extract main article content only (strip navbars & footers)"
                  checked={onlyMainContent}
                  onChange={setOnlyMainContent}
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={webLoading}
              className="mt-4 inline-flex h-8 w-full items-center justify-center gap-1.5 rounded-md bg-[var(--cta-bg)] px-3 text-xs font-medium text-[var(--cta-fg)] transition hover:bg-[var(--cta-hover)] disabled:opacity-40 cursor-pointer"
            >
              {webLoading ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Globe className="h-3.5 w-3.5" />
              )}
              <span>
                {webLoading
                  ? webMode === "crawl"
                    ? `Crawling up to ${crawlLimit} pages...`
                    : "Scraping & Indexing..."
                  : webMode === "crawl"
                    ? "Crawl Docs Site & Index"
                    : "Scrape URL(s) & Index"}
              </span>
            </button>

            {webFeedback && (
              <div
                className={`mt-3 space-y-1.5 rounded-md border px-3 py-2 text-xs ${
                  webFeedback.kind === "success"
                    ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
                    : "border-rose-500/30 bg-rose-500/10 text-rose-700 dark:text-rose-300"
                }`}
              >
                <div className="flex items-start gap-2">
                  {webFeedback.kind === "success" ? (
                    <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  ) : (
                    <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  )}
                  <span>{webFeedback.message}</span>
                </div>
                {webFeedback.urls && webFeedback.urls.length > 0 && (
                  <ul className="space-y-0.5 pl-5 font-mono text-[10.5px]">
                    {webFeedback.urls.slice(0, 8).map((u) => (
                      <li key={u} className="truncate">
                        <a
                          href={u}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="underline hover:opacity-80"
                        >
                          {u}
                        </a>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </form>
        </div>

        {/* Indexed Documents Table */}
        <div className="overflow-hidden rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-surface)]">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--border-subtle)] px-4 py-3">
            <div className="flex items-center gap-2">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-secondary)]">
                Indexed Sources
              </h3>
              <span className="rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)] px-1.5 py-0.2 font-mono text-[10.5px] text-[var(--text-muted)] tabular-nums">
                {filteredDocs.length}
              </span>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <div className="relative">
                <Search className="pointer-events-none absolute top-1/2 left-2.5 h-3 w-3 -translate-y-1/2 text-[var(--text-muted)]" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Filter sources..."
                  className="h-7.5 w-52 rounded-md border border-[var(--border-subtle)] bg-[var(--bg-canvas)]/70 pr-3 pl-7 text-xs text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:border-[var(--text-muted)] focus:outline-none"
                />
              </div>

              <div className="inline-flex shrink-0 whitespace-nowrap rounded-md border border-[var(--border-subtle)] bg-[var(--bg-canvas)] p-0.5 text-[11px]">
                {(
                  [
                    ["all", "All"],
                    ["local", "Local"],
                    ["web", "Web"],
                  ] as const
                ).map(([val, lbl]) => (
                  <button
                    key={val}
                    type="button"
                    onClick={() => setOriginFilter(val)}
                    className={`rounded px-2.5 py-0.5 font-medium transition cursor-pointer ${
                      originFilter === val
                        ? "bg-[var(--bg-elevated)] text-[var(--text-primary)]"
                        : "text-[var(--text-muted)] hover:text-[var(--text-primary)]"
                    }`}
                  >
                    {lbl}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {(documentsError || deleteError) && (
            <div className="border-b border-rose-500/30 bg-rose-500/10 px-4 py-2 text-xs text-rose-500">
              {deleteError || documentsError}
            </div>
          )}

          {documentsLoading && docs.length === 0 ? (
            <div className="flex items-center justify-center gap-2 py-12 text-xs text-[var(--text-muted)]">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              <span>Loading indexed documents from Qdrant...</span>
            </div>
          ) : filteredDocs.length === 0 ? (
            <div className="py-12 text-center text-xs text-[var(--text-muted)]">
              {docs.length === 0
                ? "No documents indexed yet. Upload files, index data/raw, or scrape a URL above."
                : "No documents match your current search filter."}
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="border-b border-[var(--border-subtle)] bg-[var(--bg-elevated)]/45 font-mono text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
                  <tr>
                    <th className="px-4 py-2.5">Source</th>
                    <th className="px-3 py-2.5">Type</th>
                    <th className="px-3 py-2.5">Lang</th>
                    <th className="px-3 py-2.5 text-right">Chunks</th>
                    <th className="px-3 py-2.5 text-right">Tokens</th>
                    <th className="px-4 py-2.5 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--border-subtle)]">
                  {filteredDocs.map((doc) => {
                    const isWeb = doc.source_origin === "web";
                    const isConfirming = confirmDeleteId === doc.doc_id;
                    const isDeleting = deletingId === doc.doc_id;
                    return (
                      <tr
                        key={doc.doc_id}
                        className="transition hover:bg-[var(--bg-elevated)]/35"
                      >
                        <td className="max-w-md px-4 py-2.5">
                          <div className="truncate font-medium text-[var(--text-primary)]">
                            {doc.title}
                          </div>
                          <div className="mt-0.5 flex items-center gap-2 font-mono text-[10px] text-[var(--text-muted)]">
                            <span className="truncate">{doc.doc_id}</span>
                            {doc.chunk_strategy === "parent_child" && (
                              <span className="shrink-0 rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)] px-1 py-0.2 text-[9.5px] font-medium text-[var(--text-secondary)]">
                                Small→Big
                              </span>
                            )}
                            {doc.source_url && (
                              <a
                                href={doc.source_url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex shrink-0 items-center gap-0.5 text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:underline"
                              >
                                <ExternalLink className="h-2.5 w-2.5" />
                                <span>URL</span>
                              </a>
                            )}
                          </div>
                        </td>
                        <td className="px-3 py-2.5">
                          <span className="inline-flex items-center gap-1 rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)]/60 px-1.5 py-0.5 font-mono text-[10px] text-[var(--text-secondary)]">
                            {isWeb ? (
                              <Globe className="h-2.5 w-2.5 text-amber-500" />
                            ) : (
                              <FileText className="h-2.5 w-2.5 text-[var(--text-muted)]" />
                            )}
                            {isWeb ? "WEB" : doc.doc_type.toUpperCase()}
                          </span>
                        </td>
                        <td className="px-3 py-2.5">
                          <span className="font-mono text-[10.5px] uppercase text-[var(--text-secondary)]">
                            {doc.language}
                          </span>
                        </td>
                        <td className="px-3 py-2.5 text-right font-mono font-medium text-[var(--text-primary)] tabular-nums">
                          {doc.chunk_count}
                        </td>
                        <td className="px-3 py-2.5 text-right font-mono text-[var(--text-secondary)] tabular-nums">
                          {doc.total_tokens.toLocaleString("en-US")}
                        </td>
                        <td className="px-4 py-2.5 text-right">
                          {isConfirming ? (
                            <div className="inline-flex items-center gap-1.5">
                              <button
                                type="button"
                                disabled={isDeleting}
                                onClick={() => void handleDeleteDoc(doc)}
                                className="rounded bg-rose-600 px-2 py-0.5 text-[10.5px] font-medium text-white hover:bg-rose-500 disabled:opacity-50 cursor-pointer"
                              >
                                {isDeleting ? "..." : "Confirm"}
                              </button>
                              <button
                                type="button"
                                disabled={isDeleting}
                                onClick={() => setConfirmDeleteId(null)}
                                className="rounded border border-[var(--border-subtle)] px-2 py-0.5 text-[10.5px] text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer"
                              >
                                Cancel
                              </button>
                            </div>
                          ) : (
                            <button
                              type="button"
                              onClick={() => setConfirmDeleteId(doc.doc_id)}
                              title={`Delete ${doc.doc_id} from Qdrant`}
                              className="inline-flex items-center rounded p-1 text-[var(--text-muted)] transition hover:bg-rose-500/10 hover:text-rose-500 cursor-pointer"
                            >
                              <Trash2 className="h-3.5 w-3.5" />
                            </button>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
