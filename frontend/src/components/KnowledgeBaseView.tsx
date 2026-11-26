import React, { useMemo, useRef, useState } from "react";
import {
  ExternalLink,
  FileText,
  FolderSync,
  Globe,
  RefreshCw,
  Search,
  Trash2,
  UploadCloud,
  X,
} from "lucide-react";
import { buildWebIngestPayload, deleteDocument, ingestPayload, uploadFiles } from "../lib/api";
import { formatDuration, formatInt, languageName } from "../lib/format";
import type {
  DocumentListResponse,
  HealthResponse,
  IndexedDocumentInfo,
  IngestResponse,
  PdfOcrMode,
} from "../types/api";
import {
  Button,
  Callout,
  cn,
  EmptyState,
  Field,
  IconButton,
  INPUT_CLASS,
  Segmented,
  Spinner,
  StatusDot,
  Switch,
  Tag,
} from "./ui/primitives";

interface KnowledgeBaseViewProps {
  apiUrl: string;
  health: HealthResponse | null;
  documentsData: DocumentListResponse | null;
  documentsLoading: boolean;
  documentsError: string | null;
  onRefreshAll: () => Promise<void>;
}

const ACCEPTED_EXTS = [".md", ".markdown", ".txt", ".pdf"];

const OCR_MODES: { value: PdfOcrMode; label: string; hint: string }[] = [
  {
    value: "auto",
    label: "Auto",
    hint: "Reads the text layer first, uses OCRia-nano for scanned pages and a vision model as a last resort.",
  },
  { value: "ocria_nano", label: "OCRia-nano", hint: "Local ONNX OCR for every scanned page. No API cost." },
  { value: "vision_llm", label: "Vision LLM", hint: "Sends scanned pages to the configured vision model." },
  { value: "off", label: "Off", hint: "Uses only the embedded text layer. Scanned pages come out empty." },
];

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

interface IngestFeedback {
  kind: "success" | "error";
  title: string;
  lines: string[];
  urls?: string[];
}

function describeIngest(res: IngestResponse, what: string): IngestFeedback {
  const lines = [
    `${formatInt(res.documents_processed)} ${what} split into ${formatInt(res.chunks_created)} passages.`,
  ];
  if (res.parent_child_enabled) {
    lines.push(`${formatInt(res.unique_parents_created ?? 0)} parent sections created.`);
  }
  if ((res.ocr_pages_processed ?? 0) > 0) {
    const backends =
      res.ocr_backends_used && res.ocr_backends_used.length > 0
        ? ` with ${res.ocr_backends_used.join(" and ")}`
        : "";
    lines.push(`${formatInt(res.ocr_pages_processed ?? 0)} scanned pages read${backends}.`);
  }
  lines.push(
    `The collection now holds ${formatInt(res.total_collection_points)} passages. Took ${formatDuration(res.elapsed_ms)}.`
  );
  return { kind: "success", title: "Indexing finished", lines, urls: res.source_urls };
}

function ResultCallout({ feedback }: { feedback: IngestFeedback }) {
  return (
    <Callout
      live
      tone={feedback.kind === "success" ? "ok" : "bad"}
      title={feedback.title}
      className="mt-4"
    >
      <ul className="space-y-0.5">
        {feedback.lines.map((line) => (
          <li key={line}>{line}</li>
        ))}
      </ul>
      {feedback.urls && feedback.urls.length > 0 && (
        <ul className="mt-1.5 space-y-0.5 text-small">
          {feedback.urls.slice(0, 8).map((u) => (
            <li key={u} className="truncate">
              <a href={u} target="_blank" rel="noopener noreferrer" className="text-accent hover:underline">
                {u}
              </a>
            </li>
          ))}
        </ul>
      )}
    </Callout>
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
  const [parentChildChunking, setParentChildChunking] = useState<boolean>(() =>
    Boolean(health?.parent_child_retrieval?.enabled)
  );
  const [pdfOcrMode, setPdfOcrMode] = useState<PdfOcrMode>(
    () => health?.pdf_ocr?.default_mode ?? "auto"
  );
  const [uploading, setUploading] = useState(false);
  const [indexingLocal, setIndexingLocal] = useState(false);
  const [uploadFeedback, setUploadFeedback] = useState<IngestFeedback | null>(null);

  const [webMode, setWebMode] = useState<"scrape" | "crawl">("scrape");
  const [urlsText, setUrlsText] = useState("");
  const [crawlUrl, setCrawlUrl] = useState("");
  const [crawlLimit, setCrawlLimit] = useState(8);
  const [crawlDepth, setCrawlDepth] = useState(2);
  const [onlyMainContent, setOnlyMainContent] = useState(true);
  const [webLoading, setWebLoading] = useState(false);
  const [webFeedback, setWebFeedback] = useState<IngestFeedback | null>(null);

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
  const crawlerReady = fcStatus === "ok";
  const passages = health?.qdrant?.points_count ?? documentsData?.total_chunks ?? 0;
  const collection = health?.qdrant?.collection || documentsData?.collection || "adaptiverag_docs";
  const embeddingMismatch = health?.qdrant?.embedding_mismatch;
  const parentSizes = health?.parent_child_retrieval;
  const ocrHint = OCR_MODES.find((m) => m.value === pdfOcrMode)?.hint;

  const filteredDocs = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    return docs.filter((doc) => {
      if (originFilter !== "all" && doc.source_origin !== originFilter) return false;
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
      if (ACCEPTED_EXTS.some((ext) => lower.endsWith(ext))) valid.push(file);
      else rejected.push(file.name);
    }
    setUploadFeedback(
      rejected.length > 0
        ? {
            kind: "error",
            title: `Skipped ${rejected.length} unsupported ${rejected.length === 1 ? "file" : "files"}`,
            lines: [`${rejected.join(", ")}. Use Markdown, plain text or PDF.`],
          }
        : null
    );
    if (valid.length > 0) {
      setQueuedFiles((prev) => {
        const names = new Set(prev.map((f) => f.name));
        return [...prev, ...valid.filter((f) => !names.has(f.name))];
      });
    }
  };

  const errorFeedback = (err: unknown, title: string): IngestFeedback => ({
    kind: "error",
    title,
    lines: [err instanceof Error ? err.message : String(err)],
  });

  const handleUploadSubmit = async () => {
    if (queuedFiles.length === 0) return;
    setUploading(true);
    setUploadFeedback(null);
    try {
      const res = await uploadFiles(apiUrl, queuedFiles, {
        recreateCollection,
        parentChild: parentChildChunking,
        ocrMode: pdfOcrMode,
      });
      setQueuedFiles([]);
      setUploadFeedback(describeIngest(res, res.documents_processed === 1 ? "file" : "files"));
      await onRefreshAll();
    } catch (err) {
      setUploadFeedback(errorFeedback(err, "Upload failed"));
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
        ocr_mode: pdfOcrMode,
      });
      setUploadFeedback(
        describeIngest(res, res.documents_processed === 1 ? `document from ${res.source}` : `documents from ${res.source}`)
      );
      await onRefreshAll();
    } catch (err) {
      setUploadFeedback(errorFeedback(err, "Re-indexing failed"));
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
      setWebFeedback(errorFeedback(err, "Check the address"));
      return;
    }

    setWebLoading(true);
    try {
      const res = await ingestPayload(apiUrl, payload);
      setWebFeedback(describeIngest(res, res.documents_processed === 1 ? "web page" : "web pages"));
      await onRefreshAll();
    } catch (err) {
      setWebFeedback(errorFeedback(err, webMode === "crawl" ? "Crawl failed" : "Import failed"));
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
    <div className="flex-1 overflow-y-auto bg-paper">
      <div className="mx-auto max-w-6xl px-4 pt-8 pb-14 sm:px-8">
        {/* Header */}
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-heading font-semibold tracking-[-0.01em] text-ink">Knowledge base</h1>
            <p className="mt-1 max-w-[40rem] text-ui text-ink-2">
              The documents and web pages the pipeline can retrieve from. Everything here is split into
              passages and indexed for hybrid search.
            </p>
          </div>
          <Button
            icon={RefreshCw}
            size="sm"
            loading={documentsLoading}
            onClick={() => void onRefreshAll()}
          >
            Refresh
          </Button>
        </div>

        {embeddingMismatch && (
          <Callout live tone="bad" className="mt-6" title="This collection was indexed with a different model">
            {embeddingMismatch} Questions are refused until then, and so are uploads that don't
            replace everything.
          </Callout>
        )}

        {/* Collection summary */}
        <dl className="mt-6 grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-line bg-line lg:grid-cols-4">
          <div className="bg-paper px-4 py-3.5">
            <dt className="text-caption text-ink-3">Documents</dt>
            <dd className="mt-1 text-heading font-semibold tabular-nums text-ink">
              {formatInt(documentsData?.total_documents ?? 0)}
            </dd>
            <dd className="mt-0.5 text-small text-ink-3">
              {localCount} from files, {webCount} from the web
            </dd>
          </div>
          <div className="bg-paper px-4 py-3.5">
            <dt className="text-caption text-ink-3">Passages</dt>
            <dd className="mt-1 text-heading font-semibold tabular-nums text-ink">
              {formatInt(passages)}
            </dd>
            <dd className="mt-0.5 truncate text-small text-ink-3">
              in <span className="font-mono text-[11.5px]">{collection}</span>
            </dd>
          </div>
          <div className="bg-paper px-4 py-3.5">
            <dt className="text-caption text-ink-3">Search index</dt>
            <dd className="mt-1 text-body font-medium text-ink">Dense + BM25, fused with RRF</dd>
            <dd
              className="mt-0.5 truncate font-mono text-[11.5px] text-ink-3"
              title={health?.dense_embedding_model}
            >
              {health?.dense_embedding_model?.split("/").pop() ||
                "paraphrase-multilingual-mpnet-base-v2"}
            </dd>
          </div>
          <div className="bg-paper px-4 py-3.5">
            <dt className="text-caption text-ink-3">Web crawler</dt>
            <dd className="mt-1 flex items-center gap-2 text-body font-medium text-ink">
              <StatusDot tone={crawlerReady ? "ok" : fcStatus === "unreachable" ? "warn" : "neutral"} />
              {crawlerReady ? "Ready" : fcStatus === "unreachable" ? "Unreachable" : "Not configured"}
            </dd>
            <dd className="mt-0.5 truncate font-mono text-[11.5px] text-ink-3">{fcBaseUrl}</dd>
          </div>
        </dl>

        {/* Add documents */}
        <h2 className="mt-10 text-body font-semibold text-ink">Add documents</h2>
        <div className="mt-3 grid grid-cols-1 gap-4 lg:grid-cols-2">
          {/* Files */}
          <section className="flex flex-col rounded-xl border border-line bg-paper p-5" aria-labelledby="kb-files">
            <h3 id="kb-files" className="text-ui font-semibold text-ink">
              Upload files
            </h3>
            <p className="mt-0.5 text-small text-ink-3">
              Markdown, plain text and PDF. Headings and tables are kept when splitting.
            </p>

            <div
              role="button"
              tabIndex={0}
              aria-label="Choose files to upload"
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  fileInputRef.current?.click();
                }
              }}
              onDragOver={(e) => {
                e.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={() => setDragActive(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragActive(false);
                if (e.dataTransfer.files.length > 0) addFiles(e.dataTransfer.files);
              }}
              onClick={() => fileInputRef.current?.click()}
              className={cn(
                "mt-4 flex cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed px-4 py-7 text-center transition-colors",
                dragActive ? "border-accent bg-accent-soft" : "border-line-strong bg-sunken hover:tint"
              )}
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
              <UploadCloud className="h-5 w-5 text-ink-3" aria-hidden />
              <p className="mt-2 text-ui font-medium text-ink">Drop files here or click to choose</p>
              <p className="mt-0.5 text-small text-ink-3">Up to 20 files, 15 MB each</p>
            </div>

            {queuedFiles.length > 0 && (
              <ul className="mt-3 divide-y divide-line rounded-lg border border-line">
                {queuedFiles.map((file) => (
                  <li key={file.name} className="flex items-center justify-between gap-2 px-3 py-2">
                    <span className="flex min-w-0 items-center gap-2">
                      <FileText className="h-4 w-4 shrink-0 text-ink-3" aria-hidden />
                      <span className="truncate text-ui text-ink">{file.name}</span>
                      <span className="shrink-0 text-small tabular-nums text-ink-3">
                        {formatBytes(file.size)}
                      </span>
                    </span>
                    <IconButton
                      icon={X}
                      size="sm"
                      tone="bad"
                      label={`Remove ${file.name}`}
                      onClick={() => setQueuedFiles((prev) => prev.filter((f) => f.name !== file.name))}
                    />
                  </li>
                ))}
              </ul>
            )}

            <div className="mt-5 space-y-4 border-t border-line pt-4">
              <div>
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="flex items-center gap-2 text-ui text-ink">
                    PDF text recognition
                    {health?.pdf_ocr?.ocria_nano_installed && <Tag tone="ok">OCRia-nano ready</Tag>}
                  </span>
                  <Segmented
                    ariaLabel="PDF OCR mode"
                    value={pdfOcrMode}
                    onChange={setPdfOcrMode}
                    options={OCR_MODES.map((m) => ({ value: m.value, label: m.label }))}
                  />
                </div>
                {ocrHint && <p className="mt-1.5 text-small text-ink-3">{ocrHint}</p>}
              </div>
              <Switch
                label="Hierarchical parent-child chunking"
                description={`Index small ${parentSizes?.child_max_tokens ?? 256}-token passages and answer from their ${formatInt(parentSizes?.parent_max_tokens ?? 1000)}-token parent sections.`}
                checked={parentChildChunking}
                onChange={setParentChildChunking}
              />
              <Switch
                label="Replace everything in the collection"
                description="Keeps only these files. Current passages stay searchable until indexing succeeds, then they're deleted. This can't be undone."
                checked={recreateCollection}
                onChange={setRecreateCollection}
              />
            </div>

            <div className="mt-5 flex flex-wrap gap-2">
              <Button
                variant="primary"
                icon={UploadCloud}
                loading={uploading}
                disabled={queuedFiles.length === 0}
                onClick={() => void handleUploadSubmit()}
                className="min-w-[10rem] flex-1"
              >
                {uploading
                  ? "Indexing…"
                  : queuedFiles.length > 0
                    ? `Upload and index ${queuedFiles.length} ${queuedFiles.length === 1 ? "file" : "files"}`
                    : "Upload and index"}
              </Button>
              <Button
                icon={FolderSync}
                loading={indexingLocal}
                onClick={() => void handleReindexLocalCorpus()}
                title="Index every supported file in the server's data/raw folder"
              >
                Re-index data/raw
              </Button>
            </div>

            {uploadFeedback && <ResultCallout feedback={uploadFeedback} />}
          </section>

          {/* Web */}
          <form
            onSubmit={(e) => void handleWebIngest(e)}
            className="flex flex-col rounded-xl border border-line bg-paper p-5"
            aria-labelledby="kb-web"
          >
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <h3 id="kb-web" className="text-ui font-semibold text-ink">
                  Import from the web
                </h3>
                <p className="mt-0.5 text-small text-ink-3">
                  Fetched with Firecrawl Local. Private and internal addresses are blocked.
                </p>
              </div>
              <Segmented
                ariaLabel="Web import mode"
                value={webMode}
                onChange={setWebMode}
                options={[
                  { value: "scrape", label: "Pages" },
                  { value: "crawl", label: "Whole site" },
                ]}
              />
            </div>

            {!crawlerReady && (
              <Callout tone="warn" className="mt-4" title="The web crawler isn't reachable">
                Start it with{" "}
                <code className="rounded-[4px] bg-paper px-1 font-mono text-[11.5px] text-ink">
                  FIRECRAWL_CLOUD=off firecrawl-local serve
                </code>{" "}
                and expect it at <span className="font-mono text-[11.5px]">{fcBaseUrl}</span>.
              </Callout>
            )}

            <div className="mt-4 flex-1 space-y-4">
              {webMode === "scrape" ? (
                <Field
                  label="Page addresses"
                  htmlFor="kb-urls"
                  hint="One URL per line, up to 25. Each page becomes one document."
                >
                  <textarea
                    id="kb-urls"
                    rows={5}
                    value={urlsText}
                    onChange={(e) => setUrlsText(e.target.value)}
                    placeholder={"https://docs.qdrant.tech/concepts/hybrid-queries/\nhttps://docs.firecrawl.dev/quickstarts/firecrawl-local"}
                    className={cn(INPUT_CLASS, "resize-y px-3 py-2 font-mono text-small leading-relaxed")}
                  />
                </Field>
              ) : (
                <>
                  <Field
                    label="Start page"
                    htmlFor="kb-crawl-url"
                    hint="Links are followed only within this site."
                  >
                    <input
                      id="kb-crawl-url"
                      type="url"
                      value={crawlUrl}
                      onChange={(e) => setCrawlUrl(e.target.value)}
                      placeholder="https://docs.qdrant.tech/concepts/"
                      className={cn(INPUT_CLASS, "h-8 px-3 font-mono text-small")}
                    />
                  </Field>
                  <div className="grid grid-cols-2 gap-4">
                    <Field
                      label="Pages"
                      htmlFor="kb-crawl-limit"
                      aside={<span className="text-small font-medium tabular-nums text-ink">{crawlLimit}</span>}
                    >
                      <input
                        id="kb-crawl-limit"
                        type="range"
                        min={1}
                        max={25}
                        value={crawlLimit}
                        onChange={(e) => setCrawlLimit(Number(e.target.value))}
                        className="ar-range w-full"
                      />
                    </Field>
                    <Field
                      label="Link depth"
                      htmlFor="kb-crawl-depth"
                      aside={<span className="text-small font-medium tabular-nums text-ink">{crawlDepth}</span>}
                    >
                      <input
                        id="kb-crawl-depth"
                        type="range"
                        min={1}
                        max={4}
                        value={crawlDepth}
                        onChange={(e) => setCrawlDepth(Number(e.target.value))}
                        className="ar-range w-full"
                      />
                    </Field>
                  </div>
                </>
              )}

              <div className="border-t border-line pt-4">
                <Switch
                  label="Main content only"
                  description="Drops navigation bars, footers and cookie banners."
                  checked={onlyMainContent}
                  onChange={setOnlyMainContent}
                />
              </div>
            </div>

            <Button
              type="submit"
              variant="primary"
              icon={Globe}
              loading={webLoading}
              className="mt-5 w-full"
            >
              {webLoading
                ? webMode === "crawl"
                  ? `Crawling up to ${crawlLimit} pages…`
                  : "Importing…"
                : webMode === "crawl"
                  ? "Crawl and index the site"
                  : "Import and index pages"}
            </Button>

            {webFeedback && <ResultCallout feedback={webFeedback} />}
          </form>
        </div>

        {/* Documents table */}
        <section className="mt-10" aria-labelledby="kb-docs">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 id="kb-docs" className="flex items-baseline gap-2 text-body font-semibold text-ink">
              Indexed documents
              <span className="text-ui font-normal tabular-nums text-ink-3">{filteredDocs.length}</span>
            </h2>
            <div className="flex flex-wrap items-center gap-2">
              <div className="relative">
                <Search
                  className="pointer-events-none absolute top-1/2 left-2.5 h-3.5 w-3.5 -translate-y-1/2 text-ink-3"
                  aria-hidden
                />
                <input
                  type="search"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Filter by title, ID or URL"
                  aria-label="Filter documents"
                  className={cn(INPUT_CLASS, "h-8 w-60 pr-3 pl-8")}
                />
              </div>
              <Segmented
                ariaLabel="Filter by origin"
                value={originFilter}
                onChange={setOriginFilter}
                options={[
                  { value: "all", label: "All" },
                  { value: "local", label: "Files" },
                  { value: "web", label: "Web" },
                ]}
              />
            </div>
          </div>

          {(documentsError || deleteError) && (
            <Callout live tone="bad" className="mt-3" title="Couldn't load or change documents">
              {deleteError || documentsError}
            </Callout>
          )}

          <div className="mt-3 overflow-hidden rounded-xl border border-line">
            {documentsLoading && docs.length === 0 ? (
              <div className="flex items-center justify-center gap-2 py-14 text-ui text-ink-3">
                <Spinner />
                Loading documents…
              </div>
            ) : filteredDocs.length === 0 ? (
              <EmptyState
                className="m-4 border-0"
                icon={FileText}
                title={docs.length === 0 ? "Nothing indexed yet" : "No documents match this filter"}
              >
                {docs.length === 0
                  ? "Upload files, re-index data/raw, or import a web page above."
                  : "Try a different word or clear the filter."}
              </EmptyState>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full min-w-[640px] text-left text-ui">
                  <thead className="border-b border-line bg-sunken text-small text-ink-3">
                    <tr>
                      <th scope="col" className="px-4 py-2.5 font-medium">Document</th>
                      <th scope="col" className="px-3 py-2.5 font-medium">Type</th>
                      <th scope="col" className="px-3 py-2.5 font-medium">Language</th>
                      <th scope="col" className="px-3 py-2.5 text-right font-medium">Passages</th>
                      <th scope="col" className="px-3 py-2.5 text-right font-medium">Tokens</th>
                      <th scope="col" className="w-px px-4 py-2.5">
                        <span className="sr-only">Actions</span>
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-line bg-paper">
                    {filteredDocs.map((doc) => {
                      const isWeb = doc.source_origin === "web";
                      const isConfirming = confirmDeleteId === doc.doc_id;
                      const isDeleting = deletingId === doc.doc_id;
                      return (
                        <tr key={doc.doc_id} className="transition-colors hover:bg-hover">
                          <td className="max-w-md px-4 py-3">
                            <div className="truncate font-medium text-ink">{doc.title}</div>
                            <div className="mt-0.5 flex items-center gap-2 text-small text-ink-3">
                              <span className="truncate font-mono text-[11.5px]">{doc.doc_id}</span>
                              {doc.chunk_strategy === "parent_child" && <Tag tone="accent">Small→Big</Tag>}
                              {doc.source_url && (
                                <a
                                  href={doc.source_url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="inline-flex shrink-0 items-center gap-1 text-accent hover:underline"
                                >
                                  <ExternalLink className="h-3 w-3" aria-hidden />
                                  Source
                                </a>
                              )}
                            </div>
                          </td>
                          <td className="px-3 py-3">
                            <span className="inline-flex items-center gap-1.5 text-ink-2">
                              {isWeb ? (
                                <Globe className="h-3.5 w-3.5 text-ink-3" aria-hidden />
                              ) : (
                                <FileText className="h-3.5 w-3.5 text-ink-3" aria-hidden />
                              )}
                              {isWeb ? "Web page" : doc.doc_type.toUpperCase()}
                            </span>
                          </td>
                          <td className="px-3 py-3 text-ink-2">{languageName(doc.language)}</td>
                          <td className="px-3 py-3 text-right font-medium tabular-nums text-ink">
                            {formatInt(doc.chunk_count)}
                          </td>
                          <td className="px-3 py-3 text-right tabular-nums text-ink-2">
                            {formatInt(doc.total_tokens)}
                          </td>
                          <td className="px-4 py-3 text-right whitespace-nowrap">
                            {isConfirming ? (
                              <span className="inline-flex items-center gap-1.5">
                                <Button
                                  variant="danger"
                                  size="xs"
                                  loading={isDeleting}
                                  onClick={() => void handleDeleteDoc(doc)}
                                >
                                  Delete
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="xs"
                                  disabled={isDeleting}
                                  onClick={() => setConfirmDeleteId(null)}
                                >
                                  Cancel
                                </Button>
                              </span>
                            ) : (
                              <IconButton
                                icon={Trash2}
                                tone="bad"
                                label={`Delete ${doc.title}`}
                                onClick={() => setConfirmDeleteId(doc.doc_id)}
                              />
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
        </section>
      </div>
    </div>
  );
};
