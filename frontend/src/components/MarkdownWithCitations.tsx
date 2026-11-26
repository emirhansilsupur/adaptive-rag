import React, { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";
import { Check, Copy, FileText, Globe } from "lucide-react";
import { splitTextWithCitations } from "../lib/citations";
import { excerpt } from "../lib/format";
import type { EvidenceChunk } from "../types/api";
import { CitationMark, isWebCitation } from "./ui/CitationMark";
import { Popover } from "./ui/Popover";

/**
 * Normalize LaTeX \[...\] and \(...\) math delimiters emitted by LLMs into
 * remark-math compatible $$...$$ and $...$ while leaving fenced code untouched.
 */
function normalizeMathDelimiters(markdown: string): string {
  const parts = markdown.split(/(```[\s\S]*?```)/g);
  return parts
    .map((part, idx) => {
      if (idx % 2 === 1) return part; // fenced code block
      return part
        .replace(/\\\[([\s\S]*?)\\\]/g, (_m, expr: string) => `\n$$\n${expr.trim()}\n$$\n`)
        .replace(/\\\((.+?)\\\)/g, (_m, expr: string) => `$${expr.trim()}$`);
    })
    .join("");
}

interface MarkdownWithCitationsProps {
  content: string;
  evidenceChunks: EvidenceChunk[];
  onCitationClick?: (citationId: string) => void;
  density?: "regular" | "compact";
}

function scoreLabel(chunk: EvidenceChunk): string | null {
  if (chunk.rerank_score != null) return `Relevance ${chunk.rerank_score.toFixed(2)}`;
  if (chunk.fusion_score != null) return `RRF ${chunk.fusion_score.toFixed(3)}`;
  return null;
}

/** Hover or focus a mark to read the passage it points to. */
function CitationPill({
  citationId,
  chunk,
  onClick,
}: {
  citationId: string;
  chunk?: EvidenceChunk;
  onClick?: (id: string) => void;
}) {
  const anchorRef = useRef<HTMLButtonElement>(null);
  const openTimer = useRef<number | undefined>(undefined);
  const closeTimer = useRef<number | undefined>(undefined);
  const [open, setOpen] = useState(false);
  const web = isWebCitation(citationId, chunk?.source_type);

  useEffect(
    () => () => {
      window.clearTimeout(openTimer.current);
      window.clearTimeout(closeTimer.current);
    },
    []
  );

  const scheduleOpen = () => {
    if (!chunk) return;
    window.clearTimeout(closeTimer.current);
    openTimer.current = window.setTimeout(() => setOpen(true), 260);
  };
  const scheduleClose = () => {
    window.clearTimeout(openTimer.current);
    closeTimer.current = window.setTimeout(() => setOpen(false), 140);
  };
  const keepOpen = () => window.clearTimeout(closeTimer.current);

  const score = chunk ? scoreLabel(chunk) : null;

  return (
    <>
      <CitationMark
        ref={anchorRef}
        citationId={citationId}
        web={web}
        data-citation-id={citationId}
        title={chunk ? undefined : `Inspect citation ${citationId}`}
        onMouseEnter={scheduleOpen}
        onMouseLeave={scheduleClose}
        onFocus={() => chunk && setOpen(true)}
        onBlur={scheduleClose}
        onClick={(e) => {
          e.stopPropagation();
          window.clearTimeout(openTimer.current);
          setOpen(false);
          onClick?.(citationId);
        }}
        className="mx-px -translate-y-px align-middle"
      />
      {chunk && (
        <Popover
          open={open}
          onClose={() => setOpen(false)}
          anchorRef={anchorRef}
          placement="bottom-start"
          width={360}
          role="tooltip"
          onMouseEnter={keepOpen}
          onMouseLeave={scheduleClose}
          className="p-3.5"
        >
          <div className="flex items-start gap-2.5">
            <CitationMark citationId={citationId} web={web} static size="md" />
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline justify-between gap-3">
                <span className="truncate text-ui font-medium text-ink">
                  {chunk.title || chunk.doc_id || "Untitled source"}
                </span>
                {score && (
                  <span className="shrink-0 text-caption tabular-nums text-ink-3">{score}</span>
                )}
              </div>
              <div className="mt-0.5 flex items-center gap-1.5 text-small text-ink-3">
                {web ? (
                  <Globe className="h-3 w-3 shrink-0" aria-hidden />
                ) : (
                  <FileText className="h-3 w-3 shrink-0" aria-hidden />
                )}
                <span className="truncate">
                  {chunk.section_path || chunk.url || (web ? "Web result" : "Local document")}
                </span>
              </div>
            </div>
          </div>
          <p className="prose-passage mt-2.5 border-l-2 border-mark-strong pl-3 text-[13px] leading-[1.6] whitespace-normal text-ink-2">
            {excerpt(chunk.text, 320)}
          </p>
          <p className="mt-2.5 text-caption text-ink-3">Click the mark to open this passage in the inspector.</p>
        </Popover>
      )}
    </>
  );
}

function injectCitationsInChildren(
  children: React.ReactNode,
  chunkMap: Map<string, EvidenceChunk>,
  onCitationClick?: (id: string) => void
): React.ReactNode {
  return React.Children.map(children, (child, idx) => {
    if (typeof child === "string") {
      const segments = splitTextWithCitations(child);
      if (segments.length === 1 && segments[0].type === "text") {
        return child;
      }
      const consumedLeading = new Set<number>();
      return (
        <React.Fragment key={`seg-${idx}`}>
          {segments.map((seg, sIdx) => {
            if (seg.type === "text") {
              const val = consumedLeading.has(sIdx) ? seg.value.slice(1) : seg.value;
              return val ? <React.Fragment key={sIdx}>{val}</React.Fragment> : null;
            }
            const nextSeg = segments[sIdx + 1];
            let gluedPunct = "";
            if (nextSeg && nextSeg.type === "text" && /^[.,;:!?)]/.test(nextSeg.value)) {
              gluedPunct = nextSeg.value[0];
              consumedLeading.add(sIdx + 1);
            }
            return (
              <span key={sIdx} className="inline whitespace-nowrap">
                {seg.ids.map((cid) => (
                  <CitationPill
                    key={cid}
                    citationId={cid}
                    chunk={chunkMap.get(cid)}
                    onClick={onCitationClick}
                  />
                ))}
                {gluedPunct}
              </span>
            );
          })}
        </React.Fragment>
      );
    }

    if (React.isValidElement<{ children?: React.ReactNode; className?: string }>(child)) {
      // Never rewrite citations inside code, pre, or KaTeX math blocks
      if (
        child.type === "code" ||
        child.type === "pre" ||
        (typeof child.props.className === "string" && child.props.className.includes("katex"))
      ) {
        return child;
      }
      if (child.props.children) {
        return React.cloneElement(child, {
          ...child.props,
          children: injectCitationsInChildren(child.props.children, chunkMap, onCitationClick),
        });
      }
    }
    return child;
  });
}

function CodeBlock({ className, children }: { className?: string; children: React.ReactNode }) {
  const [copied, setCopied] = useState(false);
  const langMatch = /language-(\w+)/.exec(className || "");
  const language = langMatch ? langMatch[1] : "";
  const rawCode = String(children).replace(/\n$/, "");

  const handleCopy = () => {
    void navigator.clipboard?.writeText(rawCode);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  };

  return (
    <div className="my-4 overflow-hidden rounded-lg border border-line bg-sunken font-sans">
      <div className="flex h-8 items-center justify-between border-b border-line px-3 text-small text-ink-3">
        <span>{language || "Code"}</span>
        <button
          type="button"
          onClick={handleCopy}
          className="inline-flex cursor-pointer items-center gap-1 rounded-md px-1.5 py-0.5 transition-colors hover:bg-hover hover:text-ink"
        >
          {copied ? (
            <>
              <Check className="h-3.5 w-3.5 text-ok" aria-hidden />
              <span>Copied</span>
            </>
          ) : (
            <>
              <Copy className="h-3.5 w-3.5" aria-hidden />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>
      <pre className="overflow-x-auto p-3.5 font-mono text-[12.5px] leading-relaxed text-ink">
        <code className={className}>{rawCode}</code>
      </pre>
    </div>
  );
}

export const MarkdownWithCitations: React.FC<MarkdownWithCitationsProps> = ({
  content,
  evidenceChunks,
  onCitationClick,
  density = "regular",
}) => {
  const chunkMap = useMemo(() => {
    const map = new Map<string, EvidenceChunk>();
    for (const ch of evidenceChunks) {
      if (ch.citation_id) {
        map.set(ch.citation_id, ch);
      }
    }
    return map;
  }, [evidenceChunks]);

  const components: Components = useMemo(
    () => ({
      p: ({ children }) => <p>{injectCitationsInChildren(children, chunkMap, onCitationClick)}</p>,
      li: ({ children }) => (
        <li>{injectCitationsInChildren(children, chunkMap, onCitationClick)}</li>
      ),
      td: ({ children }) => (
        <td>{injectCitationsInChildren(children, chunkMap, onCitationClick)}</td>
      ),
      th: ({ children }) => (
        <th>{injectCitationsInChildren(children, chunkMap, onCitationClick)}</th>
      ),
      blockquote: ({ children }) => (
        <blockquote>{injectCitationsInChildren(children, chunkMap, onCitationClick)}</blockquote>
      ),
      a: ({ href, children }) => (
        <a href={href} target="_blank" rel="noopener noreferrer">
          {children}
        </a>
      ),
      pre: ({ children }) => <>{children}</>,
      code: ({ className, children, ...rest }) => {
        const isBlock = Boolean(className) || String(children).includes("\n");
        if (isBlock) {
          return <CodeBlock className={className}>{children}</CodeBlock>;
        }
        return (
          <code
            {...rest}
            className="rounded-[4px] border border-line bg-sunken px-1 py-px font-mono text-[0.84em] text-ink"
          >
            {children}
          </code>
        );
      },
    }),
    [chunkMap, onCitationClick]
  );

  const normalizedContent = useMemo(() => normalizeMathDelimiters(content), [content]);

  return (
    <div className="prose-answer" data-density={density}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={components}
      >
        {normalizedContent}
      </ReactMarkdown>
    </div>
  );
};
