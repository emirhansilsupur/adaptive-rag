import React, { useMemo, useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";
import { Check, Copy, Globe } from "lucide-react";
import { splitTextWithCitations } from "../lib/citations";
import type { EvidenceChunk } from "../types/api";

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
}

function CitationPill({
  citationId,
  chunk,
  onClick,
}: {
  citationId: string;
  chunk?: EvidenceChunk;
  onClick?: (id: string) => void;
}) {
  const isWeb = citationId.startsWith("W") || chunk?.source_type === "web";
  const scoreLabel =
    chunk?.rerank_score != null
      ? `rerank ${chunk.rerank_score.toFixed(2)}`
      : chunk?.fusion_score != null
        ? `fusion ${chunk.fusion_score.toFixed(3)}`
        : null;
  const tooltip = chunk
    ? `[${citationId}] ${chunk.title || chunk.doc_id}${scoreLabel ? ` · ${scoreLabel}` : ""}`
    : `Inspect citation [${citationId}]`;

  return (
    <button
      type="button"
      onClick={(e) => {
        e.stopPropagation();
        onClick?.(citationId);
      }}
      title={tooltip}
      data-citation-id={citationId}
      className={`mx-0.5 inline-flex -translate-y-[1px] items-center gap-0.5 rounded-[4px] border px-1.25 py-0.5 font-mono text-[10px] font-semibold leading-none transition-colors focus:outline-none focus:ring-1 focus:ring-sky-500/50 cursor-pointer ${
        isWeb
          ? "border-amber-500/35 bg-amber-500/10 text-amber-700 hover:bg-amber-500/20 dark:text-amber-300"
          : "border-sky-500/35 bg-sky-500/10 text-sky-700 hover:bg-sky-500/20 dark:text-sky-300"
      }`}
    >
      {isWeb && <Globe className="h-2.5 w-2.5 shrink-0" />}
      <span>{citationId}</span>
    </button>
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
            if (
              nextSeg &&
              nextSeg.type === "text" &&
              /^[.,;:!?)]/.test(nextSeg.value)
            ) {
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
        (typeof child.props.className === "string" &&
          child.props.className.includes("katex"))
      ) {
        return child;
      }
      if (child.props.children) {
        return React.cloneElement(child, {
          ...child.props,
          children: injectCitationsInChildren(
            child.props.children,
            chunkMap,
            onCitationClick
          ),
        });
      }
    }
    return child;
  });
}

function CodeBlock({
  className,
  children,
}: {
  className?: string;
  children: React.ReactNode;
}) {
  const [copied, setCopied] = useState(false);
  const langMatch = /language-(\w+)/.exec(className || "");
  const language = langMatch ? langMatch[1] : "";
  const rawCode = String(children).replace(/\n$/, "");

  const handleCopy = () => {
    void navigator.clipboard?.writeText(rawCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  return (
    <div className="my-3 overflow-hidden rounded-md border border-[var(--border-subtle)] bg-[#0c0c0f] text-zinc-100">
      <div className="flex items-center justify-between border-b border-white/[0.07] bg-white/[0.02] px-3 py-1.5 font-mono text-[10.5px] text-zinc-400">
        <span>{language || "code"}</span>
        <button
          type="button"
          onClick={handleCopy}
          className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-zinc-400 transition hover:bg-white/[0.08] hover:text-zinc-100 cursor-pointer"
        >
          {copied ? (
            <>
              <Check className="h-3 w-3 text-emerald-400" />
              <span className="text-emerald-400">Copied</span>
            </>
          ) : (
            <>
              <Copy className="h-3 w-3" />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>
      <pre className="overflow-x-auto p-3 text-[12px] leading-relaxed">
        <code className={className}>{rawCode}</code>
      </pre>
    </div>
  );
}

export const MarkdownWithCitations: React.FC<MarkdownWithCitationsProps> = ({
  content,
  evidenceChunks,
  onCitationClick,
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
      p: ({ children }) => (
        <p>{injectCitationsInChildren(children, chunkMap, onCitationClick)}</p>
      ),
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
        <blockquote>
          {injectCitationsInChildren(children, chunkMap, onCitationClick)}
        </blockquote>
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
            className="rounded-[4px] border border-[var(--border-subtle)] bg-[var(--bg-elevated)] px-1.25 py-0.25 font-mono text-[0.84em] text-[var(--text-primary)]"
          >
            {children}
          </code>
        );
      },
    }),
    [chunkMap, onCitationClick]
  );

  const normalizedContent = useMemo(
    () => normalizeMathDelimiters(content),
    [content]
  );

  return (
    <div className="ar-markdown">
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
