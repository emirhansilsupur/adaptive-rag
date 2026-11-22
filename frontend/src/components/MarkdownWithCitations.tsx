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
      className={`mx-0.5 inline-flex -translate-y-[1px] items-center gap-0.5 rounded-md border px-1.5 py-0.5 font-mono text-[11px] font-semibold leading-none transition-all duration-150 hover:scale-[1.04] focus:outline-none focus:ring-2 focus:ring-blue-500/50 cursor-pointer ${
        isWeb
          ? "border-purple-500/40 bg-purple-500/12 text-purple-600 hover:bg-purple-500/22 dark:text-purple-300"
          : "border-blue-500/40 bg-blue-500/12 text-blue-600 hover:bg-blue-500/22 dark:text-blue-300"
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
      return (
        <React.Fragment key={`seg-${idx}`}>
          {segments.map((seg, sIdx) => {
            if (seg.type === "text") {
              return <React.Fragment key={sIdx}>{seg.value}</React.Fragment>;
            }
            return (
              <span key={sIdx} className="inline-flex items-baseline">
                {seg.ids.map((cid) => (
                  <CitationPill
                    key={cid}
                    citationId={cid}
                    chunk={chunkMap.get(cid)}
                    onClick={onCitationClick}
                  />
                ))}
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
    <div className="my-3 overflow-hidden rounded-xl border border-[var(--border-subtle)] bg-[#0b0f17] text-slate-100">
      <div className="flex items-center justify-between border-b border-white/10 bg-white/[0.03] px-3.5 py-1.5 text-[11px] font-mono text-slate-400">
        <span>{language || "code"}</span>
        <button
          type="button"
          onClick={handleCopy}
          className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-slate-400 transition hover:bg-white/10 hover:text-slate-100 cursor-pointer"
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
      <pre className="overflow-x-auto p-3.5 text-[12.5px] leading-relaxed">
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
            className="rounded-md border border-[var(--border-subtle)] bg-[var(--bg-elevated)] px-1.5 py-0.5 font-mono text-[0.83em] text-blue-600 dark:text-blue-300"
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
