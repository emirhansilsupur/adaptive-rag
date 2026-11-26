import React, { forwardRef } from "react";
import { Globe } from "lucide-react";
import { cn } from "./primitives";

export function isWebCitation(citationId: string, sourceType?: string | null): boolean {
  return sourceType === "web" || citationId.startsWith("W");
}

interface CitationMarkProps extends Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, "id"> {
  citationId: string;
  web?: boolean;
  size?: "sm" | "md";
  /** Render as a static span instead of a button. */
  static?: boolean;
}

const BASE =
  "inline-flex shrink-0 items-center gap-0.5 rounded-[3px] font-sans font-semibold leading-none tabular-nums transition-colors";

function markTone(web: boolean, interactive: boolean): string {
  if (web) {
    return cn("bg-web text-web-ink", interactive && "hover:bg-web-strong");
  }
  return cn("bg-mark text-mark-ink", interactive && "hover:bg-mark-strong");
}

/**
 * The highlighter mark. The only element in the Studio that uses the
 * highlighter colour: it always means "backed by this passage".
 */
export const CitationMark = forwardRef<HTMLButtonElement, CitationMarkProps>(function CitationMark(
  { citationId, web = false, size = "sm", static: isStatic = false, className, type = "button", ...rest },
  ref
) {
  const sizing = size === "md" ? "h-5 px-1.5 text-small" : "h-[17px] px-1 text-caption";
  const content = (
    <>
      {web && <Globe className="h-2.5 w-2.5 shrink-0" aria-hidden />}
      <span>{citationId}</span>
    </>
  );

  if (isStatic) {
    return <span className={cn(BASE, sizing, markTone(web, false), className)}>{content}</span>;
  }

  return (
    <button
      ref={ref}
      type={type}
      className={cn(BASE, sizing, markTone(web, true), "cursor-pointer", className)}
      {...rest}
    >
      {content}
    </button>
  );
});
