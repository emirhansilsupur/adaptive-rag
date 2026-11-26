import { cn } from "./primitives";

/**
 * AdaptiveRAG mark: a citation bracket holding a highlighted passage.
 * The tile stays ink-coloured in both themes so the brand reads the same.
 */
export function LogoMark({ className }: { className?: string }) {
  return (
    <span
      aria-hidden
      className={cn(
        "inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-[7px] bg-[#161C2D] ring-1 ring-white/10 ring-inset",
        className
      )}
    >
      <svg viewBox="0 0 32 32" className="h-full w-full">
        <path
          d="M11.5 8H8v16h3.5M20.5 8H24v16h-3.5"
          fill="none"
          stroke="#FFFFFF"
          strokeWidth="2.4"
          strokeLinecap="square"
        />
        <rect x="11.5" y="13" width="9" height="6" rx="1" fill="var(--brand-mark)" />
      </svg>
    </span>
  );
}
