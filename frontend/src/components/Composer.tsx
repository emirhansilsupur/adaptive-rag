import React, { useEffect, useRef } from "react";
import { ArrowUp, Square } from "lucide-react";
import { cn, Kbd } from "./ui/primitives";

interface ComposerProps {
  value: string;
  onChange: (next: string) => void;
  onSubmit: () => void;
  onStop: () => void;
  isRunning: boolean;
  placeholder: string;
  submitLabel: string;
  stopLabel: string;
  /** Controls rendered inside the composer, left of the send button. */
  toolbar?: React.ReactNode;
  /** Secondary line under the composer (right side shows keyboard hints). */
  footer?: React.ReactNode;
  maxHeight?: number;
  ariaLabel?: string;
}

/** Shared question box for the Workbench and the Arena. */
export const Composer: React.FC<ComposerProps> = ({
  value,
  onChange,
  onSubmit,
  onStop,
  isRunning,
  placeholder,
  submitLabel,
  stopLabel,
  toolbar,
  footer,
  maxHeight = 176,
  ariaLabel = "Question",
}) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, maxHeight)}px`;
  }, [value, maxHeight]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      onSubmit();
    }
  };

  const canSend = value.trim().length > 0 && !isRunning;

  return (
    <div>
      <div
        className={cn(
          "rounded-xl border border-line-strong bg-paper shadow-[0_1px_2px_rgba(22,28,45,0.04)] transition-[border-color,box-shadow]",
          "focus-within:border-accent focus-within:shadow-[0_0_0_3px_var(--accent-soft)]"
        )}
      >
        <textarea
          ref={textareaRef}
          rows={1}
          value={value}
          aria-label={ariaLabel}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isRunning}
          placeholder={placeholder}
          style={{ maxHeight }}
          className="block w-full resize-none overflow-y-auto bg-transparent px-3.5 pt-3 pb-1 text-body text-ink focus:outline-none disabled:opacity-60"
        />
        <div className="flex items-end justify-between gap-2 px-2 pb-2">
          <div className="flex min-w-0 flex-wrap items-center gap-1">{toolbar}</div>
          {isRunning ? (
            <button
              type="button"
              onClick={onStop}
              aria-label={stopLabel}
              title={stopLabel}
              className="flex h-8 w-8 shrink-0 cursor-pointer items-center justify-center rounded-lg bg-bad text-on-bad transition-opacity hover:opacity-90"
            >
              <Square className="h-3 w-3 fill-current" aria-hidden />
            </button>
          ) : (
            <button
              type="button"
              onClick={onSubmit}
              disabled={!canSend}
              aria-label={submitLabel}
              title={submitLabel}
              className="flex h-8 w-8 shrink-0 cursor-pointer items-center justify-center rounded-lg bg-cta text-cta-ink transition-colors hover:bg-cta-hover disabled:cursor-not-allowed disabled:bg-line disabled:text-ink-4"
            >
              <ArrowUp className="h-4 w-4" aria-hidden />
            </button>
          )}
        </div>
      </div>
      <div className="mt-1.5 flex min-h-5 flex-wrap items-center justify-between gap-x-3 gap-y-1 px-1 text-caption text-ink-3">
        <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1">{footer}</div>
        <div className="hidden items-center gap-1.5 sm:flex">
          <Kbd>↵</Kbd>
          <span>to send</span>
          <Kbd className="ml-1.5">⇧↵</Kbd>
          <span>for a new line</span>
        </div>
      </div>
    </div>
  );
};

/** Small pill button used inside the composer toolbar. */
export const ComposerChip = React.forwardRef<
  HTMLButtonElement,
  React.ButtonHTMLAttributes<HTMLButtonElement> & {
    active?: boolean;
    icon?: React.ComponentType<{ className?: string }>;
  }
>(function ComposerChip({ active, icon: Icon, className, children, type = "button", ...rest }, ref) {
  return (
    <button
      ref={ref}
      type={type}
      className={cn(
        "inline-flex h-7 max-w-[220px] shrink-0 cursor-pointer items-center gap-1.5 rounded-md px-2 text-small transition-colors disabled:cursor-not-allowed disabled:opacity-50",
        active
          ? "bg-selected text-ink"
          : "text-ink-2 hover:bg-hover hover:text-ink",
        className
      )}
      {...rest}
    >
      {Icon && <Icon className="h-3.5 w-3.5 shrink-0 text-ink-3" />}
      {children}
    </button>
  );
});
