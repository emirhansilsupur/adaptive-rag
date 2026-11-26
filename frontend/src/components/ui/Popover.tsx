import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { cn } from "./primitives";

type Placement = "bottom-start" | "bottom-end" | "top-start" | "top-end";

interface PopoverProps {
  open: boolean;
  onClose: () => void;
  anchorRef: React.RefObject<HTMLElement | null>;
  placement?: Placement;
  /** Preferred width in px; clamped to the viewport. */
  width: number;
  offset?: number;
  /** Move focus into the popover when it opens (dialog-style popovers). */
  autoFocus?: boolean;
  /** Close when the pointer goes down outside the anchor and the popover. */
  dismissOnOutside?: boolean;
  role?: "dialog" | "tooltip" | "menu" | "listbox";
  ariaLabel?: string;
  testId?: string;
  className?: string;
  onMouseEnter?: () => void;
  onMouseLeave?: () => void;
  children: React.ReactNode;
}

interface Position {
  left: number;
  top?: number;
  bottom?: number;
  maxHeight: number;
  width: number;
}

const VIEWPORT_MARGIN = 8;

function computePosition(
  anchor: DOMRect,
  placement: Placement,
  preferredWidth: number,
  offset: number
): Position {
  const vw = window.innerWidth || 1024;
  const vh = window.innerHeight || 768;
  const width = Math.min(preferredWidth, vw - VIEWPORT_MARGIN * 2);

  const spaceBelow = vh - anchor.bottom - offset - VIEWPORT_MARGIN;
  const spaceAbove = anchor.top - offset - VIEWPORT_MARGIN;
  let vertical: "top" | "bottom" = placement.startsWith("top") ? "top" : "bottom";
  if (vertical === "bottom" && spaceBelow < 220 && spaceAbove > spaceBelow) vertical = "top";
  if (vertical === "top" && spaceAbove < 220 && spaceBelow > spaceAbove) vertical = "bottom";

  const alignEnd = placement.endsWith("end");
  let left = alignEnd ? anchor.right - width : anchor.left;
  left = Math.max(VIEWPORT_MARGIN, Math.min(left, vw - width - VIEWPORT_MARGIN));

  if (vertical === "bottom") {
    return {
      left,
      top: anchor.bottom + offset,
      maxHeight: Math.max(160, spaceBelow),
      width,
    };
  }
  return {
    left,
    bottom: vh - anchor.top + offset,
    maxHeight: Math.max(160, spaceAbove),
    width,
  };
}

const FOCUSABLE =
  'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/**
 * Anchored floating layer rendered in a portal. Handles viewport clamping,
 * flipping, outside-click and Escape dismissal, and focus return.
 */
export function Popover({
  open,
  onClose,
  anchorRef,
  placement = "bottom-end",
  width,
  offset = 6,
  autoFocus = false,
  dismissOnOutside = true,
  role = "dialog",
  ariaLabel,
  testId,
  className,
  onMouseEnter,
  onMouseLeave,
  children,
}: PopoverProps) {
  const contentRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState<Position | null>(null);

  const reposition = useCallback(() => {
    const anchor = anchorRef.current;
    if (!anchor) return;
    setPosition(computePosition(anchor.getBoundingClientRect(), placement, width, offset));
  }, [anchorRef, placement, width, offset]);

  useLayoutEffect(() => {
    if (!open) return;
    reposition();
  }, [open, reposition]);

  useEffect(() => {
    if (!open) return;
    window.addEventListener("resize", reposition);
    window.addEventListener("scroll", reposition, true);
    return () => {
      window.removeEventListener("resize", reposition);
      window.removeEventListener("scroll", reposition, true);
    };
  }, [open, reposition]);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: MouseEvent) => {
      if (!dismissOnOutside) return;
      const target = e.target as Node;
      if (contentRef.current?.contains(target)) return;
      if (anchorRef.current?.contains(target)) return;
      onClose();
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
        anchorRef.current?.focus();
      }
    };
    window.addEventListener("mousedown", onPointerDown);
    // Capture phase so page-level Escape handlers see the event as handled.
    window.addEventListener("keydown", onKeyDown, true);
    return () => {
      window.removeEventListener("mousedown", onPointerDown);
      window.removeEventListener("keydown", onKeyDown, true);
    };
  }, [open, onClose, anchorRef, dismissOnOutside]);

  useEffect(() => {
    if (!open || !autoFocus) return;
    const timer = window.setTimeout(() => {
      const first = contentRef.current?.querySelector<HTMLElement>(FOCUSABLE);
      (first ?? contentRef.current)?.focus();
    }, 0);
    return () => window.clearTimeout(timer);
  }, [open, autoFocus]);

  if (!open || typeof document === "undefined") return null;

  const style: React.CSSProperties = position
    ? {
        position: "fixed",
        left: position.left,
        top: position.top,
        bottom: position.bottom,
        width: position.width,
        maxHeight: position.maxHeight,
      }
    : { position: "fixed", left: -9999, top: -9999, width };

  return createPortal(
    <div
      ref={contentRef}
      role={role}
      aria-label={ariaLabel}
      data-testid={testId}
      tabIndex={-1}
      style={style}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      className={cn(
        "anim-pop z-50 overflow-y-auto rounded-xl border border-line-strong bg-raised text-ui text-ink shadow-[var(--shadow-float)] focus:outline-none",
        className
      )}
    >
      {children}
    </div>,
    document.body
  );
}
