import React, { forwardRef, useId } from "react";
import {
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  Info,
  Loader2,
  type LucideIcon,
} from "lucide-react";

/* -------------------------------------------------------------------------- */
/* Helpers                                                                    */
/* -------------------------------------------------------------------------- */

export function cn(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

export type Tone = "neutral" | "ok" | "warn" | "bad" | "accent";

/** Shared input styling: hairline border, accent border + soft ring on focus. */
export const INPUT_CLASS =
  "w-full rounded-md border border-line-strong bg-paper text-ui text-ink transition-[border-color,box-shadow] hover:border-ink-4 focus:border-accent focus:shadow-[0_0_0_3px_var(--accent-soft)] focus:outline-none disabled:cursor-not-allowed disabled:opacity-60";

export const SELECT_CLASS = `ar-select ${INPUT_CLASS} h-8 cursor-pointer pl-2.5`;

/* -------------------------------------------------------------------------- */
/* Buttons                                                                    */
/* -------------------------------------------------------------------------- */

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger" | "danger-ghost";
type ButtonSize = "xs" | "sm" | "md";

const BUTTON_VARIANTS: Record<ButtonVariant, string> = {
  primary:
    "bg-cta text-cta-ink hover:bg-cta-hover disabled:bg-line disabled:text-ink-3 disabled:hover:bg-line",
  secondary: "border border-line-strong bg-raised text-ink hover:tint disabled:opacity-50",
  ghost: "text-ink-2 hover:bg-hover hover:text-ink disabled:opacity-40",
  danger: "bg-bad text-on-bad hover:opacity-90 disabled:opacity-50",
  "danger-ghost": "text-bad hover:bg-bad-soft disabled:opacity-40",
};

const BUTTON_SIZES: Record<ButtonSize, string> = {
  xs: "h-6 gap-1 px-2 text-caption",
  sm: "h-7 gap-1.5 px-2.5 text-small",
  md: "h-8 gap-1.5 px-3 text-ui",
};

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  icon?: LucideIcon;
  iconRight?: LucideIcon;
  loading?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  {
    variant = "secondary",
    size = "md",
    icon: Icon,
    iconRight: IconRight,
    loading = false,
    className,
    children,
    type = "button",
    disabled,
    ...rest
  },
  ref
) {
  return (
    <button
      ref={ref}
      type={type}
      disabled={disabled}
      aria-busy={loading || undefined}
      className={cn(
        "inline-flex shrink-0 cursor-pointer items-center justify-center whitespace-nowrap rounded-md font-medium transition-colors select-none disabled:cursor-not-allowed",
        BUTTON_VARIANTS[variant],
        BUTTON_SIZES[size],
        loading && "pointer-events-none",
        className
      )}
      {...rest}
    >
      {loading ? (
        <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
      ) : Icon ? (
        <Icon className="h-3.5 w-3.5 shrink-0" aria-hidden />
      ) : null}
      {children}
      {IconRight && <IconRight className="h-3.5 w-3.5 shrink-0 opacity-70" aria-hidden />}
    </button>
  );
});

export interface IconButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  icon: LucideIcon;
  label: string;
  size?: "sm" | "md" | "lg";
  pressed?: boolean;
  tone?: "default" | "bad";
}

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(function IconButton(
  { icon: Icon, label, size = "md", pressed, tone = "default", className, type = "button", ...rest },
  ref
) {
  const box = size === "sm" ? "h-6 w-6" : size === "lg" ? "h-8 w-8" : "h-7 w-7";
  const glyph = size === "sm" ? "h-3.5 w-3.5" : "h-4 w-4";
  return (
    <button
      ref={ref}
      type={type}
      aria-label={label}
      title={rest.title ?? label}
      aria-pressed={pressed}
      className={cn(
        "inline-flex shrink-0 cursor-pointer items-center justify-center rounded-md transition-colors disabled:cursor-not-allowed disabled:opacity-40",
        box,
        pressed
          ? "bg-selected text-ink"
          : tone === "bad"
            ? "text-ink-3 hover:bg-bad-soft hover:text-bad"
            : "text-ink-3 hover:bg-hover hover:text-ink",
        className
      )}
      {...rest}
    >
      <Icon className={glyph} aria-hidden />
    </button>
  );
});

/* -------------------------------------------------------------------------- */
/* Switch                                                                     */
/* -------------------------------------------------------------------------- */

export function Switch({
  checked,
  onChange,
  label,
  description,
  disabled,
  className,
}: {
  checked: boolean;
  onChange: (next: boolean) => void;
  label: React.ReactNode;
  description?: React.ReactNode;
  disabled?: boolean;
  className?: string;
}) {
  const id = useId();
  return (
    <label
      htmlFor={id}
      className={cn(
        "group flex cursor-pointer items-start justify-between gap-4 select-none",
        disabled && "cursor-not-allowed opacity-55",
        className
      )}
    >
      <span className="min-w-0">
        <span className="block text-ui text-ink">{label}</span>
        {description && (
          <span className="mt-0.5 block text-small text-ink-3">{description}</span>
        )}
      </span>
      <span className="relative mt-px inline-flex h-[18px] w-[30px] shrink-0">
        <input
          id={id}
          type="checkbox"
          role="switch"
          checked={checked}
          disabled={disabled}
          onChange={(e) => onChange(e.target.checked)}
          className="peer absolute inset-0 z-10 m-0 h-full w-full cursor-pointer opacity-0 disabled:cursor-not-allowed"
        />
        <span
          aria-hidden
          className="pointer-events-none h-full w-full rounded-full border border-line-strong bg-sunken transition-colors peer-checked:border-transparent peer-checked:bg-ink peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-accent"
        />
        <span
          aria-hidden
          className="pointer-events-none absolute top-[3px] left-[3px] h-3 w-3 rounded-full bg-ink-3 transition-transform peer-checked:translate-x-3 peer-checked:bg-paper"
        />
      </span>
    </label>
  );
}

/* -------------------------------------------------------------------------- */
/* Segmented control (buttons with aria-pressed)                              */
/* -------------------------------------------------------------------------- */

export interface SegmentOption<T extends string> {
  value: T;
  label: React.ReactNode;
  count?: number;
  title?: string;
  disabled?: boolean;
}

export function Segmented<T extends string>({
  value,
  onChange,
  options,
  ariaLabel,
  size = "sm",
  fullWidth = false,
  className,
}: {
  value: T;
  onChange: (next: T) => void;
  options: SegmentOption<T>[];
  ariaLabel: string;
  size?: "xs" | "sm";
  fullWidth?: boolean;
  className?: string;
}) {
  return (
    <div
      role="group"
      aria-label={ariaLabel}
      className={cn(
        "items-center gap-0.5 rounded-md border border-line bg-sunken p-0.5",
        fullWidth ? "flex w-full" : "inline-flex",
        className
      )}
    >
      {options.map((opt) => {
        const active = opt.value === value;
        return (
          <button
            key={opt.value}
            type="button"
            aria-pressed={active}
            title={opt.title}
            disabled={opt.disabled}
            onClick={() => onChange(opt.value)}
            className={cn(
              "inline-flex cursor-pointer items-center justify-center gap-1.5 whitespace-nowrap rounded-[5px] px-2.5 font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-45",
              size === "xs" ? "h-6 text-caption" : "h-6.5 text-small",
              fullWidth && "min-w-0 flex-1",
              active
                ? "bg-raised text-ink shadow-[0_0_0_1px_var(--line),0_1px_2px_rgba(0,0,0,0.05)]"
                : "text-ink-3 hover:text-ink"
            )}
          >
            {opt.label}
            {opt.count !== undefined && (
              <span aria-hidden="true" className="tabular-nums text-ink-3">
                {opt.count}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Tags, dots, keycaps                                                        */
/* -------------------------------------------------------------------------- */

const TAG_TONES: Record<Tone | "outline", string> = {
  neutral: "border-line bg-sunken text-ink-2",
  ok: "border-transparent bg-ok-soft text-ok",
  warn: "border-transparent bg-warn-soft text-warn",
  bad: "border-transparent bg-bad-soft text-bad",
  accent: "border-transparent bg-accent-soft text-accent",
  outline: "border-line-strong bg-transparent text-ink-2",
};

export function Tag({
  tone = "neutral",
  icon: Icon,
  children,
  className,
  title,
  mono = false,
}: {
  tone?: Tone | "outline";
  icon?: LucideIcon;
  children: React.ReactNode;
  className?: string;
  title?: string;
  mono?: boolean;
}) {
  return (
    <span
      title={title}
      className={cn(
        "inline-flex h-5 max-w-full shrink-0 items-center gap-1 rounded-sm border px-1.5 text-caption font-medium whitespace-nowrap",
        TAG_TONES[tone],
        mono && "font-mono font-normal",
        className
      )}
    >
      {Icon && <Icon className="h-3 w-3 shrink-0" aria-hidden />}
      <span className="truncate">{children}</span>
    </span>
  );
}

const DOT_TONES: Record<Tone, string> = {
  neutral: "bg-ink-4",
  ok: "bg-ok",
  warn: "bg-warn",
  bad: "bg-bad",
  accent: "bg-accent",
};

export function StatusDot({
  tone = "neutral",
  pulse = false,
  className,
}: {
  tone?: Tone;
  pulse?: boolean;
  className?: string;
}) {
  return (
    <span
      aria-hidden
      className={cn(
        "inline-block h-2 w-2 shrink-0 rounded-full",
        DOT_TONES[tone],
        pulse && "live-dot",
        className
      )}
    />
  );
}

export function Kbd({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <kbd
      className={cn(
        "inline-flex h-[18px] min-w-[18px] items-center justify-center rounded-[4px] border border-line-strong bg-paper px-1 font-sans text-caption leading-none text-ink-3",
        className
      )}
    >
      {children}
    </kbd>
  );
}

export function Spinner({ className }: { className?: string }) {
  return <Loader2 className={cn("h-3.5 w-3.5 animate-spin", className)} aria-hidden />;
}

/* -------------------------------------------------------------------------- */
/* Callouts & empty states                                                    */
/* -------------------------------------------------------------------------- */

const CALLOUT_TONES: Record<
  "info" | "ok" | "warn" | "bad",
  { box: string; icon: LucideIcon; iconClass: string }
> = {
  info: { box: "border-line bg-sunken", icon: Info, iconClass: "text-ink-3" },
  ok: { box: "border-ok/25 bg-ok-soft", icon: CheckCircle2, iconClass: "text-ok" },
  warn: { box: "border-warn/30 bg-warn-soft", icon: AlertTriangle, iconClass: "text-warn" },
  bad: { box: "border-bad/30 bg-bad-soft", icon: AlertCircle, iconClass: "text-bad" },
};

export function Callout({
  tone = "info",
  title,
  children,
  className,
  icon,
  action,
  live = false,
}: {
  tone?: "info" | "ok" | "warn" | "bad";
  title?: React.ReactNode;
  children?: React.ReactNode;
  className?: string;
  icon?: LucideIcon;
  action?: React.ReactNode;
  live?: boolean;
}) {
  const config = CALLOUT_TONES[tone];
  const Icon = icon ?? config.icon;
  return (
    <div
      role={live ? (tone === "bad" ? "alert" : "status") : undefined}
      className={cn(
        "flex items-start gap-2.5 rounded-lg border px-3 py-2.5 text-ui",
        config.box,
        className
      )}
    >
      <Icon className={cn("mt-0.5 h-4 w-4 shrink-0", config.iconClass)} aria-hidden />
      <div className="min-w-0 flex-1">
        {title && <div className="font-medium text-ink">{title}</div>}
        {children && (
          <div className={cn("text-ink-2 break-words", title ? "mt-0.5" : undefined)}>
            {children}
          </div>
        )}
      </div>
      {action}
    </div>
  );
}

export function EmptyState({
  icon: Icon,
  title,
  children,
  action,
  className,
}: {
  icon?: LucideIcon;
  title: React.ReactNode;
  children?: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center rounded-lg border border-dashed border-line-strong px-6 py-8 text-center",
        className
      )}
    >
      {Icon && <Icon className="h-5 w-5 text-ink-4" aria-hidden />}
      <p className={cn("text-ui font-medium text-ink", Icon && "mt-2.5")}>{title}</p>
      {children && <p className="mt-1 max-w-sm text-small text-ink-3">{children}</p>}
      {action && <div className="mt-3">{action}</div>}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Layout atoms                                                               */
/* -------------------------------------------------------------------------- */

export function SectionHeading({
  children,
  aside,
  className,
  as: Heading = "h3",
}: {
  children: React.ReactNode;
  aside?: React.ReactNode;
  className?: string;
  as?: "h2" | "h3" | "h4";
}) {
  return (
    <div className={cn("flex items-center justify-between gap-3", className)}>
      <Heading className="text-ui font-semibold text-ink">{children}</Heading>
      {aside}
    </div>
  );
}

export function Field({
  label,
  htmlFor,
  hint,
  aside,
  children,
  className,
}: {
  label: React.ReactNode;
  htmlFor?: string;
  hint?: React.ReactNode;
  aside?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={className}>
      <div className="mb-1.5 flex items-baseline justify-between gap-2">
        <label htmlFor={htmlFor} className="text-small font-medium text-ink">
          {label}
        </label>
        {aside}
      </div>
      {children}
      {hint && <p className="mt-1.5 text-small text-ink-3">{hint}</p>}
    </div>
  );
}
