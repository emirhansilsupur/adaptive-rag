import React, { useRef, useState } from "react";
import { Check, ChevronDown, Workflow } from "lucide-react";
import { PRESET_OPTIONS } from "../constants/pipeline";
import type { AblationConfig, PresetSelection } from "../types/api";
import { ComposerChip } from "./Composer";
import { Popover } from "./ui/Popover";
import { cn, Field, SELECT_CLASS, Switch } from "./ui/primitives";

interface PresetPickerProps {
  preset: PresetSelection;
  onPresetChange: (preset: PresetSelection) => void;
  customAblation: AblationConfig;
  onCustomAblationChange: (cfg: AblationConfig) => void;
  disabled?: boolean;
}

type StageKey =
  | "enable_query_routing"
  | "enable_multi_query"
  | "enable_rerank"
  | "enable_dynamic_k"
  | "enable_confidence_gating"
  | "enable_web_fallback"
  | "enable_citation_validation";

const STAGE_SWITCHES: { key: StageKey; label: string; description: string }[] = [
  { key: "enable_query_routing", label: "Query routing", description: "Pick a route per question" },
  { key: "enable_multi_query", label: "Multi-query decomposition", description: "Split analytical questions" },
  { key: "enable_rerank", label: "Cross-encoder rerank", description: "Rescore passages locally" },
  { key: "enable_dynamic_k", label: "Dynamic top-k", description: "Keep only passages above the cutoff" },
  { key: "enable_confidence_gating", label: "Confidence gate (CRAG)", description: "Rewrite or fall back when weak" },
  { key: "enable_web_fallback", label: "Web search fallback", description: "Search the web when the corpus misses" },
  { key: "enable_citation_validation", label: "Citation audit", description: "Check every claim against its source" },
];

/** Composer control for choosing which pipeline stages run on the next question. */
export const PresetPicker: React.FC<PresetPickerProps> = ({
  preset,
  onPresetChange,
  customAblation,
  onCustomAblationChange,
  disabled,
}) => {
  const anchorRef = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  const active = PRESET_OPTIONS.find((p) => p.value === preset) ?? PRESET_OPTIONS[0];

  const update = <K extends keyof AblationConfig>(key: K, value: AblationConfig[K]) => {
    onCustomAblationChange({ ...customAblation, [key]: value });
  };

  return (
    <>
      <ComposerChip
        ref={anchorRef}
        icon={Workflow}
        active={open}
        disabled={disabled}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label={`Pipeline preset: ${active.shortLabel}`}
        onClick={() => setOpen((prev) => !prev)}
      >
        <span className="truncate">{active.shortLabel}</span>
        <ChevronDown className="h-3 w-3 shrink-0 text-ink-3" aria-hidden />
      </ComposerChip>

      <Popover
        open={open}
        onClose={() => setOpen(false)}
        anchorRef={anchorRef}
        placement="top-start"
        width={360}
        autoFocus
        ariaLabel="Pipeline preset"
        className="p-1.5"
      >
        <div className="px-2.5 pt-2 pb-1.5">
          <div className="text-ui font-semibold text-ink">Pipeline</div>
          <p className="mt-0.5 text-small text-ink-3">
            Choose which stages run on the next question.
          </p>
        </div>

        <div role="radiogroup" aria-label="Pipeline preset" className="space-y-px">
          {PRESET_OPTIONS.map((opt) => {
            const selected = opt.value === preset;
            return (
              <button
                key={opt.value}
                type="button"
                role="radio"
                aria-checked={selected}
                onClick={() => {
                  onPresetChange(opt.value);
                  if (opt.value !== "custom") setOpen(false);
                }}
                className={cn(
                  "flex w-full cursor-pointer items-start gap-2.5 rounded-lg px-2.5 py-2 text-left transition-colors",
                  selected ? "bg-selected" : "hover:bg-hover"
                )}
              >
                <span
                  aria-hidden
                  className={cn(
                    "mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full border",
                    selected ? "border-ink bg-ink text-paper" : "border-line-strong"
                  )}
                >
                  {selected && <Check className="h-2.5 w-2.5" strokeWidth={3} />}
                </span>
                <span className="min-w-0">
                  <span className="block text-ui font-medium text-ink">{opt.label}</span>
                  <span className="mt-0.5 block text-small text-ink-3">{opt.description}</span>
                </span>
              </button>
            );
          })}
        </div>

        {preset === "custom" && (
          <div className="mx-1 mt-1.5 mb-1 space-y-3 rounded-lg border border-line bg-sunken p-3">
            <div className="space-y-2.5">
              {STAGE_SWITCHES.map(({ key, label, description }) => (
                <Switch
                  key={key}
                  label={label}
                  description={description}
                  checked={customAblation[key]}
                  onChange={(val) => update(key, val)}
                />
              ))}
            </div>

            {!customAblation.enable_dynamic_k && (
              <Field
                label="Fixed top-k"
                htmlFor="preset-fixed-k"
                aside={
                  <span className="text-small font-medium tabular-nums text-ink">
                    {customAblation.fixed_top_k}
                  </span>
                }
              >
                <input
                  id="preset-fixed-k"
                  type="range"
                  min={1}
                  max={15}
                  value={customAblation.fixed_top_k}
                  onChange={(e) => update("fixed_top_k", Number(e.target.value))}
                  className="ar-range w-full"
                />
              </Field>
            )}

            <Field label="Retrieval mode" htmlFor="preset-retrieval-mode">
              <select
                id="preset-retrieval-mode"
                value={customAblation.retrieval_mode}
                onChange={(e) =>
                  update("retrieval_mode", e.target.value as AblationConfig["retrieval_mode"])
                }
                className={SELECT_CLASS}
              >
                <option value="hybrid">Hybrid (dense + BM25)</option>
                <option value="dense_only">Dense only</option>
                <option value="sparse_only">BM25 only</option>
              </select>
            </Field>

            <Field label="Parent sections" htmlFor="preset-parent-child">
              <select
                id="preset-parent-child"
                value={
                  customAblation.enable_parent_child === true
                    ? "enabled"
                    : customAblation.enable_parent_child === false
                      ? "disabled"
                      : "default"
                }
                onChange={(e) =>
                  update(
                    "enable_parent_child",
                    e.target.value === "enabled" ? true : e.target.value === "disabled" ? false : null
                  )
                }
                className={SELECT_CLASS}
              >
                <option value="default">Server default</option>
                <option value="enabled">Expand matches to parent sections</option>
                <option value="disabled">Child chunks only</option>
              </select>
            </Field>

            <Field label="Corpus language" htmlFor="preset-language">
              <select
                id="preset-language"
                value={customAblation.language_filter ?? "all"}
                onChange={(e) =>
                  update(
                    "language_filter",
                    e.target.value === "all" ? null : (e.target.value as "en" | "tr")
                  )
                }
                className={SELECT_CLASS}
              >
                <option value="all">All languages</option>
                <option value="en">English only</option>
                <option value="tr">Turkish only</option>
              </select>
            </Field>
          </div>
        )}
      </Popover>
    </>
  );
};
