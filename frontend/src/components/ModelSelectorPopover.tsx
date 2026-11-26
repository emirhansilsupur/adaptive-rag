import React, { useRef, useState } from "react";
import { ChevronDown, Eye, EyeOff, Lock, RefreshCw, RotateCcw, Server } from "lucide-react";
import type {
  HealthResponse,
  LLMOverride,
  LLMProvider,
  ModelsCatalogResponse,
  ProviderApiKeys,
  ProviderCatalogItem,
} from "../types/api";
import { Popover } from "./ui/Popover";
import {
  Button,
  Callout,
  cn,
  Field,
  IconButton,
  INPUT_CLASS,
  StatusDot,
  Switch,
  Tag,
  type Tone,
} from "./ui/primitives";

interface ModelSelectorPopoverProps {
  catalog: ModelsCatalogResponse | null;
  catalogLoading: boolean;
  health: HealthResponse | null;
  override: LLMOverride | null;
  providerApiKeys?: ProviderApiKeys;
  onOverrideChange: (next: LLMOverride | null) => void;
  onProviderApiKeyChange?: (provider: LLMProvider, apiKey: string) => void;
  onRefreshCatalog: () => void;
  /* Run options (optional: shown when handlers are provided) */
  useStreaming?: boolean;
  onUseStreamingChange?: (next: boolean) => void;
  enableBudgetCap?: boolean;
  onEnableBudgetCapChange?: (next: boolean) => void;
  maxCostUsd?: number;
  onMaxCostUsdChange?: (next: number) => void;
}

function shortModelDisplay(modelId: string): string {
  const trimmed = modelId.trim();
  if (!trimmed) return "auto";
  const parts = trimmed.split("/");
  return parts[parts.length - 1] || trimmed;
}

function providerShortLabel(id: LLMProvider): string {
  switch (id) {
    case "openrouter":
      return "OpenRouter";
    case "ollama":
      return "Local";
    case "openai":
      return "OpenAI";
    case "gemini":
      return "Gemini";
    case "anthropic":
      return "Anthropic";
  }
}

function SuggestionChips({
  models,
  active,
  onPick,
}: {
  models: string[];
  active: string;
  onPick: (modelId: string) => void;
}) {
  if (models.length === 0) return null;
  return (
    <div className="mt-1.5 flex flex-wrap gap-1">
      {models.map((modelId) => {
        const selected = modelId === active;
        return (
          <button
            key={modelId}
            type="button"
            title={modelId}
            aria-pressed={selected}
            onClick={() => onPick(modelId)}
            className={cn(
              "h-6 cursor-pointer rounded-md border px-1.5 font-mono text-[11px] transition-colors",
              selected
                ? "border-accent bg-accent-soft text-ink"
                : "border-line bg-paper text-ink-2 hover:border-line-strong hover:text-ink"
            )}
          >
            {shortModelDisplay(modelId)}
          </button>
        );
      })}
    </div>
  );
}

export const ModelSelectorPopover: React.FC<ModelSelectorPopoverProps> = ({
  catalog,
  catalogLoading,
  health,
  override,
  providerApiKeys = {},
  onOverrideChange,
  onProviderApiKeyChange,
  onRefreshCatalog,
  useStreaming,
  onUseStreamingChange,
  enableBudgetCap,
  onEnableBudgetCapChange,
  maxCostUsd,
  onMaxCostUsdChange,
}) => {
  const [open, setOpen] = useState(false);
  const [showKey, setShowKey] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);

  const defaultProvider = (catalog?.default_provider ?? health?.llm_provider ?? "openrouter") as LLMProvider;
  const defaultFast = catalog?.default_fast_model ?? health?.fast_model ?? "";
  const defaultGen = catalog?.default_generation_model ?? health?.generation_model ?? "";
  const defaultTemp = catalog?.default_temperature ?? 0.0;
  const allowClientKeys = catalog?.allow_client_provider_keys ?? true;

  const activeProvider: LLMProvider = override?.provider ?? defaultProvider;
  const providers: ProviderCatalogItem[] = catalog?.providers ?? [];
  const activeProviderMeta = providers.find((p) => p.id === activeProvider) ?? null;

  const activeSessionKey = (providerApiKeys[activeProvider] ?? "").trim();
  const activeSdkInstalled = activeProviderMeta?.sdk_installed ?? true;
  const activeRequiresKey = activeProviderMeta?.requires_api_key ?? activeProvider !== "ollama";
  const activeHasCredentials =
    !activeRequiresKey || Boolean(activeProviderMeta?.configured) || Boolean(activeSessionKey);

  const providerMatchesDefault = activeProvider === defaultProvider;
  const effectiveFast =
    override?.fast_model ??
    (providerMatchesDefault ? defaultFast : (activeProviderMeta?.default_fast_model ?? ""));
  const effectiveGen =
    override?.generation_model ??
    (providerMatchesDefault ? defaultGen : (activeProviderMeta?.default_generation_model ?? ""));
  const effectiveTemp = typeof override?.temperature === "number" ? override.temperature : defaultTemp;

  const isOverridden = Boolean(
    override &&
      (override.provider !== undefined ||
        override.fast_model !== undefined ||
        override.generation_model !== undefined ||
        override.temperature !== undefined)
  );

  const commitIfChanged = (candidate: {
    provider: LLMProvider;
    fast_model: string;
    generation_model: string;
    temperature: number;
  }) => {
    const matchesServerDefault =
      candidate.provider === defaultProvider &&
      candidate.fast_model.trim() === defaultFast.trim() &&
      candidate.generation_model.trim() === defaultGen.trim() &&
      Math.abs(candidate.temperature - defaultTemp) < 1e-6;

    if (matchesServerDefault) {
      onOverrideChange(null);
      return;
    }

    onOverrideChange({
      provider: candidate.provider,
      fast_model: candidate.fast_model.trim() || undefined,
      generation_model: candidate.generation_model.trim() || undefined,
      temperature: Number(candidate.temperature.toFixed(2)),
    });
  };

  const current = {
    provider: activeProvider,
    fast_model: effectiveFast,
    generation_model: effectiveGen,
    temperature: effectiveTemp,
  };

  const handleSelectProvider = (target: ProviderCatalogItem) => {
    const canSelect = target.configured || allowClientKeys;
    if (!canSelect) return;

    setShowKey(false);
    const isTargetDefault = target.id === defaultProvider;
    const nextFast = isTargetDefault ? defaultFast : target.detected_models[0] || target.default_fast_model;
    const nextGen = isTargetDefault
      ? defaultGen
      : target.detected_models[0] || target.default_generation_model;
    const nextTemp = isTargetDefault
      ? defaultTemp
      : target.is_local && nextGen.toLowerCase().includes("inkling")
        ? 1.0
        : effectiveTemp;

    commitIfChanged({
      provider: target.id,
      fast_model: nextFast,
      generation_model: nextGen,
      temperature: nextTemp,
    });
  };

  const fastSuggestions = activeProviderMeta
    ? Array.from(
        new Set([...activeProviderMeta.detected_models, ...activeProviderMeta.recommended_fast_models])
      ).slice(0, 6)
    : [];
  const genSuggestions = activeProviderMeta
    ? Array.from(
        new Set([
          ...activeProviderMeta.detected_models,
          ...activeProviderMeta.recommended_generation_models,
        ])
      ).slice(0, 6)
    : [];

  const localOffline = activeProvider === "ollama" && activeProviderMeta?.reachable === false;
  const needsSetup = !activeSdkInstalled || !activeHasCredentials || localOffline;
  const triggerTone: Tone = needsSetup
    ? "warn"
    : isOverridden || Boolean(activeSessionKey)
      ? "accent"
      : "ok";

  const modeLabel = activeSessionKey ? "Session key" : isOverridden ? "Custom" : "Server default";
  const hasRunOptions = Boolean(onUseStreamingChange || onEnableBudgetCapChange);

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        aria-expanded={open}
        aria-haspopup="dialog"
        aria-label="Configure LLM provider and models"
        title={`${providerShortLabel(activeProvider)}: ${effectiveGen || "default"} (fast tier ${effectiveFast || "default"}, temperature ${effectiveTemp.toFixed(1)})`}
        className={cn(
          "inline-flex h-8 max-w-[260px] cursor-pointer items-center gap-2 rounded-md border px-2.5 text-ui transition-colors",
          open
            ? "border-line-strong bg-selected text-ink"
            : "border-line bg-paper text-ink hover:border-line-strong"
        )}
      >
        <StatusDot tone={triggerTone} />
        <span className="hidden shrink-0 text-small text-ink-3 sm:inline">
          {providerShortLabel(activeProvider)}
        </span>
        <span className="truncate font-mono text-[12px] text-ink">
          {shortModelDisplay(effectiveGen || effectiveFast)}
        </span>
        <ChevronDown
          className={cn("h-3.5 w-3.5 shrink-0 text-ink-3 transition-transform", open && "rotate-180")}
          aria-hidden
        />
      </button>

      <Popover
        open={open}
        onClose={() => setOpen(false)}
        anchorRef={triggerRef}
        placement="bottom-end"
        width={392}
        ariaLabel="Model runtime settings"
        testId="model-selector-popover"
        className="p-0"
      >
        {/* Header */}
        <div className="flex items-center justify-between gap-2 border-b border-line px-4 py-3">
          <div className="flex items-center gap-2">
            <span className="text-ui font-semibold text-ink">Runtime</span>
            <Tag tone={isOverridden || activeSessionKey ? "accent" : "neutral"}>{modeLabel}</Tag>
          </div>
          <div className="flex items-center gap-1">
            {isOverridden && (
              <Button variant="ghost" size="xs" icon={RotateCcw} onClick={() => onOverrideChange(null)}>
                Reset
              </Button>
            )}
            <IconButton
              icon={RefreshCw}
              size="sm"
              label="Refresh providers and detect local models"
              onClick={onRefreshCatalog}
              disabled={catalogLoading}
              className={catalogLoading ? "[&>svg]:animate-spin" : undefined}
            />
          </div>
        </div>

        <div className="space-y-5 px-4 py-4">
          {/* Provider */}
          <div>
            <div className="mb-2 flex items-center justify-between gap-2">
              <span className="text-small font-medium text-ink">Provider</span>
              {activeProvider === "ollama" && catalog?.ollama_base_url && (
                <span className="inline-flex items-center gap-1 font-mono text-[11px] text-ink-3">
                  <Server className="h-3 w-3" aria-hidden />
                  {catalog.ollama_base_url.replace(/^https?:\/\//, "")}
                </span>
              )}
            </div>

            {providers.length > 0 ? (
              <div className="grid grid-cols-3 gap-1.5">
                {providers.map((prov) => {
                  const selected = prov.id === activeProvider;
                  const provSessionKey = (providerApiKeys[prov.id] ?? "").trim();
                  const sdkOk = prov.sdk_installed ?? true;
                  const ready = sdkOk && (prov.configured || Boolean(provSessionKey));
                  const disabled = !prov.configured && !allowClientKeys;
                  const tone: Tone =
                    !sdkOk || (prov.is_local && prov.reachable === false)
                      ? "warn"
                      : provSessionKey
                        ? "accent"
                        : ready
                          ? "ok"
                          : "neutral";
                  const status = !sdkOk
                    ? prov.status_detail || `Install the providers extra to use ${prov.label}`
                    : provSessionKey
                      ? `${prov.label}: session key active`
                      : prov.status_detail ||
                        (prov.is_local
                          ? `${prov.label}: ${prov.detected_models.length} models detected`
                          : `${prov.label}: ready via server key`);
                  return (
                    <button
                      key={prov.id}
                      type="button"
                      disabled={disabled}
                      aria-pressed={selected}
                      aria-label={providerShortLabel(prov.id)}
                      onClick={() => handleSelectProvider(prov)}
                      title={status}
                      className={cn(
                        "flex h-8 items-center justify-between gap-1.5 rounded-md border px-2.5 text-left text-small font-medium transition-colors",
                        disabled
                          ? "cursor-not-allowed border-line bg-sunken text-ink-4"
                          : selected
                            ? "cursor-pointer border-ink bg-paper text-ink shadow-[0_0_0_1px_var(--ink)]"
                            : "cursor-pointer border-line bg-paper text-ink-2 hover:border-line-strong hover:text-ink"
                      )}
                    >
                      <span className="truncate">{providerShortLabel(prov.id)}</span>
                      <StatusDot tone={tone} />
                    </button>
                  );
                })}
              </div>
            ) : (
              <Callout tone="warn">
                Can't load the provider list. Start the API with{" "}
                <code className="font-mono text-[11.5px]">make api</code>, then refresh.
              </Callout>
            )}

            {!activeSdkInstalled && activeProviderMeta?.status_detail && (
              <Callout tone="warn" className="mt-2">
                {activeProviderMeta.status_detail}
              </Callout>
            )}
            {activeSdkInstalled && localOffline && activeProviderMeta?.status_detail && (
              <Callout tone="warn" className="mt-2">
                {activeProviderMeta.status_detail}
              </Callout>
            )}
          </div>

          {/* Session-only provider key */}
          {activeRequiresKey && allowClientKeys && onProviderApiKeyChange && (
            <Field
              label={`${providerShortLabel(activeProvider)} API key`}
              htmlFor="studio-provider-api-key-input"
              aside={
                activeSessionKey ? (
                  <span className="text-small text-accent">Session key active</span>
                ) : activeProviderMeta?.configured ? (
                  <span className="text-small text-ok">Using the server key</span>
                ) : (
                  <span className="text-small text-warn">Key required</span>
                )
              }
            >
              <div className="flex items-center gap-1.5">
                <div className="relative flex-1">
                  <input
                    id="studio-provider-api-key-input"
                    type={showKey ? "text" : "password"}
                    autoComplete="off"
                    spellCheck={false}
                    value={providerApiKeys[activeProvider] ?? ""}
                    onChange={(e) => onProviderApiKeyChange(activeProvider, e.target.value)}
                    placeholder={
                      activeProviderMeta?.configured
                        ? "Optional: overrides the server key"
                        : `Paste a ${providerShortLabel(activeProvider)} key for this tab`
                    }
                    className={cn(INPUT_CLASS, "h-8 pr-8 pl-2.5 font-mono text-small")}
                  />
                  <IconButton
                    icon={showKey ? EyeOff : Eye}
                    size="sm"
                    label={showKey ? "Hide API key" : "Show API key"}
                    onClick={() => setShowKey((prev) => !prev)}
                    className="absolute top-1/2 right-1 -translate-y-1/2"
                  />
                </div>
                {activeSessionKey && (
                  <Button size="sm" onClick={() => onProviderApiKeyChange(activeProvider, "")}>
                    Clear
                  </Button>
                )}
              </div>
            </Field>
          )}

          {/* Models */}
          <Field
            label="Fast model"
            htmlFor="studio-fast-model-input"
            aside={
              activeProvider === "ollama" && (activeProviderMeta?.detected_models.length ?? 0) > 0 ? (
                <span className="text-small text-ok">
                  {activeProviderMeta?.detected_models.length} running locally
                </span>
              ) : (
                <span className="text-small text-ink-3">Routing, grading and audit</span>
              )
            }
          >
            <input
              id="studio-fast-model-input"
              type="text"
              value={effectiveFast}
              spellCheck={false}
              onChange={(e) => commitIfChanged({ ...current, fast_model: e.target.value })}
              placeholder="e.g. qwen/qwen3-30b-a3b-instruct-2507"
              className={cn(INPUT_CLASS, "h-8 px-2.5 font-mono text-small")}
            />
            <SuggestionChips
              models={fastSuggestions}
              active={effectiveFast}
              onPick={(modelId) => commitIfChanged({ ...current, fast_model: modelId })}
            />
          </Field>

          <Field
            label="Generation model"
            htmlFor="studio-gen-model-input"
            aside={
              effectiveFast && effectiveGen !== effectiveFast ? (
                <button
                  type="button"
                  onClick={() => commitIfChanged({ ...current, generation_model: effectiveFast })}
                  className="cursor-pointer text-small text-accent hover:underline"
                >
                  Use the fast model
                </button>
              ) : (
                <span className="text-small text-ink-3">Writes the answer</span>
              )
            }
          >
            <input
              id="studio-gen-model-input"
              type="text"
              value={effectiveGen}
              spellCheck={false}
              onChange={(e) => commitIfChanged({ ...current, generation_model: e.target.value })}
              placeholder="e.g. deepseek/deepseek-v3.2"
              className={cn(INPUT_CLASS, "h-8 px-2.5 font-mono text-small")}
            />
            <SuggestionChips
              models={genSuggestions}
              active={effectiveGen}
              onPick={(modelId) => commitIfChanged({ ...current, generation_model: modelId })}
            />
          </Field>

          {/* Temperature */}
          <div>
            <div className="mb-1.5 flex items-center justify-between gap-2">
              <label htmlFor="studio-temperature" className="text-small font-medium text-ink">
                Temperature
              </label>
              <div className="flex items-center gap-1">
                {[0, 0.3, 1].map((val) => (
                  <button
                    key={val}
                    type="button"
                    aria-pressed={Math.abs(effectiveTemp - val) < 1e-6}
                    onClick={() => commitIfChanged({ ...current, temperature: val })}
                    className={cn(
                      "h-6 cursor-pointer rounded-md px-1.5 text-small tabular-nums transition-colors",
                      Math.abs(effectiveTemp - val) < 1e-6
                        ? "bg-selected font-medium text-ink"
                        : "text-ink-3 hover:bg-hover hover:text-ink"
                    )}
                  >
                    {val.toFixed(1)}
                  </button>
                ))}
                <span className="ml-1 w-9 text-right text-small font-medium tabular-nums text-ink">
                  {effectiveTemp.toFixed(2)}
                </span>
              </div>
            </div>
            <input
              id="studio-temperature"
              aria-label="Sampling temperature"
              type="range"
              min={0}
              max={2}
              step={0.05}
              value={effectiveTemp}
              onChange={(e) => commitIfChanged({ ...current, temperature: Number(e.target.value) })}
              className="ar-range w-full"
            />
          </div>

          {/* Run options */}
          {hasRunOptions && (
            <div className="space-y-3.5 border-t border-line pt-4">
              {onUseStreamingChange && (
                <Switch
                  label="Stream pipeline steps"
                  description="Show each step as it finishes instead of waiting for the full answer."
                  checked={Boolean(useStreaming)}
                  onChange={onUseStreamingChange}
                />
              )}
              {onEnableBudgetCapChange && (
                <div>
                  <Switch
                    label="Spend cap per question"
                    description="Stops a run that would cost more than the cap."
                    checked={Boolean(enableBudgetCap)}
                    onChange={onEnableBudgetCapChange}
                  />
                  {enableBudgetCap && onMaxCostUsdChange && typeof maxCostUsd === "number" && (
                    <div className="mt-2.5">
                      <div className="mb-1 flex items-center justify-between text-small">
                        <label htmlFor="studio-spend-cap" className="text-ink-2">
                          Cap
                        </label>
                        <span className="font-medium tabular-nums text-ink">${maxCostUsd.toFixed(2)}</span>
                      </div>
                      <input
                        id="studio-spend-cap"
                        type="range"
                        min={0.01}
                        max={0.5}
                        step={0.01}
                        value={maxCostUsd}
                        onChange={(e) => onMaxCostUsdChange(Number(e.target.value))}
                        className="ar-range w-full"
                      />
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-start gap-2 border-t border-line bg-sunken px-4 py-2.5 text-small text-ink-3">
          <Lock className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
          <span>
            {allowClientKeys
              ? "Session keys stay in this tab's sessionStorage and are sent via X-Provider-API-Key header only."
              : "Server-managed keys only (BYOK disabled by server policy)."}
          </span>
        </div>
      </Popover>
    </>
  );
};
