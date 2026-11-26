import React, { useEffect, useRef, useState } from "react";
import {
  ChevronDown,
  Cpu,
  Eye,
  EyeOff,
  KeyRound,
  Lock,
  RefreshCw,
  RotateCcw,
  Server,
} from "lucide-react";
import type {
  HealthResponse,
  LLMOverride,
  LLMProvider,
  ModelsCatalogResponse,
  ProviderApiKeys,
  ProviderCatalogItem,
} from "../types/api";

interface ModelSelectorPopoverProps {
  catalog: ModelsCatalogResponse | null;
  catalogLoading: boolean;
  health: HealthResponse | null;
  override: LLMOverride | null;
  providerApiKeys?: ProviderApiKeys;
  onOverrideChange: (next: LLMOverride | null) => void;
  onProviderApiKeyChange?: (provider: LLMProvider, apiKey: string) => void;
  onRefreshCatalog: () => void;
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

export const ModelSelectorPopover: React.FC<ModelSelectorPopoverProps> = ({
  catalog,
  catalogLoading,
  health,
  override,
  providerApiKeys = {},
  onOverrideChange,
  onProviderApiKeyChange,
  onRefreshCatalog,
}) => {
  const [open, setOpen] = useState(false);
  const [showKey, setShowKey] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpen(false);
      }
    };
    window.addEventListener("mousedown", onPointerDown);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("mousedown", onPointerDown);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  const defaultProvider = (catalog?.default_provider ??
    health?.llm_provider ??
    "openrouter") as LLMProvider;
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
    (providerMatchesDefault
      ? defaultFast
      : activeProviderMeta?.default_fast_model ?? "");
  const effectiveGen =
    override?.generation_model ??
    (providerMatchesDefault
      ? defaultGen
      : activeProviderMeta?.default_generation_model ?? "");
  const effectiveTemp =
    typeof override?.temperature === "number" ? override.temperature : defaultTemp;

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

  const handleSelectProvider = (target: ProviderCatalogItem) => {
    const canSelect = target.configured || allowClientKeys;
    if (!canSelect) return;

    setShowKey(false);
    const isTargetDefault = target.id === defaultProvider;
    const nextFast = isTargetDefault
      ? defaultFast
      : target.detected_models[0] || target.default_fast_model;
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
        new Set([
          ...activeProviderMeta.detected_models,
          ...activeProviderMeta.recommended_fast_models,
        ])
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

  const localOffline =
    activeProvider === "ollama" && activeProviderMeta?.reachable === false;
  const needsSetup = !activeSdkInstalled || !activeHasCredentials || localOffline;

  const dotClass = needsSetup
    ? "bg-amber-500"
    : isOverridden || Boolean(activeSessionKey)
      ? "bg-sky-500"
      : "bg-emerald-500";

  return (
    <div ref={containerRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        aria-expanded={open}
        aria-haspopup="dialog"
        aria-label="Configure LLM provider and models"
        title={`LLM Runtime: ${activeProvider} · fast=${effectiveFast || "default"} · gen=${effectiveGen || "default"} · temp=${effectiveTemp.toFixed(1)}`}
        className={`inline-flex h-7.5 max-w-[230px] items-center gap-1.5 rounded-md border px-2 text-xs transition cursor-pointer ${
          open || isOverridden || Boolean(activeSessionKey)
            ? "border-[var(--border-strong)] bg-[var(--bg-elevated)] text-[var(--text-primary)]"
            : "border-[var(--border-subtle)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:border-[var(--border-strong)] hover:text-[var(--text-primary)]"
        }`}
      >
        <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${dotClass}`} />
        <Cpu className="hidden h-3 w-3 shrink-0 text-[var(--text-muted)] sm:inline" />
        <span className="hidden shrink-0 font-mono text-[10.5px] text-[var(--text-muted)] sm:inline">
          {providerShortLabel(activeProvider)}
        </span>
        <span className="hidden text-[var(--text-muted)]/40 sm:inline">/</span>
        <span className="truncate font-mono text-[11px] font-medium text-[var(--text-primary)]">
          {shortModelDisplay(effectiveGen || effectiveFast)}
        </span>
        <ChevronDown
          className={`h-3 w-3 shrink-0 text-[var(--text-muted)] transition-transform ${
            open ? "rotate-180" : ""
          }`}
        />
      </button>

      {open && (
        <div
          role="dialog"
          aria-label="Model Runtime Settings"
          data-testid="model-selector-popover"
          className="absolute right-0 top-full z-50 mt-1.5 w-[380px] max-w-[calc(100vw-1.5rem)] rounded-lg border border-[var(--border-strong)] bg-[var(--bg-panel)] p-3.5 text-xs shadow-xl"
        >
          {/* Header */}
          <div className="flex items-center justify-between gap-2 border-b border-[var(--border-subtle)] pb-2.5">
            <div className="flex items-center gap-1.5">
              <Cpu className="h-3.5 w-3.5 text-[var(--text-muted)]" />
              <span className="font-semibold text-[var(--text-primary)]">
                Model Runtime
              </span>
              <span
                className={`rounded border px-1.5 py-0.2 font-mono text-[9.5px] font-medium ${
                  isOverridden || Boolean(activeSessionKey)
                    ? "border-sky-500/40 bg-sky-500/10 text-sky-600 dark:text-sky-300"
                    : "border-[var(--border-subtle)] bg-[var(--bg-surface)] text-[var(--text-muted)]"
                }`}
              >
                {activeSessionKey ? "byok" : isOverridden ? "override" : ".env default"}
              </span>
            </div>

            <div className="flex items-center gap-1">
              {isOverridden && (
                <button
                  type="button"
                  onClick={() => onOverrideChange(null)}
                  title="Reset to server .env defaults"
                  className="inline-flex items-center gap-1 rounded border border-[var(--border-subtle)] px-1.5 py-0.5 font-mono text-[10px] text-[var(--text-muted)] transition hover:border-[var(--border-strong)] hover:text-[var(--text-primary)] cursor-pointer"
                >
                  <RotateCcw className="h-2.5 w-2.5" />
                  <span>Reset</span>
                </button>
              )}
              <button
                type="button"
                onClick={onRefreshCatalog}
                disabled={catalogLoading}
                title="Refresh provider readiness & detect local models"
                className="rounded border border-[var(--border-subtle)] p-1 text-[var(--text-muted)] transition hover:border-[var(--border-strong)] hover:text-[var(--text-primary)] disabled:opacity-50 cursor-pointer"
              >
                <RefreshCw
                  className={`h-3 w-3 ${catalogLoading ? "animate-spin" : ""}`}
                />
              </button>
            </div>
          </div>

          {/* Provider Selector */}
          <div className="mt-3">
            <div className="mb-1.5 flex items-center justify-between">
              <span className="font-mono text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
                Provider
              </span>
              {activeProvider === "ollama" && catalog?.ollama_base_url && (
                <span className="inline-flex items-center gap-1 font-mono text-[10px] text-[var(--text-muted)]">
                  <Server className="h-2.5 w-2.5" />
                  <span>{catalog.ollama_base_url.replace(/^https?:\/\//, "")}</span>
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
                  return (
                    <button
                      key={prov.id}
                      type="button"
                      disabled={disabled}
                      onClick={() => handleSelectProvider(prov)}
                      title={
                        !sdkOk
                          ? prov.status_detail || `Install uv sync --extra providers for ${prov.label}`
                          : provSessionKey
                            ? `${prov.label} (session API key active)`
                            : prov.status_detail ||
                              (prov.is_local
                                ? `${prov.label} (${prov.detected_models.length} model(s) detected)`
                                : `${prov.label} ready via .env`)
                      }
                      className={`flex items-center justify-between gap-1 rounded-md border px-2 py-1.5 text-left text-[11px] font-medium transition ${
                        disabled
                          ? "cursor-not-allowed border-[var(--border-subtle)] bg-[var(--bg-surface)]/40 text-[var(--text-muted)] opacity-50"
                          : selected
                            ? "cursor-pointer border-[var(--border-strong)] bg-[var(--bg-elevated)] text-[var(--text-primary)]"
                            : "cursor-pointer border-[var(--border-subtle)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:border-[var(--border-strong)] hover:text-[var(--text-primary)]"
                      }`}
                    >
                      <span className="truncate">{providerShortLabel(prov.id)}</span>
                      <span
                        className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                          !sdkOk
                            ? "bg-amber-500"
                            : prov.is_local && prov.reachable === false
                              ? "bg-amber-500"
                              : provSessionKey
                                ? "bg-sky-500"
                                : ready
                                  ? "bg-emerald-500"
                                  : "bg-zinc-500/45"
                        }`}
                      />
                    </button>
                  );
                })}
              </div>
            ) : (
              <p className="rounded-md border border-[var(--border-subtle)] bg-[var(--bg-surface)] px-2.5 py-2 text-[11px] text-[var(--text-muted)]">
                Backend catalog unavailable. Start FastAPI (`make api`) and click refresh.
              </p>
            )}

            {/* SDK missing or local unreachable banner */}
            {!activeSdkInstalled && activeProviderMeta?.status_detail && (
              <p className="mt-2 rounded border border-amber-500/35 bg-amber-500/10 px-2.5 py-1.5 text-[10.5px] leading-snug text-amber-700 dark:text-amber-300">
                {activeProviderMeta.status_detail}
              </p>
            )}
            {activeSdkInstalled && localOffline && activeProviderMeta?.status_detail && (
              <p className="mt-2 rounded border border-amber-500/35 bg-amber-500/10 px-2.5 py-1.5 text-[10.5px] leading-snug text-amber-700 dark:text-amber-300">
                {activeProviderMeta.status_detail}
              </p>
            )}
          </div>

          {/* Session-Only Provider API Key (BYOK) for Cloud Providers */}
          {activeRequiresKey && allowClientKeys && onProviderApiKeyChange && (
            <div className="mt-3 space-y-1.5 border-t border-[var(--border-subtle)] pt-2.5">
              <div className="flex items-center justify-between gap-2">
                <label
                  htmlFor="studio-provider-api-key-input"
                  className="inline-flex items-center gap-1 font-mono text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]"
                >
                  <KeyRound className="h-2.5 w-2.5" />
                  <span>{providerShortLabel(activeProvider)} API Key</span>
                </label>
                {activeSessionKey ? (
                  <span className="font-mono text-[9.5px] font-medium text-sky-600 dark:text-sky-400">
                    session key active
                  </span>
                ) : activeProviderMeta?.configured ? (
                  <span className="font-mono text-[9.5px] text-emerald-600 dark:text-emerald-400">
                    using server .env
                  </span>
                ) : (
                  <span className="font-mono text-[9.5px] text-amber-600 dark:text-amber-400">
                    key required
                  </span>
                )}
              </div>

              <div className="flex items-center gap-1">
                <div className="relative flex-1">
                  <input
                    id="studio-provider-api-key-input"
                    type={showKey ? "text" : "password"}
                    autoComplete="off"
                    spellCheck={false}
                    value={providerApiKeys[activeProvider] ?? ""}
                    onChange={(e) =>
                      onProviderApiKeyChange(activeProvider, e.target.value)
                    }
                    placeholder={
                      activeProviderMeta?.configured
                        ? "Optional override (uses server .env)"
                        : `Paste ${providerShortLabel(activeProvider)} key (session-only)`
                    }
                    className="h-8 w-full rounded-md border border-[var(--border-strong)] bg-[var(--bg-surface)] pl-2.5 pr-7 font-mono text-[11px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:border-[var(--text-muted)] focus:outline-none"
                  />
                  <button
                    type="button"
                    aria-label={showKey ? "Hide API key" : "Show API key"}
                    onClick={() => setShowKey((prev) => !prev)}
                    className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded p-1 text-[var(--text-muted)] hover:text-[var(--text-primary)] cursor-pointer"
                  >
                    {showKey ? (
                      <EyeOff className="h-3 w-3" />
                    ) : (
                      <Eye className="h-3 w-3" />
                    )}
                  </button>
                </div>
                {activeSessionKey && (
                  <button
                    type="button"
                    onClick={() => onProviderApiKeyChange(activeProvider, "")}
                    className="h-8 rounded-md border border-[var(--border-subtle)] bg-[var(--bg-surface)] px-2 font-mono text-[10px] text-[var(--text-muted)] transition hover:border-[var(--border-strong)] hover:text-[var(--text-primary)] cursor-pointer"
                  >
                    Clear
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Fast Model Tier */}
          <div className="mt-3 space-y-1.5">
            <label
              htmlFor="studio-fast-model-input"
              className="flex items-center justify-between font-mono text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]"
            >
              <span>Fast Tier (Router · CRAG · Audit)</span>
              {activeProvider === "ollama" &&
                (activeProviderMeta?.detected_models.length ?? 0) > 0 && (
                  <span className="text-emerald-600 dark:text-emerald-400">
                    {activeProviderMeta?.detected_models.length} live
                  </span>
                )}
            </label>
            <input
              id="studio-fast-model-input"
              type="text"
              value={effectiveFast}
              onChange={(e) =>
                commitIfChanged({
                  provider: activeProvider,
                  fast_model: e.target.value,
                  generation_model: effectiveGen,
                  temperature: effectiveTemp,
                })
              }
              placeholder="e.g. qwen/qwen3-30b-a3b-instruct-2507"
              className="h-8 w-full rounded-md border border-[var(--border-strong)] bg-[var(--bg-surface)] px-2.5 font-mono text-[11px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:border-[var(--text-muted)] focus:outline-none"
            />
            {fastSuggestions.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {fastSuggestions.map((modelId) => {
                  const active = modelId === effectiveFast;
                  return (
                    <button
                      key={modelId}
                      type="button"
                      onClick={() =>
                        commitIfChanged({
                          provider: activeProvider,
                          fast_model: modelId,
                          generation_model: effectiveGen,
                          temperature: effectiveTemp,
                        })
                      }
                      className={`rounded border px-1.5 py-0.5 font-mono text-[10px] transition cursor-pointer ${
                        active
                          ? "border-sky-500/45 bg-sky-500/12 text-sky-700 dark:text-sky-300"
                          : "border-[var(--border-subtle)] bg-[var(--bg-surface)] text-[var(--text-muted)] hover:border-[var(--border-strong)] hover:text-[var(--text-primary)]"
                      }`}
                    >
                      {shortModelDisplay(modelId)}
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {/* Generation Model Tier */}
          <div className="mt-3 space-y-1.5">
            <div className="flex items-center justify-between">
              <label
                htmlFor="studio-gen-model-input"
                className="font-mono text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]"
              >
                Generation Tier (Synthesis)
              </label>
              {effectiveFast && effectiveGen !== effectiveFast && (
                <button
                  type="button"
                  onClick={() =>
                    commitIfChanged({
                      provider: activeProvider,
                      fast_model: effectiveFast,
                      generation_model: effectiveFast,
                      temperature: effectiveTemp,
                    })
                  }
                  className="font-mono text-[10px] text-[var(--text-muted)] hover:text-[var(--text-primary)] cursor-pointer"
                >
                  Same as fast
                </button>
              )}
            </div>
            <input
              id="studio-gen-model-input"
              type="text"
              value={effectiveGen}
              onChange={(e) =>
                commitIfChanged({
                  provider: activeProvider,
                  fast_model: effectiveFast,
                  generation_model: e.target.value,
                  temperature: effectiveTemp,
                })
              }
              placeholder="e.g. deepseek/deepseek-v3.2"
              className="h-8 w-full rounded-md border border-[var(--border-strong)] bg-[var(--bg-surface)] px-2.5 font-mono text-[11px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:border-[var(--text-muted)] focus:outline-none"
            />
            {genSuggestions.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {genSuggestions.map((modelId) => {
                  const active = modelId === effectiveGen;
                  return (
                    <button
                      key={modelId}
                      type="button"
                      onClick={() =>
                        commitIfChanged({
                          provider: activeProvider,
                          fast_model: effectiveFast,
                          generation_model: modelId,
                          temperature: effectiveTemp,
                        })
                      }
                      className={`rounded border px-1.5 py-0.5 font-mono text-[10px] transition cursor-pointer ${
                        active
                          ? "border-emerald-500/45 bg-emerald-500/12 text-emerald-700 dark:text-emerald-300"
                          : "border-[var(--border-subtle)] bg-[var(--bg-surface)] text-[var(--text-muted)] hover:border-[var(--border-strong)] hover:text-[var(--text-primary)]"
                      }`}
                    >
                      {shortModelDisplay(modelId)}
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {/* Temperature Control */}
          <div className="mt-3 border-t border-[var(--border-subtle)] pt-2.5">
            <div className="mb-1.5 flex items-center justify-between">
              <span className="font-mono text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
                Temperature
              </span>
              <div className="flex items-center gap-1">
                {(
                  [
                    [0.0, "0.0"],
                    [0.3, "0.3"],
                    [1.0, "1.0"],
                  ] as const
                ).map(([val, lbl]) => (
                  <button
                    key={lbl}
                    type="button"
                    onClick={() =>
                      commitIfChanged({
                        provider: activeProvider,
                        fast_model: effectiveFast,
                        generation_model: effectiveGen,
                        temperature: val,
                      })
                    }
                    className={`rounded border px-1.5 py-0.2 font-mono text-[10px] tabular-nums transition cursor-pointer ${
                      Math.abs(effectiveTemp - val) < 1e-6
                        ? "border-[var(--border-strong)] bg-[var(--bg-elevated)] text-[var(--text-primary)]"
                        : "border-[var(--border-subtle)] text-[var(--text-muted)] hover:text-[var(--text-primary)]"
                    }`}
                  >
                    {lbl}
                  </button>
                ))}
                <span className="ml-1 w-8 text-right font-mono text-[11px] font-semibold text-[var(--text-primary)] tabular-nums">
                  {effectiveTemp.toFixed(2)}
                </span>
              </div>
            </div>
            <input
              aria-label="Sampling temperature"
              type="range"
              min={0}
              max={2}
              step={0.05}
              value={effectiveTemp}
              onChange={(e) =>
                commitIfChanged({
                  provider: activeProvider,
                  fast_model: effectiveFast,
                  generation_model: effectiveGen,
                  temperature: Number(e.target.value),
                })
              }
              className="ar-range w-full"
            />
          </div>

          {/* Security Footer */}
          <div className="mt-3 flex items-center gap-1.5 border-t border-[var(--border-subtle)] pt-2 text-[10.5px] text-[var(--text-muted)]">
            <Lock className="h-3 w-3 shrink-0" />
            <span>
              {allowClientKeys
                ? "Session keys stay in sessionStorage, sent via X-Provider-API-Key header only."
                : "Server-managed .env keys only (BYOK disabled by server policy)."}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
