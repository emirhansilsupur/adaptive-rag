import React, { useRef, useState } from "react";
import {
  Columns2,
  Database,
  MessageSquare,
  PanelLeftClose,
  PenLine,
  Plus,
  RefreshCw,
  Settings2,
  Trash2,
  X,
  type LucideIcon,
} from "lucide-react";
import { resolveEffectiveInstructions } from "../constants/pipeline";
import { formatInt, formatRelativeTime } from "../lib/format";
import type { AppView, ChatSession, HealthResponse } from "../types/api";
import { LogoMark } from "./ui/Logo";
import { Popover } from "./ui/Popover";
import {
  Button,
  Callout,
  cn,
  Field,
  IconButton,
  INPUT_CLASS,
  StatusDot,
  type Tone,
} from "./ui/primitives";

interface SidebarProps {
  activeView: AppView;
  onChangeView: (view: AppView) => void;
  sessions: ChatSession[];
  activeSessionId: string;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  onDeleteSession: (id: string) => void;
  apiUrl: string;
  onApiUrlChange: (url: string) => void;
  apiKey?: string;
  onApiKeyChange?: (key: string) => void;
  health: HealthResponse | null;
  healthError: string | null;
  healthLoading: boolean;
  onRefreshHealth: () => void;
  documentCount?: number;
  onCollapse: () => void;
  /** "drawer" renders a close button instead of the collapse control. */
  variant?: "docked" | "drawer";
}

const NAV_ITEMS: { id: AppView; label: string; icon: LucideIcon; hint: string }[] = [
  { id: "workbench", label: "Workbench", icon: MessageSquare, hint: "Ask questions and inspect answers" },
  { id: "compare", label: "Arena", icon: Columns2, hint: "Run two presets side by side" },
  { id: "knowledge", label: "Knowledge base", icon: Database, hint: "Add and manage documents" },
];

function hostLabel(apiUrl: string): string {
  if (!apiUrl) return "Same origin";
  try {
    return new URL(apiUrl).host;
  } catch {
    return apiUrl;
  }
}

/* -------------------------------------------------------------------------- */
/* Connection panel                                                           */
/* -------------------------------------------------------------------------- */

function ServiceRow({ label, tone, value }: { label: string; tone: Tone; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 py-1">
      <span className="flex min-w-0 items-center gap-2 text-ui text-ink">
        <StatusDot tone={tone} />
        <span className="truncate">{label}</span>
      </span>
      <span className="shrink-0 text-small text-ink-3">{value}</span>
    </div>
  );
}

function ConnectionPanel({
  apiUrl,
  onApiUrlChange,
  apiKey,
  onApiKeyChange,
  health,
  healthError,
  healthLoading,
  onRefreshHealth,
}: Pick<
  SidebarProps,
  | "apiUrl"
  | "onApiUrlChange"
  | "apiKey"
  | "onApiKeyChange"
  | "health"
  | "healthError"
  | "healthLoading"
  | "onRefreshHealth"
>) {
  const [apiInput, setApiInput] = useState(apiUrl);
  const [apiKeyInput, setApiKeyInput] = useState(apiKey ?? "");
  const authRequired = Boolean(health?.security?.api_key_auth_enabled);
  const keySet = Boolean(apiKey?.trim());

  const lfMode = health?.langfuse?.mode ?? (health?.langfuse_enabled ? "cloud" : "disabled");
  const fcStatus = health?.firecrawl?.status;
  const parentChild = health?.parent_child_retrieval;

  return (
    <div>
      <div className="flex items-center justify-between gap-2 border-b border-line px-4 py-3">
        <span className="text-ui font-semibold text-ink">Connection</span>
        <Button
          variant="ghost"
          size="xs"
          icon={RefreshCw}
          loading={healthLoading}
          onClick={onRefreshHealth}
        >
          Check again
        </Button>
      </div>

      <div className="space-y-4 px-4 py-4">
        <Field
          label="API endpoint"
          htmlFor="studio-endpoint"
          hint="Leave empty when the Studio is served by the API itself."
        >
          <form
            className="flex gap-1.5"
            onSubmit={(e) => {
              e.preventDefault();
              onApiUrlChange(apiInput.trim().replace(/\/+$/, ""));
            }}
          >
            <input
              id="studio-endpoint"
              type="text"
              value={apiInput}
              spellCheck={false}
              placeholder="http://localhost:8000"
              onChange={(e) => setApiInput(e.target.value)}
              className={cn(INPUT_CLASS, "h-8 min-w-0 flex-1 px-2.5 font-mono text-small")}
            />
            <Button type="submit" size="sm">
              Save
            </Button>
          </form>
        </Field>

        {onApiKeyChange && (
          <Field
            label="API key"
            htmlFor="studio-api-key"
            aside={
              keySet ? (
                <span className="text-small text-ok">Saved for this tab</span>
              ) : authRequired ? (
                <span className="text-small text-warn">Required by the server</span>
              ) : (
                <span className="text-small text-ink-3">Optional</span>
              )
            }
            hint="Sent as X-API-Key. Kept in sessionStorage, so it's cleared when the tab closes."
          >
            <form
              className="flex gap-1.5"
              onSubmit={(e) => {
                e.preventDefault();
                onApiKeyChange(apiKeyInput);
              }}
            >
              <input
                id="studio-api-key"
                type="password"
                value={apiKeyInput}
                autoComplete="off"
                placeholder={authRequired ? "Paste the server's API_AUTH_KEY" : "Not needed for this server"}
                onChange={(e) => setApiKeyInput(e.target.value)}
                className={cn(INPUT_CLASS, "h-8 min-w-0 flex-1 px-2.5 font-mono text-small")}
              />
              <Button type="submit" size="sm">
                Save
              </Button>
            </form>
          </Field>
        )}

        {health ? (
          <>
            <div>
              <div className="mb-1 text-small font-medium text-ink">Services</div>
              <ServiceRow
                label="Vector store"
                tone={health.qdrant?.status === "ok" ? "ok" : "bad"}
                value={
                  health.qdrant?.status === "ok"
                    ? `${formatInt(health.qdrant?.points_count ?? 0)} passages`
                    : health.qdrant?.status === "embedding_mismatch"
                      ? "Needs re-indexing"
                      : "Unavailable"
                }
              />
              <ServiceRow
                label="Web crawler"
                tone={fcStatus === "ok" ? "ok" : fcStatus === "unreachable" ? "warn" : "neutral"}
                value={fcStatus === "ok" ? "Ready" : fcStatus === "unreachable" ? "Unreachable" : "Off"}
              />
              <ServiceRow
                label="Tracing (Langfuse)"
                tone={lfMode === "cloud" || lfMode === "self_hosted" ? "ok" : lfMode === "misconfigured" ? "warn" : "neutral"}
                value={
                  lfMode === "cloud"
                    ? "Cloud"
                    : lfMode === "self_hosted"
                      ? "Self-hosted"
                      : lfMode === "misconfigured"
                        ? "Missing keys"
                        : "Off"
                }
              />
              <ServiceRow
                label="Guardrails"
                tone={health.security?.guardrails_enabled ? "ok" : "neutral"}
                value={
                  health.security?.guardrails_enabled
                    ? `On, PII ${health.security.pii_mode === "block" ? "blocked" : health.security.pii_mode === "off" ? "allowed" : "redacted"}`
                    : "Off"
                }
              />
              <ServiceRow
                label="Access"
                tone={authRequired && !keySet ? "warn" : "neutral"}
                value={[
                  authRequired ? "API key" : "Open",
                  health.security?.rate_limit_enabled
                    ? `${health.security.rate_limit_requests_per_minute ?? 60}/min`
                    : null,
                ]
                  .filter(Boolean)
                  .join(", ")}
              />
            </div>

            <div>
              <div className="mb-1.5 text-small font-medium text-ink">Models</div>
              <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1 text-small">
                <dt className="text-ink-3">Fast</dt>
                <dd className="truncate text-right font-mono text-[11.5px] text-ink" title={health.fast_model}>
                  {health.fast_model}
                </dd>
                <dt className="text-ink-3">Generation</dt>
                <dd className="truncate text-right font-mono text-[11.5px] text-ink" title={health.generation_model}>
                  {health.generation_model}
                </dd>
                <dt className="text-ink-3">Reranker</dt>
                <dd className="truncate text-right font-mono text-[11.5px] text-ink" title={health.reranker_model}>
                  {health.reranker_model.split("/").pop()}
                </dd>
                <dt className="text-ink-3">Web search</dt>
                <dd className="truncate text-right text-ink">
                  {health.web_fallback_providers.join(", ") || "None"}
                </dd>
                <dt className="text-ink-3">New uploads</dt>
                <dd className="truncate text-right text-ink">
                  {parentChild?.enabled
                    ? `${parentChild.child_max_tokens} → ${formatInt(parentChild.parent_max_tokens ?? 1000)} tokens`
                    : "Flat passages"}
                </dd>
              </dl>
            </div>
          </>
        ) : (
          <Callout tone="bad" title="Can't reach the API">
            {healthError || "No response."} Start it with{" "}
            <code className="font-mono text-[11.5px] text-ink">make api</code> and check the endpoint
            above.
          </Callout>
        )}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Sidebar                                                                    */
/* -------------------------------------------------------------------------- */

export const Sidebar: React.FC<SidebarProps> = ({
  activeView,
  onChangeView,
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  apiUrl,
  onApiUrlChange,
  apiKey = "",
  onApiKeyChange,
  health,
  healthError,
  healthLoading,
  onRefreshHealth,
  documentCount,
  onCollapse,
  variant = "docked",
}) => {
  const [connectionOpen, setConnectionOpen] = useState(false);
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
  const footerRef = useRef<HTMLButtonElement>(null);

  const authRequiredMissingKey = Boolean(health?.security?.api_key_auth_enabled) && !apiKey.trim();
  const statusTone: Tone = !health
    ? "bad"
    : authRequiredMissingKey || health.status !== "ok"
      ? "warn"
      : "ok";
  const statusLabel = !health
    ? healthLoading
      ? "Connecting…"
      : "Offline"
    : authRequiredMissingKey
      ? "API key required"
      : health.status === "ok"
        ? "Connected"
        : "Degraded";

  return (
    <aside
      aria-label="Studio navigation"
      className="flex h-full w-[272px] shrink-0 flex-col border-r border-line bg-desk"
    >
      <div className="flex h-12 shrink-0 items-center justify-between gap-2 pr-2 pl-3.5">
        <div className="flex min-w-0 items-center gap-2.5">
          <LogoMark />
          <span className="truncate text-ui font-semibold text-ink">
            AdaptiveRAG <span className="font-normal text-ink-3">Studio</span>
          </span>
        </div>
        <IconButton
          icon={variant === "drawer" ? X : PanelLeftClose}
          label={variant === "drawer" ? "Close menu" : "Collapse sidebar"}
          onClick={onCollapse}
        />
      </div>

      <div className="px-3 pt-1.5 pb-3">
        <Button
          icon={Plus}
          className="w-full justify-start"
          onClick={() => {
            onNewSession();
            onChangeView("workbench");
          }}
        >
          New thread
        </Button>
      </div>

      <nav aria-label="Workspace" className="space-y-px px-2">
        {NAV_ITEMS.map(({ id, label, icon: Icon, hint }) => {
          const active = activeView === id;
          return (
            <button
              key={id}
              type="button"
              data-nav={id}
              title={hint}
              aria-current={active ? "page" : undefined}
              onClick={() => onChangeView(id)}
              className={cn(
                "flex h-8 w-full cursor-pointer items-center gap-2.5 rounded-md px-2.5 text-ui transition-colors",
                active
                  ? "bg-paper font-medium text-ink shadow-[0_0_0_1px_var(--line)]"
                  : "text-ink-2 hover:bg-hover hover:text-ink"
              )}
            >
              <Icon className={cn("h-4 w-4 shrink-0", active ? "text-ink" : "text-ink-3")} aria-hidden />
              <span className="flex-1 text-left">{label}</span>
              {id === "knowledge" && documentCount !== undefined && (
                <span className="text-small tabular-nums text-ink-3">{formatInt(documentCount)}</span>
              )}
            </button>
          );
        })}
      </nav>

      <div className="mt-6 flex min-h-0 flex-1 flex-col">
        <div className="flex items-center justify-between px-4 pb-1.5">
          <h2 className="text-small font-medium text-ink-3">Threads</h2>
          <span className="text-small tabular-nums text-ink-3">{sessions.length}</span>
        </div>
        <ul className="flex-1 space-y-px overflow-y-auto px-2 pb-3">
          {sessions.map((sess) => {
            const isActive = sess.id === activeSessionId && activeView === "workbench";
            const hasCustomStyle = Boolean(resolveEffectiveInstructions(sess.context));
            const scopedCount = sess.context?.scopedDocIds?.length ?? 0;
            const hasCustomContext = hasCustomStyle || scopedCount > 0;
            const isConfirming = pendingDeleteId === sess.id;

            if (isConfirming) {
              return (
                <li
                  key={sess.id}
                  className="flex h-8 items-center justify-between gap-1.5 rounded-md bg-bad-soft pr-1 pl-2.5"
                >
                  <span className="truncate text-small text-ink">Delete this thread?</span>
                  <span className="flex shrink-0 items-center gap-1">
                    <Button
                      variant="danger"
                      size="xs"
                      onClick={() => {
                        onDeleteSession(sess.id);
                        setPendingDeleteId(null);
                      }}
                    >
                      Delete
                    </Button>
                    <IconButton icon={X} size="sm" label="Keep thread" onClick={() => setPendingDeleteId(null)} />
                  </span>
                </li>
              );
            }

            return (
              <li key={sess.id} className="group relative">
                <button
                  type="button"
                  aria-current={isActive ? "true" : undefined}
                  title={
                    hasCustomContext
                      ? `${sess.title} (${[
                          hasCustomStyle ? "custom style" : null,
                          scopedCount > 0 ? `${scopedCount} scoped documents` : null,
                        ]
                          .filter(Boolean)
                          .join(", ")})`
                      : sess.title
                  }
                  onClick={() => {
                    onSelectSession(sess.id);
                    onChangeView("workbench");
                  }}
                  className={cn(
                    "flex h-8 w-full cursor-pointer items-center gap-2 rounded-md pr-2 pl-2.5 text-left text-ui transition-colors",
                    isActive
                      ? "bg-paper text-ink shadow-[0_0_0_1px_var(--line)]"
                      : "text-ink-2 hover:bg-hover hover:text-ink"
                  )}
                >
                  <span className="min-w-0 flex-1 truncate">{sess.title}</span>
                  {hasCustomContext && (
                    <PenLine className="h-3 w-3 shrink-0 text-ink-3" aria-label="Has thread settings" />
                  )}
                  <span className="shrink-0 text-caption tabular-nums text-ink-3 group-focus-within:opacity-0 group-hover:opacity-0">
                    {sess.turns.length === 0 ? "Empty" : formatRelativeTime(sess.updatedAt)}
                  </span>
                </button>
                {sessions.length > 1 && (
                  <IconButton
                    icon={Trash2}
                    size="sm"
                    tone="bad"
                    label={`Delete thread "${sess.title}"`}
                    onClick={(e) => {
                      e.stopPropagation();
                      setPendingDeleteId(sess.id);
                    }}
                    className="absolute top-1 right-1 opacity-0 group-focus-within:opacity-100 group-hover:opacity-100 focus-visible:opacity-100"
                  />
                )}
              </li>
            );
          })}
        </ul>
      </div>

      <div className="shrink-0 border-t border-line p-2">
        <button
          ref={footerRef}
          type="button"
          aria-haspopup="dialog"
          aria-expanded={connectionOpen}
          onClick={() => setConnectionOpen((prev) => !prev)}
          className={cn(
            "flex w-full cursor-pointer items-center gap-2.5 rounded-md px-2.5 py-1.5 text-left transition-colors",
            connectionOpen ? "bg-hover" : "hover:bg-hover"
          )}
        >
          <StatusDot tone={statusTone} pulse={healthLoading} />
          <span className="min-w-0 flex-1">
            <span className="block text-ui font-medium text-ink">{statusLabel}</span>
            <span className="block truncate text-caption text-ink-3">{hostLabel(apiUrl)}</span>
          </span>
          <Settings2 className="h-4 w-4 shrink-0 text-ink-3" aria-hidden />
        </button>
        <Popover
          open={connectionOpen}
          onClose={() => setConnectionOpen(false)}
          anchorRef={footerRef}
          placement="top-start"
          width={340}
          ariaLabel="Connection settings"
          className="p-0"
        >
          <ConnectionPanel
            apiUrl={apiUrl}
            onApiUrlChange={onApiUrlChange}
            apiKey={apiKey}
            onApiKeyChange={onApiKeyChange}
            health={health}
            healthError={healthError}
            healthLoading={healthLoading}
            onRefreshHealth={onRefreshHealth}
          />
        </Popover>
      </div>
    </aside>
  );
};
