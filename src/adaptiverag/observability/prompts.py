"""Versioned Langfuse v5 Prompt Management with zero-downtime local fallback & CLI sync.

Manages all 8 LLM system prompts across AdaptiveRAG:
1. ``adaptiverag.router`` — Query Router & Planner (classify_and_plan)
2. ``adaptiverag.decompose`` — Analytical Multi-Query Decomposer (decompose_queries)
3. ``adaptiverag.corrective_rewrite`` — CRAG Corrective Query Rewriter (corrective_rewrite)
4. ``adaptiverag.crag_grader`` — Ambiguous-Band CRAG Sufficiency Grader (score_confidence)
5. ``adaptiverag.conversational`` — Zero-Retrieval Conversational Fast Path (conversational_respond)
6. ``adaptiverag.grounded_synthesis`` — Strictly Grounded Citation Synthesis (generate_answer)
7. ``adaptiverag.strict_regeneration`` — Strict Corrective Citation Repair (regenerate_strict)
8. ``adaptiverag.citation_entailment`` — Tier-2 Semantic Citation Entailment Auditor (validate_citations)

Design guarantees:
- **Zero-downtime fail-open:** When Langfuse or prompt management is disabled, misconfigured,
  unreachable, or a prompt has not yet been seeded, ``PromptManager.resolve()`` transparently
  renders the local canonical template (``source="local_fallback"``).
- **Mustache ``{{variable}}`` compilation:** Uses double-curly-brace syntax matching Langfuse
  Text Prompts, so JSON schemas and braces in prompts never clash with Python ``str.format``.
- **Native trace-to-prompt version linking:** Returns the live ``TextPromptClient`` object so
  ``invoke_text`` / ``invoke_structured`` bind it via LangChain ``langfuse_prompt`` metadata,
  enabling per-version quality & cost analytics in Langfuse UI.
- **Idempotent CLI seeding:** ``python -m adaptiverag.observability.prompts --seed`` skips
  unchanged prompts so repeated runs never spam duplicate versions.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import structlog

from adaptiverag.config import Settings
from adaptiverag.observability.tracing import LangfuseObservability

logger = structlog.get_logger(__name__)

PromptSource = Literal["langfuse", "local_fallback"]
ModelTier = Literal["fast", "generation"]

_MUSTACHE_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")

# Canonical prompt names
PROMPT_ROUTER = "adaptiverag.router"
PROMPT_DECOMPOSE = "adaptiverag.decompose"
PROMPT_CORRECTIVE_REWRITE = "adaptiverag.corrective_rewrite"
PROMPT_CRAG_GRADER = "adaptiverag.crag_grader"
PROMPT_CONVERSATIONAL = "adaptiverag.conversational"
PROMPT_GROUNDED_SYNTHESIS = "adaptiverag.grounded_synthesis"
PROMPT_STRICT_REGENERATION = "adaptiverag.strict_regeneration"
PROMPT_CITATION_ENTAILMENT = "adaptiverag.citation_entailment"


@dataclass(frozen=True)
class PromptDefinition:
    """Immutable canonical definition of a managed AdaptiveRAG prompt template."""

    name: str
    description: str
    node: str
    model_tier: ModelTier
    template: str
    variables: tuple[str, ...] = ()
    tags: tuple[str, ...] = ("adaptiverag",)


@dataclass(frozen=True)
class ResolvedPrompt:
    """Compiled prompt ready for LLM invocation, carrying version & Langfuse link metadata."""

    name: str
    text: str
    source: PromptSource
    version: int | None = None
    label: str | None = None
    langfuse_prompt: Any | None = field(default=None, repr=False, compare=False)

    def telemetry_metadata(self) -> dict[str, Any]:
        """Return compact JSON-serializable metadata for trace events & Langfuse spans."""
        payload: dict[str, Any] = {
            "name": self.name,
            "source": self.source,
        }
        if self.version is not None:
            payload["version"] = self.version
        if self.label is not None:
            payload["label"] = self.label
        return payload


@dataclass(frozen=True)
class PromptSyncItem:
    """Per-prompt outcome from ``sync_prompts_to_langfuse``."""

    name: str
    action: Literal["created", "unchanged", "would_create", "error"]
    version: int | None = None
    labels: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class PromptSyncResult:
    """Summary returned after seeding/syncing local prompt definitions to Langfuse."""

    total: int
    created: int
    unchanged: int
    errors: int
    dry_run: bool
    label: str
    items: tuple[PromptSyncItem, ...]


DEFAULT_PROMPTS: dict[str, PromptDefinition] = {
    PROMPT_ROUTER: PromptDefinition(
        name=PROMPT_ROUTER,
        description="Multilingual (TR+EN) query classifier, coreference resolver, and freshness detector.",
        node="classify_and_plan",
        model_tier="fast",
        variables=(),
        tags=("adaptiverag", "routing", "structured"),
        template="""You are the Query Router for AdaptiveRAG, a multilingual (Turkish & English) technical knowledge assistant.

Classify the user query into exactly one `query_type`:
- "conversational": Greetings, thanks, small talk, or pure meta-requests about prior assistant turns (set `needs_retrieval=false`).
- "factual": Direct definition, specific API/parameter/default lookup, exact formula, or single-concept question (set `needs_retrieval=true`).
- "analytical": Comparisons, trade-offs, architectural design choices, multi-step troubleshooting, or questions combining multiple systems/concepts (set `needs_retrieval=true`).

Also produce:
- `language`: "tr" for Turkish, "en" for English, or "other".
- `standalone_query`: Rewrite the user query so it is completely self-contained by resolving any pronouns ("it", "this", "onun", "bunlar") using chat history. Preserve the query's language and technical terms.
- `requires_fresh_web_info`: Set `true` ONLY when the query explicitly asks for current/latest releases, live benchmarks, recent news, or real-time prices unlikely to exist in static documentation. Otherwise `false`.
- `reasoning`: One concise sentence justifying your routing choice.""",
    ),
    PROMPT_DECOMPOSE: PromptDefinition(
        name=PROMPT_DECOMPOSE,
        description="Analytical multi-query decomposer for hybrid BM25 + dense retrieval.",
        node="decompose_queries",
        model_tier="fast",
        variables=("max_sub_queries", "language"),
        tags=("adaptiverag", "decomposition", "structured"),
        template="""You are a Query Decomposition Specialist for a hybrid BM25 + Vector search engine.

Decompose the complex analytical question into 2 to {{max_sub_queries}} distinct, focused sub-queries that cover orthogonal aspects (e.g., each system/concept being compared, specific mechanisms, latency/memory/operational trade-offs).

Rules:
- Keep every sub-query concise and keyword-rich so both BM25 lexical search and dense embeddings perform well.
- Avoid redundant duplicates.
- Preserve exact technical terms, library names, and acronyms.
- Write sub-queries primarily in the language of the input question (language={{language}}), keeping standard English technical identifiers intact.""",
    ),
    PROMPT_CORRECTIVE_REWRITE: PromptDefinition(
        name=PROMPT_CORRECTIVE_REWRITE,
        description="CRAG corrective query rewriter for cross-lingual & synonym recovery.",
        node="corrective_rewrite",
        model_tier="fast",
        variables=("cross_lingual_rule",),
        tags=("adaptiverag", "crag", "rewrite", "structured"),
        template="""You are a Corrective Retrieval Rewriter (CRAG query optimizer) for a multilingual (TR/EN) hybrid BM25 + dense vector search engine.

The initial retrieval produced low or partial confidence. Reformulate 2 to 3 improved search queries that use better technical synonyms, canonical terminology, or clearer aspect separation.

Rules:
- Do NOT simply repeat the prior queries verbatim.
- Target any missing aspects identified in the confidence diagnosis.
{{cross_lingual_rule}}
- Keep every query concise and retrieval-oriented.""",
    ),
    PROMPT_CRAG_GRADER: PromptDefinition(
        name=PROMPT_CRAG_GRADER,
        description="Ambiguous-band CRAG retrieval sufficiency evaluator.",
        node="score_confidence",
        model_tier="fast",
        variables=(),
        tags=("adaptiverag", "crag", "verification", "structured"),
        template="""You are a strict Retrieval Sufficiency Grader (CRAG evaluator) for a multilingual (Turkish & English) technical RAG system.

Given a user question and retrieved evidence passages, judge whether the passages contain sufficient factual evidence to answer the question accurately WITHOUT guessing or relying on outside memory.

Grading rubric:
- "sufficient": Passages directly and completely cover the core question (and all major sub-parts if analytical/comparative).
- "partial": Passages cover only part of the question, are somewhat tangential, or miss key details/comparisons needed for a complete answer.
- "insufficient": Passages are off-topic, outdated relative to what is asked, or do not contain the required facts.

List any concrete missing aspects in `missing_aspects` (empty list when sufficient) and give a concise 1-sentence justification in `reasoning`.""",
    ),
    PROMPT_CONVERSATIONAL: PromptDefinition(
        name=PROMPT_CONVERSATIONAL,
        description="Zero-retrieval conversational fast-path system prompt.",
        node="conversational_respond",
        model_tier="fast",
        variables=("language_instruction",),
        tags=("adaptiverag", "conversational"),
        template=(
            "You are AdaptiveRAG, a helpful and concise AI engineering assistant. "
            "Handle conversational messages, greetings, or meta-questions about prior turns naturally. "
            "Do NOT fabricate document citations like [C1] or [W1] because no retrieval was performed. "
            "{{language_instruction}}"
        ),
    ),
    PROMPT_GROUNDED_SYNTHESIS: PromptDefinition(
        name=PROMPT_GROUNDED_SYNTHESIS,
        description="Strictly grounded technical synthesis prompt requiring inline [C#]/[W#] citations.",
        node="generate_answer",
        model_tier="generation",
        variables=("language_instruction", "style_hint"),
        tags=("adaptiverag", "generation", "citations"),
        template="""You are AdaptiveRAG, a strictly grounded technical AI assistant.

Grounding & Citation Rules (MANDATORY):
1. Answer ONLY using facts explicitly stated in the provided Evidence Passages below. Do NOT rely on unverified parametric memory.
2. Attach exact inline citations like [C1], [W1], or [C1, C2] to EVERY factual sentence or bullet point.
3. NEVER invent, guess, or alter citation IDs. Only use IDs that appear in the Evidence Passages.
4. If the evidence only partially covers the question, answer the supported part with citations and explicitly note what is not covered in the provided sources.
5. {{language_instruction}}
6. {{style_hint}}""",
    ),
    PROMPT_STRICT_REGENERATION: PromptDefinition(
        name=PROMPT_STRICT_REGENERATION,
        description="Strict corrective regeneration prompt repairing failed citation audits.",
        node="regenerate_strict",
        model_tier="generation",
        variables=("valid_labels", "language_instruction"),
        tags=("adaptiverag", "generation", "citations", "repair"),
        template="""You are revising a draft answer that FAILED strict citation verification.

Mandatory Revision Instructions:
1. Rewrite the answer so EVERY factual statement is 100% entailed by the Evidence Passages and cites valid passage IDs inline.
2. Allowed citation IDs ONLY: {{valid_labels}}. Never emit any other ID.
3. REMOVE or explicitly qualify any claim that was flagged as unsupported or contradicted.
4. Do NOT add new unsupported facts. Prefer a shorter, 100% verified answer over a longer speculative one.
5. {{language_instruction}}""",
    ),
    PROMPT_CITATION_ENTAILMENT: PromptDefinition(
        name=PROMPT_CITATION_ENTAILMENT,
        description="Tier-2 claim-level semantic citation entailment auditor.",
        node="validate_citations",
        model_tier="fast",
        variables=(),
        tags=("adaptiverag", "citations", "verification", "structured"),
        template="""You are a strict Citation Entailment Auditor for a multilingual (Turkish & English) technical RAG engine.

Given a draft answer and the retrieved evidence passages labeled [C1], [C2], ..., [W1], ..., extract the distinct atomic factual claims made in the answer (up to 10 key claims) and verify EACH claim strictly against the evidence passages it cites.

Verdict definitions:
- "supported": The claim is directly and accurately entailed by the cited passage(s).
- "partially_supported": Only part of the claim is supported by the cited passage(s), or minor extrapolations/numbers are not explicitly stated.
- "unsupported": The claim is NOT stated in the cited passage(s), cites a non-existent ID, or makes a factual assertion with NO citation at all.
- "contradicted": The claim directly contradicts the evidence passages.

Rules:
- Be rigorous with numbers, version strings, formulas, thresholds, and technical comparisons.
- Do NOT judge pure greetings, section headings, or honest abstention/uncertainty statements as factual claims.
- Record the exact citation IDs attached to each claim in `cited_ids` (uppercase, e.g. ["C1", "W1"], or [] if uncited).
- Keep `explanation` concise (1 short sentence).""",
    ),
}


def compile_mustache_template(
    template: str,
    variables: Mapping[str, Any] | None = None,
) -> str:
    """Render ``{{variable}}`` placeholders safely without touching single-brace JSON."""
    if not variables:
        return template

    normalized = {key: str(val) for key, val in variables.items()}

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in normalized:
            return normalized[key]
        return match.group(0)

    return _MUSTACHE_VAR_RE.sub(_replace, template)


class PromptManager:
    """Fail-open Langfuse v5 Prompt Management adapter with local canonical fallback."""

    def __init__(
        self,
        *,
        observability: LangfuseObservability | None = None,
        enabled: bool = False,
        label: str = "production",
        cache_ttl_seconds: int = 300,
        fetch_timeout_seconds: float = 3.0,
        registry: Mapping[str, PromptDefinition] | None = None,
    ) -> None:
        self._observability = observability
        self._enabled = bool(enabled)
        self._label = label.strip() or "production"
        self._cache_ttl_seconds = max(0, int(cache_ttl_seconds))
        self._fetch_timeout_seconds = max(1, round(fetch_timeout_seconds))
        self._registry: dict[str, PromptDefinition] = dict(registry or DEFAULT_PROMPTS)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        observability: LangfuseObservability | None = None,
    ) -> PromptManager:
        """Construct a PromptManager bound to Settings and a shared LangfuseObservability."""
        resolved_obs = observability or LangfuseObservability.from_settings(settings)
        return cls(
            observability=resolved_obs,
            enabled=settings.langfuse_prompt_management_enabled,
            label=settings.langfuse_prompt_label,
            cache_ttl_seconds=settings.langfuse_prompt_cache_ttl_seconds,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def active(self) -> bool:
        """True when prompt management is enabled AND Langfuse is configured."""
        return bool(
            self._enabled and self._observability is not None and self._observability.configured
        )

    @property
    def label(self) -> str:
        return self._label

    @property
    def cache_ttl_seconds(self) -> int:
        return self._cache_ttl_seconds

    @property
    def prompt_names(self) -> list[str]:
        return list(self._registry.keys())

    def get_definition(self, name: str) -> PromptDefinition:
        """Return the canonical local PromptDefinition for ``name``."""
        return self._registry[name]

    def health_summary(self) -> dict[str, Any]:
        """Return non-secret prompt management readiness metadata for GET /v1/health."""
        return {
            "enabled": self._enabled,
            "active": self.active,
            "mode": "langfuse" if self.active else "local_fallback",
            "label": self._label,
            "cache_ttl_seconds": self._cache_ttl_seconds,
            "managed_prompts": len(self._registry),
        }

    def resolve(
        self,
        name: str,
        *,
        variables: Mapping[str, Any] | None = None,
        label: str | None = None,
        version: int | None = None,
    ) -> ResolvedPrompt:
        """Resolve and compile a managed prompt, falling back locally on any error."""
        defn = self._registry[name]
        vars_dict = {k: str(v) for k, v in (variables or {}).items()}
        target_label = label.strip() if label and label.strip() else self._label

        if self.active and self._observability is not None:
            client = self._observability.get_client()
            if client is not None:
                try:
                    get_kwargs: dict[str, Any] = {
                        "type": "text",
                        "cache_ttl_seconds": self._cache_ttl_seconds,
                        "fallback": defn.template,
                        "fetch_timeout_seconds": self._fetch_timeout_seconds,
                    }
                    if version is not None:
                        get_kwargs["version"] = version
                    else:
                        get_kwargs["label"] = target_label

                    prompt_client = client.get_prompt(name, **get_kwargs)
                    is_fallback = bool(getattr(prompt_client, "is_fallback", False))

                    compile_fn = getattr(prompt_client, "compile", None)
                    if callable(compile_fn):
                        compiled_raw = compile_fn(**vars_dict)
                        compiled = str(compiled_raw)
                    else:
                        raw_tpl = str(getattr(prompt_client, "prompt", defn.template))
                        compiled = compile_mustache_template(raw_tpl, vars_dict)

                    if is_fallback:
                        return ResolvedPrompt(
                            name=name,
                            text=compiled,
                            source="local_fallback",
                            version=None,
                            label=None,
                            langfuse_prompt=None,
                        )

                    raw_version = getattr(prompt_client, "version", None)
                    resolved_version = raw_version if isinstance(raw_version, int) else None
                    return ResolvedPrompt(
                        name=name,
                        text=compiled,
                        source="langfuse",
                        version=resolved_version,
                        label=None if version is not None else target_label,
                        langfuse_prompt=prompt_client,
                    )
                except Exception as exc:
                    logger.warning(
                        "langfuse_prompt_fetch_failed_using_fallback",
                        prompt_name=name,
                        label=target_label,
                        error=str(exc),
                    )

        compiled_local = compile_mustache_template(defn.template, vars_dict)
        return ResolvedPrompt(
            name=name,
            text=compiled_local,
            source="local_fallback",
            version=None,
            label=None,
            langfuse_prompt=None,
        )

    def clear_cache(self) -> None:
        """Clear the underlying Langfuse SDK in-memory prompt cache safely."""
        if not self.active or self._observability is None:
            return
        client = self._observability.get_client()
        if client is None:
            return
        clear_fn = getattr(client, "clear_prompt_cache", None)
        if callable(clear_fn):
            try:
                clear_fn()
            except Exception as exc:
                logger.debug("langfuse_prompt_cache_clear_failed", error=str(exc))


def sync_prompts_to_langfuse(
    client: Any,
    *,
    label: str = "production",
    extra_labels: list[str] | None = None,
    commit_message: str = "Sync canonical AdaptiveRAG prompt templates",
    force: bool = False,
    dry_run: bool = False,
    registry: Mapping[str, PromptDefinition] | None = None,
) -> PromptSyncResult:
    """Idempotently seed or update all managed AdaptiveRAG prompts in Langfuse.

    Compares each local template against the currently active remote prompt under ``label``
    so repeated runs skip unchanged templates instead of creating redundant versions.
    """
    prompts = dict(registry or DEFAULT_PROMPTS)
    primary_label = label.strip() or "production"
    target_labels = list(
        dict.fromkeys(
            [primary_label, *(lbl.strip() for lbl in (extra_labels or []) if lbl.strip())]
        )
    )

    items: list[PromptSyncItem] = []
    created = 0
    unchanged = 0
    errors = 0

    for name, defn in prompts.items():
        remote_template: str | None = None
        remote_version: int | None = None

        if not force:
            try:
                existing = client.get_prompt(
                    name,
                    label=primary_label,
                    type="text",
                    cache_ttl_seconds=0,
                )
                if not bool(getattr(existing, "is_fallback", False)):
                    raw_remote = getattr(existing, "prompt", None)
                    if isinstance(raw_remote, str):
                        remote_template = raw_remote
                    raw_ver = getattr(existing, "version", None)
                    if isinstance(raw_ver, int):
                        remote_version = raw_ver
            except Exception:
                # Prompt does not exist remotely yet (404) — will be created below.
                remote_template = None

        if (
            not force
            and remote_template is not None
            and remote_template.strip() == defn.template.strip()
        ):
            unchanged += 1
            items.append(
                PromptSyncItem(
                    name=name,
                    action="unchanged",
                    version=remote_version,
                    labels=tuple(target_labels),
                )
            )
            continue

        if dry_run:
            items.append(
                PromptSyncItem(
                    name=name,
                    action="would_create",
                    version=None,
                    labels=tuple(target_labels),
                )
            )
            continue

        try:
            created_prompt = client.create_prompt(
                name=name,
                prompt=defn.template,
                type="text",
                labels=target_labels,
                tags=list(defn.tags),
                config={
                    "node": defn.node,
                    "model_tier": defn.model_tier,
                    "variables": list(defn.variables),
                    "description": defn.description,
                },
                commit_message=commit_message,
            )
            raw_ver = getattr(created_prompt, "version", None)
            new_version = raw_ver if isinstance(raw_ver, int) else None
            created += 1
            items.append(
                PromptSyncItem(
                    name=name,
                    action="created",
                    version=new_version,
                    labels=tuple(target_labels),
                )
            )
        except Exception as exc:
            errors += 1
            logger.warning("langfuse_prompt_sync_failed", prompt_name=name, error=str(exc))
            items.append(
                PromptSyncItem(
                    name=name,
                    action="error",
                    version=None,
                    labels=tuple(target_labels),
                    error=str(exc),
                )
            )

    flush_fn = getattr(client, "flush", None)
    if callable(flush_fn):
        with contextlib.suppress(Exception):
            flush_fn()

    return PromptSyncResult(
        total=len(prompts),
        created=created,
        unchanged=unchanged,
        errors=errors,
        dry_run=dry_run,
        label=primary_label,
        items=tuple(items),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adaptiverag.observability.prompts",
        description="Inspect and idempotently seed AdaptiveRAG prompts into Langfuse Prompt Management.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all canonical managed prompts, variables, nodes, and model tiers ($0 offline).",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Idempotently push local canonical prompt templates to Langfuse Prompt Management.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Langfuse prompt label to assign (defaults to LANGFUSE_PROMPT_LABEL / 'production').",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Create a new prompt version even if the remote template matches locally.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview which prompts would be created vs. skipped without writing to Langfuse.",
    )
    parser.add_argument(
        "--commit-message",
        default="Sync canonical AdaptiveRAG prompt templates",
        help="Commit message recorded on newly created Langfuse prompt versions.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit machine-readable JSON output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for listing and seeding AdaptiveRAG Langfuse prompts."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.list or not args.seed:
        rows = [
            {
                "name": defn.name,
                "node": defn.node,
                "model_tier": defn.model_tier,
                "variables": list(defn.variables),
                "description": defn.description,
            }
            for defn in DEFAULT_PROMPTS.values()
        ]
        if args.as_json:
            print(json.dumps(rows, indent=2, ensure_ascii=False))
        else:
            print(f"AdaptiveRAG Managed Prompts ({len(rows)} total):")
            for row in rows:
                vars_text = ", ".join(row["variables"]) if row["variables"] else "(none)"
                print(
                    f"  - {row['name']}  [node={row['node']}, tier={row['model_tier']}, "
                    f"vars={vars_text}]"
                )
        return 0

    settings = Settings()
    label = (args.label or settings.langfuse_prompt_label).strip() or "production"
    obs = LangfuseObservability(
        enabled=True,
        host=settings.langfuse_host,
        public_key=settings.langfuse_public_key,
        secret_key=(
            settings.langfuse_secret_key.get_secret_value()
            if settings.langfuse_secret_key is not None
            else None
        ),
        environment=settings.langfuse_environment,
        release=settings.langfuse_release,
    )
    client = obs.get_client()
    if client is None:
        print(
            "Error: Langfuse credentials are not configured. Set LANGFUSE_PUBLIC_KEY and "
            "LANGFUSE_SECRET_KEY (plus LANGFUSE_HOST for self-hosted) in .env.",
            file=sys.stderr,
        )
        return 1

    result = sync_prompts_to_langfuse(
        client,
        label=label,
        commit_message=args.commit_message,
        force=args.force,
        dry_run=args.dry_run,
    )
    if args.as_json:
        print(
            json.dumps(
                {
                    "total": result.total,
                    "created": result.created,
                    "unchanged": result.unchanged,
                    "errors": result.errors,
                    "dry_run": result.dry_run,
                    "label": result.label,
                    "items": [
                        {
                            "name": item.name,
                            "action": item.action,
                            "version": item.version,
                            "labels": list(item.labels),
                            "error": item.error,
                        }
                        for item in result.items
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        prefix = "[DRY-RUN] " if result.dry_run else ""
        print(
            f"{prefix}Langfuse Prompt Sync ({obs.host}, label={result.label}): "
            f"total={result.total}, created={result.created}, "
            f"unchanged={result.unchanged}, errors={result.errors}"
        )
        for item in result.items:
            ver_suffix = f" (v{item.version})" if item.version is not None else ""
            err_suffix = f" — {item.error}" if item.error else ""
            print(f"  - {item.name}: {item.action}{ver_suffix}{err_suffix}")

    return 1 if result.errors > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
