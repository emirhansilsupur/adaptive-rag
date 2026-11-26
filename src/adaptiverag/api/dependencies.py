"""Lifespan-managed dependency container and FastAPI dependency providers."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from threading import Lock

from fastapi import Request

from adaptiverag.config import Settings, StrategiesConfig, get_settings, load_strategies
from adaptiverag.graph.workflow import AdaptiveRAGWorkflow
from adaptiverag.ingestion.pdf_parser import VisionOcrFn
from adaptiverag.ingestion.web_loader import FirecrawlClient
from adaptiverag.llm.factory import (
    LLMProvider,
    ModelBundle,
    alist_local_models,
    check_provider_configured,
    create_model_bundle,
    resolve_effective_settings,
)
from adaptiverag.observability.prompts import PromptManager
from adaptiverag.observability.tracing import LangfuseObservability
from adaptiverag.retrieval.fallback import FallbackRetriever, WebFallbackRetriever
from adaptiverag.retrieval.qdrant_store import QdrantHybridStore

LocalModelsProbeFn = Callable[[str], Awaitable[tuple[bool, list[str], str | None]]]
ModelBundleFactoryFn = Callable[[Settings], ModelBundle]

_MAX_OVERRIDE_BUNDLES = 16


@dataclass
class AppContainer:
    """Shared concurrency-safe runtime container bound to FastAPI app.state."""

    settings: Settings
    strategies: StrategiesConfig
    store: QdrantHybridStore
    fallback_retriever: FallbackRetriever
    workflow: AdaptiveRAGWorkflow | None = None
    firecrawl_client: FirecrawlClient | None = None
    observability: LangfuseObservability | None = None
    prompt_manager: PromptManager | None = None
    vision_ocr_fn: VisionOcrFn | None = None
    local_models_probe_fn: LocalModelsProbeFn | None = None
    model_bundle_factory: ModelBundleFactoryFn | None = None
    owns_store: bool = True
    owns_firecrawl: bool = True
    _workflow_lock: Lock = field(default_factory=Lock, repr=False)
    _bundle_cache: OrderedDict[tuple[str, str, str, float, str], ModelBundle] = field(
        default_factory=OrderedDict,
        repr=False,
    )

    def get_observability(self) -> LangfuseObservability:
        """Return the shared LangfuseObservability adapter (lazily initialized if absent)."""
        if self.observability is None:
            if self.workflow is not None and self.workflow.runtime.observability is not None:
                self.observability = self.workflow.runtime.observability
            else:
                self.observability = LangfuseObservability.from_settings(self.settings)
        return self.observability

    def get_prompt_manager(self) -> PromptManager:
        """Return the shared PromptManager bound to observability (lazily initialized if absent)."""
        if self.prompt_manager is None:
            if self.workflow is not None and self.workflow.runtime.prompt_manager is not None:
                self.prompt_manager = self.workflow.runtime.prompt_manager
            else:
                self.prompt_manager = PromptManager.from_settings(
                    self.settings,
                    observability=self.get_observability(),
                )
        return self.prompt_manager

    def ensure_workflow(
        self,
        *,
        models_override: ModelBundle | None = None,
    ) -> AdaptiveRAGWorkflow:
        """Return the active AdaptiveRAGWorkflow, building it lazily on first query."""
        if self.workflow is not None:
            return self.workflow
        with self._workflow_lock:
            if self.workflow is None:
                self.workflow = AdaptiveRAGWorkflow.from_settings(
                    self.settings,
                    strategies=self.strategies,
                    models=models_override,
                    store=self.store,
                    fallback_retriever=self.fallback_retriever,
                    observability=self.get_observability(),
                    prompt_manager=self.get_prompt_manager(),
                )
            return self.workflow

    def resolve_model_bundle(
        self,
        *,
        provider: LLMProvider | None = None,
        fast_model: str | None = None,
        generation_model: str | None = None,
        temperature: float | None = None,
        api_key_override: str | None = None,
    ) -> tuple[ModelBundle | None, float | None]:
        """Resolve a cached, concurrency-safe per-request ModelBundle when overrides are given."""
        cleaned_key = api_key_override.strip() if api_key_override else ""
        if (
            provider is None
            and fast_model is None
            and generation_model is None
            and temperature is None
            and not cleaned_key
        ):
            return None, None

        if cleaned_key and not self.settings.allow_client_provider_keys:
            raise PermissionError(
                "Client-supplied provider API keys are disabled by server policy "
                "(set ALLOW_CLIENT_PROVIDER_KEYS=true to enable BYOK)."
            )

        effective = resolve_effective_settings(
            self.settings,
            provider=provider,
            fast_model=fast_model,
            generation_model=generation_model,
            temperature=temperature,
            api_key_override=cleaned_key or None,
        )

        # If effective selection matches server default (and no client key override), skip rebuild.
        if (
            self.workflow is not None
            and not cleaned_key
            and effective.llm_provider == self.settings.llm_provider
            and effective.llm_fast_model == self.settings.llm_fast_model
            and effective.llm_generation_model == self.settings.llm_generation_model
            and effective.llm_temperature == self.settings.llm_temperature
        ):
            return None, effective.llm_temperature

        configured, reason = check_provider_configured(effective, effective.llm_provider)
        if not configured:
            raise ValueError(reason or f"Provider '{effective.llm_provider}' is not configured.")

        # Fingerprint client keys with SHA-256 so raw secrets never appear in cache keys or logs.
        key_fingerprint = (
            f"client:{hashlib.sha256(cleaned_key.encode('utf-8')).hexdigest()[:16]}"
            if cleaned_key
            else "server"
        )
        cache_key = (
            effective.llm_provider,
            effective.llm_fast_model,
            effective.llm_generation_model,
            round(float(effective.llm_temperature), 4),
            key_fingerprint,
        )
        with self._workflow_lock:
            cached = self._bundle_cache.get(cache_key)
            if cached is not None:
                self._bundle_cache.move_to_end(cache_key)
                return cached, effective.llm_temperature

            factory = self.model_bundle_factory or create_model_bundle
            bundle = factory(effective)
            self._bundle_cache[cache_key] = bundle
            if len(self._bundle_cache) > _MAX_OVERRIDE_BUNDLES:
                self._bundle_cache.popitem(last=False)
            return bundle, effective.llm_temperature

    async def probe_local_models(self) -> tuple[bool, list[str], str | None]:
        """List live models on the server-configured local OpenAI-compatible runtime."""
        if self.local_models_probe_fn is not None:
            return await self.local_models_probe_fn(self.settings.ollama_base_url)
        return await alist_local_models(self.settings.ollama_base_url)

    def check_llm_configured(self) -> tuple[bool, str | None]:
        """Check whether LLM credentials/runtime are configured without executing API calls."""
        if self.workflow is not None:
            return True, None
        return check_provider_configured(self.settings, self.settings.llm_provider)

    async def aclose(self) -> None:
        """Release underlying network clients on application shutdown."""
        if self.observability is not None:
            self.observability.flush()
        if self.firecrawl_client is not None and self.owns_firecrawl:
            await self.firecrawl_client.aclose()
        if self.owns_store:
            await self.store.close()


def create_app_container(
    settings: Settings | None = None,
    *,
    in_memory_store: bool = False,
) -> AppContainer:
    """Construct the default runtime container from Settings & configs/strategies.yaml."""
    resolved_settings = settings or get_settings()
    strategies = load_strategies(resolved_settings.strategies_config_path)
    store = QdrantHybridStore.from_settings(resolved_settings, in_memory=in_memory_store)
    fallback_retriever = WebFallbackRetriever.from_settings(resolved_settings)
    firecrawl_client = FirecrawlClient.from_settings(resolved_settings)
    observability = LangfuseObservability.from_settings(resolved_settings)
    prompt_manager = PromptManager.from_settings(resolved_settings, observability=observability)

    container = AppContainer(
        settings=resolved_settings,
        strategies=strategies,
        store=store,
        fallback_retriever=fallback_retriever,
        workflow=None,
        firecrawl_client=firecrawl_client,
        observability=observability,
        prompt_manager=prompt_manager,
        owns_store=True,
        owns_firecrawl=True,
    )

    # Eagerly build workflow when credentials are configured so startup surfaces wiring errors early.
    llm_ok, _ = container.check_llm_configured()
    if llm_ok:
        container.ensure_workflow()

    return container


def get_container(request: Request) -> AppContainer:
    """FastAPI dependency returning the active AppContainer from app.state."""
    container: AppContainer | None = getattr(request.app.state, "container", None)
    if container is None:
        raise RuntimeError("AdaptiveRAG AppContainer is not initialized on app.state.")
    return container
