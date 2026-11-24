"""Lifespan-managed dependency container and FastAPI dependency providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

from fastapi import Request

from adaptiverag.config import Settings, StrategiesConfig, get_settings, load_strategies
from adaptiverag.graph.workflow import AdaptiveRAGWorkflow
from adaptiverag.ingestion.pdf_parser import VisionOcrFn
from adaptiverag.ingestion.web_loader import FirecrawlClient
from adaptiverag.observability.prompts import PromptManager
from adaptiverag.observability.tracing import LangfuseObservability
from adaptiverag.retrieval.fallback import FallbackRetriever, WebFallbackRetriever
from adaptiverag.retrieval.qdrant_store import QdrantHybridStore


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
    owns_store: bool = True
    owns_firecrawl: bool = True
    _workflow_lock: Lock = field(default_factory=Lock, repr=False)

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

    def ensure_workflow(self) -> AdaptiveRAGWorkflow:
        """Return the active AdaptiveRAGWorkflow, building it lazily on first query."""
        if self.workflow is not None:
            return self.workflow
        with self._workflow_lock:
            if self.workflow is None:
                self.workflow = AdaptiveRAGWorkflow.from_settings(
                    self.settings,
                    strategies=self.strategies,
                    store=self.store,
                    fallback_retriever=self.fallback_retriever,
                    observability=self.get_observability(),
                    prompt_manager=self.get_prompt_manager(),
                )
            return self.workflow

    def check_llm_configured(self) -> tuple[bool, str | None]:
        """Check whether LLM credentials/runtime are configured without executing API calls."""
        if self.workflow is not None:
            return True, None
        provider = self.settings.llm_provider
        if provider == "ollama":
            return True, None
        key_map = {
            "openrouter": (self.settings.openrouter_api_key, "OPENROUTER_API_KEY"),
            "openai": (self.settings.openai_api_key, "OPENAI_API_KEY"),
            "gemini": (self.settings.gemini_api_key, "GEMINI_API_KEY"),
            "anthropic": (self.settings.anthropic_api_key, "ANTHROPIC_API_KEY"),
        }
        secret, env_var = key_map[provider]
        if secret is None or not secret.get_secret_value().strip():
            return False, f"Missing required API key: set {env_var} in your environment or .env"
        return True, None

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
