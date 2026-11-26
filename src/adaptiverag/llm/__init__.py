"""LLM provider factory, caching, cost tracking, and structured output utilities."""

from adaptiverag.llm.cache import CachedResponse, SQLiteLLMCache
from adaptiverag.llm.cost import BudgetExceededError, CostTracker, estimate_cost_usd
from adaptiverag.llm.factory import (
    PROVIDER_CATALOG,
    SUPPORTED_PROVIDERS,
    LLMProvider,
    ModelBundle,
    ModelTier,
    ProviderCatalogEntry,
    alist_local_models,
    check_provider_configured,
    check_provider_sdk_installed,
    create_chat_model,
    create_model_bundle,
    resolve_effective_settings,
)
from adaptiverag.llm.structured import (
    LLMStructuredResult,
    LLMTextResult,
    extract_json_payload,
    invoke_structured,
    invoke_text,
)

__all__ = [
    "PROVIDER_CATALOG",
    "SUPPORTED_PROVIDERS",
    "BudgetExceededError",
    "CachedResponse",
    "CostTracker",
    "LLMProvider",
    "LLMStructuredResult",
    "LLMTextResult",
    "ModelBundle",
    "ModelTier",
    "ProviderCatalogEntry",
    "SQLiteLLMCache",
    "alist_local_models",
    "check_provider_configured",
    "check_provider_sdk_installed",
    "create_chat_model",
    "create_model_bundle",
    "estimate_cost_usd",
    "extract_json_payload",
    "invoke_structured",
    "invoke_text",
    "resolve_effective_settings",
]
