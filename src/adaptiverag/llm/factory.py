"""Provider-agnostic LLM factory supporting OpenRouter, OpenAI, Ollama, Gemini, and Anthropic."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, cast

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from adaptiverag.config import Settings

LLMProvider = Literal["openrouter", "openai", "ollama", "gemini", "anthropic"]
SUPPORTED_PROVIDERS: tuple[LLMProvider, ...] = (
    "openrouter",
    "ollama",
    "openai",
    "gemini",
    "anthropic",
)


class LLMNotConfiguredError(RuntimeError):
    """A run has no models: the server has no LLM of its own and the request brought none."""


class ModelTier(StrEnum):
    """Tiered model selection for FinOps optimization."""

    FAST = "fast"
    GENERATION = "generation"


@dataclass
class ModelBundle:
    """Pair of fast (routing/grading) and generation (synthesis/verification) models."""

    fast_model: BaseChatModel
    fast_model_name: str
    generation_model: BaseChatModel
    generation_model_name: str
    provider: LLMProvider = "openrouter"
    temperature: float | None = None


@dataclass(frozen=True)
class ProviderCatalogEntry:
    """Non-secret provider metadata and curated default/recommended models for Studio UI."""

    id: LLMProvider
    label: str
    is_local: bool
    env_var: str | None
    default_fast_model: str
    default_generation_model: str
    recommended_fast_models: tuple[str, ...]
    recommended_generation_models: tuple[str, ...]
    optional_package: str | None = None


PROVIDER_CATALOG: dict[LLMProvider, ProviderCatalogEntry] = {
    "openrouter": ProviderCatalogEntry(
        id="openrouter",
        label="OpenRouter",
        is_local=False,
        env_var="OPENROUTER_API_KEY",
        default_fast_model="qwen/qwen3-30b-a3b-instruct-2507",
        default_generation_model="deepseek/deepseek-v3.2",
        recommended_fast_models=(
            "qwen/qwen3-30b-a3b-instruct-2507",
            "google/gemini-2.5-flash",
            "openai/gpt-4o-mini",
        ),
        recommended_generation_models=(
            "deepseek/deepseek-v3.2",
            "google/gemini-2.5-pro",
            "anthropic/claude-sonnet-4",
            "openai/gpt-4.1-mini",
        ),
    ),
    "ollama": ProviderCatalogEntry(
        id="ollama",
        label="Local (Ollama / llama-server)",
        is_local=True,
        env_var=None,
        default_fast_model="qwen2.5:7b-instruct",
        default_generation_model="qwen2.5:14b-instruct",
        recommended_fast_models=(
            "inkling-2-mini",
            "qwen2.5:7b-instruct",
            "llama3.1:8b",
        ),
        recommended_generation_models=(
            "inkling-2-mini",
            "qwen2.5:14b-instruct",
            "llama3.1:8b",
        ),
    ),
    "openai": ProviderCatalogEntry(
        id="openai",
        label="OpenAI",
        is_local=False,
        env_var="OPENAI_API_KEY",
        default_fast_model="gpt-4o-mini",
        default_generation_model="gpt-4.1",
        recommended_fast_models=("gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano"),
        recommended_generation_models=("gpt-4.1", "gpt-4o", "gpt-4.1-mini"),
    ),
    "gemini": ProviderCatalogEntry(
        id="gemini",
        label="Google Gemini",
        is_local=False,
        env_var="GEMINI_API_KEY",
        default_fast_model="gemini-2.5-flash",
        default_generation_model="gemini-2.5-pro",
        recommended_fast_models=("gemini-2.5-flash", "gemini-2.0-flash"),
        recommended_generation_models=("gemini-2.5-pro", "gemini-2.5-flash"),
        optional_package="langchain_google_genai",
    ),
    "anthropic": ProviderCatalogEntry(
        id="anthropic",
        label="Anthropic",
        is_local=False,
        env_var="ANTHROPIC_API_KEY",
        default_fast_model="claude-3-5-haiku-latest",
        default_generation_model="claude-sonnet-4-20250514",
        recommended_fast_models=("claude-3-5-haiku-latest",),
        recommended_generation_models=("claude-sonnet-4-20250514", "claude-3-7-sonnet-latest"),
        optional_package="langchain_anthropic",
    ),
}


def check_provider_sdk_installed(provider: LLMProvider) -> tuple[bool, str | None]:
    """Check whether the optional Python SDK required by ``provider`` is installed."""
    entry = PROVIDER_CATALOG[provider]
    if entry.optional_package and importlib.util.find_spec(entry.optional_package) is None:
        return (
            False,
            f"Install optional extra 'providers' (`uv sync --extra providers`) to use {entry.label}.",
        )
    return True, None


def check_provider_configured(
    settings: Settings,
    provider: LLMProvider,
) -> tuple[bool, str | None]:
    """Check whether ``provider`` has credentials (and optional SDK) ready without API calls."""
    sdk_ok, sdk_err = check_provider_sdk_installed(provider)
    if not sdk_ok:
        return False, sdk_err

    entry = PROVIDER_CATALOG[provider]
    if provider == "ollama":
        if not settings.ollama_base_url.strip():
            return False, "Set OLLAMA_BASE_URL in .env to use a local OpenAI-compatible server."
        return True, None

    secret_map: dict[str, SecretStr | None] = {
        "openrouter": settings.openrouter_api_key,
        "openai": settings.openai_api_key,
        "gemini": settings.gemini_api_key,
        "anthropic": settings.anthropic_api_key,
    }
    secret = secret_map.get(provider)
    env_var = entry.env_var or "API_KEY"
    if secret is None or not secret.get_secret_value().strip():
        return False, f"Set {env_var} in .env or enter a session API key to use {entry.label}."
    return True, None


async def alist_local_models(
    base_url: str,
    *,
    timeout_seconds: float = 1.5,
) -> tuple[bool, list[str], str | None]:
    """Probe the server-configured local OpenAI-compatible endpoint's ``/models`` list."""
    cleaned_base = base_url.strip().rstrip("/")
    if not cleaned_base:
        return False, [], "OLLAMA_BASE_URL is not configured."

    models_url = f"{cleaned_base}/models"
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            resp = await client.get(models_url, headers={"Accept": "application/json"})
            resp.raise_for_status()
            payload: Any = resp.json()
    except Exception as exc:
        return False, [], f"Local runtime unreachable at {cleaned_base} ({type(exc).__name__})."

    model_ids: list[str] = []
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    raw_id = item.get("id") or item.get("name") or item.get("model")
                    if isinstance(raw_id, str) and raw_id.strip():
                        model_ids.append(raw_id.strip())
        models_list = payload.get("models")
        if not model_ids and isinstance(models_list, list):
            for item in models_list:
                if isinstance(item, dict):
                    raw_name = item.get("name") or item.get("model") or item.get("id")
                    if isinstance(raw_name, str) and raw_name.strip():
                        model_ids.append(raw_name.strip())

    unique_ids = list(dict.fromkeys(model_ids))
    return True, unique_ids, None


def resolve_effective_settings(
    settings: Settings,
    *,
    provider: LLMProvider | None = None,
    fast_model: str | None = None,
    generation_model: str | None = None,
    temperature: float | None = None,
    api_key_override: str | None = None,
) -> Settings:
    """Return a derived Settings copy applying optional per-request LLM overrides safely."""
    target_provider: LLMProvider = provider or settings.llm_provider
    provider_changed = target_provider != settings.llm_provider
    entry = PROVIDER_CATALOG[target_provider]

    cleaned_fast = fast_model.strip() if fast_model else ""
    cleaned_gen = generation_model.strip() if generation_model else ""
    cleaned_key = api_key_override.strip() if api_key_override else ""

    effective_fast = (
        cleaned_fast
        if cleaned_fast
        else (entry.default_fast_model if provider_changed else settings.llm_fast_model)
    )
    effective_gen = (
        cleaned_gen
        if cleaned_gen
        else (entry.default_generation_model if provider_changed else settings.llm_generation_model)
    )
    effective_temp = settings.llm_temperature if temperature is None else float(temperature)

    if (
        not provider_changed
        and not cleaned_key
        and effective_fast == settings.llm_fast_model
        and effective_gen == settings.llm_generation_model
        and effective_temp == settings.llm_temperature
    ):
        return settings

    updates: dict[str, Any] = {
        "llm_provider": target_provider,
        "llm_fast_model": effective_fast,
        "llm_generation_model": effective_gen,
        "llm_temperature": effective_temp,
    }
    if cleaned_key and target_provider != "ollama":
        key_field_map: dict[str, str] = {
            "openrouter": "openrouter_api_key",
            "openai": "openai_api_key",
            "gemini": "gemini_api_key",
            "anthropic": "anthropic_api_key",
        }
        field_name = key_field_map.get(target_provider)
        if field_name:
            updates[field_name] = SecretStr(cleaned_key)

    return settings.model_copy(update=updates)


def _require_secret(secret: SecretStr | None, env_var: str) -> str:
    if secret is None or not secret.get_secret_value().strip():
        raise ValueError(f"Missing required API key: set {env_var} in your environment or .env")
    return secret.get_secret_value()


def create_chat_model(
    settings: Settings,
    tier: ModelTier = ModelTier.FAST,
    *,
    model_override: str | None = None,
) -> BaseChatModel:
    """Instantiate a chat model according to settings.llm_provider and tier."""
    cleaned_override = model_override.strip() if model_override else ""
    model_name = (
        cleaned_override
        if cleaned_override
        else (settings.llm_fast_model if tier == ModelTier.FAST else settings.llm_generation_model)
    )
    provider = settings.llm_provider

    if provider == "openrouter":
        api_key = _require_secret(settings.openrouter_api_key, "OPENROUTER_API_KEY")
        return ChatOpenAI(
            model=model_name,
            api_key=SecretStr(api_key),
            base_url=settings.openrouter_base_url,
            temperature=settings.llm_temperature,
            timeout=settings.llm_request_timeout,
            default_headers={
                "HTTP-Referer": "https://github.com/adaptiverag",
                "X-Title": "AdaptiveRAG Portfolio Engine",
            },
        )

    if provider == "openai":
        api_key = _require_secret(settings.openai_api_key, "OPENAI_API_KEY")
        return ChatOpenAI(
            model=model_name,
            api_key=SecretStr(api_key),
            temperature=settings.llm_temperature,
            timeout=settings.llm_request_timeout,
        )

    if provider == "ollama":
        return ChatOpenAI(
            model=model_name,
            api_key=SecretStr("ollama"),
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
            timeout=settings.llm_request_timeout,
        )

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError(
                "Install optional extra 'providers' (`uv sync --extra providers`) to use Gemini."
            ) from exc
        api_key = _require_secret(settings.gemini_api_key, "GEMINI_API_KEY")
        return cast(
            BaseChatModel,
            ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=settings.llm_temperature,
                timeout=settings.llm_request_timeout,
            ),
        )

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError(
                "Install optional extra 'providers' (`uv sync --extra providers`) to use Anthropic."
            ) from exc
        api_key = _require_secret(settings.anthropic_api_key, "ANTHROPIC_API_KEY")
        return cast(
            BaseChatModel,
            ChatAnthropic(
                model_name=model_name,
                api_key=SecretStr(api_key),
                temperature=settings.llm_temperature,
                timeout=settings.llm_request_timeout,
                stop=None,
            ),
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")


def create_model_bundle(
    settings: Settings,
    *,
    provider: LLMProvider | None = None,
    fast_model: str | None = None,
    generation_model: str | None = None,
    temperature: float | None = None,
    api_key_override: str | None = None,
) -> ModelBundle:
    """Create both fast and generation chat models from runtime settings & optional overrides."""
    effective = resolve_effective_settings(
        settings,
        provider=provider,
        fast_model=fast_model,
        generation_model=generation_model,
        temperature=temperature,
        api_key_override=api_key_override,
    )
    return ModelBundle(
        fast_model=create_chat_model(effective, ModelTier.FAST),
        fast_model_name=effective.llm_fast_model,
        generation_model=create_chat_model(effective, ModelTier.GENERATION),
        generation_model_name=effective.llm_generation_model,
        provider=effective.llm_provider,
        temperature=effective.llm_temperature,
    )
