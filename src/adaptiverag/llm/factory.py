"""Provider-agnostic LLM factory supporting OpenRouter, OpenAI, Ollama, Gemini, and Anthropic."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from adaptiverag.config import Settings


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


def _require_secret(secret: SecretStr | None, env_var: str) -> str:
    if secret is None or not secret.get_secret_value().strip():
        raise ValueError(f"Missing required API key: set {env_var} in your environment or .env")
    return secret.get_secret_value()


def create_chat_model(settings: Settings, tier: ModelTier = ModelTier.FAST) -> BaseChatModel:
    """Instantiate a chat model according to settings.llm_provider and tier."""
    model_name = (
        settings.llm_fast_model if tier == ModelTier.FAST else settings.llm_generation_model
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


def create_model_bundle(settings: Settings) -> ModelBundle:
    """Create both fast and generation chat models from runtime settings."""
    return ModelBundle(
        fast_model=create_chat_model(settings, ModelTier.FAST),
        fast_model_name=settings.llm_fast_model,
        generation_model=create_chat_model(settings, ModelTier.GENERATION),
        generation_model_name=settings.llm_generation_model,
    )
