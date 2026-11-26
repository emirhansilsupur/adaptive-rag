"""Resilient structured output & text invocation wrapper for heterogeneous LLM providers."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from pydantic import BaseModel, ValidationError

from adaptiverag.llm.cache import SQLiteLLMCache
from adaptiverag.llm.cost import CostTracker, estimate_cost_usd
from adaptiverag.schemas.response import TokenUsage

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


@dataclass
class LLMTextResult:
    """Result returned by invoke_text."""

    text: str
    usage: TokenUsage
    from_cache: bool = False


@dataclass
class LLMStructuredResult[T: BaseModel]:
    """Result returned by invoke_structured."""

    parsed: T
    raw_text: str
    usage: TokenUsage
    from_cache: bool = False


class StructuredOutputError(ValueError):
    """No reply parsed into the schema. ``usage`` is what the attempts still cost."""

    def __init__(self, message: str, *, usage: TokenUsage) -> None:
        super().__init__(message)
        self.usage = usage


def failed_call_usage(exc: BaseException) -> TokenUsage:
    """Tokens a failed LLM call used anyway; the budget tracker has already counted them.

    Callers that fall back after a failure add this to their own usage, so an answer's
    usage matches what the run spent.
    """
    return exc.usage if isinstance(exc, StructuredOutputError) else TokenUsage()


def _serialize_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for msg in messages:
        serialized.append({"type": msg.type, "content": msg.content})
    return serialized


def _extract_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "\n".join(parts)
    return str(content)


def _extract_token_usage(
    message: BaseMessage, model_name: str, provider: str | None = None
) -> TokenUsage:
    prompt_tokens = 0
    completion_tokens = 0

    usage_meta = getattr(message, "usage_metadata", None)
    if isinstance(usage_meta, dict):
        prompt_tokens = int(usage_meta.get("input_tokens", 0) or 0)
        completion_tokens = int(usage_meta.get("output_tokens", 0) or 0)
    else:
        resp_meta = getattr(message, "response_metadata", {}) or {}
        token_usage = resp_meta.get("token_usage") or resp_meta.get("usage") or {}
        if isinstance(token_usage, dict):
            prompt_tokens = int(
                token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0)) or 0
            )
            completion_tokens = int(
                token_usage.get("completion_tokens", token_usage.get("output_tokens", 0)) or 0
            )

    total_tokens = prompt_tokens + completion_tokens
    cost = estimate_cost_usd(model_name, prompt_tokens, completion_tokens, provider=provider)
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=cost,
    )


def extract_json_payload(raw_text: str) -> str:
    """Extract a JSON object/array string from raw model output, handling code fences or prose."""
    stripped = raw_text.strip()
    if not stripped:
        raise ValueError("Empty LLM response; expected JSON payload.")

    # 1. Direct valid JSON
    if stripped.startswith(("{", "[")):
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            pass

    # 2. Fenced ```json ... ``` block
    fence_match = _CODE_FENCE_RE.search(stripped)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            stripped = candidate

    # 3. Scan for balanced outer JSON object
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        candidate = stripped[start : end + 1]
        json.loads(candidate)  # validates syntax
        return candidate

    raise ValueError(f"Could not extract valid JSON object from LLM response: {raw_text[:200]!r}")


async def _ainvoke_model(
    model: BaseChatModel,
    messages: list[BaseMessage],
    *,
    callbacks: list[Any] | None = None,
    langfuse_prompt: Any | None = None,
    prompt_metadata: dict[str, Any] | None = None,
) -> BaseMessage:
    """Invoke a chat model and natively bind a Langfuse managed prompt object when present.

    Langfuse v5's LangChain ``CallbackHandler`` registers ``metadata["langfuse_prompt"]``
    from an ``on_chain_start`` event and links it to the immediate child generation span.
    Wrapping the model with a zero-overhead ``RunnableLambda | model`` sequence *only* when
    a live Langfuse prompt object is supplied preserves exact generation-to-prompt version
    attribution without altering local-fallback behavior.
    """
    invoke_config: RunnableConfig = {}
    if callbacks:
        invoke_config["callbacks"] = callbacks
    if prompt_metadata:
        invoke_config["metadata"] = {"prompt": dict(prompt_metadata)}
    resolved_config: RunnableConfig | None = invoke_config or None

    if langfuse_prompt is not None:
        prompt_name = (
            str(prompt_metadata.get("name", "managed"))
            if isinstance(prompt_metadata, dict)
            else "managed"
        )
        step_metadata: dict[str, Any] = {"langfuse_prompt": langfuse_prompt}
        if prompt_metadata:
            step_metadata["prompt"] = dict(prompt_metadata)

        def _identity(msgs: list[BaseMessage]) -> list[BaseMessage]:
            return msgs

        prompt_step = RunnableLambda(_identity).with_config(
            run_name=f"prompt:{prompt_name}",
            metadata=step_metadata,
        )
        chain = prompt_step | model
        return await chain.ainvoke(messages, config=resolved_config)

    return await model.ainvoke(messages, config=resolved_config)


async def invoke_text(
    model: BaseChatModel,
    messages: list[BaseMessage],
    *,
    model_name: str,
    provider: str | None = None,
    temperature: float = 0.0,
    cache: SQLiteLLMCache | None = None,
    cost_tracker: CostTracker | None = None,
    callbacks: list[Any] | None = None,
    langfuse_prompt: Any | None = None,
    prompt_metadata: dict[str, Any] | None = None,
) -> LLMTextResult:
    """Invoke a chat model for plain text with optional SQLite caching and budget tracking.

    ``provider`` prices the tokens: a local provider's are free whatever the model is called.
    A reply with no text is neither cached nor served from the cache, so asking the same
    thing again reaches the model instead of replaying the empty reply.
    """
    cache_key: str | None = None
    if cache is not None:
        cache_key = SQLiteLLMCache.build_key(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            schema_name="text",
            messages=_serialize_messages(messages),
        )
        cached = await cache.aget(cache_key)
        if cached is not None and cached.response_text.strip():
            usage = TokenUsage(cache_hits=1)
            if cost_tracker is not None:
                cost_tracker.record(usage)
            return LLMTextResult(text=cached.response_text, usage=usage, from_cache=True)

    if cost_tracker is not None:
        cost_tracker.assert_within_budget()

    response = await _ainvoke_model(
        model,
        messages,
        callbacks=callbacks,
        langfuse_prompt=langfuse_prompt,
        prompt_metadata=prompt_metadata,
    )
    text = _extract_text(response)
    usage = _extract_token_usage(response, model_name, provider)

    if cost_tracker is not None:
        cost_tracker.record(usage)

    if cache is not None and cache_key is not None and text.strip():
        await cache.aset(
            cache_key=cache_key,
            model_name=model_name,
            schema_name="text",
            response_text=text,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )

    return LLMTextResult(text=text, usage=usage, from_cache=False)


async def invoke_structured[T: BaseModel](
    model: BaseChatModel,
    schema: type[T],
    messages: list[BaseMessage],
    *,
    model_name: str,
    provider: str | None = None,
    temperature: float = 0.0,
    max_repair_attempts: int = 1,
    cache: SQLiteLLMCache | None = None,
    cache_if: Callable[[T], bool] | None = None,
    cost_tracker: CostTracker | None = None,
    callbacks: list[Any] | None = None,
    langfuse_prompt: Any | None = None,
    prompt_metadata: dict[str, Any] | None = None,
) -> LLMStructuredResult[T]:
    """Invoke a chat model and parse output into a strict Pydantic v2 schema.

    Designed for high reliability across OpenRouter (Qwen/DeepSeek), OpenAI, Gemini,
    Anthropic, and Ollama without depending on vendor-specific strict json_schema quirks.
    ``provider`` prices the tokens as in ``invoke_text``. If no attempt parses, raises
    StructuredOutputError carrying the usage of every attempt.

    ``cache_if`` tells the cache which replies the caller can use. A reply it turns down,
    such as a judge's empty verdict list, is neither stored nor served from an older entry,
    so asking the same thing again reaches the model instead of replaying the reply.
    """
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    format_instruction = SystemMessage(
        content=(
            "You MUST respond with ONLY a single valid JSON object matching this JSON Schema. "
            "Do not include markdown commentary or extra keys outside the schema.\n"
            f"JSON Schema:\n{schema_json}"
        )
    )
    enriched_messages: list[BaseMessage] = [format_instruction, *messages]

    cache_key: str | None = None
    if cache is not None:
        cache_key = SQLiteLLMCache.build_key(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            schema_name=schema.__name__,
            messages=_serialize_messages(enriched_messages),
        )
        cached = await cache.aget(cache_key)
        if cached is not None:
            parsed = schema.model_validate_json(extract_json_payload(cached.response_text))
            if cache_if is None or cache_if(parsed):
                usage = TokenUsage(cache_hits=1)
                if cost_tracker is not None:
                    cost_tracker.record(usage)
                return LLMStructuredResult(
                    parsed=parsed,
                    raw_text=cached.response_text,
                    usage=usage,
                    from_cache=True,
                )

    cumulative_usage = TokenUsage()
    conversation = list(enriched_messages)
    last_error: Exception | None = None
    last_raw = ""

    for attempt in range(max_repair_attempts + 1):
        if cost_tracker is not None:
            cost_tracker.assert_within_budget()

        response = await _ainvoke_model(
            model,
            conversation,
            callbacks=callbacks,
            langfuse_prompt=langfuse_prompt,
            prompt_metadata=prompt_metadata,
        )
        raw_text = _extract_text(response)
        last_raw = raw_text

        call_usage = _extract_token_usage(response, model_name, provider)
        cumulative_usage = cumulative_usage.add(call_usage)
        if cost_tracker is not None:
            cost_tracker.record(call_usage)

        try:
            payload = extract_json_payload(raw_text)
            parsed = schema.model_validate_json(payload)
            if (
                cache is not None
                and cache_key is not None
                and (cache_if is None or cache_if(parsed))
            ):
                await cache.aset(
                    cache_key=cache_key,
                    model_name=model_name,
                    schema_name=schema.__name__,
                    response_text=payload,
                    prompt_tokens=cumulative_usage.prompt_tokens,
                    completion_tokens=cumulative_usage.completion_tokens,
                )
            return LLMStructuredResult(
                parsed=parsed,
                raw_text=payload,
                usage=cumulative_usage,
                from_cache=False,
            )
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            if attempt >= max_repair_attempts:
                break
            conversation.append(AIMessage(content=raw_text))
            conversation.append(
                HumanMessage(
                    content=(
                        f"Your previous response failed JSON/schema validation with error:\n{exc}\n"
                        "Fix the error and return ONLY the corrected valid JSON object."
                    )
                )
            )

    raise StructuredOutputError(
        f"Failed to parse structured output into {schema.__name__} after "
        f"{max_repair_attempts + 1} attempts. Last error: {last_error}. Raw: {last_raw[:250]!r}",
        usage=cumulative_usage,
    ) from last_error
