"""Unit tests for resilient structured output, SQLite prompt cache, and FinOps budget guards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, Field

from adaptiverag.llm import (
    PROVIDER_CATALOG,
    BudgetExceededError,
    CostTracker,
    SQLiteLLMCache,
    StructuredOutputError,
    estimate_cost_usd,
    extract_json_payload,
    failed_call_usage,
    invoke_structured,
    invoke_text,
)
from adaptiverag.llm.cost import LOCAL_PROVIDERS
from adaptiverag.schemas import QueryType, RouterClassification


class ScriptedChatModel(BaseChatModel):
    """Deterministic fake chat model returning pre-scripted responses at $0 cost."""

    responses: list[str] = Field(default_factory=list)
    prompt_tokens: int = 100
    completion_tokens: int = 50
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        msg = AIMessage(
            content=self.responses[idx],
            usage_metadata={
                "input_tokens": self.prompt_tokens,
                "output_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


class SimpleOutput(BaseModel):
    label: str
    score: float


def test_extract_json_payload_handles_markdown_fences_and_prose() -> None:
    raw = 'Here is the result:\n```json\n{"label": "ok", "score": 0.9}\n```\nThanks!'
    extracted = extract_json_payload(raw)
    parsed = SimpleOutput.model_validate_json(extracted)
    assert parsed.label == "ok"
    assert parsed.score == pytest.approx(0.9)


async def test_invoke_structured_repairs_invalid_json_once() -> None:
    model = ScriptedChatModel(
        responses=[
            '{"query_type": "invalid_route", "needs_retrieval": true}',
            (
                "```json\n"
                '{"query_type": "factual", "needs_retrieval": true, "language": "tr", '
                '"standalone_query": "Qdrant ef_construct varsayılan değeri nedir?", '
                '"requires_fresh_web_info": false, "reasoning": "Tek bir teknik parametre soruluyor."}'
                "\n```"
            ),
        ]
    )

    result = await invoke_structured(
        model,
        RouterClassification,
        [HumanMessage(content="Qdrant ef_construct varsayılanı nedir?")],
        model_name="qwen/qwen-2.5-7b-instruct",
        max_repair_attempts=1,
    )

    assert model.call_count == 2
    assert result.parsed.query_type == QueryType.FACTUAL
    assert result.parsed.language == "tr"
    assert result.usage.prompt_tokens == 200
    assert result.usage.estimated_cost_usd > 0.0


async def test_sqlite_cache_prevents_duplicate_llm_calls(tmp_path: Path) -> None:
    cache = SQLiteLLMCache(tmp_path / "cache.sqlite")
    model = ScriptedChatModel(responses=['{"label": "cached", "score": 1.0}'])

    first = await invoke_structured(
        model,
        SimpleOutput,
        [HumanMessage(content="Test prompt")],
        model_name="deepseek/deepseek-chat",
        cache=cache,
    )
    second = await invoke_structured(
        model,
        SimpleOutput,
        [HumanMessage(content="Test prompt")],
        model_name="deepseek/deepseek-chat",
        cache=cache,
    )

    assert model.call_count == 1
    assert first.from_cache is False
    assert second.from_cache is True
    assert second.parsed == first.parsed
    assert second.usage.cache_hits == 1
    assert second.usage.estimated_cost_usd == 0.0


async def test_cost_tracker_enforces_hard_budget_cap() -> None:
    tracker = CostTracker(max_cost_usd=0.0001)
    model = ScriptedChatModel(
        responses=["Hello world"],
        prompt_tokens=100_000,
        completion_tokens=10_000,
    )

    with pytest.raises(BudgetExceededError):
        await invoke_text(
            model,
            [HumanMessage(content="Trigger budget cap")],
            model_name="deepseek/deepseek-chat",
            cost_tracker=tracker,
        )


def test_estimate_cost_zero_for_ollama_and_free_tier() -> None:
    assert estimate_cost_usd("ollama/qwen2.5:7b", 10_000, 5_000) == 0.0
    assert estimate_cost_usd("qwen/qwen3-30b-a3b-instruct-2507:free", 10_000, 5_000) == 0.0
    assert estimate_cost_usd(
        "qwen/qwen3-30b-a3b-instruct-2507", 1_000_000, 1_000_000
    ) == pytest.approx(0.40)
    assert estimate_cost_usd("deepseek/deepseek-v3.2", 1_000_000, 1_000_000) == pytest.approx(0.64)


def test_a_local_provider_is_free_whatever_the_model_is_called() -> None:
    assert estimate_cost_usd("inkling-2-mini", 10_000, 5_000, provider="ollama") == 0.0
    # Without the provider, an unknown name still gets the fallback rate.
    assert estimate_cost_usd("inkling-2-mini", 10_000, 5_000) > 0.0
    assert estimate_cost_usd(
        "deepseek/deepseek-v3.2", 1_000_000, 1_000_000, provider="openrouter"
    ) == pytest.approx(0.64)


def test_local_providers_match_the_provider_catalog() -> None:
    catalog_local = {pid for pid, entry in PROVIDER_CATALOG.items() if entry.is_local}
    assert catalog_local == LOCAL_PROVIDERS


async def test_a_failed_structured_call_reports_the_tokens_it_used() -> None:
    model = ScriptedChatModel(responses=["not json", "still not json"])
    tracker = CostTracker(max_cost_usd=1.0)

    with pytest.raises(StructuredOutputError) as excinfo:
        await invoke_structured(
            model,
            SimpleOutput,
            [HumanMessage(content="Classify this.")],
            model_name="deepseek/deepseek-v3.2",
            max_repair_attempts=1,
            cost_tracker=tracker,
        )

    # Callers that catch ValueError still catch it.
    assert isinstance(excinfo.value, ValueError)
    usage = failed_call_usage(excinfo.value)
    assert model.call_count == 2
    assert usage.total_tokens == 2 * 150
    assert usage == tracker.usage
    assert failed_call_usage(RuntimeError("network down")).total_tokens == 0
