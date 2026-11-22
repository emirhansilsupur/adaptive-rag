"""Token cost estimation and hard budget guardrails (FinOps protection)."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from adaptiverag.schemas.response import TokenUsage


@dataclass(frozen=True)
class ModelPricing:
    """USD price per 1 million tokens."""

    input_per_1m: float
    output_per_1m: float


# Approximate reference pricing (USD / 1M tokens) for OpenRouter & major providers.
# Local Ollama and mock models default to $0.00.
MODEL_PRICING: dict[str, ModelPricing] = {
    # OpenRouter — Qwen family (verified reference rates)
    "qwen/qwen3-30b-a3b-instruct-2507": ModelPricing(0.10, 0.30),
    "qwen/qwen3-235b-a22b-instruct-2507": ModelPricing(0.12, 0.58),
    "qwen/qwen-2.5-7b-instruct": ModelPricing(0.10, 0.20),
    "qwen/qwen-2.5-72b-instruct": ModelPricing(0.15, 0.40),
    "qwen/qwen3-32b": ModelPricing(0.10, 0.30),
    "qwen/qwen3-8b": ModelPricing(0.05, 0.15),
    # OpenRouter — DeepSeek family
    "deepseek/deepseek-v3.2": ModelPricing(0.26, 0.38),
    "deepseek/deepseek-chat-v3.1": ModelPricing(0.25, 0.95),
    "deepseek/deepseek-chat": ModelPricing(0.26, 1.03),
    "deepseek/deepseek-chat-v3-0324": ModelPricing(0.26, 1.03),
    "deepseek/deepseek-r1": ModelPricing(0.55, 2.19),
    # OpenAI
    "gpt-4o-mini": ModelPricing(0.15, 0.60),
    "gpt-4.1-mini": ModelPricing(0.40, 1.60),
    "gpt-4o": ModelPricing(2.50, 10.00),
    # Google Gemini
    "gemini-2.5-flash": ModelPricing(0.15, 0.60),
    "google/gemini-2.5-flash": ModelPricing(0.15, 0.60),
    # Anthropic
    "claude-3-5-haiku-latest": ModelPricing(0.80, 4.00),
    "claude-3-7-sonnet-latest": ModelPricing(3.00, 15.00),
}

DEFAULT_FALLBACK_PRICING = ModelPricing(0.30, 1.00)


class BudgetExceededError(RuntimeError):
    """Raised when cumulative LLM spend exceeds the configured budget cap."""


def estimate_cost_usd(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Estimate USD cost for a single LLM invocation."""
    normalized = model_name.strip().lower()
    if normalized.startswith(("ollama/", "fake", "mock")) or ":free" in normalized:
        return 0.0

    pricing = MODEL_PRICING.get(normalized)
    if pricing is None:
        # Prefix match for versioned tags such as "deepseek/deepseek-chat:nitro"
        base = normalized.split(":", maxsplit=1)[0]
        pricing = MODEL_PRICING.get(base, DEFAULT_FALLBACK_PRICING)

    cost = (prompt_tokens * pricing.input_per_1m + completion_tokens * pricing.output_per_1m) / 1_000_000.0
    return round(cost, 8)


class CostTracker:
    """Thread-safe cumulative token & USD budget tracker with hard stop."""

    def __init__(self, max_cost_usd: float | None = None) -> None:
        self._max_cost_usd = max_cost_usd
        self._usage = TokenUsage()
        self._lock = Lock()

    @property
    def usage(self) -> TokenUsage:
        with self._lock:
            return self._usage.model_copy()

    @property
    def max_cost_usd(self) -> float | None:
        return self._max_cost_usd

    def assert_within_budget(self) -> None:
        with self._lock:
            if (
                self._max_cost_usd is not None
                and self._usage.estimated_cost_usd >= self._max_cost_usd
            ):
                raise BudgetExceededError(
                    f"LLM budget cap reached: ${self._usage.estimated_cost_usd:.4f} >= "
                    f"${self._max_cost_usd:.4f}"
                )

    def record(self, delta: TokenUsage) -> TokenUsage:
        with self._lock:
            self._usage = self._usage.add(delta)
            current = self._usage.model_copy()
            if (
                self._max_cost_usd is not None
                and current.estimated_cost_usd > self._max_cost_usd
            ):
                raise BudgetExceededError(
                    f"LLM budget cap exceeded: ${current.estimated_cost_usd:.4f} > "
                    f"${self._max_cost_usd:.4f}"
                )
            return current
