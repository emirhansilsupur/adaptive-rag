"""Unit tests for Pydantic v2 domain schemas and YAML strategy config loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from adaptiverag.config import ConfidenceConfig, Settings, load_strategies
from adaptiverag.schemas import (
    QueryType,
    StrategyProfile,
    TokenUsage,
)


def test_load_default_strategies_yaml() -> None:
    cfg = load_strategies(Path("configs/strategies.yaml"))

    conv = cfg.for_query_type(QueryType.CONVERSATIONAL)
    factual = cfg.for_query_type(QueryType.FACTUAL)
    analytical = cfg.for_query_type(QueryType.ANALYTICAL)

    assert conv.needs_retrieval is False
    assert factual.needs_retrieval is True
    assert factual.rerank_max_k < analytical.rerank_max_k
    assert factual.alpha_relative > analytical.alpha_relative
    assert analytical.max_sub_queries >= 2


def test_strategy_profile_rejects_inverted_k_bounds() -> None:
    with pytest.raises(ValidationError, match="rerank_min_k"):
        StrategyProfile(rerank_min_k=8, rerank_max_k=4)


def test_confidence_config_requires_unit_weights() -> None:
    with pytest.raises(ValidationError, match=r"sum to 1\.0"):
        ConfidenceConfig(weight_top1=0.5, weight_mean_topk=0.5, weight_subquery_coverage=0.2)


def test_confidence_config_rejects_inverted_thresholds() -> None:
    with pytest.raises(ValidationError, match="low_threshold"):
        ConfidenceConfig(low_threshold=0.8, high_threshold=0.7)


def test_token_usage_addition() -> None:
    a = TokenUsage(
        prompt_tokens=100, completion_tokens=50, total_tokens=150, estimated_cost_usd=0.001
    )
    b = TokenUsage(
        prompt_tokens=200,
        completion_tokens=25,
        total_tokens=225,
        estimated_cost_usd=0.002,
        cache_hits=1,
    )
    combined = a.add(b)
    assert combined.prompt_tokens == 300
    assert combined.completion_tokens == 75
    assert combined.total_tokens == 375
    assert combined.estimated_cost_usd == pytest.approx(0.003)
    assert combined.cache_hits == 1


def test_settings_defaults_to_openrouter_multilingual() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.llm_provider == "openrouter"
    assert settings.llm_fast_model == "qwen/qwen3-30b-a3b-instruct-2507"
    assert settings.llm_generation_model == "deepseek/deepseek-v3.2"
    assert "multilingual" in settings.dense_embedding_model
    assert "m3" in settings.reranker_model or "multilingual" in settings.reranker_model
    assert settings.sparse_embedding_model == "Qdrant/bm25"
