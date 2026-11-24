"""Query classification, strategy profile, and query transformation schemas."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class QueryType(StrEnum):
    """High-level intent class that determines the retrieval strategy."""

    CONVERSATIONAL = "conversational"
    FACTUAL = "factual"
    ANALYTICAL = "analytical"


class ChatMessage(BaseModel):
    """Single message in conversation history."""

    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1, max_length=16_000)


class StrategyProfile(BaseModel):
    """Tunable retrieval and dynamic top-k hyperparameters for a query class."""

    needs_retrieval: bool = True
    max_sub_queries: int = Field(default=1, ge=0, le=8)
    dense_prefetch_limit: int = Field(default=20, ge=0, le=200)
    sparse_prefetch_limit: int = Field(default=20, ge=0, le=200)
    fusion_limit: int = Field(default=20, ge=0, le=200)
    rerank_min_k: int = Field(default=2, ge=0, le=30)
    rerank_max_k: int = Field(default=5, ge=0, le=30)
    tau_min: float = Field(default=0.35, ge=0.0, le=1.0)
    alpha_relative: float = Field(default=0.60, ge=0.0, le=1.0)
    delta_gap: float = Field(default=0.25, ge=0.0, le=1.0)
    max_context_tokens: int = Field(default=3000, ge=256, le=32000)

    @model_validator(mode="after")
    def _validate_bounds(self) -> StrategyProfile:
        if self.rerank_min_k > self.rerank_max_k:
            raise ValueError(
                f"rerank_min_k ({self.rerank_min_k}) must be <= rerank_max_k ({self.rerank_max_k})"
            )
        return self


class RouterClassification(BaseModel):
    """Structured output produced by the Query Router LLM."""

    query_type: QueryType
    needs_retrieval: bool = Field(
        description="False only for greetings, chit-chat, or pure meta-commands on prior turns."
    )
    language: Literal["tr", "en", "other"] = Field(
        default="en",
        description="Primary language of the user query so the final answer matches it.",
    )
    standalone_query: str = Field(
        min_length=1,
        description="Self-contained query with pronouns/coreferences resolved from chat history.",
    )
    requires_fresh_web_info: bool = Field(
        default=False,
        description="True when the query explicitly asks about recent releases, live prices, or breaking news unlikely to be in static docs.",
    )
    reasoning: str = Field(
        min_length=1,
        description="Concise 1-sentence justification for the selected route.",
    )


class QueryPlan(BaseModel):
    """Full execution plan produced by the routing and query-planning stage."""

    original_query: str
    standalone_query: str
    query_type: QueryType
    needs_retrieval: bool
    language: Literal["tr", "en", "other"] = "en"
    requires_fresh_web_info: bool = False
    reasoning: str = ""
    strategy: StrategyProfile
    sub_queries: list[str] = Field(default_factory=list)


class SubQueryDecomposition(BaseModel):
    """Structured output for analytical multi-query decomposition."""

    sub_queries: list[str] = Field(
        min_length=1,
        max_length=5,
        description="Distinct, focused search queries covering different facets of the analytical question.",
    )
    rationale: str = Field(default="")


class CorrectiveRewriteResult(BaseModel):
    """Structured output for corrective query reformulation after weak retrieval."""

    rewritten_queries: list[str] = Field(
        min_length=1,
        max_length=4,
        description="Alternative technical phrasing, synonyms, or narrower/broader queries.",
    )
    diagnosis: str = Field(
        description="Brief diagnosis of why the prior retrieval likely missed relevant evidence."
    )
