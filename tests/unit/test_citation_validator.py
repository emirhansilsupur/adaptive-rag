"""Unit tests for Two-Tier Citation Validation (Tier-1 structural + Tier-2 semantic entailment)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from adaptiverag.config import CitationValidationConfig
from adaptiverag.llm import BudgetExceededError, CostTracker, SQLiteLLMCache
from adaptiverag.schemas import ClaimVerdict, EvidenceChunk, SemanticAuditStatus, SourceType
from adaptiverag.verification import (
    extract_citations,
    find_uncited_factual_sentences,
    validate_citations,
    validate_citations_structural,
)


class _ScriptedAuditModel(BaseChatModel):
    responses: list[str] = Field(default_factory=list)
    fail_with_error: bool = False
    fail_with_timeout: bool = False
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-citation-audit"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.call_count += 1
        if self.fail_with_error:
            raise RuntimeError("Upstream OpenRouter 503 error")
        if self.fail_with_timeout:
            raise TimeoutError("judge did not answer in time")
        idx = min(self.call_count - 1, len(self.responses) - 1)
        msg = AIMessage(
            content=self.responses[idx],
            usage_metadata={
                "input_tokens": 180,
                "output_tokens": 70,
                "total_tokens": 250,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _make_chunks() -> list[EvidenceChunk]:
    return [
        EvidenceChunk(
            citation_id="C1",
            chunk_id="chunk-1",
            doc_id="qdrant_guide",
            title="Qdrant Hybrid Search",
            section_path="RRF",
            text="Qdrant fuses dense and BM25 sparse vectors natively using Reciprocal Rank Fusion.",
            source_type=SourceType.LOCAL,
            rerank_score=0.91,
        ),
        EvidenceChunk(
            citation_id="W1",
            chunk_id="web-1",
            doc_id="web::qdrant.tech",
            title="Qdrant Release Notes",
            section_path="qdrant.tech",
            url="https://qdrant.tech/releases",
            text="Qdrant 1.13 added improved sparse index quantization for lower RAM footprint.",
            source_type=SourceType.WEB,
            rerank_score=0.84,
        ),
    ]


def test_extract_citations_handles_single_grouped_and_ignores_code_and_links() -> None:
    text = (
        "Qdrant uses native RRF [C1] and supports grouped tags [c1, W1; C2][W2].\n"
        "See [C1](https://qdrant.tech) for docs.\n"
        "```python\nvalue = items[C99]\n```\n"
    )
    assert extract_citations(text) == ["C1", "C1", "W1", "C2", "W2"]


def test_tier1_structural_passes_for_clean_cited_answer() -> None:
    answer = (
        "## Overview\n"
        "Key points:\n"
        "- Qdrant combines dense embeddings and BM25 sparse vectors natively via RRF [C1].\n"
        "- Recent releases also introduced sparse quantization to reduce RAM usage [W1].\n"
    )
    report = validate_citations_structural(answer, _make_chunks())

    assert report.passed is True
    assert report.total_citations_found == 2
    assert report.valid_citation_ids == ["C1", "W1"]
    assert report.invalid_citation_ids == []
    assert report.uncited_factual_sentences == []
    assert report.citation_precision == 1.0
    assert report.claim_support_rate == 1.0


def test_tier1_structural_catches_ghost_citation_id() -> None:
    answer = (
        "Qdrant executes Reciprocal Rank Fusion on the server side [C1]. "
        "It automatically enables GPU acceleration on macOS by default [C99]."
    )
    report = validate_citations_structural(answer, _make_chunks())

    assert report.passed is False
    assert report.valid_citation_ids == ["C1"]
    assert report.invalid_citation_ids == ["C99"]
    assert report.citation_precision == 0.5


def test_tier1_detects_uncited_factual_sentences_while_ignoring_abstentions_and_code() -> None:
    answer = (
        "### Architecture Summary\n"
        "Qdrant merges dense cosine vectors and BM25 sparse vectors natively via RRF [C1]. "
        "It also ships with a built-in SQL query optimizer for relational joins across shards.\n"
        "```python\nx = [1, 2, 3]\n```\n"
        "Yeterli doğrulanmış kanıt bulunamadığı için bu konuda tahminde bulunmuyorum."
    )
    uncited = find_uncited_factual_sentences(answer)
    assert len(uncited) == 1
    assert "SQL query optimizer" in uncited[0]

    report = validate_citations_structural(answer, _make_chunks())
    assert report.passed is False
    assert report.claim_support_rate == 0.5


@pytest.mark.asyncio
async def test_tier2_semantic_entailment_passes_all_supported() -> None:
    payload = json.dumps(
        {
            "verifications": [
                {
                    "claim": "Qdrant fuses dense and BM25 sparse vectors natively via RRF.",
                    "cited_ids": ["C1"],
                    "verdict": "supported",
                    "explanation": "Directly stated in C1.",
                },
                {
                    "claim": "Qdrant 1.13 improved sparse quantization to lower RAM footprint.",
                    "cited_ids": ["W1"],
                    "verdict": "supported",
                    "explanation": "Directly stated in W1.",
                },
            ]
        }
    )
    model = _ScriptedAuditModel(responses=[payload])
    answer = (
        "Qdrant fuses dense and BM25 sparse vectors natively via RRF [C1]. "
        "Qdrant 1.13 added improved sparse quantization to lower RAM usage [W1]."
    )

    report, usage = await validate_citations(
        answer=answer,
        evidence_chunks=_make_chunks(),
        config=CitationValidationConfig(min_claim_support_rate=0.85),
        fast_model=model,
        fast_model_name="qwen/qwen3-30b-a3b-instruct-2507",
    )

    assert model.call_count == 1
    assert report.passed is True
    assert report.structural_passed is True
    assert report.semantic_audit == SemanticAuditStatus.COMPLETED
    assert report.semantic_audit_error is None
    assert report.claim_support_rate == 1.0
    assert len(report.claim_verifications) == 2
    assert usage.total_tokens == 250


@pytest.mark.asyncio
async def test_tier2_contradicted_claim_always_fails() -> None:
    payload = json.dumps(
        {
            "verifications": [
                {
                    "claim": "Qdrant uses RRF to merge dense and sparse vectors.",
                    "cited_ids": ["C1"],
                    "verdict": "supported",
                    "explanation": "Supported by C1.",
                },
                {
                    "claim": "Qdrant 1.13 increased RAM usage by removing sparse quantization.",
                    "cited_ids": ["W1"],
                    "verdict": "contradicted",
                    "explanation": "W1 states it lowered RAM footprint.",
                },
            ]
        }
    )
    model = _ScriptedAuditModel(responses=[payload])
    answer = (
        "Qdrant uses RRF to merge dense and sparse vectors [C1]. "
        "Qdrant 1.13 increased RAM usage by removing sparse quantization [W1]."
    )

    # Even with a permissive min_claim_support_rate=0.40, any CONTRADICTED claim must fail.
    report, _ = await validate_citations(
        answer=answer,
        evidence_chunks=_make_chunks(),
        config=CitationValidationConfig(min_claim_support_rate=0.40),
        fast_model=model,
        fast_model_name="qwen/qwen3-30b-a3b-instruct-2507",
    )

    assert report.passed is False
    assert report.claim_verifications[1].verdict == ClaimVerdict.CONTRADICTED


@pytest.mark.asyncio
async def test_tier2_allow_partial_as_supported_toggle() -> None:
    payload = json.dumps(
        {
            "verifications": [
                {
                    "claim": "Qdrant merges vectors natively via RRF with zero latency overhead.",
                    "cited_ids": ["C1"],
                    "verdict": "partially_supported",
                    "explanation": "RRF is stated, but zero latency overhead is not.",
                }
            ]
        }
    )
    answer = "Qdrant merges vectors natively via RRF with zero latency overhead [C1]."

    strict_model = _ScriptedAuditModel(responses=[payload])
    strict_report, _ = await validate_citations(
        answer=answer,
        evidence_chunks=_make_chunks(),
        config=CitationValidationConfig(
            min_claim_support_rate=0.85,
            allow_partial_as_supported=False,
        ),
        fast_model=strict_model,
        fast_model_name="qwen/qwen3-30b-a3b-instruct-2507",
    )
    assert strict_report.passed is False
    assert strict_report.claim_support_rate == 0.0

    lenient_model = _ScriptedAuditModel(responses=[payload])
    lenient_report, _ = await validate_citations(
        answer=answer,
        evidence_chunks=_make_chunks(),
        config=CitationValidationConfig(
            min_claim_support_rate=0.85,
            allow_partial_as_supported=True,
        ),
        fast_model=lenient_model,
        fast_model_name="qwen/qwen3-30b-a3b-instruct-2507",
    )
    assert lenient_report.passed is True
    assert lenient_report.claim_support_rate == 1.0


@pytest.mark.asyncio
async def test_tier2_finops_fast_fail_skips_llm_when_zero_valid_citations() -> None:
    model = _ScriptedAuditModel(responses=["{}"])
    answer = (
        "Qdrant natively merges dense and sparse vectors using Reciprocal Rank Fusion "
        "without requiring any external proxy server."
    )

    report, usage = await validate_citations(
        answer=answer,
        evidence_chunks=_make_chunks(),
        config=CitationValidationConfig(enable_semantic_entailment=True),
        fast_model=model,
        fast_model_name="qwen/qwen3-30b-a3b-instruct-2507",
    )

    assert model.call_count == 0
    assert report.passed is False
    assert report.structural_passed is False
    assert report.semantic_audit == SemanticAuditStatus.SKIPPED
    assert report.claim_support_rate == 0.0
    assert usage.total_tokens == 0


@pytest.mark.asyncio
async def test_tier2_propagates_budget_error_and_degrades_on_provider_error() -> None:
    answer = "Qdrant merges dense and BM25 sparse vectors natively via RRF [C1]."

    # 1. BudgetExceededError must NOT be swallowed
    from adaptiverag.schemas import TokenUsage

    exhausted = CostTracker(max_cost_usd=0.0001)
    exhausted.record(
        TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            estimated_cost_usd=0.0001,
        )
    )
    model = _ScriptedAuditModel(responses=["{}"])
    with pytest.raises(BudgetExceededError):
        await validate_citations(
            answer=answer,
            evidence_chunks=_make_chunks(),
            config=CitationValidationConfig(enable_semantic_entailment=True),
            fast_model=model,
            fast_model_name="qwen/qwen3-30b-a3b-instruct-2507",
            cost_tracker=exhausted,
        )

    # 2. A provider error keeps the Tier-1 findings but fails closed instead of passing
    failing_model = _ScriptedAuditModel(fail_with_error=True)
    report, usage = await validate_citations(
        answer=answer,
        evidence_chunks=_make_chunks(),
        config=CitationValidationConfig(enable_semantic_entailment=True),
        fast_model=failing_model,
        fast_model_name="qwen/qwen3-30b-a3b-instruct-2507",
    )
    assert failing_model.call_count == 1
    assert report.passed is False
    assert report.structural_passed is True
    assert report.semantic_audit == SemanticAuditStatus.FAILED
    assert report.semantic_audit_error == "The judge model call failed (RuntimeError)."
    assert "503" not in report.semantic_audit_error  # provider text stays in server logs
    assert report.valid_citation_ids == ["C1"]
    assert usage.total_tokens == 0


_WRONG_NUMBER_EVIDENCE = [
    EvidenceChunk(
        citation_id="C1",
        chunk_id="c1",
        doc_id="qdrant_hnsw",
        title="Qdrant HNSW",
        text="The default value of ef_construct in Qdrant HNSW is 100.",
        source_type=SourceType.LOCAL,
    )
]
# Cites [C1] correctly but states the wrong number, so only the semantic judge can catch it.
_WRONG_NUMBER_ANSWER = "The default value of ef_construct in Qdrant HNSW is 512 [C1]."


@pytest.mark.asyncio
async def test_tier2_unparsable_judge_reply_fails_closed() -> None:
    judge = _ScriptedAuditModel(responses=["sorry, I cannot produce JSON"])

    report, _ = await validate_citations(
        answer=_WRONG_NUMBER_ANSWER,
        evidence_chunks=_WRONG_NUMBER_EVIDENCE,
        config=CitationValidationConfig(),
        fast_model=judge,
        fast_model_name="local-judge",
    )

    assert judge.call_count == 2  # first reply + one JSON repair attempt
    assert report.passed is False
    assert report.structural_passed is True
    assert report.semantic_audit == SemanticAuditStatus.FAILED
    assert report.semantic_audit_error is not None
    assert "could not be parsed" in report.semantic_audit_error
    assert report.claim_verifications == []


@pytest.mark.asyncio
async def test_tier2_judge_timeout_fails_closed() -> None:
    judge = _ScriptedAuditModel(fail_with_timeout=True)

    report, _ = await validate_citations(
        answer=_WRONG_NUMBER_ANSWER,
        evidence_chunks=_WRONG_NUMBER_EVIDENCE,
        config=CitationValidationConfig(),
        fast_model=judge,
        fast_model_name="local-judge",
    )

    assert report.passed is False
    assert report.semantic_audit == SemanticAuditStatus.FAILED
    assert report.semantic_audit_error == "The judge model timed out."


@pytest.mark.asyncio
async def test_tier2_empty_verdicts_for_cited_answer_fails_closed() -> None:
    judge = _ScriptedAuditModel(responses=[json.dumps({"verifications": []})])

    report, usage = await validate_citations(
        answer=_WRONG_NUMBER_ANSWER,
        evidence_chunks=_WRONG_NUMBER_EVIDENCE,
        config=CitationValidationConfig(),
        fast_model=judge,
        fast_model_name="local-judge",
    )

    assert report.passed is False
    assert report.structural_passed is True
    assert report.semantic_audit == SemanticAuditStatus.FAILED
    assert report.semantic_audit_error is not None
    assert "no claim verdicts" in report.semantic_audit_error
    assert usage.total_tokens == 250  # the judge call succeeded, so its tokens still count


@pytest.mark.asyncio
async def test_tier2_completed_judge_catches_the_wrong_number() -> None:
    judge = _ScriptedAuditModel(
        responses=[
            json.dumps(
                {
                    "verifications": [
                        {
                            "claim": "The default ef_construct in Qdrant HNSW is 512.",
                            "cited_ids": ["C1"],
                            "verdict": "contradicted",
                            "explanation": "C1 says the default is 100.",
                        }
                    ]
                }
            )
        ]
    )

    report, _ = await validate_citations(
        answer=_WRONG_NUMBER_ANSWER,
        evidence_chunks=_WRONG_NUMBER_EVIDENCE,
        config=CitationValidationConfig(),
        fast_model=judge,
        fast_model_name="local-judge",
    )

    assert report.passed is False
    assert report.structural_passed is True
    assert report.semantic_audit == SemanticAuditStatus.COMPLETED


@pytest.mark.asyncio
async def test_an_empty_answer_fails_without_asking_the_judge() -> None:
    judge = _ScriptedAuditModel(responses=[json.dumps({"verifications": []})])

    report, usage = await validate_citations(
        answer="  \n",
        evidence_chunks=_make_chunks(),
        config=CitationValidationConfig(),
        fast_model=judge,
        fast_model_name="local-judge",
    )

    assert report.empty_answer is True
    assert report.passed is False
    assert report.structural_passed is False
    assert report.semantic_audit == SemanticAuditStatus.SKIPPED
    assert judge.call_count == 0
    assert usage.total_tokens == 0


@pytest.mark.asyncio
async def test_tier2_verdicts_are_reused_from_the_cache(tmp_path: Path) -> None:
    cache = SQLiteLLMCache(tmp_path / "cache.sqlite")
    judge = _ScriptedAuditModel(
        responses=[
            json.dumps(
                {
                    "verifications": [
                        {
                            "claim": "The default ef_construct in Qdrant HNSW is 512.",
                            "cited_ids": ["C1"],
                            "verdict": "contradicted",
                            "explanation": "C1 says the default is 100.",
                        }
                    ]
                }
            )
        ]
    )

    for _ in range(2):
        report, _ = await validate_citations(
            answer=_WRONG_NUMBER_ANSWER,
            evidence_chunks=_WRONG_NUMBER_EVIDENCE,
            config=CitationValidationConfig(),
            fast_model=judge,
            fast_model_name="local-judge",
            cache=cache,
        )
        assert report.semantic_audit == SemanticAuditStatus.COMPLETED
        assert report.passed is False

    assert judge.call_count == 1


@pytest.mark.asyncio
async def test_semantic_audit_disabled_reports_skipped_structural_verdict() -> None:
    judge = _ScriptedAuditModel(responses=["{}"])

    report, usage = await validate_citations(
        answer=_WRONG_NUMBER_ANSWER,
        evidence_chunks=_WRONG_NUMBER_EVIDENCE,
        config=CitationValidationConfig(enable_semantic_entailment=False),
        fast_model=judge,
        fast_model_name="local-judge",
    )

    assert judge.call_count == 0
    assert report.passed is True
    assert report.structural_passed is True
    assert report.semantic_audit == SemanticAuditStatus.SKIPPED
    assert report.semantic_audit_error is None
    assert usage.total_tokens == 0
