"""Web search fallback retriever supporting Tavily primary + DuckDuckGo zero-key backup."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, runtime_checkable
from urllib.parse import urlparse
from uuid import UUID, uuid5

import structlog

from adaptiverag.config import Settings
from adaptiverag.ingestion.chunker import count_tokens
from adaptiverag.schemas.retrieval import EvidenceChunk, SourceType
from adaptiverag.verification.guardrails import sanitize_untrusted_passage

logger = structlog.get_logger(__name__)

WEB_CHUNK_NAMESPACE = UUID("8b7d42e1-52b9-4f7a-9b30-2a4c6e8f1d03")


@dataclass(frozen=True)
class WebSearchResult:
    """Normalized raw web search hit returned by any external search provider."""

    title: str
    url: str
    content: str
    score: float | None = None
    provider: str = ""


@runtime_checkable
class WebSearchProvider(Protocol):
    """Synchronous provider interface wrapped by WebFallbackRetriever."""

    @property
    def provider_name(self) -> str: ...

    def search(self, query: str, *, max_results: int = 4) -> list[WebSearchResult]: ...


@runtime_checkable
class FallbackRetriever(Protocol):
    """Async fallback retrieval protocol returning standardized EvidenceChunks."""

    async def asearch(
        self,
        query: str,
        *,
        max_results: int | None = None,
        start_citation_index: int = 1,
    ) -> list[EvidenceChunk]: ...


def _clamp_optional_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return round(min(1.0, max(0.0, numeric)), 6)


def _extract_domain(url: str) -> str:
    parsed = urlparse(url.strip())
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain or "web"


class TavilySearchProvider:
    """Primary RAG-optimized web search provider using the Tavily Python SDK."""

    def __init__(
        self,
        api_key: str,
        *,
        search_depth: Literal["basic", "advanced"] = "basic",
        timeout: float = 15.0,
    ) -> None:
        self._api_key = api_key
        self._search_depth = search_depth
        self._timeout = timeout

    @property
    def provider_name(self) -> str:
        return "tavily"

    def search(self, query: str, *, max_results: int = 4) -> list[WebSearchResult]:
        from tavily import TavilyClient

        client = TavilyClient(api_key=self._api_key)
        response = client.search(
            query=query,
            search_depth=self._search_depth,
            max_results=max_results,
            include_answer=False,
            timeout=self._timeout,
        )
        raw_results = response.get("results", []) if isinstance(response, dict) else []
        parsed: list[WebSearchResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            url = str(item.get("url") or "").strip()
            title = str(item.get("title") or url).strip()
            if not content:
                continue
            parsed.append(
                WebSearchResult(
                    title=title or "Web Result",
                    url=url,
                    content=content,
                    score=_clamp_optional_score(item.get("score")),
                    provider=self.provider_name,
                )
            )
        return parsed


class DuckDuckGoSearchProvider:
    """Zero-API-key fallback search provider powered by duckduckgo-search (DDGS)."""

    def __init__(
        self,
        *,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        timeout: float = 15.0,
    ) -> None:
        self._region = region
        self._safesearch = safesearch
        self._timeout = max(1, int(timeout))

    @property
    def provider_name(self) -> str:
        return "duckduckgo"

    def search(self, query: str, *, max_results: int = 4) -> list[WebSearchResult]:
        from duckduckgo_search import DDGS

        with DDGS(timeout=self._timeout) as ddgs:
            raw_results = ddgs.text(
                keywords=query,
                region=self._region,
                safesearch=self._safesearch,
                max_results=max_results,
            )

        parsed: list[WebSearchResult] = []
        for item in raw_results or []:
            if not isinstance(item, dict):
                continue
            content = str(item.get("body") or item.get("snippet") or "").strip()
            url = str(item.get("href") or item.get("url") or "").strip()
            title = str(item.get("title") or url).strip()
            if not content:
                continue
            parsed.append(
                WebSearchResult(
                    title=title or "Web Result",
                    url=url,
                    content=content,
                    score=None,
                    provider=self.provider_name,
                )
            )
        return parsed


class WebFallbackRetriever:
    """Resilient prioritized web fallback chain converting web hits into EvidenceChunks."""

    def __init__(
        self,
        providers: Sequence[WebSearchProvider],
        *,
        default_max_results: int = 4,
        timeout_seconds: float = 15.0,
        enabled: bool = True,
    ) -> None:
        self._providers = list(providers)
        self._default_max_results = max(1, default_max_results)
        self._timeout_seconds = max(0.5, timeout_seconds)
        self._enabled = enabled

    @classmethod
    def from_settings(cls, settings: Settings) -> WebFallbackRetriever:
        """Construct the default provider chain (Tavily -> DuckDuckGo) from Settings."""
        providers: list[WebSearchProvider] = []
        if settings.tavily_api_key is not None:
            api_key = settings.tavily_api_key.get_secret_value().strip()
            if api_key:
                providers.append(TavilySearchProvider(api_key=api_key))
        providers.append(DuckDuckGoSearchProvider())
        return cls(
            providers=providers,
            default_max_results=settings.web_fallback_max_results,
            enabled=settings.web_fallback_enabled,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def provider_names(self) -> list[str]:
        return [p.provider_name for p in self._providers]

    async def asearch(
        self,
        query: str,
        *,
        max_results: int | None = None,
        start_citation_index: int = 1,
    ) -> list[EvidenceChunk]:
        """Execute prioritized web search and normalize hits to ``W1..Wn`` EvidenceChunks."""
        if not self._enabled or not self._providers:
            return []

        cleaned_query = query.strip()
        limit = self._default_max_results if max_results is None else max_results
        if not cleaned_query or limit <= 0:
            return []

        for provider in self._providers:
            try:
                raw_hits = await asyncio.wait_for(
                    asyncio.to_thread(provider.search, cleaned_query, max_results=limit),
                    timeout=self._timeout_seconds,
                )
            except TimeoutError:
                logger.warning(
                    "web_fallback_provider_timeout",
                    provider=provider.provider_name,
                    timeout_seconds=self._timeout_seconds,
                )
                continue
            except Exception as exc:
                logger.warning(
                    "web_fallback_provider_failed",
                    provider=provider.provider_name,
                    error=str(exc),
                )
                continue

            chunks = self._normalize_results(
                raw_hits,
                query=cleaned_query,
                provider_name=provider.provider_name,
                limit=limit,
                start_citation_index=start_citation_index,
            )
            if chunks:
                logger.info(
                    "web_fallback_succeeded",
                    provider=provider.provider_name,
                    result_count=len(chunks),
                )
                return chunks

            logger.warning(
                "web_fallback_provider_empty",
                provider=provider.provider_name,
                query=cleaned_query[:100],
            )

        logger.warning("web_fallback_all_providers_exhausted", query=cleaned_query[:100])
        return []

    def _normalize_results(
        self,
        results: Sequence[WebSearchResult],
        *,
        query: str,
        provider_name: str,
        limit: int,
        start_citation_index: int,
    ) -> list[EvidenceChunk]:
        retrieved_at = datetime.now(UTC).isoformat()
        seen_urls: set[str] = set()
        chunks: list[EvidenceChunk] = []

        for hit in results:
            content = sanitize_untrusted_passage(hit.content).strip()
            if not content:
                continue

            norm_url = hit.url.strip()
            dedup_key = norm_url.lower() if norm_url else content[:120].lower()
            if dedup_key in seen_urls:
                continue
            seen_urls.add(dedup_key)

            idx = len(chunks)
            domain = _extract_domain(norm_url) if norm_url else "web"
            chunk_id = str(uuid5(WEB_CHUNK_NAMESPACE, f"{norm_url}::{content[:120]}"))
            rank_score = round(1.0 / (1.0 + 0.15 * idx), 4)

            chunks.append(
                EvidenceChunk(
                    citation_id=f"W{start_citation_index + idx}",
                    chunk_id=chunk_id,
                    text=content,
                    source_type=SourceType.WEB,
                    doc_id=f"web::{domain}",
                    title=hit.title.strip() or domain,
                    section_path=domain,
                    url=norm_url or None,
                    metadata={
                        "provider": hit.provider or provider_name,
                        "domain": domain,
                        "retrieved_at": retrieved_at,
                    },
                    fusion_score=rank_score,
                    rerank_score=_clamp_optional_score(hit.score),
                    matched_sub_queries=[query],
                    token_count=max(1, count_tokens(content)),
                )
            )
            if len(chunks) >= limit:
                break

        return chunks
