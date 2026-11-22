"""$0 offline unit tests for Firecrawl Local web loader, SSRF protection, and CLI flags."""

from __future__ import annotations

import json
import socket
from typing import Any

import httpx
import pytest

from adaptiverag.config import Settings
from adaptiverag.ingestion.indexer import _parse_args
from adaptiverag.ingestion.web_loader import (
    FirecrawlClient,
    FirecrawlError,
    FirecrawlUnavailableError,
    UnsafeURLError,
    make_web_doc_id,
    validate_public_url,
)


def test_validate_public_url_accepts_public_and_strips_fragment() -> None:
    normalized = validate_public_url("  https://docs.qdrant.tech/concepts/hybrid-queries/#rrf  ")
    assert normalized == "https://docs.qdrant.tech/concepts/hybrid-queries/"

    root = validate_public_url("http://example.com")
    assert root == "http://example.com/"


@pytest.mark.parametrize(
    "unsafe_url",
    [
        "",
        "   ",
        "file:///etc/passwd",
        "ftp://example.com/file.txt",
        "javascript:alert(1)",
        "http://localhost:3002/v2/scrape",
        "http://sub.localhost/test",
        "http://127.0.0.1:6333/collections",
        "http://10.0.0.5/internal",
        "http://172.16.0.10/admin",
        "http://192.168.1.1/router",
        "http://169.254.169.254/latest/meta-data/",
        "http://0.0.0.0:8000/",
        "http://[::1]:8000/",
        "http://[::ffff:127.0.0.1]/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://printer.local/status",
        "http://service.internal/metrics",
        "https://user:secret@example.com/docs",
    ],
)
def test_validate_public_url_blocks_ssrf_and_unsafe_schemes(unsafe_url: str) -> None:
    with pytest.raises(UnsafeURLError):
        validate_public_url(unsafe_url)


def test_validate_public_url_blocks_dns_rebinding_to_private_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_getaddrinfo(*args: Any, **kwargs: Any) -> list[Any]:
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("127.0.0.1", 0),
            )
        ]

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo)
    with pytest.raises(UnsafeURLError, match="non-public IP"):
        validate_public_url("https://rebind.example.com/docs", resolve_dns=True)


def test_validate_public_url_allows_private_when_explicitly_enabled() -> None:
    assert (
        validate_public_url("http://127.0.0.1:8080/docs", allow_private=True)
        == "http://127.0.0.1:8080/docs"
    )


def test_make_web_doc_id_is_deterministic_and_distinct() -> None:
    a = make_web_doc_id("https://docs.qdrant.tech/concepts/hybrid-queries/")
    b = make_web_doc_id("https://docs.qdrant.tech/concepts/hybrid-queries")
    c = make_web_doc_id("https://docs.qdrant.tech/concepts/hybrid-queries/?v=2")
    assert a == b
    assert a != c
    assert a.startswith("web_docs_qdrant_tech_concepts_hybrid_queries_")


@pytest.mark.asyncio
async def test_firecrawl_scrape_url_normalizes_en_and_tr_markdown() -> None:
    seen_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v2/scrape"
        assert request.headers.get("authorization") == "Bearer test-secret"
        body = json.loads(request.content.decode("utf-8"))
        seen_bodies.append(body)
        target = body["url"]
        if "turkce" in target:
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        "markdown": (
                            "# Hibrit Arama Mimarisi\n\n"
                            "Yoğun vektörler ve BM25 seyrek vektörleri RRF ile birleştirilir.\n"
                        ),
                        "metadata": {
                            "sourceURL": target,
                            "statusCode": 200,
                        },
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "markdown": "# Hybrid Queries\n\n## RRF\n\nQdrant combines dense and sparse prefetch.",
                    "metadata": {
                        "title": "Qdrant Hybrid Queries",
                        "description": "Hybrid search documentation",
                        "language": "en-US",
                        "sourceURL": target,
                        "statusCode": 200,
                    },
                },
            },
        )

    transport = httpx.MockTransport(handler)
    async with FirecrawlClient(
        "http://localhost:3002",
        api_key="test-secret",
        resolve_dns=False,
        transport=transport,
    ) as client:
        en_doc = await client.ascrape_url("https://docs.qdrant.tech/concepts/hybrid-queries/#rrf")
        tr_doc = await client.ascrape_url("https://docs.example.com/turkce-rehber")

    assert len(seen_bodies) == 2
    assert seen_bodies[0]["formats"] == ["markdown"]
    assert seen_bodies[0]["onlyMainContent"] is True

    assert en_doc.title == "Qdrant Hybrid Queries"
    assert en_doc.doc_type == "markdown"
    assert en_doc.language == "en"
    assert en_doc.source_path == "https://docs.qdrant.tech/concepts/hybrid-queries/"
    assert en_doc.metadata["source_url"] == "https://docs.qdrant.tech/concepts/hybrid-queries/"
    assert en_doc.metadata["source_origin"] == "firecrawl_local"
    assert en_doc.metadata["description"] == "Hybrid search documentation"
    assert en_doc.metadata["status_code"] == 200

    # Turkish page falls back to H1 heading and detects TR from diacritics
    assert tr_doc.title == "Hibrit Arama Mimarisi"
    assert tr_doc.language == "tr"


@pytest.mark.asyncio
async def test_firecrawl_scrape_urls_deduplicates_and_fails_fast_on_ssrf() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "markdown": f"# Page {calls}\n\nContent for {body['url']}",
                    "metadata": {"sourceURL": body["url"]},
                },
            },
        )

    transport = httpx.MockTransport(handler)
    async with FirecrawlClient(
        "http://localhost:3002", resolve_dns=False, transport=transport
    ) as client:
        # Unsafe second URL fails before ANY request is made
        with pytest.raises(UnsafeURLError):
            await client.ascrape_urls(
                [
                    "https://docs.example.com/a",
                    "http://169.254.169.254/latest/meta-data/",
                ]
            )
        assert calls == 0

        # Fragment duplicate collapses into one scrape
        docs = await client.ascrape_urls(
            [
                "https://docs.example.com/a#intro",
                "https://docs.example.com/a#details",
                "https://docs.example.com/b",
            ]
        )
        assert len(docs) == 2
        assert calls == 2


@pytest.mark.asyncio
async def test_firecrawl_scrape_errors_and_unavailable_mapping() -> None:
    def empty_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"success": True, "data": {"markdown": "   "}})

    async with FirecrawlClient(
        "http://localhost:3002",
        resolve_dns=False,
        transport=httpx.MockTransport(empty_handler),
    ) as client:
        with pytest.raises(FirecrawlError, match="empty markdown"):
            await client.ascrape_url("https://docs.example.com/empty")

    def server_error_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"success": False, "error": "Chromium crashed"})

    async with FirecrawlClient(
        "http://localhost:3002",
        resolve_dns=False,
        transport=httpx.MockTransport(server_error_handler),
    ) as client:
        with pytest.raises(FirecrawlError, match="HTTP 500"):
            await client.ascrape_url("https://docs.example.com/boom")

    def connect_error_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    async with FirecrawlClient(
        "http://localhost:3002",
        resolve_dns=False,
        transport=httpx.MockTransport(connect_error_handler),
    ) as client:
        with pytest.raises(FirecrawlUnavailableError, match="firecrawl-local serve"):
            await client.ascrape_url("https://docs.example.com/offline")


@pytest.mark.asyncio
async def test_firecrawl_crawl_site_polls_paginates_and_deduplicates() -> None:
    poll_count = 0
    seen_crawl_payload: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal poll_count, seen_crawl_payload
        if request.method == "POST" and request.url.path == "/v2/crawl":
            seen_crawl_payload = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "id": "job-42",
                    "url": "http://localhost:3002/v2/crawl/job-42",
                },
            )
        if request.method == "GET" and request.url.path == "/v2/crawl/job-42":
            if request.url.query == b"skip=2":
                return httpx.Response(
                    200,
                    json={
                        "status": "completed",
                        "next": None,
                        "data": [
                            {
                                "markdown": "# Page 2\n\nSecond crawled docs page.",
                                "metadata": {
                                    "title": "Page 2",
                                    "sourceURL": "https://docs.example.com/page-2",
                                },
                            },
                            # Redirected alias pointing to the same canonical metadata.url should deduplicate
                            {
                                "markdown": "# Page 1 Redirected Alias",
                                "metadata": {
                                    "title": "Page 1",
                                    "sourceURL": "https://docs.example.com/old-page-1",
                                    "url": "https://docs.example.com/page-1",
                                },
                            },
                        ],
                    },
                )
            poll_count += 1
            if poll_count == 1:
                return httpx.Response(
                    200,
                    json={"status": "scraping", "total": 2, "completed": 1, "data": []},
                )
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "total": 2,
                    "completed": 2,
                    "next": "http://localhost:3002/v2/crawl/job-42?skip=2",
                    "data": [
                        {
                            "markdown": "# Page 1\n\nFirst crawled docs page.",
                            "metadata": {
                                "title": "Page 1",
                                "sourceURL": "https://docs.example.com/page-1",
                            },
                        },
                        # Empty page should be skipped cleanly
                        {
                            "markdown": "   ",
                            "metadata": {"sourceURL": "https://docs.example.com/blank"},
                        },
                    ],
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    async with FirecrawlClient(
        "http://localhost:3002",
        resolve_dns=False,
        transport=httpx.MockTransport(handler),
    ) as client:
        docs = await client.acrawl_site(
            "https://docs.example.com/",
            limit=5,
            max_depth=2,
            poll_interval_seconds=0.0,
        )

    assert seen_crawl_payload["limit"] == 5
    assert seen_crawl_payload["maxDiscoveryDepth"] == 2
    assert seen_crawl_payload["scrapeOptions"]["formats"] == ["markdown"]
    assert poll_count == 2
    assert len(docs) == 2
    assert [d.title for d in docs] == ["Page 1", "Page 2"]
    assert all(d.metadata["crawl_root_url"] == "https://docs.example.com/" for d in docs)


@pytest.mark.asyncio
async def test_firecrawl_crawl_site_raises_on_failed_job() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"success": True, "id": "job-fail"})
        return httpx.Response(200, json={"status": "failed", "error": "Blocked by robots.txt"})

    async with FirecrawlClient(
        "http://localhost:3002",
        resolve_dns=False,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(FirecrawlError, match=r"Blocked by robots\.txt"):
            await client.acrawl_site(
                "https://docs.example.com/",
                poll_interval_seconds=0.0,
            )


@pytest.mark.asyncio
async def test_firecrawl_health_check_and_settings_disabled() -> None:
    # Empty FIRECRAWL_LOCAL_URL disables the client cleanly
    disabled_settings = Settings(firecrawl_local_url="   ")
    assert disabled_settings.firecrawl_local_url is None
    assert FirecrawlClient.from_settings(disabled_settings) is None

    # 404 on /health still confirms local HTTP listener is up
    async with FirecrawlClient(
        "http://localhost:3002",
        resolve_dns=False,
        transport=httpx.MockTransport(lambda req: httpx.Response(404)),
    ) as ok_client:
        health = await ok_client.ahealth_check()
        assert health["status"] == "ok"
        assert health["base_url"] == "http://localhost:3002"

    def _refused(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    async with FirecrawlClient(
        "http://localhost:3002",
        resolve_dns=False,
        transport=httpx.MockTransport(_refused),
    ) as down_client:
        health_down = await down_client.ahealth_check()
        assert health_down["status"] == "unreachable"


def test_indexer_cli_parses_web_flags_and_rejects_conflicts() -> None:
    url_args = _parse_args(
        ["--url", "https://docs.example.com/a", "--url", "https://docs.example.com/b"]
    )
    assert url_args.urls == ["https://docs.example.com/a", "https://docs.example.com/b"]
    assert url_args.crawl_url is None

    crawl_args = _parse_args(
        ["--crawl-url", "https://docs.example.com/", "--crawl-limit", "7", "--crawl-max-depth", "2"]
    )
    assert crawl_args.crawl_url == "https://docs.example.com/"
    assert crawl_args.crawl_limit == 7
    assert crawl_args.crawl_max_depth == 2

    with pytest.raises(SystemExit):
        _parse_args(
            ["--url", "https://docs.example.com/a", "--crawl-url", "https://docs.example.com/"]
        )
