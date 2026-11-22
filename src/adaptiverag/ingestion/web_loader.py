"""Async Firecrawl Local (v2 API) client for live URL & documentation site indexing.

Converts scraped/crawled web pages into AdaptiveRAG ``LoadedDocument`` instances
(Markdown + deterministic URL-based ``doc_id`` + source URL metadata) so they flow
directly through the existing Structure-Aware Chunker and Qdrant Hybrid Store.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import re
import socket
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
import structlog

from adaptiverag.config import Settings
from adaptiverag.ingestion.loader import LoadedDocument, detect_document_language

logger = structlog.get_logger(__name__)

_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})
_BLOCKED_HOSTNAMES: frozenset[str] = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "metadata.google.internal",
        "metadata",
    }
)
_BLOCKED_HOST_SUFFIXES: tuple[str, ...] = (
    ".localhost",
    ".local",
    ".internal",
    ".home.arpa",
)
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_H1_RE = re.compile(r"^#\s+(.+?)\s*$")
_NON_SLUG_RE = re.compile(r"[^a-z0-9]+")
_MAX_CRAWL_PAGINATION_HOPS = 50


class UnsafeURLError(ValueError):
    """Raised when a user-supplied URL fails scheme or SSRF safety checks."""


class FirecrawlError(RuntimeError):
    """Raised when Firecrawl Local returns an API error, timeout, or empty result."""


class FirecrawlUnavailableError(FirecrawlError):
    """Raised when the configured firecrawl-local server cannot be reached."""


def _is_disallowed_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if an IP address is non-global, private, loopback, or metadata-scoped."""
    if (
        isinstance(ip, ipaddress.IPv6Address)
        and ip.ipv4_mapped is not None
        and _is_disallowed_ip(ip.ipv4_mapped)
    ):
        return True
    return bool(
        not ip.is_global
        or ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_public_url(
    url: str,
    *,
    allow_private: bool = False,
    resolve_dns: bool = False,
) -> str:
    """Validate and normalize an HTTP/HTTPS URL with SSRF protection.

    Blocks non-HTTP(S) schemes, embedded credentials, localhost/internal domains,
    and private/loopback/link-local/metadata IP literals (plus resolved DNS IPs
    when ``resolve_dns=True`` and DNS is reachable).
    """
    cleaned = url.strip()
    if not cleaned:
        raise UnsafeURLError("URL must not be empty.")

    parsed = urlsplit(cleaned)
    scheme = parsed.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise UnsafeURLError(
            f"Unsupported URL scheme '{parsed.scheme or '<missing>'}'. "
            "Only 'http' and 'https' URLs are permitted."
        )

    if parsed.username is not None or parsed.password is not None:
        raise UnsafeURLError("URLs with embedded credentials are not permitted.")

    hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        raise UnsafeURLError(f"URL is missing a valid hostname: '{cleaned}'")

    if not allow_private:
        if hostname in _BLOCKED_HOSTNAMES or any(
            hostname.endswith(suffix) for suffix in _BLOCKED_HOST_SUFFIXES
        ):
            raise UnsafeURLError(
                f"Blocked private or internal hostname '{hostname}' (SSRF protection)."
            )

        try:
            ip_literal = ipaddress.ip_address(hostname)
        except ValueError:
            ip_literal = None

        if ip_literal is not None:
            if _is_disallowed_ip(ip_literal):
                raise UnsafeURLError(
                    f"Blocked non-public IP address '{ip_literal}' (SSRF protection)."
                )
        elif resolve_dns:
            try:
                addr_infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
            except OSError:
                # Graceful pass-through when offline / DNS unavailable locally;
                # Firecrawl itself will report host resolution failure if unreachable.
                addr_infos = []
            for info in addr_infos:
                raw_ip = str(info[4][0]).split("%", 1)[0]
                try:
                    resolved_ip = ipaddress.ip_address(raw_ip)
                except ValueError:
                    continue
                if _is_disallowed_ip(resolved_ip):
                    raise UnsafeURLError(
                        f"Hostname '{hostname}' resolved to non-public IP '{resolved_ip}' "
                        "(SSRF protection)."
                    )

    # Strip fragment so identical pages map to one canonical URL & doc_id.
    normalized = urlunsplit((scheme, parsed.netloc, parsed.path or "/", parsed.query, ""))
    return normalized


def make_web_doc_id(url: str) -> str:
    """Create a deterministic, readable, collision-resistant doc_id for a web URL."""
    parsed = urlsplit(url.strip())
    host = (parsed.hostname or "web").lower()
    path = parsed.path.rstrip("/") or ""
    canonical = f"{host}{path}"
    if parsed.query:
        canonical = f"{canonical}?{parsed.query}"

    slug = _NON_SLUG_RE.sub("_", f"{host}{path}".lower()).strip("_")[:48].rstrip("_")
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]
    prefix = slug or "page"
    return f"web_{prefix}_{digest}"


def _extract_h1_heading(markdown: str) -> str | None:
    """Return the first Markdown H1 heading outside fenced code blocks, if present."""
    in_fence = False
    fence_char = ""
    fence_len = 0

    for line in markdown.splitlines():
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if not in_fence:
                in_fence = True
                fence_char = marker[0]
                fence_len = len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_len:
                in_fence = False
            continue
        if not in_fence:
            h1 = _H1_RE.match(line)
            if h1:
                heading = h1.group(1).strip()
                if heading:
                    return heading
    return None


def _prettify_url_title(url: str) -> str:
    """Fallback human-readable title derived from URL path or hostname."""
    parsed = urlsplit(url)
    segments = [seg for seg in parsed.path.split("/") if seg]
    if segments:
        tail = segments[-1]
        tail = re.sub(r"\.[a-zA-Z0-9]+$", "", tail)
        pretty = tail.replace("-", " ").replace("_", " ").strip()
        if pretty:
            return pretty.title()
    return (parsed.hostname or url).strip()


def _detect_web_language(markdown: str, source_url: str, metadata: dict[str, Any]) -> str:
    """Resolve document language (tr / en) from page metadata and Markdown diacritics."""
    raw_lang = metadata.get("language")
    if isinstance(raw_lang, str):
        norm = raw_lang.strip().lower()
        if norm == "tr" or norm.startswith(("tr-", "tr_")):
            return "tr"
    return detect_document_language(markdown, source_url)


def _page_to_loaded_document(
    page: dict[str, Any],
    *,
    fallback_url: str,
    crawled_at: str,
    crawl_root_url: str | None = None,
) -> LoadedDocument | None:
    """Convert a single Firecrawl v2 page payload into a LoadedDocument."""
    markdown = str(page.get("markdown") or "").strip()
    if not markdown:
        return None

    raw_meta = page.get("metadata")
    meta: dict[str, Any] = dict(raw_meta) if isinstance(raw_meta, dict) else {}

    # Prefer the final post-redirect canonical URL (`metadata.url`) over the
    # pre-redirect `sourceURL` so redirected docs pages deduplicate cleanly.
    source_url = str(
        meta.get("url") or meta.get("sourceURL") or page.get("url") or fallback_url
    ).strip()
    if not source_url:
        source_url = fallback_url

    raw_title = meta.get("title")
    title = (
        str(raw_title).strip()
        if isinstance(raw_title, str) and raw_title.strip()
        else (_extract_h1_heading(markdown) or _prettify_url_title(source_url))
    )

    language = _detect_web_language(markdown, source_url, meta)

    doc_metadata: dict[str, Any] = {
        "source_url": source_url,
        "source_origin": "firecrawl_local",
        "crawled_at": crawled_at,
    }
    if crawl_root_url:
        doc_metadata["crawl_root_url"] = crawl_root_url
    raw_desc = meta.get("description")
    if isinstance(raw_desc, str) and raw_desc.strip():
        doc_metadata["description"] = raw_desc.strip()
    raw_status = meta.get("statusCode")
    if isinstance(raw_status, int):
        doc_metadata["status_code"] = raw_status

    return LoadedDocument(
        doc_id=make_web_doc_id(source_url),
        title=title,
        content=markdown,
        source_path=source_url,
        doc_type="markdown",
        language=language,
        metadata=doc_metadata,
    )


def _extract_relative_api_path(url_or_path: str) -> str:
    """Extract a safe relative path + query string to call against our configured base_url."""
    parsed = urlsplit(url_or_path.strip())
    path = parsed.path or "/"
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{path}?{parsed.query}" if parsed.query else path


class FirecrawlClient:
    """Async HTTP client targeting Firecrawl Local (v2 API: /v2/scrape and /v2/crawl)."""

    def __init__(
        self,
        base_url: str = "http://localhost:3002",
        *,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        allow_private_urls: bool = False,
        resolve_dns: bool = True,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        cleaned_base = base_url.strip().rstrip("/")
        if not cleaned_base:
            raise ValueError("FirecrawlClient base_url must not be empty.")
        self._base_url = cleaned_base
        self._timeout_seconds = float(timeout_seconds)
        self._allow_private_urls = allow_private_urls
        self._resolve_dns = resolve_dns

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(self._timeout_seconds, connect=min(10.0, self._timeout_seconds)),
            transport=transport,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> FirecrawlClient | None:
        """Construct a FirecrawlClient from runtime Settings, or None when disabled."""
        if not settings.firecrawl_local_url:
            return None
        api_key = (
            settings.firecrawl_api_key.get_secret_value().strip() or None
            if settings.firecrawl_api_key is not None
            else None
        )
        return cls(
            base_url=settings.firecrawl_local_url,
            api_key=api_key,
            timeout_seconds=settings.firecrawl_timeout_seconds,
            allow_private_urls=settings.firecrawl_allow_private_urls,
        )

    @property
    def base_url(self) -> str:
        return self._base_url

    async def aclose(self) -> None:
        """Close the underlying async HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> FirecrawlClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def ahealth_check(self, *, timeout_seconds: float = 1.5) -> dict[str, Any]:
        """Non-crashing reachability check for the configured firecrawl-local instance."""
        try:
            resp = await self._client.get("/health", timeout=timeout_seconds)
            # Minimal local servers may return 404 on /health while still serving /v2/*.
            if resp.status_code < 500:
                return {
                    "status": "ok",
                    "base_url": self._base_url,
                    "http_status": resp.status_code,
                }
            return {
                "status": "degraded",
                "base_url": self._base_url,
                "http_status": resp.status_code,
            }
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
            return {
                "status": "unreachable",
                "base_url": self._base_url,
                "error": str(exc),
            }
        except Exception as exc:
            return {
                "status": "unreachable",
                "base_url": self._base_url,
                "error": str(exc),
            }

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute an HTTP request against Firecrawl and map errors cleanly."""
        try:
            response = await self._client.request(method, path, json=json_body)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise FirecrawlUnavailableError(
                f"Cannot connect to Firecrawl Local at '{self._base_url}'. "
                "Start it with `firecrawl-local serve` (install via `cargo install firecrawl-local`)."
            ) from exc
        except httpx.TimeoutException as exc:
            raise FirecrawlError(
                f"Firecrawl request '{method} {path}' timed out after {self._timeout_seconds:.0f}s."
            ) from exc
        except httpx.HTTPError as exc:
            raise FirecrawlError(
                f"Firecrawl HTTP transport error on '{method} {path}': {exc}"
            ) from exc

        try:
            payload: Any = response.json()
        except ValueError:
            payload = {"error": response.text.strip()[:400]}

        if response.status_code >= 400:
            detail = (
                payload.get("error") or payload.get("message") or response.text.strip()[:400]
                if isinstance(payload, dict)
                else str(payload)[:400]
            )
            raise FirecrawlError(
                f"Firecrawl '{method} {path}' failed with HTTP {response.status_code}: {detail}"
            )

        if not isinstance(payload, dict):
            raise FirecrawlError(
                f"Unexpected non-object JSON payload from Firecrawl '{method} {path}'."
            )
        if payload.get("success") is False:
            detail = str(payload.get("error") or payload.get("message") or "Unknown error")
            raise FirecrawlError(f"Firecrawl reported failure on '{method} {path}': {detail}")

        return payload

    async def ascrape_url(
        self,
        url: str,
        *,
        only_main_content: bool = True,
    ) -> LoadedDocument:
        """Scrape a single public URL into clean Markdown via POST /v2/scrape."""
        safe_url = validate_public_url(
            url,
            allow_private=self._allow_private_urls,
            resolve_dns=self._resolve_dns,
        )
        started = time.perf_counter()
        payload = await self._request_json(
            "POST",
            "/v2/scrape",
            json_body={
                "url": safe_url,
                "formats": ["markdown"],
                "onlyMainContent": only_main_content,
            },
        )
        raw_data = payload.get("data")
        if not isinstance(raw_data, dict):
            raise FirecrawlError(f"Firecrawl /v2/scrape returned missing 'data' for {safe_url}.")

        crawled_at = datetime.now(tz=UTC).isoformat()
        doc = _page_to_loaded_document(
            raw_data,
            fallback_url=safe_url,
            crawled_at=crawled_at,
        )
        if doc is None:
            raise FirecrawlError(f"Firecrawl returned empty markdown content for {safe_url}.")

        logger.info(
            "firecrawl_scrape_complete",
            url=safe_url,
            doc_id=doc.doc_id,
            language=doc.language,
            chars=len(doc.content),
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )
        return doc

    async def ascrape_urls(
        self,
        urls: Sequence[str],
        *,
        only_main_content: bool = True,
        concurrency: int = 3,
    ) -> list[LoadedDocument]:
        """Validate, deduplicate, and concurrently scrape a list of public URLs."""
        if not urls:
            return []

        # Fail fast on any unsafe URL before dispatching any HTTP requests.
        normalized_unique: list[str] = list(
            dict.fromkeys(
                validate_public_url(
                    u,
                    allow_private=self._allow_private_urls,
                    resolve_dns=self._resolve_dns,
                )
                for u in urls
            )
        )

        sem = asyncio.Semaphore(max(1, concurrency))

        async def _guarded_scrape(target_url: str) -> LoadedDocument:
            async with sem:
                return await self.ascrape_url(target_url, only_main_content=only_main_content)

        return list(await asyncio.gather(*(_guarded_scrape(u) for u in normalized_unique)))

    async def _collect_paginated_pages(
        self,
        initial_payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Collect all scraped page objects across optional Firecrawl 'next' pagination links."""
        pages: list[dict[str, Any]] = []
        raw_data = initial_payload.get("data")
        if isinstance(raw_data, list):
            pages.extend(item for item in raw_data if isinstance(item, dict))

        next_ref = initial_payload.get("next")
        hops = 0
        seen_paths: set[str] = set()
        while isinstance(next_ref, str) and next_ref.strip() and hops < _MAX_CRAWL_PAGINATION_HOPS:
            rel_path = _extract_relative_api_path(next_ref)
            if rel_path in seen_paths:
                break
            seen_paths.add(rel_path)
            hops += 1
            page_payload = await self._request_json("GET", rel_path)
            batch = page_payload.get("data")
            if isinstance(batch, list):
                pages.extend(item for item in batch if isinstance(item, dict))
            next_ref = page_payload.get("next")

        return pages

    async def acrawl_site(
        self,
        url: str,
        *,
        limit: int = 10,
        max_depth: int | None = None,
        only_main_content: bool = True,
        poll_interval_seconds: float = 1.0,
        max_wait_seconds: float | None = None,
    ) -> list[LoadedDocument]:
        """Crawl a documentation/website root URL via POST /v2/crawl and poll until complete."""
        safe_url = validate_public_url(
            url,
            allow_private=self._allow_private_urls,
            resolve_dns=self._resolve_dns,
        )
        started = time.perf_counter()

        body: dict[str, Any] = {
            "url": safe_url,
            "limit": limit,
            "scrapeOptions": {
                "formats": ["markdown"],
                "onlyMainContent": only_main_content,
            },
        }
        if max_depth is not None:
            body["maxDiscoveryDepth"] = max_depth

        init_payload = await self._request_json("POST", "/v2/crawl", json_body=body)

        # Support both async job initiation ({id, url}) and immediate completed payloads.
        final_payload = init_payload
        if not (
            init_payload.get("status") == "completed" and isinstance(init_payload.get("data"), list)
        ):
            job_id = str(init_payload.get("id") or "").strip()
            poll_url = str(init_payload.get("url") or "").strip()
            if not job_id and not poll_url:
                raise FirecrawlError(
                    f"Firecrawl /v2/crawl did not return a job 'id' or 'url' for {safe_url}."
                )
            status_path = (
                _extract_relative_api_path(poll_url) if poll_url else f"/v2/crawl/{job_id}"
            )
            deadline = time.monotonic() + (
                max_wait_seconds
                if max_wait_seconds is not None
                else max(self._timeout_seconds * 2.0, 120.0)
            )

            while True:
                status_payload = await self._request_json("GET", status_path)
                job_status = str(status_payload.get("status") or "").lower()
                if job_status == "completed":
                    final_payload = status_payload
                    break
                if job_status in {"failed", "cancelled"}:
                    err_detail = str(
                        status_payload.get("error")
                        or status_payload.get("message")
                        or f"crawl {job_status}"
                    )
                    raise FirecrawlError(
                        f"Firecrawl crawl job '{job_id or status_path}' {job_status} for "
                        f"{safe_url}: {err_detail}"
                    )
                if time.monotonic() >= deadline:
                    raise FirecrawlError(
                        f"Firecrawl crawl job '{job_id or status_path}' timed out waiting for "
                        f"completion on {safe_url}."
                    )
                await asyncio.sleep(max(0.0, poll_interval_seconds))

        raw_pages = await self._collect_paginated_pages(final_payload)
        crawled_at = datetime.now(tz=UTC).isoformat()

        docs: list[LoadedDocument] = []
        seen_doc_ids: set[str] = set()
        for page in raw_pages:
            doc = _page_to_loaded_document(
                page,
                fallback_url=safe_url,
                crawled_at=crawled_at,
                crawl_root_url=safe_url,
            )
            if doc is None or doc.doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc.doc_id)
            docs.append(doc)
            if len(docs) >= limit:
                break

        if not docs:
            raise FirecrawlError(
                f"Firecrawl crawl completed for {safe_url} but returned zero usable Markdown pages."
            )

        logger.info(
            "firecrawl_crawl_complete",
            crawl_url=safe_url,
            pages_indexed=len(docs),
            limit=limit,
            max_depth=max_depth,
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )
        return docs
