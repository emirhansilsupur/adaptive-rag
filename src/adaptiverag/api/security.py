"""API security middleware: local-only guard, optional API key auth & per-IP rate limiting.

Implemented as pure ASGI middleware (rather than Starlette's ``BaseHTTPMiddleware``) so
Server-Sent Events (SSE) streams on ``/v1/query/stream`` pass through with zero body buffering.
"""

from __future__ import annotations

import hmac
import ipaddress
import math
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from threading import Lock
from urllib.parse import urlsplit

import structlog
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from adaptiverag.config import Settings

logger = structlog.get_logger(__name__)

# Versioned API endpoints that remain public for probes / metadata even when auth is enabled.
PUBLIC_API_PATHS: frozenset[str] = frozenset({"/v1/health"})


# Studio's Vite dev server. Used when CORS_ALLOW_ORIGINS is empty; never widened to "*".
DEFAULT_CORS_ORIGINS: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")

_DEFAULT_PORTS: dict[str, int] = {"http": 80, "https": 443}


def parse_cors_origins(raw: str) -> list[str]:
    """Parse comma-separated CORS origins; an empty value means the local Studio dev origins.

    Allowing every origin must be an explicit ``*``: an empty or missing setting never does it.
    Trailing slashes are dropped because browsers send ``Origin`` without one.
    """
    origins = [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]
    return origins or list(DEFAULT_CORS_ORIGINS)


def parse_host_list(raw: str) -> frozenset[str]:
    """Parse comma-separated host names into a lowercase set (``*`` is kept as a wildcard)."""
    return frozenset(item.strip().lower().rstrip(".") for item in raw.split(",") if item.strip())


def api_auth_enabled(settings: Settings) -> bool:
    """True when ``API_AUTH_KEY`` holds a non-blank key."""
    key = settings.api_auth_key
    return key is not None and bool(key.get_secret_value().strip())


def local_only_mode(settings: Settings) -> bool:
    """True when the API has no key and answers only loopback (plus explicitly listed) hosts."""
    return not api_auth_enabled(settings) and "*" not in parse_host_list(
        settings.allowed_hosts_without_auth
    )


def split_host_port(value: str) -> tuple[str, int | None]:
    """Split a Host header (``name``, ``name:port`` or ``[v6]:port``) into host and port."""
    value = value.strip().lower()
    port_text = ""
    if value.startswith("["):
        end = value.find("]")
        host = value[1:end] if end != -1 else value
        rest = value[end + 1 :] if end != -1 else ""
        port_text = rest[1:] if rest.startswith(":") else ""
    elif value.count(":") == 1:
        host, _, port_text = value.partition(":")
    else:
        host = value  # bare name, or an unbracketed IPv6 literal without a port
    port = int(port_text) if port_text.isascii() and port_text.isdigit() else None
    return host.rstrip("."), port


def is_loopback_host(host: str) -> bool:
    """True for ``localhost``, ``*.localhost`` and loopback IP literals (127.0.0.0/8, ::1)."""
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _origin_matches_host(origin: str, host_header: str) -> bool:
    """True when a browser ``Origin`` names the same host (and port) the request was sent to.

    A Host header without a port only occurs on default ports, typically behind a TLS proxy,
    so the port is compared only when the Host header states one.
    """
    try:
        parts = urlsplit(origin)
        origin_port = parts.port or _DEFAULT_PORTS.get(parts.scheme)
    except ValueError:
        return False
    if parts.scheme not in _DEFAULT_PORTS or not parts.hostname:
        return False
    host, port = split_host_port(host_header)
    return parts.hostname.rstrip(".") == host and (port is None or port == origin_port)


@dataclass(frozen=True)
class RateLimitDecision:
    """Result of evaluating a single request against the sliding-window rate limiter."""

    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: int
    reset_epoch_seconds: int


class SlidingWindowRateLimiter:
    """Thread-safe in-memory sliding-window rate limiter keyed by client IP."""

    def __init__(
        self,
        max_requests: int = 60,
        window_seconds: float = 60.0,
        *,
        max_tracked_clients: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        if max_requests < 1:
            raise ValueError("max_requests must be >= 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self.max_requests = max_requests
        self.window_seconds = float(window_seconds)
        self.max_tracked_clients = max_tracked_clients
        self._clock = clock
        self._wall_clock = wall_clock
        self._buckets: dict[str, deque[float]] = {}
        self._lock = Lock()

    def _evict_expired_buckets(self, cutoff: float) -> None:
        stale_keys = [
            key
            for key, timestamps in self._buckets.items()
            if not timestamps or timestamps[-1] <= cutoff
        ]
        for key in stale_keys:
            self._buckets.pop(key, None)

        # Hard memory cap fallback under randomized IP floods
        if len(self._buckets) >= self.max_tracked_clients:
            oldest_key = min(
                self._buckets,
                key=lambda k: self._buckets[k][-1] if self._buckets[k] else 0.0,
            )
            self._buckets.pop(oldest_key, None)

    def check(self, client_key: str) -> RateLimitDecision:
        """Record a hit for ``client_key`` if under limit and return quota telemetry."""
        now = self._clock()
        now_wall = self._wall_clock()
        cutoff = now - self.window_seconds

        with self._lock:
            bucket = self._buckets.get(client_key)
            if bucket is None:
                if len(self._buckets) >= self.max_tracked_clients:
                    self._evict_expired_buckets(cutoff)
                bucket = deque()
                self._buckets[client_key] = bucket

            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

            if len(bucket) >= self.max_requests:
                oldest = bucket[0]
                retry_after = max(1, math.ceil(self.window_seconds - (now - oldest)))
                reset_epoch = int(now_wall + retry_after)
                return RateLimitDecision(
                    allowed=False,
                    limit=self.max_requests,
                    remaining=0,
                    retry_after_seconds=retry_after,
                    reset_epoch_seconds=reset_epoch,
                )

            bucket.append(now)
            remaining = max(0, self.max_requests - len(bucket))
            oldest = bucket[0]
            reset_in = max(1, math.ceil(self.window_seconds - (now - oldest)))
            return RateLimitDecision(
                allowed=True,
                limit=self.max_requests,
                remaining=remaining,
                retry_after_seconds=0,
                reset_epoch_seconds=int(now_wall + reset_in),
            )


def _extract_header(scope: Scope, target_name: bytes) -> str | None:
    for name, value in scope.get("headers", ()):
        if name.lower() == target_name:
            decoded: str = value.decode("latin-1")
            return decoded.strip()
    return None


def _extract_client_ip(scope: Scope, *, trust_proxy_headers: bool) -> str:
    if trust_proxy_headers:
        forwarded_for = _extract_header(scope, b"x-forwarded-for")
        if forwarded_for:
            first_ip = forwarded_for.split(",", 1)[0].strip()
            if first_ip:
                return first_ip
        real_ip = _extract_header(scope, b"x-real-ip")
        if real_ip:
            return real_ip

    client = scope.get("client")
    if client and len(client) >= 1 and client[0]:
        return str(client[0])
    return "unknown"


def _extract_provided_api_key(scope: Scope) -> str | None:
    x_api_key = _extract_header(scope, b"x-api-key")
    if x_api_key:
        return x_api_key

    auth_header = _extract_header(scope, b"authorization")
    if auth_header:
        scheme, _, token = auth_header.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            return token.strip()
    return None


class ApiSecurityMiddleware:
    """Pure ASGI middleware enforcing optional API key auth & per-IP rate limiting on ``/v1/*``."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        api_key: str | None = None,
        rate_limiter: SlidingWindowRateLimiter | None = None,
        trust_proxy_headers: bool = False,
        public_paths: frozenset[str] = PUBLIC_API_PATHS,
    ) -> None:
        self.app = app
        cleaned_key = (api_key or "").strip()
        self._expected_key_bytes: bytes | None = (
            cleaned_key.encode("utf-8") if cleaned_key else None
        )
        self._rate_limiter = rate_limiter
        self._trust_proxy_headers = trust_proxy_headers
        self._public_paths = public_paths

    def _is_protected_path(self, path: str) -> bool:
        if not path.startswith("/v1/"):
            return False
        return path not in self._public_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method", "GET")).upper()
        path = str(scope.get("path", ""))

        # Never block CORS preflight requests or non-protected/public paths.
        if method == "OPTIONS" or not self._is_protected_path(path):
            await self.app(scope, receive, send)
            return

        # 1. Optional constant-time API Key verification
        if self._expected_key_bytes is not None:
            provided = _extract_provided_api_key(scope)
            if not provided or not hmac.compare_digest(
                provided.encode("utf-8"),
                self._expected_key_bytes,
            ):
                client_ip = _extract_client_ip(scope, trust_proxy_headers=self._trust_proxy_headers)
                logger.warning("api_auth_rejected", path=path, client_ip=client_ip)
                response = JSONResponse(
                    status_code=401,
                    content={
                        "detail": (
                            "Invalid or missing API key. "
                            "Provide X-API-Key or Authorization: Bearer <key>."
                        )
                    },
                    headers={"WWW-Authenticate": "Bearer"},
                )
                await response(scope, receive, send)
                return

        # 2. Optional per-IP sliding-window rate limiting
        if self._rate_limiter is None:
            await self.app(scope, receive, send)
            return

        client_ip = _extract_client_ip(scope, trust_proxy_headers=self._trust_proxy_headers)
        decision = self._rate_limiter.check(client_ip)
        rate_headers = [
            (b"x-ratelimit-limit", str(decision.limit).encode("ascii")),
            (b"x-ratelimit-remaining", str(decision.remaining).encode("ascii")),
            (b"x-ratelimit-reset", str(decision.reset_epoch_seconds).encode("ascii")),
        ]

        if not decision.allowed:
            logger.warning(
                "api_rate_limit_exceeded",
                path=path,
                client_ip=client_ip,
                limit=decision.limit,
                retry_after=decision.retry_after_seconds,
            )
            response = JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        f"Rate limit exceeded ({decision.limit} requests/min). "
                        f"Retry after {decision.retry_after_seconds} seconds."
                    )
                },
                headers={
                    "Retry-After": str(decision.retry_after_seconds),
                    "X-RateLimit-Limit": str(decision.limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(decision.reset_epoch_seconds),
                },
            )
            await response(scope, receive, send)
            return

        async def send_with_rate_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                existing = list(message.get("headers", []))
                existing.extend(rate_headers)
                message = {**message, "headers": existing}
            await send(message)

        await self.app(scope, receive, send_with_rate_headers)


class LocalOnlyGuardMiddleware:
    """Pure ASGI guard for an API running without ``API_AUTH_KEY``.

    CORS only stops a browser from reading responses. It does not stop the request itself,
    and browsers send multipart or form POSTs to another origin without a preflight. So a page
    on any site could upload to, or wipe, a local API that has no key. This guard rejects:

    1. Requests addressed to a host other than a loopback name or one of ``extra_hosts``. That
       blocks LAN clients and DNS-rebinding pages, whose Host header is the attacker's name.
    2. Browser requests whose ``Origin`` is neither allowed for CORS nor this API itself.

    Clients without a Host or Origin header (curl, SDKs, server-side code) are not browsers
    and pass the matching check. ``*`` in either list turns that check off.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        allowed_origins: Iterable[str],
        extra_hosts: Iterable[str] = (),
    ) -> None:
        self.app = app
        self._allowed_origins = frozenset(origin.rstrip("/") for origin in allowed_origins)
        self._any_origin = "*" in self._allowed_origins
        self._extra_hosts = frozenset(extra_hosts)
        self._any_host = "*" in self._extra_hosts

    def _host_allowed(self, host_header: str) -> bool:
        host, _ = split_host_port(host_header)
        return is_loopback_host(host) or host in self._extra_hosts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path", ""))
        host_header = _extract_header(scope, b"host")
        if host_header and not self._any_host and not self._host_allowed(host_header):
            logger.warning("local_only_host_rejected", host=host_header, path=path)
            response = JSONResponse(
                status_code=403,
                content={
                    "detail": (
                        "This API answers only local requests because API_AUTH_KEY is not set. "
                        "Address it as localhost, 127.0.0.1 or [::1], add this host to "
                        "ALLOWED_HOSTS_WITHOUT_AUTH, or set API_AUTH_KEY."
                    )
                },
            )
            await response(scope, receive, send)
            return

        origin = _extract_header(scope, b"origin")
        if (
            origin
            and not self._any_origin
            and origin.rstrip("/") not in self._allowed_origins
            and not _origin_matches_host(origin, host_header or "")
        ):
            logger.warning("local_only_origin_rejected", origin=origin, path=path)
            response = JSONResponse(
                status_code=403,
                content={
                    "detail": (
                        f"Browser requests from {origin} are not allowed. "
                        "Add the origin to CORS_ALLOW_ORIGINS."
                    )
                },
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
