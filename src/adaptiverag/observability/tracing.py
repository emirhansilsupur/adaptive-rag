"""Structured logging (structlog) and optional Langfuse tracing setup."""

from __future__ import annotations

import logging
from typing import Any

import structlog

from adaptiverag.config import Settings


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structured console logging via structlog."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        cache_logger_on_first_use=True,
    )


def get_langfuse_callback(
    settings: Settings,
) -> Any | None:
    """Return a Langfuse v3+/v5 LangChain CallbackHandler when enabled, else None."""
    if not settings.langfuse_enabled:
        return None
    if not settings.langfuse_public_key or settings.langfuse_secret_key is None:
        return None

    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler
    except ImportError:
        return None

    # Initialize the singleton Langfuse client with explicit credentials/host.
    Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        host=settings.langfuse_host,
    )
    return CallbackHandler(public_key=settings.langfuse_public_key)
