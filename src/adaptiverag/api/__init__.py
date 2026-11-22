"""FastAPI HTTP and Server-Sent Events (SSE) streaming API."""

from adaptiverag.api.dependencies import AppContainer, create_app_container
from adaptiverag.api.main import app, create_app
from adaptiverag.api.schemas import (
    AblationPreset,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    InlineDocument,
    QueryRequest,
)

__all__ = [
    "AblationPreset",
    "AppContainer",
    "HealthResponse",
    "IngestRequest",
    "IngestResponse",
    "InlineDocument",
    "QueryRequest",
    "app",
    "create_app",
    "create_app_container",
]
