# ==============================================================================
# AdaptiveRAG Multi-Stage Production Dockerfile
# Stage 1: Build React 19 + TypeScript + Vite Glass-Box Studio SPA (same-origin)
# Stage 2: Minimal Python 3.12 + uv FastAPI runtime serving API + built SPA
# ==============================================================================

# --- Stage 1: Frontend SPA Builder ---
FROM node:22-alpine AS frontend-builder

WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./

# Empty string configures the built SPA to issue same-origin /v1/* requests
ARG VITE_API_URL=""
ENV VITE_API_URL=${VITE_API_URL}

RUN npm run build


# --- Stage 2: Python 3.12 FastAPI + ONNX Runtime ---
FROM python:3.12-slim AS runtime

# Install libgomp1 required by ONNX Runtime / FastEmbed on Debian slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:${PATH}" \
    STATIC_UI_DIR="/app/frontend/dist" \
    FASTEMBED_CACHE_PATH="/app/.adaptiverag_cache/fastembed" \
    LLM_CACHE_PATH="/app/.adaptiverag_cache/llm_cache.sqlite"

# 1. Install locked production dependencies first for optimal Docker layer caching
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# 2. Copy application source, strategy YAML, and built-in multilingual corpus
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/raw/ ./data/raw/
RUN uv sync --frozen --no-dev

# 3. Copy built React 19 Glass-Box Studio SPA from Stage 1
COPY --from=frontend-builder /frontend/dist /app/frontend/dist

# 4. Run as an unprivileged non-root system user
RUN addgroup --system adaptiverag \
    && adduser --system --ingroup adaptiverag --home /app adaptiverag \
    && mkdir -p /app/.adaptiverag_cache/fastembed \
    && chown -R adaptiverag:adaptiverag /app

USER adaptiverag

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/health', timeout=3)"]

CMD ["/app/.venv/bin/uvicorn", "adaptiverag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
