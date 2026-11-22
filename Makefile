.PHONY: install up up-full down ingest api ui test test-unit lint typecheck ci eval

install:
	uv sync --extra ui --extra eval --group dev

up:
	docker compose up -d qdrant

up-full:
	docker compose --profile observability up -d

down:
	docker compose --profile observability down

ingest:
	uv run python -m adaptiverag.ingestion.indexer

api:
	uv run uvicorn adaptiverag.api.main:app --reload --host 0.0.0.0 --port 8000

ui:
	uv run streamlit run ui/app.py --server.port 8501

test:
	uv run pytest

test-unit:
	uv run pytest tests/unit

lint:
	uv run ruff check src tests eval ui
	uv run ruff format --check src tests eval ui

typecheck:
	uv run mypy src eval

ci: lint typecheck
	uv run python -m py_compile ui/app.py
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LANGFUSE_ENABLED=false uv run pytest -v --cov=adaptiverag --cov=eval --cov-report=term-missing

eval:
	uv run python -m eval.run_benchmark
