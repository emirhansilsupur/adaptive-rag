.PHONY: install up up-full down ingest api ui ui-install ui-build ui-test ui-legacy test test-unit lint typecheck ci eval eval-langfuse prompts-list prompts-sync

install:
	uv sync --extra ui --extra eval --group dev
	npm --prefix frontend install

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

ui-install:
	npm --prefix frontend install

ui:
	npm --prefix frontend run dev

ui-build:
	npm --prefix frontend run build

ui-test:
	npm --prefix frontend run typecheck
	npm --prefix frontend run test

ui-legacy:
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

ci: lint typecheck ui-test ui-build
	uv run python -m py_compile ui/app.py
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LANGFUSE_ENABLED=false uv run pytest -v --cov=adaptiverag --cov=eval --cov-report=term-missing

eval:
	uv run python -m eval.run_benchmark

eval-langfuse:
	uv run python -m eval.run_benchmark --mode offline-ir --in-memory --langfuse-sync

prompts-list:
	uv run python -m adaptiverag.observability.prompts --list

prompts-sync:
	uv run python -m adaptiverag.observability.prompts --seed
