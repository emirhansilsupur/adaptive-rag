"""Idempotent corpus ingestion CLI and programmatic indexing pipeline."""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import structlog
from pydantic import BaseModel, Field

from adaptiverag.config import get_settings
from adaptiverag.ingestion.chunker import ChunkingConfig, chunk_documents
from adaptiverag.ingestion.loader import load_directory
from adaptiverag.observability.tracing import configure_logging
from adaptiverag.retrieval.qdrant_store import QdrantHybridStore

logger = structlog.get_logger(__name__)


class IndexingReport(BaseModel):
    """Summary returned after indexing a document directory into Qdrant."""

    data_dir: str
    collection_name: str
    documents_loaded: int = Field(ge=0)
    chunks_created: int = Field(ge=0)
    points_upserted: int = Field(ge=0)
    total_collection_points: int = Field(ge=0)
    collection_created: bool = False
    elapsed_seconds: float = Field(ge=0.0)


async def index_corpus(
    data_dir: Path,
    store: QdrantHybridStore,
    *,
    chunk_config: ChunkingConfig | None = None,
    recreate: bool = False,
    batch_size: int = 64,
) -> IndexingReport:
    """Load, structure-chunk, embed (Dense + BM25), and upsert a corpus into Qdrant."""
    started = time.perf_counter()

    docs = load_directory(data_dir)
    chunks = chunk_documents(docs, config=chunk_config)

    created = await store.ensure_collection(recreate=recreate)
    upserted = await store.upsert_chunks(
        chunks,
        batch_size=batch_size,
        replace_existing_docs=not recreate,
    )
    total_points = await store.count_points()
    elapsed = round(time.perf_counter() - started, 3)

    report = IndexingReport(
        data_dir=str(data_dir),
        collection_name=store.collection_name,
        documents_loaded=len(docs),
        chunks_created=len(chunks),
        points_upserted=upserted,
        total_collection_points=total_points,
        collection_created=created,
        elapsed_seconds=elapsed,
    )
    logger.info("corpus_indexing_complete", **report.model_dump())
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index Markdown/PDF/text documents into Qdrant with Dense + BM25 Hybrid vectors."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing source documents (default: data/raw).",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Override target Qdrant collection name.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the Qdrant collection before indexing.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per chunk (default: 512).",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=64,
        help="Token overlap between consecutive chunks in the same section (default: 64).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Upsert batch size (default: 64).",
    )
    return parser.parse_args(argv)


async def _run_cli(args: argparse.Namespace) -> IndexingReport:
    settings = get_settings()
    configure_logging(settings.log_level)

    store = QdrantHybridStore.from_settings(settings, collection_name=args.collection)
    chunk_cfg = ChunkingConfig(
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    try:
        return await index_corpus(
            data_dir=args.data_dir,
            store=store,
            chunk_config=chunk_cfg,
            recreate=args.recreate,
            batch_size=args.batch_size,
        )
    finally:
        await store.close()


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: ``python -m adaptiverag.ingestion.indexer``."""
    args = _parse_args(argv)
    report = asyncio.run(_run_cli(args))
    print(
        f"Indexed {report.documents_loaded} documents -> {report.chunks_created} chunks "
        f"into Qdrant collection '{report.collection_name}' "
        f"(total points: {report.total_collection_points}, {report.elapsed_seconds:.2f}s)."
    )


if __name__ == "__main__":
    main()
