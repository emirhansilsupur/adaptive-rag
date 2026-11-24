"""Idempotent corpus and live web/docs ingestion CLI and programmatic indexing pipeline."""

from __future__ import annotations

import argparse
import asyncio
import time
from collections.abc import Sequence
from pathlib import Path

import structlog
from pydantic import BaseModel, Field

from adaptiverag.config import get_settings
from adaptiverag.ingestion.chunker import ChunkingConfig, chunk_documents
from adaptiverag.ingestion.loader import LoadedDocument, load_directory
from adaptiverag.ingestion.web_loader import FirecrawlClient
from adaptiverag.observability.tracing import configure_logging
from adaptiverag.retrieval.qdrant_store import QdrantHybridStore

logger = structlog.get_logger(__name__)


class IndexingReport(BaseModel):
    """Summary returned after indexing local files or web/docs pages into Qdrant."""

    data_dir: str
    collection_name: str
    documents_loaded: int = Field(ge=0)
    chunks_created: int = Field(ge=0)
    points_upserted: int = Field(ge=0)
    total_collection_points: int = Field(ge=0)
    collection_created: bool = False
    elapsed_seconds: float = Field(ge=0.0)
    source_urls: list[str] = Field(default_factory=list)
    parent_child_enabled: bool = False
    unique_parents_created: int = Field(default=0, ge=0)


async def index_documents(
    docs: Sequence[LoadedDocument],
    store: QdrantHybridStore,
    *,
    source: str,
    chunk_config: ChunkingConfig | None = None,
    recreate: bool = False,
    batch_size: int = 64,
    started_at: float | None = None,
) -> IndexingReport:
    """Structure-chunk, embed (Dense + BM25), and idempotently upsert LoadedDocuments into Qdrant."""
    started = started_at if started_at is not None else time.perf_counter()
    doc_list = list(docs)
    chunks = chunk_documents(doc_list, config=chunk_config)

    created = await store.ensure_collection(recreate=recreate)
    upserted = await store.upsert_chunks(
        chunks,
        batch_size=batch_size,
        replace_existing_docs=not recreate,
    )
    total_points = await store.count_points()
    elapsed = round(time.perf_counter() - started, 3)

    source_urls = [
        str(doc.metadata["source_url"])
        for doc in doc_list
        if isinstance(doc.metadata.get("source_url"), str) and doc.metadata["source_url"].strip()
    ]

    unique_parents = len({c.parent_id for c in chunks if c.parent_id})
    report = IndexingReport(
        data_dir=source,
        collection_name=store.collection_name,
        documents_loaded=len(doc_list),
        chunks_created=len(chunks),
        points_upserted=upserted,
        total_collection_points=total_points,
        collection_created=created,
        elapsed_seconds=elapsed,
        source_urls=source_urls,
        parent_child_enabled=bool(chunk_config.parent_child_enabled) if chunk_config else False,
        unique_parents_created=unique_parents,
    )
    logger.info("corpus_indexing_complete", **report.model_dump())
    return report


async def index_corpus(
    data_dir: Path,
    store: QdrantHybridStore,
    *,
    chunk_config: ChunkingConfig | None = None,
    recreate: bool = False,
    batch_size: int = 64,
) -> IndexingReport:
    """Load, structure-chunk, embed (Dense + BM25), and upsert a local corpus into Qdrant."""
    started = time.perf_counter()
    docs = load_directory(data_dir)
    return await index_documents(
        docs,
        store,
        source=str(data_dir),
        chunk_config=chunk_config,
        recreate=recreate,
        batch_size=batch_size,
        started_at=started,
    )


async def index_web_urls(
    urls: Sequence[str],
    store: QdrantHybridStore,
    client: FirecrawlClient,
    *,
    chunk_config: ChunkingConfig | None = None,
    recreate: bool = False,
    batch_size: int = 64,
    only_main_content: bool = True,
) -> IndexingReport:
    """Scrape one or more public URLs via Firecrawl Local and upsert chunks into Qdrant."""
    started = time.perf_counter()
    docs = await client.ascrape_urls(urls, only_main_content=only_main_content)
    return await index_documents(
        docs,
        store,
        source="firecrawl_scrape",
        chunk_config=chunk_config,
        recreate=recreate,
        batch_size=batch_size,
        started_at=started,
    )


async def index_web_crawl(
    crawl_url: str,
    store: QdrantHybridStore,
    client: FirecrawlClient,
    *,
    limit: int = 10,
    max_depth: int | None = None,
    chunk_config: ChunkingConfig | None = None,
    recreate: bool = False,
    batch_size: int = 64,
    only_main_content: bool = True,
) -> IndexingReport:
    """Crawl a live documentation/website root URL via Firecrawl Local and upsert into Qdrant."""
    started = time.perf_counter()
    docs = await client.acrawl_site(
        crawl_url,
        limit=limit,
        max_depth=max_depth,
        only_main_content=only_main_content,
    )
    return await index_documents(
        docs,
        store,
        source=f"firecrawl_crawl:{crawl_url}",
        chunk_config=chunk_config,
        recreate=recreate,
        batch_size=batch_size,
        started_at=started,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Index local Markdown/PDF/text files or live web/docs URLs (via firecrawl-local) "
            "into Qdrant with Dense + BM25 Hybrid vectors."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing local source documents (default: data/raw).",
    )
    parser.add_argument(
        "--url",
        action="append",
        dest="urls",
        default=None,
        help="Public HTTP/HTTPS URL to scrape via firecrawl-local (repeatable).",
    )
    parser.add_argument(
        "--crawl-url",
        type=str,
        default=None,
        help="Root documentation/website URL to crawl via firecrawl-local.",
    )
    parser.add_argument(
        "--crawl-limit",
        type=int,
        default=None,
        help="Maximum pages to crawl when --crawl-url is specified (default from settings).",
    )
    parser.add_argument(
        "--crawl-max-depth",
        type=int,
        default=None,
        help="Optional maximum link discovery depth for --crawl-url.",
    )
    parser.add_argument(
        "--firecrawl-url",
        type=str,
        default=None,
        help="Override FIRECRAWL_LOCAL_URL (default: http://localhost:3002).",
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
        "--parent-child",
        action="store_true",
        default=None,
        help="Enable hierarchical Parent-Child (Small-to-Big) chunking (defaults to ENABLE_PARENT_CHILD_RETRIEVAL).",
    )
    parser.add_argument(
        "--parent-max-tokens",
        type=int,
        default=None,
        help="Maximum tokens per parent passage when --parent-child is enabled (default: 1000).",
    )
    parser.add_argument(
        "--child-max-tokens",
        type=int,
        default=None,
        help="Maximum tokens per indexed child chunk when --parent-child is enabled (default: 256).",
    )
    parser.add_argument(
        "--child-overlap-tokens",
        type=int,
        default=None,
        help="Token overlap between sibling child chunks (default: 40).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Upsert batch size (default: 64).",
    )
    args = parser.parse_args(argv)
    if args.urls and args.crawl_url:
        parser.error("Provide either --url or --crawl-url, not both.")
    return args


async def _run_cli(args: argparse.Namespace) -> IndexingReport:
    settings = get_settings()
    configure_logging(settings.log_level)

    store = QdrantHybridStore.from_settings(settings, collection_name=args.collection)
    use_parent_child = (
        args.parent_child
        if args.parent_child is not None
        else settings.enable_parent_child_retrieval
    )
    chunk_cfg = ChunkingConfig(
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        parent_child_enabled=use_parent_child,
        parent_max_tokens=(
            args.parent_max_tokens
            if args.parent_max_tokens is not None
            else settings.parent_max_tokens
        ),
        child_max_tokens=(
            args.child_max_tokens
            if args.child_max_tokens is not None
            else settings.child_max_tokens
        ),
        child_overlap_tokens=(
            args.child_overlap_tokens
            if args.child_overlap_tokens is not None
            else settings.child_overlap_tokens
        ),
    )
    firecrawl_client: FirecrawlClient | None = None
    try:
        if args.urls or args.crawl_url:
            base_url = args.firecrawl_url or settings.firecrawl_local_url
            if not base_url:
                raise ValueError(
                    "FIRECRAWL_LOCAL_URL is disabled. Pass --firecrawl-url or set "
                    "FIRECRAWL_LOCAL_URL=http://localhost:3002."
                )
            api_key = (
                settings.firecrawl_api_key.get_secret_value().strip() or None
                if settings.firecrawl_api_key is not None
                else None
            )
            firecrawl_client = FirecrawlClient(
                base_url=base_url,
                api_key=api_key,
                timeout_seconds=settings.firecrawl_timeout_seconds,
                allow_private_urls=settings.firecrawl_allow_private_urls,
            )
            if args.urls:
                return await index_web_urls(
                    urls=args.urls,
                    store=store,
                    client=firecrawl_client,
                    chunk_config=chunk_cfg,
                    recreate=args.recreate,
                    batch_size=args.batch_size,
                )
            crawl_limit = (
                args.crawl_limit
                if args.crawl_limit is not None
                else settings.firecrawl_default_crawl_limit
            )
            return await index_web_crawl(
                crawl_url=args.crawl_url,
                store=store,
                client=firecrawl_client,
                limit=crawl_limit,
                max_depth=args.crawl_max_depth,
                chunk_config=chunk_cfg,
                recreate=args.recreate,
                batch_size=args.batch_size,
            )

        return await index_corpus(
            data_dir=args.data_dir,
            store=store,
            chunk_config=chunk_cfg,
            recreate=args.recreate,
            batch_size=args.batch_size,
        )
    finally:
        if firecrawl_client is not None:
            await firecrawl_client.aclose()
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
