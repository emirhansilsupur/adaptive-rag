"""Document loaders for Markdown, plain text, and PDF technical corpora."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from adaptiverag.config import Settings
from adaptiverag.ingestion.pdf_parser import (
    PdfOcrMode,
    PdfParseConfig,
    VisionOcrFn,
    aparse_pdf_bytes,
    parse_pdf_bytes,
)
from adaptiverag.llm.cost import CostTracker

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".md", ".markdown", ".txt", ".pdf"})

_TURKISH_DIACRITICS = frozenset("ğĞşŞıİöÖüÜçÇ")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_H1_RE = re.compile(r"^#\s+(.+?)\s*$")
_NON_SLUG_RE = re.compile(r"[^a-z0-9]+")


class LoadedDocument(BaseModel):
    """Normalized raw document ready for structure-aware chunking."""

    doc_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source_path: str
    doc_type: Literal["markdown", "pdf", "text"]
    language: str = Field(default="en", description="Detected document language (en / tr).")
    metadata: dict[str, Any] = Field(default_factory=dict)


def _slugify_doc_id(path: Path, base_dir: Path | None = None) -> str:
    """Create a deterministic, filesystem-independent document identifier."""
    if base_dir is not None:
        try:
            rel = path.resolve().relative_to(base_dir.resolve())
            raw = rel.with_suffix("").as_posix()
        except ValueError:
            raw = path.stem
    else:
        raw = path.stem
    slug = _NON_SLUG_RE.sub("_", raw.lower()).strip("_")
    return slug or "document"


def _prettify_filename(path: Path) -> str:
    """Fallback human-readable title derived from filename."""
    cleaned = re.sub(r"^\d+[_-]+", "", path.stem)
    return cleaned.replace("_", " ").replace("-", " ").strip().title() or path.stem


def _extract_markdown_title(text: str, fallback_path: Path) -> str:
    """Return the first top-level H1 heading outside fenced code blocks."""
    in_fence = False
    fence_char = ""
    fence_len = 0

    for line in text.splitlines():
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
                return h1.group(1).strip()

    return _prettify_filename(fallback_path)


def detect_document_language(text: str, filename: str = "") -> str:
    """Lightweight deterministic TR vs EN language detector for corpus metadata."""
    lower_name = filename.lower()
    if "_tr_" in lower_name or lower_name.startswith("tr_") or lower_name.endswith("_tr.md"):
        return "tr"

    sample = text[:3000]
    alpha_chars = [ch for ch in sample if ch.isalpha()]
    if not alpha_chars:
        return "en"
    tr_hits = sum(1 for ch in alpha_chars if ch in _TURKISH_DIACRITICS)
    # Turkish prose typically has >3% Turkish-specific diacritics.
    if (tr_hits / len(alpha_chars)) >= 0.025:
        return "tr"
    return "en"


def _resolve_pdf_title(content: str, extra_meta: dict[str, Any], fallback_path: Path) -> str:
    meta_title = extra_meta.get("pdf_title")
    if isinstance(meta_title, str) and meta_title.strip():
        return meta_title.strip()
    return _extract_markdown_title(content, fallback_path)


def _validate_upload_inputs(filename: str, data: bytes) -> tuple[str, Path, str]:
    safe_name = Path(filename or "").name.strip()
    if not safe_name:
        raise ValueError("Uploaded file must have a valid filename.")

    pseudo_path = Path(safe_name)
    suffix = pseudo_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported document extension '{suffix or '(none)'}' for {safe_name}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if not data:
        raise ValueError(f"Uploaded file '{safe_name}' is empty.")

    return safe_name, pseudo_path, suffix


async def aload_document_bytes(
    filename: str,
    data: bytes,
    *,
    ocr_mode: PdfOcrMode | None = None,
    pdf_config: PdfParseConfig | None = None,
    vision_ocr_fn: VisionOcrFn | None = None,
    settings: Settings | None = None,
    cost_tracker: CostTracker | None = None,
) -> LoadedDocument:
    """Asynchronously load an uploaded document from raw bytes (.md, .markdown, .txt, .pdf)."""
    safe_name, pseudo_path, suffix = _validate_upload_inputs(filename, data)

    extra_meta: dict[str, Any] = {}
    if suffix in {".md", ".markdown"}:
        try:
            content = data.decode("utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            raise ValueError(f"File '{safe_name}' is not valid UTF-8 text.") from exc
        doc_type: Literal["markdown", "pdf", "text"] = "markdown"
        title = _extract_markdown_title(content, pseudo_path)
    elif suffix == ".txt":
        try:
            content = data.decode("utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            raise ValueError(f"File '{safe_name}' is not valid UTF-8 text.") from exc
        doc_type = "text"
        title = _prettify_filename(pseudo_path)
    else:
        effective_cfg = pdf_config or PdfParseConfig.from_settings(
            settings,
            ocr_mode_override=ocr_mode,
        )
        try:
            parsed = await aparse_pdf_bytes(
                data,
                config=effective_cfg,
                vision_ocr_fn=vision_ocr_fn,
                settings=settings,
                cost_tracker=cost_tracker,
            )
        except ValueError as exc:
            raise ValueError(f"Failed to parse PDF '{safe_name}': {exc}") from exc
        content = parsed.markdown
        extra_meta = parsed.metadata
        doc_type = "pdf"
        title = _resolve_pdf_title(content, extra_meta, pseudo_path)

    if not content:
        hint = (
            " (enable OCR mode 'auto', 'ocria_nano', or 'vision_llm' if this is a scanned PDF)"
            if suffix == ".pdf"
            and (ocr_mode == "off" or (pdf_config and pdf_config.ocr_mode == "off"))
            else ""
        )
        raise ValueError(f"Document has no extractable text content: {safe_name}{hint}")

    language = detect_document_language(content, safe_name)
    return LoadedDocument(
        doc_id=_slugify_doc_id(pseudo_path),
        title=title,
        content=content,
        source_path=f"upload://{safe_name}",
        doc_type=doc_type,
        language=language,
        metadata={
            "file_name": safe_name,
            "file_size_bytes": len(data),
            "source": "upload",
            **extra_meta,
        },
    )


def load_document_bytes(
    filename: str,
    data: bytes,
    *,
    ocr_mode: PdfOcrMode | None = None,
    pdf_config: PdfParseConfig | None = None,
    vision_ocr_fn: VisionOcrFn | None = None,
    settings: Settings | None = None,
    cost_tracker: CostTracker | None = None,
) -> LoadedDocument:
    """Load an uploaded document from raw bytes (.md, .markdown, .txt, .pdf)."""
    safe_name, pseudo_path, suffix = _validate_upload_inputs(filename, data)
    if suffix != ".pdf":
        return asyncio.run(aload_document_bytes(filename, data))

    effective_cfg = pdf_config or PdfParseConfig.from_settings(
        settings,
        ocr_mode_override=ocr_mode,
    )
    try:
        parsed = parse_pdf_bytes(
            data,
            config=effective_cfg,
            vision_ocr_fn=vision_ocr_fn,
            settings=settings,
            cost_tracker=cost_tracker,
        )
    except ValueError as exc:
        raise ValueError(f"Failed to parse PDF '{safe_name}': {exc}") from exc

    content = parsed.markdown
    extra_meta = parsed.metadata
    if not content:
        hint = (
            " (enable OCR mode 'auto', 'ocria_nano', or 'vision_llm' if this is a scanned PDF)"
            if effective_cfg.ocr_mode == "off"
            else ""
        )
        raise ValueError(f"Document has no extractable text content: {safe_name}{hint}")

    title = _resolve_pdf_title(content, extra_meta, pseudo_path)
    language = detect_document_language(content, safe_name)
    return LoadedDocument(
        doc_id=_slugify_doc_id(pseudo_path),
        title=title,
        content=content,
        source_path=f"upload://{safe_name}",
        doc_type="pdf",
        language=language,
        metadata={
            "file_name": safe_name,
            "file_size_bytes": len(data),
            "source": "upload",
            **extra_meta,
        },
    )


async def aload_document(
    path: Path,
    base_dir: Path | None = None,
    *,
    ocr_mode: PdfOcrMode | None = None,
    pdf_config: PdfParseConfig | None = None,
    vision_ocr_fn: VisionOcrFn | None = None,
    settings: Settings | None = None,
    cost_tracker: CostTracker | None = None,
) -> LoadedDocument:
    """Asynchronously load a single supported document (.md, .markdown, .txt, .pdf)."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Document not found: {resolved}")

    suffix = resolved.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported document extension '{suffix}' for {resolved.name}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    extra_meta: dict[str, Any] = {}
    if suffix in {".md", ".markdown"}:
        raw_text = await asyncio.to_thread(resolved.read_text, encoding="utf-8-sig")
        content = raw_text.strip()
        doc_type: Literal["markdown", "pdf", "text"] = "markdown"
        title = _extract_markdown_title(content, resolved)
    elif suffix == ".txt":
        raw_text = await asyncio.to_thread(resolved.read_text, encoding="utf-8-sig")
        content = raw_text.strip()
        doc_type = "text"
        title = _prettify_filename(resolved)
    else:
        raw_bytes = await asyncio.to_thread(resolved.read_bytes)
        effective_cfg = pdf_config or PdfParseConfig.from_settings(
            settings,
            ocr_mode_override=ocr_mode,
        )
        parsed = await aparse_pdf_bytes(
            raw_bytes,
            config=effective_cfg,
            vision_ocr_fn=vision_ocr_fn,
            settings=settings,
            cost_tracker=cost_tracker,
        )
        content = parsed.markdown
        extra_meta = parsed.metadata
        doc_type = "pdf"
        title = _resolve_pdf_title(content, extra_meta, resolved)

    if not content:
        raise ValueError(f"Document has no extractable text content: {resolved}")

    rel_source = str(resolved)
    if base_dir is not None:
        try:
            rel_source = resolved.relative_to(base_dir.resolve()).as_posix()
        except ValueError:
            rel_source = resolved.name

    language = detect_document_language(content, resolved.name)
    file_size = await asyncio.to_thread(lambda: resolved.stat().st_size)

    return LoadedDocument(
        doc_id=_slugify_doc_id(resolved, base_dir=base_dir),
        title=title,
        content=content,
        source_path=rel_source,
        doc_type=doc_type,
        language=language,
        metadata={
            "file_name": resolved.name,
            "file_size_bytes": file_size,
            **extra_meta,
        },
    )


def load_document(
    path: Path,
    base_dir: Path | None = None,
    *,
    ocr_mode: PdfOcrMode | None = None,
    pdf_config: PdfParseConfig | None = None,
    vision_ocr_fn: VisionOcrFn | None = None,
    settings: Settings | None = None,
    cost_tracker: CostTracker | None = None,
) -> LoadedDocument:
    """Load a single supported document (.md, .markdown, .txt, .pdf)."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Document not found: {resolved}")

    suffix = resolved.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported document extension '{suffix}' for {resolved.name}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    extra_meta: dict[str, Any] = {}
    if suffix in {".md", ".markdown"}:
        content = resolved.read_text(encoding="utf-8-sig").strip()
        doc_type: Literal["markdown", "pdf", "text"] = "markdown"
        title = _extract_markdown_title(content, resolved)
    elif suffix == ".txt":
        content = resolved.read_text(encoding="utf-8-sig").strip()
        doc_type = "text"
        title = _prettify_filename(resolved)
    else:
        effective_cfg = pdf_config or PdfParseConfig.from_settings(
            settings,
            ocr_mode_override=ocr_mode,
        )
        parsed = parse_pdf_bytes(
            resolved.read_bytes(),
            config=effective_cfg,
            vision_ocr_fn=vision_ocr_fn,
            settings=settings,
            cost_tracker=cost_tracker,
        )
        content = parsed.markdown
        extra_meta = parsed.metadata
        doc_type = "pdf"
        title = _resolve_pdf_title(content, extra_meta, resolved)

    if not content:
        raise ValueError(f"Document has no extractable text content: {resolved}")

    rel_source = str(resolved)
    if base_dir is not None:
        try:
            rel_source = resolved.relative_to(base_dir.resolve()).as_posix()
        except ValueError:
            rel_source = resolved.name

    language = detect_document_language(content, resolved.name)

    return LoadedDocument(
        doc_id=_slugify_doc_id(resolved, base_dir=base_dir),
        title=title,
        content=content,
        source_path=rel_source,
        doc_type=doc_type,
        language=language,
        metadata={
            "file_name": resolved.name,
            "file_size_bytes": resolved.stat().st_size,
            **extra_meta,
        },
    )


async def aload_directory(
    directory: Path,
    *,
    recursive: bool = True,
    ocr_mode: PdfOcrMode | None = None,
    pdf_config: PdfParseConfig | None = None,
    vision_ocr_fn: VisionOcrFn | None = None,
    settings: Settings | None = None,
    cost_tracker: CostTracker | None = None,
) -> list[LoadedDocument]:
    """Asynchronously load all supported documents from a directory in deterministic sorted order."""
    root = directory.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {root}")

    def _scan() -> list[Path]:
        iterator = root.rglob("*") if recursive else root.glob("*")
        return sorted(
            p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    paths = await asyncio.to_thread(_scan)
    docs: list[LoadedDocument] = []
    for p in paths:
        docs.append(
            await aload_document(
                p,
                base_dir=root,
                ocr_mode=ocr_mode,
                pdf_config=pdf_config,
                vision_ocr_fn=vision_ocr_fn,
                settings=settings,
                cost_tracker=cost_tracker,
            )
        )
    return docs


def load_directory(
    directory: Path,
    *,
    recursive: bool = True,
    ocr_mode: PdfOcrMode | None = None,
    pdf_config: PdfParseConfig | None = None,
    vision_ocr_fn: VisionOcrFn | None = None,
    settings: Settings | None = None,
    cost_tracker: CostTracker | None = None,
) -> list[LoadedDocument]:
    """Load all supported documents from a directory in deterministic sorted order."""
    root = directory.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {root}")

    iterator = root.rglob("*") if recursive else root.glob("*")
    paths = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    return [
        load_document(
            p,
            base_dir=root,
            ocr_mode=ocr_mode,
            pdf_config=pdf_config,
            vision_ocr_fn=vision_ocr_fn,
            settings=settings,
            cost_tracker=cost_tracker,
        )
        for p in paths
    ]
