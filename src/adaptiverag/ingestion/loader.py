"""Document loaders for Markdown, plain text, and PDF technical corpora."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from pypdf import PdfReader

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


def _read_pdf_text(path: Path) -> tuple[str, dict[str, Any]]:
    """Extract text page-by-page from a PDF file using pypdf."""
    reader = PdfReader(str(path))
    pages: list[str] = []
    for idx, page in enumerate(reader.pages, start=1):
        extracted = (page.extract_text() or "").strip()
        if extracted:
            pages.append(f"## Page {idx}\n\n{extracted}")
    meta: dict[str, Any] = {"page_count": len(reader.pages)}
    if reader.metadata and reader.metadata.title:
        meta["pdf_title"] = str(reader.metadata.title).strip()
    return "\n\n".join(pages), meta


def load_document(path: Path, base_dir: Path | None = None) -> LoadedDocument:
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
        content, extra_meta = _read_pdf_text(resolved)
        doc_type = "pdf"
        title = str(extra_meta.get("pdf_title") or _prettify_filename(resolved))

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


def load_directory(directory: Path, *, recursive: bool = True) -> list[LoadedDocument]:
    """Load all supported documents from a directory in deterministic sorted order."""
    root = directory.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {root}")

    iterator = root.rglob("*") if recursive else root.glob("*")
    paths = sorted(
        p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return [load_document(p, base_dir=root) for p in paths]
