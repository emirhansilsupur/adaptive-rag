"""Structure-aware hierarchical Markdown & text chunker with deterministic IDs."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import structlog
import tiktoken
from pydantic import BaseModel, Field, model_validator

from adaptiverag.ingestion.loader import LoadedDocument
from adaptiverag.schemas.retrieval import EvidenceChunk, SourceType

logger = structlog.get_logger(__name__)

# Fixed namespace UUID for deterministic, idempotent Qdrant point IDs across re-runs.
ADAPTIVERAG_CHUNK_NAMESPACE = uuid.UUID("7f4c9a12-6b3e-4d8a-9c51-2a8e1f0b4d63")

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_FALLBACK_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class ChunkingConfig(BaseModel):
    """Parameters governing token-budgeted, structure-aware (flat or parent-child) chunking."""

    max_tokens: int = Field(default=512, ge=64, le=4096)
    overlap_tokens: int = Field(default=64, ge=0, le=512)
    min_chunk_tokens: int = Field(default=40, ge=1, le=512)
    encoding_name: str = "cl100k_base"

    # Optional Hierarchical Parent-Child (Small-to-Big) mode
    parent_child_enabled: bool = False
    parent_max_tokens: int = Field(default=1000, ge=128, le=8192)
    child_max_tokens: int = Field(default=256, ge=64, le=2048)
    child_overlap_tokens: int = Field(default=40, ge=0, le=512)

    @model_validator(mode="after")
    def _validate_bounds(self) -> ChunkingConfig:
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError(
                f"overlap_tokens ({self.overlap_tokens}) must be < max_tokens ({self.max_tokens})"
            )
        if self.min_chunk_tokens > self.max_tokens:
            raise ValueError(
                f"min_chunk_tokens ({self.min_chunk_tokens}) must be <= max_tokens ({self.max_tokens})"
            )
        if self.parent_child_enabled:
            if self.child_overlap_tokens >= self.child_max_tokens:
                raise ValueError(
                    f"child_overlap_tokens ({self.child_overlap_tokens}) must be < "
                    f"child_max_tokens ({self.child_max_tokens})"
                )
            if self.child_max_tokens >= self.parent_max_tokens:
                raise ValueError(
                    f"child_max_tokens ({self.child_max_tokens}) must be < "
                    f"parent_max_tokens ({self.parent_max_tokens})"
                )
            if self.min_chunk_tokens > self.child_max_tokens:
                raise ValueError(
                    f"min_chunk_tokens ({self.min_chunk_tokens}) must be <= "
                    f"child_max_tokens ({self.child_max_tokens})"
                )
        return self


class DocumentChunk(BaseModel):
    """A structure-aware chunk ready for dense+sparse embedding and Qdrant upsert."""

    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    section_path: str = ""
    text: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    token_count: int = Field(ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Optional Parent-Child (Small-to-Big) lineage populated when parent_child_enabled=True
    parent_id: str | None = None
    parent_text: str | None = None
    parent_section_path: str | None = None
    parent_token_count: int | None = Field(default=None, ge=1)
    child_index: int | None = Field(default=None, ge=0)
    child_count: int | None = Field(default=None, ge=1)

    @property
    def embedding_text(self) -> str:
        """Context-enriched text used for dense and BM25 indexing."""
        header_parts = [self.title]
        if self.section_path and self.section_path != self.title:
            header_parts.append(self.section_path)
        header = " | ".join(header_parts)
        return f"[{header}]\n{self.text}"

    def to_evidence_chunk(self) -> EvidenceChunk:
        """Convert to runtime EvidenceChunk representation."""
        source_url = self.metadata.get("source_url") or self.metadata.get("url")
        return EvidenceChunk(
            chunk_id=self.chunk_id,
            text=self.text,
            source_type=SourceType.LOCAL,
            doc_id=self.doc_id,
            title=self.title,
            section_path=self.section_path,
            url=str(source_url) if source_url else None,
            metadata=dict(self.metadata),
            token_count=self.token_count,
            parent_id=self.parent_id,
            parent_text=self.parent_text,
            parent_section_path=self.parent_section_path,
            parent_token_count=self.parent_token_count,
        )


@lru_cache(maxsize=4)
def _load_tiktoken_encoding(encoding_name: str) -> tiktoken.Encoding | None:
    """Attempt to load tiktoken encoding; return None gracefully if offline/uncached."""
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as exc:
        logger.warning(
            "tiktoken_unavailable_using_fallback",
            encoding=encoding_name,
            error=str(exc),
        )
        return None


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens using tiktoken when available, else deterministic regex fallback."""
    if not text:
        return 0
    enc = _load_tiktoken_encoding(encoding_name)
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    # Deterministic subword-aware fallback (~1.15x regex tokens for agglutinative TR/EN text)
    raw_tokens = len(_FALLBACK_TOKEN_RE.findall(text))
    return max(1, round(raw_tokens * 1.15))


@dataclass(frozen=True)
class _AtomicBlock:
    kind: Literal["paragraph", "code", "table"]
    text: str


@dataclass
class _MarkdownSection:
    section_path: str
    blocks: list[_AtomicBlock]
    heading_level: int = 0
    heading_text: str = ""


def _flush_paragraph(buf: list[str], blocks: list[_AtomicBlock]) -> None:
    text = "\n".join(buf).strip()
    if text:
        blocks.append(_AtomicBlock(kind="paragraph", text=text))
    buf.clear()


def _parse_markdown_sections(content: str, doc_title: str) -> list[_MarkdownSection]:
    """Parse Markdown into heading-scoped sections preserving code blocks and tables."""
    lines = content.splitlines()
    heading_stack: list[tuple[int, str]] = []
    sections: list[_MarkdownSection] = []

    current_path = doc_title
    current_level = 0
    current_heading = doc_title
    current_blocks: list[_AtomicBlock] = []
    para_buf: list[str] = []

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        # 1. Fenced code block (atomic, immune to '# comment' headings inside code)
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            _flush_paragraph(para_buf, current_blocks)
            marker = fence_match.group(1)
            fence_char = marker[0]
            fence_len = len(marker)
            code_lines = [line]
            i += 1
            while i < n:
                code_lines.append(lines[i])
                close_match = _FENCE_RE.match(lines[i])
                if (
                    close_match
                    and close_match.group(1)[0] == fence_char
                    and len(close_match.group(1)) >= fence_len
                    and lines[i].strip().strip(fence_char) == ""
                ):
                    i += 1
                    break
                i += 1
            current_blocks.append(_AtomicBlock(kind="code", text="\n".join(code_lines).strip()))
            continue

        # 2. Markdown heading (H1-H4) outside code fences
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            _flush_paragraph(para_buf, current_blocks)
            if current_blocks:
                sections.append(
                    _MarkdownSection(
                        section_path=current_path,
                        blocks=current_blocks,
                        heading_level=current_level,
                        heading_text=current_heading,
                    )
                )
                current_blocks = []

            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text))
            current_path = " > ".join(h_text for _, h_text in heading_stack)
            current_level = level
            current_heading = heading_text
            i += 1
            continue

        # 3. Markdown pipe table (kept intact as an atomic block)
        if _TABLE_LINE_RE.match(line):
            _flush_paragraph(para_buf, current_blocks)
            table_lines = [line]
            i += 1
            while i < n and _TABLE_LINE_RE.match(lines[i]):
                table_lines.append(lines[i])
                i += 1
            current_blocks.append(_AtomicBlock(kind="table", text="\n".join(table_lines).strip()))
            continue

        # 4. Blank line ends current paragraph
        if not line.strip():
            _flush_paragraph(para_buf, current_blocks)
            i += 1
            continue

        # 5. Regular prose / list line
        para_buf.append(line)
        i += 1

    _flush_paragraph(para_buf, current_blocks)
    if current_blocks:
        sections.append(
            _MarkdownSection(
                section_path=current_path,
                blocks=current_blocks,
                heading_level=current_level,
                heading_text=current_heading,
            )
        )

    return sections


def _split_oversized_code(code_text: str, max_tokens: int, encoding_name: str) -> list[str]:
    """Split an oversized fenced code block along line boundaries keeping valid fences."""
    lines = code_text.splitlines()
    if len(lines) <= 2:
        return [code_text]

    opening_fence = lines[0]
    closing_fence = lines[-1] if _FENCE_RE.match(lines[-1]) else "```"
    body_lines = lines[1:-1] if _FENCE_RE.match(lines[-1]) else lines[1:]
    fence_overhead = count_tokens(f"{opening_fence}\n{closing_fence}", encoding_name)
    budget = max(32, max_tokens - fence_overhead)

    parts: list[str] = []
    buf: list[str] = []
    for line in body_lines:
        candidate = "\n".join([*buf, line])
        if buf and count_tokens(candidate, encoding_name) > budget:
            parts.append("\n".join([opening_fence, *buf, closing_fence]))
            buf = [line]
        else:
            buf.append(line)
    if buf:
        parts.append("\n".join([opening_fence, *buf, closing_fence]))
    return parts


def _split_oversized_table(table_text: str, max_tokens: int, encoding_name: str) -> list[str]:
    """Split an oversized Markdown table by rows while repeating the table header."""
    lines = table_text.splitlines()
    if len(lines) <= 2:
        return [table_text]

    header = lines[:2]
    rows = lines[2:]
    header_tokens = count_tokens("\n".join(header), encoding_name)
    budget = max(32, max_tokens - header_tokens)

    parts: list[str] = []
    buf: list[str] = []
    for row in rows:
        candidate = "\n".join([*buf, row])
        if buf and count_tokens(candidate, encoding_name) > budget:
            parts.append("\n".join([*header, *buf]))
            buf = [row]
        else:
            buf.append(row)
    if buf:
        parts.append("\n".join([*header, *buf]))
    return parts


def _split_oversized_paragraph(text: str, max_tokens: int, encoding_name: str) -> list[str]:
    """Split an oversized paragraph on sentence boundaries, then words if needed."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if len(sentences) <= 1:
        # Word-level split fallback for single mega-sentences
        words = text.split()
        parts: list[str] = []
        buf: list[str] = []
        for word in words:
            candidate = " ".join([*buf, word])
            if buf and count_tokens(candidate, encoding_name) > max_tokens:
                parts.append(" ".join(buf))
                buf = [word]
            else:
                buf.append(word)
        if buf:
            parts.append(" ".join(buf))
        return parts

    parts = []
    sent_buf: list[str] = []
    for sent in sentences:
        if count_tokens(sent, encoding_name) > max_tokens:
            if sent_buf:
                parts.append(" ".join(sent_buf))
                sent_buf = []
            parts.extend(_split_oversized_paragraph(sent, max_tokens, encoding_name))
            continue

        candidate = " ".join([*sent_buf, sent])
        if sent_buf and count_tokens(candidate, encoding_name) > max_tokens:
            parts.append(" ".join(sent_buf))
            sent_buf = [sent]
        else:
            sent_buf.append(sent)
    if sent_buf:
        parts.append(" ".join(sent_buf))
    return parts


def _expand_oversized_block(
    block: _AtomicBlock,
    max_tokens: int,
    encoding_name: str,
) -> list[_AtomicBlock]:
    """Ensure every atomic block fits within max_tokens while preserving structure."""
    if count_tokens(block.text, encoding_name) <= max_tokens:
        return [block]
    if block.kind == "code":
        return [
            _AtomicBlock(kind="code", text=p)
            for p in _split_oversized_code(block.text, max_tokens, encoding_name)
        ]
    if block.kind == "table":
        return [
            _AtomicBlock(kind="table", text=p)
            for p in _split_oversized_table(block.text, max_tokens, encoding_name)
        ]
    return [
        _AtomicBlock(kind="paragraph", text=p)
        for p in _split_oversized_paragraph(block.text, max_tokens, encoding_name)
    ]


def _extract_overlap_tail(
    blocks: list[_AtomicBlock], overlap_tokens: int, encoding_name: str
) -> str:
    """Extract clean prose overlap from the tail of the previous chunk (never broken fences)."""
    if overlap_tokens <= 0 or not blocks:
        return ""
    # Do not copy partial fenced code blocks or pipe tables as overlap prefixes.
    tail_block = blocks[-1]
    if tail_block.kind != "paragraph":
        return ""

    words = tail_block.text.split()
    selected: list[str] = []
    for word in reversed(words):
        candidate = " ".join(reversed([*selected, word]))
        if selected and count_tokens(candidate, encoding_name) > overlap_tokens:
            break
        selected.append(word)
    return " ".join(reversed(selected))


def _content_hash(text: str) -> str:
    """Short deterministic SHA-256 fingerprint of chunk content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _make_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Create a deterministic UUIDv5 chunk identifier compatible with Qdrant PointId."""
    return str(uuid.uuid5(ADAPTIVERAG_CHUNK_NAMESPACE, f"{doc_id}::{chunk_index}"))


def _make_child_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Create a deterministic UUIDv5 child chunk ID disjoint from flat chunk IDs."""
    return str(uuid.uuid5(ADAPTIVERAG_CHUNK_NAMESPACE, f"{doc_id}::child::{chunk_index}"))


def _make_parent_id(doc_id: str, parent_index: int) -> str:
    """Create a deterministic UUIDv5 parent passage identifier for Small-to-Big retrieval."""
    return str(uuid.uuid5(ADAPTIVERAG_CHUNK_NAMESPACE, f"{doc_id}::parent::{parent_index}"))


def _pack_blocks(
    blocks: list[_AtomicBlock],
    *,
    max_tokens: int,
    overlap_tokens: int,
    min_chunk_tokens: int,
    encoding_name: str,
) -> list[tuple[list[_AtomicBlock], str]]:
    """Pack atomic blocks into token-budgeted windows preserving code fences and tables."""
    expanded: list[_AtomicBlock] = []
    for blk in blocks:
        expanded.extend(_expand_oversized_block(blk, max_tokens, encoding_name))

    packed: list[tuple[list[_AtomicBlock], str]] = []
    current_blocks: list[_AtomicBlock] = []
    overlap_prefix = ""

    for blk in expanded:
        parts = [overlap_prefix] if overlap_prefix else []
        parts.extend(b.text for b in current_blocks)
        parts.append(blk.text)
        candidate_text = "\n\n".join(parts)

        if current_blocks and count_tokens(candidate_text, encoding_name) > max_tokens:
            finalized_parts = [overlap_prefix] if overlap_prefix else []
            finalized_parts.extend(b.text for b in current_blocks)
            packed.append((list(current_blocks), "\n\n".join(finalized_parts)))

            overlap_prefix = _extract_overlap_tail(current_blocks, overlap_tokens, encoding_name)
            if (
                overlap_prefix
                and count_tokens(f"{overlap_prefix}\n\n{blk.text}", encoding_name) > max_tokens
            ):
                overlap_prefix = ""
            current_blocks = [blk]
        else:
            current_blocks.append(blk)

    if current_blocks:
        finalized_parts = [overlap_prefix] if overlap_prefix else []
        finalized_parts.extend(b.text for b in current_blocks)
        packed.append((list(current_blocks), "\n\n".join(finalized_parts)))

    # Merge tiny trailing chunk within the same scope when safely within budget
    if len(packed) >= 2:
        last_blocks, last_text = packed[-1]
        if count_tokens(last_text, encoding_name) < min_chunk_tokens:
            prev_blocks, prev_text = packed[-2]
            merged_text = f"{prev_text}\n\n" + "\n\n".join(b.text for b in last_blocks)
            if count_tokens(merged_text, encoding_name) <= int(max_tokens * 1.15):
                packed[-2] = (prev_blocks + last_blocks, merged_text)
                packed.pop()

    return packed


@dataclass(frozen=True)
class _ScopedBlock:
    section_path: str
    subheading_line: str | None
    block: _AtomicBlock


def _group_sections_into_parent_scopes(
    sections: list[_MarkdownSection],
) -> list[tuple[str, list[_MarkdownSection]]]:
    """Group H3/H4 subsections under their enclosing H1/H2 parent section."""
    groups: list[tuple[str, list[_MarkdownSection]]] = []
    for section in sections:
        starts_new_parent = not groups or section.heading_level <= 2
        if starts_new_parent:
            groups.append((section.section_path, [section]))
        else:
            groups[-1][1].append(section)
    return groups


def _render_parent_window_text(items: list[_ScopedBlock]) -> str:
    """Render cohesive Markdown text for a parent window, keeping H3/H4 subheadings intact."""
    parts: list[str] = []
    emitted_headings: set[tuple[str, str]] = set()
    for item in items:
        if item.subheading_line:
            key = (item.section_path, item.subheading_line)
            if key not in emitted_headings:
                emitted_headings.add(key)
                parts.append(item.subheading_line)
        parts.append(item.block.text)
    return "\n\n".join(parts).strip()


def _chunk_document_parent_child(
    doc: LoadedDocument,
    cfg: ChunkingConfig,
) -> list[DocumentChunk]:
    """Build cohesive parent windows (<= parent_max_tokens) and focused child chunks (<= child_max_tokens)."""
    sections = _parse_markdown_sections(doc.content, doc_title=doc.title)
    parent_groups = _group_sections_into_parent_scopes(sections)

    chunks: list[DocumentChunk] = []
    parent_idx = 0
    global_child_idx = 0

    for parent_section_path, group_sections in parent_groups:
        # Flatten group blocks while preserving subsection breadcrumbs and atomic constraints
        scoped_items: list[_ScopedBlock] = []
        for sub_sec in group_sections:
            subheading_line: str | None = None
            if (
                sub_sec.section_path != parent_section_path
                and sub_sec.heading_level >= 3
                and sub_sec.heading_text
            ):
                subheading_line = f"{'#' * sub_sec.heading_level} {sub_sec.heading_text}"
            for blk in sub_sec.blocks:
                for piece in _expand_oversized_block(blk, cfg.parent_max_tokens, cfg.encoding_name):
                    scoped_items.append(
                        _ScopedBlock(
                            section_path=sub_sec.section_path,
                            subheading_line=subheading_line,
                            block=piece,
                        )
                    )

        if not scoped_items:
            continue

        # Pack scoped items into parent windows up to parent_max_tokens
        parent_windows: list[list[_ScopedBlock]] = []
        current_window: list[_ScopedBlock] = []
        for item in scoped_items:
            candidate_window = [*current_window, item]
            candidate_text = _render_parent_window_text(candidate_window)
            if (
                current_window
                and count_tokens(candidate_text, cfg.encoding_name) > cfg.parent_max_tokens
            ):
                parent_windows.append(current_window)
                current_window = [item]
            else:
                current_window.append(item)
        if current_window:
            parent_windows.append(current_window)

        for window_items in parent_windows:
            parent_text = _render_parent_window_text(window_items)
            if not parent_text:
                continue
            parent_id = _make_parent_id(doc.doc_id, parent_idx)
            parent_tokens = max(1, count_tokens(parent_text, cfg.encoding_name))
            parent_idx += 1

            # Slice this parent window into child chunks respecting contiguous subsection boundaries
            subsection_runs: list[tuple[str, list[_AtomicBlock]]] = []
            for item in window_items:
                if subsection_runs and subsection_runs[-1][0] == item.section_path:
                    subsection_runs[-1][1].append(item.block)
                else:
                    subsection_runs.append((item.section_path, [item.block]))

            raw_children: list[tuple[str, str, bool, bool]] = []
            for sub_path, sub_blocks in subsection_runs:
                packed_children = _pack_blocks(
                    sub_blocks,
                    max_tokens=cfg.child_max_tokens,
                    overlap_tokens=cfg.child_overlap_tokens,
                    min_chunk_tokens=cfg.min_chunk_tokens,
                    encoding_name=cfg.encoding_name,
                )
                for child_blocks, child_text in packed_children:
                    cleaned = child_text.strip()
                    if not cleaned:
                        continue
                    has_code = any(b.kind == "code" for b in child_blocks)
                    has_table = any(b.kind == "table" for b in child_blocks)
                    raw_children.append((sub_path, cleaned, has_code, has_table))

            child_total = len(raw_children)
            for local_idx, (child_section_path, child_text, has_code, has_table) in enumerate(
                raw_children
            ):
                child_tokens = max(1, count_tokens(child_text, cfg.encoding_name))
                chunks.append(
                    DocumentChunk(
                        chunk_id=_make_child_chunk_id(doc.doc_id, global_child_idx),
                        doc_id=doc.doc_id,
                        title=doc.title,
                        section_path=child_section_path,
                        text=child_text,
                        chunk_index=global_child_idx,
                        token_count=child_tokens,
                        parent_id=parent_id,
                        parent_text=parent_text,
                        parent_section_path=parent_section_path,
                        parent_token_count=parent_tokens,
                        child_index=local_idx,
                        child_count=child_total,
                        metadata={
                            **doc.metadata,
                            "source_path": doc.source_path,
                            "doc_type": doc.doc_type,
                            "language": doc.language,
                            "content_hash": _content_hash(child_text),
                            "has_code_block": has_code,
                            "has_table": has_table,
                            "chunk_strategy": "parent_child",
                            "parent_id": parent_id,
                            "parent_token_count": parent_tokens,
                            "child_index": local_idx,
                            "child_count": child_total,
                        },
                    )
                )
                global_child_idx += 1

    return chunks


def chunk_document(
    doc: LoadedDocument,
    config: ChunkingConfig | None = None,
) -> list[DocumentChunk]:
    """Split a LoadedDocument into structure-aware, token-budgeted DocumentChunks."""
    cfg = config or ChunkingConfig()
    if cfg.parent_child_enabled:
        return _chunk_document_parent_child(doc, cfg)

    sections = _parse_markdown_sections(doc.content, doc_title=doc.title)
    raw_chunks: list[tuple[str, str, bool, bool]] = []  # (section_path, text, has_code, has_table)

    for section in sections:
        section_packed = _pack_blocks(
            section.blocks,
            max_tokens=cfg.max_tokens,
            overlap_tokens=cfg.overlap_tokens,
            min_chunk_tokens=cfg.min_chunk_tokens,
            encoding_name=cfg.encoding_name,
        )
        for packed_blocks, chunk_text in section_packed:
            has_code = any(b.kind == "code" for b in packed_blocks)
            has_table = any(b.kind == "table" for b in packed_blocks)
            raw_chunks.append((section.section_path, chunk_text.strip(), has_code, has_table))

    chunks: list[DocumentChunk] = []
    for idx, (section_path, text, has_code, has_table) in enumerate(raw_chunks):
        if not text:
            continue
        tok_count = max(1, count_tokens(text, cfg.encoding_name))
        chunks.append(
            DocumentChunk(
                chunk_id=_make_chunk_id(doc.doc_id, idx),
                doc_id=doc.doc_id,
                title=doc.title,
                section_path=section_path,
                text=text,
                chunk_index=idx,
                token_count=tok_count,
                metadata={
                    **doc.metadata,
                    "source_path": doc.source_path,
                    "doc_type": doc.doc_type,
                    "language": doc.language,
                    "content_hash": _content_hash(text),
                    "has_code_block": has_code,
                    "has_table": has_table,
                },
            )
        )

    return chunks


def chunk_documents(
    docs: list[LoadedDocument],
    config: ChunkingConfig | None = None,
) -> list[DocumentChunk]:
    """Chunk a list of LoadedDocuments into a flat list of DocumentChunks."""
    all_chunks: list[DocumentChunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, config=config))
    return all_chunks
