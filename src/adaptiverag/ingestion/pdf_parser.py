"""Adaptive 2-Tier PDF-to-Markdown parser (LiteParse + OCRia-nano-1 + Vision-LLM fallback).

Routing hierarchy for PDF pages:
1. Born-digital pages (`needs_ocr == False`):
   Parsed in ~2-5 ms/page at $0 cost via LiteParse's native Rust/PDFium layout engine,
   preserving `# / ##` headings, multi-column reading order, and Markdown `|...|` tables.
2. Tier-1 Local ONNX OCR (`OCRia-nano-1`, ~8.1M params + `LayoutNet` / `TableNet`, $0):
   Executed locally on CPU *only* for pages flagged by complexity detection (`scanned`,
   `sparse-text`, `garbled`), supporting Latin/Turkish diacritics (`ğ, ş, ı, İ, ö, ü, ç`).
3. Tier-2 Multimodal Vision-LLM Escalation (optional / configurable):
   Triggered in `vision_llm` mode or automatically in `auto` mode when Tier-1 returns
   near-empty text on a non-blank scanned page (e.g., degraded scans, handwriting, or
   intricate tables), bounded by `max_ocr_pages` and `CostTracker`.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import io
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from pypdf import PdfReader

from adaptiverag.config import Settings, get_settings
from adaptiverag.llm.cost import CostTracker
from adaptiverag.llm.factory import ModelTier, create_chat_model
from adaptiverag.llm.structured import invoke_text
from adaptiverag.schemas.response import TokenUsage
from adaptiverag.verification.guardrails import sanitize_untrusted_passage

logger = structlog.get_logger(__name__)

PdfOcrMode = Literal["auto", "ocria_nano", "vision_llm", "off"]

VisionOcrResult = tuple[str, TokenUsage]
VisionOcrFn = Callable[
    [int, bytes],
    VisionOcrResult | Awaitable[VisionOcrResult],
]

_DEFAULT_MODEL_DIR = Path(
    os.environ.get("LITEPARSE_MODEL_DIR", ".adaptiverag_cache/liteparse")
).resolve()
os.environ.setdefault("LITEPARSE_MODEL_DIR", str(_DEFAULT_MODEL_DIR))

_MODEL_PULL_LOCK = Lock()
_HEADING_LINE_RE = re.compile(r"^\s*#{1,6}\s+\S+", re.MULTILINE)
_BROKEN_LOCAL_IMG_RE = re.compile(r"!\[[^\]]*\]\(img_[^)]+\)")
_EMPTY_CODE_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*```")
_OUTER_MD_FENCE_RE = re.compile(
    r"^\s*```(?:markdown|md)?\s*\n(.*?)\n```\s*$",
    re.DOTALL | re.IGNORECASE,
)
_GENERIC_PDF_TITLES = frozenset(
    {
        "",
        "untitled",
        "document",
        "microsoft word",
        "powerpoint presentation",
        "none",
    }
)

_VISION_OCR_SYSTEM_PROMPT = (
    "You are a precision document OCR and layout-to-Markdown transcription engine. "
    "Transcribe ALL visible text from the provided document page image into clean "
    "GitHub-Flavored Markdown.\n"
    "Rules:\n"
    "1. Preserve structural hierarchy using `#`, `##`, `###` headings, bullet/numbered lists, "
    "code blocks, and inline/block LaTeX math where present.\n"
    "2. Reconstruct all tables accurately as Markdown pipe tables (`| col | col |`).\n"
    "3. Preserve the exact original language and all diacritics (including Turkish characters "
    "`ğ, Ğ, ş, Ş, ı, İ, ö, Ö, ü, Ü, ç, Ç`) without translating or summarizing.\n"
    "4. Output ONLY the transcribed Markdown content with no preamble, commentary, or outer "
    "code fences. If the page is completely blank, output an empty string."
)


class PdfParseConfig(BaseModel):
    """Runtime configuration for Adaptive 2-Tier PDF parsing and OCR."""

    ocr_mode: PdfOcrMode = "auto"
    auto_pull_models: bool = True
    model_dir: Path = Field(default_factory=lambda: _DEFAULT_MODEL_DIR)
    vision_fallback_enabled: bool = True
    vision_model: str = "google/gemini-2.5-flash"
    dpi: int = Field(default=200, ge=72, le=400)
    max_ocr_pages: int = Field(default=25, ge=1, le=500)
    min_page_chars: int = Field(default=32, ge=1, le=500)

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        ocr_mode_override: PdfOcrMode | None = None,
    ) -> PdfParseConfig:
        resolved = settings or get_settings()
        return cls(
            ocr_mode=ocr_mode_override if ocr_mode_override is not None else resolved.pdf_ocr_mode,
            auto_pull_models=resolved.pdf_ocr_auto_pull,
            model_dir=resolved.pdf_ocr_model_dir.resolve(),
            vision_fallback_enabled=resolved.pdf_ocr_vision_fallback_enabled,
            vision_model=resolved.pdf_ocr_vision_model,
            dpi=resolved.pdf_ocr_dpi,
            max_ocr_pages=resolved.pdf_ocr_max_pages,
            min_page_chars=resolved.pdf_ocr_min_chars,
        )


@dataclass
class PdfParseOutput:
    """Normalized Markdown and per-page telemetry returned by the Adaptive PDF parser."""

    markdown: str
    metadata: dict[str, Any]
    usage: TokenUsage


def _configure_model_dir_env(model_dir: Path) -> Path:
    resolved = model_dir.resolve()
    os.environ["LITEPARSE_MODEL_DIR"] = str(resolved)
    return resolved


def is_ocria_nano_installed(model_dir: Path | None = None) -> bool:
    """Return True when the local `ocria-nano` ONNX model bundle is complete on disk."""
    target_dir = _configure_model_dir_env(model_dir or _DEFAULT_MODEL_DIR)
    manifest = target_dir / "ocria-nano" / "manifest.json"
    det = target_dir / "ocria-nano" / "det.onnx"
    rec_latin = target_dir / "ocria-nano" / "rec_latin.onnx"
    if manifest.is_file() and det.is_file() and rec_latin.is_file():
        return True
    try:
        from liteparse import models as lp_models

        return any(m.name == "ocria-nano" and m.complete for m in lp_models.list())
    except Exception:
        return False


def _resolve_table_model_path(model_dir: Path) -> Path | None:
    candidate = model_dir.resolve() / "ocria-nano" / "tablenet.bin"
    return candidate if candidate.is_file() else None


def ensure_ocria_nano_installed(model_dir: Path, *, auto_pull: bool) -> bool:
    """Ensure `ocria-nano` weights exist locally, optionally pulling once thread-safely."""
    resolved_dir = _configure_model_dir_env(model_dir)
    if is_ocria_nano_installed(resolved_dir):
        return True
    if not auto_pull:
        return False

    with _MODEL_PULL_LOCK:
        if is_ocria_nano_installed(resolved_dir):
            return True
        try:
            resolved_dir.mkdir(parents=True, exist_ok=True)
            from liteparse import models as lp_models

            logger.info("pdf_ocr_pulling_ocria_nano", model_dir=str(resolved_dir))
            lp_models.pull("ocria-nano", quiet=True)
            return is_ocria_nano_installed(resolved_dir)
        except Exception as exc:
            logger.warning("pdf_ocr_pull_failed", error=str(exc), model_dir=str(resolved_dir))
            return False


def _has_llm_credentials(settings: Settings) -> bool:
    if settings.llm_provider == "ollama":
        return True
    key_map = {
        "openrouter": settings.openrouter_api_key,
        "openai": settings.openai_api_key,
        "gemini": settings.gemini_api_key,
        "anthropic": settings.anthropic_api_key,
    }
    secret = key_map.get(settings.llm_provider)
    return bool(secret is not None and secret.get_secret_value().strip())


def build_vision_ocr_fn(
    settings: Settings | None = None,
    *,
    cost_tracker: CostTracker | None = None,
    model_override: str | None = None,
) -> VisionOcrFn | None:
    """Construct an async Tier-2 multimodal Vision-LLM OCR callable when credentials exist."""
    resolved = settings or get_settings()
    if not _has_llm_credentials(resolved):
        return None

    vision_model_name = (model_override or resolved.pdf_ocr_vision_model).strip()
    chat_model = create_chat_model(
        resolved,
        ModelTier.FAST,
        model_override=vision_model_name,
    )

    async def _transcribe_page(page_num: int, png_bytes: bytes) -> VisionOcrResult:
        encoded = base64.b64encode(png_bytes).decode("ascii")
        messages = [
            SystemMessage(content=_VISION_OCR_SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": f"Transcribe page {page_num} into structured Markdown.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                ]
            ),
        ]
        result = await invoke_text(
            chat_model,
            messages,
            model_name=vision_model_name,
            provider=resolved.llm_provider,
            temperature=0.0,
            cost_tracker=cost_tracker,
        )
        cleaned = _clean_page_markdown(result.text)
        return cleaned, result.usage

    return _transcribe_page


def health_pdf_ocr_summary(settings: Settings | None = None) -> dict[str, Any]:
    """Return operational readiness metadata for `/v1/health`."""
    resolved = settings or get_settings()
    model_dir = resolved.pdf_ocr_model_dir.resolve()
    return {
        "parser": "liteparse",
        "default_mode": resolved.pdf_ocr_mode,
        "ocria_nano_installed": is_ocria_nano_installed(model_dir),
        "auto_pull_enabled": resolved.pdf_ocr_auto_pull,
        "vision_fallback_enabled": resolved.pdf_ocr_vision_fallback_enabled,
        "vision_llm_configured": _has_llm_credentials(resolved),
        "vision_model": resolved.pdf_ocr_vision_model,
        "dpi": resolved.pdf_ocr_dpi,
        "max_ocr_pages": resolved.pdf_ocr_max_pages,
    }


def _extract_pdf_metadata_title(data: bytes) -> str | None:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.metadata and reader.metadata.title:
            candidate = str(reader.metadata.title).strip()
            if candidate and candidate.lower() not in _GENERIC_PDF_TITLES:
                return candidate
    except Exception:
        return None
    return None


def _visible_char_count(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def _clean_page_markdown(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    fence_match = _OUTER_MD_FENCE_RE.match(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    cleaned = _EMPTY_CODE_FENCE_RE.sub("", cleaned)
    cleaned = _BROKEN_LOCAL_IMG_RE.sub("", cleaned)
    cleaned = sanitize_untrusted_passage(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _page_genuinely_needs_ocr(comp: Any, cleaned_native: str) -> bool:
    """Return True only when a page contains raster/vector/garbled content needing OCR.

    LiteParse's raw `needs_ocr` also flags short born-digital pages (`sparse-text`) even
    when the page has clean native glyphs and zero images. Skipping OCR on pure-text,
    zero-image, non-garbled pages avoids redundant CPU/Vision-LLM work.
    """
    if comp is None:
        return _visible_char_count(cleaned_native) == 0
    if not comp.needs_ocr:
        return False
    if comp.full_page_image or comp.has_substantial_images or comp.image_block_count > 0:
        return True
    if comp.is_garbled:
        return True
    if (comp.uncovered_vector_area or 0.0) > 500.0:
        return True
    # Zero-text pages without images may still carry vector-outlined glyphs; solid-fill
    # probing filters out genuinely blank white pages right afterwards.
    return int(comp.text_length) == 0


def _format_target_pages(page_numbers: list[int]) -> str:
    return ",".join(str(p) for p in sorted(set(page_numbers)))


def _run_native_layout_pass(data: bytes, model_dir: Path) -> Any:
    from liteparse import LiteParse

    _configure_model_dir_env(model_dir)
    table_model = _resolve_table_model_path(model_dir)
    parser = LiteParse(
        ocr_enabled=False,
        output_format="markdown",
        include_complexity=True,
        keep_headers_footers=False,
        table_model=table_model,
        quiet=True,
    )
    return parser.parse(data)


def _render_page_screenshots(
    data: bytes,
    page_numbers: list[int],
    *,
    dpi: int,
    model_dir: Path,
) -> dict[int, tuple[bytes, bool]]:
    """Render 1-indexed PDF pages to PNG in memory, returning `{page_num: (png_bytes, is_solid_fill)}`."""
    if not page_numbers:
        return {}
    from liteparse import LiteParse

    _configure_model_dir_env(model_dir)
    parser = LiteParse(
        ocr_enabled=False,
        output_format="markdown",
        extract_screenshots=True,
        target_pages=_format_target_pages(page_numbers),
        dpi=float(dpi),
        quiet=True,
    )
    res = parser.parse(data)
    return {shot.page_num: (shot.image_bytes, bool(shot.is_solid_fill)) for shot in res.screenshots}


def _run_ocria_nano_pages(
    data: bytes,
    page_numbers: list[int],
    *,
    dpi: int,
    model_dir: Path,
    fatal_on_error: bool,
) -> dict[int, str]:
    """Run local `ocria-nano` ONNX OCR + LayoutNet/TableNet strictly on `page_numbers`.

    Uses `pool_size=1` process isolation so OCRia-nano's bundled ONNX Runtime never
    collides with FastEmbed's in-process `onnxruntime` shared library and enforces a
    hard 90-second worker timeout on pathological scans.
    """
    if not page_numbers:
        return {}
    from liteparse import LiteParse

    _configure_model_dir_env(model_dir)
    table_model = _resolve_table_model_path(model_dir)
    with LiteParse(
        ocr_enabled=True,
        ocr_backend="ocria-nano",
        output_format="markdown",
        target_pages=_format_target_pages(page_numbers),
        dpi=float(dpi),
        keep_headers_footers=False,
        table_model=table_model,
        ocr_failure_fatal=fatal_on_error,
        pool_size=1,
        parse_timeout=90.0,
        quiet=True,
    ) as parser:
        res = parser.parse(data)
    return {page.page_num: _clean_page_markdown(page.markdown or page.text) for page in res.pages}


async def aparse_pdf_bytes(
    data: bytes,
    *,
    config: PdfParseConfig | None = None,
    vision_ocr_fn: VisionOcrFn | None = None,
    settings: Settings | None = None,
    cost_tracker: CostTracker | None = None,
) -> PdfParseOutput:
    """Asynchronously parse PDF bytes into structure-preserving Markdown with 2-Tier OCR."""
    if not data:
        raise ValueError("PDF payload is empty.")

    resolved_settings = settings or get_settings()
    cfg = config or PdfParseConfig.from_settings(resolved_settings)
    model_dir = _configure_model_dir_env(cfg.model_dir)

    pdf_title = await asyncio.to_thread(_extract_pdf_metadata_title, data)
    try:
        native_res = await asyncio.to_thread(_run_native_layout_pass, data, model_dir)
    except Exception as exc:
        raise ValueError(f"Failed to parse PDF stream: {exc}") from exc

    total_pages = int(native_res.total_pages or len(native_res.pages))
    page_md: dict[int, str] = {}
    raw_ocr_candidates: list[int] = []
    maybe_blank_pages: list[int] = []

    for page in native_res.pages:
        cleaned_native = _clean_page_markdown(page.markdown or page.text)
        page_md[page.page_num] = cleaned_native
        comp = page.complexity
        if _page_genuinely_needs_ocr(comp, cleaned_native):
            raw_ocr_candidates.append(page.page_num)
            if (
                comp is not None
                and comp.text_length == 0
                and not comp.full_page_image
                and not comp.has_substantial_images
                and comp.image_block_count == 0
                and not comp.is_garbled
            ):
                maybe_blank_pages.append(page.page_num)

    ocr_pages_processed: set[int] = set()
    backends_used: list[str] = []
    cumulative_usage = TokenUsage()
    skipped_ocr_pages = 0

    if cfg.ocr_mode != "off" and raw_ocr_candidates:
        # Filter out genuinely blank solid-fill pages before spending CPU or Vision-LLM budget
        cached_screenshots: dict[int, tuple[bytes, bool]] = {}
        if maybe_blank_pages:
            probe_shots = await asyncio.to_thread(
                _render_page_screenshots,
                data,
                maybe_blank_pages,
                dpi=min(cfg.dpi, 150),
                model_dir=model_dir,
            )
            cached_screenshots.update(probe_shots)

        non_blank_candidates = [
            p for p in raw_ocr_candidates if not cached_screenshots.get(p, (b"", False))[1]
        ]

        selected_ocr_pages = non_blank_candidates[: cfg.max_ocr_pages]
        skipped_ocr_pages = max(0, len(non_blank_candidates) - len(selected_ocr_pages))

        escalated_to_vision: list[int] = []

        if selected_ocr_pages and cfg.ocr_mode in ("auto", "ocria_nano"):
            ocria_ready = await asyncio.to_thread(
                ensure_ocria_nano_installed,
                model_dir,
                auto_pull=cfg.auto_pull_models,
            )
            if ocria_ready:
                try:
                    ocria_map = await asyncio.to_thread(
                        _run_ocria_nano_pages,
                        data,
                        selected_ocr_pages,
                        dpi=cfg.dpi,
                        model_dir=model_dir,
                        fatal_on_error=(cfg.ocr_mode == "ocria_nano"),
                    )
                    if ocria_map:
                        backends_used.append("ocria-nano")
                    for pno in selected_ocr_pages:
                        ocr_text = ocria_map.get(pno, "")
                        if _visible_char_count(ocr_text) >= _visible_char_count(
                            page_md.get(pno, "")
                        ):
                            page_md[pno] = ocr_text
                        if _visible_char_count(page_md.get(pno, "")) > 0:
                            ocr_pages_processed.add(pno)
                        if _visible_char_count(page_md.get(pno, "")) < cfg.min_page_chars:
                            escalated_to_vision.append(pno)
                except Exception as exc:
                    if cfg.ocr_mode == "ocria_nano":
                        raise ValueError(f"Local OCRia-nano-1 OCR failed: {exc}") from exc
                    logger.warning("pdf_ocria_nano_failed_escalating", error=str(exc))
                    escalated_to_vision = list(selected_ocr_pages)
            else:
                if cfg.ocr_mode == "ocria_nano":
                    raise ValueError(
                        "OCRia-nano-1 model weights are not installed locally. "
                        "Run `uv run lit models pull ocria-nano`, enable PDF_OCR_AUTO_PULL=true, "
                        "or set ocr_mode='vision_llm'."
                    )
                escalated_to_vision = list(selected_ocr_pages)
        elif selected_ocr_pages and cfg.ocr_mode == "vision_llm":
            escalated_to_vision = list(selected_ocr_pages)

        should_run_vision = bool(escalated_to_vision) and (
            cfg.ocr_mode == "vision_llm" or (cfg.ocr_mode == "auto" and cfg.vision_fallback_enabled)
        )
        if should_run_vision:
            effective_vision_fn = vision_ocr_fn
            if effective_vision_fn is None:
                effective_vision_fn = build_vision_ocr_fn(
                    resolved_settings,
                    cost_tracker=cost_tracker,
                    model_override=cfg.vision_model,
                )

            if effective_vision_fn is None:
                if cfg.ocr_mode == "vision_llm":
                    raise ValueError(
                        "Vision-LLM OCR mode requires configured LLM credentials "
                        "(e.g., OPENROUTER_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY)."
                    )
            else:
                shots = await asyncio.to_thread(
                    _render_page_screenshots,
                    data,
                    escalated_to_vision,
                    dpi=cfg.dpi,
                    model_dir=model_dir,
                )
                vision_used = False
                for pno in escalated_to_vision:
                    shot_entry = shots.get(pno)
                    if shot_entry is None:
                        continue
                    png_bytes, is_solid_fill = shot_entry
                    if is_solid_fill or not png_bytes:
                        continue
                    raw_out = effective_vision_fn(pno, png_bytes)
                    if inspect.isawaitable(raw_out):
                        v_text, v_usage = await raw_out
                    else:
                        v_text, v_usage = raw_out
                    cumulative_usage = cumulative_usage.add(v_usage)
                    cleaned_v = _clean_page_markdown(v_text)
                    vision_used = True
                    if _visible_char_count(cleaned_v) >= _visible_char_count(page_md.get(pno, "")):
                        page_md[pno] = cleaned_v
                    if _visible_char_count(page_md.get(pno, "")) > 0:
                        ocr_pages_processed.add(pno)

                if vision_used and "vision_llm" not in backends_used:
                    backends_used.append("vision_llm")

    non_empty_pages: list[tuple[int, str]] = [
        (pno, md) for pno, md in sorted(page_md.items()) if _visible_char_count(md) > 0
    ]
    has_structural_headings = any(_HEADING_LINE_RE.search(md) for _, md in non_empty_pages)

    if has_structural_headings:
        assembled = "\n\n".join(md for _, md in non_empty_pages)
    else:
        assembled = "\n\n".join(f"## Page {pno}\n\n{md}" for pno, md in non_empty_pages)

    digital_pages = sum(1 for pno, _ in non_empty_pages if pno not in ocr_pages_processed)
    meta: dict[str, Any] = {
        "page_count": total_pages,
        "digital_pages": digital_pages,
        "ocr_pages": len(ocr_pages_processed),
        "ocr_page_numbers": sorted(ocr_pages_processed),
        "skipped_ocr_pages": skipped_ocr_pages,
        "pdf_parser": "liteparse",
        "ocr_mode": cfg.ocr_mode,
        "ocr_backends_used": backends_used,
        "ocr_cost_usd": round(cumulative_usage.estimated_cost_usd, 8),
    }
    if pdf_title:
        meta["pdf_title"] = pdf_title

    return PdfParseOutput(
        markdown=assembled.strip(),
        metadata=meta,
        usage=cumulative_usage,
    )


def parse_pdf_bytes(
    data: bytes,
    *,
    config: PdfParseConfig | None = None,
    vision_ocr_fn: VisionOcrFn | None = None,
    settings: Settings | None = None,
    cost_tracker: CostTracker | None = None,
) -> PdfParseOutput:
    """Synchronous entry point for `aparse_pdf_bytes` (safe outside running event loops)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            aparse_pdf_bytes(
                data,
                config=config,
                vision_ocr_fn=vision_ocr_fn,
                settings=settings,
                cost_tracker=cost_tracker,
            )
        )

    # Fallback if invoked synchronously inside a thread that happens to host an active loop:
    # run in a dedicated worker thread with its own event loop.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            asyncio.run,
            aparse_pdf_bytes(
                data,
                config=config,
                vision_ocr_fn=vision_ocr_fn,
                settings=settings,
                cost_tracker=cost_tracker,
            ),
        )
        return future.result()
