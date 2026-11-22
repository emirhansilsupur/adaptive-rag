"""Golden dataset loaders and external JSONL / HuggingFace benchmark adapters."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from adaptiverag.ingestion.loader import LoadedDocument
from adaptiverag.schemas.query import QueryType
from eval.schemas import EvalStratum, GoldenQuery

_NON_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str, fallback: str = "doc") -> str:
    slug = _NON_SLUG_RE.sub("_", text.strip().lower()).strip("_")
    return slug[:80] or fallback


def load_golden_jsonl(path: Path) -> list[GoldenQuery]:
    """Load and validate an AdaptiveRAG golden dataset from JSONL."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Golden dataset not found: {resolved}")

    queries: list[GoldenQuery] = []
    seen_ids: set[str] = set()
    with resolved.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {resolved.name}:{line_no}: {exc}") from exc
            item = GoldenQuery.model_validate(payload)
            if item.id in seen_ids:
                raise ValueError(f"Duplicate golden query id '{item.id}' at line {line_no}.")
            seen_ids.add(item.id)
            queries.append(item)

    if not queries:
        raise ValueError(f"Golden dataset '{resolved}' contains no valid records.")
    return queries


def _first_present(row: Mapping[str, Any], candidates: Sequence[str]) -> Any:
    for key in candidates:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _coerce_doc_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if isinstance(value, Sequence):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def adapt_generic_row(
    row: Mapping[str, Any],
    *,
    index: int,
    query_field: str | None = None,
    answer_field: str | None = None,
    doc_ids_field: str | None = None,
    default_stratum: EvalStratum = EvalStratum.FACTUAL,
    default_language: str = "en",
) -> GoldenQuery:
    """Adapt an arbitrary QA JSON/HF row into a validated GoldenQuery."""
    query_keys = ([query_field] if query_field else []) + [
        "query",
        "question",
        "user_input",
        "input",
        "prompt",
    ]
    answer_keys = ([answer_field] if answer_field else []) + [
        "reference_answer",
        "reference",
        "answer",
        "ground_truth",
        "response",
        "expected_answer",
    ]
    doc_keys = ([doc_ids_field] if doc_ids_field else []) + [
        "expected_doc_ids",
        "doc_ids",
        "relevant_doc_ids",
        "document_ids",
    ]

    raw_query = _first_present(row, query_keys)
    if not raw_query or not str(raw_query).strip():
        raise ValueError(f"Row {index} is missing a non-empty query field (tried {query_keys}).")

    raw_answer = _first_present(row, answer_keys)
    if isinstance(raw_answer, Sequence) and not isinstance(raw_answer, str):
        reference_answer = " ".join(str(x).strip() for x in raw_answer if str(x).strip())
    else:
        reference_answer = str(raw_answer).strip() if raw_answer is not None else ""

    raw_id = _first_present(row, ["id", "qid", "question_id", "_id"])
    query_id = (
        str(raw_id).strip() if raw_id is not None and str(raw_id).strip() else f"ext_{index:04d}"
    )

    raw_stratum = row.get("stratum")
    if isinstance(raw_stratum, str) and raw_stratum.lower() in {s.value for s in EvalStratum}:
        stratum = EvalStratum(raw_stratum.lower())
    else:
        stratum = default_stratum

    raw_qtype = row.get("expected_query_type") or row.get("query_type")
    if isinstance(raw_qtype, str) and raw_qtype.lower() in {t.value for t in QueryType}:
        expected_qtype = QueryType(raw_qtype.lower())
    elif stratum == EvalStratum.ANALYTICAL:
        expected_qtype = QueryType.ANALYTICAL
    elif stratum == EvalStratum.CONVERSATIONAL:
        expected_qtype = QueryType.CONVERSATIONAL
    else:
        expected_qtype = QueryType.FACTUAL

    needs_retrieval = bool(row.get("needs_retrieval", stratum != EvalStratum.CONVERSATIONAL))
    if stratum == EvalStratum.CONVERSATIONAL:
        needs_retrieval = False

    expected_doc_ids = _coerce_doc_ids(_first_present(row, doc_keys))
    language = str(row.get("language") or default_language).strip().lower() or "en"
    key_facts = _coerce_doc_ids(row.get("key_facts"))

    return GoldenQuery(
        id=query_id,
        query=str(raw_query).strip(),
        language=language,
        stratum=stratum,
        expected_query_type=expected_qtype,
        needs_retrieval=needs_retrieval,
        expected_doc_ids=expected_doc_ids,
        reference_answer=reference_answer,
        key_facts=key_facts,
        expect_web_fallback=bool(
            row.get("expect_web_fallback", stratum == EvalStratum.WEB_FALLBACK)
        ),
        expect_abstention=bool(row.get("expect_abstention", stratum == EvalStratum.UNANSWERABLE)),
    )


def load_external_jsonl(
    path: Path,
    *,
    query_field: str | None = None,
    answer_field: str | None = None,
    doc_ids_field: str | None = None,
    default_stratum: EvalStratum = EvalStratum.FACTUAL,
    default_language: str = "en",
) -> list[GoldenQuery]:
    """Load an external JSONL file using flexible field-name mapping."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"External JSONL dataset not found: {resolved}")

    queries: list[GoldenQuery] = []
    seen_ids: set[str] = set()
    with resolved.open("r", encoding="utf-8") as handle:
        for idx, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, Mapping):
                raise ValueError(f"Expected JSON object at {resolved.name}:{idx}.")
            item = adapt_generic_row(
                row,
                index=idx,
                query_field=query_field,
                answer_field=answer_field,
                doc_ids_field=doc_ids_field,
                default_stratum=default_stratum,
                default_language=default_language,
            )
            if item.id in seen_ids:
                item = item.model_copy(update={"id": f"{item.id}_{idx}"})
            seen_ids.add(item.id)
            queries.append(item)

    if not queries:
        raise ValueError(f"External JSONL dataset '{resolved}' contains no valid records.")
    return queries


def adapt_hotpotqa_row(
    row: Mapping[str, Any],
    *,
    index: int,
) -> tuple[GoldenQuery, list[LoadedDocument]]:
    """Adapt a HuggingFace HotpotQA row into a multi-hop GoldenQuery and companion documents."""
    qid = str(row.get("id") or f"hotpot_{index:04d}").strip()
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    if not question:
        raise ValueError(f"HotpotQA row {index} missing 'question'.")

    qtype_raw = str(row.get("type") or "").strip().lower()
    is_multi_hop = qtype_raw in {"comparison", "bridge"} or True
    stratum = EvalStratum.ANALYTICAL if is_multi_hop else EvalStratum.FACTUAL
    expected_qtype = QueryType.ANALYTICAL if is_multi_hop else QueryType.FACTUAL

    supporting = row.get("supporting_facts") or {}
    supporting_titles: list[str] = []
    if isinstance(supporting, Mapping):
        supporting_titles = [str(t).strip() for t in supporting.get("title", []) if str(t).strip()]
    elif isinstance(supporting, Sequence):
        for entry in supporting:
            if isinstance(entry, Sequence) and entry:
                supporting_titles.append(str(entry[0]).strip())

    # Deduplicate supporting titles preserving order
    expected_doc_ids = list(
        dict.fromkeys(_slugify(t, f"hotpot_doc_{i}") for i, t in enumerate(supporting_titles))
    )

    docs: list[LoadedDocument] = []
    context = row.get("context") or {}
    if isinstance(context, Mapping):
        titles = context.get("title") or []
        sentences_list = context.get("sentences") or []
        for idx_doc, (title_raw, sents) in enumerate(zip(titles, sentences_list, strict=False)):
            title = str(title_raw).strip() or f"Hotpot Doc {idx_doc}"
            body = " ".join(str(s).strip() for s in sents if str(s).strip())
            if not body:
                continue
            doc_id = _slugify(title, f"hotpot_{index}_{idx_doc}")
            docs.append(
                LoadedDocument(
                    doc_id=doc_id,
                    title=title,
                    content=f"# {title}\n\n{body}",
                    source_path=f"hf://hotpot_qa/{doc_id}.md",
                    doc_type="markdown",
                    language="en",
                    metadata={"source_dataset": "hotpot_qa", "question_id": qid},
                )
            )

    golden = GoldenQuery(
        id=qid,
        query=question,
        language="en",
        stratum=stratum,
        expected_query_type=expected_qtype,
        needs_retrieval=True,
        expected_doc_ids=expected_doc_ids,
        reference_answer=answer,
        key_facts=[answer] if answer else [],
        expect_web_fallback=False,
        expect_abstention=False,
    )
    return golden, docs


def adapt_ragbench_row(
    row: Mapping[str, Any],
    *,
    index: int,
) -> tuple[GoldenQuery, list[LoadedDocument]]:
    """Adapt a HuggingFace RAGBench (rungalileo/ragbench) row into GoldenQuery + LoadedDocuments."""
    qid = str(row.get("id") or f"ragbench_{index:04d}").strip()
    question = str(row.get("question") or "").strip()
    answer = str(row.get("response") or row.get("answer") or "").strip()
    if not question:
        raise ValueError(f"RAGBench row {index} missing 'question'.")

    raw_docs = row.get("documents") or []
    docs: list[LoadedDocument] = []
    expected_doc_ids: list[str] = []

    if isinstance(raw_docs, Sequence) and not isinstance(raw_docs, str):
        for doc_idx, passage in enumerate(raw_docs):
            text = str(passage).strip()
            if not text:
                continue
            doc_id = _slugify(f"{qid}_doc_{doc_idx}", f"ragbench_{index}_{doc_idx}")
            expected_doc_ids.append(doc_id)
            docs.append(
                LoadedDocument(
                    doc_id=doc_id,
                    title=f"RAGBench {qid} Passage {doc_idx + 1}",
                    content=f"# RAGBench {qid} Passage {doc_idx + 1}\n\n{text}",
                    source_path=f"hf://ragbench/{doc_id}.md",
                    doc_type="markdown",
                    language="en",
                    metadata={"source_dataset": "ragbench", "question_id": qid},
                )
            )

    is_multi_doc = len(expected_doc_ids) >= 2
    golden = GoldenQuery(
        id=qid,
        query=question,
        language="en",
        stratum=EvalStratum.ANALYTICAL if is_multi_doc else EvalStratum.FACTUAL,
        expected_query_type=QueryType.ANALYTICAL if is_multi_doc else QueryType.FACTUAL,
        needs_retrieval=True,
        expected_doc_ids=expected_doc_ids,
        reference_answer=answer,
        key_facts=[],
        expect_web_fallback=False,
        expect_abstention=False,
    )
    return golden, docs


def _detect_hf_preset(
    dataset_name: str, sample_row: Mapping[str, Any]
) -> Literal["hotpotqa", "ragbench", "generic"]:
    lowered = dataset_name.lower()
    if "hotpot" in lowered or ("supporting_facts" in sample_row and "context" in sample_row):
        return "hotpotqa"
    if "ragbench" in lowered or ("documents" in sample_row and "question" in sample_row):
        return "ragbench"
    return "generic"


def load_huggingface_dataset(
    dataset_name: str,
    *,
    subset: str | None = None,
    split: str = "validation",
    limit: int = 50,
    preset: Literal["auto", "hotpotqa", "ragbench", "generic"] = "auto",
    export_corpus_dir: Path | None = None,
) -> tuple[list[GoldenQuery], list[LoadedDocument]]:
    """Load and adapt a HuggingFace QA dataset into GoldenQuery items and companion documents.

    Requires the ``eval`` optional dependency group (``datasets``). Companion context
    passages are deduplicated by ``doc_id`` and optionally written as Markdown files
    to ``export_corpus_dir`` for reproducible Qdrant ingestion.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, subset, split=split)
    rows: list[Mapping[str, Any]] = []
    for idx, row in enumerate(ds):
        if limit > 0 and idx >= limit:
            break
        if isinstance(row, Mapping):
            rows.append(row)

    if not rows:
        raise ValueError(f"HuggingFace dataset '{dataset_name}' ({split}) yielded zero rows.")

    resolved_preset: Literal["hotpotqa", "ragbench", "generic"] = (
        _detect_hf_preset(dataset_name, rows[0]) if preset == "auto" else preset
    )

    queries: list[GoldenQuery] = []
    docs_by_id: dict[str, LoadedDocument] = {}

    for idx, row in enumerate(rows, start=1):
        if resolved_preset == "hotpotqa":
            q, row_docs = adapt_hotpotqa_row(row, index=idx)
        elif resolved_preset == "ragbench":
            q, row_docs = adapt_ragbench_row(row, index=idx)
        else:
            q = adapt_generic_row(row, index=idx)
            row_docs = []
        queries.append(q)
        for doc in row_docs:
            docs_by_id.setdefault(doc.doc_id, doc)

    documents = list(docs_by_id.values())
    if export_corpus_dir is not None and documents:
        export_corpus_dir.mkdir(parents=True, exist_ok=True)
        for doc in documents:
            target = export_corpus_dir / f"{doc.doc_id}.md"
            target.write_text(doc.content, encoding="utf-8")

    return queries, documents


def filter_queries(
    queries: Iterable[GoldenQuery],
    *,
    strata: Sequence[str] | None = None,
    language: str | None = None,
    limit: int | None = None,
) -> list[GoldenQuery]:
    """Filter golden queries deterministically by stratum, language, and max count."""
    allowed_strata = {s.strip().lower() for s in strata} if strata else None
    lang_norm = language.strip().lower() if language else None

    filtered: list[GoldenQuery] = []
    for item in queries:
        if allowed_strata is not None and item.stratum.value not in allowed_strata:
            continue
        if lang_norm is not None and item.language != lang_norm:
            continue
        filtered.append(item)

    if limit is not None and limit > 0:
        return filtered[:limit]
    return filtered
