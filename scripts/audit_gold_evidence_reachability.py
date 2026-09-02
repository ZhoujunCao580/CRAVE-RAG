"""Audit MMLongBench-Doc Gold-page reachability in serialized SoftDocs.

This stage-3A entry point is deliberately read-only with respect to source PDFs
and SoftDocs.  Every invocation creates a new run directory and refuses to
overwrite an existing run.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

from PIL import Image
import pypdfium2 as pdfium

from softdoc.external_data import MMLongBenchDocAdapter
from softdoc.models import Document, Element, ElementType, Page
from softdoc.retrieval import SearchUnitBuilder, html_to_text
from softdoc.serialization import load_document
from softdoc.store import DocumentStore


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = ROOT / "data" / "raw" / "mmlongbench_doc" / "questions.json"
DEFAULT_DOCUMENTS = ROOT / "data" / "raw" / "mmlongbench_doc" / "documents"
DEFAULT_SOFTDOCS = ROOT / "data" / "processed" / "mmlongbench_doc" / "softdocs"
DEFAULT_PROGRESS = ROOT / ".runlogs" / "stage3_mineru_local" / "progress.json"
DEFAULT_OUTPUT_ROOT = ROOT / ".runlogs" / "stage3_softdoc_audit" / "runs"
DEFAULT_GOLD_PAGE_RESOLUTION = (
    ROOT
    / "configs"
    / "evaluation"
    / "mmlongbench_doc_gold_page_resolution_v0_1.json"
)
AUDIT_VERSION = "gold-evidence-reachability-v0.2"


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--documents-root", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--softdocs-root", type=Path, default=DEFAULT_SOFTDOCS)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--gold-page-resolution",
        type=Path,
        default=DEFAULT_GOLD_PAGE_RESOLUTION,
    )
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)

    questions_path = args.questions.resolve()
    documents_root = args.documents_root.resolve()
    softdocs_root = args.softdocs_root.resolve()
    progress_path = args.progress.resolve()
    output_root = args.output_root.resolve()
    gold_page_resolution_path = args.gold_page_resolution.resolve()
    gold_page_resolutions = _load_gold_page_resolutions(
        gold_page_resolution_path
    )
    run_id = args.run_id or (
        "reachability-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("run-id may contain only letters, numbers, dot, underscore, and dash")
    run_dir = output_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True)

    started_at = datetime.now(timezone.utc).isoformat()
    input_snapshot = {
        "audit_version": AUDIT_VERSION,
        "run_id": run_id,
        "started_at": started_at,
        "questions_path": str(questions_path),
        "questions_sha256": _sha256(questions_path),
        "documents_root": str(documents_root),
        "softdocs_root": str(softdocs_root),
        "progress_path": str(progress_path),
        "progress": _read_json_if_present(progress_path),
        "gold_page_resolution_path": str(gold_page_resolution_path),
        "gold_page_resolution_sha256": _sha256(gold_page_resolution_path),
    }
    _write_json(run_dir / "run_state.json", {**input_snapshot, "state": "running"})

    adapter = MMLongBenchDocAdapter()
    manifest = adapter.build_manifest(
        questions_path=questions_path,
        documents_root=documents_root,
        softdocs_root=softdocs_root,
        path_root=ROOT,
        manifest_path=ROOT / "external_dataset_manifest.json",
        hash_sources=False,
    )
    # The adapter normally writes a relative root beside its manifest.  Stage 3A
    # permits an output root on another Windows drive, so persist the canonical
    # absolute root instead of attempting an invalid cross-drive relative path.
    manifest.path_root = ROOT
    _write_json(
        run_dir / "external_dataset_manifest.json",
        manifest.model_dump(mode="json"),
    )

    documents = {item.document_id: item for item in manifest.documents}
    questions_by_document: dict[str, list[Any]] = defaultdict(list)
    for question in manifest.questions:
        questions_by_document[question.document_id].append(question)

    page_rows: list[dict[str, Any]] = []
    unreachable_rows: list[dict[str, Any]] = []
    document_summaries: list[dict[str, Any]] = []
    image_cache: dict[Path, bool] = {}

    for document_id, questions in questions_by_document.items():
        external_document = documents[document_id]
        source_pdf = _resolve_from_root(ROOT, external_document.source_path)
        softdoc_dir = _resolve_from_root(ROOT, external_document.softdoc_dir)
        document_audit = _audit_document(
            document_id=document_id,
            source_pdf=source_pdf,
            softdoc_dir=softdoc_dir,
        )
        document_summaries.append(document_audit["summary"])
        document = document_audit["document"]
        pages_by_number: dict[int, Page] = document_audit["pages_by_number"]
        elements_by_page: dict[str, list[Element]] = document_audit["elements_by_page"]
        search_units_by_page: dict[str, list[Any]] = document_audit[
            "search_units_by_page"
        ]

        for question in questions:
            question_page_rows: list[dict[str, Any]] = []
            raw_gold_pages = list(question.evidence_pages)
            resolution = gold_page_resolutions.get(question.case_id)
            gold_pages = _resolved_gold_pages(
                question.case_id,
                raw_gold_pages,
                resolution,
            )
            pages_to_check: list[int | None] = gold_pages or [None]
            for gold_page in pages_to_check:
                row = _audit_question_page(
                    question=question,
                    gold_page=gold_page,
                    source_pdf=source_pdf,
                    softdoc_dir=softdoc_dir,
                    document=document,
                    document_audit=document_audit,
                    pages_by_number=pages_by_number,
                    elements_by_page=elements_by_page,
                    search_units_by_page=search_units_by_page,
                    image_cache=image_cache,
                    raw_gold_pages=raw_gold_pages,
                    gold_page_resolution=resolution,
                )
                question_page_rows.append(row)

            page_reachability = [
                bool(row["reachable"])
                for row in question_page_rows
                if row["gold_page_id"] is not None
            ]
            all_reachable = bool(page_reachability) and all(page_reachability)
            any_reachable = any(page_reachability)
            for row in question_page_rows:
                row["all_gold_pages_reachable"] = all_reachable
                row["any_gold_page_reachable"] = any_reachable
                page_rows.append(row)

            if gold_pages and not all_reachable:
                unreachable_rows.append(
                    {
                        "question_id": question.question_id,
                        "case_id": question.case_id,
                        "question_index": question.metadata.get("source_index"),
                        "document_id": question.document_id,
                        "raw_gold_page_ids": raw_gold_pages,
                        "gold_page_ids": gold_pages,
                        "gold_page_resolution": resolution,
                        "reachable_gold_page_ids": [
                            row["gold_page_id"]
                            for row in question_page_rows
                            if row["reachable"]
                        ],
                        "unreachable_gold_page_ids": [
                            row["gold_page_id"]
                            for row in question_page_rows
                            if not row["reachable"]
                        ],
                        "all_gold_pages_reachable": all_reachable,
                        "any_gold_page_reachable": any_reachable,
                        "failure_reasons": sorted(
                            {
                                reason
                                for row in question_page_rows
                                for reason in row["failure_reasons"]
                            }
                        ),
                    }
                )

    summary = _build_summary(
        page_rows=page_rows,
        unreachable_rows=unreachable_rows,
        document_summaries=document_summaries,
        total_questions=len(manifest.questions),
        input_snapshot=input_snapshot,
    )
    _write_jsonl(run_dir / "gold_evidence_reachability.jsonl", page_rows)
    _write_json(run_dir / "gold_evidence_reachability_summary.json", summary)
    _write_jsonl(run_dir / "gold_evidence_unreachable.jsonl", unreachable_rows)
    completed_at = datetime.now(timezone.utc).isoformat()
    _write_json(
        run_dir / "run_state.json",
        {
            **input_snapshot,
            "state": "completed",
            "completed_at": completed_at,
            "outputs": {
                "reachability": "gold_evidence_reachability.jsonl",
                "summary": "gold_evidence_reachability_summary.json",
                "unreachable": "gold_evidence_unreachable.jsonl",
            },
        },
    )
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    print(run_dir)
    return 0


def _audit_document(
    *, document_id: str, source_pdf: Path, softdoc_dir: Path
) -> dict[str, Any]:
    source_pdf_available = False
    source_pdf_page_count: int | None = None
    source_pdf_error: str | None = None
    if not source_pdf.is_file():
        source_pdf_error = "source_pdf_missing"
    else:
        try:
            pdf = pdfium.PdfDocument(str(source_pdf))
            try:
                source_pdf_page_count = len(pdf)
                source_pdf_available = source_pdf_page_count > 0
            finally:
                pdf.close()
            if not source_pdf_available:
                source_pdf_error = "source_pdf_empty"
        except Exception as exc:
            source_pdf_error = f"source_pdf_unreadable:{type(exc).__name__}:{exc}"

    document: Document | None = None
    softdoc_available = False
    softdoc_references_valid = False
    softdoc_source_identity_matches = False
    softdoc_error: str | None = None
    reference_errors: list[str] = []
    if not (softdoc_dir / "document.json").is_file():
        softdoc_error = "softdoc_missing"
    else:
        try:
            document = load_document(softdoc_dir)
            softdoc_available = True
            reference_errors = DocumentStore(document).validate_references()
            softdoc_references_valid = not reference_errors
            softdoc_source_identity_matches = (
                _document_key(document.source_path.name) == _document_key(document_id)
            )
        except Exception as exc:
            softdoc_error = f"softdoc_load_error:{type(exc).__name__}:{exc}"

    pages_by_number: dict[int, Page] = {}
    elements_by_page: dict[str, list[Element]] = defaultdict(list)
    search_units_by_page: dict[str, list[Any]] = defaultdict(list)
    search_unit_error: str | None = None
    if document is not None:
        pages_by_number = {page.page_number: page for page in document.pages}
        for element in document.elements:
            elements_by_page[element.page_id].append(element)
        try:
            build_result = SearchUnitBuilder().build(document)
            for unit in build_result.units:
                if unit.search_text.strip() and unit.content_text.strip():
                    search_units_by_page[unit.page_id].append(unit)
        except Exception as exc:
            search_unit_error = f"search_unit_build_error:{type(exc).__name__}:{exc}"

    summary = {
        "document_id": document_id,
        "source_pdf_path": str(source_pdf),
        "source_pdf_available": source_pdf_available,
        "source_pdf_page_count": source_pdf_page_count,
        "source_pdf_error": source_pdf_error,
        "softdoc_dir": str(softdoc_dir),
        "softdoc_available": softdoc_available,
        "softdoc_document_id": document.document_id if document else None,
        "softdoc_source_path": str(document.source_path) if document else None,
        "softdoc_source_identity_matches": softdoc_source_identity_matches,
        "softdoc_references_valid": softdoc_references_valid,
        "softdoc_reference_error_count": len(reference_errors),
        "softdoc_reference_errors": reference_errors,
        "softdoc_error": softdoc_error,
        "search_unit_error": search_unit_error,
        "softdoc_page_count": len(document.pages) if document else None,
        "search_unit_count": sum(len(items) for items in search_units_by_page.values()),
    }
    return {
        "summary": summary,
        "document": document,
        "pages_by_number": pages_by_number,
        "elements_by_page": elements_by_page,
        "search_units_by_page": search_units_by_page,
    }


def _audit_question_page(
    *,
    question: Any,
    gold_page: int | None,
    source_pdf: Path,
    softdoc_dir: Path,
    document: Document | None,
    document_audit: dict[str, Any],
    pages_by_number: dict[int, Page],
    elements_by_page: dict[str, list[Element]],
    search_units_by_page: dict[str, list[Any]],
    image_cache: dict[Path, bool],
    raw_gold_pages: list[int],
    gold_page_resolution: dict[str, Any] | None,
) -> dict[str, Any]:
    summary = document_audit["summary"]
    page = pages_by_number.get(gold_page) if gold_page is not None else None
    elements = elements_by_page.get(page.page_id, []) if page else []
    readable_text_or_table = any(_has_readable_text_or_table(item) for item in elements)
    visual_paths: list[Path] = []
    if page is not None:
        if page.image_path is not None:
            visual_paths.append(_resolve_asset(softdoc_dir, page.image_path))
        for element in elements:
            for value in (element.image_path, element.crop_image_path):
                if value is not None:
                    visual_paths.append(_resolve_asset(softdoc_dir, value))
    readable_visual_paths = [
        path for path in dict.fromkeys(visual_paths) if _image_decodable(path, image_cache)
    ]
    readable_visual_asset = bool(readable_visual_paths)
    nonempty_search_units = search_units_by_page.get(page.page_id, []) if page else []
    nonempty_search_unit = bool(nonempty_search_units)
    source_pdf_available = bool(summary["source_pdf_available"])
    source_page_count = summary["source_pdf_page_count"]
    gold_page_in_source_pdf = bool(
        gold_page is not None
        and source_page_count is not None
        and 1 <= gold_page <= source_page_count
    )
    softdoc_available = bool(summary["softdoc_available"])
    gold_page_mapped = page is not None

    failure_reasons: list[str] = []
    if gold_page is None:
        failure_reasons.append("no_gold_evidence_pages")
    else:
        if not source_pdf_available:
            failure_reasons.append(
                "source_pdf_missing"
                if summary["source_pdf_error"] == "source_pdf_missing"
                else "source_pdf_unreadable"
            )
        elif not gold_page_in_source_pdf:
            failure_reasons.append(
                "source_pdf_variant_incomplete"
                if gold_page_resolution
                and gold_page_resolution.get("status")
                == "source_variant_mismatch"
                else "gold_page_out_of_source_pdf_range"
            )
        if not softdoc_available:
            failure_reasons.append(
                "softdoc_missing"
                if summary["softdoc_error"] == "softdoc_missing"
                else "softdoc_load_error"
            )
        else:
            if not summary["softdoc_references_valid"]:
                failure_reasons.append("softdoc_reference_error")
            if not summary["softdoc_source_identity_matches"]:
                failure_reasons.append("softdoc_source_identity_mismatch")
        if (
            softdoc_available
            and not gold_page_mapped
            and not (
                gold_page_resolution
                and gold_page_resolution.get("status")
                == "source_variant_mismatch"
            )
        ):
            failure_reasons.append("gold_page_not_mapped")
        if gold_page_mapped and not (readable_text_or_table or readable_visual_asset):
            failure_reasons.append("no_readable_text_table_or_visual")
        if gold_page_mapped and not nonempty_search_unit:
            failure_reasons.append("no_nonempty_search_unit")
        if summary["search_unit_error"]:
            failure_reasons.append("search_unit_build_error")

    reachable = bool(
        gold_page is not None
        and source_pdf_available
        and gold_page_in_source_pdf
        and softdoc_available
        and summary["softdoc_references_valid"]
        and summary["softdoc_source_identity_matches"]
        and gold_page_mapped
        and (readable_text_or_table or readable_visual_asset)
        and nonempty_search_unit
        and not summary["search_unit_error"]
    )
    return {
        "question_id": question.question_id,
        "case_id": question.case_id,
        "question_index": question.metadata.get("source_index"),
        "document_id": question.document_id,
        "softdoc_document_id": document.document_id if document else None,
        "gold_page_id": gold_page,
        "raw_gold_page_ids": raw_gold_pages,
        "gold_page_resolution": gold_page_resolution,
        "audit_applicable": gold_page is not None,
        "source_pdf_path": str(source_pdf),
        "source_pdf_available": source_pdf_available,
        "gold_page_in_source_pdf": gold_page_in_source_pdf,
        "softdoc_dir": str(softdoc_dir),
        "softdoc_available": softdoc_available,
        "softdoc_references_valid": bool(summary["softdoc_references_valid"]),
        "softdoc_source_identity_matches": bool(
            summary["softdoc_source_identity_matches"]
        ),
        "gold_page_mapped": gold_page_mapped,
        "softdoc_page_id": page.page_id if page else None,
        "readable_text_or_table": readable_text_or_table,
        "readable_visual_asset": readable_visual_asset,
        "readable_visual_asset_paths": [str(path) for path in readable_visual_paths],
        "nonempty_search_unit": nonempty_search_unit,
        "nonempty_search_unit_count": len(nonempty_search_units),
        "reachable": reachable,
        "failure_reasons": failure_reasons,
    }


def _load_gold_page_resolutions(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != (
        "mmlongbench-doc-gold-page-resolution-v0.1"
    ):
        raise ValueError(f"Unsupported Gold-page resolution schema: {path}")
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise ValueError("Gold-page resolution entries must be an object")
    return entries


def _resolved_gold_pages(
    case_id: str,
    raw_gold_pages: list[int],
    resolution: dict[str, Any] | None,
) -> list[int]:
    if resolution is None:
        return raw_gold_pages
    expected = resolution.get("raw_evidence_pages")
    if expected != raw_gold_pages:
        raise ValueError(
            f"Gold-page resolution for {case_id} is stale: "
            f"expected {expected}, observed {raw_gold_pages}"
        )
    resolved = resolution.get("resolved_evidence_pages")
    if not isinstance(resolved, list) or not all(
        isinstance(page, int) for page in resolved
    ):
        raise ValueError(
            f"Gold-page resolution for {case_id} must contain integer pages"
        )
    return list(dict.fromkeys(resolved))


def _build_summary(
    *,
    page_rows: list[dict[str, Any]],
    unreachable_rows: list[dict[str, Any]],
    document_summaries: list[dict[str, Any]],
    total_questions: int,
    input_snapshot: dict[str, Any],
) -> dict[str, Any]:
    question_flags: dict[str, tuple[bool, bool]] = {}
    for row in page_rows:
        question_flags[row["question_id"]] = (
            bool(row["all_gold_pages_reachable"]),
            bool(row["any_gold_page_reachable"]),
        )
    total_gold_page_checks = sum(row["gold_page_id"] is not None for row in page_rows)
    reachable_page_checks = sum(
        row["gold_page_id"] is not None and row["reachable"] for row in page_rows
    )
    fully_reachable = sum(all_reachable for all_reachable, _ in question_flags.values())
    partially_reachable = sum(
        (not all_reachable) and any_reachable
        for all_reachable, any_reachable in question_flags.values()
    )
    no_gold = sum(row["gold_page_id"] is None for row in page_rows)
    failure_counts = Counter(
        reason
        for row in page_rows
        if row["gold_page_id"] is not None and not row["reachable"]
        for reason in row["failure_reasons"]
    )
    questions_with_gold = total_questions - no_gold
    not_fully_reachable_with_gold = questions_with_gold - fully_reachable
    zero_reachable_with_gold = (
        questions_with_gold - fully_reachable - partially_reachable
    )
    counts = {
        "total_questions": total_questions,
        "questions_with_gold_pages": questions_with_gold,
        "questions_without_gold_pages": no_gold,
        "fully_reachable_questions": fully_reachable,
        "partially_reachable_questions": partially_reachable,
        "zero_reachable_questions_with_gold_pages": zero_reachable_with_gold,
        "not_fully_reachable_questions": not_fully_reachable_with_gold,
        "total_gold_page_checks": total_gold_page_checks,
        "reachable_gold_page_checks": reachable_page_checks,
        "unreachable_gold_page_checks": total_gold_page_checks - reachable_page_checks,
        "gold_page_reachability_rate": (
            reachable_page_checks / total_gold_page_checks
            if total_gold_page_checks
            else None
        ),
        "question_all_pages_reachability_rate": (
            fully_reachable / questions_with_gold if questions_with_gold else None
        ),
        "referenced_documents": len(document_summaries),
        "source_pdf_available_documents": sum(
            item["source_pdf_available"] for item in document_summaries
        ),
        "softdoc_available_documents": sum(
            item["softdoc_available"] for item in document_summaries
        ),
        "softdoc_reference_valid_documents": sum(
            item["softdoc_references_valid"] for item in document_summaries
        ),
        "softdoc_source_identity_matching_documents": sum(
            item["softdoc_source_identity_matches"] for item in document_summaries
        ),
    }
    return {
        "audit_version": AUDIT_VERSION,
        "run_id": input_snapshot["run_id"],
        "started_at": input_snapshot["started_at"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "inputs": input_snapshot,
        "counts": counts,
        "failure_reason_counts": dict(sorted(failure_counts.items())),
        "documents": document_summaries,
    }


def _has_readable_text_or_table(element: Element) -> bool:
    if (element.text or "").strip():
        return True
    return bool(
        element.element_type == ElementType.TABLE
        and html_to_text(element.html or "").strip()
    )


def _image_decodable(path: Path, cache: dict[Path, bool]) -> bool:
    path = path.resolve()
    if path not in cache:
        try:
            with Image.open(path) as image:
                image.verify()
            cache[path] = True
        except Exception:
            cache[path] = False
    return cache[path]


def _resolve_asset(softdoc_dir: Path, value: Path) -> Path:
    return value if value.is_absolute() else softdoc_dir / value


def _resolve_from_root(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _document_key(value: str) -> str:
    stem = Path(value).stem.casefold()
    return stem[: -len("_origin")] if stem.endswith("_origin") else stem


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_if_present(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
