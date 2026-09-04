"""Portable external-dataset manifests, adapters, and fail-fast auditing.

The reading runner consumes one question and one serialized SoftDoc at a time.
This module handles the separate ingestion concern: map a benchmark's native
layout into a canonical manifest, verify every referenced artifact, and export
Gold-free JSONL cases for ``scripts/run_model_batch.py``.
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal, Sequence

import pypdfium2 as pdfium
from pydantic import BaseModel, ConfigDict, Field, model_validator

from softdoc.serialization import load_document
from softdoc.store import DocumentStore


EXTERNAL_DATASET_MANIFEST_VERSION = "external-dataset-manifest-v0.1"
EXTERNAL_DATASET_AUDIT_VERSION = "external-dataset-audit-v0.1"
_IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class ExternalDataModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExternalDocument(ExternalDataModel):
    document_id: str = Field(min_length=1)
    source_kind: Literal["pdf", "image_directory"]
    source_path: Path
    softdoc_dir: Path
    content_sha256: str | None = None


class ExternalQuestion(ExternalDataModel):
    case_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    evidence_pages: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExternalDatasetManifest(ExternalDataModel):
    schema_version: Literal[EXTERNAL_DATASET_MANIFEST_VERSION] = (
        EXTERNAL_DATASET_MANIFEST_VERSION
    )
    dataset_id: str = Field(min_length=1)
    dataset_version: str | None = None
    adapter: str = Field(min_length=1)
    path_root: Path = Path(".")
    documents: list[ExternalDocument]
    questions: list[ExternalQuestion]

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "ExternalDatasetManifest":
        _require_unique([item.document_id for item in self.documents], "document_id")
        _require_unique([item.case_id for item in self.questions], "case_id")
        _require_unique([item.question_id for item in self.questions], "question_id")
        return self


class ExternalDataAuditIssue(ExternalDataModel):
    severity: Literal["error", "warning"]
    code: str
    message: str
    document_id: str | None = None
    question_id: str | None = None
    path: Path | None = None


class ExternalDataAuditReport(ExternalDataModel):
    schema_version: Literal[EXTERNAL_DATASET_AUDIT_VERSION] = (
        EXTERNAL_DATASET_AUDIT_VERSION
    )
    dataset_id: str
    status: Literal["passed", "failed"]
    document_count: int
    question_count: int
    audited_document_count: int
    audited_question_count: int
    issues: list[ExternalDataAuditIssue]


def load_external_dataset_manifest(path: Path) -> ExternalDatasetManifest:
    with Path(path).open("r", encoding="utf-8") as handle:
        return ExternalDatasetManifest.model_validate(json.load(handle))


def write_external_dataset_manifest(
    manifest: ExternalDatasetManifest,
    path: Path,
) -> None:
    _write_json(Path(path), manifest.model_dump(mode="json"))


def resolve_manifest_root(
    manifest: ExternalDatasetManifest,
    manifest_path: Path,
    *,
    path_root_override: Path | None = None,
) -> Path:
    if path_root_override is not None:
        return Path(path_root_override).resolve()
    root = manifest.path_root
    if not root.is_absolute():
        root = Path(manifest_path).resolve().parent / root
    return root.resolve()


def audit_external_dataset(
    manifest: ExternalDatasetManifest,
    *,
    manifest_path: Path,
    path_root_override: Path | None = None,
) -> ExternalDataAuditReport:
    """Audit all external files and SoftDoc references without silent skips."""

    root = resolve_manifest_root(
        manifest,
        manifest_path,
        path_root_override=path_root_override,
    )
    issues: list[ExternalDataAuditIssue] = []
    documents = {item.document_id: item for item in manifest.documents}
    page_counts: dict[str, int] = {}
    valid_document_ids: set[str] = set()
    audited_documents = 0

    for item in manifest.documents:
        source = _resolve(root, item.source_path)
        softdoc_dir = _resolve(root, item.softdoc_dir)
        source_ok = _audit_source(item, source, issues)
        softdoc_ok, softdoc_page_count = _audit_softdoc(
            item, softdoc_dir, source, issues
        )
        if source_ok:
            try:
                source_page_count = _source_page_count(
                    source, item.source_kind
                )
                page_counts[item.document_id] = source_page_count
                if source_page_count == 0:
                    issues.append(
                        _issue(
                            "error",
                            "source_has_no_pages",
                            "Source contains no readable PDF or image pages.",
                            document=item,
                            path=source,
                        )
                    )
                    source_ok = False
                if (
                    softdoc_page_count is not None
                    and source_page_count != softdoc_page_count
                ):
                    issues.append(
                        _issue(
                            "error",
                            "page_count_mismatch",
                            (
                                f"Source has {source_page_count} pages but SoftDoc "
                                f"has {softdoc_page_count}."
                            ),
                            document=item,
                            path=softdoc_dir / "document.json",
                        )
                    )
                    softdoc_ok = False
            except Exception as exc:
                issues.append(
                    _issue(
                        "error",
                        "source_unreadable",
                        f"Cannot count source pages: {type(exc).__name__}: {exc}",
                        document=item,
                        path=source,
                    )
                )
        if source_ok and softdoc_ok:
            audited_documents += 1
            valid_document_ids.add(item.document_id)

    audited_questions = 0
    for question in manifest.questions:
        document = documents.get(question.document_id)
        if document is None:
            issues.append(
                ExternalDataAuditIssue(
                    severity="error",
                    code="unknown_question_document",
                    message=(
                        f"Question references unknown document_id "
                        f"{question.document_id!r}."
                    ),
                    question_id=question.question_id,
                )
            )
            continue
        if question.document_id not in valid_document_ids:
            continue
        page_count = page_counts.get(question.document_id)
        zero_pages = [page for page in question.evidence_pages if page == 0]
        if zero_pages and manifest.dataset_id == "mmlongbench-doc":
            issues.append(
                ExternalDataAuditIssue(
                    severity="warning",
                    code="ambiguous_evidence_page_zero",
                    message=(
                        "MMLongBench-Doc contains mixed page-numbering conventions; "
                        "preserving Gold page 0 without silently converting it."
                    ),
                    document_id=question.document_id,
                    question_id=question.question_id,
                )
            )
        invalid_pages = [
            page
            for page in question.evidence_pages
            if page < 0
            or (
                page > 0
                and page_count is not None
                and page > page_count
            )
        ]
        if invalid_pages:
            if manifest.dataset_id == "mmlongbench-doc":
                issues.append(
                    ExternalDataAuditIssue(
                        severity="warning",
                        code="non_physical_evidence_page_reference",
                        message=(
                            f"MMLongBench-Doc Gold pages {invalid_pages} are outside "
                            f"the physical range 1..{page_count}; preserving the raw "
                            "annotation because this benchmark also uses printed page "
                            "labels and contains known delimiter errors."
                        ),
                        document_id=question.document_id,
                        question_id=question.question_id,
                    )
                )
            else:
                issues.append(
                    ExternalDataAuditIssue(
                        severity="error",
                        code="evidence_page_out_of_range",
                        message=(
                            f"Evidence pages {invalid_pages} are outside the "
                            f"1-based source-page range 1..{page_count}."
                        ),
                        document_id=question.document_id,
                        question_id=question.question_id,
                    )
                )
                continue
        audited_questions += 1

    status = "failed" if any(item.severity == "error" for item in issues) else "passed"
    return ExternalDataAuditReport(
        dataset_id=manifest.dataset_id,
        status=status,
        document_count=len(manifest.documents),
        question_count=len(manifest.questions),
        audited_document_count=audited_documents,
        audited_question_count=audited_questions,
        issues=issues,
    )


def export_batch_cases(
    manifest: ExternalDatasetManifest,
    *,
    manifest_path: Path,
    output: Path,
    path_root_override: Path | None = None,
) -> int:
    """Export runner input while deliberately excluding Gold-only fields."""

    root = resolve_manifest_root(
        manifest,
        manifest_path,
        path_root_override=path_root_override,
    )
    documents = {item.document_id: item for item in manifest.documents}
    rows: list[dict[str, str]] = []
    for question in manifest.questions:
        document = documents.get(question.document_id)
        if document is None:
            raise ValueError(
                f"Question {question.question_id!r} references unknown document "
                f"{question.document_id!r}"
            )
        rows.append(
            {
                "case_id": question.case_id,
                "question_id": question.question_id,
                "document_dir": str(_resolve(root, document.softdoc_dir)),
                "question": question.question,
            }
        )
    _write_jsonl(Path(output), rows)
    return len(rows)


class MMLongBenchDocAdapter:
    """Map native MMLongBench-Doc questions, PDFs, and SoftDocs to a manifest."""

    adapter_name = "mmlongbench-doc-v0.1"

    def build_manifest(
        self,
        *,
        questions_path: Path,
        documents_root: Path,
        softdocs_root: Path,
        path_root: Path,
        manifest_path: Path,
        question_indices: set[int] | None = None,
        hash_sources: bool = False,
    ) -> ExternalDatasetManifest:
        questions_path = Path(questions_path).resolve()
        documents_root = Path(documents_root).resolve()
        softdocs_root = Path(softdocs_root).resolve()
        path_root = Path(path_root).resolve()
        manifest_path = Path(manifest_path).resolve()
        with questions_path.open("r", encoding="utf-8-sig") as handle:
            native = json.load(handle)
        if not isinstance(native, list):
            raise ValueError("MMLongBench-Doc questions.json must contain a list")

        selected: list[tuple[int, dict[str, Any]]] = []
        for index, row in enumerate(native):
            if question_indices is not None and index not in question_indices:
                continue
            if not isinstance(row, dict):
                raise ValueError(f"questions.json index {index} is not an object")
            selected.append((index, row))
        if not selected:
            raise ValueError("No MMLongBench-Doc questions were selected")

        softdoc_index = _index_softdocs(softdocs_root)
        document_ids = list(dict.fromkeys(str(row.get("doc_id", "")) for _, row in selected))
        documents: list[ExternalDocument] = []
        for document_id in document_ids:
            if not document_id:
                raise ValueError("MMLongBench-Doc row has a blank doc_id")
            source = documents_root / document_id
            softdoc = softdoc_index.get(_document_key(document_id))
            if softdoc is None:
                softdoc = softdocs_root / Path(document_id).stem
            documents.append(
                ExternalDocument(
                    document_id=document_id,
                    source_kind="pdf",
                    source_path=_relative_or_absolute(source, path_root),
                    softdoc_dir=_relative_or_absolute(softdoc, path_root),
                    content_sha256=(
                        _sha256(source) if hash_sources and source.is_file() else None
                    ),
                )
            )

        cases: list[ExternalQuestion] = []
        for index, row in selected:
            question = str(row.get("question", "")).strip()
            if not question:
                raise ValueError(f"questions.json index {index} has a blank question")
            cases.append(
                ExternalQuestion(
                    case_id=f"Q{index}",
                    question_id=f"mmlongbench-doc:Q{index}",
                    document_id=str(row["doc_id"]),
                    question=question,
                    evidence_pages=_parse_evidence_pages(
                        row.get("evidence_pages"), index=index
                    ),
                    metadata={
                        "source_index": index,
                        "doc_type": row.get("doc_type"),
                        "evidence_sources": row.get("evidence_sources"),
                        "answer_format": row.get("answer_format"),
                    },
                )
            )

        path_root_hint = Path(
            os.path.relpath(path_root, start=manifest_path.parent)
        )
        return ExternalDatasetManifest(
            dataset_id="mmlongbench-doc",
            dataset_version=None,
            adapter=self.adapter_name,
            path_root=path_root_hint,
            documents=documents,
            questions=cases,
        )


def _audit_source(
    item: ExternalDocument,
    source: Path,
    issues: list[ExternalDataAuditIssue],
) -> bool:
    expected = source.is_file() if item.source_kind == "pdf" else source.is_dir()
    if not expected:
        noun = "file" if item.source_kind == "pdf" else "directory"
        issues.append(
            _issue(
                "error",
                "missing_source",
                f"Required source {noun} does not exist.",
                document=item,
                path=source,
            )
        )
        return False
    if item.content_sha256 is not None and _sha256(source) != item.content_sha256:
        issues.append(
            _issue(
                "error",
                "source_hash_mismatch",
                "Source SHA-256 does not match the manifest.",
                document=item,
                path=source,
            )
        )
        return False
    return True


def _audit_softdoc(
    item: ExternalDocument,
    softdoc_dir: Path,
    source: Path,
    issues: list[ExternalDataAuditIssue],
) -> tuple[bool, int | None]:
    issue_count_before = len(issues)
    document_json = softdoc_dir / "document.json"
    if not document_json.is_file():
        issues.append(
            _issue(
                "error",
                "missing_softdoc",
                "Serialized SoftDoc document.json does not exist.",
                document=item,
                path=document_json,
            )
        )
        return False, None
    try:
        document = load_document(softdoc_dir)
        reference_errors = DocumentStore(document).validate_references()
    except Exception as exc:
        issues.append(
            _issue(
                "error",
                "invalid_softdoc",
                f"SoftDoc cannot be loaded: {type(exc).__name__}: {exc}",
                document=item,
                path=document_json,
            )
        )
        return False, None
    for error in reference_errors:
        issues.append(
            _issue(
                "error",
                "invalid_softdoc_reference",
                error,
                document=item,
                path=document_json,
            )
        )
    missing_assets: set[Path] = set()
    for raw_path in [
        *(page.image_path for page in document.pages),
        *(element.image_path for element in document.elements),
        *(element.crop_image_path for element in document.elements),
    ]:
        if raw_path is None:
            continue
        # A serialized SoftDoc is portable across operating systems.  Pathlib
        # on Linux treats a Windows-authored backslash as a literal character,
        # so normalize stored relative asset paths before auditing them.
        portable_path = Path(str(raw_path).replace("\\", "/"))
        asset = (
            portable_path
            if portable_path.is_absolute()
            else softdoc_dir / portable_path
        )
        if not asset.is_file():
            missing_assets.add(asset)
    for asset in sorted(missing_assets, key=str):
        issues.append(
            _issue(
                "error",
                "missing_softdoc_asset",
                "SoftDoc references a visual asset that does not exist.",
                document=item,
                path=asset,
            )
        )
    if item.source_kind == "pdf" and source.is_file():
        source_name = _document_key(source.name)
        softdoc_name = _document_key(document.source_path.name)
        if source_name != softdoc_name:
            issues.append(
                _issue(
                    "error",
                    "source_identity_mismatch",
                    (
                        f"SoftDoc source {document.source_path.name!r} does not match "
                        f"manifest source {source.name!r}."
                    ),
                    document=item,
                    path=document_json,
                )
            )
    return len(issues) == issue_count_before, len(document.pages)


def _source_page_count(source: Path, source_kind: str) -> int:
    if source_kind == "pdf":
        document = pdfium.PdfDocument(str(source))
        try:
            return len(document)
        finally:
            document.close()
    return sum(
        1
        for path in source.iterdir()
        if path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES
    )


def _index_softdocs(root: Path) -> dict[str, Path]:
    candidates: dict[str, list[Path]] = defaultdict(list)
    if not root.is_dir():
        return {}
    for document_json in root.glob("*/document.json"):
        directory = document_json.parent
        keys = {_document_key(directory.name)}
        try:
            payload = json.loads(document_json.read_text(encoding="utf-8"))
            source_path = payload.get("source_path")
            if isinstance(source_path, str) and source_path:
                keys.add(_document_key(Path(source_path).name))
        except (OSError, json.JSONDecodeError):
            pass
        for key in keys:
            candidates[key].append(directory)
    result: dict[str, Path] = {}
    for key, paths in candidates.items():
        unique = sorted(set(paths), key=str)
        if len(unique) == 1:
            result[key] = unique[0]
    return result


def _document_key(value: str) -> str:
    stem = Path(value).stem.casefold()
    if stem.endswith("_origin"):
        stem = stem[: -len("_origin")]
    return stem


def _parse_evidence_pages(value: Any, *, index: int) -> list[int]:
    if value is None or value == "":
        return []
    parsed = value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"questions.json index {index} has invalid evidence_pages {value!r}"
            ) from exc
    if not isinstance(parsed, list) or not all(
        isinstance(page, int) and not isinstance(page, bool) for page in parsed
    ):
        raise ValueError(
            f"questions.json index {index} evidence_pages must be a list of integers"
        )
    return parsed


def _relative_or_absolute(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return path.resolve()


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        raise ValueError("SHA-256 is only supported for source files")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _issue(
    severity: Literal["error", "warning"],
    code: str,
    message: str,
    *,
    document: ExternalDocument,
    path: Path | None = None,
) -> ExternalDataAuditIssue:
    return ExternalDataAuditIssue(
        severity=severity,
        code=code,
        message=message,
        document_id=document.document_id,
        path=path,
    )


def _require_unique(values: Sequence[str], field: str) -> None:
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate {field} values: {duplicates}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
