"""Strict semantic comparison for serialized SoftDoc documents."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import Field

from softdoc.models import SoftDocModel


_MAX_RECORDED_DIFFERENCES = 200


class SemanticDocumentDiff(SoftDocModel):
    document: str
    equal: bool
    before_sha256: str
    after_sha256: str
    difference_count: int = Field(ge=0)
    differences: list[str] = Field(default_factory=list)


class SemanticDiffReport(SoftDocModel):
    schema_version: str = "1.0"
    baseline_root: Path
    candidate_root: Path
    document_count: int = Field(ge=0)
    equal_document_count: int = Field(ge=0)
    changed_document_count: int = Field(ge=0)
    missing_from_baseline: list[str] = Field(default_factory=list)
    missing_from_candidate: list[str] = Field(default_factory=list)
    documents: list[SemanticDocumentDiff] = Field(default_factory=list)

    @property
    def semantically_equal(self) -> bool:
        return (
            self.changed_document_count == 0
            and not self.missing_from_baseline
            and not self.missing_from_candidate
        )


def compare_softdoc_roots(
    baseline_root: Path,
    candidate_root: Path,
) -> SemanticDiffReport:
    baseline_root = Path(baseline_root)
    candidate_root = Path(candidate_root)
    baseline = _document_paths(baseline_root)
    candidate = _document_paths(candidate_root)
    names = sorted(set(baseline) & set(candidate))
    rows: list[SemanticDocumentDiff] = []
    for name in names:
        before = _load_json(baseline[name])
        after = _load_json(candidate[name])
        differences: list[str] = []
        difference_count = _deep_difference_count(
            before,
            after,
            path="",
            recorded=differences,
        )
        rows.append(
            SemanticDocumentDiff(
                document=name,
                equal=difference_count == 0,
                before_sha256=_canonical_sha256(before),
                after_sha256=_canonical_sha256(after),
                difference_count=difference_count,
                differences=differences,
            )
        )
    equal_count = sum(row.equal for row in rows)
    return SemanticDiffReport(
        baseline_root=baseline_root,
        candidate_root=candidate_root,
        document_count=len(rows),
        equal_document_count=equal_count,
        changed_document_count=len(rows) - equal_count,
        missing_from_baseline=sorted(set(candidate) - set(baseline)),
        missing_from_candidate=sorted(set(baseline) - set(candidate)),
        documents=rows,
    )


def write_semantic_diff_reports(
    report: SemanticDiffReport,
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "semantic_diff_report.json").write_text(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "semantic_diff_report.md").write_text(
        semantic_diff_markdown(report),
        encoding="utf-8",
    )


def semantic_diff_markdown(report: SemanticDiffReport) -> str:
    verdict = "PASS" if report.semantically_equal else "FAIL"
    lines = [
        "# SoftDoc JSON semantic diff",
        "",
        f"- Verdict: **{verdict}**",
        f"- Compared documents: {report.document_count}",
        f"- Semantically equal: {report.equal_document_count}",
        f"- Changed: {report.changed_document_count}",
        f"- Missing from baseline: {len(report.missing_from_baseline)}",
        f"- Missing from candidate: {len(report.missing_from_candidate)}",
        "",
        "| Document | Equal | Differences | Before SHA-256 | After SHA-256 |",
        "|---|---:|---:|---|---|",
    ]
    for row in report.documents:
        lines.append(
            f"| `{row.document}` | {str(row.equal).lower()} | "
            f"{row.difference_count} | `{row.before_sha256}` | "
            f"`{row.after_sha256}` |"
        )
    if report.missing_from_baseline:
        lines.extend(
            (
                "",
                "## Missing from baseline",
                "",
                *[f"- `{name}`" for name in report.missing_from_baseline],
            )
        )
    if report.missing_from_candidate:
        lines.extend(
            (
                "",
                "## Missing from candidate",
                "",
                *[f"- `{name}`" for name in report.missing_from_candidate],
            )
        )
    lines.append("")
    return "\n".join(lines)


def _document_paths(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise NotADirectoryError(root)
    return {
        directory.name: document_path
        for directory in sorted(root.iterdir())
        if directory.is_dir()
        and (document_path := directory / "document.json").is_file()
    }


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _deep_difference_count(
    before: Any,
    after: Any,
    *,
    path: str,
    recorded: list[str],
) -> int:
    if type(before) is not type(after):
        _record(recorded, path or "/", "type")
        return 1
    if isinstance(before, dict):
        count = 0
        keys = sorted(set(before) | set(after))
        for key in keys:
            child_path = f"{path}/{_escape_pointer(str(key))}"
            if key not in before:
                _record(recorded, child_path, "added")
                count += 1
            elif key not in after:
                _record(recorded, child_path, "removed")
                count += 1
            else:
                count += _deep_difference_count(
                    before[key],
                    after[key],
                    path=child_path,
                    recorded=recorded,
                )
        return count
    if isinstance(before, list):
        count = 0
        maximum = max(len(before), len(after))
        for index in range(maximum):
            child_path = f"{path}/{index}"
            if index >= len(before):
                _record(recorded, child_path, "added")
                count += 1
            elif index >= len(after):
                _record(recorded, child_path, "removed")
                count += 1
            else:
                count += _deep_difference_count(
                    before[index],
                    after[index],
                    path=child_path,
                    recorded=recorded,
                )
        return count
    if before != after:
        _record(recorded, path or "/", "changed")
        return 1
    return 0


def _record(recorded: list[str], path: str, kind: str) -> None:
    if len(recorded) < _MAX_RECORDED_DIFFERENCES:
        recorded.append(f"{kind}: {path}")


def _escape_pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")
