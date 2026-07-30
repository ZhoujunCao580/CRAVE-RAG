from __future__ import annotations

import json
from pathlib import Path

from softdoc.semantic_diff import (
    compare_softdoc_roots,
    write_semantic_diff_reports,
)


def _write_document(root: Path, name: str, payload: dict) -> None:
    destination = root / name
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "document.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_semantic_diff_ignores_json_formatting_but_not_list_order(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_document(
        baseline,
        "doc",
        {"document_id": "doc:1", "reading_order": ["a", "b"]},
    )
    _write_document(
        candidate,
        "doc",
        {"reading_order": ["a", "b"], "document_id": "doc:1"},
    )

    equal = compare_softdoc_roots(baseline, candidate)
    assert equal.semantically_equal
    assert equal.documents[0].before_sha256 == equal.documents[0].after_sha256

    _write_document(
        candidate,
        "doc",
        {"document_id": "doc:1", "reading_order": ["b", "a"]},
    )
    changed = compare_softdoc_roots(baseline, candidate)
    assert not changed.semantically_equal
    assert changed.documents[0].difference_count == 2
    assert changed.documents[0].differences == [
        "changed: /reading_order/0",
        "changed: /reading_order/1",
    ]


def test_semantic_diff_reports_are_written(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    output = tmp_path / "report"
    _write_document(baseline, "doc", {"value": 1})
    _write_document(candidate, "doc", {"value": 1})

    report = compare_softdoc_roots(baseline, candidate)
    write_semantic_diff_reports(report, output)

    assert (output / "semantic_diff_report.json").is_file()
    assert "**PASS**" in (
        output / "semantic_diff_report.md"
    ).read_text(encoding="utf-8")
