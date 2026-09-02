from __future__ import annotations

import json
from pathlib import Path

import pypdf
import pytest

from scripts.audit_gold_evidence_reachability import (
    _resolved_gold_pages,
    main,
)
from softdoc.serialization import write_document


def test_stage3a_cli_writes_contract_and_refuses_overwrite(
    parsed_document,
    tmp_path: Path,
) -> None:
    documents_root = tmp_path / "documents"
    softdocs_root = tmp_path / "softdocs"
    documents_root.mkdir()
    source_pdf = documents_root / "sample.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with source_pdf.open("wb") as handle:
        writer.write(handle)

    document = parsed_document.model_copy(deep=True)
    document.source_path = Path("sample_origin.pdf")
    write_document(document, softdocs_root / "sample", render_overlays=False)
    questions = [
        {
            "doc_id": "sample.pdf",
            "doc_type": "fixture",
            "question": "What is shown?",
            "answer": "fixture",
            "evidence_pages": "[1]",
            "evidence_sources": "['Text']",
            "answer_format": "Str",
        },
        {
            "doc_id": "sample.pdf",
            "doc_type": "fixture",
            "question": "This question has no Gold page.",
            "answer": "fixture",
            "evidence_pages": "[]",
            "evidence_sources": "[]",
            "answer_format": "Str",
        },
    ]
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(json.dumps(questions), encoding="utf-8")
    progress_path = tmp_path / "progress.json"
    progress_path.write_text('{"state":"completed"}', encoding="utf-8")
    output_root = tmp_path / "runs"
    argv = [
        "--questions",
        str(questions_path),
        "--documents-root",
        str(documents_root),
        "--softdocs-root",
        str(softdocs_root),
        "--progress",
        str(progress_path),
        "--output-root",
        str(output_root),
        "--run-id",
        "smoke",
    ]

    assert main(argv) == 0
    run_dir = output_root / "smoke"
    rows = [
        json.loads(line)
        for line in (run_dir / "gold_evidence_reachability.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    summary = json.loads(
        (run_dir / "gold_evidence_reachability_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert rows[0]["question_id"] == "mmlongbench-doc:Q0"
    assert rows[0]["gold_page_id"] == 1
    assert rows[0]["audit_applicable"] is True
    assert rows[0]["reachable"] is True
    assert rows[0]["nonempty_search_unit"] is True
    assert rows[1]["question_id"] == "mmlongbench-doc:Q1"
    assert rows[1]["audit_applicable"] is False
    assert rows[1]["failure_reasons"] == ["no_gold_evidence_pages"]
    assert summary["counts"]["fully_reachable_questions"] == 1
    assert summary["counts"]["questions_with_gold_pages"] == 1
    assert summary["counts"]["questions_without_gold_pages"] == 1
    assert summary["counts"]["question_all_pages_reachability_rate"] == 1.0
    assert (run_dir / "gold_evidence_unreachable.jsonl").read_text(
        encoding="utf-8"
    ) == ""

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        main(argv)


def test_gold_page_resolution_preserves_raw_and_rejects_stale_overlay() -> None:
    resolution = {
        "raw_evidence_pages": [289, 290],
        "resolved_evidence_pages": [8, 9],
    }

    assert _resolved_gold_pages("Q787", [289, 290], resolution) == [8, 9]
    assert _resolved_gold_pages("Q1", [3], None) == [3]
    with pytest.raises(ValueError, match="stale"):
        _resolved_gold_pages("Q787", [289], resolution)
