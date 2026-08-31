from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from softdoc.external_data import (
    ExternalDatasetManifest,
    ExternalDocument,
    ExternalQuestion,
    MMLongBenchDocAdapter,
    audit_external_dataset,
    export_batch_cases,
    load_external_dataset_manifest,
    write_external_dataset_manifest,
)
from softdoc.serialization import write_document


def test_external_manifest_audit_and_gold_free_batch_export(
    parsed_document,
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source-pages"
    source_dir.mkdir()
    for page in parsed_document.pages:
        Image.new("RGB", (16, 16), "white").save(
            source_dir / f"page_{page.page_number:04d}.png"
        )
    softdoc_dir = tmp_path / "adapter_output"
    write_document(parsed_document, softdoc_dir, render_overlays=False)
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest = ExternalDatasetManifest(
        dataset_id="fixture",
        adapter="fixture-v0",
        documents=[
            ExternalDocument(
                document_id="doc-1",
                source_kind="image_directory",
                source_path=source_dir.relative_to(tmp_path),
                softdoc_dir=softdoc_dir.relative_to(tmp_path),
            )
        ],
        questions=[
            ExternalQuestion(
                case_id="Q1",
                question_id="fixture:Q1",
                document_id="doc-1",
                question="What is reported?",
                evidence_pages=[1],
                metadata={"gold_only_note": "not exported"},
            )
        ],
    )
    write_external_dataset_manifest(manifest, manifest_path)

    restored = load_external_dataset_manifest(manifest_path)
    report = audit_external_dataset(restored, manifest_path=manifest_path)
    assert report.status == "passed"
    assert report.audited_document_count == 1
    assert report.audited_question_count == 1

    cases_path = tmp_path / "cases.jsonl"
    assert export_batch_cases(
        restored,
        manifest_path=manifest_path,
        output=cases_path,
    ) == 1
    row = json.loads(cases_path.read_text(encoding="utf-8"))
    assert row == {
        "case_id": "Q1",
        "question_id": "fixture:Q1",
        "document_dir": str(softdoc_dir.resolve()),
        "question": "What is reported?",
    }


def test_external_audit_fails_loudly_for_missing_corpus(tmp_path: Path) -> None:
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest = ExternalDatasetManifest(
        dataset_id="missing-fixture",
        adapter="fixture-v0",
        documents=[
            ExternalDocument(
                document_id="doc-1",
                source_kind="pdf",
                source_path=Path("missing.pdf"),
                softdoc_dir=Path("missing-softdoc"),
            )
        ],
        questions=[
            ExternalQuestion(
                case_id="Q1",
                question_id="fixture:Q1",
                document_id="doc-1",
                question="Question?",
            )
        ],
    )

    report = audit_external_dataset(manifest, manifest_path=manifest_path)

    assert report.status == "failed"
    assert {issue.code for issue in report.issues} == {
        "missing_source",
        "missing_softdoc",
    }
    assert report.audited_document_count == 0


def test_mmlongbench_adapter_maps_origin_softdoc_and_keeps_gold_out(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    documents = raw / "documents"
    documents.mkdir(parents=True)
    (documents / "manual.pdf").write_bytes(b"not opened during mapping")
    questions = raw / "questions.json"
    questions.write_text(
        "\ufeff"
        + json.dumps(
            [
                {
                    "doc_id": "manual.pdf",
                    "doc_type": "Guidebook",
                    "question": "What is the value?",
                    "answer": "secret Gold",
                    "evidence_pages": "[2]",
                    "evidence_sources": "['Table']",
                    "answer_format": "Float",
                }
            ]
        ),
        encoding="utf-8",
    )
    softdoc = tmp_path / "softdocs" / "manual"
    softdoc.mkdir(parents=True)
    (softdoc / "document.json").write_text(
        json.dumps({"source_path": "manual_origin.pdf"}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"

    manifest = MMLongBenchDocAdapter().build_manifest(
        questions_path=questions,
        documents_root=documents,
        softdocs_root=tmp_path / "softdocs",
        path_root=tmp_path,
        manifest_path=manifest_path,
    )

    assert manifest.documents[0].softdoc_dir == Path("softdocs/manual")
    assert manifest.questions[0].case_id == "Q0"
    assert manifest.questions[0].evidence_pages == [2]
    serialized = manifest.model_dump_json()
    assert "secret Gold" not in serialized
    assert '"answer"' not in serialized
