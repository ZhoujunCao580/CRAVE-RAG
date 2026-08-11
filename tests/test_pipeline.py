from __future__ import annotations

from pathlib import Path

import pytest

from softdoc.adapters import MinerUAdapter
from softdoc.coverage import CoverageRecoveryResult
from softdoc.pipeline import (
    CoverageRecoveryPass,
    FloatingSectionPass,
    PageLabelPass,
    PassContext,
    RelationPass,
    RuleAuditPass,
    SoftDocPipeline,
    StructurePass,
    ValidationPass,
)


def _passes():
    return (
        CoverageRecoveryPass(),
        PageLabelPass(),
        StructurePass(),
        RelationPass(),
        FloatingSectionPass(),
        ValidationPass(),
        RuleAuditPass(),
    )


def test_pipeline_is_the_only_full_orchestration_entry(
    mineru_fixture_dir: Path,
    tmp_path: Path,
) -> None:
    adapter = MinerUAdapter()
    raw = adapter.parse(mineru_fixture_dir, tmp_path / "raw")
    result = SoftDocPipeline(adapter, passes=_passes()).run(
        mineru_fixture_dir,
        tmp_path / "final",
    )

    assert raw.sections == []
    assert raw.relations == []
    assert result.document.sections
    assert result.document.relations
    assert [report.name for report in result.pass_reports] == [
        "coverage_recovery",
        "page_label_resolver",
        "document_structure",
        "relation_builder",
        "floating_section_resolver",
        "reference_validation",
        "rule_audit",
    ]


def test_every_document_pass_is_idempotent(
    mineru_fixture_dir: Path,
    tmp_path: Path,
) -> None:
    document = MinerUAdapter().parse(
        mineru_fixture_dir,
        tmp_path / "raw",
    )
    context = PassContext(
        input_path=mineru_fixture_dir,
        output_dir=tmp_path / "final",
        available={"raw_document"},
    )

    for document_pass in _passes():
        assert document_pass.requires <= context.available
        first = document_pass.apply(document, context)
        after_first = document.model_dump(mode="json")
        second = document_pass.apply(document, context)
        after_second = document.model_dump(mode="json")

        assert after_second == after_first, document_pass.name
        assert second.changed is False, document_pass.name
        assert first.name == second.name == document_pass.name
        context.available.update(document_pass.provides)


def test_coverage_pass_preserves_first_result_on_second_apply(
    tmp_path: Path,
    parsed_document,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_pdf = tmp_path / "fixture_origin.pdf"
    source_pdf.write_bytes(b"mocked")
    recovered_id = parsed_document.elements[0].element_id
    calls = 0

    def fake_recovery(document, source):
        nonlocal calls
        calls += 1
        return CoverageRecoveryResult(
            scanned_line_count=7,
            recovered_element_ids=(recovered_id,),
        )

    monkeypatch.setattr(
        "softdoc.pipeline.recover_pdf_text_layer_coverage",
        fake_recovery,
    )
    context = PassContext(
        input_path=tmp_path,
        output_dir=tmp_path / "output",
        available={"raw_document"},
    )
    document = parsed_document.model_copy(deep=True)
    document.metadata.pop("coverage_recovery", None)
    document_pass = CoverageRecoveryPass()

    first = document_pass.apply(document, context)
    first_payload = document.model_dump(mode="json")
    second = document_pass.apply(document, context)

    assert calls == 1
    assert first.changed
    assert second.skipped
    assert second.changed is False
    assert document.model_dump(mode="json") == first_payload
    assert document.metadata["coverage_recovery"]["recovered_count"] == 1


def test_pipeline_rejects_unsatisfied_pass_dependencies(
    mineru_fixture_dir: Path,
    tmp_path: Path,
) -> None:
    pipeline = SoftDocPipeline(
        MinerUAdapter(),
        passes=(RelationPass(),),
    )

    with pytest.raises(RuntimeError, match="requires unavailable"):
        pipeline.run(mineru_fixture_dir, tmp_path / "output")
