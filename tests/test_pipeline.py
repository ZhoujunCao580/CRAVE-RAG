from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from softdoc.adapters import MinerUAdapter
from softdoc.coverage import CoverageRecoveryResult
from softdoc.models import ElementType
from softdoc.pipeline import (
    CoverageRecoveryPass,
    FloatingSectionPass,
    PageLabelPass,
    PassContext,
    RelationPass,
    RuleAuditPass,
    SoftDocPipeline,
    StructurePass,
    TableFragmentReconciliationPass,
    ValidationPass,
    VisualAssetRecoveryPass,
)


def _passes():
    return (
        CoverageRecoveryPass(),
        PageLabelPass(),
        StructurePass(),
        VisualAssetRecoveryPass(),
        TableFragmentReconciliationPass(),
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
        "visual_asset_recovery",
        "table_fragment_reconciliation",
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


def test_visual_asset_pass_recovers_normalized_visual_element(
    parsed_document,
    tmp_path: Path,
) -> None:
    document = parsed_document.model_copy(deep=True)
    page = document.pages[0]
    element = next(
        item for item in document.elements if item.page_id == page.page_id
    )
    element.element_type = ElementType.FIGURE
    element.image_path = None
    element.crop_image_path = None
    output_dir = tmp_path / "visual_recovery"
    page_asset = output_dir / "assets" / "pages" / "page.png"
    page_asset.parent.mkdir(parents=True)
    Image.new("RGB", (1000, 1000), "white").save(page_asset)
    page.image_path = page_asset.relative_to(output_dir)
    context = PassContext(
        input_path=tmp_path,
        output_dir=output_dir,
        available={"document_structure"},
    )
    document_pass = VisualAssetRecoveryPass()

    first = document_pass.apply(document, context)
    second = document_pass.apply(document, context)

    assert first.changed
    assert element.element_id in first.details["recovered_element_ids"]
    assert element.crop_image_path is not None
    assert (output_dir / element.crop_image_path).is_file()
    assert element.metadata["visual_asset_recovery"]["status"] == "recovered"
    assert second.changed is False


def test_visual_asset_pass_recovers_empty_textual_element(
    parsed_document,
    tmp_path: Path,
) -> None:
    document = parsed_document.model_copy(deep=True)
    page = document.pages[0]
    element = next(
        item for item in document.elements if item.page_id == page.page_id
    )
    element.element_type = ElementType.PARAGRAPH
    element.text = None
    element.html = None
    element.image_path = None
    element.crop_image_path = None
    output_dir = tmp_path / "empty_text_recovery"
    page_asset = output_dir / "assets" / "pages" / "page.png"
    page_asset.parent.mkdir(parents=True)
    Image.new("RGB", (1000, 1000), "white").save(page_asset)
    page.image_path = page_asset.relative_to(output_dir)
    context = PassContext(
        input_path=tmp_path,
        output_dir=output_dir,
        available={"document_structure"},
    )

    report = VisualAssetRecoveryPass().apply(document, context)

    assert report.changed
    assert element.element_id in report.details["recovered_element_ids"]
    assert element.crop_image_path is not None
    assert (output_dir / element.crop_image_path).is_file()
    assert element.metadata["visual_asset_recovery"]["status"] == "recovered"


def test_element_visual_asset_path_unifies_parser_and_fallback_assets(
    parsed_document,
) -> None:
    element = parsed_document.elements[0].model_copy(deep=True)
    element.image_path = Path("parser.png")
    element.crop_image_path = Path("fallback.png")
    assert element.visual_asset_path == Path("parser.png")

    element.image_path = None
    assert element.visual_asset_path == Path("fallback.png")
