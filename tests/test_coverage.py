from __future__ import annotations

from pathlib import Path

import pytest

from softdoc.coverage import PdfTextLayerCoverageRecovery, PdfTextLine
from softdoc.ids import bbox_id, document_id, element_id, page_id, provenance_id
from softdoc.models import BoundingBox, Document, Element, ElementType, Page, Provenance


def _provenance(source: Path, locator: str) -> Provenance:
    return Provenance(
        provenance_id=provenance_id("test", source.as_posix(), locator),
        adapter="test",
        source_path=source,
        source_locator=locator,
    )


def _document(source: Path) -> Document:
    doc_id = document_id("coverage.pdf")
    page_id_value = page_id(doc_id, 0)
    existing_id = element_id(doc_id, 0, 0, "paragraph")
    page = Page(
        page_id=page_id_value,
        document_id=doc_id,
        page_index=0,
        page_number=1,
        width=612,
        height=792,
        element_ids=[existing_id],
        reading_order=[existing_id],
        provenance=_provenance(source, "page[0]"),
    )
    existing = Element(
        element_id=existing_id,
        document_id=doc_id,
        page_id=page.page_id,
        page_number=1,
        element_type=ElementType.PARAGRAPH,
        reading_order=0,
        bbox=BoundingBox.from_raw(
            bbox_id=bbox_id("existing"), raw=(40, 40, 500, 70), page_width=612, page_height=792
        ),
        text="3. Name of this Capital Asset:",
        provenance=_provenance(source, "block[0]"),
    )
    return Document(
        document_id=doc_id,
        source_path=source,
        pages=[page],
        elements=[existing],
        provenance=_provenance(source, "document"),
    )


def test_recovery_adds_only_uncovered_text_and_updates_page_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = (tmp_path / "coverage.pdf").resolve()
    source.write_bytes(b"not read because extraction is mocked")
    document = _document(source)
    recovery = PdfTextLayerCoverageRecovery(source)
    monkeypatch.setattr(
        recovery,
        "_extract_lines",
        lambda: {
            0: [
                PdfTextLine(0, 0, "3. Name of this Capital Asset:", (40, 722, 300, 752)),
                PdfTextLine(0, 1, "4. Name of this Capital Asset:", (40, 672, 300, 702)),
            ]
        },
    )

    result = recovery.recover(document)

    assert result.scanned_line_count == 2
    assert result.recovered_count == 1
    recovered = next(element for element in document.elements if element.element_id in result.recovered_element_ids)
    assert recovered.text == "4. Name of this Capital Asset:"
    assert recovered.element_type == ElementType.PARAGRAPH
    assert recovered.parse_status.value == "degraded"
    assert recovered.provenance.source_path == source
    assert recovered.metadata["coverage_recovery"]["warning_code"] == "pdf_text_layer_uncovered"
    page = document.pages[0]
    assert page.element_ids == page.reading_order
    assert page.reading_order == [element.element_id for element in sorted(document.elements, key=lambda item: item.reading_order)]
    assert [element.reading_order for element in document.elements] == [0, 1]


def test_recovery_requires_an_absolute_pdf_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        PdfTextLayerCoverageRecovery(Path("relative.pdf"))


def test_recovery_keeps_only_uncovered_part_of_a_form_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (tmp_path / "coverage.pdf").resolve()
    source.write_bytes(b"not read because extraction is mocked")
    document = _document(source)
    existing = document.elements[0]
    existing.text = "Capital Asset Management (CAM)"
    existing.bbox = BoundingBox.from_raw(
        bbox_id=bbox_id("existing-value"),
        raw=(330, 86, 540, 118),
        page_width=612,
        page_height=792,
    )
    recovery = PdfTextLayerCoverageRecovery(source)
    line_text = "4. Name of this Capital Asset: Capital Asset Management"
    monkeypatch.setattr(
        recovery,
        "_extract_lines",
        lambda: {0: [PdfTextLine(0, 0, line_text, (40, 672, 540, 702))]},
    )

    result = recovery.recover(document)

    assert result.recovered_count == 1
    recovered = next(
        element for element in document.elements if element.element_id in result.recovered_element_ids
    )
    assert recovered.text == "4. Name of this Capital Asset:"
    assert document.pages[0].reading_order[0] == recovered.element_id


def test_recovery_does_not_duplicate_table_text_or_page_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (tmp_path / "coverage.pdf").resolve()
    source.write_bytes(b"not read because extraction is mocked")
    document = _document(source)
    original = document.elements[0]
    table = original.model_copy(
        update={
            "element_type": ElementType.TABLE,
            "text": None,
            "html": "<table><tr><th>Measure</th><th>Value</th></tr><tr><td>A</td><td>1</td></tr></table>",
            "bbox": BoundingBox.from_raw(
                bbox_id=bbox_id("table"),
                raw=(40, 100, 560, 500),
                page_width=612,
                page_height=792,
            ),
        }
    )
    document.elements = [table]
    recovery = PdfTextLayerCoverageRecovery(source)
    monkeypatch.setattr(
        recovery,
        "_extract_lines",
        lambda: {
            0: [
                PdfTextLine(0, 0, "Measure Value", (60, 650, 300, 675)),
                PdfTextLine(0, 1, "A 1", (60, 610, 160, 635)),
                PdfTextLine(0, 2, "Page 1 of 15", (260, 20, 350, 35)),
            ]
        },
    )

    result = recovery.recover(document)

    assert result.recovered_count == 0


def test_recovery_clips_negative_pdf_coordinates_without_aborting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = (tmp_path / "coverage.pdf").resolve()
    source.write_bytes(b"not read because extraction is mocked")
    document = _document(source)
    recovery = PdfTextLayerCoverageRecovery(source)
    monkeypatch.setattr(
        recovery,
        "_extract_lines",
        lambda: {
            0: [
                PdfTextLine(
                    0,
                    0,
                    "Visible text with a glyph overhang",
                    (-24, 620, 320, 650),
                    source_page_width=612,
                    source_page_height=792,
                )
            ]
        },
    )

    result = recovery.recover(document)

    assert result.recovered_count == 1
    recovered = next(
        element
        for element in document.elements
        if element.element_id in result.recovered_element_ids
    )
    assert recovered.bbox is not None
    assert recovered.bbox.normalized[0] == 0.0
    assert recovered.provenance.raw_payload["pdf_bbox"][0] == -24
    assert recovered.provenance.raw_payload["clipped_pdf_bbox"][0] == 0.0
    assert recovered.metadata["coverage_recovery"]["bbox_clipped_to_page"] is True


def test_recovery_skips_fully_outside_line_and_keeps_valid_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = (tmp_path / "coverage.pdf").resolve()
    source.write_bytes(b"not read because extraction is mocked")
    document = _document(source)
    recovery = PdfTextLayerCoverageRecovery(source)
    monkeypatch.setattr(
        recovery,
        "_extract_lines",
        lambda: {
            0: [
                PdfTextLine(
                    0,
                    0,
                    "Decorative text outside the media box",
                    (-200, 620, -20, 650),
                    source_page_width=612,
                    source_page_height=792,
                ),
                PdfTextLine(
                    0,
                    1,
                    "A valid uncovered line remains recoverable",
                    (40, 620, 320, 650),
                    source_page_width=612,
                    source_page_height=792,
                ),
            ]
        },
    )

    result = recovery.recover(document)

    assert result.recovered_count == 1
    recovered = next(
        element
        for element in document.elements
        if element.element_id in result.recovered_element_ids
    )
    assert recovered.text == "A valid uncovered line remains recoverable"
