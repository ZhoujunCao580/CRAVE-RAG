from __future__ import annotations

from pathlib import Path

from softdoc.models import BoundingBox, Element, ElementType, Page, Provenance
from softdoc.profiles import DocumentProfile, DocumentProfileDetector


def _provenance(locator: str) -> Provenance:
    return Provenance(
        provenance_id=f"prov:{locator}",
        adapter="test",
        source_path=Path("sample.pdf"),
        source_locator=locator,
    )


def _pages(count: int, *, landscape: bool = True) -> list[Page]:
    width, height = (1600.0, 900.0) if landscape else (900.0, 1600.0)
    return [
        Page(
            page_id=f"page:{page_number}",
            document_id="doc:test",
            page_index=page_number - 1,
            page_number=page_number,
            width=width,
            height=height,
            provenance=_provenance(f"page:{page_number}"),
        )
        for page_number in range(1, count + 1)
    ]


def _element(
    page_number: int,
    order: int,
    *,
    element_type: ElementType = ElementType.PARAGRAPH,
    text: str | None = None,
    html: str | None = None,
    bbox: tuple[float, float, float, float] = (0.1, 0.2, 0.9, 0.4),
) -> Element:
    return Element(
        element_id=f"element:{page_number}:{order}",
        document_id="doc:test",
        page_id=f"page:{page_number}",
        page_number=page_number,
        element_type=element_type,
        reading_order=order,
        text=text,
        html=html,
        bbox=BoundingBox.from_raw(
            bbox_id=f"bbox:{page_number}:{order}",
            raw=bbox,
            page_width=1.0,
            page_height=1.0,
        ),
        provenance=_provenance(f"element:{page_number}:{order}"),
    )


def _slide_elements(page_count: int = 5) -> list[Element]:
    elements: list[Element] = []
    for page_number in range(1, page_count + 1):
        elements.extend(
            [
                _element(
                    page_number,
                    0,
                    element_type=ElementType.HEADING,
                    text=f"Topic {page_number}",
                    bbox=(0.08, 0.05, 0.72, 0.16),
                ),
                _element(
                    page_number,
                    1,
                    text="A concise slide bullet.",
                    bbox=(0.12, 0.30, 0.75, 0.45),
                ),
                _element(
                    page_number,
                    2,
                    text=f"Workshop 2026 - {page_number}",
                    bbox=(0.08, 0.90, 0.40, 0.95),
                ),
            ]
        )
    return elements


def test_sparse_landscape_repeated_template_is_slides() -> None:
    decision = DocumentProfileDetector().detect(_pages(5), _slide_elements())

    assert decision.profile == DocumentProfile.SLIDES
    assert decision.evidence["rule"] == "sparse_landscape_repeated_page_pattern"
    assert decision.evidence["landscape_page_ratio"] == 1.0
    assert decision.evidence["sparse_page_ratio"] == 1.0


def test_dense_landscape_document_falls_back_to_report() -> None:
    elements: list[Element] = []
    for page_number in range(1, 6):
        elements.append(
            _element(
                page_number,
                0,
                element_type=ElementType.HEADING,
                text=f"Operational review {page_number}",
                bbox=(0.08, 0.05, 0.72, 0.16),
            )
        )
        for order in range(1, 27):
            elements.append(
                _element(
                    page_number,
                    order,
                    text=(
                        "This report paragraph contains sustained analysis, "
                        "supporting detail, and explanatory context."
                    ),
                    bbox=(0.08, 0.18, 0.92, 0.82),
                )
            )

    decision = DocumentProfileDetector().detect(_pages(5), elements)

    assert decision.profile == DocumentProfile.REPORT
    assert decision.evidence["rule"] == "conservative_report_fallback"
    assert decision.evidence["median_element_count_per_page"] == 27.0


def test_dense_report_with_brochure_vocabulary_still_falls_back_to_report() -> None:
    elements: list[Element] = []
    for page_number in range(1, 6):
        elements.append(
            _element(
                page_number,
                0,
                element_type=ElementType.HEADING,
                text=(
                    "Annual Overview"
                    if page_number == 1
                    else "Admissions Overview"
                ),
                bbox=(0.08, 0.05, 0.72, 0.16),
            )
        )
        for order in range(1, 24):
            paragraph = (
                "The annual report contains detailed operational analysis, "
                "historical comparisons, governance notes, and risk context."
                if page_number == 1
                else (
                    "The programme report discusses admissions and "
                    "prospectus distribution with detailed annual analysis."
                )
            )
            elements.append(
                _element(
                    page_number,
                    order,
                    text=paragraph,
                    bbox=(0.08, 0.18, 0.92, 0.82),
                )
            )

    decision = DocumentProfileDetector().detect(_pages(5), elements)

    assert decision.profile == DocumentProfile.REPORT
    assert decision.evidence["sparse_page_ratio"] == 0.0


def test_sparse_landscape_without_repeated_pattern_is_report() -> None:
    elements = [
        _element(
            page_number,
            0,
            text=f"Unique body content for page {page_number}",
            bbox=(0.20, 0.35, 0.75, 0.48),
        )
        for page_number in range(1, 6)
    ]

    decision = DocumentProfileDetector().detect(_pages(5), elements)

    assert decision.profile == DocumentProfile.REPORT
    assert decision.evidence["repeated_page_pattern"] is None


def test_academic_profile_precedes_slide_signals() -> None:
    elements = _slide_elements()
    elements.extend(
        [
            _element(
                1,
                10,
                element_type=ElementType.HEADING,
                text="Abstract",
            ),
            _element(
                5,
                10,
                element_type=ElementType.HEADING,
                text="References",
            ),
        ]
    )

    decision = DocumentProfileDetector().detect(_pages(5), elements)

    assert decision.profile == DocumentProfile.ACADEMIC
    assert decision.evidence["rule"] == "academic_outline_signals"


def test_manual_profile_precedes_slide_signals() -> None:
    elements = _slide_elements()
    elements.extend(
        [
            _element(
                2,
                10,
                element_type=ElementType.HEADING,
                text="Chapter 1 Setup",
            ),
            _element(
                4,
                10,
                element_type=ElementType.HEADING,
                text="Chapter 2 Operation",
            ),
        ]
    )

    decision = DocumentProfileDetector().detect(_pages(5), elements)

    assert decision.profile == DocumentProfile.MANUAL
    assert decision.evidence["rule"] == "chapter_contents_index_structure"


def test_form_profile_precedes_slide_signals() -> None:
    elements = _slide_elements()
    for page_number in range(1, 4):
        elements.extend(
            [
                _element(
                    page_number,
                    10,
                    text=f"Section {page_number}: Required information",
                ),
                _element(
                    page_number,
                    11,
                    element_type=ElementType.TABLE,
                    html="<table><tr><td>Field</td></tr></table>",
                    bbox=(0.05, 0.20, 0.95, 0.85),
                ),
            ]
        )

    decision = DocumentProfileDetector().detect(_pages(5), elements)

    assert decision.profile == DocumentProfile.FORM
    assert decision.evidence["rule"] == "section_labels_with_dense_tables"


def test_brochure_profile_precedes_slide_signals() -> None:
    decision = DocumentProfileDetector().detect(
        _pages(5),
        _slide_elements(),
        source_path=Path("graduate-prospectus-brochure.pdf"),
    )

    assert decision.profile == DocumentProfile.BROCHURE
    assert decision.evidence["rule"] == "strong_brochure_identity"
