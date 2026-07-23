from __future__ import annotations

from pathlib import Path

from softdoc.hierarchy import HeadingAction, HeadingHierarchyBuilder
from softdoc.models import (
    BoundingBox,
    Element,
    ElementType,
    Page,
    Provenance,
)
from softdoc.outline import build_document_outline, outline_markdown
from softdoc.visualization import _element_label


def _provenance(locator: str) -> Provenance:
    return Provenance(
        provenance_id=f"prov:{locator}",
        adapter="test",
        source_path=Path("fixture.json"),
        source_locator=locator,
    )


def _element(
    index: int,
    text: str,
    element_type: ElementType,
    *,
    level: int | None = None,
    y1: float | None = None,
    height: float = 0.02,
) -> Element:
    element_id = f"element:{index}"
    bbox = None
    if y1 is not None:
        bbox = BoundingBox(
            bbox_id=f"bbox:{index}",
            raw=(100.0, y1 * 1000, 900.0, (y1 + height) * 1000),
            normalized=(0.1, y1, 0.9, y1 + height),
        )
    return Element(
        element_id=element_id,
        document_id="doc:test",
        page_id="page:0",
        page_number=1,
        element_type=element_type,
        reading_order=index,
        bbox=bbox,
        heading_level=level,
        text=text,
        provenance=_provenance(element_id),
    )


def _page(elements: list[Element]) -> Page:
    element_ids = [element.element_id for element in elements]
    return Page(
        page_id="page:0",
        document_id="doc:test",
        page_index=0,
        page_number=1,
        width=1000,
        height=1000,
        element_ids=element_ids,
        reading_order=element_ids,
        provenance=_provenance("page:0"),
    )


def test_document_title_is_not_a_section_and_front_matter_stays_unassigned() -> None:
    elements = [
        _element(
            0,
            "A Research Paper",
            ElementType.HEADING,
            level=1,
            y1=0.05,
            height=0.04,
        ),
        _element(1, "Alice and Bob", ElementType.PARAGRAPH, y1=0.11),
        _element(2, "Abstract", ElementType.HEADING, level=2, y1=0.20),
        _element(3, "Summary text.", ElementType.PARAGRAPH, y1=0.24),
    ]

    result = HeadingHierarchyBuilder().build(
        "doc:test",
        [_page(elements)],
        elements,
    )

    assert result.document_title == "A Research Paper"
    assert [section.title for section in result.sections] == ["Abstract"]
    assert elements[0].section_id is None
    assert elements[1].section_id is None
    assert elements[2].heading_level == 1
    assert elements[3].section_id == result.sections[0].section_id
    assert result.decisions[0].action == HeadingAction.DOCUMENT_TITLE
    assert _element_label(elements[0]).startswith("0 TITLE ")
    assert _element_label(elements[2]).startswith("2 H1 ")


def test_numbering_and_unnumbered_subhead_build_h1_h2_h3_tree() -> None:
    elements = [
        _element(0, "3 Method", ElementType.HEADING, level=2, y1=0.05),
        _element(1, "3.1 Setup", ElementType.HEADING, level=2, y1=0.10),
        _element(
            2,
            "An important observation.",
            ElementType.HEADING,
            level=2,
            y1=0.15,
        ),
        _element(3, "4 Results", ElementType.HEADING, level=2, y1=0.20),
    ]

    result = HeadingHierarchyBuilder().build(
        "doc:test",
        [_page(elements)],
        elements,
    )

    assert [section.level for section in result.sections] == [1, 2, 3, 1]
    method, setup, observation, results = result.sections
    assert setup.parent_section_id == method.section_id
    assert observation.parent_section_id == setup.section_id
    assert observation.section_path == [
        "3 Method",
        "3.1 Setup",
        "An important observation.",
    ]
    assert results.parent_section_id is None


def test_appendix_prompt_headings_share_the_same_parent() -> None:
    elements = [
        _element(0, "E Prompts", ElementType.HEADING, level=2, y1=0.05),
        _element(1, "Direct Prompting", ElementType.HEADING, level=2, y1=0.10),
        _element(2, "Prompt body", ElementType.ALGORITHM, y1=0.15),
        _element(3, "ZS-CoT Prompting", ElementType.HEADING, level=2, y1=0.20),
        _element(4, "Prompt body", ElementType.ALGORITHM, y1=0.25),
    ]

    result = HeadingHierarchyBuilder().build(
        "doc:test",
        [_page(elements)],
        elements,
    )

    appendix, direct, zero_shot = result.sections
    assert [appendix.level, direct.level, zero_shot.level] == [1, 2, 2]
    assert direct.parent_section_id == appendix.section_id
    assert zero_shot.parent_section_id == appendix.section_id


def test_checked_question_heading_candidate_is_demoted() -> None:
    elements = [
        _element(
            0,
            "C \x03<sup>✓</sup> Did you run computational experiments?",
            ElementType.HEADING,
            level=2,
            y1=0.10,
        )
    ]

    result = HeadingHierarchyBuilder().build(
        "doc:test",
        [_page(elements)],
        elements,
    )

    assert result.sections == []
    assert elements[0].element_type == ElementType.PARAGRAPH
    assert elements[0].heading_level is None
    assert result.decisions[0].action == HeadingAction.DEMOTED_TO_PARAGRAPH


def test_outline_uses_normalized_section_tree(parsed_document) -> None:
    outline = build_document_outline(parsed_document)
    markdown = outline_markdown(outline)

    assert outline.title == "Soft Structure Fixture"
    assert outline.sections[0].title == "1 Introduction"
    assert outline.sections[0].level == 1
    assert "- H1 1 Introduction" in markdown
    assert parsed_document.metadata["heading_decisions"]
