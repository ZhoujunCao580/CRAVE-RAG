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
from softdoc.repetition import RepeatedHeaderFooterDetector
from softdoc.sections import SectionBuilder
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
    page_index: int = 0,
    x1: float = 0.1,
    style: dict | None = None,
) -> Element:
    element_id = f"element:{page_index}:{index}"
    bbox = None
    if y1 is not None:
        bbox = BoundingBox(
            bbox_id=f"bbox:{index}",
            raw=(x1 * 1000, y1 * 1000, 900.0, (y1 + height) * 1000),
            normalized=(x1, y1, 0.9, y1 + height),
        )
    return Element(
        element_id=element_id,
        document_id="doc:test",
        page_id=f"page:{page_index}",
        page_number=page_index + 1,
        element_type=element_type,
        reading_order=index,
        bbox=bbox,
        heading_level=level,
        text=text,
        provenance=_provenance(element_id),
        metadata={"style": style} if style else {},
    )


def _page(elements: list[Element], page_index: int = 0) -> Page:
    element_ids = [element.element_id for element in elements]
    return Page(
        page_id=f"page:{page_index}",
        document_id="doc:test",
        page_index=page_index,
        page_number=page_index + 1,
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
    sections = SectionBuilder().build("doc:test", [_page(elements)], elements)

    assert result.document_title == "A Research Paper"
    assert [section.title for section in sections] == ["Abstract"]
    assert elements[0].section_id is None
    assert elements[1].section_id is None
    assert elements[2].heading_level == 1
    assert elements[3].section_id == sections[0].section_id
    assert result.decisions[0].action == HeadingAction.DOCUMENT_TITLE
    assert _element_label(elements[0]).startswith("0 TITLE ")
    assert _element_label(elements[2]).startswith("2 H1 raw=2 ")


def test_unnumbered_same_style_stays_sibling_instead_of_becoming_h3() -> None:
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
    sections = SectionBuilder().build("doc:test", [_page(elements)], elements)

    assert [section.level for section in sections] == [1, 2, 2, 1]
    method, setup, observation, results = sections
    assert setup.parent_section_id == method.section_id
    assert observation.parent_section_id == method.section_id
    assert observation.section_path == [
        "3 Method",
        "An important observation.",
    ]
    assert results.parent_section_id is None


def test_short_all_caps_table_caption_creates_float_local_section() -> None:
    heading = _element(
        0,
        "ABSOLUTE MAXIMUM RATINGS",
        ElementType.HEADING,
        level=1,
        y1=0.05,
    )
    table = _element(
        1,
        "voltage limits",
        ElementType.TABLE,
        y1=0.20,
        height=0.20,
    )
    caption = _element(
        2,
        "OPERATING CONDITIONS",
        ElementType.CAPTION,
        y1=0.17,
    )
    caption.metadata["target_element_id"] = table.element_id
    elements = [heading, table, caption]

    sections = SectionBuilder().build(
        "doc:test",
        [_page(elements)],
        elements,
    )

    assert [section.title for section in sections] == [
        "ABSOLUTE MAXIMUM RATINGS",
        "OPERATING CONDITIONS",
    ]
    assert sections[1].metadata["section_scope"] == "floating_table_block"
    assert table.section_id == sections[1].section_id
    assert caption.section_id == sections[1].section_id


def test_numbered_table_caption_does_not_create_section() -> None:
    heading = _element(
        0,
        "3 Results",
        ElementType.HEADING,
        level=1,
        y1=0.05,
    )
    table = _element(
        1,
        "scores",
        ElementType.TABLE,
        y1=0.20,
        height=0.20,
    )
    caption = _element(
        2,
        "Table 1: Main results",
        ElementType.CAPTION,
        y1=0.17,
    )
    caption.metadata["target_element_id"] = table.element_id
    elements = [heading, table, caption]

    sections = SectionBuilder().build(
        "doc:test",
        [_page(elements)],
        elements,
    )

    assert [section.title for section in sections] == ["3 Results"]
    assert table.section_id == sections[0].section_id
    assert caption.section_id == sections[0].section_id


def test_table_caption_section_title_drops_test_conditions_suffix() -> None:
    heading = _element(
        0,
        "ERASURE CHARACTERISTICS",
        ElementType.HEADING,
        level=1,
        y1=0.05,
    )
    table = _element(
        1,
        "programming limits",
        ElementType.TABLE,
        y1=0.20,
        height=0.20,
    )
    caption = _element(
        2,
        (
            "EPROM PROGRAMMING AND VERIFICATION CHARACTERISTICS "
            "TA = 21\u00b0C to 27\u00b0C"
        ),
        ElementType.CAPTION,
        y1=0.17,
    )
    caption.metadata["target_element_id"] = table.element_id
    elements = [heading, table, caption]

    sections = SectionBuilder().build(
        "doc:test",
        [_page(elements)],
        elements,
    )

    assert sections[-1].title == (
        "EPROM PROGRAMMING AND VERIFICATION CHARACTERISTICS"
    )


def test_terminal_page_header_can_anchor_numbered_checklist_section() -> None:
    prior = [
        _element(
            0,
            "E Prompts",
            ElementType.HEADING,
            level=1,
            y1=0.08,
            page_index=0,
        ),
        _element(
            1,
            "Prompt body",
            ElementType.PARAGRAPH,
            y1=0.20,
            page_index=0,
        ),
    ]
    checklist_page = [
        _element(
            0,
            "A1. Did you describe the limitations?",
            ElementType.PARAGRAPH,
            y1=0.12,
            page_index=1,
        ),
        _element(
            1,
            "A2. Did you discuss potential risks?",
            ElementType.PARAGRAPH,
            y1=0.22,
            page_index=1,
        ),
        _element(
            2,
            "ACL 2023 Responsible NLP Checklist",
            ElementType.PARAGRAPH,
            y1=0.05,
            page_index=1,
        ),
        _element(
            3,
            "Checklist provenance note.",
            ElementType.FOOTNOTE,
            y1=0.90,
            page_index=1,
        ),
    ]
    checklist_page[2].metadata["mineru_type"] = "page_header"
    continuation = [
        _element(
            0,
            "B1. Did you cite the artifacts?",
            ElementType.PARAGRAPH,
            y1=0.10,
            page_index=2,
        )
    ]
    pages = [
        _page(prior, 0),
        _page(checklist_page, 1),
        _page(continuation, 2),
    ]
    elements = prior + checklist_page + continuation

    sections = SectionBuilder().build("doc:test", pages, elements)

    assert [section.title for section in sections] == [
        "E Prompts",
        "ACL 2023 Responsible NLP Checklist",
    ]
    checklist = sections[-1]
    assert checklist.metadata["section_scope"] == "terminal_checklist"
    assert prior[1].section_id == sections[0].section_id
    assert all(
        element.section_id == checklist.section_id
        for element in checklist_page + continuation
    )
    assert checklist.page_ids == ["page:1", "page:2"]


def test_page_header_mention_without_numbered_items_is_not_a_section() -> None:
    heading = _element(
        0,
        "3 Evaluation",
        ElementType.HEADING,
        level=1,
        y1=0.08,
        page_index=0,
    )
    title = _element(
        1,
        "Project Checklist",
        ElementType.PARAGRAPH,
        y1=0.05,
        page_index=0,
    )
    title.metadata["mineru_type"] = "page_header"
    body = _element(
        2,
        "We used this checklist during evaluation.",
        ElementType.PARAGRAPH,
        y1=0.20,
        page_index=0,
    )
    elements = [heading, title, body]

    sections = SectionBuilder().build(
        "doc:test",
        [_page(elements)],
        elements,
    )

    assert [section.title for section in sections] == ["3 Evaluation"]
    assert title.section_id == sections[0].section_id


def test_appendix_prompt_headings_share_the_same_parent() -> None:
    elements = [
        _element(
            0,
            "E Prompts",
            ElementType.HEADING,
            level=2,
            y1=0.05,
            height=0.035,
        ),
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
    sections = SectionBuilder().build("doc:test", [_page(elements)], elements)

    appendix, direct, zero_shot = sections
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
    sections = SectionBuilder().build("doc:test", [_page(elements)], elements)

    assert sections == []
    assert elements[0].element_type == ElementType.PARAGRAPH
    assert elements[0].heading_level is None
    assert result.decisions[0].action == HeadingAction.DEMOTED_TO_PARAGRAPH


def test_repeated_pew_header_is_excluded_but_preserved() -> None:
    pages: list[Page] = []
    elements: list[Element] = []
    for page_index in range(4):
        page_elements = [
            _element(
                0,
                "PEW RESEARCH CENTER",
                ElementType.HEADING,
                level=1,
                y1=0.02,
                height=0.018,
                page_index=page_index,
            ),
            _element(
                1,
                f"{page_index + 1} Findings",
                ElementType.HEADING,
                level=2,
                y1=0.16,
                height=0.03,
                page_index=page_index,
            ),
            _element(
                2,
                "Body",
                ElementType.PARAGRAPH,
                y1=0.22,
                page_index=page_index,
            ),
        ]
        pages.append(_page(page_elements, page_index))
        elements.extend(page_elements)

    detected = RepeatedHeaderFooterDetector().detect(pages, elements)
    result = HeadingHierarchyBuilder().build("doc:test", pages, elements)
    sections = SectionBuilder().build("doc:test", pages, elements)

    headers = [
        element
        for element in elements
        if element.text == "PEW RESEARCH CENTER"
    ]
    assert len(detected.decisions) == 4
    assert all(item.metadata["repeated_region"] == "page_header" for item in headers)
    assert all(item.heading_level is None for item in headers)
    assert all(item.section_id is None for item in headers)
    assert [section.level for section in sections] == [1, 1, 1, 1]
    assert sum(
        decision.action == HeadingAction.EXCLUDED_REPEATED_REGION
        for decision in result.decisions
    ) == 4


def test_slide_titles_with_same_style_are_siblings_across_pages() -> None:
    pages: list[Page] = []
    elements: list[Element] = []
    for page_index, text in enumerate(
        ["Deck title", "Market overview", "Architecture", "Results"]
    ):
        page_elements = [
            _element(
                0,
                text,
                ElementType.HEADING,
                level=page_index + 1,
                y1=0.08,
                height=0.055,
                page_index=page_index,
                style={"font_size": 30, "bold": True},
            )
        ]
        pages.append(_page(page_elements, page_index))
        elements.extend(page_elements)

    result = HeadingHierarchyBuilder().build("doc:test", pages, elements)
    sections = SectionBuilder().build("doc:test", pages, elements)

    assert result.document_title == "Deck title"
    assert [section.title for section in sections] == [
        "Market overview",
        "Architecture",
        "Results",
    ]
    assert [section.level for section in sections] == [1, 1, 1]
    assert all(section.parent_section_id is None for section in sections)


def test_part_and_costco_item_patterns_create_two_levels() -> None:
    elements = [
        _element(0, "PART I", ElementType.HEADING, y1=0.22),
        _element(1, "Item 1鈥擝usiness", ElementType.HEADING, y1=0.28),
        _element(2, "Item 1A鈥擱isk Factors", ElementType.HEADING, y1=0.34),
        _element(3, "PART II", ElementType.HEADING, y1=0.40),
    ]
    pages = [_page(elements)]

    HeadingHierarchyBuilder().build("doc:test", pages, elements)
    sections = SectionBuilder().build("doc:test", pages, elements)

    assert [section.level for section in sections] == [1, 2, 2, 1]
    assert sections[1].parent_section_id == sections[0].section_id
    assert sections[2].parent_section_id == sections[0].section_id


def test_non_explicit_heading_inference_is_conservatively_capped_at_h3() -> None:
    elements = [
        _element(
            0,
            "Introduction",
            ElementType.HEADING,
            level=2,
            y1=None,
            style={"font_size": 18},
        ),
        _element(
            1,
            "Visually uncertain subheading",
            ElementType.HEADING,
            level=8,
            y1=None,
            style={"font_size": 11},
        ),
    ]
    pages = [_page(elements)]

    result = HeadingHierarchyBuilder().build("doc:test", pages, elements)

    assert elements[0].heading_level == 1
    assert elements[1].heading_level == 3
    uncertain = next(
        item for item in result.decisions if item.element_id == elements[1].element_id
    )
    assert uncertain.evidence["conservative_max_inferred_level"] == 3


def test_parser_header_text_excludes_repeated_branding_at_variable_positions() -> None:
    pages: list[Page] = []
    elements: list[Element] = []
    for page_index, y1 in enumerate([0.84, 0.52, 0.68]):
        page_elements = [
            _element(
                0,
                "PEW RESEARCH CENTER",
                ElementType.PARAGRAPH,
                y1=0.05,
                page_index=page_index,
            ),
            _element(
                1,
                "PEW RESEARCH CENTER",
                ElementType.HEADING,
                level=2,
                y1=y1,
                page_index=page_index,
            ),
        ]
        page_elements[0].metadata["mineru_type"] = "page_header"
        pages.append(_page(page_elements, page_index))
        elements.extend(page_elements)

    detected = RepeatedHeaderFooterDetector().detect(pages, elements)
    result = HeadingHierarchyBuilder().build("doc:test", pages, elements)
    sections = SectionBuilder().build("doc:test", pages, elements)

    assert len(detected.decisions) == 6
    assert sections == []
    assert sum(
        item.action == HeadingAction.EXCLUDED_REPEATED_REGION
        for item in result.decisions
    ) == 3


def test_gpl_brochure_indentation_is_only_a_fallback_signal() -> None:
    elements = [
        _element(
            0,
            "Graduate Programs",
            ElementType.HEADING,
            y1=0.25,
            height=0.04,
            x1=0.08,
        ),
        _element(
            1,
            "Program Requirements",
            ElementType.HEADING,
            y1=0.32,
            x1=0.15,
        ),
        _element(
            2,
            "Admissions",
            ElementType.HEADING,
            y1=0.39,
            x1=0.15,
        ),
    ]
    pages = [_page(elements)]

    HeadingHierarchyBuilder().build("doc:test", pages, elements)
    sections = SectionBuilder().build("doc:test", pages, elements)

    assert [section.level for section in sections] == [1, 2, 2]
    assert sections[1].parent_section_id == sections[0].section_id
    assert sections[2].parent_section_id == sections[0].section_id


def test_outline_uses_normalized_section_tree(parsed_document) -> None:
    outline = build_document_outline(parsed_document)
    markdown = outline_markdown(outline)

    assert outline.title == "Soft Structure Fixture"
    assert outline.sections[0].title == "1 Introduction"
    assert outline.sections[0].level == 1
    assert "- H1 1 Introduction" in markdown
    assert parsed_document.metadata["heading_decisions"]
