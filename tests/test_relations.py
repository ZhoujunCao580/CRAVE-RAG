from __future__ import annotations

import pytest

from softdoc.ids import bbox_id, element_id, provenance_id
from softdoc.models import (
    BoundingBox,
    Element,
    ElementType,
    Provenance,
    RelationSource,
    RelationStatus,
    RelationType,
)
from softdoc.relations import FootnoteRelationValidator, RelationBuilder


def _code_continuation_document(
    parsed_document,
    source_type: ElementType,
    target_type: ElementType,
    *,
    source_bottom: float = 1350,
    target_top: float = 50,
    source_text: str = "the generated program continues with",
    target_text: str = "the remaining generated steps",
):
    document = parsed_document.model_copy(deep=True)
    source_page = document.pages[0]
    target_page = document.pages[1]
    source_id = element_id(
        document.document_id,
        source_page.page_index,
        90,
        source_type.value,
        "continuation-source",
    )
    target_id = element_id(
        document.document_id,
        target_page.page_index,
        90,
        target_type.value,
        "continuation-target",
    )
    source = Element(
        element_id=source_id,
        document_id=document.document_id,
        page_id=source_page.page_id,
        page_number=source_page.page_number,
        element_type=source_type,
        reading_order=0,
        bbox=BoundingBox.from_raw(
            bbox_id=bbox_id(source_id),
            raw=(100, 100, 900, source_bottom),
            page_width=1000,
            page_height=1400,
        ),
        text=source_text,
        provenance=document.provenance.model_copy(deep=True),
    )
    target = Element(
        element_id=target_id,
        document_id=document.document_id,
        page_id=target_page.page_id,
        page_number=target_page.page_number,
        element_type=target_type,
        reading_order=0,
        bbox=BoundingBox.from_raw(
            bbox_id=bbox_id(target_id),
            raw=(100, target_top, 900, 900),
            page_width=1000,
            page_height=1400,
        ),
        text=target_text,
        provenance=document.provenance.model_copy(deep=True),
    )
    document.pages = [
        source_page.model_copy(
            update={
                "element_ids": [source_id],
                "reading_order": [source_id],
            }
        ),
        target_page.model_copy(
            update={
                "element_ids": [target_id],
                "reading_order": [target_id],
            }
        ),
    ]
    document.elements = [source, target]
    document.sections = []
    document.relations = []
    return document, source, target


def _two_page_continuation_document(
    parsed_document,
    *,
    source_text: str,
    target_text: str,
    source_type: ElementType = ElementType.PARAGRAPH,
    target_type: ElementType = ElementType.PARAGRAPH,
    source_bottom: float = 1350,
    target_top: float = 50,
    section_path: list[str] | None = None,
):
    document = parsed_document.model_copy(deep=True)
    source_page, target_page = document.pages[:2]
    source_id = element_id(
        document.document_id,
        source_page.page_index,
        90,
        source_type.value,
        "boundary-source",
    )
    target_id = element_id(
        document.document_id,
        target_page.page_index,
        90,
        target_type.value,
        "boundary-target",
    )
    shared_section_id = "section:shared-boundary"
    source = Element(
        element_id=source_id,
        document_id=document.document_id,
        page_id=source_page.page_id,
        page_number=source_page.page_number,
        element_type=source_type,
        reading_order=0,
        bbox=BoundingBox.from_raw(
            bbox_id=bbox_id(source_id),
            raw=(100, max(50, source_bottom - 400), 900, source_bottom),
            page_width=1000,
            page_height=1400,
        ),
        text=source_text,
        section_id=shared_section_id,
        section_path=section_path,
        provenance=document.provenance.model_copy(deep=True),
    )
    target = Element(
        element_id=target_id,
        document_id=document.document_id,
        page_id=target_page.page_id,
        page_number=target_page.page_number,
        element_type=target_type,
        reading_order=0,
        bbox=BoundingBox.from_raw(
            bbox_id=bbox_id(target_id),
            raw=(100, target_top, 900, 400),
            page_width=1000,
            page_height=1400,
        ),
        text=target_text,
        section_id=shared_section_id,
        section_path=section_path,
        provenance=document.provenance.model_copy(deep=True),
    )
    document.pages = [
        source_page.model_copy(
            update={
                "element_ids": [source_id],
                "reading_order": [source_id],
            }
        ),
        target_page.model_copy(
            update={
                "element_ids": [target_id],
                "reading_order": [target_id],
            }
        ),
    ]
    document.elements = [source, target]
    document.sections = []
    document.relations = []
    return document, source, target


def test_cross_page_explicit_reference(parsed_document) -> None:
    relations = [
        relation for relation in parsed_document.relations if relation.relation_type == RelationType.REFERS_TO
    ]
    match = next(
        relation
        for relation in relations
        if relation.evidence[0].data.get("matched_text") == "Figure 3"
    )
    elements = {element.element_id: element for element in parsed_document.elements}
    assert elements[match.source_id].page_number == 2
    assert elements[match.target_id].page_number == 3
    assert match.created_by == RelationSource.EXPLICIT_REFERENCE
    assert match.status == RelationStatus.CONFIRMED


def test_page_13_paragraph_can_reference_page_12_figure_3(parsed_document) -> None:
    document = parsed_document.model_copy(deep=True)
    page_12, page_13, page_14 = document.pages
    page_12.page_number = 12
    page_13.page_number = 13
    page_14.page_number = 14
    for element in document.elements:
        if element.page_id == page_12.page_id:
            element.page_number = 12
        elif element.page_id == page_13.page_id:
            element.page_number = 13
        else:
            element.page_number = 14
    page_12_figure = next(
        element
        for element in document.elements
        if element.page_id == page_12.page_id and element.element_type == ElementType.FIGURE
    )
    page_12_caption = next(
        element
        for element in document.elements
        if element.page_id == page_12.page_id and element.element_type == ElementType.CAPTION
    )
    page_12_caption.text = "Figure 3. Target on page 12."
    page_14_caption = next(
        element
        for element in document.elements
        if element.page_id == page_14.page_id and element.element_type == ElementType.CAPTION
    )
    page_14_caption.text = "Figure 4. Another target."
    document.relations = []
    RelationBuilder(document).build_all()
    reference = next(
        relation
        for relation in document.relations
        if relation.relation_type == RelationType.REFERS_TO
        and relation.evidence[0].data.get("matched_text") == "Figure 3"
    )
    source = next(element for element in document.elements if element.element_id == reference.source_id)
    assert source.page_number == 13
    assert reference.target_id == page_12_figure.element_id
    assert page_12_figure.page_number == 12


@pytest.mark.parametrize(
    ("mention", "target_label", "expected_kind", "expected_subreference"),
    [
        ("Figure 1a", "Figure 1", "figure", "a"),
        ("Figure 1(a)", "Figure 1", "figure", "a"),
        ("Figure 3b-c", "Figure 3", "figure", "b-c"),
        ("Fig. 3(b–c)", "Figure 3", "figure", "b-c"),
        ("图3", "Figure 3", "figure", None),
        ("表2", "Table 2", "table", None),
        ("第4.1节", "Section 4.1", "section", None),
    ],
)
def test_explicit_reference_variants_and_main_target_fallback(
    parsed_document,
    mention: str,
    target_label: str,
    expected_kind: str,
    expected_subreference: str | None,
) -> None:
    document = parsed_document.model_copy(deep=True)
    source = next(
        element
        for element in document.elements
        if element.element_type == ElementType.PARAGRAPH
        and element.text
        and "cross-page example" in element.text
    )
    source.text = f"See {mention} for details."

    if expected_kind == "section":
        target = document.sections[0]
        target.title = "4.1 Methods"
        target_id = target.section_id
    else:
        caption = next(
            element
            for element in document.elements
            if element.element_type == ElementType.CAPTION
            and element.text
            and element.text.startswith(target_label)
        )
        target_id = str(caption.metadata["target_element_id"])

    references = RelationBuilder(document).build_explicit_reference_relations()
    relation = next(
        item
        for item in references
        if item.source_id == source.element_id
        and item.evidence[0].data["matched_text"] == mention
    )

    assert relation.target_id == target_id
    assert relation.metadata["reference_kind"] == expected_kind
    assert relation.metadata["reference_base_number"] in {"1", "2", "3", "4.1"}
    assert relation.metadata["fallback_to_main"] is (
        expected_subreference is not None
    )
    if expected_subreference is not None:
        assert relation.metadata["subfigure_number"] == expected_subreference
        assert relation.evidence[0].data["subreference"] == expected_subreference


def test_section_reference_allows_sentence_period_after_decimal_number(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    source = next(
        element
        for element in document.elements
        if element.element_type == ElementType.PARAGRAPH
        and element.text
        and "cross-page example" in element.text
    )
    source.text = "The details are given in Section 4.1."
    target = document.sections[0]
    target.title = "4.1 Methods"

    references = RelationBuilder(document).build_explicit_reference_relations()
    relation = next(
        item
        for item in references
        if item.source_id == source.element_id
        and item.evidence[0].data["matched_text"] == "Section 4.1"
    )

    assert relation.target_id == target.section_id
    assert relation.metadata["reference_number"] == "4.1"


def test_paragraph_misclassified_caption_can_anchor_figure_number(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    source = next(
        element
        for element in document.elements
        if element.element_type == ElementType.PARAGRAPH
        and element.text
        and "cross-page example" in element.text
    )
    source.text = "See Figure 3 for details."
    caption = next(
        element
        for element in document.elements
        if element.element_type == ElementType.CAPTION
        and element.text
        and element.text.startswith("Figure 3")
    )
    target_id = str(caption.metadata["target_element_id"])
    caption.element_type = ElementType.PARAGRAPH
    caption.metadata.pop("target_element_id", None)
    document.relations = []

    references = RelationBuilder(document).build_explicit_reference_relations()
    relation = next(
        item
        for item in references
        if item.source_id == source.element_id
        and item.evidence[0].data["matched_text"] == "Figure 3"
    )

    assert relation.target_id == target_id
    assert relation.metadata["target_resolution_rule"] == (
        "paragraph_caption_layout_heuristic"
    )
    assert relation.metadata["target_label_source_id"] == caption.element_id
    assert caption.element_type == ElementType.PARAGRAPH


def test_footnote_validator_confirms_strong_parser_binding(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    footnote = next(
        element
        for element in document.elements
        if element.element_type == ElementType.FOOTNOTE
    )

    relation = next(
        item
        for item in FootnoteRelationValidator(document).build_relations()
        if item.source_id == footnote.element_id
    )

    assert relation.status == RelationStatus.CONFIRMED
    assert relation.created_by == RelationSource.DETERMINISTIC_RULE
    assert relation.metadata["parser_binding"] is True
    assert relation.evidence[0].data["same_page"] is True


def test_footnote_validator_downgrades_weak_parser_binding_to_candidate(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    footnote = next(
        element
        for element in document.elements
        if element.element_type == ElementType.FOOTNOTE
    )
    footnote.page_id = document.pages[-1].page_id
    footnote.page_number = document.pages[-1].page_number
    footnote.text = "Additional detail without a marker."
    original_element_id = footnote.element_id

    relation = next(
        item
        for item in FootnoteRelationValidator(document).build_relations()
        if item.source_id == original_element_id
    )

    assert relation.status == RelationStatus.CANDIDATE
    assert relation.metadata["parser_binding"] is True
    assert relation.evidence[0].data["same_page"] is False
    assert any(
        element.element_id == original_element_id
        for element in document.elements
    )


def test_footnote_validator_can_use_exact_superscript_marker(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    page = document.pages[0]
    target = next(
        element
        for element in document.elements
        if element.page_id == page.page_id
        and element.element_type == ElementType.PARAGRAPH
    )
    target.text = f"{target.text}<sup>7</sup>"
    footnote_id = element_id(
        document.document_id,
        page.page_index,
        99,
        ElementType.FOOTNOTE.value,
        "marker-footnote",
    )
    footnote = Element(
        element_id=footnote_id,
        document_id=document.document_id,
        page_id=page.page_id,
        page_number=page.page_number,
        element_type=ElementType.FOOTNOTE,
        reading_order=99,
        bbox=BoundingBox.from_raw(
            bbox_id=bbox_id(footnote_id),
            raw=(100, 1250, 900, 1320),
            page_width=1000,
            page_height=1400,
        ),
        text="<sup>7</sup>Marker-linked note.",
        provenance=document.provenance.model_copy(deep=True),
    )
    document.elements.append(footnote)

    relation = next(
        item
        for item in FootnoteRelationValidator(document).build_relations()
        if item.source_id == footnote_id
    )

    assert relation.target_id == target.element_id
    assert relation.status == RelationStatus.CONFIRMED
    assert relation.metadata["parser_binding"] is False
    assert relation.metadata["marker_binding"] is True
    assert relation.metadata["footnote_marker"] == "7"


def test_footnote_validator_binds_markerless_aligned_continuation(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    page = document.pages[0]
    target = next(
        element
        for element in document.elements
        if element.page_id == page.page_id
        and element.element_type == ElementType.PARAGRAPH
    )
    target.text = f"{target.text}<sup>4</sup>"
    anchor_id = element_id(
        document.document_id,
        page.page_index,
        97,
        ElementType.FOOTNOTE.value,
        "anchor-footnote",
    )
    continuation_id = element_id(
        document.document_id,
        page.page_index,
        98,
        ElementType.FOOTNOTE.value,
        "continued-footnote",
    )
    anchor = Element(
        element_id=anchor_id,
        document_id=document.document_id,
        page_id=page.page_id,
        page_number=page.page_number,
        element_type=ElementType.FOOTNOTE,
        reading_order=97,
        bbox=BoundingBox.from_raw(
            bbox_id=bbox_id(anchor_id),
            raw=(500, 1200, 850, 1240),
            page_width=1000,
            page_height=1400,
        ),
        text="<sup>4</sup>https://example.test/model/",
        provenance=document.provenance.model_copy(deep=True),
    )
    continuation = Element(
        element_id=continuation_id,
        document_id=document.document_id,
        page_id=page.page_id,
        page_number=page.page_number,
        element_type=ElementType.FOOTNOTE,
        reading_order=98,
        bbox=BoundingBox.from_raw(
            bbox_id=bbox_id(continuation_id),
            raw=(500, 1238, 900, 1280),
            page_width=1000,
            page_height=1400,
        ),
        text="remaining-checkpoint-name",
        provenance=document.provenance.model_copy(deep=True),
    )
    document.elements.extend([anchor, continuation])

    relations = FootnoteRelationValidator(document).build_relations()
    continued = next(
        relation
        for relation in relations
        if relation.source_id == continuation_id
    )

    assert continued.target_id == target.element_id
    assert continued.status == RelationStatus.CONFIRMED
    assert (
        continued.evidence[0].rule
        == "markerless_footnote_line_continuation"
    )


def test_shared_visual_note_builds_one_to_many_footnote_relations(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    page = document.pages[0]
    visual_ids: list[str] = []
    visual_boxes = (
        (100, 600, 340, 900),
        (380, 600, 620, 900),
        (660, 600, 900, 900),
    )
    for index, raw_bbox in enumerate(visual_boxes):
        visual_id = element_id(
            document.document_id,
            page.page_index,
            80 + index,
            ElementType.CHART.value,
            f"shared-{index}",
        )
        visual_ids.append(visual_id)
        visual = Element(
            element_id=visual_id,
            document_id=document.document_id,
            page_id=page.page_id,
            page_number=page.page_number,
            element_type=ElementType.CHART,
            reading_order=80 + index,
            bbox=BoundingBox.from_raw(
                bbox_id=bbox_id(visual_id),
                raw=raw_bbox,
                page_width=1000,
                page_height=1400,
            ),
            text=f"Chart panel {index + 1}",
            provenance=document.provenance.model_copy(deep=True),
        )
        document.elements.append(visual)

    footnote_id = element_id(
        document.document_id,
        page.page_index,
        90,
        ElementType.FOOTNOTE.value,
        "shared-note",
    )
    footnote = Element(
        element_id=footnote_id,
        document_id=document.document_id,
        page_id=page.page_id,
        page_number=page.page_number,
        element_type=ElementType.FOOTNOTE,
        reading_order=90,
        bbox=BoundingBox.from_raw(
            bbox_id=bbox_id(footnote_id),
            raw=(90, 910, 910, 980),
            page_width=1000,
            page_height=1400,
        ),
        text="Notes: Values may not sum to 100%.",
        provenance=document.provenance.model_copy(deep=True),
        metadata={"target_element_id": visual_ids[-1]},
    )
    document.elements.append(footnote)

    relations = [
        relation
        for relation in FootnoteRelationValidator(document).build_relations()
        if relation.source_id == footnote_id
    ]

    assert {relation.target_id for relation in relations} == set(visual_ids)
    assert all(
        relation.status == RelationStatus.CONFIRMED
        for relation in relations
    )
    assert all(
        relation.metadata["shared_visual_note"] is True
        for relation in relations
    )
    assert len(
        {
            relation.metadata["visual_group_id"]
            for relation in relations
        }
    ) == 1
    assert any(
        evidence.rule == "shared_visual_group_footnote"
        for relation in relations
        for evidence in relation.evidence
    )


def test_function_relation_can_cross_adjacent_pages(parsed_document) -> None:
    document = parsed_document.model_copy(deep=True)
    figure = next(
        element
        for element in document.elements
        if element.element_type == ElementType.FIGURE and element.page_number == 3
    )
    page_two = document.pages[1]
    caption_id = element_id(document.document_id, page_two.page_index, 99, "caption", "cross-page")
    caption = Element(
        element_id=caption_id,
        document_id=document.document_id,
        page_id=page_two.page_id,
        page_number=page_two.page_number,
        element_type=ElementType.CAPTION,
        reading_order=len(page_two.reading_order),
        bbox=BoundingBox.from_raw(
            bbox_id=bbox_id(caption_id),
            raw=(100, 900, 900, 950),
            page_width=1000,
            page_height=1400,
        ),
        text="Figure 3. Caption placed on the previous page.",
        provenance=Provenance(
            provenance_id=provenance_id("test", "fixture", caption_id),
            adapter="test",
            source_path="fixture",
            source_locator=caption_id,
        ),
        metadata={"target_element_id": figure.element_id},
    )
    document.elements.append(caption)
    page_two.element_ids.append(caption_id)
    page_two.reading_order.append(caption_id)
    relations = RelationBuilder(document).build_caption_relations()
    relation = next(item for item in relations if item.source_id == caption_id)
    assert relation.target_id == figure.element_id
    assert relation.evidence[0].data["cross_page"] is True


def test_table_continuation_is_candidate_only(parsed_document) -> None:
    document = parsed_document.model_copy(deep=True)
    source = next(element for element in document.elements if element.element_type == ElementType.TABLE)
    page_three = document.pages[2]
    target_id = element_id(document.document_id, page_three.page_index, 90, "table", "continued")
    target = Element(
        element_id=target_id,
        document_id=document.document_id,
        page_id=page_three.page_id,
        page_number=page_three.page_number,
        element_type=ElementType.TABLE,
        reading_order=0,
        bbox=BoundingBox.from_raw(
            bbox_id=bbox_id(target_id),
            raw=(100, 80, 900, 500),
            page_width=1000,
            page_height=1400,
        ),
        text="Table 2 continued",
        html=source.html,
        reference_label="Table 2",
        provenance=source.provenance.model_copy(deep=True),
        metadata={"column_count": 2},
    )
    document.elements.append(target)
    page_three.element_ids.insert(0, target_id)
    page_three.reading_order.insert(0, target_id)
    candidates = RelationBuilder(document).build_cross_page_continuation_candidates()
    table_candidate = next(
        relation
        for relation in candidates
        if relation.source_id == source.element_id and relation.target_id == target_id
    )
    assert table_candidate.relation_type == RelationType.CONTINUED_ON
    assert table_candidate.status == RelationStatus.CANDIDATE
    assert table_candidate.created_by == RelationSource.LAYOUT_HEURISTIC
    assert table_candidate.evidence


def test_table_header_similarity_recovers_continuation_with_column_drift(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    source = next(
        element
        for element in document.elements
        if element.element_type == ElementType.TABLE
    )
    source.reference_label = None
    source.html = (
        "<table><tr><th>Exhibit Number</th><th>Exhibit Description</th>"
        "<th>Filed Herewith</th></tr><tr><td>3.1</td><td>Articles</td>"
        "<td>Yes</td></tr></table>"
    )
    source.metadata["column_count"] = 3
    target_page = document.pages[2]
    target_id = element_id(
        document.document_id,
        target_page.page_index,
        90,
        ElementType.TABLE.value,
        "header-drift",
    )
    target = Element(
        element_id=target_id,
        document_id=document.document_id,
        page_id=target_page.page_id,
        page_number=target_page.page_number,
        element_type=ElementType.TABLE,
        reading_order=0,
        bbox=BoundingBox.from_raw(
            bbox_id=bbox_id(target_id),
            raw=(100, 80, 900, 500),
            page_width=1000,
            page_height=1400,
        ),
        html=(
            "<table><tr><th colspan='2'>Exhibit Number</th>"
            "<th>Exhibit Description</th><th>Filed Herewith</th></tr>"
            "<tr><td>3.2</td><td></td><td>Bylaws</td><td>No</td></tr>"
            "</table>"
        ),
        provenance=source.provenance.model_copy(deep=True),
        metadata={"column_count": 4},
    )
    document.elements.append(target)
    target_page.element_ids.insert(0, target_id)
    target_page.reading_order.insert(0, target_id)

    candidates = RelationBuilder(
        document
    ).build_cross_page_continuation_candidates()
    relation = next(
        item
        for item in candidates
        if item.source_id == source.element_id
        and item.target_id == target.element_id
    )

    assert relation.status == RelationStatus.CANDIDATE
    assert relation.evidence[0].data["column_count_match"] is False
    assert relation.evidence[0].data["table_headers_match"] is True
    assert relation.evidence[0].data["header_token_similarity"] >= 0.5


def test_slide_footer_and_next_slide_title_are_not_continuation(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    source_page, target_page = document.pages[:2]
    source = next(
        element
        for element in document.elements
        if element.page_id == source_page.page_id
        and element.element_type == ElementType.PARAGRAPH
    )
    target = next(
        element
        for element in document.elements
        if element.page_id == target_page.page_id
        and element.element_type == ElementType.PARAGRAPH
    )
    source.text = "THEBIGDATAGROUP.COM"
    source.bbox = BoundingBox.from_raw(
        bbox_id=bbox_id(source.element_id),
        raw=(100, 1250, 300, 1320),
        page_width=1000,
        page_height=1400,
    )
    target.text = "WHAT IS BIG DATA?"
    target.bbox = BoundingBox.from_raw(
        bbox_id=bbox_id(target.element_id),
        raw=(700, 80, 920, 180),
        page_width=1000,
        page_height=1400,
    )
    document.pages = [
        source_page.model_copy(
            update={
                "element_ids": [source.element_id],
                "reading_order": [source.element_id],
            }
        ),
        target_page.model_copy(
            update={
                "element_ids": [target.element_id],
                "reading_order": [target.element_id],
            }
        ),
    ]
    document.elements = [source, target]

    candidates = RelationBuilder(
        document
    ).build_cross_page_continuation_candidates()

    assert not any(
        item.source_id == source.element_id
        and item.target_id == target.element_id
        for item in candidates
    )


def test_paragraph_boundary_search_ignores_later_chart_title(
    parsed_document,
) -> None:
    document, source, target = _two_page_continuation_document(
        parsed_document,
        source_text=(
            "These shares changed only modestly, when no more than "
            "about a third of the"
        ),
        target_text="public identified as independents.",
        source_bottom=650,
    )
    source_page = document.pages[0]
    decoy_id = element_id(
        document.document_id,
        source_page.page_index,
        91,
        ElementType.PARAGRAPH.value,
        "chart-title-decoy",
    )
    decoy = Element(
        element_id=decoy_id,
        document_id=document.document_id,
        page_id=source_page.page_id,
        page_number=source_page.page_number,
        element_type=ElementType.PARAGRAPH,
        reading_order=1,
        bbox=BoundingBox.from_raw(
            bbox_id=bbox_id(decoy_id),
            raw=(100, 700, 900, 780),
            page_width=1000,
            page_height=1400,
        ),
        text=(
            "Independents Outnumber Republicans and Democrats, "
            "But Few Are Truly Independent"
        ),
        section_id=source.section_id,
        section_path=source.section_path,
        provenance=document.provenance.model_copy(deep=True),
    )
    document.elements.append(decoy)
    document.pages[0] = source_page.model_copy(
        update={
            "element_ids": [source.element_id, decoy.element_id],
            "reading_order": [source.element_id, decoy.element_id],
        }
    )

    relations = RelationBuilder(
        document
    ).build_cross_page_continuation_candidates()
    relation = next(
        item
        for item in relations
        if item.relation_type == RelationType.CONTINUED_ON
    )

    assert relation.source_id == source.element_id
    assert relation.target_id == target.element_id
    assert relation.evidence[0].data["evaluated_pair_count"] == 2


def test_short_complete_target_is_not_paragraph_continuation(
    parsed_document,
) -> None:
    document, source, target = _two_page_continuation_document(
        parsed_document,
        source_text=(
            "The shares are traded under both depository systems. "
            "A total of 34% of the share capital stands dematerialised"
        ),
        target_text=(
            "Under the Depository System, the identification number "
            "allotted to the shares is INE260B01010."
        ),
    )

    relations = RelationBuilder(
        document
    ).build_cross_page_continuation_candidates()

    assert not any(
        item.source_id == source.element_id
        and item.target_id == target.element_id
        for item in relations
    )


def test_slide_profile_resets_standalone_output_and_code(
    parsed_document,
) -> None:
    document, source, target = _two_page_continuation_document(
        parsed_document,
        source_text="already after 5 min: valid_y_misclass: 0.0619999989867",
        target_text="In [319]: plt.plot(score_train[2:])",
    )
    document.metadata["document_profile"] = {"profile": "slides"}

    relations = RelationBuilder(
        document
    ).build_cross_page_continuation_candidates()

    assert not any(
        item.source_id == source.element_id
        and item.target_id == target.element_id
        for item in relations
    )


@pytest.mark.parametrize(
    "target_text",
    [
        "15",
        "inlA",
        "https://www.example.org/admissions",
        "(a) PathMNIST (b) OrganAMNIST (c) BloodMNIST",
    ],
)
def test_non_prose_target_is_not_paragraph_continuation(
    parsed_document,
    target_text: str,
) -> None:
    document, source, target = _two_page_continuation_document(
        parsed_document,
        source_text=(
            "The evaluation uses multiple datasets and the next page "
            "reports results for"
        ),
        target_text=target_text,
    )

    relations = RelationBuilder(
        document
    ).build_cross_page_continuation_candidates()

    assert not any(
        item.source_id == source.element_id
        and item.target_id == target.element_id
        for item in relations
    )


def test_unreadable_ocr_is_not_paragraph_continuation(
    parsed_document,
) -> None:
    document, source, target = _two_page_continuation_document(
        parsed_document,
        source_text=".-.\x15\x14< .\x184-4 '\x18 .-='\x154- .-+",
        target_text=(
            "government policy and export potential are discussed here."
        ),
    )

    relations = RelationBuilder(
        document
    ).build_cross_page_continuation_candidates()

    assert not any(
        item.source_id == source.element_id
        and item.target_id == target.element_id
        for item in relations
    )


def test_page_bottom_footnote_is_not_linked_to_next_page_body(
    parsed_document,
) -> None:
    document, source, target = _two_page_continuation_document(
        parsed_document,
        source_text="1The program code and data are publicly available at",
        target_text=(
            "The generated reasoning program serves as a guide for "
            "verifying the claim."
        ),
    )

    relations = RelationBuilder(
        document
    ).build_cross_page_continuation_candidates()

    assert not any(
        item.source_id == source.element_id
        and item.target_id == target.element_id
        for item in relations
    )


def test_numbered_form_question_can_continue_across_page(
    parsed_document,
) -> None:
    document, source, target = _two_page_continuation_document(
        parsed_document,
        source_text=(
            "13. Does this investment directly support one of the PMA"
        ),
        target_text="initiatives?",
    )

    relations = RelationBuilder(
        document
    ).build_cross_page_continuation_candidates()

    assert any(
        item.source_id == source.element_id
        and item.target_id == target.element_id
        for item in relations
    )


def test_bibliography_list_fragments_continue_across_page(
    parsed_document,
) -> None:
    document, source, target = _two_page_continuation_document(
        parsed_document,
        source_text=(
            "Liu et al. 2020. Fine-grained fact verification. "
            "In Proceedings"
        ),
        target_text=(
            "of the 58th Annual Meeting of the Association for "
            "Computational Linguistics, pages 7342-7351."
        ),
        source_type=ElementType.LIST,
        target_type=ElementType.LIST,
        section_path=["References"],
    )

    relations = RelationBuilder(
        document
    ).build_cross_page_continuation_candidates()
    relation = next(
        item
        for item in relations
        if item.source_id == source.element_id
        and item.target_id == target.element_id
    )

    assert relation.status == RelationStatus.CANDIDATE
    assert relation.evidence[0].rule == (
        "bounded_cross_page_list_continuation"
    )
    assert relation.evidence[0].data["context_kind"] == "bibliography"


def test_index_paragraph_can_continue_as_list_on_next_page(
    parsed_document,
) -> None:
    document, source, target = _two_page_continuation_document(
        parsed_document,
        source_text="F F1 to F12 function keys 29 Fast-forward key 29",
        target_text=(
            "flashing question mark 41\nForce Quit 40\n"
            "forward delete 32"
        ),
        source_type=ElementType.PARAGRAPH,
        target_type=ElementType.LIST,
        section_path=["Index"],
    )

    relations = RelationBuilder(
        document
    ).build_cross_page_continuation_candidates()
    relation = next(
        item
        for item in relations
        if item.source_id == source.element_id
        and item.target_id == target.element_id
    )

    assert relation.evidence[0].data["context_kind"] == "index"
    assert relation.evidence[0].data["source_element_type"] == "paragraph"
    assert relation.evidence[0].data["target_element_type"] == "list"


@pytest.mark.parametrize(
    ("source_type", "target_type"),
    [
        (ElementType.CODE, ElementType.CODE),
        (ElementType.CODE, ElementType.ALGORITHM),
        (ElementType.ALGORITHM, ElementType.CODE),
        (ElementType.ALGORITHM, ElementType.ALGORITHM),
    ],
)
def test_code_and_algorithm_cross_page_pairs_are_compatible_candidates(
    parsed_document,
    source_type: ElementType,
    target_type: ElementType,
) -> None:
    document, source, target = _code_continuation_document(
        parsed_document,
        source_type,
        target_type,
    )

    candidates = RelationBuilder(
        document
    ).build_cross_page_continuation_candidates()
    relation = next(
        item
        for item in candidates
        if item.source_id == source.element_id
        and item.target_id == target.element_id
    )

    assert relation.relation_type == RelationType.CONTINUED_ON
    assert relation.status == RelationStatus.CANDIDATE
    assert relation.created_by == RelationSource.LAYOUT_HEURISTIC
    assert relation.evidence[0].rule == "bounded_cross_page_code_continuation"
    assert relation.evidence[0].data["content_family"] == "code_like"
    assert relation.evidence[0].data["source_element_type"] == source_type.value
    assert relation.evidence[0].data["target_element_type"] == target_type.value
    assert relation.evidence[0].data["compatible_type_pair"] is True


def test_code_continuation_requires_page_boundary_layout(parsed_document) -> None:
    document, source, target = _code_continuation_document(
        parsed_document,
        ElementType.ALGORITHM,
        ElementType.CODE,
        source_bottom=700,
        target_top=500,
    )

    candidates = RelationBuilder(
        document
    ).build_cross_page_continuation_candidates()

    assert not any(
        relation.source_id == source.element_id
        and relation.target_id == target.element_id
        for relation in candidates
    )


def test_target_caption_can_confirm_multi_page_listing_layout(
    parsed_document,
) -> None:
    document, source, target = _code_continuation_document(
        parsed_document,
        ElementType.ALGORITHM,
        ElementType.CODE,
        source_text="label = Predict(fact_1)",
        target_text="# The next example begins here",
    )
    without_caption = RelationBuilder(
        document
    ).build_cross_page_continuation_candidates()
    assert not any(
        relation.source_id == source.element_id
        and relation.target_id == target.element_id
        for relation in without_caption
    )

    target_page = document.pages[1]
    caption_id = element_id(
        document.document_id,
        target_page.page_index,
        91,
        ElementType.CAPTION.value,
        "listing-caption",
    )
    caption = Element(
        element_id=caption_id,
        document_id=document.document_id,
        page_id=target_page.page_id,
        page_number=target_page.page_number,
        element_type=ElementType.CAPTION,
        reading_order=1,
        bbox=BoundingBox.from_raw(
            bbox_id=bbox_id(caption_id),
            raw=(200, 920, 800, 980),
            page_width=1000,
            page_height=1400,
        ),
        text="Listing 1: Complete prompt.",
        provenance=document.provenance.model_copy(deep=True),
        metadata={"target_element_id": target.element_id},
    )
    document.elements.append(caption)
    document.pages[1] = target_page.model_copy(
        update={
            "element_ids": [target.element_id, caption_id],
            "reading_order": [target.element_id, caption_id],
        }
    )

    with_caption = RelationBuilder(
        document
    ).build_cross_page_continuation_candidates()
    relation = next(
        item
        for item in with_caption
        if item.source_id == source.element_id
        and item.target_id == target.element_id
    )
    assert relation.status == RelationStatus.CANDIDATE
    assert relation.evidence[0].data["target_has_caption"] is True
    assert relation.evidence[0].data["target_starts_new_construct"] is True


def test_source_caption_prevents_linking_to_next_listing(parsed_document) -> None:
    document, source, target = _code_continuation_document(
        parsed_document,
        ElementType.CODE,
        ElementType.ALGORITHM,
    )
    source_page = document.pages[0]
    caption_id = element_id(
        document.document_id,
        source_page.page_index,
        91,
        ElementType.CAPTION.value,
        "completed-listing-caption",
    )
    caption = Element(
        element_id=caption_id,
        document_id=document.document_id,
        page_id=source_page.page_id,
        page_number=source_page.page_number,
        element_type=ElementType.CAPTION,
        reading_order=1,
        bbox=BoundingBox.from_raw(
            bbox_id=bbox_id(caption_id),
            raw=(200, 1360, 800, 1390),
            page_width=1000,
            page_height=1400,
        ),
        text="Listing 1: Completed listing.",
        provenance=document.provenance.model_copy(deep=True),
        metadata={"target_element_id": source.element_id},
    )
    document.elements.append(caption)
    document.pages[0] = source_page.model_copy(
        update={
            "element_ids": [source.element_id, caption_id],
            "reading_order": [source.element_id, caption_id],
        }
    )

    candidates = RelationBuilder(
        document
    ).build_cross_page_continuation_candidates()

    assert not any(
        relation.source_id == source.element_id
        and relation.target_id == target.element_id
        for relation in candidates
    )


def test_plural_listing_references_resolve_each_number(
    parsed_document,
) -> None:
    document, first, second = _code_continuation_document(
        parsed_document,
        ElementType.CODE,
        ElementType.CODE,
    )
    source_page, target_page = document.pages
    reference_id = element_id(
        document.document_id,
        source_page.page_index,
        91,
        ElementType.PARAGRAPH.value,
        "listing-reference",
    )
    reference = Element(
        element_id=reference_id,
        document_id=document.document_id,
        page_id=source_page.page_id,
        page_number=source_page.page_number,
        element_type=ElementType.PARAGRAPH,
        reading_order=2,
        text="The complete prompts are shown in Listings 1 and 2.",
        provenance=document.provenance.model_copy(deep=True),
    )
    captions: list[Element] = []
    for index, (target, page, number) in enumerate(
        (
            (first, source_page, 1),
            (second, target_page, 2),
        ),
        start=1,
    ):
        caption_id = element_id(
            document.document_id,
            page.page_index,
            92 + index,
            ElementType.CAPTION.value,
            f"listing-{number}",
        )
        captions.append(
            Element(
                element_id=caption_id,
                document_id=document.document_id,
                page_id=page.page_id,
                page_number=page.page_number,
                element_type=ElementType.CAPTION,
                reading_order=1,
                text=f"Listing {number}: Complete prompt.",
                provenance=document.provenance.model_copy(deep=True),
                metadata={"target_element_id": target.element_id},
            )
        )
    document.elements.extend([reference, *captions])

    builder = RelationBuilder(document)
    caption_relations = builder.build_caption_relations()
    references = builder.build_explicit_reference_relations(
        caption_relations
    )

    listing_relations = [
        relation
        for relation in references
        if relation.source_id == reference_id
        and relation.metadata.get("reference_kind") == "listing"
    ]
    assert {
        relation.metadata["reference_number"]
        for relation in listing_relations
    } == {"1", "2"}
    assert {
        relation.target_id for relation in listing_relations
    } == {first.element_id, second.element_id}


def test_numbered_caption_spans_aligned_chart_panels(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    page = document.pages[0]
    panels: list[Element] = []
    for index, x1 in enumerate((100, 370, 640)):
        panel_id = element_id(
            document.document_id,
            page.page_index,
            90 + index,
            ElementType.CHART.value,
            f"panel-{index}",
        )
        panels.append(
            Element(
                element_id=panel_id,
                document_id=document.document_id,
                page_id=page.page_id,
                page_number=page.page_number,
                element_type=ElementType.CHART,
                reading_order=index,
                bbox=BoundingBox.from_raw(
                    bbox_id=bbox_id(panel_id),
                    raw=(x1, 100, x1 + 220, 300),
                    page_width=1000,
                    page_height=1400,
                ),
                text=f"panel {index}",
                provenance=document.provenance.model_copy(deep=True),
            )
        )
    caption_id = element_id(
        document.document_id,
        page.page_index,
        94,
        ElementType.CAPTION.value,
        "panel-caption",
    )
    caption = Element(
        element_id=caption_id,
        document_id=document.document_id,
        page_id=page.page_id,
        page_number=page.page_number,
        element_type=ElementType.CAPTION,
        reading_order=3,
        bbox=BoundingBox.from_raw(
            bbox_id=bbox_id(caption_id),
            raw=(90, 320, 880, 380),
            page_width=1000,
            page_height=1400,
        ),
        text="Figure 4: Left, middle, and right results.",
        provenance=document.provenance.model_copy(deep=True),
        metadata={"target_element_id": panels[-1].element_id},
    )
    document.elements = [*panels, caption]
    document.pages = [
        page.model_copy(
            update={
                "element_ids": [
                    panel.element_id for panel in panels
                ] + [caption_id],
                "reading_order": [
                    panel.element_id for panel in panels
                ] + [caption_id],
            }
        )
    ]
    document.sections = []
    document.relations = []

    relations = RelationBuilder(document).build_caption_relations()

    assert {
        relation.target_id
        for relation in relations
        if relation.source_id == caption_id
    } == {panel.element_id for panel in panels}


def test_no_pairwise_near_or_semantic_relations(parsed_document) -> None:
    relation_values = {relation.relation_type.value for relation in parsed_document.relations}
    assert "near" not in relation_values
    assert "semantic_similar" not in relation_values


def test_every_relation_has_audit_fields(parsed_document) -> None:
    for relation in parsed_document.relations:
        assert relation.created_by
        assert 0 <= relation.confidence <= 1
        assert relation.status
        assert relation.evidence
