from __future__ import annotations

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
from softdoc.relations import RelationBuilder


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
