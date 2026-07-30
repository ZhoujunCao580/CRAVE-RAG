from __future__ import annotations

import json
from pathlib import Path

from softdoc.floating_sections import (
    FloatingContentSectionResolver,
    SectionResolutionStatus,
)
from softdoc.ids import relation_id
from softdoc.models import (
    Document,
    Element,
    ElementType,
    Page,
    Provenance,
    Relation,
    RelationEvidence,
    RelationSource,
    RelationStatus,
    RelationType,
    Section,
)
from softdoc.serialization import write_document


def _provenance(locator: str) -> Provenance:
    return Provenance(
        provenance_id=f"prov:{locator}",
        adapter="test",
        source_path=Path("floating.json"),
        source_locator=locator,
    )


def _element(
    element_id: str,
    page_index: int,
    reading_order: int,
    element_type: ElementType,
    section_id: str,
    text: str,
) -> Element:
    return Element(
        element_id=element_id,
        document_id="doc:floating",
        page_id=f"page:{page_index}",
        page_number=page_index + 1,
        element_type=element_type,
        reading_order=reading_order,
        section_id=section_id,
        section_path=[section_id],
        text=text,
        provenance=_provenance(element_id),
    )


def _relation(
    source_id: str,
    target_id: str,
    relation_type: RelationType,
    *,
    status: RelationStatus = RelationStatus.CONFIRMED,
    confidence: float = 1.0,
    created_by: RelationSource = RelationSource.DETERMINISTIC_RULE,
    metadata: dict | None = None,
) -> Relation:
    return Relation(
        relation_id=relation_id(
            relation_type.value,
            source_id,
            target_id,
            status.value,
            created_by.value,
        ),
        source_id=source_id,
        target_id=target_id,
        relation_type=relation_type,
        confidence=confidence,
        status=status,
        created_by=created_by,
        evidence=[
            RelationEvidence(
                rule="test_relation",
                description="Test evidence.",
                source_ids=[source_id, target_id],
            )
        ],
        metadata=metadata or {},
    )


def _document(
    *,
    target_type: ElementType = ElementType.FIGURE,
    with_caption: bool = False,
    with_footnote: bool = False,
) -> Document:
    section_a = "section:a"
    section_b = "section:b"
    heading_a = _element(
        "heading:a", 0, 0, ElementType.HEADING, section_a, "1 Earlier section"
    )
    reference_a = _element(
        "paragraph:a",
        0,
        1,
        ElementType.PARAGRAPH,
        section_a,
        "See Figure 7.",
    )
    continuation_source = _element(
        "source:a",
        0,
        2,
        target_type,
        section_a,
        "Source floating content",
    )
    heading_b = _element(
        "heading:b", 1, 0, ElementType.HEADING, section_b, "2 Later section"
    )
    target = _element(
        "target:float",
        1,
        1,
        target_type,
        section_b,
        "Floating target",
    )
    elements = [
        heading_a,
        reference_a,
        continuation_source,
        heading_b,
        target,
    ]
    if with_caption:
        elements.append(
            _element(
                "caption:float",
                1,
                2,
                ElementType.CAPTION,
                section_b,
                "Figure 7: Results",
            )
        )
    if with_footnote:
        elements.append(
            _element(
                "footnote:float",
                1,
                3,
                ElementType.FOOTNOTE,
                section_b,
                "1 Table note",
            )
        )

    page_zero_elements = [
        item for item in elements if item.page_id == "page:0"
    ]
    page_one_elements = [
        item for item in elements if item.page_id == "page:1"
    ]
    pages = [
        Page(
            page_id="page:0",
            document_id="doc:floating",
            page_index=0,
            page_number=1,
            width=1000,
            height=1000,
            element_ids=[item.element_id for item in page_zero_elements],
            reading_order=[item.element_id for item in page_zero_elements],
            provenance=_provenance("page:0"),
        ),
        Page(
            page_id="page:1",
            document_id="doc:floating",
            page_index=1,
            page_number=2,
            width=1000,
            height=1000,
            element_ids=[item.element_id for item in page_one_elements],
            reading_order=[item.element_id for item in page_one_elements],
            provenance=_provenance("page:1"),
        ),
    ]
    sections = [
        Section(
            section_id=section_a,
            document_id="doc:floating",
            title="1 Earlier section",
            level=1,
            heading_element_id=heading_a.element_id,
            section_path=["1 Earlier section"],
            page_ids=["page:0"],
            element_ids=[
                heading_a.element_id,
                reference_a.element_id,
                continuation_source.element_id,
            ],
            provenance=heading_a.provenance,
        ),
        Section(
            section_id=section_b,
            document_id="doc:floating",
            title="2 Later section",
            level=1,
            heading_element_id=heading_b.element_id,
            section_path=["2 Later section"],
            page_ids=["page:1"],
            element_ids=[item.element_id for item in page_one_elements],
            provenance=heading_b.provenance,
        ),
    ]
    return Document(
        document_id="doc:floating",
        source_path=Path("floating.pdf"),
        pages=pages,
        sections=sections,
        elements=elements,
        relations=[],
        provenance=_provenance("document"),
    )


def test_cross_page_figure_uses_unique_preceding_reference_and_writes_debug(
    tmp_path: Path,
) -> None:
    document = _document()
    document.relations = [
        _relation(
            "paragraph:a",
            "target:float",
            RelationType.REFERS_TO,
            created_by=RelationSource.EXPLICIT_REFERENCE,
        )
    ]

    decisions = FloatingContentSectionResolver(document).resolve()
    target = next(
        item for item in document.elements if item.element_id == "target:float"
    )
    decision = decisions[0]

    assert target.section_id == "section:a"
    assert decision.original_section_id == "section:b"
    assert decision.resolved_section_id == "section:a"
    assert decision.status == SectionResolutionStatus.CONFIRMED
    assert decision.created_by == RelationSource.EXPLICIT_REFERENCE
    assert decision.evidence_relation_ids == [
        document.relations[0].relation_id
    ]

    output_dir = tmp_path / "output"
    write_document(document, output_dir, render_overlays=False)
    payload = json.loads(
        (
            output_dir / "debug" / "section_resolution_decisions.json"
        ).read_text(encoding="utf-8")
    )
    assert payload[0]["element_id"] == "target:float"


def test_explicit_reference_can_resolve_algorithm_target() -> None:
    document = _document(target_type=ElementType.ALGORITHM)
    document.relations = [
        _relation(
            "paragraph:a",
            "target:float",
            RelationType.REFERS_TO,
            created_by=RelationSource.EXPLICIT_REFERENCE,
        )
    ]

    FloatingContentSectionResolver(document).resolve()

    target = next(
        item for item in document.elements
        if item.element_id == "target:float"
    )
    assert target.section_id == "section:a"


def test_following_same_page_reference_can_resolve_boundary_table() -> None:
    document = _document(target_type=ElementType.TABLE)
    target = next(
        item for item in document.elements
        if item.element_id == "target:float"
    )
    heading_b = next(
        item for item in document.elements
        if item.element_id == "heading:b"
    )
    target.reading_order = 0
    target.section_id = "section:a"
    target.section_path = ["section:a"]
    heading_b.reading_order = 1
    reference = _element(
        "paragraph:following",
        1,
        2,
        ElementType.PARAGRAPH,
        "section:b",
        "Table 1 reports the result.",
    )
    document.elements.append(reference)
    ordered_ids = [
        target.element_id,
        heading_b.element_id,
        reference.element_id,
    ]
    document.pages[1] = document.pages[1].model_copy(
        update={
            "element_ids": ordered_ids,
            "reading_order": ordered_ids,
        }
    )
    document.relations = [
        _relation(
            reference.element_id,
            target.element_id,
            RelationType.REFERS_TO,
            created_by=RelationSource.EXPLICIT_REFERENCE,
            metadata={
                "reference_kind": "table",
                "target_label_source_id": "caption:table",
            },
        )
    ]

    FloatingContentSectionResolver(document).resolve()

    assert target.section_id == "section:b"


def test_following_reference_does_not_override_own_local_heading() -> None:
    document = _document(target_type=ElementType.TABLE)
    target = next(
        item for item in document.elements
        if item.element_id == "target:float"
    )
    reference = _element(
        "paragraph:later",
        1,
        2,
        ElementType.PARAGRAPH,
        "section:a",
        "Table 1 is reused later.",
    )
    document.elements.append(reference)
    document.pages[1].element_ids.append(reference.element_id)
    document.pages[1].reading_order.append(reference.element_id)
    document.relations = [
        _relation(
            reference.element_id,
            target.element_id,
            RelationType.REFERS_TO,
            created_by=RelationSource.EXPLICIT_REFERENCE,
            metadata={
                "reference_kind": "table",
                "target_label_source_id": "caption:table",
            },
        )
    ]

    decisions = FloatingContentSectionResolver(document).resolve()

    assert target.section_id == "section:b"
    assert decisions == []


def test_caption_follows_figure_final_section() -> None:
    document = _document(with_caption=True)
    document.relations = [
        _relation(
            "paragraph:a",
            "target:float",
            RelationType.REFERS_TO,
            created_by=RelationSource.EXPLICIT_REFERENCE,
        ),
        _relation(
            "caption:float",
            "target:float",
            RelationType.CAPTION_OF,
            created_by=RelationSource.PARSER,
        ),
    ]

    decisions = FloatingContentSectionResolver(document).resolve()
    elements = {item.element_id: item for item in document.elements}

    assert elements["target:float"].section_id == "section:a"
    assert elements["caption:float"].section_id == "section:a"
    assert any(
        item.rule == "caption_inherits_target_final_section"
        for item in decisions
    )


def test_footnote_follows_table_final_section() -> None:
    document = _document(
        target_type=ElementType.TABLE,
        with_footnote=True,
    )
    document.relations = [
        _relation(
            "paragraph:a",
            "target:float",
            RelationType.REFERS_TO,
            created_by=RelationSource.EXPLICIT_REFERENCE,
        ),
        _relation(
            "footnote:float",
            "target:float",
            RelationType.FOOTNOTE_OF,
            confidence=0.75,
        ),
    ]

    decisions = FloatingContentSectionResolver(document).resolve()
    elements = {item.element_id: item for item in document.elements}

    assert elements["target:float"].section_id == "section:a"
    assert elements["footnote:float"].section_id == "section:a"
    assert any(
        item.rule == "footnote_inherits_target_final_section"
        for item in decisions
    )


def test_listing_anchor_propagates_back_across_candidate_code_chain() -> None:
    document = _document(
        target_type=ElementType.CODE,
        with_caption=True,
    )
    source = next(
        item for item in document.elements
        if item.element_id == "source:a"
    )
    source.section_id = "section:b"
    source.section_path = ["section:b"]
    caption = next(
        item for item in document.elements
        if item.element_id == "caption:float"
    )
    caption.text = "Listing 1: Full prompt"
    document.relations = [
        _relation(
            "paragraph:a",
            "target:float",
            RelationType.REFERS_TO,
            created_by=RelationSource.EXPLICIT_REFERENCE,
            metadata={
                "reference_kind": "listing",
                "reference_number": "1",
                "target_label_source_id": caption.element_id,
            },
        ),
        _relation(
            source.element_id,
            "target:float",
            RelationType.CONTINUED_ON,
            status=RelationStatus.CANDIDATE,
            confidence=0.80,
            created_by=RelationSource.LAYOUT_HEURISTIC,
        ),
        _relation(
            caption.element_id,
            "target:float",
            RelationType.CAPTION_OF,
            created_by=RelationSource.PARSER,
        ),
    ]

    decisions = FloatingContentSectionResolver(document).resolve()
    elements = {item.element_id: item for item in document.elements}

    assert elements["target:float"].section_id == "section:a"
    assert elements[source.element_id].section_id == "section:a"
    assert elements[caption.element_id].section_id == "section:a"
    assert any(
        item.rule == "listing_chain_inherits_explicit_anchor"
        for item in decisions
    )


def test_multi_panel_caption_propagates_anchored_section() -> None:
    document = _document(
        target_type=ElementType.CHART,
        with_caption=True,
    )
    sibling = _element(
        "target:panel",
        1,
        3,
        ElementType.CHART,
        "section:b",
        "Second panel",
    )
    document.elements.append(sibling)
    document.pages[1].element_ids.append(sibling.element_id)
    document.pages[1].reading_order.append(sibling.element_id)
    document.relations = [
        _relation(
            "paragraph:a",
            "target:float",
            RelationType.REFERS_TO,
            created_by=RelationSource.EXPLICIT_REFERENCE,
        ),
        _relation(
            "caption:float",
            "target:float",
            RelationType.CAPTION_OF,
            created_by=RelationSource.PARSER,
        ),
        _relation(
            "caption:float",
            sibling.element_id,
            RelationType.CAPTION_OF,
            confidence=0.96,
            created_by=RelationSource.LAYOUT_HEURISTIC,
        ),
    ]

    decisions = FloatingContentSectionResolver(document).resolve()
    elements = {item.element_id: item for item in document.elements}

    assert elements["target:float"].section_id == "section:a"
    assert elements[sibling.element_id].section_id == "section:a"
    assert elements["caption:float"].section_id == "section:a"
    assert any(
        item.rule
        == "caption_group_inherits_anchored_target_section"
        for item in decisions
    )


def test_caption_can_anchor_table_to_matching_next_page_section() -> None:
    document = _document(
        target_type=ElementType.TABLE,
        with_caption=True,
    )
    caption = next(
        item for item in document.elements
        if item.element_id == "caption:float"
    )
    caption.text = (
        "DC CHARACTERISTICS (Over Operating Conditions) "
        "All parameter values apply."
    )
    heading = _element(
        "heading:dc",
        2,
        0,
        ElementType.HEADING,
        "section:dc",
        "DC CHARACTERISTICS (Over Operating Conditions)",
    )
    document.elements.append(heading)
    document.pages.append(
        Page(
            page_id="page:2",
            document_id=document.document_id,
            page_index=2,
            page_number=3,
            width=1000,
            height=1000,
            element_ids=[heading.element_id],
            reading_order=[heading.element_id],
            provenance=_provenance("page:2"),
        )
    )
    document.sections.append(
        Section(
            section_id="section:dc",
            document_id=document.document_id,
            title="DC CHARACTERISTICS (Over Operating Conditions)",
            level=1,
            heading_element_id=heading.element_id,
            section_path=[
                "DC CHARACTERISTICS (Over Operating Conditions)"
            ],
            page_ids=[heading.page_id],
            element_ids=[heading.element_id],
            provenance=heading.provenance,
        )
    )
    document.relations = [
        _relation(
            caption.element_id,
            "target:float",
            RelationType.CAPTION_OF,
            created_by=RelationSource.PARSER,
        )
    ]

    decisions = FloatingContentSectionResolver(document).resolve()
    elements = {item.element_id: item for item in document.elements}

    assert elements["target:float"].section_id == "section:dc"
    assert elements[caption.element_id].section_id == "section:dc"
    assert any(
        item.rule == "caption_matches_adjacent_section_title"
        for item in decisions
    )


def test_confirmed_continued_table_inherits_source_section() -> None:
    document = _document(target_type=ElementType.TABLE)
    document.relations = [
        _relation(
            "source:a",
            "target:float",
            RelationType.CONTINUED_ON,
            status=RelationStatus.CONFIRMED,
            confidence=0.92,
        )
    ]

    decisions = FloatingContentSectionResolver(document).resolve()
    target = next(
        item for item in document.elements if item.element_id == "target:float"
    )

    assert target.section_id == "section:a"
    assert decisions[0].rule == (
        "confirmed_continued_on_inherits_source_section"
    )


def test_candidate_continued_table_records_candidate_without_migration() -> None:
    document = _document(target_type=ElementType.TABLE)
    document.relations = [
        _relation(
            "source:a",
            "target:float",
            RelationType.CONTINUED_ON,
            status=RelationStatus.CANDIDATE,
            confidence=0.81,
            created_by=RelationSource.LAYOUT_HEURISTIC,
        )
    ]

    decisions = FloatingContentSectionResolver(document).resolve()
    target = next(
        item for item in document.elements if item.element_id == "target:float"
    )

    assert target.section_id == "section:b"
    assert target.metadata["candidate_section_id"] == "section:a"
    assert decisions[0].status == SectionResolutionStatus.CANDIDATE
    assert decisions[0].resolved_section_id == "section:b"


def test_equal_strength_references_from_multiple_sections_are_ambiguous() -> None:
    document = _document()
    competing = _element(
        "paragraph:b",
        0,
        3,
        ElementType.PARAGRAPH,
        "section:b",
        "Figure 7 is also discussed here.",
    )
    document.elements.append(competing)
    document.pages[0].element_ids.append(competing.element_id)
    document.pages[0].reading_order.append(competing.element_id)
    document.relations = [
        _relation(
            "paragraph:a",
            "target:float",
            RelationType.REFERS_TO,
            created_by=RelationSource.EXPLICIT_REFERENCE,
        ),
        _relation(
            "paragraph:b",
            "target:float",
            RelationType.REFERS_TO,
            created_by=RelationSource.EXPLICIT_REFERENCE,
        ),
    ]

    decisions = FloatingContentSectionResolver(document).resolve()
    target = next(
        item for item in document.elements if item.element_id == "target:float"
    )

    assert target.section_id == "section:b"
    assert decisions[0].status == SectionResolutionStatus.AMBIGUOUS
    assert decisions[0].metadata["candidate_section_ids"] == [
        "section:a",
        "section:b",
    ]


def test_page_adjacency_without_relation_does_not_change_section() -> None:
    document = _document()

    decisions = FloatingContentSectionResolver(document).resolve()
    target = next(
        item for item in document.elements if item.element_id == "target:float"
    )

    assert target.section_id == "section:b"
    assert decisions == []


def test_final_belongs_to_section_matches_every_element_section_id() -> None:
    document = _document(with_caption=True)
    document.relations = [
        _relation(
            "paragraph:a",
            "target:float",
            RelationType.REFERS_TO,
            created_by=RelationSource.EXPLICIT_REFERENCE,
        ),
        _relation(
            "caption:float",
            "target:float",
            RelationType.CAPTION_OF,
            created_by=RelationSource.PARSER,
        ),
    ]

    FloatingContentSectionResolver(document).resolve()
    membership = {
        relation.source_id: relation.target_id
        for relation in document.relations
        if relation.relation_type == RelationType.BELONGS_TO_SECTION
    }

    assert membership == {
        element.element_id: element.section_id
        for element in document.elements
        if element.section_id
    }
    target_membership = next(
        relation
        for relation in document.relations
        if relation.relation_type == RelationType.BELONGS_TO_SECTION
        and relation.source_id == "target:float"
    )
    assert target_membership.evidence[0].rule == (
        "resolved_floating_content_membership"
    )
