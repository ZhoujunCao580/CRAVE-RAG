from __future__ import annotations

from pathlib import Path

from softdoc.adapters import MinerUAdapter
from softdoc.models import ElementType, RelationType
from softdoc.parser import DocumentParser


def test_adapter_satisfies_parser_protocol() -> None:
    assert isinstance(MinerUAdapter(), DocumentParser)


def test_stable_ids(mineru_fixture_dir: Path, tmp_path: Path) -> None:
    first = MinerUAdapter().parse(mineru_fixture_dir, tmp_path / "first")
    second = MinerUAdapter().parse(mineru_fixture_dir, tmp_path / "second")
    assert first.document_id == second.document_id
    assert [page.page_id for page in first.pages] == [page.page_id for page in second.pages]
    assert [element.element_id for element in first.elements] == [element.element_id for element in second.elements]
    assert [relation.relation_id for relation in first.relations] == [relation.relation_id for relation in second.relations]


def test_adapter_preserves_structure_and_raw_payload(parsed_document) -> None:
    assert len(parsed_document.pages) == 3
    assert any(element.element_type == ElementType.HEADING for element in parsed_document.elements)
    assert sum(element.element_type == ElementType.PARAGRAPH for element in parsed_document.elements) == 3
    assert any(element.element_type == ElementType.TABLE and element.html for element in parsed_document.elements)
    assert parsed_document.metadata["adapter_warnings"]
    paragraph = next(
        element
        for element in parsed_document.elements
        if element.provenance.raw_payload.get("experimental_field")
    )
    assert paragraph.provenance.raw_payload["experimental_field"].startswith("retained")


def test_caption_footnote_and_reading_order(parsed_document) -> None:
    caption_relations = [
        relation for relation in parsed_document.relations if relation.relation_type == RelationType.CAPTION_OF
    ]
    footnote_relations = [
        relation for relation in parsed_document.relations if relation.relation_type == RelationType.FOOTNOTE_OF
    ]
    assert len(caption_relations) == 3
    assert len(footnote_relations) == 1
    elements = {element.element_id: element for element in parsed_document.elements}
    assert all(elements[relation.source_id].element_type == ElementType.CAPTION for relation in caption_relations)
    assert all(
        elements[relation.target_id].element_type in {ElementType.FIGURE, ElementType.TABLE}
        for relation in caption_relations
    )
    for page in parsed_document.pages:
        assert page.reading_order == page.element_ids
        orders = [
            element.reading_order
            for element in parsed_document.elements
            if element.page_id == page.page_id
        ]
        assert orders == list(range(len(orders)))


def test_section_membership_crosses_pages(parsed_document) -> None:
    assert len(parsed_document.sections) == 1
    section = parsed_document.sections[0]
    assert len(section.page_ids) == 3
    assert all(element.section_id == section.section_id for element in parsed_document.elements)
    assert any(
        relation.relation_type == RelationType.BELONGS_TO_SECTION
        for relation in parsed_document.relations
    )


def test_document_page_and_page_element_relations(parsed_document) -> None:
    contains = [
        relation for relation in parsed_document.relations if relation.relation_type == RelationType.CONTAINS
    ]
    next_pages = [
        relation for relation in parsed_document.relations if relation.relation_type == RelationType.NEXT_PAGE
    ]
    assert sum(relation.source_id == parsed_document.document_id for relation in contains) == 3
    assert len(next_pages) == 2
    assert next_pages[0].source_id == parsed_document.pages[0].page_id
    assert next_pages[0].target_id == parsed_document.pages[1].page_id
