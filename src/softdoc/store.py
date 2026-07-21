"""In-memory indexed document access and referential validation."""

from __future__ import annotations

from typing import Any

from softdoc.models import Document, Element, Page, Relation, RelationType, Section


class DocumentValidationError(ValueError):
    pass


class DocumentStore:
    def __init__(self, document: Document):
        self._document = document
        self._pages = _unique_index(document.pages, "page_id")
        self._elements = _unique_index(document.elements, "element_id")
        self._sections = _unique_index(document.sections, "section_id")
        self._relations = _unique_index(document.relations, "relation_id")
        self._objects: dict[str, Any] = {
            document.document_id: document,
            **self._pages,
            **self._elements,
            **self._sections,
        }

    def get_document(self) -> Document:
        return self._document

    def get_page(self, page_id: str) -> Page:
        return self._pages[page_id]

    def get_element(self, element_id: str) -> Element:
        return self._elements[element_id]

    def get_section(self, section_id: str) -> Section:
        return self._sections[section_id]

    def get_relations_from(
        self,
        element_id: str,
        relation_type: RelationType | None = None,
    ) -> list[Relation]:
        return [
            relation
            for relation in self._document.relations
            if relation.source_id == element_id
            and (relation_type is None or relation.relation_type == relation_type)
        ]

    def get_relations_to(
        self,
        element_id: str,
        relation_type: RelationType | None = None,
    ) -> list[Relation]:
        return [
            relation
            for relation in self._document.relations
            if relation.target_id == element_id
            and (relation_type is None or relation.relation_type == relation_type)
        ]

    def follow_relation(self, element_id: str, relation_type: RelationType) -> list[Any]:
        return [
            self._objects[relation.target_id]
            for relation in self.get_relations_from(element_id, relation_type)
            if relation.target_id in self._objects
        ]

    def validate_references(self, *, raise_on_error: bool = False) -> list[str]:
        errors: list[str] = []
        for relation in self._document.relations:
            if relation.source_id not in self._objects:
                errors.append(f"Relation {relation.relation_id} has missing source {relation.source_id}")
            if relation.target_id not in self._objects:
                errors.append(f"Relation {relation.relation_id} has missing target {relation.target_id}")
        for page in self._document.pages:
            for element_id in page.element_ids:
                if element_id not in self._elements:
                    errors.append(f"Page {page.page_id} references missing element {element_id}")
                elif self._elements[element_id].page_id != page.page_id:
                    errors.append(f"Element {element_id} points to a different page")
        for element in self._document.elements:
            if element.page_id not in self._pages:
                errors.append(f"Element {element.element_id} references missing page {element.page_id}")
            if element.section_id and element.section_id not in self._sections:
                errors.append(f"Element {element.element_id} references missing section {element.section_id}")
        for section in self._document.sections:
            if section.heading_element_id not in self._elements:
                errors.append(f"Section {section.section_id} references missing heading {section.heading_element_id}")
            if section.parent_section_id and section.parent_section_id not in self._sections:
                errors.append(f"Section {section.section_id} references missing parent {section.parent_section_id}")
            for page_id in section.page_ids:
                if page_id not in self._pages:
                    errors.append(f"Section {section.section_id} references missing page {page_id}")
            for element_id in section.element_ids:
                if element_id not in self._elements:
                    errors.append(f"Section {section.section_id} references missing element {element_id}")
        if errors and raise_on_error:
            raise DocumentValidationError("\n".join(errors))
        return errors


def _unique_index(items: list[Any], id_field: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        item_id = str(getattr(item, id_field))
        if item_id in result:
            raise DocumentValidationError(f"Duplicate ID: {item_id}")
        result[item_id] = item
    return result
