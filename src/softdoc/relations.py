"""Deterministic relation construction, separate from parser adapters and models."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from softdoc.ids import relation_id
from softdoc.models import (
    Document,
    Element,
    ElementType,
    Relation,
    RelationEvidence,
    RelationSource,
    RelationStatus,
    RelationType,
)


_REFERENCE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("figure", re.compile(r"\b(?:Figure|Fig\.)\s*([0-9]+[A-Za-z]?)\b", re.IGNORECASE)),
    ("table", re.compile(r"\bTable\s*([0-9]+[A-Za-z]?)\b", re.IGNORECASE)),
    ("section", re.compile(r"\bSection\s*([0-9]+(?:\.[0-9]+)*)\b", re.IGNORECASE)),
    ("figure", re.compile(r"图\s*([0-9]+[A-Za-z]?)")),
    ("table", re.compile(r"表\s*([0-9]+[A-Za-z]?)")),
    ("section", re.compile(r"第\s*([0-9]+(?:\.[0-9]+)*)\s*节")),
)

_LABEL_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "figure": (
        re.compile(r"^\s*(?:Figure|Fig\.)\s*([0-9]+[A-Za-z]?)\b", re.IGNORECASE),
        re.compile(r"^\s*图\s*([0-9]+[A-Za-z]?)"),
    ),
    "table": (
        re.compile(r"^\s*Table\s*([0-9]+[A-Za-z]?)\b", re.IGNORECASE),
        re.compile(r"^\s*表\s*([0-9]+[A-Za-z]?)"),
    ),
    "section": (
        re.compile(r"^\s*([0-9]+(?:\.[0-9]+)*)\b"),
        re.compile(r"^\s*第\s*([0-9]+(?:\.[0-9]+)*)\s*节"),
    ),
}

_CODE_LIKE_TYPES = frozenset({ElementType.CODE, ElementType.ALGORITHM})


class RelationBuilder:
    """Build only deterministic relations and bounded cross-page candidates."""

    def __init__(self, document: Document, *, continuation_page_window: int = 2):
        self.document = document
        self.continuation_page_window = max(1, min(int(continuation_page_window), 2))
        self.elements = {element.element_id: element for element in document.elements}
        self.pages = {page.page_id: page for page in document.pages}
        self.sections = {section.section_id: section for section in document.sections}

    def build_all(self) -> list[Relation]:
        generated: list[Relation] = []
        generated.extend(self.build_containment_relations())
        generated.extend(self.build_page_order_relations())
        generated.extend(self.build_reading_order_relations())
        generated.extend(self.build_section_membership_relations())
        caption_relations = self.build_caption_relations()
        footnote_relations = self.build_footnote_relations()
        generated.extend(caption_relations)
        generated.extend(footnote_relations)
        generated.extend(self.build_explicit_reference_relations(caption_relations))
        generated.extend(self.build_cross_page_continuation_candidates())
        by_id = {relation.relation_id: relation for relation in self.document.relations}
        by_id.update({relation.relation_id: relation for relation in generated})
        self.document.relations = list(by_id.values())
        return self.document.relations

    def build_containment_relations(self) -> list[Relation]:
        relations: list[Relation] = []
        for page in self.document.pages:
            relations.append(
                self._relation(
                    self.document.document_id,
                    page.page_id,
                    RelationType.CONTAINS,
                    confidence=1.0,
                    created_by=RelationSource.PARSER,
                    rule="document_contains_page",
                    description="The parsed page belongs to the document.",
                )
            )
            for element_id in page.element_ids:
                relations.append(
                    self._relation(
                        page.page_id,
                        element_id,
                        RelationType.CONTAINS,
                        confidence=1.0,
                        created_by=RelationSource.PARSER,
                        rule="page_contains_element",
                        description="The parser placed the element on this page.",
                    )
                )
        return relations

    def build_page_order_relations(self) -> list[Relation]:
        pages = sorted(self.document.pages, key=lambda page: page.page_index)
        return [
            self._relation(
                source.page_id,
                target.page_id,
                RelationType.NEXT_PAGE,
                confidence=1.0,
                created_by=RelationSource.DETERMINISTIC_RULE,
                rule="consecutive_page_index",
                description="Pages are consecutive in document order.",
                data={"source_page_index": source.page_index, "target_page_index": target.page_index},
            )
            for source, target in zip(pages, pages[1:])
        ]

    def build_reading_order_relations(self) -> list[Relation]:
        relations: list[Relation] = []
        for page in self.document.pages:
            for source_id, target_id in zip(page.reading_order, page.reading_order[1:]):
                relations.append(
                    self._relation(
                        source_id,
                        target_id,
                        RelationType.NEXT_IN_READING_ORDER,
                        confidence=1.0,
                        created_by=RelationSource.PARSER,
                        rule="parser_reading_order",
                        description="Elements are consecutive in parser reading order.",
                        data={"page_id": page.page_id},
                    )
                )
        return relations

    def build_section_membership_relations(self) -> list[Relation]:
        relations: list[Relation] = []
        for element in self.document.elements:
            if element.section_id:
                relations.append(
                    self._relation(
                        element.element_id,
                        element.section_id,
                        RelationType.BELONGS_TO_SECTION,
                        confidence=1.0,
                        created_by=RelationSource.DETERMINISTIC_RULE,
                        rule="heading_stack_membership",
                        description="The active heading stack assigned the element to this section.",
                        data={"section_path": element.section_path or []},
                    )
                )
        return relations

    def build_caption_relations(self) -> list[Relation]:
        return self._build_function_relations(ElementType.CAPTION, RelationType.CAPTION_OF)

    def build_footnote_relations(self) -> list[Relation]:
        return self._build_function_relations(ElementType.FOOTNOTE, RelationType.FOOTNOTE_OF)

    def _build_function_relations(
        self,
        source_type: ElementType,
        relation_type: RelationType,
    ) -> list[Relation]:
        relations: list[Relation] = []
        for source in (element for element in self.document.elements if element.element_type == source_type):
            target_id = str(source.metadata.get("target_element_id") or "")
            target = self.elements.get(target_id)
            created_by = RelationSource.PARSER
            confidence = 1.0
            rule = "parser_declared_function_target"
            if (
                target is None
                and source_type == ElementType.FOOTNOTE
                and source.metadata.get("mineru_type") == "page_footnote"
            ):
                continue
            if target is None:
                target = self._nearest_function_target(source)
                created_by = RelationSource.LAYOUT_HEURISTIC
                confidence = 0.95 if target and target.page_id == source.page_id else 0.78
                rule = "bounded_nearest_compatible_element"
            if target is None:
                continue
            relations.append(
                self._relation(
                    source.element_id,
                    target.element_id,
                    relation_type,
                    confidence=confidence,
                    created_by=created_by,
                    rule=rule,
                    description=f"{source_type.value} is associated with a compatible document element.",
                    data={
                        "source_page_id": source.page_id,
                        "target_page_id": target.page_id,
                        "cross_page": source.page_id != target.page_id,
                    },
                )
            )
        return relations

    def _nearest_function_target(self, source: Element) -> Element | None:
        compatible_types = _compatible_target_types(source)
        source_page_index = self.pages[source.page_id].page_index
        candidates = [
            element
            for element in self.document.elements
            if element.element_type in compatible_types
            and abs(self.pages[element.page_id].page_index - source_page_index) <= 1
        ]
        if not candidates:
            return None
        source_position = source_page_index * 1_000_000 + source.reading_order

        def candidate_key(element: Element) -> tuple[int, int, int]:
            page_index = self.pages[element.page_id].page_index
            position = page_index * 1_000_000 + element.reading_order
            same_page_penalty = 0 if page_index == source_page_index else 1
            preceding_penalty = 0 if position <= source_position else 1
            return same_page_penalty, preceding_penalty, abs(position - source_position)

        return min(candidates, key=candidate_key)

    def build_explicit_reference_relations(
        self,
        caption_relations: Iterable[Relation] | None = None,
    ) -> list[Relation]:
        targets = self._reference_targets(caption_relations or self.build_caption_relations())
        relations: list[Relation] = []
        searchable_types = {ElementType.PARAGRAPH, ElementType.LIST, ElementType.FOOTNOTE}
        for source in self.document.elements:
            if source.element_type not in searchable_types or not source.text:
                continue
            for reference_kind, pattern in _REFERENCE_PATTERNS:
                for match in pattern.finditer(source.text):
                    number = _normalize_reference_number(match.group(1))
                    target_id = targets.get((reference_kind, number))
                    if not target_id or target_id == source.element_id:
                        continue
                    relations.append(
                        self._relation(
                            source.element_id,
                            target_id,
                            RelationType.REFERS_TO,
                            confidence=1.0,
                            created_by=RelationSource.EXPLICIT_REFERENCE,
                            rule="explicit_numbered_reference",
                            description=f"Text explicitly names {match.group(0)}.",
                            data={
                                "matched_text": match.group(0),
                                "reference_kind": reference_kind,
                                "reference_number": number,
                                "source_page_id": source.page_id,
                                "target_page_id": self._object_page_id(target_id),
                            },
                        )
                    )
        return _deduplicate(relations)

    def _reference_targets(self, caption_relations: Iterable[Relation]) -> dict[tuple[str, str], str]:
        targets: dict[tuple[str, str], str] = {}
        for element in self.document.elements:
            if element.reference_label:
                kind = _element_reference_kind(element)
                number = _extract_label_number(kind, element.reference_label) if kind else None
                if kind and number:
                    targets[(kind, number)] = element.element_id
        for relation in caption_relations:
            caption = self.elements.get(relation.source_id)
            target = self.elements.get(relation.target_id)
            if not caption or not target or not caption.text:
                continue
            kind = _element_reference_kind(target)
            number = _extract_label_number(kind, caption.text) if kind else None
            if kind and number:
                targets[(kind, number)] = target.element_id
        for section in self.document.sections:
            number = _extract_label_number("section", section.title)
            if number:
                targets[("section", number)] = section.section_id
        return targets

    def build_cross_page_continuation_candidates(self) -> list[Relation]:
        relations: list[Relation] = []
        pages = sorted(self.document.pages, key=lambda page: page.page_index)
        page_elements = {
            page.page_id: sorted(
                (self.elements[element_id] for element_id in page.reading_order),
                key=lambda element: element.reading_order,
            )
            for page in pages
        }
        for source_offset, source_page in enumerate(pages):
            for distance in range(1, self.continuation_page_window + 1):
                target_offset = source_offset + distance
                if target_offset >= len(pages):
                    break
                target_page = pages[target_offset]
                relations.extend(
                    self._paragraph_continuation_candidate(
                        source_page.page_id,
                        target_page.page_id,
                        page_elements,
                        distance,
                    )
                )
                relations.extend(
                    self._table_continuation_candidate(
                        source_page.page_id,
                        target_page.page_id,
                        page_elements,
                        distance,
                    )
                )
                relations.extend(
                    self._code_continuation_candidate(
                        source_page.page_id,
                        target_page.page_id,
                        page_elements,
                        distance,
                    )
                )
        return relations

    def _paragraph_continuation_candidate(
        self,
        source_page_id: str,
        target_page_id: str,
        page_elements: dict[str, list[Element]],
        page_distance: int,
    ) -> list[Relation]:
        source_paragraphs = [
            element for element in page_elements[source_page_id] if element.element_type == ElementType.PARAGRAPH
        ]
        target_paragraphs = [
            element for element in page_elements[target_page_id] if element.element_type == ElementType.PARAGRAPH
        ]
        if not source_paragraphs or not target_paragraphs:
            return []
        source = source_paragraphs[-1]
        target = target_paragraphs[0]
        heading_before_target = any(
            element.element_type == ElementType.HEADING and element.reading_order < target.reading_order
            for element in page_elements[target_page_id]
        )
        missing_terminal = not bool(re.search(r"[.!?。！？;；:]\s*$", (source.text or "").strip()))
        same_section = bool(source.section_id and source.section_id == target.section_id)
        same_column = source.column_index == target.column_index if None not in (source.column_index, target.column_index) else None
        style_match = _style_signature(source) and _style_signature(source) == _style_signature(target)
        score = 0.10
        score += 0.35 if missing_terminal else 0.0
        score += 0.15 if not heading_before_target else 0.0
        score += 0.15 if same_section else 0.0
        score += 0.10 if same_column is True else 0.0
        score += 0.10 if style_match else 0.0
        score += 0.05 if page_distance == 1 else 0.0
        if score < 0.5:
            return []
        data = {
            "content_type": "paragraph",
            "page_distance": page_distance,
            "missing_terminal_punctuation": missing_terminal,
            "heading_before_target": heading_before_target,
            "same_section": same_section,
            "same_column": same_column,
            "style_match": bool(style_match),
        }
        return [
            self._relation(
                source.element_id,
                target.element_id,
                RelationType.CONTINUED_ON,
                confidence=round(min(score, 0.9), 3),
                status=RelationStatus.CANDIDATE,
                created_by=RelationSource.LAYOUT_HEURISTIC,
                rule="bounded_cross_page_paragraph_continuation",
                description="Last paragraph and first paragraph on nearby pages may be continuous.",
                data=data,
            )
        ]

    def _table_continuation_candidate(
        self,
        source_page_id: str,
        target_page_id: str,
        page_elements: dict[str, list[Element]],
        page_distance: int,
    ) -> list[Relation]:
        source_tables = [element for element in page_elements[source_page_id] if element.element_type == ElementType.TABLE]
        target_tables = [element for element in page_elements[target_page_id] if element.element_type == ElementType.TABLE]
        if not source_tables or not target_tables:
            return []
        source, target = source_tables[-1], target_tables[0]
        source_label = _table_label_for_element(source, self.document.elements)
        target_label = _table_label_for_element(target, self.document.elements)
        same_label = bool(source_label and source_label == target_label)
        continuation_word = bool(
            re.search(r"\bcontinued\b|续", " ".join(filter(None, [source.text, target.text])), re.IGNORECASE)
        )
        source_columns, target_columns = _table_column_count(source), _table_column_count(target)
        columns_match = bool(source_columns and source_columns == target_columns)
        bbox_match = _bbox_width_and_position_match(source, target)
        score = 0.15
        score += 0.30 if continuation_word else 0.0
        score += 0.25 if same_label else 0.0
        score += 0.15 if columns_match else 0.0
        score += 0.10 if bbox_match else 0.0
        score += 0.05 if page_distance == 1 else 0.0
        if score < 0.45:
            return []
        data = {
            "content_type": "table",
            "page_distance": page_distance,
            "same_table_label": same_label,
            "continuation_marker": continuation_word,
            "column_count_match": columns_match,
            "source_column_count": source_columns,
            "target_column_count": target_columns,
            "bbox_width_and_position_match": bbox_match,
        }
        return [
            self._relation(
                source.element_id,
                target.element_id,
                RelationType.CONTINUED_ON,
                confidence=round(min(score, 0.9), 3),
                status=RelationStatus.CANDIDATE,
                created_by=RelationSource.LAYOUT_HEURISTIC,
                rule="bounded_cross_page_table_continuation",
                description="Tables on nearby pages share deterministic continuation signals.",
                data=data,
            )
        ]

    def _code_continuation_candidate(
        self,
        source_page_id: str,
        target_page_id: str,
        page_elements: dict[str, list[Element]],
        page_distance: int,
    ) -> list[Relation]:
        # Code continuations are intentionally limited to adjacent pages.
        # Looking two pages ahead is useful for prose, but too permissive for
        # independent listings that happen to share a visual style.
        if page_distance != 1:
            return []
        source_blocks = [
            element
            for element in page_elements[source_page_id]
            if element.element_type in _CODE_LIKE_TYPES
        ]
        target_blocks = [
            element
            for element in page_elements[target_page_id]
            if element.element_type in _CODE_LIKE_TYPES
        ]
        if not source_blocks or not target_blocks:
            return []

        source, target = source_blocks[-1], target_blocks[0]
        source_at_page_end = bool(
            source.bbox and source.bbox.normalized[3] >= 0.80
        )
        target_at_page_start = bool(
            target.bbox and target.bbox.normalized[1] <= 0.20
        )
        bbox_match = _bbox_width_and_position_match(source, target)
        same_section = bool(
            source.section_id and source.section_id == target.section_id
        )
        heading_before_target = any(
            element.element_type == ElementType.HEADING
            and element.reading_order < target.reading_order
            for element in page_elements[target_page_id]
        )
        source_caption = _caption_for_element(
            source.element_id,
            self.document.elements,
        )
        target_caption = _caption_for_element(
            target.element_id,
            self.document.elements,
        )
        caption_text = " ".join(
            caption.text or ""
            for caption in (source_caption, target_caption)
            if caption is not None
        )
        continuation_marker = bool(
            re.search(r"\bcontinued\b|续", caption_text, re.IGNORECASE)
        )
        source_lacks_block_terminator = _code_lacks_block_terminator(
            source.text or ""
        )
        target_starts_new_construct = _code_starts_new_construct(
            target.text or ""
        )

        # A caption attached to the source usually marks the end of that
        # listing. Only an explicit continuation marker may override it.
        if source_caption is not None and not continuation_marker:
            return []
        boundary_layout_match = (
            source_at_page_end and target_at_page_start and bbox_match
        )
        if not boundary_layout_match and not continuation_marker:
            return []
        if (
            target_starts_new_construct
            and not source_lacks_block_terminator
            and not continuation_marker
            and target_caption is None
        ):
            return []

        score = 0.10
        score += 0.15  # adjacent pages
        score += 0.15 if source_at_page_end else 0.0
        score += 0.15 if target_at_page_start else 0.0
        score += 0.15 if bbox_match else 0.0
        score += 0.10 if source_lacks_block_terminator else 0.0
        score += 0.05 if same_section else 0.0
        score += 0.05 if not heading_before_target else 0.0
        score += 0.05 if target_caption is not None else 0.0
        score += 0.15 if continuation_marker else 0.0
        score -= 0.10 if target_starts_new_construct else 0.0
        if score < 0.60:
            return []

        data = {
            "content_family": "code_like",
            "page_distance": page_distance,
            "source_element_type": source.element_type.value,
            "target_element_type": target.element_type.value,
            "compatible_type_pair": True,
            "source_at_page_end": source_at_page_end,
            "target_at_page_start": target_at_page_start,
            "bbox_width_and_position_match": bbox_match,
            "source_lacks_block_terminator": source_lacks_block_terminator,
            "target_starts_new_construct": target_starts_new_construct,
            "heading_before_target": heading_before_target,
            "same_section": same_section,
            "source_has_caption": source_caption is not None,
            "target_has_caption": target_caption is not None,
            "caption_text": caption_text or None,
            "continuation_marker": continuation_marker,
        }
        return [
            self._relation(
                source.element_id,
                target.element_id,
                RelationType.CONTINUED_ON,
                confidence=round(min(score, 0.9), 3),
                status=RelationStatus.CANDIDATE,
                created_by=RelationSource.LAYOUT_HEURISTIC,
                rule="bounded_cross_page_code_continuation",
                description=(
                    "Adjacent code or algorithm blocks share page-boundary "
                    "layout and textual continuation signals."
                ),
                data=data,
            )
        ]

    def _object_page_id(self, object_id: str) -> str | None:
        element = self.elements.get(object_id)
        if element:
            return element.page_id
        section = self.sections.get(object_id)
        return section.page_ids[0] if section and section.page_ids else None

    def _relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        *,
        confidence: float,
        created_by: RelationSource,
        rule: str,
        description: str,
        status: RelationStatus = RelationStatus.CONFIRMED,
        data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
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
                    rule=rule,
                    description=description,
                    source_ids=[source_id, target_id],
                    data=data or {},
                )
            ],
            metadata=metadata or {},
        )


def _compatible_target_types(source: Element) -> set[ElementType]:
    text = (source.text or "").strip()
    if re.match(r"^(?:Table|表)\s*\d", text, re.IGNORECASE):
        return {ElementType.TABLE}
    if re.match(r"^(?:Figure|Fig\.|图)\s*\d", text, re.IGNORECASE):
        return {
            ElementType.FIGURE,
            ElementType.CHART,
            ElementType.CODE,
            ElementType.ALGORITHM,
        }
    return {
        ElementType.FIGURE,
        ElementType.CHART,
        ElementType.CODE,
        ElementType.ALGORITHM,
        ElementType.TABLE,
    }


def _element_reference_kind(element: Element) -> str | None:
    if element.element_type in {
        ElementType.FIGURE,
        ElementType.CHART,
        ElementType.CODE,
        ElementType.ALGORITHM,
    }:
        return "figure"
    if element.element_type == ElementType.TABLE:
        return "table"
    return None


def _extract_label_number(kind: str, text: str) -> str | None:
    for pattern in _LABEL_PATTERNS.get(kind, ()):
        match = pattern.search(text)
        if match:
            return _normalize_reference_number(match.group(1))
    return None


def _normalize_reference_number(number: str) -> str:
    return number.strip().lower()


def _style_signature(element: Element) -> tuple[Any, ...] | None:
    style = element.metadata.get("style")
    if not isinstance(style, dict):
        return None
    signature = (style.get("font"), style.get("font_size"), style.get("weight"))
    return signature if any(value is not None for value in signature) else None


def _caption_for_element(
    element_id: str,
    elements: Iterable[Element],
) -> Element | None:
    return next(
        (
            element
            for element in elements
            if element.element_type == ElementType.CAPTION
            and element.metadata.get("target_element_id") == element_id
        ),
        None,
    )


def _code_lacks_block_terminator(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    return not bool(re.search(r"(?:[;:{}\[\]()]|```)\s*$", stripped))


def _code_starts_new_construct(text: str) -> bool:
    stripped = text.lstrip()
    return bool(
        re.match(
            r"(?:```|#\s|//\s|def\s+|class\s+|function\s+|"
            r"procedure\s+|algorithm\s+|import\s+|from\s+\S+\s+import\s+)",
            stripped,
            re.IGNORECASE,
        )
    )


def _table_column_count(element: Element) -> int | None:
    declared = element.metadata.get("column_count")
    if isinstance(declared, int) and declared > 0:
        return declared
    html = element.html or ""
    first_row = re.search(r"<tr\b[^>]*>(.*?)</tr>", html, re.IGNORECASE | re.DOTALL)
    if not first_row:
        return None
    count = len(re.findall(r"<t[hd]\b", first_row.group(1), re.IGNORECASE))
    return count or None


def _bbox_width_and_position_match(source: Element, target: Element) -> bool:
    if not source.bbox or not target.bbox:
        return False
    source_x1, _, source_x2, _ = source.bbox.normalized
    target_x1, _, target_x2, _ = target.bbox.normalized
    width_difference = abs((source_x2 - source_x1) - (target_x2 - target_x1))
    center_difference = abs(((source_x1 + source_x2) / 2) - ((target_x1 + target_x2) / 2))
    return width_difference <= 0.1 and center_difference <= 0.1


def _table_label_for_element(table: Element, elements: Iterable[Element]) -> str | None:
    if table.reference_label:
        return _extract_label_number("table", table.reference_label)
    for element in elements:
        if element.element_type != ElementType.CAPTION:
            continue
        if element.metadata.get("target_element_id") == table.element_id and element.text:
            return _extract_label_number("table", element.text)
    return None


def _deduplicate(relations: Iterable[Relation]) -> list[Relation]:
    return list({relation.relation_id: relation for relation in relations}.values())
