"""Deterministic relation construction, separate from parser adapters and models."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from softdoc.ids import relation_id, stable_digest
from softdoc.labels import (
    LabelRegistry,
    ReferenceTarget,
    build_label_registry,
    extract_label_number as _extract_label_number,
    normalize_reference_number as _normalize_reference_number,
    reference_kind_from_label as _reference_kind_from_label,
    reference_number_parts as _reference_number_parts,
)
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
from softdoc.table_fragments import FRAGMENT_METADATA_KEY, RULE_ID as TABLE_FRAGMENT_RULE_ID


_SUBREFERENCE = r"[A-Za-z](?:\s*[-\u2012\u2013\u2014]\s*[A-Za-z])?"
_REFERENCE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "figure",
        re.compile(
            rf"(?<!\w)(?:Figure|Fig\.?)\s*(?P<base>[0-9]+)"
            rf"(?:\s*(?:\(\s*(?P<paren_sub>{_SUBREFERENCE})\s*\)"
            rf"|(?P<suffix_sub>{_SUBREFERENCE})))?(?![A-Za-z0-9_])",
            re.IGNORECASE,
        ),
    ),
    (
        "table",
        re.compile(
            rf"(?<!\w)Table\s*(?P<base>[0-9]+)"
            rf"(?:\s*(?:\(\s*(?P<paren_sub>{_SUBREFERENCE})\s*\)"
            rf"|(?P<suffix_sub>{_SUBREFERENCE})))?(?![A-Za-z0-9_])",
            re.IGNORECASE,
        ),
    ),
    (
        "section",
        re.compile(
            r"(?<!\w)Section\s*(?P<base>[0-9]+(?:\.[0-9]+)*)"
            r"(?=$|[^\d.]|\.(?!\d))",
            re.IGNORECASE,
        ),
    ),
    (
        "listing",
        re.compile(
            r"(?<!\w)Listings?\s*(?P<base>[0-9]+)"
            r"(?![A-Za-z0-9_])",
            re.IGNORECASE,
        ),
    ),
    (
        "figure",
        re.compile(
            rf"图\s*(?P<base>[0-9]+)"
            rf"(?:\s*(?:\(\s*(?P<paren_sub>{_SUBREFERENCE})\s*\)"
            rf"|(?P<suffix_sub>{_SUBREFERENCE})))?(?![A-Za-z0-9_])",
            re.IGNORECASE,
        ),
    ),
    (
        "table",
        re.compile(
            rf"表\s*(?P<base>[0-9]+)"
            rf"(?:\s*(?:\(\s*(?P<paren_sub>{_SUBREFERENCE})\s*\)"
            rf"|(?P<suffix_sub>{_SUBREFERENCE})))?(?![A-Za-z0-9_])",
            re.IGNORECASE,
        ),
    ),
    (
        "section",
        re.compile(r"第\s*(?P<base>[0-9]+(?:\.[0-9]+)*)\s*节"),
    ),
)

_CODE_LIKE_TYPES = frozenset({ElementType.CODE, ElementType.ALGORITHM})


class FootnoteRelationValidator:
    """Validate deterministic footnote targets without deleting source elements.

    A parser-declared target is useful evidence, but is not sufficient on its
    own.  Strong relations combine the declaration or an explicit marker with
    page, layout, prefix, and reading-distance signals.  Weak bindings remain
    inspectable as candidates.
    """

    def __init__(self, document: Document) -> None:
        self.document = document
        self.elements = {
            element.element_id: element for element in document.elements
        }

    def build_relations(self) -> list[Relation]:
        relations: list[Relation] = []
        for footnote in self.document.elements:
            if (
                footnote.element_type != ElementType.FOOTNOTE
                or footnote.metadata.get("excluded_from_relations") is True
            ):
                continue
            declared_target_id = str(
                footnote.metadata.get("target_element_id") or ""
            )
            target = self.elements.get(declared_target_id)
            parser_binding = target is not None
            marker = _footnote_prefix_marker(footnote.text or "")
            marker_binding = False
            if target is None and marker is not None:
                target = self._marker_target(footnote, marker)
                marker_binding = target is not None
            if target is None:
                footnote.metadata["footnote_anchor_status"] = "unresolved"
                continue
            footnote.metadata["footnote_anchor_status"] = "resolved"

            signals = self._signals(
                footnote,
                target,
                parser_binding=parser_binding,
                marker_binding=marker_binding,
                marker=marker,
            )
            confidence = round(min(float(signals["score"]), 0.98), 3)
            status = (
                RelationStatus.CONFIRMED
                if confidence >= 0.75
                else RelationStatus.CANDIDATE
            )
            rule = (
                "parser_footnote_target_validated"
                if parser_binding
                else "explicit_marker_footnote_target"
            )
            created_by = RelationSource.DETERMINISTIC_RULE
            relations.append(
                Relation(
                    relation_id=relation_id(
                        RelationType.FOOTNOTE_OF.value,
                        footnote.element_id,
                        target.element_id,
                        status.value,
                        created_by.value,
                    ),
                    source_id=footnote.element_id,
                    target_id=target.element_id,
                    relation_type=RelationType.FOOTNOTE_OF,
                    confidence=confidence,
                    status=status,
                    created_by=created_by,
                    evidence=[
                        RelationEvidence(
                            rule=rule,
                            description=(
                                "Footnote target validated with parser, text, "
                                "page, and layout signals."
                            ),
                            source_ids=[
                                footnote.element_id,
                                target.element_id,
                            ],
                            data=signals,
                        )
                    ],
                    metadata={
                        "parser_binding": parser_binding,
                        "marker_binding": marker_binding,
                        "footnote_marker": marker,
                        "validation_threshold": 0.75,
                    },
                )
            )
        relations.extend(self._markerless_continuation_relations(relations))
        relations.extend(self._shared_visual_group_relations(relations))
        return _deduplicate(relations)

    def _shared_visual_group_relations(
        self,
        anchored_relations: list[Relation],
    ) -> list[Relation]:
        """Expand a visually shared Note/Source line to every group member.

        A binary ``footnote_of`` edge remains the storage primitive.  Shared
        notes are therefore represented by one edge per visual target, all
        carrying the same deterministic ``visual_group_id``.  Expansion is
        deliberately limited to same-page Figure/Chart rows whose combined
        width is covered by the note.
        """

        shared_relations: list[Relation] = []
        primary_by_source: dict[str, Relation] = {}
        for relation in anchored_relations:
            if relation.relation_type != RelationType.FOOTNOTE_OF:
                continue
            primary_by_source.setdefault(relation.source_id, relation)

        for footnote_id, primary in primary_by_source.items():
            footnote = self.elements.get(footnote_id)
            target = self.elements.get(primary.target_id)
            if (
                footnote is None
                or target is None
                or footnote.bbox is None
                or target.bbox is None
                or footnote.page_id != target.page_id
                or target.element_type
                not in {ElementType.FIGURE, ElementType.CHART}
                or not _looks_like_shared_visual_note(footnote.text or "")
            ):
                continue

            members = self._shared_visual_members(footnote, target)
            if len(members) < 2:
                continue
            union_x1 = min(item.bbox.normalized[0] for item in members if item.bbox)
            union_x2 = max(item.bbox.normalized[2] for item in members if item.bbox)
            note_x1, _, note_x2, _ = footnote.bbox.normalized
            union_width = union_x2 - union_x1
            covered_width = max(
                0.0,
                min(note_x2, union_x2) - max(note_x1, union_x1),
            )
            group_coverage = (
                covered_width / union_width if union_width > 0 else 0.0
            )
            if group_coverage < 0.80:
                continue

            group_id = str(
                target.metadata.get("visual_group_id")
                or f"visual-group:{stable_digest(footnote.element_id, length=12)}"
            )
            member_ids = [member.element_id for member in members]
            footnote.metadata["visual_group_id"] = group_id
            footnote.metadata["visual_group_member_ids"] = member_ids
            footnote.metadata["shared_visual_note"] = True
            for member in members:
                member.metadata["visual_group_id"] = group_id

            shared_evidence = RelationEvidence(
                rule="shared_visual_group_footnote",
                description=(
                    "A Note/Source line immediately below a same-row visual "
                    "group spans the combined group width."
                ),
                source_ids=[footnote.element_id, *member_ids],
                data={
                    "visual_group_id": group_id,
                    "visual_group_member_ids": member_ids,
                    "visual_group_member_count": len(members),
                    "normalized_group_coverage": round(group_coverage, 4),
                    "same_page": True,
                    "note_prefix_signal": True,
                    "maximum_vertical_gap": round(
                        max(
                            _normalized_vertical_gap(footnote, member)
                            for member in members
                        ),
                        4,
                    ),
                    "minimum_row_overlap": round(
                        min(
                            _normalized_vertical_overlap(target, member)
                            for member in members
                        ),
                        4,
                    ),
                },
            )
            primary.metadata.update(
                {
                    "shared_visual_note": True,
                    "visual_group_id": group_id,
                    "visual_group_member_ids": member_ids,
                }
            )
            primary.evidence.append(shared_evidence)

            for member in members:
                if member.element_id == primary.target_id:
                    continue
                confidence = round(
                    min(primary.confidence, 0.75 + 0.15 * group_coverage),
                    3,
                )
                status = (
                    RelationStatus.CONFIRMED
                    if primary.status == RelationStatus.CONFIRMED
                    and confidence >= 0.75
                    else RelationStatus.CANDIDATE
                )
                shared_relations.append(
                    Relation(
                        relation_id=relation_id(
                            RelationType.FOOTNOTE_OF.value,
                            footnote.element_id,
                            member.element_id,
                            status.value,
                            RelationSource.DETERMINISTIC_RULE.value,
                        ),
                        source_id=footnote.element_id,
                        target_id=member.element_id,
                        relation_type=RelationType.FOOTNOTE_OF,
                        confidence=confidence,
                        status=status,
                        created_by=RelationSource.DETERMINISTIC_RULE,
                        evidence=[shared_evidence.model_copy(deep=True)],
                        metadata={
                            "shared_visual_note": True,
                            "visual_group_id": group_id,
                            "visual_group_member_ids": member_ids,
                            "primary_target_id": primary.target_id,
                        },
                    )
                )
        return shared_relations

    def _shared_visual_members(
        self,
        footnote: Element,
        target: Element,
    ) -> list[Element]:
        declared_group_id = target.metadata.get("visual_group_id")
        declared_members = [
            element
            for element in self.document.elements
            if declared_group_id
            and element.metadata.get("visual_group_id")
            == declared_group_id
            and element.page_id == footnote.page_id
            and element.element_type
            in {ElementType.FIGURE, ElementType.CHART}
            and element.bbox is not None
        ]
        if len(declared_members) >= 2:
            return sorted(
                declared_members,
                key=lambda element: element.bbox.normalized[0],
            )

        members = [
            element
            for element in self.document.elements
            if element.page_id == footnote.page_id
            and element.element_type
            in {ElementType.FIGURE, ElementType.CHART}
            and element.bbox is not None
            and element.bbox.normalized[3]
            <= footnote.bbox.normalized[1] + 0.02
            and _normalized_vertical_gap(footnote, element) <= 0.08
            and _normalized_vertical_overlap(target, element) >= 0.70
            and _normalized_width_ratio(target, element) <= 1.80
            and element.metadata.get("excluded_from_relations") is not True
        ]
        return sorted(
            members,
            key=lambda element: element.bbox.normalized[0],
        )

    def _markerless_continuation_relations(
        self,
        anchored_relations: list[Relation],
    ) -> list[Relation]:
        by_source = {
            relation.source_id: relation
            for relation in anchored_relations
            if relation.relation_type == RelationType.FOOTNOTE_OF
        }
        anchored_footnotes = [
            self.elements[source_id]
            for source_id in by_source
            if source_id in self.elements
        ]
        continued: list[Relation] = []
        for footnote in self.document.elements:
            if (
                footnote.element_type != ElementType.FOOTNOTE
                or footnote.element_id in by_source
                or footnote.metadata.get("excluded_from_relations") is True
                or _footnote_prefix_marker(footnote.text or "") is not None
                or footnote.bbox is None
            ):
                continue
            candidates: list[tuple[float, Element, Relation]] = []
            for anchor in anchored_footnotes:
                if (
                    anchor.page_id != footnote.page_id
                    or anchor.bbox is None
                    or footnote.bbox.normalized[1]
                    < anchor.bbox.normalized[1]
                ):
                    continue
                vertical_gap = _normalized_vertical_gap(footnote, anchor)
                horizontal_overlap = _normalized_horizontal_overlap(
                    footnote, anchor
                )
                if vertical_gap <= 0.035 and horizontal_overlap >= 0.60:
                    candidates.append(
                        (
                            vertical_gap,
                            anchor,
                            by_source[anchor.element_id],
                        )
                    )
            if not candidates:
                continue
            vertical_gap, anchor, anchor_relation = min(
                candidates,
                key=lambda item: (
                    item[0],
                    abs(
                        footnote.reading_order
                        - item[1].reading_order
                    ),
                ),
            )
            confidence = min(anchor_relation.confidence, 0.9)
            continued.append(
                Relation(
                    relation_id=relation_id(
                        RelationType.FOOTNOTE_OF.value,
                        footnote.element_id,
                        anchor_relation.target_id,
                        anchor_relation.status.value,
                        RelationSource.DETERMINISTIC_RULE.value,
                    ),
                    source_id=footnote.element_id,
                    target_id=anchor_relation.target_id,
                    relation_type=RelationType.FOOTNOTE_OF,
                    confidence=confidence,
                    status=anchor_relation.status,
                    created_by=RelationSource.DETERMINISTIC_RULE,
                    evidence=[
                        RelationEvidence(
                            rule="markerless_footnote_line_continuation",
                            description=(
                                "A markerless footnote line is aligned "
                                "directly below an already bound footnote."
                            ),
                            source_ids=[
                                footnote.element_id,
                                anchor.element_id,
                                anchor_relation.target_id,
                            ],
                            data={
                                "anchor_footnote_id": anchor.element_id,
                                "anchor_relation_id": (
                                    anchor_relation.relation_id
                                ),
                                "normalized_vertical_gap": vertical_gap,
                                "horizontal_overlap": (
                                    _normalized_horizontal_overlap(
                                        footnote, anchor
                                    )
                                ),
                            },
                        )
                    ],
                    metadata={
                        "continued_footnote_id": anchor.element_id,
                        "anchor_relation_id": anchor_relation.relation_id,
                    },
                )
            )
        return continued

    def _marker_target(
        self,
        footnote: Element,
        marker: str,
    ) -> Element | None:
        candidates = [
            element
            for element in self.document.elements
            if element.page_id == footnote.page_id
            and element.element_type in {
                ElementType.PARAGRAPH,
                ElementType.LIST,
                ElementType.TABLE,
            }
            and element.reading_order < footnote.reading_order
            and _contains_footnote_marker(
                element.text or element.html or "",
                marker,
            )
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda element: footnote.reading_order - element.reading_order,
        )

    @staticmethod
    def _signals(
        footnote: Element,
        target: Element,
        *,
        parser_binding: bool,
        marker_binding: bool,
        marker: str | None,
    ) -> dict[str, Any]:
        same_page = footnote.page_id == target.page_id
        vertical_gap = (
            _normalized_vertical_gap(footnote, target)
            if same_page
            else None
        )
        reading_distance = (
            abs(footnote.reading_order - target.reading_order)
            if same_page
            else None
        )
        page_bottom = bool(
            footnote.bbox and footnote.bbox.normalized[1] >= 0.70
        )
        prefix_signal = _looks_like_footnote_prefix(footnote.text or "")

        score = 0.0
        score += 0.35 if parser_binding else 0.0
        score += 0.45 if marker_binding else 0.0
        score += 0.15 if same_page else 0.0
        if vertical_gap is not None and vertical_gap <= 0.08:
            score += 0.15
        elif vertical_gap is not None and vertical_gap <= 0.18:
            score += 0.08
        if reading_distance is not None and reading_distance <= 2:
            score += 0.10
        elif reading_distance is not None and reading_distance <= 5:
            score += 0.05
        score += 0.10 if prefix_signal else 0.0
        score += 0.05 if page_bottom else 0.0

        return {
            "score": score,
            "parser_binding": parser_binding,
            "marker_binding": marker_binding,
            "footnote_marker": marker,
            "same_page": same_page,
            "source_page_id": footnote.page_id,
            "target_page_id": target.page_id,
            "normalized_vertical_gap": vertical_gap,
            "reading_order_distance": reading_distance,
            "footnote_prefix_signal": prefix_signal,
            "footnote_in_bottom_region": page_bottom,
        }


class RelationBuilder:
    """Build only deterministic relations and bounded cross-page candidates."""

    def __init__(self, document: Document, *, continuation_page_window: int = 2):
        self.document = document
        self.continuation_page_window = max(1, min(int(continuation_page_window), 2))
        self.elements = {element.element_id: element for element in document.elements}
        self.pages = {page.page_id: page for page in document.pages}
        self.sections = {section.section_id: section for section in document.sections}
        profile = document.metadata.get("document_profile", {})
        self.document_profile = (
            str(profile.get("profile") or "")
            if isinstance(profile, dict)
            else ""
        ).casefold()

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
        # Parser-backed aggregate reconciliation is stronger than layout
        # candidates and is intentionally appended last.
        generated.extend(self.build_confirmed_table_fragment_relations())
        by_id = {
            relation.relation_id: relation
            for relation in self.document.relations
            if relation.created_by == RelationSource.HUMAN
        }
        by_id.update({relation.relation_id: relation for relation in generated})
        self.document.relations = list(by_id.values())
        return self.document.relations

    def build_confirmed_table_fragment_relations(self) -> list[Relation]:
        """Connect adjacent page fragments proven to share aggregate HTML."""

        groups: dict[str, list[Element]] = defaultdict(list)
        for element in self.document.elements:
            metadata = element.metadata.get(FRAGMENT_METADATA_KEY)
            if (
                element.element_type == ElementType.TABLE
                and isinstance(metadata, dict)
                and metadata.get("status") == "confirmed"
                and isinstance(metadata.get("group_id"), str)
            ):
                groups[metadata["group_id"]].append(element)

        relations: list[Relation] = []
        for group_id, fragments in sorted(groups.items()):
            ordered = sorted(
                fragments,
                key=lambda element: int(
                    element.metadata[FRAGMENT_METADATA_KEY]["fragment_index"]
                ),
            )
            for source, target in zip(ordered, ordered[1:]):
                source_metadata = source.metadata[FRAGMENT_METADATA_KEY]
                target_metadata = target.metadata[FRAGMENT_METADATA_KEY]
                relations.append(
                    self._relation(
                        source.element_id,
                        target.element_id,
                        RelationType.CONTINUED_ON,
                        confidence=1.0,
                        status=RelationStatus.CONFIRMED,
                        created_by=RelationSource.DETERMINISTIC_RULE,
                        rule=TABLE_FRAGMENT_RULE_ID,
                        description=(
                            "Page-local table fragments uniquely reconstruct "
                            "the MinerU aggregate table in physical page order."
                        ),
                        data={
                            "group_id": group_id,
                            "source_page_number": source.page_number,
                            "target_page_number": target.page_number,
                            "source_aggregate_row_range": [
                                source_metadata["aggregate_row_start"],
                                source_metadata["aggregate_row_end"],
                            ],
                            "target_aggregate_row_range": [
                                target_metadata["aggregate_row_start"],
                                target_metadata["aggregate_row_end"],
                            ],
                            "aggregate_html_preserved_in_provenance": True,
                        },
                    )
                )
        return relations

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
                resolution = element.metadata.get("section_resolution")
                if isinstance(resolution, dict):
                    confidence = float(resolution.get("confidence", 1.0))
                    created_by_value = resolution.get(
                        "created_by",
                        RelationSource.DETERMINISTIC_RULE.value,
                    )
                    try:
                        created_by = RelationSource(created_by_value)
                    except ValueError:
                        created_by = RelationSource.DETERMINISTIC_RULE
                    rule = "resolved_floating_content_membership"
                    description = (
                        "A high-confidence document relation resolved the "
                        "floating element's final semantic Section."
                    )
                    data = {
                        "section_path": element.section_path or [],
                        "section_resolution_decision_id": resolution.get(
                            "decision_id"
                        ),
                        "original_section_id": resolution.get(
                            "original_section_id"
                        ),
                        "resolved_section_id": resolution.get(
                            "resolved_section_id"
                        ),
                        "evidence_relation_ids": resolution.get(
                            "evidence_relation_ids", []
                        ),
                    }
                else:
                    confidence = 1.0
                    created_by = RelationSource.DETERMINISTIC_RULE
                    rule = "heading_stack_membership"
                    description = (
                        "The active heading stack assigned the element to "
                        "this section."
                    )
                    data = {"section_path": element.section_path or []}
                relations.append(
                    self._relation(
                        element.element_id,
                        element.section_id,
                        RelationType.BELONGS_TO_SECTION,
                        confidence=confidence,
                        created_by=created_by,
                        rule=rule,
                        description=description,
                        data=data,
                    )
                )
        return relations

    def build_caption_relations(self) -> list[Relation]:
        return self._build_function_relations(ElementType.CAPTION, RelationType.CAPTION_OF)

    def build_footnote_relations(self) -> list[Relation]:
        return FootnoteRelationValidator(self.document).build_relations()

    def _build_function_relations(
        self,
        source_type: ElementType,
        relation_type: RelationType,
    ) -> list[Relation]:
        relations: list[Relation] = []
        for source in (element for element in self.document.elements if element.element_type == source_type):
            if source.metadata.get("excluded_from_relations") is True:
                continue
            # Parser output occasionally promotes a repeated running header to
            # a caption.  It is not a semantic caption merely because it was
            # assigned a caption type on one page.
            if (
                source_type == ElementType.CAPTION
                and _is_repeated_page_header_caption(
                    source,
                    self.document.elements,
                )
            ):
                continue
            target_id = str(source.metadata.get("target_element_id") or "")
            target = self.elements.get(target_id)
            if (
                target is not None
                and (
                    target.element_type not in _compatible_target_types(source)
                    or target.metadata.get("excluded_from_relations") is True
                )
            ):
                target = None
            created_by = RelationSource.PARSER
            confidence = 1.0
            rule = "parser_declared_function_target"
            if source_type == ElementType.CAPTION and target is not None:
                repaired_target = self._repair_conflicting_table_caption_target(
                    source,
                    target,
                )
                if repaired_target is not None:
                    target = repaired_target
                    created_by = RelationSource.LAYOUT_HEURISTIC
                    confidence = 0.96
                    rule = "caption_label_table_target_repaired"
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
            if source_type == ElementType.CAPTION:
                for sibling in self._caption_group_targets(source, target):
                    if sibling.element_id == target.element_id:
                        continue
                    relations.append(
                        self._relation(
                            source.element_id,
                            sibling.element_id,
                            relation_type,
                            confidence=0.96,
                            created_by=RelationSource.LAYOUT_HEURISTIC,
                            rule="shared_multi_panel_caption",
                            description=(
                                "A numbered caption spans aligned visual "
                                "panels immediately above it."
                            ),
                            data={
                                "source_page_id": source.page_id,
                                "target_page_id": sibling.page_id,
                                "cross_page": False,
                                "primary_target_id": target.element_id,
                            },
                        )
                    )
        return _deduplicate(relations)

    def _repair_conflicting_table_caption_target(
        self,
        caption: Element,
        target: Element,
    ) -> Element | None:
        """Repair a parser caption binding contradicted by explicit labels.

        A table number in the caption is stronger evidence than a parser link
        when the linked table has its own number in the extracted table title.
        Restricting the repair to the caption's page prevents an ordinary
        cross-page caption from being silently redirected.
        """
        if target.element_type != ElementType.TABLE or not caption.text:
            return None
        caption_label = _table_title_label(caption.text)
        bound_label = _table_intrinsic_label(target)
        if not caption_label or not bound_label or caption_label == bound_label:
            return None

        same_page_tables = [
            element
            for element in self.document.elements
            if element.page_id == caption.page_id
            and element.element_type == ElementType.TABLE
            and _is_relation_endpoint(element)
            and element.element_id != target.element_id
        ]
        matching = [
            element
            for element in same_page_tables
            if _table_intrinsic_label(element) == caption_label
        ]
        if matching:
            return self._nearest_caption_table(caption, matching)

        # Some parsers retain only the title text in a sibling caption.  Once
        # a bound table is known to have the wrong number, the next local
        # table is a bounded and inspectable fallback, not a cross-page guess.
        following = [
            element
            for element in same_page_tables
            if element.reading_order > caption.reading_order
        ]
        return self._nearest_caption_table(caption, following) if following else None

    @staticmethod
    def _nearest_caption_table(
        caption: Element,
        tables: Iterable[Element],
    ) -> Element:
        def key(table: Element) -> tuple[float, int, int]:
            layout_gap = (
                _normalized_vertical_gap(caption, table)
                if caption.bbox is not None and table.bbox is not None
                else 1.0
            )
            return (
                layout_gap,
                abs(table.reading_order - caption.reading_order),
                table.reading_order,
            )

        return min(tables, key=key)

    def _caption_group_targets(
        self,
        caption: Element,
        primary_target: Element,
    ) -> list[Element]:
        declared_members = caption.metadata.get("visual_group_member_ids")
        if isinstance(declared_members, list):
            members = [
                self.elements[member_id]
                for member_id in declared_members
                if isinstance(member_id, str)
                and member_id in self.elements
                and self.elements[member_id].element_type
                in {ElementType.FIGURE, ElementType.CHART}
            ]
            if members:
                return members
        if (
            caption.bbox is None
            or primary_target.bbox is None
            or primary_target.element_type
            not in {ElementType.FIGURE, ElementType.CHART}
            or _reference_kind_from_label(caption.text or "") != "figure"
        ):
            return [primary_target]
        candidates: list[Element] = []
        for element in self.document.elements:
            if (
                element.page_id != caption.page_id
                or element.element_type
                not in {ElementType.FIGURE, ElementType.CHART}
                or element.bbox is None
                or element.metadata.get("excluded_from_relations") is True
            ):
                continue
            vertical_gap = _normalized_vertical_gap(caption, element)
            horizontal_overlap = _normalized_horizontal_overlap(
                caption, element
            )
            row_overlap = _normalized_vertical_overlap(
                primary_target, element
            )
            if (
                vertical_gap <= 0.08
                and horizontal_overlap >= 0.25
                and row_overlap >= 0.65
            ):
                candidates.append(element)
        return candidates or [primary_target]

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
        targets = self._reference_targets(
            caption_relations or self.build_caption_relations()
        )
        relations: list[Relation] = []
        searchable_types = {ElementType.PARAGRAPH, ElementType.LIST, ElementType.FOOTNOTE}
        for source in self.document.elements:
            if (
                source.element_type not in searchable_types
                or not source.text
                or source.metadata.get("excluded_from_relations") is True
            ):
                continue
            for reference_kind, pattern in _REFERENCE_PATTERNS:
                for match in pattern.finditer(source.text):
                    if (
                        reference_kind == "section"
                        and _looks_like_external_section_reference(
                            source.text,
                            match,
                        )
                    ):
                        continue
                    number, base_number, subreference = _reference_number_parts(match)
                    target = targets.resolve(
                        source,
                        reference_kind,
                        number,
                        matched_text=match.group(0),
                    )
                    fell_back_to_main = False
                    if target is None and subreference is not None:
                        target = targets.resolve(
                            source,
                            reference_kind,
                            base_number,
                            matched_text=match.group(0),
                        )
                        fell_back_to_main = target is not None
                    if target is None and reference_kind == "figure":
                        target = self._composite_visual_fallback(
                            source,
                            reference_kind,
                            number,
                        )
                    if target is None or target.target_id == source.element_id:
                        continue
                    relations.append(
                        self._explicit_reference_relation(
                            source=source,
                            target=target,
                            reference_kind=reference_kind,
                            number=number,
                            base_number=base_number,
                            subreference=subreference,
                            fell_back_to_main=fell_back_to_main,
                            matched_text=match.group(0),
                        )
                    )
            for matched_text, number in _additional_listing_references(
                source.text
            ):
                target = targets.resolve(
                    source,
                    "listing",
                    number,
                    matched_text=matched_text,
                )
                if target is None or target.target_id == source.element_id:
                    continue
                relations.append(
                    self._explicit_reference_relation(
                        source=source,
                        target=target,
                        reference_kind="listing",
                        number=number,
                        base_number=number,
                        subreference=None,
                        fell_back_to_main=False,
                        matched_text=matched_text,
                    )
                )
        return _deduplicate(relations)

    def _composite_visual_fallback(
        self,
        source: Element,
        kind: str,
        number: str,
    ) -> ReferenceTarget | None:
        if kind != "figure":
            return None
        visuals = [
            element
            for element in self.document.elements
            if element.page_id == source.page_id
            and element.element_type
            in {ElementType.FIGURE, ElementType.CHART}
            and element.bbox is not None
            and element.bbox.height >= 0.25
            and element.reading_order > source.reading_order
            and _is_relation_endpoint(element)
        ]
        if len(visuals) != 1:
            return None
        visual = visuals[0]
        numbered_caption = next(
            (
                element
                for element in self.document.elements
                if element.element_type == ElementType.CAPTION
                and element.metadata.get("target_element_id")
                == visual.element_id
                and element.text
                and _extract_label_number("figure", element.text)
            ),
            None,
        )
        if numbered_caption is None:
            return None
        page_index = self.pages[visual.page_id].page_index
        section_root = (
            visual.section_path[0] if visual.section_path else None
        )
        label = f"Figure {number}"
        existing_labels = visual.metadata.get("logical_reference_labels", [])
        labels = (
            list(dict.fromkeys(existing_labels))
            if isinstance(existing_labels, list)
            else []
        )
        if label not in labels:
            labels.append(label)
        visual.metadata["logical_reference_labels"] = labels
        visual.metadata["composite_visual_candidate"] = True
        return ReferenceTarget(
            target_id=visual.element_id,
            kind=kind,
            number=number,
            resolution_rule="unique_same_page_composite_visual_fallback",
            label_source_id=numbered_caption.element_id,
            label_text=numbered_caption.text or "",
            priority=50,
            page_index=page_index,
            section_id=visual.section_id,
            section_root=section_root,
        )

    def _explicit_reference_relation(
        self,
        *,
        source: Element,
        target: ReferenceTarget,
        reference_kind: str,
        number: str,
        base_number: str,
        subreference: str | None,
        fell_back_to_main: bool,
        matched_text: str,
    ) -> Relation:
        relation_metadata: dict[str, Any] = {
            "reference_kind": reference_kind,
            "reference_number": number,
            "reference_base_number": base_number,
            "fallback_to_main": fell_back_to_main,
            "target_resolution_rule": target.resolution_rule,
        }
        if target.label_source_id is not None:
            relation_metadata["target_label_source_id"] = (
                target.label_source_id
            )
        if subreference is not None:
            metadata_key = (
                "subfigure_number"
                if reference_kind == "figure"
                else "subtable_number"
            )
            relation_metadata[metadata_key] = subreference
        return self._relation(
            source.element_id,
            target.target_id,
            RelationType.REFERS_TO,
            confidence=1.0,
            created_by=RelationSource.EXPLICIT_REFERENCE,
            rule="explicit_numbered_reference",
            description=f"Text explicitly names {matched_text}.",
            data={
                "matched_text": matched_text,
                "reference_kind": reference_kind,
                "reference_number": number,
                "reference_base_number": base_number,
                "subreference": subreference,
                "fallback_to_main": fell_back_to_main,
                "target_resolution_rule": target.resolution_rule,
                "source_page_id": source.page_id,
                "target_page_id": self._object_page_id(target.target_id),
            },
            metadata=relation_metadata,
        )

    def _reference_targets(
        self,
        caption_relations: Iterable[Relation],
    ) -> LabelRegistry:
        return build_label_registry(
            self.document,
            caption_relations=caption_relations,
        )

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
                    self._list_continuation_candidate(
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
            element
            for element in page_elements[source_page_id]
            if element.element_type == ElementType.PARAGRAPH
            and _is_relation_endpoint(element)
        ]
        target_paragraphs = [
            element
            for element in page_elements[target_page_id]
            if element.element_type == ElementType.PARAGRAPH
            and _is_relation_endpoint(element)
        ]
        if not source_paragraphs or not target_paragraphs:
            return []
        assessments: list[tuple[float, Element, Element, dict[str, Any]]] = []
        for source in source_paragraphs:
            for target in target_paragraphs:
                assessment = self._paragraph_continuation_assessment(
                    source,
                    target,
                    source_page_id=source_page_id,
                    target_page_id=target_page_id,
                    page_elements=page_elements,
                    page_distance=page_distance,
                )
                if assessment is None:
                    continue
                score, data = assessment
                assessments.append((score, source, target, data))
        if not assessments:
            return []
        assessments.sort(
            key=lambda item: (
                item[0],
                item[1].reading_order,
                -item[2].reading_order,
            ),
            reverse=True,
        )
        score, source, target, data = assessments[0]
        competing_scores = [
            candidate[0]
            for candidate in assessments[1:]
            if candidate[1].element_id != source.element_id
            or candidate[2].element_id != target.element_id
        ]
        score_margin = (
            score - max(competing_scores)
            if competing_scores
            else score
        )
        data.update(
            {
                "evaluated_pair_count": (
                    len(source_paragraphs) * len(target_paragraphs)
                ),
                "eligible_pair_count": len(assessments),
                "selected_pair_score_margin": round(score_margin, 3),
            }
        )
        return [
            self._relation(
                source.element_id,
                target.element_id,
                RelationType.CONTINUED_ON,
                confidence=round(min(score, 0.9), 3),
                status=RelationStatus.CANDIDATE,
                created_by=RelationSource.LAYOUT_HEURISTIC,
                rule="bounded_cross_page_paragraph_continuation",
                description=(
                    "A bounded source/target search selected the strongest "
                    "cross-page paragraph seam."
                ),
                data=data,
            )
        ]

    def _paragraph_continuation_assessment(
        self,
        source: Element,
        target: Element,
        *,
        source_page_id: str,
        target_page_id: str,
        page_elements: dict[str, list[Element]],
        page_distance: int,
    ) -> tuple[float, dict[str, Any]] | None:
        source_text = (source.text or "").strip()
        target_text = (target.text or "").strip()
        if not source_text or not target_text:
            return None
        heading_before_target = any(
            element.element_type == ElementType.HEADING and element.reading_order < target.reading_order
            for element in page_elements[target_page_id]
        )
        missing_terminal = not bool(
            re.search(r"[.!?。！？;；:]\s*$", source_text)
        )
        same_section = bool(source.section_id and source.section_id == target.section_id)
        same_column = source.column_index == target.column_index if None not in (source.column_index, target.column_index) else None
        style_match = _style_signature(source) and _style_signature(source) == _style_signature(target)
        source_at_page_end = bool(
            source.bbox and source.bbox.normalized[3] >= 0.72
        )
        target_at_page_start = bool(
            target.bbox and target.bbox.normalized[1] <= 0.28
        )
        boundary_layout_match = source_at_page_end and target_at_page_start
        source_ends_hyphen = bool(
            re.search(r"\w[-\u2010\u2011\u2012\u2013\u2014]\s*$", source_text)
        )
        target_starts_continuation = bool(
            re.match(r"^[a-z0-9,;:)\]}]", target_text)
        )
        target_looks_like_title = _looks_like_title_fragment(target_text)
        source_looks_like_title = _looks_like_title_fragment(source_text)
        source_is_readable = _has_readable_prose(source_text)
        target_is_readable = _has_readable_prose(target_text)
        target_is_page_marker = _looks_like_page_marker(target_text)
        target_is_url = _looks_like_standalone_url(target_text)
        target_is_too_short = len(target_text.strip()) < 8
        target_is_subfigure_labels = _looks_like_subfigure_label_line(
            target_text
        )
        source_is_marginal_footnote = _looks_like_marginal_footnote(
            source,
        )
        source_looks_like_visual_caption = (
            _looks_like_visual_caption_candidate(
                source,
                page_elements[source_page_id],
            )
        )
        checklist_transition = _looks_like_checklist_fragment(
            source_text
        ) or _looks_like_checklist_fragment(target_text)
        narrow_callout_jump = _narrow_callout_column_jump(source, target)
        lexical_seam = _paragraph_lexical_seam(source_text, target_text)
        short_independent_target = _looks_like_short_independent_paragraph(
            source_text,
            target_text,
        )
        slide_page_reset = (
            self.document_profile in {"slides", "brochure"}
            and not source_ends_hyphen
            and not lexical_seam
        )
        visual_bridge = (
            page_distance == 2
            and self._has_visual_bridge_page(
                source_page_id,
                target_page_id,
                page_elements,
            )
        )
        if page_distance == 2 and not visual_bridge:
            return None
        if not source_is_readable or not target_is_readable:
            return None
        if target_is_page_marker or target_is_url:
            return None
        if target_is_too_short and not source_ends_hyphen:
            return None
        if target_is_subfigure_labels:
            return None
        if source_is_marginal_footnote:
            return None
        if not missing_terminal:
            return None
        if source_looks_like_title and not source_ends_hyphen:
            return None
        if source_looks_like_visual_caption and not source_ends_hyphen:
            return None
        if target_looks_like_title and not source_ends_hyphen:
            return None
        if checklist_transition:
            return None
        if short_independent_target:
            return None
        if slide_page_reset:
            return None
        if (
            narrow_callout_jump
            and not source_ends_hyphen
            and not target_starts_continuation
        ):
            return None
        if (
            not boundary_layout_match
            and not source_ends_hyphen
            and not target_starts_continuation
        ):
            return None
        if heading_before_target and not source_ends_hyphen:
            return None
        if len(source_text) < 25 and not source_ends_hyphen:
            return None
        if not (
            boundary_layout_match
            or source_ends_hyphen
            or lexical_seam
        ):
            return None

        score = 0.10
        score += 0.20 if missing_terminal else 0.0
        score += 0.25 if boundary_layout_match else 0.0
        score += 0.15 if source_ends_hyphen else 0.0
        score += 0.15 if lexical_seam else 0.0
        score += 0.10 if same_section else 0.0
        score += 0.05 if same_column is True else 0.0
        score += 0.05 if style_match else 0.0
        score += 0.05 if page_distance == 1 else 0.0
        score += 0.05 if visual_bridge else 0.0
        if score < 0.55:
            return None
        data = {
            "content_type": "paragraph",
            "page_distance": page_distance,
            "missing_terminal_punctuation": missing_terminal,
            "heading_before_target": heading_before_target,
            "same_section": same_section,
            "same_column": same_column,
            "style_match": bool(style_match),
            "source_at_page_end": source_at_page_end,
            "target_at_page_start": target_at_page_start,
            "boundary_layout_match": boundary_layout_match,
            "source_ends_hyphen": source_ends_hyphen,
            "target_starts_continuation": target_starts_continuation,
            "lexical_seam": lexical_seam,
            "source_is_readable": source_is_readable,
            "target_is_readable": target_is_readable,
            "target_is_page_marker": target_is_page_marker,
            "target_is_url": target_is_url,
            "target_is_too_short": target_is_too_short,
            "target_is_subfigure_labels": target_is_subfigure_labels,
            "source_is_marginal_footnote": source_is_marginal_footnote,
            "source_looks_like_title": source_looks_like_title,
            "source_looks_like_visual_caption": (
                source_looks_like_visual_caption
            ),
            "target_looks_like_title": target_looks_like_title,
            "short_independent_target": short_independent_target,
            "slide_page_reset": slide_page_reset,
            "checklist_transition": checklist_transition,
            "narrow_callout_column_jump": narrow_callout_jump,
            "visual_bridge_page": visual_bridge,
        }
        return score, data

    def _list_continuation_candidate(
        self,
        source_page_id: str,
        target_page_id: str,
        page_elements: dict[str, list[Element]],
        page_distance: int,
    ) -> list[Relation]:
        if page_distance != 1:
            return []
        source_candidates = [
            element
            for element in page_elements[source_page_id]
            if _is_list_continuation_endpoint(element)
        ]
        target_candidates = [
            element
            for element in page_elements[target_page_id]
            if _is_list_continuation_endpoint(element)
        ]
        assessments: list[tuple[float, Element, Element, dict[str, Any]]] = []
        for source in source_candidates:
            for target in target_candidates:
                source_text = _element_text(source)
                target_text = _element_text(target)
                if not source_text or not target_text:
                    continue
                same_section = bool(
                    source.section_id
                    and source.section_id == target.section_id
                )
                source_at_page_end = bool(
                    source.bbox and source.bbox.normalized[3] >= 0.72
                )
                target_at_page_start = bool(
                    target.bbox and target.bbox.normalized[1] <= 0.30
                )
                boundary_layout_match = (
                    source_at_page_end and target_at_page_start
                )
                source_trailing_comma = bool(
                    re.search(r"[,;:]\s*$", source_text)
                )
                target_starts_lower = bool(
                    re.match(r"^[a-z]", target_text)
                )
                lexical_seam = _paragraph_lexical_seam(
                    source_text,
                    target_text,
                )
                context_kind = _list_context_kind(source, target)
                logical_page_block = (
                    context_kind == "bibliography"
                    and len(source_candidates) == 1
                    and len(target_candidates) == 1
                )
                if context_kind is None and not (
                    source_trailing_comma and boundary_layout_match
                ):
                    continue
                if not (
                    (boundary_layout_match or logical_page_block)
                    and (
                        source_trailing_comma
                        or target_starts_lower
                        or lexical_seam
                    )
                ):
                    continue
                score = 0.10
                score += 0.25 if boundary_layout_match else 0.0
                score += 0.15 if same_section else 0.0
                score += 0.20 if source_trailing_comma else 0.0
                score += 0.15 if target_starts_lower else 0.0
                score += 0.10 if lexical_seam else 0.0
                score += 0.15 if context_kind == "bibliography" else 0.0
                score += 0.10 if context_kind == "index" else 0.0
                score += 0.15 if logical_page_block else 0.0
                if score < 0.65:
                    continue
                data = {
                    "content_family": "list_like",
                    "page_distance": page_distance,
                    "source_element_type": source.element_type.value,
                    "target_element_type": target.element_type.value,
                    "context_kind": context_kind,
                    "same_section": same_section,
                    "source_at_page_end": source_at_page_end,
                    "target_at_page_start": target_at_page_start,
                    "boundary_layout_match": boundary_layout_match,
                    "logical_page_block": logical_page_block,
                    "source_trailing_comma": source_trailing_comma,
                    "target_starts_lowercase": target_starts_lower,
                    "lexical_seam": lexical_seam,
                }
                assessments.append((score, source, target, data))
        if not assessments:
            return []
        assessments.sort(
            key=lambda item: (
                item[0],
                item[1].reading_order,
                -item[2].reading_order,
            ),
            reverse=True,
        )
        score, source, target, data = assessments[0]
        data["evaluated_pair_count"] = (
            len(source_candidates) * len(target_candidates)
        )
        data["eligible_pair_count"] = len(assessments)
        return [
            self._relation(
                source.element_id,
                target.element_id,
                RelationType.CONTINUED_ON,
                confidence=round(min(score, 0.9), 3),
                status=RelationStatus.CANDIDATE,
                created_by=RelationSource.LAYOUT_HEURISTIC,
                rule="bounded_cross_page_list_continuation",
                description=(
                    "Adjacent list-like blocks share a bounded textual and "
                    "layout continuation seam."
                ),
                data=data,
            )
        ]

    def _has_visual_bridge_page(
        self,
        source_page_id: str,
        target_page_id: str,
        page_elements: dict[str, list[Element]],
    ) -> bool:
        source_index = self.pages[source_page_id].page_index
        target_index = self.pages[target_page_id].page_index
        if target_index - source_index != 2:
            return False
        bridge_page = next(
            (
                page
                for page in self.document.pages
                if page.page_index == source_index + 1
            ),
            None,
        )
        if bridge_page is None:
            return False
        bridge_elements = page_elements[bridge_page.page_id]
        has_visual = any(
            element.element_type in {ElementType.FIGURE, ElementType.CHART}
            for element in bridge_elements
        )
        has_readable_flow = any(
            element.element_type
            in {
                ElementType.HEADING,
                ElementType.PARAGRAPH,
                ElementType.LIST,
                ElementType.TABLE,
                ElementType.CODE,
                ElementType.ALGORITHM,
                ElementType.EQUATION,
            }
            and bool((element.text or element.html or "").strip())
            for element in bridge_elements
        )
        return has_visual and not has_readable_flow

    def _table_continuation_candidate(
        self,
        source_page_id: str,
        target_page_id: str,
        page_elements: dict[str, list[Element]],
        page_distance: int,
    ) -> list[Relation]:
        # Tables are only compared across the physical page boundary.  Looking
        # two pages ahead produced false links between independent tables; an
        # explicit same-label continuation can be handled later by a dedicated
        # document-specific signal if needed.
        if page_distance != 1:
            return []
        source_tables = [
            element
            for element in page_elements[source_page_id]
            if element.element_type == ElementType.TABLE
            and _is_relation_endpoint(element)
        ]
        target_tables = [
            element
            for element in page_elements[target_page_id]
            if element.element_type == ElementType.TABLE
            and _is_relation_endpoint(element)
        ]
        if not source_tables or not target_tables:
            return []
        source, target = source_tables[-1], target_tables[0]
        source_fragment = source.metadata.get(FRAGMENT_METADATA_KEY)
        target_fragment = target.metadata.get(FRAGMENT_METADATA_KEY)
        if (
            isinstance(source_fragment, dict)
            and isinstance(target_fragment, dict)
            and source_fragment.get("status") == "confirmed"
            and source_fragment.get("group_id") == target_fragment.get("group_id")
            and int(target_fragment.get("fragment_index", -1))
            == int(source_fragment.get("fragment_index", -2)) + 1
        ):
            return []
        source_label = _table_label_for_element(source, self.document.elements)
        target_label = _table_label_for_element(target, self.document.elements)
        same_label = bool(source_label and source_label == target_label)
        continuation_word = bool(
            re.search(r"\bcontinued\b|续", " ".join(filter(None, [source.text, target.text])), re.IGNORECASE)
        )
        source_columns, target_columns = _table_column_count(source), _table_column_count(target)
        columns_match = bool(source_columns and source_columns == target_columns)
        bbox_match = _bbox_width_and_position_match(source, target)
        header_similarity = _table_header_similarity(source, target)
        headers_match = header_similarity >= 0.50
        source_at_page_end = bool(
            source.bbox and source.bbox.normalized[3] >= 0.68
        )
        target_at_page_start = bool(
            target.bbox and target.bbox.normalized[1] <= 0.25
        )
        boundary_layout_match = source_at_page_end and target_at_page_start
        visual_only_chain = (
            page_distance == 1
            and bbox_match
            and boundary_layout_match
            and (
                source.content_availability is not None
                and source.content_availability.value
                in {"visual_only", "unavailable"}
                or target.content_availability is not None
                and target.content_availability.value
                in {"visual_only", "unavailable"}
            )
        )
        source_identity = _table_context_title(
            source,
            page_elements[source_page_id],
            self.document.elements,
        )
        target_identity = _table_context_title(
            target,
            page_elements[target_page_id],
            self.document.elements,
        )
        continuation_word = continuation_word or bool(
            re.search(
                r"\bcontinued\b|续",
                " ".join(
                    value
                    for value in (source_identity, target_identity)
                    if value
                ),
                re.IGNORECASE,
            )
        )
        identity_conflict = _table_identity_conflict(
            source_identity,
            target_identity,
        )
        if identity_conflict and not same_label:
            return []
        if page_distance > 1 and not (continuation_word or same_label):
            return []
        strong_structure = (
            continuation_word
            or same_label
            or headers_match
            or visual_only_chain
            or (boundary_layout_match and columns_match and bbox_match)
        )
        if not strong_structure:
            return []

        score = 0.10
        score += 0.10 if page_distance == 1 else 0.0
        score += 0.30 if continuation_word else 0.0
        score += 0.25 if same_label else 0.0
        score += 0.20 if headers_match else 0.0
        score += 0.10 if columns_match else 0.0
        score += 0.10 if bbox_match else 0.0
        score += 0.10 if boundary_layout_match else 0.0
        score += 0.15 if visual_only_chain else 0.0
        if score < 0.50:
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
            "header_token_similarity": round(header_similarity, 3),
            "table_headers_match": headers_match,
            "source_at_page_end": source_at_page_end,
            "target_at_page_start": target_at_page_start,
            "boundary_layout_match": boundary_layout_match,
            "visual_only_chain": visual_only_chain,
            "source_table_identity": source_identity,
            "target_table_identity": target_identity,
            "table_identity_conflict": identity_conflict,
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
            and _is_relation_endpoint(element)
        ]
        target_blocks = [
            element
            for element in page_elements[target_page_id]
            if element.element_type in _CODE_LIKE_TYPES
            and _is_relation_endpoint(element)
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
        comments_only = _comments_only(source.text or "") and _comments_only(
            target.text or ""
        )

        # A caption attached to the source usually marks the end of that
        # listing. Only an explicit continuation marker may override it.
        if source_caption is not None and not continuation_marker:
            return []
        if comments_only and not continuation_marker:
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
            "comments_only": comments_only,
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


def _additional_listing_references(
    text: str,
) -> list[tuple[str, str]]:
    references: list[tuple[str, str]] = []
    group_pattern = re.compile(
        r"(?<!\w)Listings\s*"
        r"(?P<numbers>[0-9]+(?:\s*(?:,|and|&)\s*[0-9]+)+)",
        re.IGNORECASE,
    )
    for match in group_pattern.finditer(text):
        numbers = re.findall(r"[0-9]+", match.group("numbers"))
        for number in numbers[1:]:
            normalized = _normalize_reference_number(number)
            references.append((f"Listing {number}", normalized))
    return references


def _looks_like_external_section_reference(
    text: str,
    match: re.Match[str],
) -> bool:
    """Reject legal/statutory Section mentions without an internal cue."""

    start = max(0, match.start() - 90)
    stop = min(len(text), match.end() + 120)
    context = text[start:stop]
    external_cue = re.search(
        r"\b(?:act|law|statute|regulation|regulations|code|clause|"
        r"sub-?section|ordinance|rule)\b",
        context,
        re.IGNORECASE,
    )
    internal_cue = re.search(
        r"\b(?:this|above|below|following|preceding)\s+section\b",
        context,
        re.IGNORECASE,
    )
    return bool(external_cue and not internal_cue)


def _normalized_vertical_gap(source: Element, target: Element) -> float:
    if source.bbox is None or target.bbox is None:
        return 1.0
    _, source_y1, _, source_y2 = source.bbox.normalized
    _, target_y1, _, target_y2 = target.bbox.normalized
    if source_y1 >= target_y2:
        return source_y1 - target_y2
    if target_y1 >= source_y2:
        return target_y1 - source_y2
    return 0.0


def _normalized_horizontal_overlap(source: Element, target: Element) -> float:
    if source.bbox is None or target.bbox is None:
        return 0.0
    source_x1, _, source_x2, _ = source.bbox.normalized
    target_x1, _, target_x2, _ = target.bbox.normalized
    overlap = max(0.0, min(source_x2, target_x2) - max(source_x1, target_x1))
    smaller_width = min(source_x2 - source_x1, target_x2 - target_x1)
    return overlap / smaller_width if smaller_width > 0 else 0.0


def _normalized_vertical_overlap(source: Element, target: Element) -> float:
    if source.bbox is None or target.bbox is None:
        return 0.0
    _, source_y1, _, source_y2 = source.bbox.normalized
    _, target_y1, _, target_y2 = target.bbox.normalized
    overlap = max(0.0, min(source_y2, target_y2) - max(source_y1, target_y1))
    smaller_height = min(source_y2 - source_y1, target_y2 - target_y1)
    return overlap / smaller_height if smaller_height > 0 else 0.0


def _normalized_width_ratio(source: Element, target: Element) -> float:
    if source.bbox is None or target.bbox is None:
        return float("inf")
    source_width = source.bbox.width
    target_width = target.bbox.width
    smaller_width = min(source_width, target_width)
    larger_width = max(source_width, target_width)
    return (
        larger_width / smaller_width
        if smaller_width > 0
        else float("inf")
    )


def _looks_like_shared_visual_note(text: str) -> bool:
    return bool(
        re.match(
            r"^\s*(?:notes?|sources?)\s*[:：]",
            text,
            re.IGNORECASE,
        )
    )


def _footnote_prefix_marker(text: str) -> str | None:
    match = re.match(
        r"^\s*(?:(?:<sup>\s*)|\(\s*)?"
        r"(?P<marker>\d+|[#*†‡])"
        r"(?:\s*</sup>|\s*\))?"
        r"(?=\s|[).:\-]|[A-Za-z])",
        text,
        re.IGNORECASE,
    )
    return match.group("marker").casefold() if match else None


def _contains_footnote_marker(text: str, marker: str) -> bool:
    escaped = re.escape(marker)
    return bool(
        re.search(
            rf"<sup>\s*{escaped}\s*</sup>|\(\s*{escaped}\s*\)"
            rf"|(?<!\w)[*†‡](?!\w)"
            if marker in {"*", "†", "‡"}
            else (
                rf"<sup>\s*{escaped}\s*</sup>"
                rf"|\(\s*{escaped}\s*\)"
            ),
            text,
            re.IGNORECASE,
        )
    )


def _looks_like_footnote_prefix(text: str) -> bool:
    return bool(
        _footnote_prefix_marker(text)
        or re.match(
            r"^\s*(?:note|notes|source|sources)\s*[:：]",
            text,
            re.IGNORECASE,
        )
    )


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


def _looks_like_title_fragment(text: str) -> bool:
    stripped = re.sub(r"<[^>]+>", "", text).strip()
    words = re.findall(r"[A-Za-z]+", stripped)
    if not words or len(words) > 12 or len(stripped) > 100:
        return False
    title_like_words = sum(
        word.isupper() or word[:1].isupper()
        for word in words
    )
    return title_like_words / len(words) >= 0.80


def _has_readable_prose(text: str) -> bool:
    stripped = re.sub(r"<[^>]+>", "", text).strip()
    if not stripped:
        return False
    visible = [character for character in stripped if not character.isspace()]
    letters = sum(character.isalpha() for character in visible)
    digits = sum(character.isdigit() for character in visible)
    if letters < 4:
        return False
    return (letters + digits) / len(visible) >= 0.35


def _looks_like_page_marker(text: str) -> bool:
    stripped = re.sub(r"<[^>]+>", "", text).strip()
    return bool(
        re.fullmatch(
            r"(?:\d{1,4}|[ivxlcdm]{1,8})",
            stripped,
            re.IGNORECASE,
        )
    )


def _looks_like_standalone_url(text: str) -> bool:
    stripped = re.sub(r"<[^>]+>", "", text).strip()
    return bool(
        re.fullmatch(
            r"(?:https?://|www\.)\S+",
            stripped,
            re.IGNORECASE,
        )
    )


def _looks_like_subfigure_label_line(text: str) -> bool:
    stripped = re.sub(r"<[^>]+>", "", text).strip()
    labels = re.findall(r"\([a-z]\)", stripped, re.IGNORECASE)
    return len(labels) >= 2 and len(stripped) <= 220


def _looks_like_marginal_footnote(element: Element) -> bool:
    text = _element_text(element)
    if not text or len(text) > 140 or element.bbox is None:
        return False
    _, _, _, y2 = element.bbox.normalized
    return y2 >= 0.88 and bool(re.match(r"^\s*\d{1,3}[A-Z]", text))


def _element_text(element: Element) -> str:
    value = element.text or element.html or ""
    return " ".join(_html_text(value).split())


def _paragraph_lexical_seam(source_text: str, target_text: str) -> bool:
    source = source_text.rstrip()
    target = target_text.lstrip()
    source_ends_open = bool(
        re.search(
            r"(?:[,;:\-\u2010-\u2014]|\b(?:a|an|and|as|at|by|for|"
            r"from|in|of|on|or|than|the|to|with))\s*$",
            source,
            re.IGNORECASE,
        )
    )
    target_starts_open = bool(
        re.match(
            r"^(?:[a-z,;:)\]}]|and\b|as\b|but\b|for\b|of\b|"
            r"or\b|that\b|than\b|to\b|which\b|with\b)",
            target,
        )
    )
    return source_ends_open or target_starts_open


def _looks_like_short_independent_paragraph(
    source_text: str,
    target_text: str,
) -> bool:
    target = target_text.strip()
    if len(target) > 220:
        return False
    starts_as_sentence = bool(
        re.match(r"^[A-Z][A-Za-z'-]*(?:\s|$)", target)
    )
    target_is_complete = bool(re.search(r"[.!?。！？]\s*$", target))
    return (
        starts_as_sentence
        and target_is_complete
        and not _paragraph_lexical_seam(source_text, target_text)
    )


def _looks_like_visual_caption_candidate(
    element: Element,
    page_elements: list[Element],
) -> bool:
    text = _element_text(element)
    if not text or len(text) > 180 or element.bbox is None:
        return False
    following_visuals = [
        candidate
        for candidate in page_elements
        if candidate.reading_order > element.reading_order
        and candidate.element_type in {ElementType.FIGURE, ElementType.CHART}
        and candidate.bbox is not None
    ]
    if not following_visuals:
        return False
    visual = min(
        following_visuals,
        key=lambda candidate: candidate.reading_order,
    )
    return (
        visual.reading_order - element.reading_order <= 2
        and _normalized_vertical_gap(element, visual) <= 0.04
        and _normalized_horizontal_overlap(element, visual) >= 0.25
    )


def _is_list_continuation_endpoint(element: Element) -> bool:
    if not _is_relation_endpoint(element):
        return False
    if element.element_type == ElementType.LIST:
        return True
    if element.element_type != ElementType.PARAGRAPH:
        return False
    text = _element_text(element)
    section_context = " ".join(element.section_path or []).casefold()
    return bool(
        "index" in section_context
        and re.match(r"^\s*[A-Z]\s+\S+", text)
    )


def _list_context_kind(
    source: Element,
    target: Element,
) -> str | None:
    context = " ".join(
        [
            *(source.section_path or []),
            *(target.section_path or []),
        ]
    ).casefold()
    if re.search(r"\b(?:references|bibliography)\b", context):
        return "bibliography"
    if re.search(r"\bindex\b", context):
        return "index"
    source_text = _element_text(source)
    target_text = _element_text(target)
    citation_signals = (
        bool(
            re.search(
                r"\b(?:19|20)\d{2}\b.*\b(?:proceedings|journal|"
                r"conference|arxiv|pages?)\b",
                source_text,
                re.IGNORECASE,
            )
        )
        or bool(
            re.search(
                r"^(?:of\s+the\s+)?(?:proceedings|journal|conference)\b",
                target_text,
                re.IGNORECASE,
            )
        )
    )
    return "bibliography" if citation_signals else None


def _looks_like_checklist_fragment(text: str) -> bool:
    if re.search(
        r"\bC\d+\.",
        text.lstrip("\x00\x01\x02\x03\x04\x05\x06\x07\x08"),
        re.IGNORECASE,
    ):
        return True
    return bool(
        re.search(
            r"(?:[✓✔☐☑]|\bC\d+\.)\s*(?:</sup>\s*)?(?:Did|Does|Is|Are|Was|Were)\b",
            text,
            re.IGNORECASE,
        )
    )


def _narrow_callout_column_jump(source: Element, target: Element) -> bool:
    if source.bbox is None or target.bbox is None:
        return False
    source_x1, _, source_x2, _ = source.bbox.normalized
    target_x1, _, target_x2, _ = target.bbox.normalized
    source_width = source_x2 - source_x1
    target_width = target_x2 - target_x1
    source_center = (source_x1 + source_x2) / 2
    target_center = (target_x1 + target_x2) / 2
    return (
        source_width < 0.30
        and target_width < 0.30
        and abs(source_center - target_center) > 0.25
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


def _table_header_similarity(source: Element, target: Element) -> float:
    source_tokens = _table_header_tokens(source.html or "")
    target_tokens = _table_header_tokens(target.html or "")
    if not source_tokens or not target_tokens:
        return 0.0
    return len(source_tokens & target_tokens) / len(source_tokens | target_tokens)


def _table_header_tokens(html: str) -> set[str]:
    rows = re.findall(
        r"<tr\b[^>]*>(.*?)</tr>",
        html,
        re.IGNORECASE | re.DOTALL,
    )[:2]
    if not rows:
        return set()
    text = re.sub(r"<[^>]+>", " ", " ".join(rows))
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9]+", text)
        if len(token) > 1
    }


def _bbox_width_and_position_match(source: Element, target: Element) -> bool:
    if not source.bbox or not target.bbox:
        return False
    source_x1, _, source_x2, _ = source.bbox.normalized
    target_x1, _, target_x2, _ = target.bbox.normalized
    width_difference = abs((source_x2 - source_x1) - (target_x2 - target_x1))
    center_difference = abs(((source_x1 + source_x2) / 2) - ((target_x1 + target_x2) / 2))
    return width_difference <= 0.1 and center_difference <= 0.1


def _table_label_for_element(table: Element, elements: Iterable[Element]) -> str | None:
    intrinsic_label = _table_intrinsic_label(table)
    if intrinsic_label:
        return intrinsic_label
    for element in elements:
        if element.element_type != ElementType.CAPTION:
            continue
        if (
            element.metadata.get("target_element_id") == table.element_id
            and element.text
            and _caption_is_reliable_for_table(element, table, elements)
        ):
            return _table_title_label(element.text)
    return None


def _table_intrinsic_label(table: Element) -> str | None:
    """Return a table label carried by the table, not by a parser binding."""
    for value in (table.reference_label, table.text, _html_text(table.html or "")):
        if value:
            label = _table_title_label(value)
            if label:
                return label
    return None


def _table_title_label(text: str) -> str | None:
    """Read either ``Table 5`` or a form-style ``5. ... Table`` title."""

    explicit = _extract_label_number("table", text)
    if explicit:
        return explicit
    plain = _html_text(text)
    match = re.match(
        r"^\s*(?P<label>\d+(?:\.\d+)*)\s*[.)]?\s+.{0,180}?\bTable\b\s*[:.]?",
        plain,
        re.IGNORECASE,
    )
    return _normalize_reference_number(match.group("label")) if match else None


def _caption_is_reliable_for_table(
    caption: Element,
    table: Element,
    all_elements: Iterable[Element],
) -> bool:
    if _is_repeated_page_header_caption(caption, all_elements):
        return False
    caption_label = _table_title_label(caption.text or "")
    table_label = _table_intrinsic_label(table)
    return not (
        caption_label
        and table_label
        and caption_label != table_label
    )


def _is_repeated_page_header_caption(
    caption: Element,
    all_elements: Iterable[Element],
) -> bool:
    if not caption.text:
        return False
    caption_text = _normalized_document_text(caption.text)
    if not caption_text:
        return False
    header_pages: dict[str, set[str]] = defaultdict(set)
    for element in all_elements:
        if not _is_page_header_element(element) or not element.text:
            continue
        header_text = _normalized_document_text(element.text)
        if header_text:
            header_pages[header_text].add(element.page_id)
    return len(header_pages.get(caption_text, set())) >= 2


def _normalized_document_text(text: str) -> str:
    return " ".join(_html_text(text).casefold().split())


def _html_text(value: str) -> str:
    return re.sub(r"<[^>]+>", " ", value)


def _is_page_header_element(element: Element) -> bool:
    parser_type = str(element.metadata.get("mineru_type") or "").casefold()
    return (
        element.metadata.get("repeated_region") == "page_header"
        or parser_type in {"page_header", "header"}
    )


def _is_relation_endpoint(element: Element) -> bool:
    return not (
        element.metadata.get("excluded_from_relations") is True
        or element.metadata.get("repeated_region")
        in {"page_header", "page_footer"}
        or str(element.metadata.get("mineru_type") or "").casefold()
        in {"page_header", "header", "page_footer", "footer", "page_number"}
        or element.metadata.get("page_decoration_candidate") is True
        or element.metadata.get("duplicate_of")
    )


def _table_context_title(
    table: Element,
    page_elements: Iterable[Element],
    all_elements: Iterable[Element],
) -> str | None:
    captions = [
        element.text or ""
        for element in all_elements
        if element.element_type == ElementType.CAPTION
        and element.metadata.get("target_element_id") == table.element_id
        and element.text
        and _caption_is_reliable_for_table(element, table, all_elements)
    ]
    headings = [
        element
        for element in page_elements
        if element.element_type == ElementType.HEADING
        and element.reading_order <= table.reading_order
        and element.metadata.get("repeated_region")
        not in {"page_header", "page_footer"}
        and element.metadata.get("excluded_from_section_hierarchy") is not True
        and element.text
    ]
    nearest_heading = (
        max(headings, key=lambda item: item.reading_order).text
        if headings
        else None
    )
    context = [
        value for value in [*captions, nearest_heading] if value
    ]
    return " ".join(context) if context else None


def _table_identity_conflict(
    source_title: str | None,
    target_title: str | None,
) -> bool:
    if not source_title or not target_title:
        return False
    identity_patterns = {
        "balance_sheet": r"\bbalance sheet\b",
        "profit_loss": r"\b(?:profit and loss|income statement)\b",
        "cash_flow": r"\b(?:cash flow|statement of cash flows)\b",
        "dc_characteristics": r"\bdc characteristics\b",
        "external_program_memory": r"\bexternal program memory\b",
        "ac_characteristics": r"\bac characteristics\b",
        "customer_wifi": r"\bcustomer\b.*\bwi-?fi\b",
        "employee_wifi": r"\bemployee\b.*\bwi-?fi\b",
    }
    source_identity = next(
        (
            name
            for name, pattern in identity_patterns.items()
            if re.search(pattern, source_title, re.IGNORECASE)
        ),
        None,
    )
    target_identity = next(
        (
            name
            for name, pattern in identity_patterns.items()
            if re.search(pattern, target_title, re.IGNORECASE)
        ),
        None,
    )
    if source_identity and target_identity:
        return source_identity != target_identity
    source_tokens = _identity_tokens(source_title)
    target_tokens = _identity_tokens(target_title)
    if len(source_tokens) >= 2 and len(target_tokens) >= 2:
        overlap = len(source_tokens & target_tokens) / min(
            len(source_tokens), len(target_tokens)
        )
        # Consecutive tables that have descriptive titles but no meaningful
        # title overlap are separate objects even when their geometry matches.
        if overlap < 0.20:
            return True
    return False


def _identity_tokens(text: str) -> set[str]:
    stop = {
        "a",
        "an",
        "and",
        "for",
        "of",
        "on",
        "table",
        "the",
        "to",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if len(token) > 1 and token not in stop
    }


def _comments_only(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(lines) and all(
        line.startswith(("#", "//", "/*", "*")) for line in lines
    )


def _deduplicate(relations: Iterable[Relation]) -> list[Relation]:
    return list({relation.relation_id: relation for relation in relations}.values())
