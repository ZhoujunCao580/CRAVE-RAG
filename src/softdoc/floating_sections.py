"""Resolve semantic Section ownership for cross-page floating content."""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
import re
from typing import Any

from pydantic import Field

from softdoc.ids import stable_digest
from softdoc.models import (
    Document,
    Element,
    ElementType,
    Relation,
    RelationSource,
    RelationStatus,
    RelationType,
    SoftDocModel,
)
from softdoc.relations import RelationBuilder


class SectionResolutionStatus(str, Enum):
    CONFIRMED = "confirmed"
    CANDIDATE = "candidate"
    AMBIGUOUS = "ambiguous"


class SectionResolutionDecision(SoftDocModel):
    decision_id: str
    element_id: str
    rule: str
    original_section_id: str | None = None
    resolved_section_id: str | None = None
    candidate_section_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    status: SectionResolutionStatus
    created_by: RelationSource
    evidence_relation_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FloatingContentSectionResolver:
    """Correct baseline heading-stack membership using confirmed relations only."""

    _FLOATING_TYPES = frozenset(
        {
            ElementType.FIGURE,
            ElementType.CHART,
            ElementType.TABLE,
            ElementType.CODE,
            ElementType.ALGORITHM,
            ElementType.EQUATION,
            ElementType.CAPTION,
            ElementType.FOOTNOTE,
        }
    )
    _REFERENCE_SOURCE_TYPES = frozenset(
        {ElementType.PARAGRAPH, ElementType.LIST}
    )
    _REFERENCE_TARGET_TYPES = frozenset(
        {
            ElementType.FIGURE,
            ElementType.CHART,
            ElementType.TABLE,
            ElementType.CODE,
            ElementType.ALGORITHM,
            ElementType.EQUATION,
        }
    )

    def __init__(
        self,
        document: Document,
        *,
        maximum_reference_page_distance: int = 2,
    ) -> None:
        self.document = document
        self.maximum_reference_page_distance = max(
            0, int(maximum_reference_page_distance)
        )
        self.elements = {
            element.element_id: element for element in document.elements
        }
        self.sections = {
            section.section_id: section for section in document.sections
        }
        self.page_indexes = {
            page.page_id: page.page_index for page in document.pages
        }
        self.decisions: list[SectionResolutionDecision] = []
        self._locked_by_continuation: set[str] = set()

    def resolve(self) -> list[SectionResolutionDecision]:
        self._restore_baseline_assignments()
        self._resolve_explicit_references()
        self._resolve_reference_anchored_listing_chains()
        self._resolve_confirmed_continuations()
        self._resolve_section_title_anchors()
        self._resolve_caption_groups()
        self._resolve_function_elements()
        self._record_candidate_continuations()
        self._synchronize_sections()
        self._rebuild_membership_relations()
        self.document.metadata["section_resolution_decisions"] = [
            decision.model_dump(mode="json") for decision in self.decisions
        ]
        return self.decisions

    def _restore_baseline_assignments(self) -> None:
        self.decisions = []
        self._locked_by_continuation = set()
        for element in self.document.elements:
            if element.element_type not in self._FLOATING_TYPES:
                continue
            if "base_section_id" not in element.metadata:
                element.metadata["base_section_id"] = element.section_id
                element.metadata["base_section_path"] = element.section_path
            element.section_id = element.metadata.get("base_section_id")
            base_path = element.metadata.get("base_section_path")
            element.section_path = list(base_path) if base_path else None
            for key in (
                "section_resolution",
                "candidate_section_id",
                "section_resolution_candidates",
                "section_resolution_ambiguities",
            ):
                element.metadata.pop(key, None)

    def _resolve_confirmed_continuations(self) -> None:
        relations = sorted(
            (
                relation
                for relation in self.document.relations
                if relation.relation_type == RelationType.CONTINUED_ON
                and relation.status == RelationStatus.CONFIRMED
            ),
            key=lambda relation: self._position_for_id(relation.target_id),
        )
        # A bounded fixpoint also handles a confirmed A -> B -> C chain when
        # externally supplied relations are not already in document order.
        for _ in range(max(1, len(relations))):
            changed = False
            for relation in relations:
                source = self.elements.get(relation.source_id)
                target = self.elements.get(relation.target_id)
                if (
                    source is None
                    or target is None
                    or target.element_type not in self._FLOATING_TYPES
                    or not source.section_id
                    or not target.section_id
                ):
                    continue
                if target.section_id == source.section_id:
                    self._locked_by_continuation.add(target.element_id)
                    continue
                source_anchored = self._has_confirmed_resolution(source)
                target_anchored = self._has_confirmed_resolution(target)
                if target_anchored and not source_anchored:
                    changed_element = source
                    resolved_section_id = target.section_id
                    rule = (
                        "confirmed_continued_on_inherits_target_anchor"
                    )
                    metadata = {"target_element_id": target.element_id}
                else:
                    changed_element = target
                    resolved_section_id = source.section_id
                    rule = (
                        "confirmed_continued_on_inherits_source_section"
                    )
                    metadata = {"source_element_id": source.element_id}
                original = changed_element.section_id
                self._assign_section(
                    changed_element,
                    resolved_section_id,
                )
                decision = self._decision(
                    element=changed_element,
                    rule=rule,
                    original_section_id=original,
                    resolved_section_id=resolved_section_id,
                    confidence=relation.confidence,
                    status=SectionResolutionStatus.CONFIRMED,
                    created_by=relation.created_by,
                    evidence_relations=[relation],
                    metadata=metadata,
                )
                self._record_confirmed(changed_element, decision)
                self._locked_by_continuation.add(
                    changed_element.element_id
                )
                changed = True
            if not changed:
                break

    def _resolve_explicit_references(self) -> None:
        by_target: dict[str, list[tuple[Relation, Element, float]]] = (
            defaultdict(list)
        )
        for relation in self.document.relations:
            if (
                relation.relation_type != RelationType.REFERS_TO
                or relation.status != RelationStatus.CONFIRMED
                or relation.created_by != RelationSource.EXPLICIT_REFERENCE
            ):
                continue
            source = self.elements.get(relation.source_id)
            target = self.elements.get(relation.target_id)
            if (
                source is None
                or target is None
                or source.element_type not in self._REFERENCE_SOURCE_TYPES
                or target.element_type not in self._REFERENCE_TARGET_TYPES
                or not source.section_id
                or target.element_id in self._locked_by_continuation
            ):
                continue
            reference = self._reference_strength(source, target, relation)
            if reference is None:
                continue
            strength, _ = reference
            by_target[target.element_id].append(
                (relation, source, strength)
            )

        for target_id, candidates in by_target.items():
            target = self.elements[target_id]
            by_section: dict[
                str, list[tuple[Relation, Element, float]]
            ] = defaultdict(list)
            for relation, source, strength in candidates:
                by_section[source.section_id].append(
                    (relation, source, strength)
                )
            section_strengths = {
                section_id: max(item[2] for item in items)
                for section_id, items in by_section.items()
            }
            best_strength = max(section_strengths.values())
            best_sections = sorted(
                section_id
                for section_id, strength in section_strengths.items()
                if strength == best_strength
            )
            best_relations = [
                relation
                for section_id in best_sections
                for relation, _, strength in by_section[section_id]
                if strength == best_strength
            ]
            if len(best_sections) > 1:
                decision = self._decision(
                    element=target,
                    rule="equally_strong_explicit_references_are_ambiguous",
                    original_section_id=target.section_id,
                    resolved_section_id=target.section_id,
                    confidence=best_strength,
                    status=SectionResolutionStatus.AMBIGUOUS,
                    created_by=RelationSource.EXPLICIT_REFERENCE,
                    evidence_relations=best_relations,
                    metadata={
                        "candidate_section_ids": best_sections,
                        "section_strengths": section_strengths,
                    },
                )
                target.metadata.setdefault(
                    "section_resolution_ambiguities", []
                ).append(decision.model_dump(mode="json"))
                self.decisions.append(decision)
                continue

            resolved_section_id = best_sections[0]
            if target.section_id == resolved_section_id:
                continue
            original = target.section_id
            self._assign_section(target, resolved_section_id)
            reference_directions = sorted(
                {
                    reference[1]
                    for relation, source, _ in candidates
                    if (
                        reference := self._reference_strength(
                            source,
                            target,
                            relation,
                        )
                    )
                    is not None
                }
            )
            decision_rule = (
                "unique_following_boundary_reference"
                if "following_boundary_float" in reference_directions
                else "unique_preceding_explicit_reference"
            )
            decision = self._decision(
                element=target,
                rule=decision_rule,
                original_section_id=original,
                resolved_section_id=resolved_section_id,
                confidence=best_strength,
                status=SectionResolutionStatus.CONFIRMED,
                created_by=RelationSource.EXPLICIT_REFERENCE,
                evidence_relations=best_relations,
                metadata={
                    "reference_count": len(candidates),
                    "maximum_page_distance": (
                        self.maximum_reference_page_distance
                    ),
                    "reference_directions": reference_directions,
                },
            )
            self._record_confirmed(target, decision)

    def _resolve_function_elements(self) -> None:
        relation_types = {
            RelationType.CAPTION_OF: ElementType.CAPTION,
            RelationType.FOOTNOTE_OF: ElementType.FOOTNOTE,
        }
        relations = sorted(
            (
                relation
                for relation in self.document.relations
                if relation.relation_type in relation_types
                and relation.status == RelationStatus.CONFIRMED
            ),
            key=lambda relation: self._position_for_id(relation.source_id),
        )
        for relation in relations:
            source = self.elements.get(relation.source_id)
            target = self.elements.get(relation.target_id)
            if (
                source is None
                or target is None
                or source.element_type
                != relation_types[relation.relation_type]
                or not target.section_id
                or source.section_id == target.section_id
            ):
                continue
            original = source.section_id
            self._assign_section(source, target.section_id)
            decision = self._decision(
                element=source,
                rule=(
                    "caption_inherits_target_final_section"
                    if source.element_type == ElementType.CAPTION
                    else "footnote_inherits_target_final_section"
                ),
                original_section_id=original,
                resolved_section_id=target.section_id,
                confidence=relation.confidence,
                status=SectionResolutionStatus.CONFIRMED,
                created_by=relation.created_by,
                evidence_relations=[relation],
                metadata={"target_element_id": target.element_id},
            )
            self._record_confirmed(source, decision)

    def _resolve_reference_anchored_listing_chains(self) -> None:
        listing_anchors: dict[str, list[Relation]] = defaultdict(list)
        for relation in self.document.relations:
            if (
                relation.relation_type == RelationType.REFERS_TO
                and relation.status == RelationStatus.CONFIRMED
                and relation.metadata.get("reference_kind") == "listing"
                and relation.target_id in self.elements
            ):
                listing_anchors[relation.target_id].append(relation)
        if not listing_anchors:
            return

        adjacency: dict[
            str, list[tuple[str, Relation]]
        ] = defaultdict(list)
        for relation in self.document.relations:
            if (
                relation.relation_type != RelationType.CONTINUED_ON
                or relation.status != RelationStatus.CANDIDATE
                or relation.confidence < 0.70
            ):
                continue
            source = self.elements.get(relation.source_id)
            target = self.elements.get(relation.target_id)
            if (
                source is None
                or target is None
                or source.element_type
                not in {ElementType.CODE, ElementType.ALGORITHM}
                or target.element_type
                not in {ElementType.CODE, ElementType.ALGORITHM}
            ):
                continue
            adjacency[source.element_id].append(
                (target.element_id, relation)
            )
            adjacency[target.element_id].append(
                (source.element_id, relation)
            )

        for anchor_id, anchor_relations in listing_anchors.items():
            anchor = self.elements[anchor_id]
            if not anchor.section_id:
                continue
            queue: list[str] = [anchor_id]
            paths: dict[str, list[Relation]] = {anchor_id: []}
            while queue:
                current_id = queue.pop(0)
                for neighbor_id, continuation in adjacency.get(
                    current_id, []
                ):
                    if neighbor_id in paths:
                        continue
                    paths[neighbor_id] = (
                        paths[current_id] + [continuation]
                    )
                    queue.append(neighbor_id)
            for member_id, path_relations in paths.items():
                if member_id == anchor_id:
                    continue
                member = self.elements[member_id]
                if member.section_id == anchor.section_id:
                    continue
                original = member.section_id
                self._assign_section(member, anchor.section_id)
                evidence = anchor_relations + path_relations
                decision = self._decision(
                    element=member,
                    rule="listing_chain_inherits_explicit_anchor",
                    original_section_id=original,
                    resolved_section_id=anchor.section_id,
                    confidence=min(
                        [relation.confidence for relation in evidence]
                    ),
                    status=SectionResolutionStatus.CONFIRMED,
                    created_by=RelationSource.DETERMINISTIC_RULE,
                    evidence_relations=evidence,
                    metadata={
                        "anchor_element_id": anchor_id,
                        "path_relation_ids": [
                            relation.relation_id
                            for relation in path_relations
                        ],
                        "anchor_relation_ids": [
                            relation.relation_id
                            for relation in anchor_relations
                        ],
                    },
                )
                self._record_confirmed(member, decision)
                self._locked_by_continuation.add(member.element_id)

    def _resolve_section_title_anchors(self) -> None:
        caption_relations = [
            relation
            for relation in self.document.relations
            if relation.relation_type == RelationType.CAPTION_OF
            and relation.status == RelationStatus.CONFIRMED
        ]
        for relation in caption_relations:
            caption = self.elements.get(relation.source_id)
            target = self.elements.get(relation.target_id)
            if (
                caption is None
                or target is None
                or not caption.text
                or target.element_type not in self._FLOATING_TYPES
            ):
                continue
            matches: list[tuple[int, int, str]] = []
            normalized_caption = self._normalized_title(caption.text)
            target_page = self.page_indexes[target.page_id]
            for section in self.document.sections:
                heading = self.elements.get(section.heading_element_id)
                if heading is None:
                    continue
                page_distance = (
                    self.page_indexes[heading.page_id] - target_page
                )
                normalized_title = self._normalized_title(section.title)
                if (
                    page_distance not in {0, 1}
                    or len(normalized_title) < 6
                    or not normalized_caption.startswith(
                        normalized_title
                    )
                ):
                    continue
                matches.append(
                    (
                        len(normalized_title),
                        -abs(page_distance),
                        section.section_id,
                    )
                )
            if not matches:
                continue
            matches.sort(reverse=True)
            best_length = matches[0][0]
            best = {
                section_id
                for length, _, section_id in matches
                if length == best_length
            }
            if len(best) != 1:
                continue
            resolved_section_id = next(iter(best))
            if target.section_id == resolved_section_id:
                continue
            original = target.section_id
            self._assign_section(target, resolved_section_id)
            decision = self._decision(
                element=target,
                rule="caption_matches_adjacent_section_title",
                original_section_id=original,
                resolved_section_id=resolved_section_id,
                confidence=0.98,
                status=SectionResolutionStatus.CONFIRMED,
                created_by=RelationSource.DETERMINISTIC_RULE,
                evidence_relations=[relation],
                metadata={
                    "caption_element_id": caption.element_id,
                    "matched_section_title": self.sections[
                        resolved_section_id
                    ].title,
                },
            )
            self._record_confirmed(target, decision)

    def _resolve_caption_groups(self) -> None:
        by_caption: dict[str, list[Relation]] = defaultdict(list)
        for relation in self.document.relations:
            if (
                relation.relation_type == RelationType.CAPTION_OF
                and relation.status == RelationStatus.CONFIRMED
            ):
                by_caption[relation.source_id].append(relation)
        relations_by_id = {
            relation.relation_id: relation
            for relation in self.document.relations
        }
        for caption_id, relations in by_caption.items():
            if len(relations) < 2:
                continue
            targets = [
                self.elements[relation.target_id]
                for relation in relations
                if relation.target_id in self.elements
            ]
            anchored = [
                target
                for target in targets
                if self._has_confirmed_resolution(target)
                and target.section_id
            ]
            anchored_sections = {
                target.section_id for target in anchored
            }
            if len(anchored_sections) != 1:
                continue
            resolved_section_id = next(iter(anchored_sections))
            anchor_evidence: list[Relation] = []
            for target in anchored:
                resolution = target.metadata.get("section_resolution", {})
                for relation_id in resolution.get(
                    "evidence_relation_ids", []
                ):
                    relation = relations_by_id.get(relation_id)
                    if relation is not None:
                        anchor_evidence.append(relation)
            for target in targets:
                if target.section_id == resolved_section_id:
                    continue
                original = target.section_id
                self._assign_section(target, resolved_section_id)
                decision = self._decision(
                    element=target,
                    rule=(
                        "caption_group_inherits_anchored_target_section"
                    ),
                    original_section_id=original,
                    resolved_section_id=resolved_section_id,
                    confidence=min(
                        relation.confidence for relation in relations
                    ),
                    status=SectionResolutionStatus.CONFIRMED,
                    created_by=RelationSource.DETERMINISTIC_RULE,
                    evidence_relations=relations + anchor_evidence,
                    metadata={
                        "caption_element_id": caption_id,
                        "group_target_ids": sorted(
                            item.element_id for item in targets
                        ),
                    },
                )
                self._record_confirmed(target, decision)

    def _record_candidate_continuations(self) -> None:
        by_target: dict[str, list[tuple[Relation, str]]] = defaultdict(list)
        for relation in self.document.relations:
            if (
                relation.relation_type != RelationType.CONTINUED_ON
                or relation.status != RelationStatus.CANDIDATE
            ):
                continue
            source = self.elements.get(relation.source_id)
            target = self.elements.get(relation.target_id)
            if (
                source is None
                or target is None
                or target.element_type not in self._FLOATING_TYPES
                or not source.section_id
                or source.section_id == target.section_id
            ):
                continue
            by_target[target.element_id].append(
                (relation, source.section_id)
            )

        for target_id, candidates in by_target.items():
            target = self.elements[target_id]
            section_ids = sorted(
                {section_id for _, section_id in candidates}
            )
            evidence_relations = [relation for relation, _ in candidates]
            if len(section_ids) > 1:
                decision = self._decision(
                    element=target,
                    rule="candidate_continuations_disagree",
                    original_section_id=target.section_id,
                    resolved_section_id=target.section_id,
                    confidence=max(
                        relation.confidence
                        for relation in evidence_relations
                    ),
                    status=SectionResolutionStatus.AMBIGUOUS,
                    created_by=RelationSource.LAYOUT_HEURISTIC,
                    evidence_relations=evidence_relations,
                    metadata={"candidate_section_ids": section_ids},
                )
                target.metadata.setdefault(
                    "section_resolution_ambiguities", []
                ).append(decision.model_dump(mode="json"))
            else:
                candidate_section_id = section_ids[0]
                decision = self._decision(
                    element=target,
                    rule="candidate_continued_on_does_not_change_section",
                    original_section_id=target.section_id,
                    resolved_section_id=target.section_id,
                    candidate_section_id=candidate_section_id,
                    confidence=max(
                        relation.confidence
                        for relation in evidence_relations
                    ),
                    status=SectionResolutionStatus.CANDIDATE,
                    created_by=evidence_relations[0].created_by,
                    evidence_relations=evidence_relations,
                )
                target.metadata["candidate_section_id"] = (
                    candidate_section_id
                )
                target.metadata.setdefault(
                    "section_resolution_candidates", []
                ).append(decision.model_dump(mode="json"))
            self.decisions.append(decision)

    def _reference_strength(
        self,
        source: Element,
        target: Element,
        relation: Relation,
    ) -> tuple[float, str] | None:
        source_page = self.page_indexes[source.page_id]
        target_page = self.page_indexes[target.page_id]
        page_distance = target_page - source_page
        reference_kind = relation.metadata.get("reference_kind")
        maximum_distance = (
            max(self.maximum_reference_page_distance, 10)
            if reference_kind == "listing"
            else self.maximum_reference_page_distance
        )
        if page_distance > maximum_distance:
            return None
        if (
            page_distance == 0
            and source.reading_order < target.reading_order
        ):
            penalty = 0.05 * page_distance
            return (
                round(
                    max(0.0, relation.confidence - penalty),
                    3,
                ),
                "preceding",
            )
        if page_distance > 0:
            penalty = 0.05 * page_distance
            return (
                round(
                    max(0.0, relation.confidence - penalty),
                    3,
                ),
                "preceding",
            )

        following_distance = abs(page_distance)
        if (
            following_distance > 1
            or "target_label_source_id" not in relation.metadata
            or self._current_section_heading_precedes_target(target)
        ):
            return None
        penalty = 0.10 + 0.05 * following_distance
        return (
            round(max(0.0, relation.confidence - penalty), 3),
            "following_boundary_float",
        )

    def _current_section_heading_precedes_target(
        self,
        target: Element,
    ) -> bool:
        section = self.sections.get(target.section_id or "")
        if section is None:
            return False
        heading = self.elements.get(section.heading_element_id)
        return bool(
            heading
            and heading.page_id == target.page_id
            and heading.reading_order < target.reading_order
        )

    @staticmethod
    def _normalized_title(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()

    @staticmethod
    def _has_confirmed_resolution(element: Element) -> bool:
        resolution = element.metadata.get("section_resolution")
        return bool(
            isinstance(resolution, dict)
            and resolution.get("status")
            == SectionResolutionStatus.CONFIRMED.value
        )

    def _assign_section(
        self,
        element: Element,
        section_id: str,
    ) -> None:
        section = self.sections.get(section_id)
        if section is None:
            return
        element.section_id = section_id
        element.section_path = list(section.section_path)

    def _synchronize_sections(self) -> None:
        for section in self.document.sections:
            section.element_ids = []
            section.page_ids = []
        ordered = sorted(
            self.document.elements,
            key=lambda element: (
                self.page_indexes[element.page_id],
                element.reading_order,
            ),
        )
        for element in ordered:
            if not element.section_id:
                continue
            section = self.sections.get(element.section_id)
            if section is None:
                continue
            section.element_ids.append(element.element_id)
            if element.page_id not in section.page_ids:
                section.page_ids.append(element.page_id)

    def _rebuild_membership_relations(self) -> None:
        retained = [
            relation
            for relation in self.document.relations
            if relation.relation_type != RelationType.BELONGS_TO_SECTION
        ]
        rebuilt = RelationBuilder(
            self.document
        ).build_section_membership_relations()
        self.document.relations = retained + rebuilt

    def _record_confirmed(
        self,
        element: Element,
        decision: SectionResolutionDecision,
    ) -> None:
        element.metadata["section_resolution"] = decision.model_dump(
            mode="json"
        )
        self.decisions.append(decision)

    def _decision(
        self,
        *,
        element: Element,
        rule: str,
        original_section_id: str | None,
        resolved_section_id: str | None,
        confidence: float,
        status: SectionResolutionStatus,
        created_by: RelationSource,
        evidence_relations: list[Relation],
        candidate_section_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SectionResolutionDecision:
        evidence_ids = sorted(
            {relation.relation_id for relation in evidence_relations}
        )
        return SectionResolutionDecision(
            decision_id=(
                "section-resolution:"
                + stable_digest(
                    element.element_id,
                    rule,
                    original_section_id,
                    resolved_section_id,
                    candidate_section_id,
                    status.value,
                    evidence_ids,
                )
            ),
            element_id=element.element_id,
            rule=rule,
            original_section_id=original_section_id,
            resolved_section_id=resolved_section_id,
            candidate_section_id=candidate_section_id,
            confidence=round(min(max(confidence, 0.0), 1.0), 3),
            status=status,
            created_by=created_by,
            evidence_relation_ids=evidence_ids,
            metadata=metadata or {},
        )

    def _position_for_id(self, element_id: str) -> tuple[int, int]:
        element = self.elements.get(element_id)
        if element is None:
            return (10**9, 10**9)
        return (
            self.page_indexes[element.page_id],
            element.reading_order,
        )
