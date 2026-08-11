"""Scope-aware numbered-label registry for explicit document references."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from softdoc.models import (
    Document,
    Element,
    ElementType,
    Relation,
    RelationStatus,
    RelationType,
)
from softdoc.profiles import DocumentProfile


_SUBREFERENCE = r"[A-Za-z](?:\s*[-\u2012\u2013\u2014]\s*[A-Za-z])?"

LABEL_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "figure": (
        re.compile(
            rf"(?:Figure|Fig\.?)\s*(?P<base>[0-9]+)"
            rf"(?:\s*(?:\(\s*(?P<paren_sub>{_SUBREFERENCE})\s*\)"
            rf"|(?P<suffix_sub>{_SUBREFERENCE})))?(?![A-Za-z0-9_])",
            re.IGNORECASE,
        ),
        re.compile(
            rf"图\s*(?P<base>[0-9]+)"
            rf"(?:\s*(?:\(\s*(?P<paren_sub>{_SUBREFERENCE})\s*\)"
            rf"|(?P<suffix_sub>{_SUBREFERENCE})))?(?![A-Za-z0-9_])",
            re.IGNORECASE,
        ),
    ),
    "table": (
        re.compile(
            rf"Table\s*(?P<base>[0-9]+)"
            rf"(?:\s*(?:\(\s*(?P<paren_sub>{_SUBREFERENCE})\s*\)"
            rf"|(?P<suffix_sub>{_SUBREFERENCE})))?(?![A-Za-z0-9_])",
            re.IGNORECASE,
        ),
        re.compile(
            rf"表\s*(?P<base>[0-9]+)"
            rf"(?:\s*(?:\(\s*(?P<paren_sub>{_SUBREFERENCE})\s*\)"
            rf"|(?P<suffix_sub>{_SUBREFERENCE})))?(?![A-Za-z0-9_])",
            re.IGNORECASE,
        ),
    ),
    "section": (
        re.compile(
            r"^\s*(?P<base>[0-9]+(?:\.[0-9]+)*)"
            r"(?=$|[^\d.]|\.(?!\d))"
        ),
        re.compile(r"第\s*(?P<base>[0-9]+(?:\.[0-9]+)*)\s*节"),
    ),
    "listing": (
        re.compile(
            r"Listing\s*(?P<base>[0-9]+)"
            r"(?![A-Za-z0-9_])",
            re.IGNORECASE,
        ),
    ),
}


@dataclass(frozen=True)
class ReferenceTarget:
    target_id: str
    kind: str
    number: str
    resolution_rule: str
    label_source_id: str | None
    label_text: str
    priority: int
    page_index: int | None
    section_id: str | None
    section_root: str | None


class LabelRegistry:
    """Retain every label candidate and resolve it in source scope."""

    def __init__(self, document: Document) -> None:
        self.document = document
        self.profile = DocumentProfile(
            str(
                document.metadata.get("document_profile", {}).get(
                    "profile", DocumentProfile.REPORT.value
                )
            )
        )
        self.pages = {
            page.page_id: page.page_index for page in document.pages
        }
        self.elements = {
            element.element_id: element for element in document.elements
        }
        self.sections = {
            section.section_id: section for section in document.sections
        }
        self._targets: dict[tuple[str, str], list[ReferenceTarget]] = {}

    def add(
        self,
        *,
        kind: str,
        number: str,
        target_id: str,
        resolution_rule: str,
        label_source_id: str | None,
        label_text: str,
        priority: int,
    ) -> None:
        element = self.elements.get(target_id)
        section = self.sections.get(target_id)
        if element is not None:
            page_index = self.pages.get(element.page_id)
            section_id = element.section_id
            section_root = (
                element.section_path[0]
                if element.section_path
                else None
            )
        elif section is not None:
            heading = self.elements.get(section.heading_element_id)
            page_index = self.pages.get(heading.page_id) if heading else None
            section_id = section.section_id
            section_root = (
                section.section_path[0] if section.section_path else None
            )
        else:
            return
        candidate = ReferenceTarget(
            target_id=target_id,
            kind=kind,
            number=number,
            resolution_rule=resolution_rule,
            label_source_id=label_source_id,
            label_text=label_text,
            priority=priority,
            page_index=page_index,
            section_id=section_id,
            section_root=section_root,
        )
        key = (kind, number)
        current = self._targets.setdefault(key, [])
        if not any(
            item.target_id == candidate.target_id
            and item.label_source_id == candidate.label_source_id
            for item in current
        ):
            current.append(candidate)

    def resolve(
        self,
        source: Element,
        kind: str,
        number: str,
        *,
        matched_text: str,
    ) -> ReferenceTarget | None:
        candidates = self._targets.get((kind, number), [])
        if not candidates:
            return None
        source_page = self.pages.get(source.page_id)
        source_root = source.section_path[0] if source.section_path else None
        source_tokens = self._tokens(source.text or "")

        def rank(candidate: ReferenceTarget) -> tuple[int, int, int, int, str]:
            same_section = int(
                bool(
                    source.section_id
                    and candidate.section_id == source.section_id
                )
            )
            same_root = int(
                bool(source_root and candidate.section_root == source_root)
            )
            distance = (
                abs(source_page - candidate.page_index)
                if source_page is not None
                and candidate.page_index is not None
                else 1_000_000
            )
            return (
                -same_section,
                -same_root,
                -candidate.priority,
                distance,
                candidate.target_id,
            )

        selected = min(candidates, key=rank)
        if (
            self.profile
            not in {DocumentProfile.ACADEMIC, DocumentProfile.SLIDES}
            and source_page is not None
            and selected.page_index is not None
            and abs(source_page - selected.page_index) > 5
            and source_root
            and selected.section_root
            and source_root != selected.section_root
        ):
            label_tokens = self._tokens(selected.label_text)
            shared = source_tokens & label_tokens
            meaningful = {
                token
                for token in shared
                if token not in {"figure", "fig", "table", "section"}
                and len(token) > 2
            }
            if not meaningful:
                return None
        return selected

    def lookup_all(self, kind: str, number: str) -> list[ReferenceTarget]:
        """Return every distinct target without source-scoped disambiguation.

        A target may have both an intrinsic label and a caption label.  Exact
        lookup treats those as two pieces of support for one target, not as an
        ambiguity, and keeps the highest-priority registration.
        """

        normalized = normalize_reference_number(number)
        by_target: dict[str, ReferenceTarget] = {}
        for candidate in self._targets.get((kind, normalized), []):
            current = by_target.get(candidate.target_id)
            if current is None or _target_preference(candidate) < _target_preference(
                current
            ):
                by_target[candidate.target_id] = candidate
        return sorted(by_target.values(), key=_target_preference)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token.casefold()
            for token in re.findall(r"[A-Za-z0-9]+", text)
            if len(token) > 1
        }


def build_label_registry(
    document: Document,
    *,
    caption_relations: Iterable[Relation] | None = None,
) -> LabelRegistry:
    """Build the shared Figure/Table/Section registry without mutation.

    RelationBuilder supplies the caption relations it has just generated.
    Read-only consumers such as ExactAnchorLookup use the relations already
    stored in the final SoftDoc.  Paragraph captions missed by the parser are
    inferred by the same deterministic layout helper in both paths.
    """

    registry = LabelRegistry(document)
    elements = {element.element_id: element for element in document.elements}

    for element in document.elements:
        if not element.reference_label:
            continue
        # Element targets are typed.  A bare numeric label on a Table/Figure
        # must not be reinterpreted as a Section merely because the Section
        # grammar also accepts leading numbers.
        kind = element_reference_kind(element, element.reference_label)
        number = extract_label_number(kind, element.reference_label) if kind else None
        if kind and number:
            registry.add(
                kind=kind,
                number=number,
                target_id=element.element_id,
                resolution_rule="element_reference_label",
                label_source_id=element.element_id,
                label_text=element.reference_label,
                priority=100,
            )

    relations = (
        list(caption_relations)
        if caption_relations is not None
        else [
            relation
            for relation in document.relations
            if relation.relation_type == RelationType.CAPTION_OF
            and relation.status == RelationStatus.CONFIRMED
        ]
    )
    for relation in relations:
        caption = elements.get(relation.source_id)
        target = elements.get(relation.target_id)
        if not caption or not target or not caption.text:
            continue
        # Caption labels inherit their namespace from the relation target.
        # Only actual Section objects below may register Section labels.
        kind = element_reference_kind(target, caption.text)
        number = extract_label_number(kind, caption.text) if kind else None
        if kind and number:
            registry.add(
                kind=kind,
                number=number,
                target_id=target.element_id,
                resolution_rule="caption_relation_label",
                label_source_id=caption.element_id,
                label_text=caption.text,
                priority=90,
            )

    for caption_candidate, target, kind, number in infer_paragraph_caption_targets(
        document
    ):
        registry.add(
            kind=kind,
            number=number,
            target_id=target.element_id,
            resolution_rule="paragraph_caption_layout_heuristic",
            label_source_id=caption_candidate.element_id,
            label_text=caption_candidate.text or "",
            priority=70,
        )

    for section in document.sections:
        number = extract_label_number("section", section.title)
        if number:
            registry.add(
                kind="section",
                number=number,
                target_id=section.section_id,
                resolution_rule="section_title_number",
                label_source_id=section.heading_element_id,
                label_text=section.title,
                priority=100,
            )
    return registry


def infer_paragraph_caption_targets(
    document: Document,
) -> list[tuple[Element, Element, str, str]]:
    """Find label-like paragraphs adjacent to compatible visual targets."""

    inferred: list[tuple[Element, Element, str, str]] = []
    for candidate in document.elements:
        if (
            candidate.element_type != ElementType.PARAGRAPH
            or not candidate.text
            or candidate.bbox is None
        ):
            continue
        label = leading_caption_label(candidate.text)
        if label is None:
            continue
        kind, number = label
        target = _nearest_visual_caption_target(document, candidate, kind)
        if target is not None:
            inferred.append((candidate, target, kind, number))
    return inferred


def element_reference_kind(
    element: Element,
    label_text: str | None = None,
) -> str | None:
    if element.element_type in {ElementType.CODE, ElementType.ALGORITHM}:
        # Code-like targets may be called either Figure or Listing by the
        # document.  The visible label selects between those two compatible
        # namespaces; neither can leak into the Section namespace.
        if label_text and reference_kind_from_label(label_text) == "listing":
            return "listing"
        return "figure"
    if element.element_type in {
        ElementType.FIGURE,
        ElementType.CHART,
    }:
        return "figure"
    if element.element_type == ElementType.TABLE:
        return "table"
    return None


def reference_kind_from_label(text: str) -> str | None:
    for kind, patterns in LABEL_PATTERNS.items():
        if any(pattern.search(text) for pattern in patterns):
            return kind
    return None


def extract_label_number(kind: str, text: str) -> str | None:
    for pattern in LABEL_PATTERNS.get(kind, ()):
        match = pattern.search(text)
        if match:
            number, _, _ = reference_number_parts(match)
            return number
    return None


def reference_number_parts(
    match: re.Match[str],
) -> tuple[str, str, str | None]:
    base_number = normalize_reference_number(match.group("base"))
    subreference = (
        match.groupdict().get("paren_sub")
        or match.groupdict().get("suffix_sub")
    )
    normalized_subreference = (
        normalize_subreference(subreference)
        if subreference is not None
        else None
    )
    number = (
        f"{base_number}{normalized_subreference}"
        if normalized_subreference is not None
        else base_number
    )
    return number, base_number, normalized_subreference


def normalize_reference_number(number: str) -> str:
    return number.strip().lower()


def normalize_subreference(value: str) -> str:
    return (
        re.sub(r"\s+", "", value)
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .lower()
    )


def leading_caption_label(text: str) -> tuple[str, str] | None:
    for kind in ("figure", "table", "listing"):
        for pattern in LABEL_PATTERNS[kind]:
            match = pattern.match(text)
            if match is None:
                continue
            remainder = text[match.end() :].lstrip()
            if not remainder or remainder[0] not in ":：.。-\u2013\u2014":
                continue
            number, _, _ = reference_number_parts(match)
            return kind, number
    return None


def _nearest_visual_caption_target(
    document: Document,
    caption_candidate: Element,
    kind: str,
) -> Element | None:
    compatible_types = (
        {ElementType.FIGURE, ElementType.CHART}
        if kind == "figure"
        else (
            {ElementType.TABLE}
            if kind == "table"
            else {ElementType.CODE, ElementType.ALGORITHM}
        )
    )
    candidates: list[tuple[float, int, Element]] = []
    for target in document.elements:
        if (
            target.page_id != caption_candidate.page_id
            or target.element_type not in compatible_types
            or target.bbox is None
        ):
            continue
        reading_gap = abs(target.reading_order - caption_candidate.reading_order)
        if reading_gap > 3:
            continue
        vertical_gap = _normalized_vertical_gap(caption_candidate, target)
        horizontal_overlap = _normalized_horizontal_overlap(
            caption_candidate,
            target,
        )
        if vertical_gap > 0.08 or horizontal_overlap < 0.25:
            continue
        candidates.append((vertical_gap, reading_gap, target))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


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


def _target_preference(target: ReferenceTarget) -> tuple[int, int, str, str]:
    return (
        -target.priority,
        target.page_index if target.page_index is not None else 1_000_000,
        target.target_id,
        target.label_source_id or "",
    )
