"""Parser-neutral normalization of heading levels and section hierarchy."""

from __future__ import annotations

import re
from enum import Enum
from statistics import median
from typing import Any

from pydantic import Field

from softdoc.ids import section_id
from softdoc.models import Element, ElementType, Page, RelationSource, Section, SoftDocModel


_ARABIC_NUMBERED_HEADING = re.compile(
    r"^\s*(?P<number>\d+(?:\.\d+)*)(?:\.)?\s+\S"
)
_APPENDIX_NUMBERED_HEADING = re.compile(
    r"^\s*(?P<letter>[A-Z])(?:\.(?P<number>\d+(?:\.\d+)*))?(?:\.)?\s+\S"
)
_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")

_KNOWN_TOP_LEVEL_HEADINGS = {
    "abstract",
    "acknowledgements",
    "acknowledgments",
    "bibliography",
    "conclusion",
    "conclusions",
    "ethics statement",
    "introduction",
    "limitations",
    "references",
}


class HeadingAction(str, Enum):
    DOCUMENT_TITLE = "document_title"
    SECTION_HEADING = "section_heading"
    DEMOTED_TO_PARAGRAPH = "demoted_to_paragraph"


class HeadingDecision(SoftDocModel):
    element_id: str
    page_id: str
    page_number: int
    text: str
    action: HeadingAction
    original_level: int | None = None
    normalized_level: int | None = Field(default=None, ge=1)
    confidence: float = Field(ge=0.0, le=1.0)
    created_by: RelationSource = RelationSource.DETERMINISTIC_RULE
    evidence: dict[str, Any] = Field(default_factory=dict)


class HeadingHierarchyResult(SoftDocModel):
    document_title: str | None = None
    sections: list[Section] = Field(default_factory=list)
    decisions: list[HeadingDecision] = Field(default_factory=list)


class HeadingHierarchyBuilder:
    """Normalize parser heading candidates without relying on parser field names.

    Explicit numbering is the strongest signal. Common unnumbered document
    sections are treated as top-level anchors. Other unnumbered headings become
    children of the nearest explicit anchor. All choices remain inspectable in
    ``HeadingDecision`` records.
    """

    def __init__(self, *, max_level: int = 6) -> None:
        self.max_level = max(1, int(max_level))

    def build(
        self,
        document_id: str,
        pages: list[Page],
        elements: list[Element],
    ) -> HeadingHierarchyResult:
        page_by_id = {page.page_id: page for page in pages}
        ordered = sorted(
            elements,
            key=lambda item: (
                page_by_id[item.page_id].page_index,
                item.reading_order,
            ),
        )
        for element in ordered:
            element.section_id = None
            element.section_path = None

        decisions: list[HeadingDecision] = []
        original_levels = {
            element.element_id: element.heading_level
            for element in ordered
            if element.element_type == ElementType.HEADING
        }

        for element in ordered:
            if element.element_type != ElementType.HEADING:
                continue
            if not self._looks_like_checklist_item(element.text or ""):
                continue
            decision = HeadingDecision(
                element_id=element.element_id,
                page_id=element.page_id,
                page_number=element.page_number,
                text=element.text or "",
                action=HeadingAction.DEMOTED_TO_PARAGRAPH,
                original_level=original_levels[element.element_id],
                normalized_level=None,
                confidence=0.99,
                evidence={
                    "rule": "checked_question_is_not_a_heading",
                    "signals": ["question_mark", "checklist_marker"],
                },
            )
            self._record_decision(element, decision)
            decisions.append(decision)
            element.element_type = ElementType.PARAGRAPH
            element.heading_level = None

        headings = [
            element
            for element in ordered
            if element.element_type == ElementType.HEADING
        ]
        document_title: str | None = None
        title_element = self._detect_document_title(headings, page_by_id, original_levels)
        if title_element is not None:
            title_element.heading_level = 1
            title_pattern = self._explicit_level(title_element.text or "")
            title_decision = HeadingDecision(
                element_id=title_element.element_id,
                page_id=title_element.page_id,
                page_number=title_element.page_number,
                text=title_element.text or "",
                action=HeadingAction.DOCUMENT_TITLE,
                original_level=original_levels[title_element.element_id],
                normalized_level=1,
                confidence=self._document_title_confidence(
                    title_element,
                    headings,
                    original_levels,
                ),
                evidence={
                    "rule": "first_page_top_heading",
                    "matched_number_pattern": (
                        title_pattern[1] if title_pattern is not None else None
                    ),
                    "number_pattern_overridden": title_pattern is not None,
                    "is_known_section_title": False,
                    "excluded_from_section_tree": True,
                },
            )
            self._record_decision(title_element, title_decision)
            decisions.append(title_decision)
            document_title = title_element.text

        section_headings = [
            heading for heading in headings if heading is not title_element
        ]
        parser_levels = [
            original_levels[heading.element_id]
            for heading in section_headings
            if original_levels[heading.element_id] is not None
        ]
        minimum_parser_level = min(parser_levels) if parser_levels else None
        last_explicit_level: int | None = None

        for heading in section_headings:
            explicit = self._explicit_level(heading.text or "")
            if explicit is not None:
                level, rule, rule_data = explicit
                confidence = 1.0 if rule != "known_top_level_heading" else 0.98
                last_explicit_level = level
            elif last_explicit_level is not None:
                level = min(last_explicit_level + 1, self.max_level)
                rule = "unnumbered_child_of_nearest_explicit_heading"
                rule_data = {"parent_anchor_level": last_explicit_level}
                confidence = 0.82
            else:
                original = original_levels[heading.element_id]
                if original is not None and minimum_parser_level is not None:
                    level = min(
                        max(1, original - minimum_parser_level + 1),
                        self.max_level,
                    )
                    rule = "normalized_parser_level_fallback"
                    rule_data = {
                        "minimum_parser_level": minimum_parser_level,
                        "parser_level": original,
                    }
                    confidence = 0.70
                else:
                    level = 1
                    rule = "top_level_fallback"
                    rule_data = {}
                    confidence = 0.60

            heading.heading_level = level
            decision = HeadingDecision(
                element_id=heading.element_id,
                page_id=heading.page_id,
                page_number=heading.page_number,
                text=heading.text or "",
                action=HeadingAction.SECTION_HEADING,
                original_level=original_levels[heading.element_id],
                normalized_level=level,
                confidence=confidence,
                evidence={"rule": rule, **rule_data},
            )
            self._record_decision(heading, decision)
            decisions.append(decision)

        sections = self._build_sections(document_id, ordered)
        return HeadingHierarchyResult(
            document_title=document_title,
            sections=sections,
            decisions=decisions,
        )

    def _build_sections(
        self,
        document_id: str,
        ordered_elements: list[Element],
    ) -> list[Section]:
        stack: list[Section] = []
        sections: list[Section] = []
        for element in ordered_elements:
            hierarchy = element.metadata.get("heading_hierarchy", {})
            if hierarchy.get("action") == HeadingAction.DOCUMENT_TITLE.value:
                continue
            if element.element_type == ElementType.HEADING:
                level = element.heading_level or 1
                while stack and stack[-1].level >= level:
                    stack.pop()
                path = [section.title for section in stack] + [element.text or ""]
                current = Section(
                    section_id=section_id(document_id, element.element_id),
                    document_id=document_id,
                    title=element.text or "",
                    level=level,
                    heading_element_id=element.element_id,
                    parent_section_id=stack[-1].section_id if stack else None,
                    section_path=path,
                    page_ids=[element.page_id],
                    element_ids=[element.element_id],
                    provenance=element.provenance,
                    metadata={
                        "created_by": RelationSource.DETERMINISTIC_RULE.value,
                        "heading_confidence": hierarchy.get("confidence"),
                        "heading_evidence": hierarchy.get("evidence", {}),
                    },
                )
                sections.append(current)
                stack.append(current)
                element.section_id = current.section_id
                element.section_path = path
                continue
            if not stack:
                continue
            current = stack[-1]
            element.section_id = current.section_id
            element.section_path = list(current.section_path)
            if element.element_id not in current.element_ids:
                current.element_ids.append(element.element_id)
            if element.page_id not in current.page_ids:
                current.page_ids.append(element.page_id)
        return sections

    def _detect_document_title(
        self,
        headings: list[Element],
        page_by_id: dict[str, Page],
        original_levels: dict[str, int | None],
    ) -> Element | None:
        if not headings:
            return None
        candidate = headings[0]
        page = page_by_id[candidate.page_id]
        text = candidate.text or ""
        if page.page_index != 0 or candidate.reading_order > 1:
            return None
        explicit = self._explicit_level(text)
        original_level = original_levels[candidate.element_id]
        followed_by_abstract = any(
            self._normalized_text(heading.text or "").casefold().rstrip(":")
            == "abstract"
            for heading in headings[1:3]
        )
        if explicit is not None:
            _, rule, _ = explicit
            # A paper title may legitimately begin with an uppercase article
            # ("A ..."). A first-page level-1 candidate followed by Abstract
            # is stronger evidence than the appendix-letter pattern.
            if not (
                rule == "appendix_section_number"
                and original_level == 1
                and followed_by_abstract
            ):
                return None
        if self._normalized_text(text) in _KNOWN_TOP_LEVEL_HEADINGS:
            return None
        if candidate.bbox is not None and candidate.bbox.normalized[1] > 0.20:
            return None

        if original_level == 1:
            return candidate
        other_heights = [
            heading.bbox.height
            for heading in headings[1:]
            if heading.bbox is not None
        ]
        if (
            candidate.bbox is not None
            and other_heights
            and candidate.bbox.height >= median(other_heights) * 1.15
        ):
            return candidate
        if len(headings) == 1:
            return candidate
        return None

    def _document_title_confidence(
        self,
        candidate: Element,
        headings: list[Element],
        original_levels: dict[str, int | None],
    ) -> float:
        if original_levels[candidate.element_id] == 1:
            return 0.98
        if len(headings) == 1:
            return 0.75
        return 0.90

    def _explicit_level(
        self,
        text: str,
    ) -> tuple[int, str, dict[str, Any]] | None:
        clean = self._normalized_text(text)
        arabic = _ARABIC_NUMBERED_HEADING.match(clean)
        if arabic:
            number = arabic.group("number")
            return (
                min(number.count(".") + 1, self.max_level),
                "arabic_section_number",
                {"section_number": number},
            )
        appendix = _APPENDIX_NUMBERED_HEADING.match(clean)
        if appendix:
            suffix = appendix.group("number")
            level = 1 if suffix is None else suffix.count(".") + 2
            label = appendix.group("letter")
            if suffix:
                label = f"{label}.{suffix}"
            return (
                min(level, self.max_level),
                "appendix_section_number",
                {"section_number": label},
            )
        if clean.casefold().rstrip(":") in _KNOWN_TOP_LEVEL_HEADINGS:
            return (1, "known_top_level_heading", {"normalized_text": clean})
        return None

    @staticmethod
    def _looks_like_checklist_item(text: str) -> bool:
        has_question = "?" in text
        has_check_marker = any(marker in text for marker in ("✓", "✗", "☐", "☑", "☒"))
        has_control_character = any(
            ord(character) < 32 and character not in "\t\n\r"
            for character in text
        )
        return has_question and (has_check_marker or has_control_character)

    @staticmethod
    def _normalized_text(text: str) -> str:
        without_tags = _HTML_TAG.sub("", text)
        without_controls = "".join(
            character if ord(character) >= 32 else " "
            for character in without_tags
        )
        return _WHITESPACE.sub(" ", without_controls).strip()

    @staticmethod
    def _record_decision(element: Element, decision: HeadingDecision) -> None:
        element.metadata["heading_hierarchy"] = decision.model_dump(mode="json")
