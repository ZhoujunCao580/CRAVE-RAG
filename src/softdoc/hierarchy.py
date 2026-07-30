"""Parser-neutral normalization of heading candidates."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from enum import Enum
from statistics import median
from typing import Any

from pydantic import Field

from softdoc.models import Element, ElementType, Page, RelationSource, SoftDocModel
from softdoc.profiles import DocumentProfile


_ARABIC_NUMBERED_HEADING = re.compile(
    r"^\s*(?P<number>\d+(?:\.\d+)*)(?:\.)?\s+\S"
)
_APPENDIX_NUMBERED_HEADING = re.compile(
    r"^\s*(?P<letter>[A-Z])(?:\.(?P<number>\d+(?:\.\d+)*))?(?:\.)?\s+\S"
)
_APPENDIX_WORD_HEADING = re.compile(
    r"^\s*appendix\s+(?P<letter>[A-Z])"
    r"(?:\.(?P<number>\d+(?:\.\d+)*))?(?:\s|$)",
    re.IGNORECASE,
)
_PART_HEADING = re.compile(
    r"^\s*part\s+(?:[ivxlcdm]+|\d+|[a-z])"
    r"(?![a-z0-9])(?:\s*[:.\-]\s*|\s+.*)?$",
    re.IGNORECASE,
)
_CHAPTER_HEADING = re.compile(
    r"^\s*chapter\s+(?:[ivxlcdm]+|\d+|[a-z])"
    r"(?![a-z0-9])(?:\s*[:.\-]\s*|\s+.*)?$",
    re.IGNORECASE,
)
_ITEM_HEADING = re.compile(
    r"^\s*item\s+\d+[a-z]?(?![a-z0-9]).*$",
    re.IGNORECASE,
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
    "executive summary",
    "introduction",
    "limitations",
    "references",
}


class HeadingAction(str, Enum):
    DOCUMENT_TITLE = "document_title"
    SECTION_HEADING = "section_heading"
    DEMOTED_TO_PARAGRAPH = "demoted_to_paragraph"
    EXCLUDED_REPEATED_REGION = "excluded_repeated_region"


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
    decisions: list[HeadingDecision] = Field(default_factory=list)


class HeadingHierarchyBuilder:
    """Normalize headings using document-wide evidence with conservative fallback.

    Priority is explicit numbering, structural labels, document-wide visual
    style, indentation, and finally the parser level. An uncertain heading is
    kept at a nearby level; it is never made one level deeper merely because it
    followed another heading.
    """

    def __init__(self, *, max_level: int = 6) -> None:
        self.max_level = max(1, int(max_level))

    def build(
        self,
        document_id: str,
        pages: list[Page],
        elements: list[Element],
        *,
        profile: DocumentProfile = DocumentProfile.REPORT,
    ) -> HeadingHierarchyResult:
        del document_id  # IDs and Sections are intentionally built elsewhere.
        page_by_id = {page.page_id: page for page in pages}
        ordered = sorted(
            elements,
            key=lambda item: (
                page_by_id[item.page_id].page_index,
                item.reading_order,
            ),
        )
        decisions: list[HeadingDecision] = []
        original_levels = {
            element.element_id: self._original_level(element)
            for element in ordered
            if element.element_type == ElementType.HEADING
        }
        for element in ordered:
            if element.element_type == ElementType.HEADING:
                element.metadata["parser_heading_level"] = original_levels[element.element_id]

        for element in ordered:
            if element.element_type != ElementType.HEADING:
                continue
            if element.metadata.get("excluded_from_heading_hierarchy") is True:
                eligibility = element.metadata.get("heading_eligibility", {})
                eligibility_reason = (
                    eligibility.get("reason")
                    if isinstance(eligibility, dict)
                    else None
                )
                decision = HeadingDecision(
                    element_id=element.element_id,
                    page_id=element.page_id,
                    page_number=element.page_number,
                    text=element.text or "",
                    action=HeadingAction.EXCLUDED_REPEATED_REGION,
                    original_level=original_levels[element.element_id],
                    confidence=0.96,
                    evidence={
                        "rule": (
                            "excluded_repeated_header_or_footer"
                            if element.metadata.get("repeated_region")
                            else "excluded_non_structural_heading"
                        ),
                        "eligibility_reason": eligibility_reason,
                        "region": element.metadata.get("repeated_region"),
                        "detector_evidence": element.metadata.get(
                            "repeated_region_evidence", {}
                        ),
                    },
                )
                self._record_decision(element, decision)
                decisions.append(decision)
                element.heading_level = None
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
            and element.metadata.get("excluded_from_heading_hierarchy") is not True
        ]
        title_element = self._detect_document_title(
            headings, page_by_id, original_levels, profile
        )
        document_title: str | None = None
        if title_element is not None:
            title_element.heading_level = 1
            title_pattern = self._explicit_level(title_element.text or "", has_part=False)
            title_decision = HeadingDecision(
                element_id=title_element.element_id,
                page_id=title_element.page_id,
                page_number=title_element.page_number,
                text=title_element.text or "",
                action=HeadingAction.DOCUMENT_TITLE,
                original_level=original_levels[title_element.element_id],
                normalized_level=1,
                confidence=self._document_title_confidence(
                    title_element, headings, original_levels
                ),
                evidence={
                    "rule": "first_page_top_heading",
                    "matched_number_pattern": (
                        title_pattern[1] if title_pattern is not None else None
                    ),
                    "number_pattern_overridden": title_pattern is not None,
                    "excluded_from_section_tree": True,
                },
            )
            self._record_decision(title_element, title_decision)
            decisions.append(title_decision)
            document_title = title_element.text

        section_headings = [
            heading for heading in headings if heading is not title_element
        ]
        has_part = any(
            _PART_HEADING.match(self._normalized_text(heading.text or ""))
            for heading in section_headings
        )
        explicit = {
            heading.element_id: self._explicit_level(
                heading.text or "", has_part=has_part
            )
            for heading in section_headings
        }
        style_profiles, style_order = self._build_style_profiles(
            section_headings, explicit
        )
        parser_levels = [
            original_levels[heading.element_id]
            for heading in section_headings
            if original_levels[heading.element_id] is not None
        ]
        minimum_parser_level = min(parser_levels) if parser_levels else None
        reliable_history: list[tuple[Element, int, str]] = []
        active_numbered_parent: tuple[Element, int] | None = None

        for heading in section_headings:
            explicit_match = explicit[heading.element_id]
            forced_level = heading.metadata.get("forced_heading_level")
            if isinstance(forced_level, int) and forced_level >= 1:
                level = forced_level
                rule = "profile_forced_sibling_level"
                rule_data = {"profile": profile.value}
                confidence = 0.94
            elif explicit_match is not None:
                level, rule, rule_data = explicit_match
                confidence = 1.0 if rule != "known_top_level_heading" else 0.98
            else:
                style_key = self._style_key(heading)
                style_profile = style_profiles.get(style_key)
                style_level = self._style_level(
                    heading,
                    style_profile,
                    style_order,
                    reliable_history,
                )
                if style_level is not None:
                    level, style_data = style_level
                    rule = "document_wide_visual_style"
                    rule_data = style_data
                    confidence = 0.86 if style_data["anchored"] else 0.80
                else:
                    indentation_level = self._indentation_level(
                        heading, reliable_history
                    )
                    if indentation_level is not None:
                        level, indentation_data = indentation_level
                        rule = "indentation_relative_to_previous_heading"
                        rule_data = indentation_data
                        confidence = 0.74
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
                            confidence = 0.66
                        elif reliable_history:
                            level = reliable_history[-1][1]
                            rule = "conservative_same_level_fallback"
                            rule_data = {
                                "previous_heading_id": reliable_history[-1][0].element_id,
                                "previous_level": reliable_history[-1][1],
                            }
                            confidence = 0.58
                        else:
                            level = 1
                            rule = "top_level_fallback"
                            rule_data = {}
                            confidence = 0.55

            if (
                explicit_match is None
                and not isinstance(forced_level, int)
                and reliable_history
            ):
                previous, previous_level, previous_rule = reliable_history[-1]
                previous_is_numbered = previous_rule in {
                    "arabic_section_number",
                    "appendix_section_number",
                }
                current_is_less_prominent = (
                    self._prominence(heading)
                    <= self._prominence(previous) * 1.10
                )
                if (
                    profile == DocumentProfile.ACADEMIC
                    and
                    previous_is_numbered
                    and current_is_less_prominent
                    and level <= previous_level
                    and not self._explicit_level(
                        heading.text or "", has_part=has_part
                    )
                ):
                    level = min(previous_level + 1, self.max_level)
                    rule_data = {
                        **rule_data,
                        "context_parent_heading_id": previous.element_id,
                        "context_parent_level": previous_level,
                    }
                    rule = "unnumbered_run_in_under_numbered_parent"
                    confidence = max(confidence, 0.86)
                elif (
                    profile == DocumentProfile.ACADEMIC
                    and active_numbered_parent is not None
                    and self._prominence(heading)
                    <= self._prominence(active_numbered_parent[0]) * 1.10
                    and level <= active_numbered_parent[1]
                ):
                    level = min(active_numbered_parent[1] + 1, self.max_level)
                    rule_data = {
                        **rule_data,
                        "context_parent_heading_id": (
                            active_numbered_parent[0].element_id
                        ),
                        "context_parent_level": active_numbered_parent[1],
                    }
                    rule = "unnumbered_run_in_under_active_numbered_parent"
                    confidence = max(confidence, 0.84)

            if explicit_match is None and level > 3:
                rule_data = {
                    **rule_data,
                    "level_before_conservative_clamp": level,
                    "conservative_max_inferred_level": 3,
                }
                level = 3
            level = min(max(1, level), self.max_level)
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
            reliable_history.append((heading, level, rule))
            if rule in {
                "arabic_section_number",
                "appendix_section_number",
            }:
                active_numbered_parent = (heading, level)

        return HeadingHierarchyResult(
            document_title=document_title,
            decisions=decisions,
        )

    def _build_style_profiles(
        self,
        headings: list[Element],
        explicit: dict[str, tuple[int, str, dict[str, Any]] | None],
    ) -> tuple[dict[str, dict[str, Any]], list[str]]:
        members: dict[str, list[Element]] = defaultdict(list)
        for heading in headings:
            members[self._style_key(heading)].append(heading)

        profiles: dict[str, dict[str, Any]] = {}
        for key, items in members.items():
            anchored_levels = [
                explicit[item.element_id][0]
                for item in items
                if explicit[item.element_id] is not None
            ]
            profiles[key] = {
                "key": key,
                "element_ids": [item.element_id for item in items],
                "count": len(items),
                "page_count": len({item.page_id for item in items}),
                "prominence": median(
                    self._prominence(item) for item in items
                ),
                "anchored_levels": anchored_levels,
                "anchored_level": (
                    Counter(anchored_levels).most_common(1)[0][0]
                    if anchored_levels and len(set(anchored_levels)) == 1
                    else None
                ),
                "has_conflicting_anchors": len(set(anchored_levels)) > 1,
            }
        style_order = [
            key
            for key, _ in sorted(
                profiles.items(),
                key=lambda item: (-item[1]["prominence"], item[0]),
            )
        ]
        return profiles, style_order

    def _style_level(
        self,
        heading: Element,
        profile: dict[str, Any] | None,
        style_order: list[str],
        history: list[tuple[Element, int, str]],
    ) -> tuple[int, dict[str, Any]] | None:
        if profile is None:
            return None
        strong_profile = profile["count"] >= 2 or bool(profile["anchored_levels"])
        if not strong_profile:
            return None
        anchored_level = profile["anchored_level"]
        if anchored_level is not None:
            level = anchored_level
            anchored = True
        elif profile["has_conflicting_anchors"]:
            previous = next(
                (
                    (item, item_level)
                    for item, item_level, _ in reversed(history)
                    if self._style_key(item) == profile["key"]
                ),
                None,
            )
            if previous is None:
                return None
            level = previous[1]
            anchored = True
        else:
            level = min(style_order.index(profile["key"]) + 1, 3)
            anchored = False
        return (
            level,
            {
                "style_signature": profile["key"],
                "style_occurrences": profile["count"],
                "style_page_count": profile["page_count"],
                "style_rank": style_order.index(profile["key"]) + 1,
                "style_prominence": round(profile["prominence"], 6),
                "anchored": anchored,
                "anchor_levels": profile["anchored_levels"],
                "same_style_is_same_level": True,
            },
        )

    def _indentation_level(
        self,
        heading: Element,
        history: list[tuple[Element, int, str]],
    ) -> tuple[int, dict[str, Any]] | None:
        if heading.bbox is None or not history:
            return None
        previous, previous_level, _ = history[-1]
        if previous.bbox is None:
            return None
        current_x = heading.bbox.normalized[0]
        previous_x = previous.bbox.normalized[0]
        delta = current_x - previous_x
        if abs(delta) <= 0.025:
            return (
                previous_level,
                {
                    "current_x1": round(current_x, 6),
                    "previous_x1": round(previous_x, 6),
                    "indent_delta": round(delta, 6),
                    "decision": "same_indent_same_level",
                },
            )
        if delta >= 0.05:
            return (
                min(previous_level + 1, self.max_level),
                {
                    "current_x1": round(current_x, 6),
                    "previous_x1": round(previous_x, 6),
                    "indent_delta": round(delta, 6),
                    "decision": "deeper_indent",
                },
            )
        if delta <= -0.05:
            return (
                max(1, previous_level - 1),
                {
                    "current_x1": round(current_x, 6),
                    "previous_x1": round(previous_x, 6),
                    "indent_delta": round(delta, 6),
                    "decision": "shallower_indent",
                },
            )
        return None

    def _detect_document_title(
        self,
        headings: list[Element],
        page_by_id: dict[str, Page],
        original_levels: dict[str, int | None],
        profile: DocumentProfile,
    ) -> Element | None:
        if not headings:
            return None
        first_page_headings = [
            heading
            for heading in headings
            if page_by_id[heading.page_id].page_index == 0
        ]
        if not first_page_headings:
            return None
        candidate = max(
            first_page_headings,
            key=lambda heading: (
                heading.bbox.height if heading.bbox else 0.0,
                self._prominence(heading),
                -heading.reading_order,
            ),
        )
        page = page_by_id[candidate.page_id]
        text = candidate.text or ""
        large_cover_heading = bool(
            candidate.bbox
            and (
                candidate.bbox.height > 0.06
                or (
                    profile
                    in {
                        DocumentProfile.SLIDES,
                        DocumentProfile.BROCHURE,
                    }
                    and candidate.bbox.width >= 0.45
                    and candidate.bbox.height >= 0.025
                )
            )
        )
        if (
            page.page_index != 0
            or (candidate.reading_order > 1 and not large_cover_heading)
        ):
            return None
        explicit = self._explicit_level(text, has_part=False)
        original_level = original_levels[candidate.element_id]
        followed_by_abstract = any(
            self._normalized_text(heading.text or "").casefold().rstrip(":")
            == "abstract"
            for heading in headings[1:3]
        )
        if explicit is not None:
            _, rule, _ = explicit
            if not (
                rule == "appendix_section_number"
                and original_level == 1
                and followed_by_abstract
            ):
                return None
        if self._normalized_text(text).casefold().rstrip(":") in _KNOWN_TOP_LEVEL_HEADINGS:
            return None
        if (
            candidate.bbox is not None
            and candidate.bbox.normalized[1] > 0.20
            and not large_cover_heading
        ):
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
        *,
        has_part: bool,
    ) -> tuple[int, str, dict[str, Any]] | None:
        clean = self._normalized_text(text)
        appendix_word = _APPENDIX_WORD_HEADING.match(clean)
        if appendix_word:
            suffix = appendix_word.group("number")
            level = 1 if suffix is None else suffix.count(".") + 2
            label = appendix_word.group("letter").upper()
            if suffix:
                label = f"{label}.{suffix}"
            return (
                min(level, self.max_level),
                "appendix_section_number",
                {"section_number": label, "structural_label": "appendix"},
            )
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
        if _PART_HEADING.match(clean):
            return (1, "part_heading", {"structural_label": "part"})
        if _CHAPTER_HEADING.match(clean):
            return (1, "chapter_heading", {"structural_label": "chapter"})
        if _ITEM_HEADING.match(clean):
            return (
                2 if has_part else 1,
                "item_heading",
                {"structural_label": "item", "part_present": has_part},
            )
        if clean.casefold().rstrip(":") in _KNOWN_TOP_LEVEL_HEADINGS:
            return (1, "known_top_level_heading", {"normalized_text": clean})
        return None

    @staticmethod
    def _original_level(element: Element) -> int | None:
        stored = element.metadata.get("parser_heading_level")
        if isinstance(stored, int) and stored >= 1:
            return stored
        return element.heading_level

    def _style_key(self, element: Element) -> str:
        style = element.metadata.get("style")
        style = style if isinstance(style, dict) else {}
        font_size = self._numeric_style_value(
            style, "font_size", "fontSize", "size", "text_size"
        )
        weight = str(
            style.get("font_weight")
            or style.get("fontWeight")
            or style.get("weight")
            or ""
        ).casefold()
        bold = bool(style.get("bold")) or weight in {
            "bold",
            "semibold",
            "600",
            "700",
            "800",
            "900",
        }
        family = str(
            style.get("font_family") or style.get("fontFamily") or ""
        ).casefold()
        if element.bbox is None:
            height_bucket = "none"
            indent_bucket = "none"
        else:
            height_bucket = str(round(element.bbox.height / 0.006))
            indent_bucket = str(round(element.bbox.normalized[0] / 0.04))
        size_bucket = "none" if font_size is None else f"{font_size:.1f}"
        return (
            f"size={size_bucket}|height={height_bucket}|bold={int(bold)}"
            f"|family={family}|indent={indent_bucket}"
        )

    def _prominence(self, element: Element) -> float:
        style = element.metadata.get("style")
        style = style if isinstance(style, dict) else {}
        font_size = self._numeric_style_value(
            style, "font_size", "fontSize", "size", "text_size"
        )
        height = element.bbox.height if element.bbox is not None else 0.02
        x1 = element.bbox.normalized[0] if element.bbox is not None else 0.1
        size_signal = (font_size / 100.0) if font_size is not None else 0.0
        return size_signal + height * 6.0 - x1 * 0.04

    @staticmethod
    def _numeric_style_value(
        style: dict[str, Any],
        *keys: str,
    ) -> float | None:
        for key in keys:
            value = style.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        return None

    @staticmethod
    def _looks_like_checklist_item(text: str) -> bool:
        has_question = "?" in text
        has_check_marker = any(
            marker in text for marker in ("✓", "✔", "☐", "☑", "□")
        )
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
