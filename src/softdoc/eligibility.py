"""Narrow heading eligibility decisions used before hierarchy construction."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from pydantic import Field

from softdoc.models import Element, ElementType, Page, RelationSource, SoftDocModel
from softdoc.profiles import DocumentProfile


class HeadingEligibilityDecision(SoftDocModel):
    element_id: str
    page_id: str
    page_number: int
    eligible: bool
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    created_by: RelationSource = RelationSource.DETERMINISTIC_RULE
    evidence: dict[str, object] = Field(default_factory=dict)


class HeadingEligibilityDetector:
    """Decide whether parser headings may create global Sections.

    This is intentionally not a general SemanticRole classifier.  It only
    protects the outline from page furniture, local visual labels, TOC/index
    entries, procedure steps, and other high-confidence non-section text.
    """

    _STRUCTURAL_PREFIX = re.compile(
        r"^\s*(?:"
        r"\d+(?:\.\d+)*\.?\s+\S|"
        r"[A-Z](?:\.\d+)*\.?\s+\S|"
        r"(?:part|chapter|item|section|appendix)\s+\S"
        r")",
        re.IGNORECASE,
    )
    _PHONE_OR_CONTACT = re.compile(
        r"^\s*(?:\+?\d[\d\s().-]{6,}|www\.|https?://|\S+@\S+)\s*$",
        re.IGNORECASE,
    )
    _PROCEDURE = re.compile(
        r"^\s*(?:step\s+\d+\s*[:.)-]?|\d+\s+(?:on|click|choose|press|select|open|insert|connect|make|use)\b)",
        re.IGNORECASE,
    )
    _LOCAL_STAT = re.compile(
        r"^\s*(?:\d+\s*(?:st|nd|rd|th)|\d+(?:\.\d+)?%|[A-Z]?\d{1,2}[A-Z]{2,})\s*$",
        re.IGNORECASE,
    )
    _SENTENCE_PREFIX = re.compile(
        r"^\s*(?:for(?:\s|(?=[A-Z]))|outline|source|note|notes)\b",
        re.IGNORECASE,
    )
    _KNOWN_GLOBAL = {
        "abstract",
        "acknowledgements",
        "acknowledgments",
        "appendix",
        "bibliography",
        "conclusion",
        "conclusions",
        "contents",
        "executive summary",
        "index",
        "introduction",
        "limitations",
        "references",
    }

    def detect(
        self,
        pages: list[Page],
        elements: list[Element],
        profile: DocumentProfile,
        *,
        trusted_parser_types: bool = False,
    ) -> list[HeadingEligibilityDecision]:
        if trusted_parser_types:
            return self._detect_with_trusted_parser_types(
                pages,
                elements,
                profile,
            )
        page_by_id = {page.page_id: page for page in pages}
        ordered = sorted(
            elements,
            key=lambda item: (
                page_by_id[item.page_id].page_index,
                item.reading_order,
            ),
        )
        headings = [
            element
            for element in ordered
            if element.element_type == ElementType.HEADING
        ]
        normalized_counts = Counter(
            self._normalized(element.text or "") for element in headings
        )
        normalized_pages: dict[str, set[str]] = defaultdict(set)
        for element in headings:
            normalized_pages[self._normalized(element.text or "")].add(
                element.page_id
            )

        first_chapter_position = self._first_matching_position(
            headings,
            r"^\s*chapter\s+\d+\b",
            page_by_id,
        )
        index_position = self._first_matching_position(
            headings,
            r"^\s*index\s*$",
            page_by_id,
        )
        contents_pages = {
            element.page_id
            for element in elements
            if re.fullmatch(
                r"\s*(?:table of )?contents\s*",
                self._plain_text(element.text or ""),
                re.IGNORECASE,
            )
        }
        slide_titles = (
            self._slide_title_ids(pages, headings)
            if profile == DocumentProfile.SLIDES
            else set()
        )
        local_stat_pages = {
            element.page_id
            for element in headings
            if self._LOCAL_STAT.match(
                self._plain_text(element.text or "")
            )
        }
        previous_by_id: dict[str, Element | None] = {}
        for page in pages:
            page_headings = sorted(
                (
                    element
                    for element in elements
                    if element.page_id == page.page_id
                ),
                key=lambda item: item.reading_order,
            )
            previous: Element | None = None
            for item in page_headings:
                previous_by_id[item.element_id] = previous
                previous = item

        decisions: list[HeadingEligibilityDecision] = []
        for element in headings:
            text = self._plain_text(element.text or "")
            normalized = self._normalized(text)
            position = (
                page_by_id[element.page_id].page_index,
                element.reading_order,
            )
            reason = "eligible_heading"
            confidence = 0.72
            eligible = True
            evidence: dict[str, object] = {"profile": profile.value}

            if element.metadata.get("excluded_from_heading_hierarchy") is True:
                eligible = False
                reason = "page_header_or_footer"
                confidence = 0.99
            elif not text:
                eligible = False
                reason = "empty_heading"
                confidence = 1.0
            elif self._PHONE_OR_CONTACT.match(text):
                eligible = False
                reason = "contact_or_url"
                confidence = 0.99
            elif self._LOCAL_STAT.match(text):
                eligible = False
                reason = "local_statistic_or_card_code"
                confidence = 0.96
            elif (
                profile == DocumentProfile.BROCHURE
                and element.page_id in local_stat_pages
                and len(text.split()) <= 3
                and text.upper() == text
                and normalized not in self._KNOWN_GLOBAL
            ):
                eligible = False
                reason = "rank_card_local_label"
                confidence = 0.93
            elif self._PROCEDURE.match(text):
                eligible = False
                reason = "procedure_step"
                confidence = 0.98
            elif self._looks_like_signature(text):
                eligible = False
                reason = "signature_or_salutation"
                confidence = 0.95
            elif (
                profile == DocumentProfile.REPORT
                and re.match(r"^\s*\d+\.\s+\S", text)
                and not re.match(r"^\s*\d+\.\d+", text)
                and (
                    len(text.split()) >= 8
                    or text.rstrip().endswith((".", "?", ";"))
                )
            ):
                eligible = False
                reason = "numbered_report_list_item"
                confidence = 0.93
            elif (
                profile == DocumentProfile.REPORT
                and (
                    re.fullmatch(
                        r"(?:address for correspondence|.+\s+plant)",
                        text,
                        re.IGNORECASE,
                    )
                    or (
                        text.rstrip().endswith(":")
                        and len(text.split()) >= 5
                        and re.match(r"^\s*(?:the|these|following)\b", text, re.IGNORECASE)
                    )
                )
            ):
                eligible = False
                reason = "report_local_label_not_section"
                confidence = 0.91
            elif (
                profile == DocumentProfile.REPORT
                and re.search(
                    r"\b(?:balance sheet|profit and loss(?: account)?|"
                    r"income statement|cash flow statement|"
                    r"statement of cash flows)\b",
                    text,
                    re.IGNORECASE,
                )
            ):
                element.metadata["forced_heading_level"] = 1
                reason = "financial_statement_sibling"
                confidence = 0.96
            elif (
                profile == DocumentProfile.BROCHURE
                and text.rstrip().endswith(":")
                and normalized not in self._KNOWN_GLOBAL
                and not self._STRUCTURAL_PREFIX.match(text)
            ):
                eligible = False
                reason = "brochure_local_sentence_label"
                confidence = 0.90
            elif (
                profile == DocumentProfile.BROCHURE
                and re.match(
                    r"^\s*for(?:\s|(?=[A-Z]))",
                    text,
                )
                and len(text.split()) >= 2
            ):
                eligible = False
                reason = "brochure_audience_sentence"
                confidence = 0.97
            elif (
                self._SENTENCE_PREFIX.match(text)
                and len(text.split()) >= 4
            ):
                eligible = False
                reason = "functional_sentence_not_section"
                confidence = 0.95
            elif (
                normalized in {"note", "notes"}
                and previous_by_id.get(element.element_id) is not None
                and previous_by_id[element.element_id].element_type
                in {ElementType.TABLE, ElementType.CAPTION, ElementType.FOOTNOTE}
            ):
                eligible = False
                reason = "table_notes_label"
                confidence = 0.96
            elif (
                normalized not in self._KNOWN_GLOBAL
                and len(text.split()) <= 4
                and len(normalized_pages[normalized]) >= 3
                and not self._STRUCTURAL_PREFIX.match(text)
            ):
                eligible = False
                reason = "repeated_local_label"
                confidence = 0.94
                evidence["distinct_page_count"] = len(
                    normalized_pages[normalized]
                )
                evidence["occurrence_count"] = normalized_counts[normalized]
            elif profile == DocumentProfile.SLIDES:
                if element.element_id in slide_titles:
                    element.metadata["forced_heading_level"] = 1
                    reason = "slide_title_sibling"
                    confidence = 0.94
                elif not self._STRUCTURAL_PREFIX.match(text):
                    eligible = False
                    reason = "slide_local_visual_label"
                    confidence = 0.88
            elif profile == DocumentProfile.BROCHURE:
                pass
            elif profile == DocumentProfile.MANUAL:
                if (
                    first_chapter_position is not None
                    and position < first_chapter_position
                    and page_by_id[element.page_id].page_index > 0
                ):
                    eligible = False
                    reason = "manual_front_matter_label"
                    confidence = 0.91
                elif (
                    element.page_id in contents_pages
                    and normalized not in {"contents", "table of contents"}
                ):
                    eligible = False
                    reason = "table_of_contents_entry"
                    confidence = 0.98
                elif (
                    index_position is not None
                    and position > index_position
                    and re.fullmatch(r"[A-Z]", text)
                ):
                    eligible = False
                    reason = "index_letter"
                    confidence = 0.99

            element.metadata["heading_eligibility"] = {
                "eligible": eligible,
                "reason": reason,
                "confidence": confidence,
                "profile": profile.value,
                "evidence": evidence,
            }
            if not eligible:
                element.metadata["excluded_from_heading_hierarchy"] = True
                element.metadata["excluded_from_section_hierarchy"] = True
                element.heading_level = None
            decisions.append(
                HeadingEligibilityDecision(
                    element_id=element.element_id,
                    page_id=element.page_id,
                    page_number=element.page_number,
                    eligible=eligible,
                    reason=reason,
                    confidence=confidence,
                    evidence=evidence,
                )
            )
        return decisions

    def _detect_with_trusted_parser_types(
        self,
        pages: list[Page],
        elements: list[Element],
        profile: DocumentProfile,
    ) -> list[HeadingEligibilityDecision]:
        """Use a small, parser-assisted policy for MinerU Hybrid output.

        Hybrid already performs stronger title/list/layout classification, so
        the post-processor only guards against document furniture, empty text,
        contact strings, and obvious sentence blocks.  It deliberately avoids
        brochure, finance, form, and named-document special cases.
        """

        page_by_id = {page.page_id: page for page in pages}
        headings = sorted(
            (
                element
                for element in elements
                if element.element_type == ElementType.HEADING
            ),
            key=lambda element: (
                page_by_id[element.page_id].page_index,
                element.reading_order,
                element.element_id,
            ),
        )
        primary_slide_titles: set[str] = set()
        if profile == DocumentProfile.SLIDES:
            for page in pages:
                candidates = [
                    element
                    for element in headings
                    if element.page_id == page.page_id
                    and element.bbox is not None
                    and element.bbox.normalized[1] <= 0.35
                ]
                if candidates:
                    primary_slide_titles.add(
                        min(
                            candidates,
                            key=lambda item: (
                                item.bbox.normalized[1],
                                -item.bbox.height,
                                item.reading_order,
                            ),
                        ).element_id
                    )

        decisions: list[HeadingEligibilityDecision] = []
        for element in headings:
            previous = element.metadata.get("heading_eligibility")
            if isinstance(previous, dict):
                element.metadata.pop("forced_heading_level", None)
                if element.metadata.get("repeated_region") is None:
                    element.metadata.pop("excluded_from_heading_hierarchy", None)
                    element.metadata.pop("excluded_from_section_hierarchy", None)

            text = self._plain_text(element.text or "")
            word_count = len(text.split())
            eligible = True
            reason = "trusted_parser_title"
            confidence = 0.90
            if element.metadata.get("repeated_region") is not None:
                eligible = False
                reason = "confirmed_repeated_page_furniture"
                confidence = 0.99
            elif not text:
                eligible = False
                reason = "empty_heading"
                confidence = 1.0
            elif self._PHONE_OR_CONTACT.match(text):
                eligible = False
                reason = "contact_or_url"
                confidence = 0.99
            elif word_count >= 18 or (
                word_count >= 9 and text.rstrip().endswith((".", ";"))
            ):
                eligible = False
                reason = "sentence_block_not_heading"
                confidence = 0.94
            elif element.element_id in primary_slide_titles:
                element.metadata["forced_heading_level"] = 1
                reason = "landscape_page_primary_title"
                confidence = 0.94

            evidence: dict[str, object] = {
                "policy": "trusted_parser_minimal_v1",
                "profile_signal": profile.value,
                "parser_backend": element.metadata.get("parser_backend"),
                "word_count": word_count,
            }
            element.metadata["heading_eligibility"] = {
                "eligible": eligible,
                "reason": reason,
                "confidence": confidence,
                "profile": profile.value,
                "evidence": evidence,
            }
            if not eligible:
                element.metadata["excluded_from_heading_hierarchy"] = True
                element.metadata["excluded_from_section_hierarchy"] = True
                element.heading_level = None
            decisions.append(
                HeadingEligibilityDecision(
                    element_id=element.element_id,
                    page_id=element.page_id,
                    page_number=element.page_number,
                    eligible=eligible,
                    reason=reason,
                    confidence=confidence,
                    evidence=evidence,
                )
            )
        return decisions

    def _slide_title_ids(
        self,
        pages: list[Page],
        headings: list[Element],
    ) -> set[str]:
        by_page: dict[str, list[Element]] = defaultdict(list)
        divider_by_page: dict[str, list[Element]] = defaultdict(list)
        for element in headings:
            if (
                element.metadata.get("semantic_marginal_override")
                == "slide_title"
            ):
                by_page[element.page_id].append(element)
                continue
            if (
                element.bbox is not None
                and element.bbox.normalized[1] <= 0.38
                and not self._PHONE_OR_CONTACT.match(
                    self._plain_text(element.text or "")
                )
            ):
                by_page[element.page_id].append(element)
            elif (
                element.bbox is not None
                and 0.38 <= element.bbox.normalized[1] <= 0.65
                and element.bbox.height >= 0.035
                and 1 <= len(self._plain_text(element.text or "").split())
                <= 14
                and self._plain_text(element.text or "").upper()
                == self._plain_text(element.text or "")
            ):
                divider_by_page[element.page_id].append(element)
        selected: set[str] = set()
        for page in pages:
            candidates = by_page.get(page.page_id, [])
            if not candidates:
                candidates = divider_by_page.get(page.page_id, [])
            if not candidates:
                continue
            best = max(
                candidates,
                key=lambda item: (
                    item.bbox.height if item.bbox else 0.0,
                    item.bbox.width if item.bbox else 0.0,
                    -item.reading_order,
                ),
            )
            if best.bbox and (
                best.bbox.height >= 0.025
                or best.bbox.width >= 0.30
                or best.reading_order <= 1
            ):
                selected.add(best.element_id)
        return selected

    @staticmethod
    def _first_matching_position(
        headings: list[Element],
        pattern: str,
        pages: dict[str, Page],
    ) -> tuple[int, int] | None:
        matcher = re.compile(pattern, re.IGNORECASE)
        for element in headings:
            if matcher.match(HeadingEligibilityDetector._plain_text(element.text or "")):
                return pages[element.page_id].page_index, element.reading_order
        return None

    @staticmethod
    def _looks_like_signature(text: str) -> bool:
        normalized = text.casefold().strip()
        if normalized in {"sincerely", "yours sincerely", "respectfully submitted"}:
            return True
        return bool(
            re.match(
                r"^(?:signed|signature|to the (?:members|stockholders)|"
                r"copyright\b)",
                normalized,
            )
            or re.match(r"^/s/\s+", normalized)
            or re.match(r"^[\u2013\u2014-]\s*[a-z][a-z .'-]{2,40}$", normalized)
            or (
                len(text.split()) <= 8
                and re.search(
                    r"\b(?:chief|president|secretary|director|officer)\b",
                    normalized,
                )
            )
        )

    @staticmethod
    def _plain_text(text: str) -> str:
        return " ".join(re.sub(r"<[^>]+>", " ", text).split())

    @classmethod
    def _normalized(cls, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", cls._plain_text(text).casefold()).strip()
