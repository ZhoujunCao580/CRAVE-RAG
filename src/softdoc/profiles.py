"""Deterministic, parser-neutral document structure profiles.

Profiles do not assign semantic roles to every element.  They only select a
small set of layout policies used by heading and relation post-processing.
"""

from __future__ import annotations

import re
from collections import Counter
from enum import Enum
from html import unescape
from math import ceil
from pathlib import Path
from statistics import median

from pydantic import Field

from softdoc.models import Element, ElementType, Page, SoftDocModel


class DocumentProfile(str, Enum):
    ACADEMIC = "academic"
    SLIDES = "slides"
    MANUAL = "manual"
    FORM = "form"
    BROCHURE = "brochure"
    REPORT = "report"


class DocumentProfileDecision(SoftDocModel):
    profile: DocumentProfile
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: dict[str, object] = Field(default_factory=dict)


class DocumentProfileDetector:
    """Choose one of a few reusable structure policies from visible evidence."""

    SLIDE_LANDSCAPE_RATIO = 0.80
    SLIDE_SPARSE_PAGE_RATIO = 0.70
    SLIDE_MEDIAN_TEXT_CHARS_MAX = 1200
    SLIDE_MEDIAN_ELEMENT_COUNT_MAX = 18
    SLIDE_SPARSE_PAGE_TEXT_CHARS_MAX = 1600
    SLIDE_SPARSE_PAGE_ELEMENT_COUNT_MAX = 24
    SLIDE_REPEATED_PATTERN_RATIO = 0.25

    def detect(
        self,
        pages: list[Page],
        elements: list[Element],
        *,
        source_path: Path | None = None,
    ) -> DocumentProfileDecision:
        headings = [
            element
            for element in elements
            if element.element_type == ElementType.HEADING
            and (element.text or "").strip()
        ]
        heading_text = "\n".join(element.text or "" for element in headings)
        all_text = "\n".join(
            element.text or ""
            for element in elements
            if element.text
        )
        page_count = max(1, len(pages))
        landscape_pages = sum(
            page.width >= page.height * 1.08 for page in pages
        )
        landscape_ratio = landscape_pages / page_count
        table_ratio = sum(
            element.element_type == ElementType.TABLE for element in elements
        ) / max(1, len(elements))
        page_text_chars, page_element_counts = self._page_density_metrics(
            pages,
            elements,
        )
        median_text_chars = float(median(page_text_chars)) if page_text_chars else 0.0
        median_element_count = (
            float(median(page_element_counts)) if page_element_counts else 0.0
        )
        sparse_page_count = sum(
            text_chars <= self.SLIDE_SPARSE_PAGE_TEXT_CHARS_MAX
            and element_count <= self.SLIDE_SPARSE_PAGE_ELEMENT_COUNT_MAX
            for text_chars, element_count in zip(
                page_text_chars,
                page_element_counts,
                strict=True,
            )
        )
        sparse_page_ratio = sparse_page_count / page_count
        repeated_pattern_ratio, repeated_pattern = self._repeated_page_pattern(
            pages,
            elements,
        )
        evidence: dict[str, object] = {
            "landscape_page_ratio": round(landscape_ratio, 4),
            "table_element_ratio": round(table_ratio, 4),
            "sparse_page_ratio": round(sparse_page_ratio, 4),
            "median_text_chars_per_page": round(median_text_chars, 2),
            "median_element_count_per_page": round(median_element_count, 2),
            "repeated_page_pattern_ratio": round(repeated_pattern_ratio, 4),
            "repeated_page_pattern": repeated_pattern,
            "page_count": len(pages),
            "heading_count": len(headings),
            "source_name": source_path.name if source_path else None,
        }
        normalized_all = re.sub(r"\s+", " ", all_text).casefold()
        source_name = source_path.name.casefold() if source_path else ""
        brochure_term_count = sum(
            len(re.findall(rf"\b{re.escape(term)}\b", normalized_all))
            for term in (
                "prospectus",
                "programme",
                "programmes",
                "admissions",
            )
        )
        cover_text = " ".join(
            element.text or ""
            for element in elements
            if pages
            and element.page_id == pages[0].page_id
            and element.text
        ).casefold()
        strong_brochure_identity = bool(
            "brochure" in source_name
            or "prospectus" in source_name
            or "prospectus" in cover_text
            or "graduate studies" in cover_text
        )
        chapter_count = len(
            re.findall(r"(?im)^\s*chapter\s+\d+\b", heading_text)
        )
        has_contents = bool(
            re.search(r"(?im)^\s*(?:table of )?contents\s*$", all_text)
        )
        has_index = bool(re.search(r"(?im)^\s*index\s*$", heading_text))
        section_labels = len(
            re.findall(
                r"(?im)\bsection\s+[A-Z0-9]+(?:\.[0-9]+)*\s*:",
                all_text,
            )
        )
        full_page_tables = sum(
            element.element_type == ElementType.TABLE
            and element.bbox is not None
            and element.bbox.width >= 0.70
            and element.bbox.height >= 0.55
            for element in elements
        )
        brochure_terms = Counter(
            term
            for term in (
                "prospectus",
                "programme",
                "programmes",
                "graduate studies",
                "admissions",
            )
            if term in normalized_all
        )
        repeated_short_headings = self._repeated_short_heading_count(headings)
        academic_outline_present = bool(
            re.search(r"(?im)^\s*abstract\s*$", heading_text)
            or re.search(r"(?im)^\s*references\s*$", heading_text)
        )
        has_abstract = bool(
            re.search(r"(?im)^\s*abstract\s*$", heading_text)
        )
        has_references = bool(
            re.search(r"(?im)^\s*references\s*$", heading_text)
        )
        numbered_headings = len(
            re.findall(
                r"(?im)^\s*\d+(?:\.\d+)*\.?\s+\S",
                heading_text,
            )
        )
        if has_abstract or (has_references and numbered_headings >= 2):
            return DocumentProfileDecision(
                profile=DocumentProfile.ACADEMIC,
                confidence=0.94,
                evidence={
                    **evidence,
                    "rule": "academic_outline_signals",
                    "has_abstract": has_abstract,
                    "has_references": has_references,
                    "numbered_heading_count": numbered_headings,
                },
            )

        if chapter_count >= 2 or (has_contents and has_index):
            return DocumentProfileDecision(
                profile=DocumentProfile.MANUAL,
                confidence=0.96,
                evidence={
                    **evidence,
                    "rule": "chapter_contents_index_structure",
                    "chapter_heading_count": chapter_count,
                    "has_contents": has_contents,
                    "has_index": has_index,
                },
            )

        if section_labels >= 3 and (table_ratio >= 0.08 or full_page_tables >= 3):
            return DocumentProfileDecision(
                profile=DocumentProfile.FORM,
                confidence=0.94,
                evidence={
                    **evidence,
                    "rule": "section_labels_with_dense_tables",
                    "section_label_count": section_labels,
                    "full_page_table_count": full_page_tables,
                },
            )

        if strong_brochure_identity:
            return DocumentProfileDecision(
                profile=DocumentProfile.BROCHURE,
                confidence=0.95,
                evidence={
                    **evidence,
                    "rule": "strong_brochure_identity",
                    "brochure_term_count": brochure_term_count,
                    "source_name_signal": (
                        "brochure" in source_name
                        or "prospectus" in source_name
                    ),
                },
            )

        if (
            brochure_terms
            and not academic_outline_present
            and brochure_term_count >= 6
            and repeated_short_headings >= 2
            and sparse_page_ratio >= 0.40
        ):
            return DocumentProfileDecision(
                profile=DocumentProfile.BROCHURE,
                confidence=0.88,
                evidence={
                    **evidence,
                    "rule": "brochure_vocabulary_and_card_layout",
                    "matched_terms": sorted(brochure_terms),
                    "brochure_term_count": brochure_term_count,
                    "repeated_short_heading_count": repeated_short_headings,
                    "minimum_sparse_page_ratio": 0.40,
                },
            )

        is_sparse_landscape_slides = bool(
            page_count >= 4
            and landscape_ratio >= self.SLIDE_LANDSCAPE_RATIO
            and sparse_page_ratio >= self.SLIDE_SPARSE_PAGE_RATIO
            and median_text_chars <= self.SLIDE_MEDIAN_TEXT_CHARS_MAX
            and median_element_count <= self.SLIDE_MEDIAN_ELEMENT_COUNT_MAX
            and repeated_pattern_ratio >= self.SLIDE_REPEATED_PATTERN_RATIO
        )
        if is_sparse_landscape_slides:
            return DocumentProfileDecision(
                profile=DocumentProfile.SLIDES,
                confidence=0.94,
                evidence={
                    **evidence,
                    "rule": "sparse_landscape_repeated_page_pattern",
                    "thresholds": {
                        "minimum_landscape_page_ratio": self.SLIDE_LANDSCAPE_RATIO,
                        "minimum_sparse_page_ratio": self.SLIDE_SPARSE_PAGE_RATIO,
                        "maximum_median_text_chars_per_page": (
                            self.SLIDE_MEDIAN_TEXT_CHARS_MAX
                        ),
                        "maximum_median_element_count_per_page": (
                            self.SLIDE_MEDIAN_ELEMENT_COUNT_MAX
                        ),
                        "minimum_repeated_page_pattern_ratio": (
                            self.SLIDE_REPEATED_PATTERN_RATIO
                        ),
                    },
                },
            )

        return DocumentProfileDecision(
            profile=DocumentProfile.REPORT,
            confidence=0.70,
            evidence={**evidence, "rule": "conservative_report_fallback"},
        )

    @staticmethod
    def _repeated_short_heading_count(headings: list[Element]) -> int:
        counts = Counter(
            re.sub(r"\s+", " ", (element.text or "")).strip().casefold()
            for element in headings
            if 1 <= len((element.text or "").split()) <= 5
        )
        return sum(count >= 3 for count in counts.values())

    @staticmethod
    def _page_density_metrics(
        pages: list[Page],
        elements: list[Element],
    ) -> tuple[list[int], list[int]]:
        elements_by_page: dict[str, list[Element]] = {
            page.page_id: [] for page in pages
        }
        for element in elements:
            if element.page_id in elements_by_page:
                elements_by_page[element.page_id].append(element)

        page_text_chars: list[int] = []
        page_element_counts: list[int] = []
        for page in pages:
            page_elements = elements_by_page[page.page_id]
            visible_text = " ".join(
                DocumentProfileDetector._visible_text(element)
                for element in page_elements
            )
            page_text_chars.append(len(re.sub(r"\s+", "", visible_text)))
            page_element_counts.append(len(page_elements))
        return page_text_chars, page_element_counts

    @staticmethod
    def _visible_text(element: Element) -> str:
        text = element.text or ""
        if element.html:
            text = f"{text} {re.sub(r'<[^>]+>', ' ', unescape(element.html))}"
        return text

    @classmethod
    def _repeated_page_pattern(
        cls,
        pages: list[Page],
        elements: list[Element],
    ) -> tuple[float, str | None]:
        if not pages:
            return 0.0, None

        signatures_by_page: dict[str, set[str]] = {
            page.page_id: set() for page in pages
        }
        for element in elements:
            if element.page_id not in signatures_by_page or element.bbox is None:
                continue
            x1, y1, x2, y2 = element.bbox.normalized
            normalized_text = cls._pattern_text(element.text or "")
            word_count = len((element.text or "").split())
            if (
                normalized_text
                and word_count <= 12
                and (y2 <= 0.15 or y1 >= 0.85)
            ):
                band = "header" if y2 <= 0.15 else "footer"
                signatures_by_page[element.page_id].add(
                    f"{band}-text:{normalized_text}"
                )
            if element.element_type == ElementType.HEADING and y1 <= 0.30:
                title_box = (
                    round(x1 * 10),
                    round(y1 * 10),
                    round(x2 * 10),
                    round(y2 * 10),
                )
                signatures_by_page[element.page_id].add(
                    "title-layout:" + ":".join(str(value) for value in title_box)
                )

        page_counts: Counter[str] = Counter()
        for signatures in signatures_by_page.values():
            page_counts.update(signatures)
        if not page_counts:
            return 0.0, None

        signature, count = page_counts.most_common(1)[0]
        required_count = max(2, ceil(len(pages) * cls.SLIDE_REPEATED_PATTERN_RATIO))
        if count < required_count:
            return count / len(pages), None
        return count / len(pages), signature.split(":", maxsplit=1)[0]

    @staticmethod
    def _pattern_text(text: str) -> str:
        normalized = re.sub(r"\d+", "#", text.casefold())
        normalized = re.sub(r"[^\w#]+", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()
