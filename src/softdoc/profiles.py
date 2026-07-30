"""Deterministic, parser-neutral document structure profiles.

Profiles do not assign semantic roles to every element.  They only select a
small set of layout policies used by heading and relation post-processing.
"""

from __future__ import annotations

import re
from collections import Counter
from enum import Enum
from pathlib import Path

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
        evidence: dict[str, object] = {
            "landscape_page_ratio": round(landscape_ratio, 4),
            "table_element_ratio": round(table_ratio, 4),
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
            or brochure_term_count >= 6
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

        if landscape_ratio >= 0.60 and page_count >= 4:
            return DocumentProfileDecision(
                profile=DocumentProfile.SLIDES,
                confidence=0.98,
                evidence={**evidence, "rule": "predominantly_landscape_pages"},
            )

        chapter_count = len(
            re.findall(r"(?im)^\s*chapter\s+\d+\b", heading_text)
        )
        has_contents = bool(
            re.search(r"(?im)^\s*(?:table of )?contents\s*$", all_text)
        )
        has_index = bool(re.search(r"(?im)^\s*index\s*$", heading_text))
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
        if (
            brochure_terms
            and not academic_outline_present
            and (
                repeated_short_headings >= 2
                or brochure_term_count >= 6
            )
        ):
            return DocumentProfileDecision(
                profile=DocumentProfile.BROCHURE,
                confidence=0.88,
                evidence={
                    **evidence,
                    "rule": "brochure_vocabulary_and_card_layout",
                    "matched_terms": sorted(brochure_terms),
                    "repeated_short_heading_count": repeated_short_headings,
                },
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
