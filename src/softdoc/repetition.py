"""Detect repeated page headers and footers from parser-neutral page geometry."""

from __future__ import annotations

import re
from collections import defaultdict
from enum import Enum
from typing import Any

from pydantic import Field

from softdoc.models import Element, ElementType, Page, SoftDocModel


_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


class RepeatedRegion(str, Enum):
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"


class RepeatedRegionDecision(SoftDocModel):
    element_id: str
    page_id: str
    page_number: int
    region: RepeatedRegion
    normalized_text: str
    occurrence_count: int = Field(ge=1)
    distinct_page_count: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: dict[str, Any] = Field(default_factory=dict)


class RepeatedHeaderFooterResult(SoftDocModel):
    decisions: list[RepeatedRegionDecision] = Field(default_factory=list)


class RepeatedHeaderFooterDetector:
    """Mark repeated marginal text while preserving the original Element."""

    _ELIGIBLE_TYPES = {ElementType.HEADING, ElementType.PARAGRAPH}

    def __init__(
        self,
        *,
        top_boundary: float = 0.13,
        bottom_boundary: float = 0.87,
        minimum_pages: int = 3,
    ) -> None:
        self.top_boundary = top_boundary
        self.bottom_boundary = bottom_boundary
        self.minimum_pages = max(2, minimum_pages)

    def detect(
        self,
        pages: list[Page],
        elements: list[Element],
    ) -> RepeatedHeaderFooterResult:
        page_count = len(pages)
        groups: dict[tuple[RepeatedRegion, str], list[Element]] = defaultdict(list)
        decisions: list[RepeatedRegionDecision] = []
        parser_votes: dict[tuple[RepeatedRegion, str], list[Element]] = (
            defaultdict(list)
        )

        for element in elements:
            element.metadata.pop("repeated_region", None)
            element.metadata.pop("repeated_region_evidence", None)
            element.metadata.pop("excluded_from_heading_hierarchy", None)
            element.metadata.pop("excluded_from_section_hierarchy", None)
            if element.element_type not in self._ELIGIBLE_TYPES:
                continue
            normalized = self._normalized_text(element.text or "").casefold()
            parser_region = self._validated_parser_region(element)
            if normalized and parser_region is not None:
                parser_votes[(parser_region, normalized)].append(element)

        threshold = min(
            self.minimum_pages,
            max(2, page_count),
        )
        parser_confirmed_ids: set[str] = set()
        for (region, normalized), occurrences in parser_votes.items():
            distinct_pages = {item.page_id for item in occurrences}
            if len(distinct_pages) < threshold:
                continue
            if not self._positions_are_similar(occurrences, region):
                continue
            evidence = {
                "rule": "repeated_parser_marginal_text",
                "normalized_text": normalized,
                "distinct_page_count": len(distinct_pages),
                "document_page_count": page_count,
                "parser_label_used_as_repeated_signal": True,
            }
            confidence = min(
                0.99,
                0.90 + len(distinct_pages) / max(1, page_count) * 0.08,
            )
            for element in occurrences:
                self._mark(element, region, evidence)
                parser_confirmed_ids.add(element.element_id)
                decisions.append(
                    RepeatedRegionDecision(
                        element_id=element.element_id,
                        page_id=element.page_id,
                        page_number=element.page_number,
                        region=region,
                        normalized_text=normalized,
                        occurrence_count=len(occurrences),
                        distinct_page_count=len(distinct_pages),
                        confidence=confidence,
                        evidence=evidence,
                    )
                )

        for element in elements:
            if element.element_type not in self._ELIGIBLE_TYPES:
                continue
            if element.element_id in parser_confirmed_ids:
                continue
            text = self._normalized_text(element.text or "")
            parser_region = self._validated_parser_region(element)
            region = self._marginal_region(element)
            if not text or region is None:
                continue
            # A parser's ``header``/``footer`` label is only a vote.  A unique
            # marginal title (for example a paper title or a chapter opener)
            # must remain readable and eligible for hierarchy construction.
            # We confirm running furniture only from document-level repetition
            # plus stable geometry below.
            if parser_region is not None and parser_region != region:
                continue
            groups[(region, text.casefold())].append(element)

        for (region, normalized), occurrences in groups.items():
            distinct_pages = {item.page_id for item in occurrences}
            if len(distinct_pages) < threshold:
                continue
            if not self._positions_are_similar(occurrences, region):
                continue
            # Large, repeated slide titles are meaningful siblings, not page
            # furniture. Marginal running text is normally shallow and small.
            median_height = sorted(
                item.bbox.height for item in occurrences if item.bbox is not None
            )[len(occurrences) // 2]
            if median_height > 0.065:
                continue
            frequency = len(distinct_pages) / max(1, page_count)
            confidence = min(0.99, 0.76 + min(0.18, frequency * 0.20))
            parser_declared = parser_votes.get((region, normalized), [])
            evidence = {
                "rule": "repeated_marginal_text",
                "normalized_text": normalized,
                "distinct_page_count": len(distinct_pages),
                "document_page_count": page_count,
                "page_frequency": round(frequency, 4),
                "median_bbox_height": round(median_height, 6),
                "parser_declared_count": len(parser_declared),
                "parser_label_used_as_signal_only": True,
                "region_boundary": (
                    self.top_boundary
                    if region == RepeatedRegion.PAGE_HEADER
                    else self.bottom_boundary
                ),
            }
            for element in occurrences:
                self._mark(element, region, evidence)
                decisions.append(
                    RepeatedRegionDecision(
                        element_id=element.element_id,
                        page_id=element.page_id,
                        page_number=element.page_number,
                        region=region,
                        normalized_text=normalized,
                        occurrence_count=len(occurrences),
                        distinct_page_count=len(distinct_pages),
                        confidence=confidence,
                        evidence=evidence,
                    )
                )
        return RepeatedHeaderFooterResult(decisions=decisions)

    @staticmethod
    def _parser_declared_region(element: Element) -> RepeatedRegion | None:
        if element.metadata.get("semantic_marginal_override") == "slide_title":
            return None
        parser_type = str(element.metadata.get("mineru_type") or "").casefold()
        if parser_type in {"page_header", "header"}:
            return RepeatedRegion.PAGE_HEADER
        if parser_type in {"page_footer", "footer"}:
            return RepeatedRegion.PAGE_FOOTER
        return None

    def _validated_parser_region(
        self,
        element: Element,
    ) -> RepeatedRegion | None:
        region = self._parser_declared_region(element)
        if region is None or element.bbox is None:
            return region
        return region if self._marginal_region(element) == region else None

    def _marginal_region(self, element: Element) -> RepeatedRegion | None:
        if element.bbox is None:
            return None
        _, y1, _, y2 = element.bbox.normalized
        if y1 <= self.top_boundary:
            return RepeatedRegion.PAGE_HEADER
        if y2 >= self.bottom_boundary:
            return RepeatedRegion.PAGE_FOOTER
        return None

    @staticmethod
    def _positions_are_similar(
        elements: list[Element],
        region: RepeatedRegion,
    ) -> bool:
        boxes = [item.bbox.normalized for item in elements if item.bbox is not None]
        if len(boxes) != len(elements):
            return False
        vertical_centers = [(box[1] + box[3]) / 2 for box in boxes]
        horizontal_centers = [(box[0] + box[2]) / 2 for box in boxes]
        vertical_spread = max(vertical_centers) - min(vertical_centers)
        horizontal_spread = max(horizontal_centers) - min(horizontal_centers)
        vertical_limit = 0.035 if region == RepeatedRegion.PAGE_HEADER else 0.045
        return vertical_spread <= vertical_limit and horizontal_spread <= 0.12

    @staticmethod
    def _mark(
        element: Element,
        region: RepeatedRegion,
        evidence: dict[str, Any],
    ) -> None:
        element.metadata["repeated_region"] = region.value
        element.metadata["repeated_region_evidence"] = evidence
        element.metadata["excluded_from_heading_hierarchy"] = True
        element.metadata["excluded_from_section_hierarchy"] = True
        element.heading_level = None

    @staticmethod
    def _normalized_text(text: str) -> str:
        return _WHITESPACE.sub(" ", _HTML_TAG.sub("", text)).strip()
