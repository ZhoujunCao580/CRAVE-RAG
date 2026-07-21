"""On-demand bbox navigation; no spatial relation edges are materialized."""

from __future__ import annotations

import math

from softdoc.models import Document, Element, Page


class SpatialNavigator:
    def __init__(self, document: Document):
        self.document = document
        self.elements = {element.element_id: element for element in document.elements}
        self.pages = {page.page_id: page for page in document.pages}

    def get_nearby_elements(self, element_id: str, max_distance: float) -> list[Element]:
        source = self._element_with_bbox(element_id)
        candidates = [
            element
            for element in self.document.elements
            if element.element_id != source.element_id
            and element.page_id == source.page_id
            and element.bbox is not None
            and _rectangle_distance(source, element) <= max_distance
        ]
        return sorted(candidates, key=lambda element: (_rectangle_distance(source, element), element.reading_order))

    def get_elements_above(self, element_id: str) -> list[Element]:
        source = self._element_with_bbox(element_id)
        source_y1 = source.bbox.normalized[1]
        candidates = self._same_page_boxed(source)
        above = [element for element in candidates if element.bbox.normalized[3] <= source_y1]
        return sorted(above, key=lambda element: (source_y1 - element.bbox.normalized[3], element.reading_order))

    def get_elements_below(self, element_id: str) -> list[Element]:
        source = self._element_with_bbox(element_id)
        source_y2 = source.bbox.normalized[3]
        candidates = self._same_page_boxed(source)
        below = [element for element in candidates if element.bbox.normalized[1] >= source_y2]
        return sorted(below, key=lambda element: (element.bbox.normalized[1] - source_y2, element.reading_order))

    def get_same_column_elements(self, element_id: str) -> list[Element]:
        source = self._element_with_bbox(element_id)
        result: list[Element] = []
        for element in self._same_page_boxed(source):
            if source.column_index is not None and element.column_index is not None:
                if source.column_index == element.column_index:
                    result.append(element)
                continue
            if _horizontal_overlap_ratio(source, element) >= 0.5:
                result.append(element)
        return sorted(result, key=lambda element: element.reading_order)

    def get_overlapping_elements(self, element_id: str) -> list[Element]:
        source = self._element_with_bbox(element_id)
        return sorted(
            [element for element in self._same_page_boxed(source) if _overlaps(source, element)],
            key=lambda element: element.reading_order,
        )

    def get_adjacent_pages(self, page_id: str, radius: int = 1) -> list[Page]:
        if radius < 1:
            raise ValueError("radius must be at least 1")
        source = self.pages[page_id]
        return [
            page
            for page in sorted(self.document.pages, key=lambda item: item.page_index)
            if page.page_id != page_id and abs(page.page_index - source.page_index) <= radius
        ]

    def _element_with_bbox(self, element_id: str) -> Element:
        element = self.elements[element_id]
        if element.bbox is None:
            raise ValueError(f"Element has no bounding box: {element_id}")
        return element

    def _same_page_boxed(self, source: Element) -> list[Element]:
        return [
            element
            for element in self.document.elements
            if element.element_id != source.element_id
            and element.page_id == source.page_id
            and element.bbox is not None
        ]


def _rectangle_distance(source: Element, target: Element) -> float:
    sx1, sy1, sx2, sy2 = source.bbox.normalized
    tx1, ty1, tx2, ty2 = target.bbox.normalized
    dx = max(tx1 - sx2, sx1 - tx2, 0.0)
    dy = max(ty1 - sy2, sy1 - ty2, 0.0)
    return math.hypot(dx, dy)


def _horizontal_overlap_ratio(source: Element, target: Element) -> float:
    sx1, _, sx2, _ = source.bbox.normalized
    tx1, _, tx2, _ = target.bbox.normalized
    overlap = max(0.0, min(sx2, tx2) - max(sx1, tx1))
    minimum_width = min(sx2 - sx1, tx2 - tx1)
    return overlap / minimum_width if minimum_width > 0 else 0.0


def _overlaps(source: Element, target: Element) -> bool:
    sx1, sy1, sx2, sy2 = source.bbox.normalized
    tx1, ty1, tx2, ty2 = target.bbox.normalized
    return min(sx2, tx2) > max(sx1, tx1) and min(sy2, ty2) > max(sy1, ty1)

