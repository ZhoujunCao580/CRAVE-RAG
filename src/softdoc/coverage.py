"""Recover text that a layout parser missed from a PDF's native text layer.

This is deliberately an opt-in post-processing step.  It never attempts OCR and
only adds text that is present in the source PDF but is not already represented
by the parser-neutral document.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from html import unescape
from pathlib import Path
import re
from typing import Iterable

import pypdfium2 as pdfium

from softdoc.ids import bbox_id, element_id, provenance_id
from softdoc.models import (
    BoundingBox,
    Document,
    Element,
    ElementParseStatus,
    ElementType,
    Page,
    Provenance,
)


_ADAPTER_NAME = "pdf_text_layer_coverage"
_RECOVERY_ROLE = "text-layer-coverage"
_MIN_MEANINGFUL_CHARACTERS = 2
_READING_ROW_TOLERANCE = 0.012
_VISUAL_CONTAINER_TYPES = {
    ElementType.TABLE,
    ElementType.FIGURE,
    ElementType.CHART,
    ElementType.CODE,
    ElementType.ALGORITHM,
    ElementType.EQUATION,
}
_AUXILIARY_PAGE_TEXT = re.compile(
    r"^\s*(?:page\s+)?\d+\s*(?:of|/)\s*\d+\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PdfTextLine:
    """One visual line extracted from a PDF page's selectable text."""

    page_index: int
    line_index: int
    text: str
    # PDF coordinates, with (0, 0) at the lower-left corner.
    bbox: tuple[float, float, float, float]
    # (character offset in ``text``, PDF bbox).  Kept so partially covered
    # lines can be split without OCR or guessed geometry.
    character_boxes: tuple[tuple[int, tuple[float, float, float, float]], ...] = ()
    fragment_index: int = 0
    source_page_width: float | None = None
    source_page_height: float | None = None


@dataclass(frozen=True)
class CoverageRecoveryResult:
    """Summary of a :class:`PdfTextLayerCoverageRecovery` run."""

    scanned_line_count: int
    recovered_element_ids: tuple[str, ...]

    @property
    def recovered_count(self) -> int:
        return len(self.recovered_element_ids)


class PdfTextLayerCoverageRecovery:
    """Add uncovered native-PDF text to an existing :class:`Document`.

    ``source_pdf`` must be absolute.  Requiring an explicit source avoids using
    ``Document.source_path`` accidentally, which is commonly relative to an
    adapter artifact directory rather than the original PDF.
    """

    def __init__(self, source_pdf: str | Path) -> None:
        self.source_pdf = Path(source_pdf)
        if not self.source_pdf.is_absolute():
            raise ValueError("source_pdf must be an absolute path to the source PDF")

    def recover(self, document: Document) -> CoverageRecoveryResult:
        """Recover text-layer lines missing from ``document`` in place."""

        if not self.source_pdf.is_file():
            raise FileNotFoundError(f"Source PDF does not exist: {self.source_pdf}")

        lines_by_page = self._extract_lines()
        elements_by_page: dict[str, list[Element]] = defaultdict(list)
        for element in document.elements:
            elements_by_page[element.page_id].append(element)

        recovered: list[Element] = []
        scanned = 0
        for page in document.pages:
            page_lines = lines_by_page.get(page.page_index, ())
            page_elements = elements_by_page[page.page_id]
            for line in page_lines:
                if (
                    not _is_meaningful(line.text)
                    or _is_auxiliary_page_text(line.text)
                    or _is_low_information_value(line.text)
                    or _is_extreme_margin_line(line, page)
                ):
                    continue
                scanned += 1
                for fragment in _uncovered_fragments(line, page, page_elements):
                    if (
                        _is_low_information_value(fragment.text)
                        or _is_extreme_margin_line(fragment, page)
                    ):
                        continue
                    element = self._recovered_element(document, page, fragment, page_elements)
                    page_elements.append(element)
                    recovered.append(element)

        if recovered:
            document.elements.extend(recovered)
            self._refresh_page_orders(document)
            warnings = document.metadata.setdefault("coverage_recovery_warnings", [])
            if not isinstance(warnings, list):
                raise ValueError("document.metadata['coverage_recovery_warnings'] must be a list")
            warnings.extend(
                {
                    "code": "pdf_text_layer_uncovered",
                    "message": "Recovered selectable PDF text missing from parser output",
                    "payload": {
                        "element_id": element.element_id,
                        "page_number": element.page_number,
                        "source_pdf": self.source_pdf.as_posix(),
                    },
                }
                for element in recovered
            )

        return CoverageRecoveryResult(
            scanned_line_count=scanned,
            recovered_element_ids=tuple(element.element_id for element in recovered),
        )

    def _extract_lines(self) -> dict[int, list[PdfTextLine]]:
        """Extract visual lines and character-derived bboxes with PDFium."""

        result: dict[int, list[PdfTextLine]] = {}
        pdf = pdfium.PdfDocument(self.source_pdf)
        try:
            for page_index in range(len(pdf)):
                page = pdf[page_index]
                try:
                    textpage = page.get_textpage()
                    try:
                        result[page_index] = _lines_from_textpage(
                            page_index,
                            textpage,
                            source_page_width=float(page.get_width()),
                            source_page_height=float(page.get_height()),
                        )
                    finally:
                        textpage.close()
                finally:
                    page.close()
        finally:
            pdf.close()
        return result

    def _recovered_element(
        self,
        document: Document,
        page: Page,
        line: PdfTextLine,
        page_elements: list[Element],
    ) -> Element:
        # Reserve a high source index so parser-provided IDs (normally dense,
        # low indexes) remain untouched.  The PDF line index makes this stable
        # across repeated parses of the same source.
        source_index = 900_000 + line.line_index * 100 + line.fragment_index
        candidate_id = element_id(
            document.document_id,
            page.page_index,
            source_index,
            ElementType.PARAGRAPH.value,
            _RECOVERY_ROLE,
        )
        existing_ids = {element.element_id for element in page_elements}
        while candidate_id in existing_ids:
            source_index += 1
            candidate_id = element_id(
                document.document_id,
                page.page_index,
                source_index,
                ElementType.PARAGRAPH.value,
                _RECOVERY_ROLE,
            )

        x1, bottom, x2, top = line.bbox
        source_width, source_height = _source_page_size(line, page)
        raw_bbox = (
            x1 / source_width * page.width,
            (source_height - top) / source_height * page.height,
            x2 / source_width * page.width,
            (source_height - bottom) / source_height * page.height,
        )
        provenance_payload = {
            "source": "pdf_text_layer",
            "page_index": line.page_index,
            "line_index": line.line_index,
            "text": line.text,
            "pdf_bbox": line.bbox,
        }
        return Element(
            element_id=candidate_id,
            document_id=document.document_id,
            page_id=page.page_id,
            page_number=page.page_number,
            element_type=ElementType.PARAGRAPH,
            reading_order=0,
            bbox=BoundingBox.from_raw(
                bbox_id=bbox_id(candidate_id),
                raw=raw_bbox,
                page_width=page.width,
                page_height=page.height,
            ),
            column_index=_nearest_column(line, page, page_elements),
            text=line.text,
            parse_status=ElementParseStatus.DEGRADED,
            provenance=Provenance(
                provenance_id=provenance_id(
                    _ADAPTER_NAME, self.source_pdf.as_posix(), f"page[{line.page_index}]/line[{line.line_index}]"
                ),
                adapter=_ADAPTER_NAME,
                source_path=self.source_pdf,
                source_locator=f"page[{line.page_index}]/line[{line.line_index}]",
                raw_payload=provenance_payload,
                metadata={"recovery_warning": "uncovered_pdf_text_layer"},
            ),
            metadata={
                "coverage_recovery": {
                    "warning_code": "pdf_text_layer_uncovered",
                    "reason": "No normalized-text and spatial coverage match in parser output",
                    "source_pdf": self.source_pdf.as_posix(),
                    "pdf_bbox": line.bbox,
                }
            },
        )

    @staticmethod
    def _refresh_page_orders(document: Document) -> None:
        by_page: dict[str, list[Element]] = defaultdict(list)
        for element in document.elements:
            by_page[element.page_id].append(element)
        for page in document.pages:
            ordered = sorted(by_page[page.page_id], key=_reading_order_key)
            for index, element in enumerate(ordered):
                element.reading_order = index
            ids = [element.element_id for element in ordered]
            # ``Page`` validates that these two fields agree on every
            # assignment.  Set the mutually dependent pair atomically after
            # it has been derived from the same ordered element list.
            object.__setattr__(page, "element_ids", ids)
            object.__setattr__(page, "reading_order", ids)


def recover_pdf_text_layer_coverage(
    document: Document, source_pdf: str | Path
) -> CoverageRecoveryResult:
    """Convenience wrapper for :class:`PdfTextLayerCoverageRecovery`."""

    return PdfTextLayerCoverageRecovery(source_pdf).recover(document)


def _lines_from_textpage(
    page_index: int,
    textpage: object,
    *,
    source_page_width: float | None = None,
    source_page_height: float | None = None,
) -> list[PdfTextLine]:
    """Convert PDFium character data into text lines with unioned bboxes."""

    # get_text_range and get_charbox share character indexes in PDFium.
    text = textpage.get_text_range()
    lines: list[PdfTextLine] = []
    line_index = 0
    for match in re.finditer(r"[^\r\n]+", text):
        chars = _character_boxes(textpage, match.start(), match.end())
        if chars:
            boxes = [box for _, box in chars]
            x1 = min(box[0] for box in boxes)
            bottom = min(box[1] for box in boxes)
            x2 = max(box[2] for box in boxes)
            top = max(box[3] for box in boxes)
            if x1 < x2 and bottom < top:
                leading = len(match.group()) - len(match.group().lstrip())
                lines.append(
                    PdfTextLine(
                        page_index=page_index,
                        line_index=line_index,
                        text=match.group().strip(),
                        bbox=(x1, bottom, x2, top),
                        character_boxes=tuple(
                            (index - match.start() - leading, box)
                            for index, box in chars
                            if leading <= index - match.start() < len(match.group().rstrip())
                        ),
                        source_page_width=source_page_width,
                        source_page_height=source_page_height,
                    )
                )
                line_index += 1
    return lines


def _character_boxes(
    textpage: object, start: int, end: int
) -> list[tuple[int, tuple[float, float, float, float]]]:
    boxes: list[tuple[int, tuple[float, float, float, float]]] = []
    for index in range(start, end):
        try:
            left, bottom, right, top = textpage.get_charbox(index)
        except Exception:  # Some control / synthetic whitespace has no box.
            continue
        x1, x2 = sorted((float(left), float(right)))
        y1, y2 = sorted((float(bottom), float(top)))
        if x1 < x2 and y1 < y2:
            boxes.append((index, (x1, y1, x2, y2)))
    return boxes


def _is_meaningful(text: str) -> bool:
    return len(_normalized_text(text)) >= _MIN_MEANINGFUL_CHARACTERS


def _normalized_text(text: str | None) -> str:
    return "".join(character for character in (text or "").casefold() if character.isalnum())


def _represented_text(element: Element) -> str:
    """Return parser-owned text, including text retained only as table HTML."""

    values = [element.text or ""]
    if element.html:
        values.append(unescape(re.sub(r"<[^>]+>", " ", element.html)))
    return _normalized_text(" ".join(values))


def _is_auxiliary_page_text(text: str) -> bool:
    """Ignore bare pagination text that is intentionally not a content element."""

    return bool(_AUXILIARY_PAGE_TEXT.fullmatch(text.strip()))


def _is_low_information_value(text: str) -> bool:
    """Avoid manufacturing context-free answer tokens such as Yes or No."""

    return bool(re.fullmatch(r"\s*(?:yes|no|n/?a)\s*", text, re.IGNORECASE))


def _is_extreme_margin_line(line: PdfTextLine, page: Page) -> bool:
    """Leave marginal running text to header/footer and table-continuation logic."""

    _, bottom, _, top = line.bbox
    _, source_height = _source_page_size(line, page)
    raw_top = source_height - top
    raw_bottom = source_height - bottom
    return raw_top <= source_height * 0.025 or raw_bottom >= source_height * 0.985


def _owned_by_visual_container(
    line: PdfTextLine,
    page: Page,
    elements: Iterable[Element],
) -> bool:
    """Treat native text inside a parsed visual or structured block as represented.

    PDF text layers often expose every table cell or chart label separately.
    Re-emitting those strings as paragraphs would duplicate a block that the
    parser already preserved as HTML or a visual crop.
    """

    for element in elements:
        if element.element_type not in _VISUAL_CONTAINER_TYPES or element.bbox is None:
            continue
        if _spatial_overlap(line, page, element) >= 0.40:
            return True
    return False


def _is_covered(line: PdfTextLine, page: Page, elements: Iterable[Element]) -> bool:
    source_text = _normalized_text(line.text)
    if not source_text or _owned_by_visual_container(line, page, elements):
        return True
    for element in elements:
        element_text = _represented_text(element)
        if not element_text:
            continue
        if source_text in element_text:
            return True
    return False


def _uncovered_fragments(line: PdfTextLine, page: Page, elements: Iterable[Element]) -> list[PdfTextLine]:
    """Split a line around text that a nearby parser element already owns."""

    if _is_covered(line, page, elements):
        return []
    normalized, raw_indexes = _normalized_with_indexes(line.text)
    covered = [False] * len(normalized)
    for element in elements:
        element_text = _represented_text(element)
        if not element_text or len(element_text) < _MIN_MEANINGFUL_CHARACTERS:
            continue
        if element.bbox is None or _spatial_overlap(line, page, element) < 0.10:
            continue
        start = normalized.find(element_text)
        matched_exactly = start >= 0
        while start >= 0:
            for index in range(start, start + len(element_text)):
                covered[index] = True
            start = normalized.find(element_text, start + 1)
        if matched_exactly:
            continue
        match = SequenceMatcher(
            None,
            normalized,
            element_text,
            autojunk=False,
        ).find_longest_match()
        if (
            match.size >= 8
            and match.size / max(1, len(element_text)) >= 0.60
            and match.size / max(1, len(normalized)) >= 0.25
        ):
            for index in range(match.a, match.a + match.size):
                covered[index] = True
    if not any(covered):
        return [line]

    fragments: list[PdfTextLine] = []
    position = 0
    fragment_index = 0
    while position < len(covered):
        while position < len(covered) and covered[position]:
            position += 1
        start = position
        while position < len(covered) and not covered[position]:
            position += 1
        if start == position:
            continue
        raw_start = raw_indexes[start]
        raw_end = raw_indexes[position - 1] + 1
        # Keep punctuation and spaces belonging to the visible uncovered span,
        # but never include the first character owned by a covered match.
        while raw_end < len(line.text) and not line.text[raw_end].isalnum():
            raw_end += 1
        text = line.text[raw_start:raw_end].strip()
        if not _is_meaningful(text):
            continue
        bbox = _fragment_bbox(line, raw_start, raw_end)
        fragments.append(
            PdfTextLine(
                page_index=line.page_index,
                line_index=line.line_index,
                text=text,
                bbox=bbox,
                character_boxes=tuple(
                    (index - raw_start, box)
                    for index, box in line.character_boxes
                    if raw_start <= index < raw_end
                ),
                fragment_index=fragment_index,
                source_page_width=line.source_page_width,
                source_page_height=line.source_page_height,
            )
        )
        fragment_index += 1
    return fragments


def _normalized_with_indexes(text: str) -> tuple[str, list[int]]:
    normalized: list[str] = []
    indexes: list[int] = []
    for index, character in enumerate(text.casefold()):
        if character.isalnum():
            normalized.append(character)
            indexes.append(index)
    return "".join(normalized), indexes


def _fragment_bbox(line: PdfTextLine, start: int, end: int) -> tuple[float, float, float, float]:
    boxes = [box for index, box in line.character_boxes if start <= index < end]
    if not boxes:
        return line.bbox
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _spatial_overlap(line: PdfTextLine, page: Page, element: Element) -> float:
    assert element.bbox is not None
    lx1, bottom, lx2, top = line.bbox
    source_width, source_height = _source_page_size(line, page)
    line_box = (
        lx1 / source_width,
        (source_height - top) / source_height,
        lx2 / source_width,
        (source_height - bottom) / source_height,
    )
    ex1, ey1, ex2, ey2 = element.bbox.normalized
    intersection = max(0.0, min(line_box[2], ex2) - max(line_box[0], ex1)) * max(
        0.0, min(line_box[3], ey2) - max(line_box[1], ey1)
    )
    area = (line_box[2] - line_box[0]) * (line_box[3] - line_box[1])
    return intersection / area if area > 0 else 0.0


def _nearest_column(line: PdfTextLine, page: Page, elements: Iterable[Element]) -> int | None:
    candidates = [element for element in elements if element.bbox is not None and element.column_index is not None]
    if not candidates:
        return None
    source_width, _ = _source_page_size(line, page)
    x_center = (line.bbox[0] + line.bbox[2]) / 2 / source_width
    nearest = min(
        candidates,
        key=lambda element: abs((element.bbox.normalized[0] + element.bbox.normalized[2]) / 2 - x_center),
    )
    return nearest.column_index


def _source_page_size(line: PdfTextLine, page: Page) -> tuple[float, float]:
    return (
        line.source_page_width or page.width,
        line.source_page_height or page.height,
    )


def _reading_order_key(element: Element) -> tuple[float, float, float, float, int, str]:
    if element.bbox is None:
        return (
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            element.reading_order,
            element.element_id,
        )
    x1, y1, _, _ = element.bbox.normalized
    column = float(element.column_index) if element.column_index is not None else 0.0
    visual_row = round(y1 / _READING_ROW_TOLERANCE)
    return (visual_row, column, x1, y1, element.reading_order, element.element_id)
