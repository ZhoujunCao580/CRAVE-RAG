"""Build deterministic retrieval-only SearchUnits from SoftDoc Elements."""

from __future__ import annotations

import re
from html.parser import HTMLParser

from softdoc.ids import stable_digest
from softdoc.models import ContentAvailability, Document, Element, ElementType
from softdoc.retrieval.models import (
    SearchUnit,
    SearchUnitBuildResult,
    SearchUnitConfig,
    SkippedSearchElement,
)
from softdoc.retrieval.tokenization import TokenSpan, chunk_token_spans


_BOUNDARY_END = frozenset(".!?。！？;；:")


class SearchUnitBuilder:
    """Create an immutable retrieval view without following Relations."""

    def __init__(self, config: SearchUnitConfig | None = None) -> None:
        self.config = config or SearchUnitConfig()

    def build(self, document: Document) -> SearchUnitBuildResult:
        pages = {page.page_id: page for page in document.pages}
        ordered = sorted(
            document.elements,
            key=lambda element: (
                pages[element.page_id].page_index
                if element.page_id in pages
                else 1_000_000,
                element.reading_order,
                element.element_id,
            ),
        )
        units: list[SearchUnit] = []
        skipped: list[SkippedSearchElement] = []
        for element in ordered:
            try:
                body, context = _searchable_content(element)
                if not body:
                    generated_visual = bool(
                        element.metadata.get("generated_visual_text")
                    )
                    skipped.append(
                        SkippedSearchElement(
                            element_id=element.element_id,
                            element_type=element.element_type,
                            reason=(
                                "unverified_generated_visual_text"
                                if generated_visual
                                else "no_searchable_visual_text"
                                if element.element_type
                                in {ElementType.FIGURE, ElementType.CHART}
                                else "empty_searchable_content"
                            ),
                        )
                    )
                    continue
                page = pages.get(element.page_id)
                if page is None:
                    skipped.append(
                        SkippedSearchElement(
                            element_id=element.element_id,
                            element_type=element.element_type,
                            reason="missing_page",
                            details={"page_id": element.page_id},
                        )
                    )
                    continue
                ranges = _part_ranges(body, self.config)
                part_count = len(ranges)
                for part_index, (start, end) in enumerate(ranges):
                    content_text = body[start:end]
                    search_text = _join_search_text(context, content_text)
                    unit_id = "search-unit:" + stable_digest(
                        document.document_id,
                        element.element_id,
                        part_index,
                        self.config.model_dump(mode="json"),
                        content_text,
                    )
                    units.append(
                        SearchUnit(
                            search_unit_id=unit_id,
                            document_id=document.document_id,
                            element_id=element.element_id,
                            part_index=part_index,
                            part_count=part_count,
                            search_text=search_text,
                            content_text=content_text,
                            source_char_start=start,
                            source_char_end=end,
                            page_id=element.page_id,
                            page_index=page.page_index,
                            page_number=element.page_number,
                            reading_order=element.reading_order,
                            section_id=element.section_id,
                            section_path=list(element.section_path or []),
                            element_type=element.element_type,
                            content_availability=(
                                element.content_availability
                                or ContentAvailability.UNAVAILABLE
                            ),
                            index_version=self.config.index_version,
                        )
                    )
            except Exception as exc:  # one malformed Element must not abort a document
                skipped.append(
                    SkippedSearchElement(
                        element_id=element.element_id,
                        element_type=element.element_type,
                        reason="search_unit_build_error",
                        details={"error": f"{type(exc).__name__}: {exc}"},
                    )
                )
        return SearchUnitBuildResult(
            document_id=document.document_id,
            index_version=self.config.index_version,
            config=self.config,
            units=units,
            skipped_elements=skipped,
        )


def _searchable_content(element: Element) -> tuple[str, list[str]]:
    section_path = _clean_components(element.section_path or [])
    text = _normalize_text(element.text or "")
    label = _normalize_text(element.reference_label or "")

    if element.element_type == ElementType.HEADING:
        if section_path and _same_text(section_path[-1], text):
            section_path = section_path[:-1]
        return text, section_path

    if element.element_type in {ElementType.FIGURE, ElementType.CHART}:
        if element.metadata.get("generated_visual_text") is True:
            text = ""
        body = _join_unique([label, text])
        return body, section_path

    if element.element_type == ElementType.TABLE:
        table_text = html_to_text(element.html or "")
        body = _join_unique([table_text, text])
        if not body:
            body = label
            context = section_path
        else:
            context = _clean_components([*section_path, label])
        return body, context

    return text, section_path


def _part_ranges(text: str, config: SearchUnitConfig) -> list[tuple[int, int]]:
    tokens = chunk_token_spans(text)
    if not tokens:
        stripped = text.strip()
        if not stripped:
            return []
        start = text.index(stripped)
        return [(start, start + len(stripped))]
    if len(tokens) <= config.split_threshold_tokens:
        return [_trim_range(text, 0, len(text))]

    ranges: list[tuple[int, int]] = []
    start_index = 0
    while start_index < len(tokens):
        raw_end = min(len(tokens), start_index + config.part_size_tokens)
        end_index = (
            _preferred_end(tokens, text, start_index, raw_end)
            if raw_end < len(tokens)
            else raw_end
        )
        if end_index <= start_index:
            end_index = raw_end
        start = tokens[start_index].start
        end = (
            tokens[end_index].start
            if end_index < len(tokens)
            else len(text)
        )
        ranges.append(_trim_range(text, start, end))
        if end_index >= len(tokens):
            break
        next_start = max(start_index + 1, end_index - config.overlap_tokens)
        start_index = next_start
    return ranges


def _preferred_end(
    tokens: list[TokenSpan],
    text: str,
    start_index: int,
    raw_end: int,
) -> int:
    minimum = start_index + max(1, int((raw_end - start_index) * 0.65))
    for end_index in range(raw_end, minimum - 1, -1):
        token = tokens[end_index - 1]
        next_start = tokens[end_index].start if end_index < len(tokens) else len(text)
        between = text[token.end:next_start]
        if "\n" in between or text[token.end - 1] in _BOUNDARY_END:
            return end_index
    return raw_end


class _PlainTextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.casefold() in {"td", "th"}:
            self.parts.append("\t")
        elif tag.casefold() in {"br", "p", "li"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"tr", "p", "li"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def html_to_text(value: str) -> str:
    if not value.strip():
        return ""
    parser = _PlainTextHTMLParser()
    parser.feed(value)
    parser.close()
    return _normalize_text("".join(parser.parts))


def _normalize_text(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in value.split("\n"):
        line = re.sub(r"[ \f\v]+", " ", line)
        line = re.sub(r" *\t *", "\t", line).strip()
        lines.append(line)
    return "\n".join(line for line in lines if line).strip()


def _trim_range(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _clean_components(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        cleaned = _normalize_text(value)
        if cleaned and (not result or not _same_text(result[-1], cleaned)):
            result.append(cleaned)
    return result


def _join_unique(values: list[str]) -> str:
    return "\n".join(_clean_components(values))


def _join_search_text(context: list[str], content: str) -> str:
    return _join_unique([*context, content])


def _same_text(left: str, right: str) -> bool:
    return re.sub(r"\s+", " ", left).strip().casefold() == re.sub(
        r"\s+", " ", right
    ).strip().casefold()
