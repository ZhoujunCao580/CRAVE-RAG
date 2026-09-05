"""Deterministic Exact Anchor Lookup over one finalized SoftDoc Document."""

from __future__ import annotations

import re
from dataclasses import dataclass

from softdoc.ids import stable_digest
from softdoc.labels import (
    LabelRegistry,
    build_label_registry,
    normalize_reference_number,
)
from softdoc.models import Document, ElementType
from softdoc.retrieval.models import (
    AnchorKind,
    AnchorResolution,
    AnchorResolutionStatus,
    AnchorTargetHandle,
    AnchorTargetType,
    ExactAnchorMatch,
    ExactLookupResult,
    ExactLookupTraceEntry,
    SubQuestionInput,
)
from softdoc.store import DocumentStore


_PAGE_MODE_PRINTED_THEN_PHYSICAL = "printed_then_physical"
_PAGE_MODE_PHYSICAL_ORDER = "physical_document_order"
_ELEMENT_MODE_PHYSICAL_ORDER = "element_physical_order"
_SECTION_MODE_APPENDIX_TITLE = "appendix_title"
_SECTION_MODE_CHAPTER_TITLE = "chapter_title"
_SECTION_MODE_NAMED_TITLE = "named_section_title"
_MAX_RANGE_EXPANSION = 32

_NUMBERED_ANCHOR_PATTERNS: tuple[
    tuple[AnchorKind, re.Pattern[str], str | None], ...
] = (
    (
        AnchorKind.PAGE,
        re.compile(
            r"(?<!\w)(?:Page|p\.)\s*\(?\s*(?P<label>[0-9]+)\s*\)?"
            r"(?![0-9]|\.[0-9])",
            re.IGNORECASE,
        ),
        _PAGE_MODE_PRINTED_THEN_PHYSICAL,
    ),
    (
        AnchorKind.PAGE,
        re.compile(r"第\s*(?P<label>[0-9]+)\s*页"),
        _PAGE_MODE_PRINTED_THEN_PHYSICAL,
    ),
    (
        AnchorKind.PAGE,
        re.compile(
            r"(?<!\w)Slide\s*\(?\s*(?P<label>[0-9]+)\s*\)?"
            r"(?![0-9]|\.[0-9])",
            re.IGNORECASE,
        ),
        _PAGE_MODE_PHYSICAL_ORDER,
    ),
    (
        AnchorKind.FIGURE,
        re.compile(
            r"(?<!\w)(?:Figure|Fig\.?)\s*"
            r"(?P<label>(?:[0-9]+(?:\s*\(\s*[A-Za-z]\s*\)|[A-Za-z])?"
            r"|(?-i:[IVXLCDM]+)(?:\s*\(\s*[A-Za-z]\s*\))?))"
            r"(?![A-Za-z0-9_]|\.[0-9])",
            re.IGNORECASE,
        ),
        None,
    ),
    (
        AnchorKind.FIGURE,
        re.compile(
            r"图\s*(?P<label>[0-9]+(?:\s*\(\s*[A-Za-z]\s*\)|[A-Za-z])?)"
            r"(?![A-Za-z0-9_]|\.[0-9])"
        ),
        None,
    ),
    (
        AnchorKind.TABLE,
        re.compile(
            r"(?<!\w)Table\s*"
            r"(?P<label>(?:[0-9]+(?:\s*\(\s*[A-Za-z]\s*\)|[A-Za-z])?"
            r"|(?-i:[IVXLCDM]+)(?:\s*\(\s*[A-Za-z]\s*\))?))"
            r"(?![A-Za-z0-9_]|\.[0-9])",
            re.IGNORECASE,
        ),
        None,
    ),
    (
        AnchorKind.TABLE,
        re.compile(
            r"表\s*(?P<label>[0-9]+(?:\s*\(\s*[A-Za-z]\s*\)|[A-Za-z])?)"
            r"(?![A-Za-z0-9_]|\.[0-9])"
        ),
        None,
    ),
    (
        AnchorKind.SECTION,
        re.compile(
            r"(?<!\w)Section\s*(?P<label>[0-9]+(?:\.[0-9]+)*)"
            r"(?=$|[^\d.]|\.(?!\d))",
            re.IGNORECASE,
        ),
        None,
    ),
    (
        AnchorKind.SECTION,
        re.compile(r"第\s*(?P<label>[0-9]+(?:\.[0-9]+)*)\s*节"),
        None,
    ),
)

_WORD_ORDINALS: dict[str, int] = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
}

_WORD_CARDINALS: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

# Positional page expressions deliberately use physical document order.  They
# are separate from ``Page N``, whose number normally denotes a printed label.
_POSITIONAL_PAGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?<!\w)(?:the\s+)?(?P<label>"
        + "|".join(_WORD_ORDINALS)
        + r")\s+page\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<!\w)(?:the\s+)?(?P<label>[1-9][0-9]*)(?:st|nd|rd|th)\s+page\b",
        re.IGNORECASE,
    ),
    re.compile(r"(?<!\w)(?:the\s+)?(?P<label>last)\s+page\b", re.IGNORECASE),
    re.compile(r"(?P<label>\u6700\u540e)\s*(?:\u4e00\s*)?\u9875"),
    re.compile(r"(?<!\w)(?P<label>cover)\s+page\b", re.IGNORECASE),
    re.compile(
        r"(?<!\w)(?P<label>cover)\b(?!\s+of\s+each\s+chapter)",
        re.IGNORECASE,
    ),
)

_WORD_PAGE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?<!\w)Page\s+(?P<label>" + "|".join(_WORD_CARDINALS) + r")\b",
            re.IGNORECASE,
        ),
        _PAGE_MODE_PRINTED_THEN_PHYSICAL,
    ),
    (
        re.compile(
            r"(?<!\w)Slide\s+(?P<label>"
            + "|".join(_WORD_CARDINALS)
            + r")\b(?!\s+positions?\s+(?:after|before)\b)",
            re.IGNORECASE,
        ),
        _PAGE_MODE_PHYSICAL_ORDER,
    ),
)

_POSITIONAL_ELEMENT_PATTERNS: tuple[tuple[AnchorKind, re.Pattern[str]], ...] = (
    (
        AnchorKind.FIGURE,
        re.compile(
            r"(?<!\w)(?:the\s+)?(?P<label>"
            + "|".join(_WORD_ORDINALS)
            + r"|last)\s+(?:figure|fig\.)\b",
            re.IGNORECASE,
        ),
    ),
    (
        AnchorKind.TABLE,
        re.compile(
            r"(?<!\w)(?:the\s+)?(?P<label>"
            + "|".join(_WORD_ORDINALS)
            + r"|last)\s+table\b",
            re.IGNORECASE,
        ),
    ),
)

_SECTION_TITLE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?<!\w)Appendix\s+(?P<label>[A-Z]|[IVXLCDM]+|[0-9]+)\b",
            re.IGNORECASE,
        ),
        _SECTION_MODE_APPENDIX_TITLE,
    ),
    (
        re.compile(
            r"(?<!\w)(?:Chapter|Ch\.?)\s+(?P<label>[0-9]+|[IVXLCDM]+)\b",
            re.IGNORECASE,
        ),
        _SECTION_MODE_CHAPTER_TITLE,
    ),
    (
        re.compile(r"附录\s*(?P<label>[A-Za-z]|[0-9]+)"),
        _SECTION_MODE_APPENDIX_TITLE,
    ),
    (
        re.compile(
            r"(?<!\w)(?:in|from|within)\s+(?:the\s+)?section\s+of\s+"
            r"(?P<label>[A-Za-z][A-Za-z0-9&/()'’+\- ]{0,79}?)"
            r"(?=\s*[,?.]|$)",
            re.IGNORECASE,
        ),
        _SECTION_MODE_NAMED_TITLE,
    ),
    (
        re.compile(
            r"(?<!\w)(?:in|from|within)\s+(?:the\s+)?section\s+(?!of\b)"
            r"(?P<label>[A-Za-z][A-Za-z0-9&/()'’+\- ]{0,79}?)"
            r"(?=\s+(?:in|of)\s+the\s+(?:slides?|document|guidebook|ppt)\b"
            r"|\s*[,?.]|$)",
            re.IGNORECASE,
        ),
        _SECTION_MODE_NAMED_TITLE,
    ),
    (
        re.compile(
            r"(?<!\w)(?:in|from|within)\s+(?!the\s+section\b)(?:the\s+)?"
            r"(?P<label>['\"]?[A-Za-z][A-Za-z0-9&/()'’+\-]*"
            r"(?:\s+[A-Za-z0-9&/()'’+\-]+){0,7}['\"]?)\s+section\b",
            re.IGNORECASE,
        ),
        _SECTION_MODE_NAMED_TITLE,
    ),
)

_COMPOUND_ANCHOR_PATTERN = re.compile(
    r"(?<!\w)(?P<prefix>Pages?|pp\.|Slides?|Figures?|Figs?\.?|Tables?)\s*"
    r"(?P<labels>(?:[0-9]+|(?-i:[IVXLCDM]+))"
    r"(?:\s*(?:[-\u2012\u2013\u2014]|to|through|and|,)\s*"
    r"(?:[0-9]+|(?-i:[IVXLCDM]+)))+)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_FIRST_PAGE_RANGE_PATTERN = re.compile(
    r"(?<!\w)(?:the\s+)?first\s+(?P<count>[1-9][0-9]*)\s+pages\b",
    re.IGNORECASE,
)

_EXAMPLE_CUE = re.compile(
    r"\b(?:for\s+example|e\.g\.)|\blike(?=\s*\[)|例如|比如",
    re.IGNORECASE,
)
_FORMAT_CONTEXT = re.compile(
    r"\b(?:answer|output|format|formatted|list)\b|答案|格式|输出",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _AnchorMention:
    anchor_text: str
    kind: AnchorKind
    normalized_label: str
    start: int
    end: int
    pattern_order: int
    page_resolution_mode: str | None = None
    resolution_mode: str | None = None


def is_complete_supported_anchor(text: str) -> bool:
    """Return whether ``text`` is exactly one supported Anchor expression.

    This exposes the Exact Lookup grammar to upstream producers such as the
    Planner without introducing a second Figure/Table/Page/Section grammar.
    """

    stripped = text.strip()
    if not stripped:
        return False
    mentions, _ = _extract_mentions(stripped)
    return bool(mentions) and all(
        mention.start == 0
        and mention.end == len(stripped)
        and mention.anchor_text == stripped
        for mention in mentions
    )


class ExactAnchorLookup:
    """Resolve explicit page and numbered document anchors without mutation."""

    def lookup(
        self,
        subquestion: SubQuestionInput,
        document: Document,
    ) -> ExactLookupResult:
        store = DocumentStore(document)
        registry = build_label_registry(document)
        mentions, duplicate_trace = _extract_mentions(subquestion.text)
        resolutions: list[AnchorResolution] = []
        trace = list(duplicate_trace)

        for mention in mentions:
            resolution_id = (
                "anchor:"
                + stable_digest(
                    document.document_id,
                    subquestion.subquestion_id,
                    mention.kind.value,
                    mention.normalized_label,
                    mention.page_resolution_mode or "not_page",
                    mention.resolution_mode or "default_resolution",
                )
            )
            handles = (
                self._page_handles(
                    mention.normalized_label,
                    document,
                    store,
                    resolution_mode=(
                        mention.page_resolution_mode
                        or _PAGE_MODE_PRINTED_THEN_PHYSICAL
                    ),
                )
                if mention.kind == AnchorKind.PAGE
                else self._positional_element_handles(
                    mention.kind,
                    mention.normalized_label,
                    document,
                    store,
                )
                if mention.resolution_mode == _ELEMENT_MODE_PHYSICAL_ORDER
                else self._section_title_handles(
                    mention.normalized_label,
                    mention.resolution_mode,
                    document,
                    store,
                )
                if mention.resolution_mode
                in {
                    _SECTION_MODE_APPENDIX_TITLE,
                    _SECTION_MODE_CHAPTER_TITLE,
                    _SECTION_MODE_NAMED_TITLE,
                }
                else self._label_handles(
                    mention.kind,
                    mention.normalized_label,
                    registry,
                    store,
                )
            )
            status, reason = _resolution_status(handles)
            resolution = AnchorResolution(
                resolution_id=resolution_id,
                anchor_text=mention.anchor_text,
                anchor_kind=mention.kind,
                normalized_label=mention.normalized_label,
                source_span=(mention.start, mention.end),
                status=status,
                matches=handles,
                reason=reason,
            )
            resolutions.append(resolution)
            trace.append(
                ExactLookupTraceEntry(
                    code=f"anchor_{status.value}",
                    description=_trace_description(status),
                    resolution_id=resolution_id,
                    data={
                        "anchor_kind": mention.kind.value,
                        "normalized_label": mention.normalized_label,
                        "page_resolution_mode": mention.page_resolution_mode,
                        "resolution_mode": mention.resolution_mode,
                        "match_count": len(handles),
                    },
                )
            )

        if not mentions and not any(
            item.code == "anchor_range_too_large" for item in trace
        ):
            trace.append(
                ExactLookupTraceEntry(
                    code="no_anchor_detected",
                    description="The SubQuestion contains no supported explicit Anchor.",
                )
            )

        exact_matches = [
            _flatten_unique_resolution(resolution)
            for resolution in resolutions
            if resolution.status == AnchorResolutionStatus.UNIQUE
        ]
        return ExactLookupResult(
            subquestion_id=subquestion.subquestion_id,
            document_id=document.document_id,
            anchor_resolutions=resolutions,
            exact_anchor_matches=exact_matches,
            trace=trace,
        )

    @staticmethod
    def _page_handles(
        normalized_label: str,
        document: Document,
        store: DocumentStore,
        *,
        resolution_mode: str,
    ) -> list[AnchorTargetHandle]:
        if resolution_mode == _PAGE_MODE_PHYSICAL_ORDER:
            if normalized_label == "last":
                pages = sorted(
                    document.pages,
                    key=lambda page: (page.page_index, page.page_id),
                )[-1:]
                resolution_method = "physical_last_page"
            else:
                page_number = int(normalized_label)
                pages = sorted(
                    (
                        page
                        for page in document.pages
                        if page.page_number == page_number
                    ),
                    key=lambda page: (page.page_index, page.page_id),
                )
                resolution_method = _PAGE_MODE_PHYSICAL_ORDER
        else:
            page_number = int(normalized_label)
            printed_pages = [
                page
                for page in document.pages
                if normalized_label in page.page_label_aliases
            ]
            pages = sorted(
                printed_pages
                or [page for page in document.pages if page.page_number == page_number],
                key=lambda page: (page.page_index, page.page_id),
            )
            resolution_method = (
                "printed_page_label" if printed_pages else "physical_page_number"
            )
        return [
            AnchorTargetHandle(
                target_id=store.get_page(page.page_id).page_id,
                target_type=AnchorTargetType.PAGE,
                page_id=page.page_id,
                page_number=page.page_number,
                resolution_method=resolution_method,
            )
            for page in pages
        ]

    @staticmethod
    def _label_handles(
        kind: AnchorKind,
        normalized_label: str,
        registry: LabelRegistry,
        store: DocumentStore,
    ) -> list[AnchorTargetHandle]:
        reference_targets = registry.lookup_all(kind.value, normalized_label)
        handles: list[AnchorTargetHandle] = []
        for target in reference_targets:
            if kind == AnchorKind.SECTION:
                try:
                    section = store.get_section(target.target_id)
                    heading = store.get_element(section.heading_element_id)
                except KeyError:
                    # A typed registry should never yield a non-Section target,
                    # but malformed legacy artifacts must degrade to unresolved
                    # rather than terminating the whole lookup.
                    continue
                handles.append(
                    AnchorTargetHandle(
                        target_id=section.section_id,
                        target_type=AnchorTargetType.SECTION,
                        page_id=heading.page_id,
                        page_number=heading.page_number,
                        section_id=section.section_id,
                        resolution_method=target.resolution_rule,
                        label_source_id=target.label_source_id,
                    )
                )
                continue
            element = store.get_element(target.target_id)
            handles.append(
                AnchorTargetHandle(
                    target_id=element.element_id,
                    target_type=_element_target_type(element.element_type),
                    page_id=element.page_id,
                    page_number=element.page_number,
                    section_id=element.section_id,
                    resolution_method=target.resolution_rule,
                    label_source_id=target.label_source_id,
                )
            )
        return handles

    @staticmethod
    def _positional_element_handles(
        kind: AnchorKind,
        normalized_label: str,
        document: Document,
        store: DocumentStore,
    ) -> list[AnchorTargetHandle]:
        allowed = (
            {ElementType.FIGURE, ElementType.CHART}
            if kind == AnchorKind.FIGURE
            else {ElementType.TABLE}
        )
        page_indexes = {page.page_id: page.page_index for page in document.pages}
        elements = sorted(
            (element for element in document.elements if element.element_type in allowed),
            key=lambda element: (
                page_indexes.get(element.page_id, 1_000_000),
                element.reading_order,
                element.element_id,
            ),
        )
        if normalized_label == "last":
            selected = elements[-1:]
        else:
            ordinal = int(normalized_label)
            selected = elements[ordinal - 1 : ordinal]
        return [
            AnchorTargetHandle(
                target_id=store.get_element(element.element_id).element_id,
                target_type=_element_target_type(element.element_type),
                page_id=element.page_id,
                page_number=element.page_number,
                section_id=element.section_id,
                resolution_method=_ELEMENT_MODE_PHYSICAL_ORDER,
            )
            for element in selected
        ]

    @staticmethod
    def _section_title_handles(
        normalized_label: str,
        resolution_mode: str,
        document: Document,
        store: DocumentStore,
    ) -> list[AnchorTargetHandle]:
        if resolution_mode == _SECTION_MODE_NAMED_TITLE:
            expected = _normalize_named_section_title(normalized_label)
            exact_sections = [
                section
                for section in document.sections
                if _normalize_named_section_title(section.title) == expected
            ]
            # MinerU section titles often retain a numeric prefix or a short
            # deck heading before the user-visible section name.  A
            # multi-token, contiguous phrase remains deterministic when it
            # occurs in exactly one title.  Single generic words such as
            # "code" or "introduction" deliberately do not use this fallback.
            if exact_sections:
                candidate_sections = exact_sections
            elif len(expected.split()) >= 2:
                padded_expected = f" {expected} "
                contained_sections = [
                    section
                    for section in document.sections
                    if padded_expected
                    in f" {_normalize_named_section_title(section.title)} "
                ]
                candidate_sections = (
                    contained_sections if len(contained_sections) == 1 else []
                )
            else:
                candidate_sections = []

        else:
            prefix = (
                r"(?:appendix|附录)"
                if resolution_mode == _SECTION_MODE_APPENDIX_TITLE
                else r"(?:chapter|ch\.?)"
            )
            pattern = re.compile(
                rf"^\s*{prefix}\s*{re.escape(normalized_label)}(?=$|\b|\s|[:.\-])",
                re.IGNORECASE,
            )

            candidate_sections = [
                section for section in document.sections if pattern.search(section.title)
            ]

        handles: list[AnchorTargetHandle] = []
        for section in candidate_sections:
            try:
                heading = store.get_element(section.heading_element_id)
            except KeyError:
                continue
            handles.append(
                AnchorTargetHandle(
                    target_id=section.section_id,
                    target_type=AnchorTargetType.SECTION,
                    page_id=heading.page_id,
                    page_number=heading.page_number,
                    section_id=section.section_id,
                    resolution_method=resolution_mode,
                    label_source_id=section.heading_element_id,
                )
            )
        return handles


def _normalize_named_section_title(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", value.casefold())
    if tokens and tokens[-1].endswith("s") and not tokens[-1].endswith("ss"):
        tokens[-1] = tokens[-1][:-1]
    tokens = ["program" if token == "programme" else token for token in tokens]
    return " ".join(tokens)


def _extract_mentions(
    text: str,
) -> tuple[list[_AnchorMention], list[ExactLookupTraceEntry]]:
    found: list[_AnchorMention] = []
    trace: list[ExactLookupTraceEntry] = []
    ignored_spans = _answer_example_spans(text)
    pattern_order = 0

    for match in _FIRST_PAGE_RANGE_PATTERN.finditer(text):
        count = int(match.group("count"))
        if count > _MAX_RANGE_EXPANSION:
            trace.append(_range_limit_trace(match.group(0), count, match.span()))
            continue
        for page_number in range(1, count + 1):
            mention = _AnchorMention(
                anchor_text=match.group(0),
                kind=AnchorKind.PAGE,
                normalized_label=str(page_number),
                start=match.start(),
                end=match.end(),
                pattern_order=pattern_order,
                page_resolution_mode=_PAGE_MODE_PHYSICAL_ORDER,
            )
            _append_or_ignore(found, trace, mention, ignored_spans)
        pattern_order += 1

    for match in _COMPOUND_ANCHOR_PATTERN.finditer(text):
        prefix = match.group("prefix").casefold().rstrip(".")
        kind, mode = _compound_kind_and_mode(prefix)
        labels = _expand_compound_labels(match.group("labels"))
        if len(labels) > _MAX_RANGE_EXPANSION:
            trace.append(_range_limit_trace(match.group(0), len(labels), match.span()))
            pattern_order += 1
            continue
        for raw_label in labels:
            mention = _AnchorMention(
                anchor_text=match.group(0),
                kind=kind,
                normalized_label=_normalize_numbered_label(kind, raw_label),
                start=match.start(),
                end=match.end(),
                pattern_order=pattern_order,
                page_resolution_mode=mode if kind == AnchorKind.PAGE else None,
            )
            _append_or_ignore(found, trace, mention, ignored_spans)
        pattern_order += 1

    for kind, pattern, page_mode in _NUMBERED_ANCHOR_PATTERNS:
        for match in pattern.finditer(text):
            raw_label = match.group("label")
            mention = _AnchorMention(
                anchor_text=match.group(0),
                kind=kind,
                normalized_label=_normalize_numbered_label(kind, raw_label),
                start=match.start(),
                end=match.end(),
                pattern_order=pattern_order,
                page_resolution_mode=page_mode,
            )
            _append_or_ignore(found, trace, mention, ignored_spans)
        pattern_order += 1

    for pattern, page_mode in _WORD_PAGE_PATTERNS:
        for match in pattern.finditer(text):
            mention = _AnchorMention(
                anchor_text=match.group(0),
                kind=AnchorKind.PAGE,
                normalized_label=str(_WORD_CARDINALS[match.group("label").casefold()]),
                start=match.start(),
                end=match.end(),
                pattern_order=pattern_order,
                page_resolution_mode=page_mode,
            )
            _append_or_ignore(found, trace, mention, ignored_spans)
        pattern_order += 1

    for pattern in _POSITIONAL_PAGE_PATTERNS:
        for match in pattern.finditer(text):
            normalized = _normalize_positional_page_label(match.group("label"))
            mention = _AnchorMention(
                anchor_text=match.group(0),
                kind=AnchorKind.PAGE,
                normalized_label=normalized,
                start=match.start(),
                end=match.end(),
                pattern_order=pattern_order,
                page_resolution_mode=_PAGE_MODE_PHYSICAL_ORDER,
            )
            _append_or_ignore(found, trace, mention, ignored_spans)
        pattern_order += 1

    for kind, pattern in _POSITIONAL_ELEMENT_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group("label").casefold()
            normalized = "last" if raw == "last" else str(_WORD_ORDINALS[raw])
            mention = _AnchorMention(
                anchor_text=match.group(0),
                kind=kind,
                normalized_label=normalized,
                start=match.start(),
                end=match.end(),
                pattern_order=pattern_order,
                resolution_mode=_ELEMENT_MODE_PHYSICAL_ORDER,
            )
            _append_or_ignore(found, trace, mention, ignored_spans)
        pattern_order += 1

    for pattern, resolution_mode in _SECTION_TITLE_PATTERNS:
        for match in pattern.finditer(text):
            mention = _AnchorMention(
                anchor_text=match.group(0),
                kind=AnchorKind.SECTION,
                normalized_label=normalize_reference_number(match.group("label")),
                start=match.start(),
                end=match.end(),
                pattern_order=pattern_order,
                resolution_mode=resolution_mode,
            )
            _append_or_ignore(found, trace, mention, ignored_spans)
        pattern_order += 1

    # Prefer the longest expression when a numbered anchor is the prefix of a
    # compound expression at the same position.  For example, ``Figure 1`` is
    # also matched inside ``Figure 1-4``; keeping the longer span ensures label
    # 1 retains the range anchor text instead of becoming a separate mention.
    found.sort(key=lambda item: (item.start, -item.end, item.pattern_order))

    unique: list[_AnchorMention] = []
    seen: set[tuple[AnchorKind, str, str | None, str | None]] = set()
    for mention in found:
        key = (
            mention.kind,
            mention.normalized_label,
            mention.page_resolution_mode,
            mention.resolution_mode,
        )
        if key in seen:
            trace.append(
                ExactLookupTraceEntry(
                    code="duplicate_anchor_ignored",
                    description="A normalized duplicate Anchor was ignored.",
                    data={
                        "anchor_text": mention.anchor_text,
                        "anchor_kind": mention.kind.value,
                        "normalized_label": mention.normalized_label,
                        "source_span": [mention.start, mention.end],
                    },
                )
            )
            continue
        seen.add(key)
        unique.append(mention)
    return unique, trace


def _append_or_ignore(
    found: list[_AnchorMention],
    trace: list[ExactLookupTraceEntry],
    mention: _AnchorMention,
    ignored_spans: list[tuple[int, int]],
) -> None:
    if any(
        start <= mention.start and mention.end <= end for start, end in ignored_spans
    ):
        trace.append(
            ExactLookupTraceEntry(
                code="example_anchor_ignored",
                description="An Anchor inside an answer-format example was ignored.",
                data={
                    "anchor_text": mention.anchor_text,
                    "anchor_kind": mention.kind.value,
                    "normalized_label": mention.normalized_label,
                    "source_span": [mention.start, mention.end],
                },
            )
        )
        return
    found.append(mention)


def _compound_kind_and_mode(prefix: str) -> tuple[AnchorKind, str | None]:
    if prefix in {"page", "pages", "pp"}:
        return AnchorKind.PAGE, _PAGE_MODE_PRINTED_THEN_PHYSICAL
    if prefix in {"slide", "slides"}:
        return AnchorKind.PAGE, _PAGE_MODE_PHYSICAL_ORDER
    if prefix.startswith("fig"):
        return AnchorKind.FIGURE, None
    return AnchorKind.TABLE, None


def _expand_compound_labels(value: str) -> list[str]:
    range_match = re.fullmatch(
        r"\s*(?P<start>[0-9]+)\s*(?:[-\u2012\u2013\u2014]|to|through)\s*"
        r"(?P<end>[0-9]+)\s*",
        value,
        re.IGNORECASE,
    )
    if range_match:
        start = int(range_match.group("start"))
        end = int(range_match.group("end"))
        if end < start:
            return []
        return [str(item) for item in range(start, end + 1)]
    return [
        item
        for item in re.split(r"\s*(?:,|and)\s*", value, flags=re.IGNORECASE)
        if item
    ]


def _normalize_numbered_label(kind: AnchorKind, raw_label: str) -> str:
    if kind == AnchorKind.PAGE:
        return str(int(raw_label))
    normalized = normalize_reference_number(raw_label)
    normalized = re.sub(r"\s+", "", normalized)
    return re.sub(r"\(([a-z])\)", r"\1", normalized)


def _range_limit_trace(
    anchor_text: str,
    requested_count: int,
    source_span: tuple[int, int],
) -> ExactLookupTraceEntry:
    return ExactLookupTraceEntry(
        code="anchor_range_too_large",
        description="The explicit Anchor range exceeds the deterministic expansion limit.",
        data={
            "anchor_text": anchor_text,
            "requested_count": requested_count,
            "maximum_count": _MAX_RANGE_EXPANSION,
            "source_span": list(source_span),
        },
    )


def _normalize_positional_page_label(raw_label: str) -> str:
    folded = raw_label.casefold()
    if folded == "last" or folded == "\u6700\u540e":
        return "last"
    if folded == "cover":
        return "1"
    if folded in _WORD_ORDINALS:
        return str(_WORD_ORDINALS[folded])
    return str(int(folded))


def _answer_example_spans(text: str) -> list[tuple[int, int]]:
    """Return spans that are examples of output syntax, not query content."""

    spans: list[tuple[int, int]] = []
    for cue in _EXAMPLE_CUE.finditer(text):
        left_context = text[max(0, cue.start() - 140) : cue.start()]
        search_end = min(len(text), cue.end() + 240)
        bracket_start = text.find("[", cue.end(), search_end)
        if bracket_start >= 0:
            bracket_end = text.find("]", bracket_start + 1, search_end)
            if bracket_end >= 0:
                spans.append((cue.start(), bracket_end + 1))
                continue
        if not _FORMAT_CONTEXT.search(left_context):
            continue
        sentence_end = len(text)
        for terminator in ("\n", "。", "?", "!"):
            position = text.find(terminator, cue.end())
            if position >= 0:
                sentence_end = min(sentence_end, position + 1)
        period = text.find(".", cue.end() + 4)
        if period >= 0:
            sentence_end = min(sentence_end, period + 1)
        spans.append((cue.start(), sentence_end))
    return spans


def _resolution_status(
    handles: list[AnchorTargetHandle],
) -> tuple[AnchorResolutionStatus, str | None]:
    if len(handles) == 1:
        return AnchorResolutionStatus.UNIQUE, None
    if len(handles) > 1:
        return AnchorResolutionStatus.AMBIGUOUS, "multiple_targets"
    return AnchorResolutionStatus.UNRESOLVED, "target_not_found"


def _trace_description(status: AnchorResolutionStatus) -> str:
    if status == AnchorResolutionStatus.UNIQUE:
        return "The Anchor resolved to one lightweight target handle."
    if status == AnchorResolutionStatus.AMBIGUOUS:
        return "The Anchor matched multiple targets and was not auto-selected."
    return "The Anchor syntax was recognized but no target exists in this Document."


def _flatten_unique_resolution(resolution: AnchorResolution) -> ExactAnchorMatch:
    target = resolution.matches[0]
    return ExactAnchorMatch(
        resolution_id=resolution.resolution_id,
        anchor_text=resolution.anchor_text,
        anchor_kind=resolution.anchor_kind,
        normalized_label=resolution.normalized_label,
        target_id=target.target_id,
        target_type=target.target_type,
        page_id=target.page_id,
        page_number=target.page_number,
        section_id=target.section_id,
        resolution_method=target.resolution_method,
    )


def _element_target_type(element_type: ElementType) -> AnchorTargetType:
    try:
        return AnchorTargetType(element_type.value)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported Exact Anchor target element type: {element_type.value}"
        ) from exc
