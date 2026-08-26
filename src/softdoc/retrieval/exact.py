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


_ANCHOR_PATTERNS: tuple[tuple[AnchorKind, re.Pattern[str]], ...] = (
    (
        AnchorKind.PAGE,
        re.compile(
            r"(?<!\w)Page\s*(?P<label>[0-9]+)(?![0-9]|\.[0-9])",
            re.IGNORECASE,
        ),
    ),
    (
        AnchorKind.PAGE,
        re.compile(r"第\s*(?P<label>[0-9]+)\s*页"),
    ),
    (
        AnchorKind.FIGURE,
        re.compile(
            r"(?<!\w)(?:Figure|Fig\.?)\s*(?P<label>[0-9]+)"
            r"(?![A-Za-z0-9_]|\.[0-9])",
            re.IGNORECASE,
        ),
    ),
    (
        AnchorKind.FIGURE,
        re.compile(r"图\s*(?P<label>[0-9]+)(?![A-Za-z0-9_]|\.[0-9])"),
    ),
    (
        AnchorKind.TABLE,
        re.compile(
            r"(?<!\w)Table\s*(?P<label>[0-9]+)(?![A-Za-z0-9_]|\.[0-9])",
            re.IGNORECASE,
        ),
    ),
    (
        AnchorKind.TABLE,
        re.compile(r"表\s*(?P<label>[0-9]+)(?![A-Za-z0-9_]|\.[0-9])"),
    ),
    (
        AnchorKind.SECTION,
        re.compile(
            r"(?<!\w)Section\s*(?P<label>[0-9]+(?:\.[0-9]+)*)"
            r"(?=$|[^\d.]|\.(?!\d))",
            re.IGNORECASE,
        ),
    ),
    (
        AnchorKind.SECTION,
        re.compile(r"第\s*(?P<label>[0-9]+(?:\.[0-9]+)*)\s*节"),
    ),
)

_PAGE_MODE_PRINTED_THEN_PHYSICAL = "printed_then_physical"
_PAGE_MODE_PHYSICAL_ORDER = "physical_document_order"

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
)

_EXAMPLE_CUE = re.compile(r"\b(?:for\s+example|e\.g\.)|例如|比如", re.IGNORECASE)
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


def is_complete_supported_anchor(text: str) -> bool:
    """Return whether ``text`` is exactly one supported Anchor expression.

    This exposes the Exact Lookup grammar to upstream producers such as the
    Planner without introducing a second Figure/Table/Page/Section grammar.
    """

    stripped = text.strip()
    if not stripped:
        return False
    mentions, _ = _extract_mentions(stripped)
    return (
        len(mentions) == 1
        and mentions[0].start == 0
        and mentions[0].end == len(stripped)
        and mentions[0].anchor_text == stripped
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
                        "match_count": len(handles),
                    },
                )
            )

        if not mentions:
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


def _extract_mentions(
    text: str,
) -> tuple[list[_AnchorMention], list[ExactLookupTraceEntry]]:
    found: list[_AnchorMention] = []
    trace: list[ExactLookupTraceEntry] = []
    ignored_spans = _answer_example_spans(text)
    for pattern_order, (kind, pattern) in enumerate(_ANCHOR_PATTERNS):
        for match in pattern.finditer(text):
            raw_label = match.group("label")
            normalized = (
                str(int(raw_label))
                if kind == AnchorKind.PAGE
                else normalize_reference_number(raw_label)
            )
            if any(start <= match.start() and match.end() <= end for start, end in ignored_spans):
                trace.append(
                    ExactLookupTraceEntry(
                        code="example_anchor_ignored",
                        description=(
                            "An Anchor inside an answer-format example was ignored."
                        ),
                        data={
                            "anchor_text": match.group(0),
                            "anchor_kind": kind.value,
                            "normalized_label": normalized,
                            "source_span": [match.start(), match.end()],
                        },
                    )
                )
                continue
            found.append(
                _AnchorMention(
                    anchor_text=match.group(0),
                    kind=kind,
                    normalized_label=normalized,
                    start=match.start(),
                    end=match.end(),
                    pattern_order=pattern_order,
                    page_resolution_mode=(
                        _PAGE_MODE_PRINTED_THEN_PHYSICAL
                        if kind == AnchorKind.PAGE
                        else None
                    ),
                )
            )

    positional_pattern_offset = len(_ANCHOR_PATTERNS)
    for offset, pattern in enumerate(_POSITIONAL_PAGE_PATTERNS):
        pattern_order = positional_pattern_offset + offset
        for match in pattern.finditer(text):
            normalized = _normalize_positional_page_label(match.group("label"))
            if any(
                start <= match.start() and match.end() <= end
                for start, end in ignored_spans
            ):
                trace.append(
                    ExactLookupTraceEntry(
                        code="example_anchor_ignored",
                        description=(
                            "An Anchor inside an answer-format example was ignored."
                        ),
                        data={
                            "anchor_text": match.group(0),
                            "anchor_kind": AnchorKind.PAGE.value,
                            "normalized_label": normalized,
                            "source_span": [match.start(), match.end()],
                        },
                    )
                )
                continue
            found.append(
                _AnchorMention(
                    anchor_text=match.group(0),
                    kind=AnchorKind.PAGE,
                    normalized_label=normalized,
                    start=match.start(),
                    end=match.end(),
                    pattern_order=pattern_order,
                    page_resolution_mode=_PAGE_MODE_PHYSICAL_ORDER,
                )
            )
    found.sort(key=lambda item: (item.start, item.end, item.pattern_order))

    unique: list[_AnchorMention] = []
    seen: set[tuple[AnchorKind, str, str | None]] = set()
    for mention in found:
        key = (
            mention.kind,
            mention.normalized_label,
            mention.page_resolution_mode,
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


def _normalize_positional_page_label(raw_label: str) -> str:
    folded = raw_label.casefold()
    if folded == "last" or folded == "\u6700\u540e":
        return "last"
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
