"""Conservative reconciliation of MinerU cross-page aggregate table HTML.

MinerU sometimes stores the HTML for several physical page fragments on the
first Table block while leaving continuation blocks empty in content-list
output.  The adapter preserves that raw payload and recovers page-local HTML
from each continuation page's preproc block.  This module reconciles the two
representations without changing the parser adapter:

* each Table Element keeps only the rows visible on its physical page;
* the original aggregate HTML remains untouched in Provenance;
* confirmed fragment-chain metadata is emitted for RelationBuilder.

Every group is transactional.  If page ownership is not unique, ordered,
complete, and free of a rowspan crossing, no Element in that group is changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from html import unescape
from html.parser import HTMLParser
import re
from typing import Any

from pydantic import Field

from softdoc.ids import stable_digest
from softdoc.models import Document, Element, ElementType, SoftDocModel


RULE_ID = "table.mineru_aggregate_fragment_reconciliation.v1"
METADATA_KEY = "cross_page_table_reconciliation"
FRAGMENT_METADATA_KEY = "cross_page_table_fragment"


class TableReconciliationStatus(str, Enum):
    CONFIRMED = "confirmed"
    SKIPPED = "skipped"


class TableFragmentAssignment(SoftDocModel):
    element_id: str
    page_number: int = Field(ge=1)
    fragment_index: int = Field(ge=0)
    aggregate_row_start: int = Field(ge=0)
    aggregate_row_end: int = Field(ge=0)
    repeated_header_rows_ignored: int = Field(default=0, ge=0)


class TableReconciliationDecision(SoftDocModel):
    group_id: str
    aggregate_source_element_id: str
    status: TableReconciliationStatus
    reason: str
    source_page_numbers: list[int] = Field(default_factory=list)
    assignments: list[TableFragmentAssignment] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)


class TableReconciliationResult(SoftDocModel):
    decisions: list[TableReconciliationDecision] = Field(default_factory=list)

    @property
    def confirmed_decisions(self) -> list[TableReconciliationDecision]:
        return [
            decision
            for decision in self.decisions
            if decision.status == TableReconciliationStatus.CONFIRMED
        ]


@dataclass(frozen=True)
class _ParsedRow:
    normalized_text: str
    rowspans: tuple[int, ...]


@dataclass(frozen=True)
class _PageMatch:
    element: Element
    start: int
    end: int
    repeated_header_rows_ignored: int


@dataclass(frozen=True)
class _Proposal:
    source: Element
    new_source_html: str
    matches: tuple[_PageMatch, ...]
    decision: TableReconciliationDecision


class _RowParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[_ParsedRow] = []
        self.nested_tables = 0
        self._table_depth = 0
        self._row_text: list[str] | None = None
        self._rowspans: list[int] | None = None
        self._cell_text: list[str] | None = None
        self._cell_rowspan = 1

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.casefold()
        attributes = dict(attrs)
        if tag == "table":
            self._table_depth += 1
            if self._table_depth > 1:
                self.nested_tables += 1
            return
        if self._table_depth != 1:
            return
        if tag == "tr":
            self._row_text = []
            self._rowspans = []
        elif tag in {"td", "th"} and self._row_text is not None:
            self._cell_text = []
            self._cell_rowspan = _positive_int(attributes.get("rowspan"))
        elif tag == "br" and self._cell_text is not None:
            self._cell_text.append(" ")

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self.handle_starttag(tag, attrs)

    def handle_data(self, data: str) -> None:
        if self._table_depth == 1 and self._cell_text is not None:
            self._cell_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if (
            tag in {"td", "th"}
            and self._table_depth == 1
            and self._cell_text is not None
            and self._row_text is not None
            and self._rowspans is not None
        ):
            self._row_text.append(" ".join(self._cell_text))
            self._rowspans.append(self._cell_rowspan)
            self._cell_text = None
            self._cell_rowspan = 1
        elif (
            tag == "tr"
            and self._table_depth == 1
            and self._row_text is not None
            and self._rowspans is not None
        ):
            self.rows.append(
                _ParsedRow(
                    normalized_text=_normalize_text(" ".join(self._row_text)),
                    rowspans=tuple(self._rowspans),
                )
            )
            self._row_text = None
            self._rowspans = None
        elif tag == "table":
            self._table_depth = max(0, self._table_depth - 1)


def reconcile_mineru_aggregate_tables(
    document: Document,
) -> TableReconciliationResult:
    """Split only aggregate groups that pass every deterministic safeguard."""

    stored = document.metadata.get(METADATA_KEY)
    if isinstance(stored, dict) and stored.get("rule_id") == RULE_ID:
        return TableReconciliationResult.model_validate(
            {"decisions": stored.get("decisions", [])}
        )

    tables = sorted(
        (
            element
            for element in document.elements
            if element.element_type == ElementType.TABLE
        ),
        key=lambda element: (element.page_number, element.reading_order),
    )
    recovered_by_page: dict[int, list[Element]] = {}
    for table in tables:
        if _has_html_recovery(table):
            recovered_by_page.setdefault(table.page_number, []).append(table)

    proposals: list[_Proposal] = []
    decisions: list[TableReconciliationDecision] = []
    claimed_targets: set[str] = set()
    for source in tables:
        aggregate_html = _raw_mineru_html(source)
        if not aggregate_html or source.element_id in claimed_targets:
            continue
        run: list[Element] = []
        page_number = source.page_number + 1
        ambiguous_page = False
        while page_number in recovered_by_page:
            candidates = [
                candidate
                for candidate in recovered_by_page[page_number]
                if candidate.element_id not in claimed_targets
            ]
            if len(candidates) != 1:
                ambiguous_page = True
                break
            run.append(candidates[0])
            page_number += 1
        if not run:
            continue

        proposal, decision = _build_proposal(
            document=document,
            source=source,
            aggregate_html=aggregate_html,
            continuation_elements=run,
            ambiguous_page=ambiguous_page,
        )
        decisions.append(decision)
        if proposal is not None:
            proposals.append(proposal)
            claimed_targets.update(
                match.element.element_id for match in proposal.matches
            )

    for proposal in proposals:
        _apply_proposal(proposal)

    document.metadata[METADATA_KEY] = {
        "rule_id": RULE_ID,
        "confirmed_group_count": len(proposals),
        "confirmed_fragment_count": sum(
            1 + len(proposal.matches) for proposal in proposals
        ),
        "decisions": [decision.model_dump(mode="json") for decision in decisions],
    }
    return TableReconciliationResult(decisions=decisions)


def _build_proposal(
    *,
    document: Document,
    source: Element,
    aggregate_html: str,
    continuation_elements: list[Element],
    ambiguous_page: bool,
) -> tuple[_Proposal | None, TableReconciliationDecision]:
    group_id = "table-group:" + stable_digest(
        document.document_id,
        source.element_id,
        *(element.element_id for element in continuation_elements),
    )
    page_numbers = [source.page_number] + [
        element.page_number for element in continuation_elements
    ]

    def skipped(reason: str, **evidence: Any) -> tuple[None, TableReconciliationDecision]:
        return None, TableReconciliationDecision(
            group_id=group_id,
            aggregate_source_element_id=source.element_id,
            status=TableReconciliationStatus.SKIPPED,
            reason=reason,
            source_page_numbers=page_numbers,
            evidence=evidence,
        )

    if ambiguous_page:
        return skipped("multiple_recovered_tables_on_continuation_page")

    aggregate_rows, nested = _parse_rows(aggregate_html)
    source_rows, source_nested = _parse_rows(source.html or "")
    if nested or source_nested:
        return skipped("nested_table_not_supported")
    if not aggregate_rows or len(source_rows) != len(aggregate_rows):
        return skipped(
            "aggregate_and_rewritten_source_row_count_mismatch",
            aggregate_row_count=len(aggregate_rows),
            source_row_count=len(source_rows),
        )

    matches: list[_PageMatch] = []
    previous_end: int | None = None
    for element in continuation_elements:
        local_rows, local_nested = _parse_rows(element.html or "")
        if local_nested or not local_rows:
            return skipped(
                "continuation_has_no_supported_local_rows",
                continuation_element_id=element.element_id,
            )
        match = _match_local_page(
            aggregate_rows,
            local_rows,
            element,
            required_start=previous_end,
        )
        if match is None:
            return skipped(
                "continuation_rows_not_uniquely_assignable",
                continuation_element_id=element.element_id,
                continuation_page_number=element.page_number,
            )
        matches.append(match)
        previous_end = match.end

    if not matches or matches[0].start <= 0:
        return skipped("aggregate_source_fragment_would_be_empty")
    if matches[-1].end != len(aggregate_rows):
        return skipped(
            "continuation_fragments_do_not_cover_aggregate_suffix",
            aggregate_row_count=len(aggregate_rows),
            final_matched_row=matches[-1].end,
        )
    cuts = {matches[0].start}
    cuts.update(match.end for match in matches[:-1])
    crossing_cuts = _rowspan_crossings(aggregate_rows, cuts)
    if crossing_cuts:
        return skipped(
            "rowspan_crosses_page_boundary",
            crossing_row_boundaries=crossing_cuts,
        )

    prefix_count = matches[0].start
    new_source_html = _keep_first_rows(source.html or "", prefix_count)
    rewritten_rows, rewritten_nested = _parse_rows(new_source_html)
    if rewritten_nested or [row.normalized_text for row in rewritten_rows] != [
        row.normalized_text for row in aggregate_rows[:prefix_count]
    ]:
        return skipped("source_fragment_round_trip_failed")

    coverage = [row.normalized_text for row in rewritten_rows]
    for match in matches:
        local_rows, _ = _parse_rows(match.element.html or "")
        coverage.extend(
            row.normalized_text
            for row in local_rows[match.repeated_header_rows_ignored :]
        )
    if coverage != [row.normalized_text for row in aggregate_rows]:
        return skipped("aggregate_content_coverage_check_failed")

    assignments = [
        TableFragmentAssignment(
            element_id=source.element_id,
            page_number=source.page_number,
            fragment_index=0,
            aggregate_row_start=0,
            aggregate_row_end=prefix_count,
        )
    ]
    assignments.extend(
        TableFragmentAssignment(
            element_id=match.element.element_id,
            page_number=match.element.page_number,
            fragment_index=index,
            aggregate_row_start=match.start,
            aggregate_row_end=match.end,
            repeated_header_rows_ignored=match.repeated_header_rows_ignored,
        )
        for index, match in enumerate(matches, start=1)
    )
    decision = TableReconciliationDecision(
        group_id=group_id,
        aggregate_source_element_id=source.element_id,
        status=TableReconciliationStatus.CONFIRMED,
        reason="unique_ordered_complete_row_assignment",
        source_page_numbers=page_numbers,
        assignments=assignments,
        evidence={
            "rule_id": RULE_ID,
            "aggregate_row_count": len(aggregate_rows),
            "aggregate_html_preserved_in_provenance": True,
            "rowspan_crossing_count": 0,
            "round_trip_verified": True,
        },
    )
    return (
        _Proposal(
            source=source,
            new_source_html=new_source_html,
            matches=tuple(matches),
            decision=decision,
        ),
        decision,
    )


def _match_local_page(
    aggregate_rows: list[_ParsedRow],
    local_rows: list[_ParsedRow],
    element: Element,
    *,
    required_start: int | None,
) -> _PageMatch | None:
    aggregate_text = [row.normalized_text for row in aggregate_rows]
    local_text = [row.normalized_text for row in local_rows]
    max_header_rows = min(3, max(0, len(local_text) - 1))
    for ignored_headers in range(max_header_rows + 1):
        body = local_text[ignored_headers:]
        if not body or not any(body):
            continue
        positions = _sequence_positions(aggregate_text, body)
        if required_start is not None:
            positions = [position for position in positions if position == required_start]
        if len(positions) != 1:
            continue
        start = positions[0]
        ignored = local_text[:ignored_headers]
        if ignored and not all(
            row and row in aggregate_text[:start] for row in ignored
        ):
            continue
        return _PageMatch(
            element=element,
            start=start,
            end=start + len(body),
            repeated_header_rows_ignored=ignored_headers,
        )
    return None


def _apply_proposal(proposal: _Proposal) -> None:
    decision = proposal.decision
    source = proposal.source
    source.html = proposal.new_source_html
    _filter_removed_embedded_assets(source)
    assignment_by_id = {
        assignment.element_id: assignment for assignment in decision.assignments
    }
    elements = [source] + [match.element for match in proposal.matches]
    for element in elements:
        assignment = assignment_by_id[element.element_id]
        element.metadata[FRAGMENT_METADATA_KEY] = {
            "rule_id": RULE_ID,
            "group_id": decision.group_id,
            "status": "confirmed",
            "aggregate_source_element_id": decision.aggregate_source_element_id,
            "fragment_index": assignment.fragment_index,
            "fragment_count": len(elements),
            "source_page_numbers": list(decision.source_page_numbers),
            "aggregate_row_start": assignment.aggregate_row_start,
            "aggregate_row_end": assignment.aggregate_row_end,
            "repeated_header_rows_ignored": (
                assignment.repeated_header_rows_ignored
            ),
            "aggregate_html_preserved_in_provenance": True,
        }


def _parse_rows(html: str) -> tuple[list[_ParsedRow], bool]:
    parser = _RowParser()
    parser.feed(html)
    parser.close()
    return parser.rows, bool(parser.nested_tables)


def _keep_first_rows(html: str, count: int) -> str:
    table_open = re.search(r"<table\b[^>]*>", html, re.IGNORECASE)
    table_close = list(re.finditer(r"</table\s*>", html, re.IGNORECASE))
    row_matches = list(
        re.finditer(r"<tr\b[^>]*>.*?</tr\s*>", html, re.IGNORECASE | re.DOTALL)
    )
    if table_open is None or not table_close or len(row_matches) < count:
        return ""
    prefix = html[: table_open.end()]
    suffix = html[table_close[-1].start() :]
    return prefix + "".join(match.group(0) for match in row_matches[:count]) + suffix


def _rowspan_crossings(
    rows: list[_ParsedRow],
    cuts: set[int],
) -> list[int]:
    crossings: list[int] = []
    for cut in sorted(cuts):
        for row_index, row in enumerate(rows):
            if row_index >= cut:
                break
            if any(row_index + rowspan > cut for rowspan in row.rowspans):
                crossings.append(cut)
                break
    return crossings


def _sequence_positions(haystack: list[str], needle: list[str]) -> list[int]:
    if len(needle) > len(haystack):
        return []
    return [
        index
        for index in range(len(haystack) - len(needle) + 1)
        if haystack[index : index + len(needle)] == needle
    ]


def _normalize_text(value: str) -> str:
    return " ".join(
        re.findall(r"[\w.%+\-]+", unescape(value).casefold(), re.UNICODE)
    )


def _positive_int(value: str | None) -> int:
    try:
        return max(1, int(value or 1))
    except ValueError:
        return 1


def _raw_mineru_html(element: Element) -> str:
    content = element.provenance.raw_payload.get("content")
    if not isinstance(content, dict):
        return ""
    value = content.get("html")
    return value if isinstance(value, str) else ""


def _has_html_recovery(element: Element) -> bool:
    return bool(
        element.metadata.get("html_recovery")
        or element.provenance.metadata.get("html_recovery")
    )


def _filter_removed_embedded_assets(element: Element) -> None:
    records = element.metadata.get("embedded_html_assets")
    if not isinstance(records, list):
        return
    html = (element.html or "").replace("\\", "/")
    kept: list[Any] = []
    for record in records:
        if not isinstance(record, dict):
            kept.append(record)
            continue
        stored = str(record.get("stored_path") or "").replace("\\", "/")
        original = str(record.get("original_src") or "").replace("\\", "/")
        if (stored and stored in html) or (original and original in html):
            kept.append(record)
    if kept:
        element.metadata["embedded_html_assets"] = kept
    else:
        element.metadata.pop("embedded_html_assets", None)
