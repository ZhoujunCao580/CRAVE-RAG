"""Auditable coverage requirements, page scopes, and source inventories.

Coverage questions (counts, exhaustive lists, extrema, and complements) need a
different completion contract from ordinary semantic QA.  This module keeps
that contract deterministic without pretending that PDF page numbers are
always unambiguous:

* the Planner records the requested operation and the scope text;
* the Environment resolves the scope to canonical ``page_id`` values;
* every resolution records the chosen page-number namespace and the unused
  alternative, so a wrong interpretation can be reviewed and overridden;
* an inventory is complete only when the scope itself resolved completely.

No model answer or Gold annotation is used here.
"""

from __future__ import annotations

from enum import Enum
import re
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from softdoc.models import Document, Element, ElementType, Page, SoftDocModel
from softdoc.table_fragments import FRAGMENT_METADATA_KEY


class CoverageOperator(str, Enum):
    COUNT = "count"
    COLLECT_ALL = "collect_all"
    TOP_K = "top_k"
    ARGMAX = "argmax"
    ARGMIN = "argmin"
    COMPLEMENT = "complement"


class CoverageSourceType(str, Enum):
    PAGE = "page"
    TABLE = "table"
    FIGURE = "figure"
    CHART = "chart"
    VISUAL = "visual"
    ELEMENT = "element"


class PageNumberNamespace(str, Enum):
    PRINTED_PAGE_LABEL = "printed_page_label"
    PHYSICAL_PAGE_ORDER = "physical_page_order"
    WHOLE_DOCUMENT = "whole_document"


class CoverageScopeStatus(str, Enum):
    RESOLVED = "resolved"
    PARTIAL = "partial"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"


class CoverageInventoryStatus(str, Enum):
    COMPLETE = "complete"
    BLOCKED = "blocked"


class CoverageRequirement(SoftDocModel):
    """Static Planner output; it never contains runtime counts or page IDs."""

    required: Literal[True] = True
    operator: CoverageOperator
    scope_text: str = Field(min_length=1)
    item_type: str = Field(min_length=1)
    source_type: CoverageSourceType
    predicate: str | None = None
    limit: int | None = Field(default=None, ge=1)

    @field_validator("scope_text", "item_type")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Coverage text fields must not be blank")
        return stripped

    @field_validator("predicate")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def validate_limit(self) -> Self:
        if self.operator == CoverageOperator.TOP_K and self.limit is None:
            raise ValueError("top_k coverage requires limit")
        if self.operator != CoverageOperator.TOP_K and self.limit is not None:
            raise ValueError("limit is only valid for top_k coverage")
        return self


class ResolvedCoveragePage(SoftDocModel):
    page_id: str = Field(min_length=1)
    physical_page_number: int = Field(ge=1)
    display_page_label: str | None = None
    page_label_aliases: list[str] = Field(default_factory=list)


class CoverageScopeCandidate(SoftDocModel):
    """One complete interpretation of a page expression."""

    namespace: PageNumberNamespace
    status: CoverageScopeStatus
    requested_labels: list[str] = Field(default_factory=list)
    resolved_pages: list[ResolvedCoveragePage] = Field(default_factory=list)
    missing_labels: list[str] = Field(default_factory=list)
    ambiguous_labels: dict[str, list[str]] = Field(default_factory=dict)
    rationale: str = Field(min_length=1)


class CoverageScopeResolution(SoftDocModel):
    """Chosen scope plus alternatives needed for audit and correction."""

    raw_scope_text: str = Field(min_length=1)
    chosen_namespace: PageNumberNamespace | None = None
    status: CoverageScopeStatus
    resolved_pages: list[ResolvedCoveragePage] = Field(default_factory=list)
    missing_labels: list[str] = Field(default_factory=list)
    ambiguous_labels: dict[str, list[str]] = Field(default_factory=dict)
    candidates: list[CoverageScopeCandidate] = Field(default_factory=list)
    requires_review: bool = False
    decision_reason: str = Field(min_length=1)

    @property
    def resolved_page_ids(self) -> list[str]:
        return [page.page_id for page in self.resolved_pages]


class CoverageInventoryItem(SoftDocModel):
    inventory_id: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)
    page_ids: list[str] = Field(min_length=1)
    physical_page_numbers: list[int] = Field(min_length=1)
    source_type: CoverageSourceType
    element_type: ElementType | None = None
    deduplication_reason: str = Field(min_length=1)


class CoverageInventory(SoftDocModel):
    status: CoverageInventoryStatus
    scope: CoverageScopeResolution
    source_type: CoverageSourceType
    items: list[CoverageInventoryItem] = Field(default_factory=list)
    structural_count: int | None = Field(default=None, ge=0)
    completion_reason: str = Field(min_length=1)


class QuestionCoveragePlan(SoftDocModel):
    """Runtime coverage artifact attached to one Root or SubQuestion."""

    question_id: str = Field(min_length=1)
    requirement: CoverageRequirement
    scope_resolution: CoverageScopeResolution
    inventory: CoverageInventory


_PAGE_RANGE = re.compile(
    r"^\s*(?P<kind>Pages?|pp\.|Slides?)\s*(?P<start>[0-9]+)\s*"
    r"(?:[-\u2012\u2013\u2014]|to|through)\s*(?P<end>[0-9]+)\s*$",
    re.IGNORECASE,
)
_PAGE_LIST = re.compile(
    r"^\s*(?P<kind>Pages?|pp\.|Slides?)\s*(?P<labels>[0-9]+"
    r"(?:\s*(?:,|and)\s*[0-9]+)+)\s*$",
    re.IGNORECASE,
)
_SINGLE_PAGE = re.compile(
    r"^\s*(?P<kind>Page|p\.|Slide)\s*(?P<label>[0-9]+)\s*$",
    re.IGNORECASE,
)
_FIRST_PAGES = re.compile(
    r"^\s*(?:the\s+)?first\s+(?P<count>[1-9][0-9]*)\s+pages\s*$",
    re.IGNORECASE,
)
_LAST_PAGE = re.compile(r"^\s*(?:the\s+)?last\s+page\s*$", re.IGNORECASE)
_WHOLE_DOCUMENT = re.compile(
    r"^\s*(?:the\s+)?(?:whole|entire|full)\s+(?:document|pdf|report)\s*$",
    re.IGNORECASE,
)


class CoverageScopeResolver:
    """Resolve page scopes without mixing printed and physical namespaces.

    ``namespace_override`` is the explicit correction path.  It makes a human
    or an audit tool, rather than a hidden heuristic, the authority when both
    interpretations remain plausible.
    """

    def resolve(
        self,
        requirement: CoverageRequirement,
        document: Document,
        *,
        namespace_override: PageNumberNamespace | None = None,
    ) -> CoverageScopeResolution:
        pages = sorted(document.pages, key=lambda item: (item.page_index, item.page_id))
        if not pages:
            return CoverageScopeResolution(
                raw_scope_text=requirement.scope_text,
                status=CoverageScopeStatus.UNRESOLVED,
                requires_review=True,
                decision_reason="document_has_no_pages",
            )

        scope = requirement.scope_text.strip()
        if _WHOLE_DOCUMENT.fullmatch(scope):
            candidate = self._whole_document_candidate(pages)
            return self._resolution(scope, candidate, [candidate], "explicit_whole_document")

        if match := _FIRST_PAGES.fullmatch(scope):
            labels = [str(value) for value in range(1, int(match.group("count")) + 1)]
            candidate = self._physical_candidate(labels, pages, "explicit_ordinal_scope")
            return self._resolution(scope, candidate, [candidate], "ordinal_scope_is_physical")

        if _LAST_PAGE.fullmatch(scope):
            label = str(len(pages))
            candidate = self._physical_candidate([label], pages, "explicit_last_page")
            return self._resolution(scope, candidate, [candidate], "last_page_is_physical")

        parsed = self._parse_numbered_scope(scope)
        if parsed is None:
            return CoverageScopeResolution(
                raw_scope_text=scope,
                status=CoverageScopeStatus.UNRESOLVED,
                requires_review=True,
                decision_reason="unsupported_scope_expression",
            )
        kind, labels = parsed
        physical = self._physical_candidate(labels, pages, "physical_pdf_order")
        printed = self._printed_candidate(labels, pages)
        candidates = [printed, physical]

        if namespace_override is not None:
            if namespace_override == PageNumberNamespace.WHOLE_DOCUMENT:
                raise ValueError("whole_document is not a valid numbered-scope override")
            chosen = (
                printed
                if namespace_override == PageNumberNamespace.PRINTED_PAGE_LABEL
                else physical
            )
            return self._resolution(
                scope,
                chosen,
                candidates,
                "explicit_namespace_override",
                requires_review=chosen.status != CoverageScopeStatus.RESOLVED,
            )

        if kind == "slide":
            return self._resolution(
                scope,
                physical,
                candidates,
                "slide_numbers_use_physical_order",
            )

        has_confirmed_printed_namespace = any(page.page_label_aliases for page in pages)
        if not has_confirmed_printed_namespace:
            return self._resolution(
                scope,
                physical,
                candidates,
                "no_confirmed_printed_labels_physical_fallback",
            )

        # Page N normally means a printed label.  We deliberately do not fall
        # back one label at a time, because that would silently mix namespaces.
        chosen = printed
        competing_complete = (
            printed.status == CoverageScopeStatus.RESOLVED
            and physical.status == CoverageScopeStatus.RESOLVED
            and printed.resolved_pages != physical.resolved_pages
        )
        return self._resolution(
            scope,
            chosen,
            candidates,
            "confirmed_printed_labels_preferred_for_page_expression",
            requires_review=(
                competing_complete
                or chosen.status != CoverageScopeStatus.RESOLVED
            ),
        )

    @staticmethod
    def _parse_numbered_scope(scope: str) -> tuple[str, list[str]] | None:
        if match := _PAGE_RANGE.fullmatch(scope):
            start, end = int(match.group("start")), int(match.group("end"))
            if end < start:
                return None
            kind = "slide" if match.group("kind").casefold().startswith("slide") else "page"
            if end - start > 10_000:
                return None
            return kind, [str(value) for value in range(start, end + 1)]
        if match := _PAGE_LIST.fullmatch(scope):
            kind = "slide" if match.group("kind").casefold().startswith("slide") else "page"
            return kind, re.findall(r"[0-9]+", match.group("labels"))
        if match := _SINGLE_PAGE.fullmatch(scope):
            kind = "slide" if match.group("kind").casefold().startswith("slide") else "page"
            return kind, [match.group("label")]
        return None

    @staticmethod
    def _page_record(page: Page) -> ResolvedCoveragePage:
        return ResolvedCoveragePage(
            page_id=page.page_id,
            physical_page_number=page.page_number,
            display_page_label=page.display_page_label,
            page_label_aliases=list(page.page_label_aliases),
        )

    def _whole_document_candidate(self, pages: list[Page]) -> CoverageScopeCandidate:
        return CoverageScopeCandidate(
            namespace=PageNumberNamespace.WHOLE_DOCUMENT,
            status=CoverageScopeStatus.RESOLVED,
            requested_labels=[],
            resolved_pages=[self._page_record(page) for page in pages],
            rationale="all_physical_pages_in_document_order",
        )

    def _physical_candidate(
        self,
        labels: list[str],
        pages: list[Page],
        rationale: str,
    ) -> CoverageScopeCandidate:
        by_number = {str(page.page_number): page for page in pages}
        resolved = [by_number[label] for label in labels if label in by_number]
        missing = [label for label in labels if label not in by_number]
        return CoverageScopeCandidate(
            namespace=PageNumberNamespace.PHYSICAL_PAGE_ORDER,
            status=self._status(len(resolved), missing, {}),
            requested_labels=labels,
            resolved_pages=[self._page_record(page) for page in resolved],
            missing_labels=missing,
            rationale=rationale,
        )

    def _printed_candidate(
        self,
        labels: list[str],
        pages: list[Page],
    ) -> CoverageScopeCandidate:
        by_label: dict[str, list[Page]] = {}
        for page in pages:
            for label in page.page_label_aliases:
                by_label.setdefault(label, []).append(page)
        resolved: list[Page] = []
        missing: list[str] = []
        ambiguous: dict[str, list[str]] = {}
        for label in labels:
            matches = by_label.get(label, [])
            if not matches:
                missing.append(label)
            elif len(matches) > 1:
                ambiguous[label] = [page.page_id for page in matches]
            else:
                resolved.append(matches[0])
        return CoverageScopeCandidate(
            namespace=PageNumberNamespace.PRINTED_PAGE_LABEL,
            status=self._status(len(resolved), missing, ambiguous),
            requested_labels=labels,
            resolved_pages=[self._page_record(page) for page in resolved],
            missing_labels=missing,
            ambiguous_labels=ambiguous,
            rationale="confirmed_page_label_aliases_only",
        )

    @staticmethod
    def _status(
        resolved_count: int,
        missing: list[str],
        ambiguous: dict[str, list[str]],
    ) -> CoverageScopeStatus:
        if ambiguous:
            return CoverageScopeStatus.AMBIGUOUS
        if missing:
            return (
                CoverageScopeStatus.PARTIAL
                if resolved_count
                else CoverageScopeStatus.UNRESOLVED
            )
        return CoverageScopeStatus.RESOLVED

    @staticmethod
    def _resolution(
        scope: str,
        chosen: CoverageScopeCandidate,
        candidates: list[CoverageScopeCandidate],
        reason: str,
        *,
        requires_review: bool = False,
    ) -> CoverageScopeResolution:
        return CoverageScopeResolution(
            raw_scope_text=scope,
            chosen_namespace=chosen.namespace,
            status=chosen.status,
            resolved_pages=list(chosen.resolved_pages),
            missing_labels=list(chosen.missing_labels),
            ambiguous_labels=dict(chosen.ambiguous_labels),
            candidates=candidates,
            requires_review=requires_review,
            decision_reason=reason,
        )


def build_coverage_inventory(
    requirement: CoverageRequirement,
    resolution: CoverageScopeResolution,
    document: Document,
) -> CoverageInventory:
    """Enumerate candidate sources only after a scope resolves completely."""

    if (
        resolution.status != CoverageScopeStatus.RESOLVED
        or resolution.requires_review
    ):
        return CoverageInventory(
            status=CoverageInventoryStatus.BLOCKED,
            scope=resolution,
            source_type=requirement.source_type,
            completion_reason=(
                "scope_not_fully_resolved"
                if resolution.status != CoverageScopeStatus.RESOLVED
                else "page_namespace_requires_review"
            ),
        )

    page_ids = set(resolution.resolved_page_ids)
    pages_by_id = {page.page_id: page for page in document.pages}
    if requirement.source_type == CoverageSourceType.PAGE:
        items = [
            CoverageInventoryItem(
                inventory_id=page.page_id,
                source_ids=[page.page_id],
                page_ids=[page.page_id],
                physical_page_numbers=[page.page_number],
                source_type=CoverageSourceType.PAGE,
                deduplication_reason="canonical_page_id",
            )
            for page in resolution.resolved_pages
        ]
    else:
        accepted_types = _accepted_element_types(requirement.source_type)
        elements = [
            element
            for element in document.elements
            if element.page_id in page_ids and element.element_type in accepted_types
        ]
        items = _deduplicate_elements(elements, pages_by_id, requirement.source_type)

    structural_count = (
        len(items) if _is_pure_structural_count(requirement) else None
    )
    return CoverageInventory(
        status=CoverageInventoryStatus.COMPLETE,
        scope=resolution,
        source_type=requirement.source_type,
        items=items,
        structural_count=structural_count,
        completion_reason="all_sources_in_resolved_scope_enumerated",
    )


def _accepted_element_types(source_type: CoverageSourceType) -> set[ElementType]:
    if source_type == CoverageSourceType.TABLE:
        return {ElementType.TABLE}
    if source_type == CoverageSourceType.FIGURE:
        return {ElementType.FIGURE}
    if source_type == CoverageSourceType.CHART:
        return {ElementType.CHART}
    if source_type == CoverageSourceType.VISUAL:
        return {ElementType.TABLE, ElementType.FIGURE, ElementType.CHART}
    return set(ElementType)


def _deduplicate_elements(
    elements: list[Element],
    pages_by_id: dict[str, Page],
    source_type: CoverageSourceType,
) -> list[CoverageInventoryItem]:
    grouped: dict[str, list[Element]] = {}
    reasons: dict[str, str] = {}
    for element in sorted(
        elements,
        key=lambda item: (item.page_number, item.reading_order, item.element_id),
    ):
        fragment = element.metadata.get(FRAGMENT_METADATA_KEY)
        if (
            element.element_type == ElementType.TABLE
            and isinstance(fragment, dict)
            and fragment.get("status") == "confirmed"
            and fragment.get("group_id")
        ):
            key = "table-group:" + str(fragment["group_id"])
            reason = "confirmed_cross_page_table_group"
        else:
            key = element.element_id
            reason = "canonical_element_id"
        grouped.setdefault(key, []).append(element)
        reasons[key] = reason

    items: list[CoverageInventoryItem] = []
    for key, group in grouped.items():
        group_pages = sorted(
            {item.page_id for item in group},
            key=lambda page_id: (pages_by_id[page_id].page_index, page_id),
        )
        items.append(
            CoverageInventoryItem(
                inventory_id=key,
                source_ids=[item.element_id for item in group],
                page_ids=group_pages,
                physical_page_numbers=[
                    pages_by_id[page_id].page_number for page_id in group_pages
                ],
                source_type=source_type,
                element_type=group[0].element_type,
                deduplication_reason=reasons[key],
            )
        )
    return items


def _is_pure_structural_count(requirement: CoverageRequirement) -> bool:
    if requirement.operator != CoverageOperator.COUNT or requirement.predicate is not None:
        return False
    normalized_item = requirement.item_type.casefold().strip().rstrip("s")
    normalized_source = requirement.source_type.value.casefold().rstrip("s")
    return normalized_item == normalized_source
