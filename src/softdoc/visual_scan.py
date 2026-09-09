"""Question-directed visual scans without reviving legacy Coverage inventory.

The Planner declares only whether a target needs an exhaustive visual scan and
the boundary of that scan.  The Environment resolves the boundary and binds
page images to stable input IDs; images themselves remain out-of-band
multimodal message content and are never represented by fake JSON fields.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from enum import StrEnum
import re
import unicodedata
from typing import Annotated, Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from softdoc.models import Document, ElementType, SoftDocModel


class VisualScanScopeKind(StrEnum):
    WHOLE_DOCUMENT = "whole_document"
    PAGES = "pages"
    SECTION = "section"


class WholeDocumentVisualScanScope(SoftDocModel):
    kind: Literal[VisualScanScopeKind.WHOLE_DOCUMENT]


class PagesVisualScanScope(SoftDocModel):
    kind: Literal[VisualScanScopeKind.PAGES]
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return _nonblank(value, "Page scope text")


class SectionVisualScanScope(SoftDocModel):
    kind: Literal[VisualScanScopeKind.SECTION]
    anchor_text: str = Field(min_length=1)

    @field_validator("anchor_text")
    @classmethod
    def strip_anchor_text(cls, value: str) -> str:
        return _nonblank(value, "Section anchor text")


VisualScanScope = Annotated[
    WholeDocumentVisualScanScope
    | PagesVisualScanScope
    | SectionVisualScanScope,
    Field(discriminator="kind"),
]


class VisualScanRequirement(SoftDocModel):
    """Minimal Planner-owned declaration for one Root or SubQuestion."""

    required: Literal[True]
    scope: VisualScanScope


class SectionAnchorStatus(StrEnum):
    READY = "ready"
    NEEDS_CONTROLLER_ANCHOR = "needs_controller_anchor"
    AMBIGUOUS = "ambiguous"


class SectionAnchorCandidate(SoftDocModel):
    source_id: str = Field(min_length=1)
    page_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    heading_text: str = Field(min_length=1)
    heading_bbox: tuple[float, float, float, float] | None = None
    match_score: float = Field(ge=0.0, le=1.0)


class SectionScanPreparation(SoftDocModel):
    """Environment result before any page image is sent to the scan VLM."""

    status: SectionAnchorStatus
    requested_anchor_text: str = Field(min_length=1)
    resolved_anchor: SectionAnchorCandidate | None = None
    candidates: list[SectionAnchorCandidate] = Field(default_factory=list)
    controller_gap: str | None = None

    @model_validator(mode="after")
    def validate_status_payload(self) -> Self:
        if self.status == SectionAnchorStatus.READY:
            if self.resolved_anchor is None or self.controller_gap is not None:
                raise ValueError("A ready Section scan requires only a resolved anchor")
        elif self.resolved_anchor is not None or not self.controller_gap:
            raise ValueError(
                "An unresolved Section scan requires a Controller gap and no resolved anchor"
            )
        return self


class VisualScanBatchInput(SoftDocModel):
    """Text portion of one multimodal scan call.

    The transport attaches one page image after each corresponding ``input_id``.
    No image path or image payload is exposed in this model-facing JSON.
    """

    target_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    scope: VisualScanScope
    batch_index: int = Field(ge=1)
    batch_count: int = Field(ge=1)
    input_ids: list[str] = Field(min_length=1)

    @field_validator("target_id", "question")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        return _nonblank(value, "Visual scan input text")

    @field_validator("input_ids")
    @classmethod
    def unique_input_ids(cls, value: list[str]) -> list[str]:
        cleaned = [_nonblank(item, "Visual scan input ID") for item in value]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("Visual scan input IDs must be unique within a batch")
        return cleaned

    @model_validator(mode="after")
    def validate_batch_position(self) -> Self:
        if self.batch_index > self.batch_count:
            raise ValueError("batch_index cannot exceed batch_count")
        return self


class VisualScanItem(SoftDocModel):
    input_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    count: int = Field(ge=1)


class VisualScanLimitation(SoftDocModel):
    input_id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class VisualScanBatchResult(SoftDocModel):
    batch_index: int = Field(ge=1)
    items: list[VisualScanItem] = Field(default_factory=list)
    partial_count: int | None = Field(default=None, ge=0)
    limitations: list[VisualScanLimitation] = Field(default_factory=list)
    section_ended: bool = False
    end_heading_text: str | None = None

    @model_validator(mode="before")
    @classmethod
    def derive_partial_count(cls, value: Any) -> Any:
        """Derive the redundant aggregate from auditable item counts."""

        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        limitations = normalized.get("limitations") or []
        if limitations:
            normalized["partial_count"] = None
            return normalized
        total = 0
        for item in normalized.get("items") or []:
            count = item.get("count") if isinstance(item, dict) else getattr(item, "count", None)
            if isinstance(count, int) and not isinstance(count, bool):
                total += count
        normalized["partial_count"] = total
        return normalized

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.limitations:
            if self.partial_count is not None:
                raise ValueError(
                    "partial_count must be null when any supplied page is unresolved"
                )
        else:
            expected = sum(item.count for item in self.items)
            if self.partial_count != expected:
                raise ValueError(
                    "partial_count must equal the sum of item counts when the batch is resolved"
                )
        if self.section_ended != bool(self.end_heading_text):
            raise ValueError(
                "section_ended and end_heading_text must be supplied together"
            )
        return self


class VisualScanAggregateResult(SoftDocModel):
    answer_candidate: str | None = None
    supporting_input_ids: list[str] = Field(default_factory=list)
    scope_complete: bool
    limitation: str | None = None

    @field_validator("supporting_input_ids")
    @classmethod
    def unique_supporting_ids(cls, value: list[str]) -> list[str]:
        cleaned = [_nonblank(item, "Supporting input ID") for item in value]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("Supporting input IDs must be unique")
        return cleaned

    @model_validator(mode="after")
    def validate_completion(self) -> Self:
        if self.scope_complete:
            if self.answer_candidate is None or self.limitation is not None:
                raise ValueError(
                    "A complete visual scan requires an answer and no limitation"
                )
        elif not (self.limitation or "").strip():
            raise ValueError("An incomplete visual scan requires a limitation")
        return self


class SectionScopeResolver:
    """Locate a section start without pretending MinerU hierarchy is reliable.

    Only Heading elements are eligible.  A unique high-confidence match becomes
    the start anchor.  The section end is intentionally *not* inferred here:
    the scan VLM observes subsequent full pages and identifies the visible next
    major section boundary during the normal scan call.
    """

    def __init__(self, *, minimum_score: float = 0.86, ambiguity_margin: float = 0.02):
        self.minimum_score = minimum_score
        self.ambiguity_margin = ambiguity_margin

    def resolve(
        self,
        document: Document,
        scope: SectionVisualScanScope,
    ) -> SectionScanPreparation:
        ranked = sorted(
            (
                self._candidate(scope.anchor_text, element)
                for element in document.elements
                if element.element_type == ElementType.HEADING
            ),
            key=lambda item: (-item.match_score, item.page_number, item.source_id),
        )
        ranked = [item for item in ranked if item.match_score >= self.minimum_score]
        gap = (
            f'Locate the section heading "{scope.anchor_text}" and read the source '
            "that marks the beginning of that section."
        )
        if not ranked:
            return SectionScanPreparation(
                status=SectionAnchorStatus.NEEDS_CONTROLLER_ANCHOR,
                requested_anchor_text=scope.anchor_text,
                controller_gap=gap,
            )

        best = ranked[0]
        competing = [
            item
            for item in ranked[1:]
            if best.match_score - item.match_score <= self.ambiguity_margin
            and item.page_id != best.page_id
        ]
        if competing:
            return SectionScanPreparation(
                status=SectionAnchorStatus.AMBIGUOUS,
                requested_anchor_text=scope.anchor_text,
                candidates=[best, *competing[:4]],
                controller_gap=gap,
            )
        return SectionScanPreparation(
            status=SectionAnchorStatus.READY,
            requested_anchor_text=scope.anchor_text,
            resolved_anchor=best,
            candidates=[best],
        )

    @staticmethod
    def _candidate(anchor_text: str, element) -> SectionAnchorCandidate:
        bbox = element.bbox.normalized if element.bbox is not None else None
        return SectionAnchorCandidate(
            source_id=element.element_id,
            page_id=element.page_id,
            page_number=element.page_number,
            heading_text=element.text or "",
            heading_bbox=bbox,
            match_score=_heading_match_score(anchor_text, element.text or ""),
        )


def validate_visual_scan_batch_result(
    batch_input: VisualScanBatchInput,
    result: VisualScanBatchResult,
) -> VisualScanBatchResult:
    """Reject hallucinated or stale page-local IDs before aggregation."""

    if result.batch_index != batch_input.batch_index:
        raise ValueError("Visual scan result batch_index does not match its input")
    visible = set(batch_input.input_ids)
    referenced = {item.input_id for item in result.items} | {
        item.input_id for item in result.limitations
    }
    hidden = sorted(referenced - visible)
    if hidden:
        raise ValueError(
            "Visual scan result references input IDs not visible in this batch: "
            + ", ".join(hidden)
        )
    return result


def _heading_match_score(query: str, candidate: str) -> float:
    query_compact = _compact_heading(query)
    candidate_compact = _compact_heading(candidate)
    if not query_compact or not candidate_compact:
        return 0.0
    if query_compact == candidate_compact:
        return 1.0
    shorter, longer = sorted((query_compact, candidate_compact), key=len)
    if len(shorter) >= 6 and longer.startswith(shorter):
        return 0.98
    return SequenceMatcher(None, query_compact, candidate_compact).ratio()


def _compact_heading(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).casefold()
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", ascii_text)


def _nonblank(value: str, label: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{label} must not be blank")
    return stripped
