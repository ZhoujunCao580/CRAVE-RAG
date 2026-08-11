"""Resolve printed PDF page labels without assuming one global offset."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
from pydantic import Field

from softdoc.coverage import PdfTextLine, extract_pdf_text_lines
from softdoc.ids import stable_digest
from softdoc.models import (
    Document,
    Element,
    Page,
    PageLabelCandidate,
    PageLabelSource,
    PageLabelStatus,
    SoftDocModel,
)


_LABEL_TEXT = re.compile(
    r"^\s*(?:(?:page)\s*)?[-\u2012\u2013\u2014]?\s*"
    r"(?P<label>[0-9]{1,5})\s*[-\u2012\u2013\u2014]?\s*$",
    re.IGNORECASE,
)
_CHINESE_LABEL_TEXT = re.compile(r"^\s*第\s*(?P<label>[0-9]{1,5})\s*页\s*$")


class PageLabelDecision(SoftDocModel):
    page_id: str
    physical_page_number: int = Field(ge=1)
    display_page_label: str | None = None
    page_label_aliases: list[str] = Field(default_factory=list)
    confirmed_candidate_ids: list[str] = Field(default_factory=list)
    candidate_candidate_ids: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)


class PageLabelResolutionResult(SoftDocModel):
    decisions: list[PageLabelDecision] = Field(default_factory=list)


@dataclass(frozen=True)
class _Observation:
    page_index: int
    label: int
    source: PageLabelSource
    confidence: float
    source_element_id: str | None
    normalized_bbox: tuple[float, float, float, float] | None
    evidence: dict[str, Any]


class PageLabelResolver:
    """Detect page labels as aliases, retaining gaps and multi-label spreads.

    The resolver intentionally does not fit one document-wide offset.  It uses
    explicit marginal observations first, fills only arithmetically bounded
    gaps, and permits more than one printed label to address the same physical
    PDF page.
    """

    def __init__(
        self,
        *,
        top_boundary: float = 0.10,
        bottom_boundary: float = 0.87,
        maximum_bounded_gap: int = 12,
        minimum_extrapolation_run: int = 3,
    ) -> None:
        self.top_boundary = top_boundary
        self.bottom_boundary = bottom_boundary
        self.maximum_bounded_gap = maximum_bounded_gap
        self.minimum_extrapolation_run = minimum_extrapolation_run

    def resolve(
        self,
        document: Document,
        *,
        source_pdf: Path | None = None,
    ) -> PageLabelResolutionResult:
        pages = {page.page_index: page for page in document.pages}
        observations = self._parser_observations(document)
        observations.extend(self._element_observations(document))
        if source_pdf is not None and source_pdf.is_file():
            observations.extend(self._pdf_observations(source_pdf, pages))
        observations = self._deduplicate(observations)

        confirmed: dict[int, set[int]] = defaultdict(set)
        confirmation_evidence: dict[tuple[int, int], list[str]] = defaultdict(list)
        for observation in observations:
            if (
                observation.source == PageLabelSource.PDF_PAGE_LABEL
                or self._has_sequence_support(observation, observations)
            ):
                confirmed[observation.page_index].add(observation.label)
                confirmation_evidence[(observation.page_index, observation.label)].append(
                    "explicit_observation_with_sequence_support"
                )

        inferred = self._missing_spread_alias_inference(
            observations,
            confirmed,
            confirmation_evidence,
        )
        inferred.extend(
            self._bounded_interpolation(
                [*observations, *inferred],
                confirmed,
                confirmation_evidence,
            )
        )
        inferred.extend(
            self._backward_sequence_inference(
                pages,
                confirmed,
                confirmation_evidence,
            )
        )

        candidates_by_page: dict[int, list[PageLabelCandidate]] = defaultdict(list)
        for observation in observations:
            status = (
                PageLabelStatus.CONFIRMED
                if observation.label in confirmed.get(observation.page_index, set())
                else PageLabelStatus.CANDIDATE
            )
            confidence = observation.confidence
            if status == PageLabelStatus.CANDIDATE:
                confidence = min(confidence, 0.64)
            candidates_by_page[observation.page_index].append(
                self._candidate(
                    document,
                    observation,
                    status=status,
                    confidence=confidence,
                )
            )
        for observation in inferred:
            candidates_by_page[observation.page_index].append(
                self._candidate(
                    document,
                    observation,
                    status=PageLabelStatus.CONFIRMED,
                    confidence=observation.confidence,
                )
            )

        decisions: list[PageLabelDecision] = []
        for page in document.pages:
            candidates = sorted(
                candidates_by_page.get(page.page_index, []),
                key=lambda item: (
                    int(item.label),
                    item.source.value,
                    item.candidate_id,
                ),
            )
            aliases = sorted(
                {
                    candidate.label
                    for candidate in candidates
                    if candidate.status == PageLabelStatus.CONFIRMED
                },
                key=int,
            )
            confidences = [
                candidate.confidence
                for candidate in candidates
                if candidate.status == PageLabelStatus.CONFIRMED
            ]
            # Assign the mutually dependent fields atomically.  The final Page
            # model is validated again by serialization and round-trip tests.
            object.__setattr__(page, "display_page_label", aliases[0] if aliases else None)
            object.__setattr__(
                page,
                "display_page_label_confidence",
                min(confidences) if confidences else None,
            )
            object.__setattr__(page, "page_label_aliases", aliases)
            object.__setattr__(page, "page_label_candidates", candidates)
            decisions.append(
                PageLabelDecision(
                    page_id=page.page_id,
                    physical_page_number=page.page_number,
                    display_page_label=page.display_page_label,
                    page_label_aliases=list(aliases),
                    confirmed_candidate_ids=[
                        item.candidate_id
                        for item in candidates
                        if item.status == PageLabelStatus.CONFIRMED
                    ],
                    candidate_candidate_ids=[
                        item.candidate_id
                        for item in candidates
                        if item.status == PageLabelStatus.CANDIDATE
                    ],
                    evidence={
                        "allows_multiple_labels": len(aliases) > 1,
                        "physical_fallback_available": True,
                    },
                )
            )
        document.metadata["page_label_decisions"] = [
            decision.model_dump(mode="json") for decision in decisions
        ]
        return PageLabelResolutionResult(decisions=decisions)

    def _parser_observations(self, document: Document) -> list[_Observation]:
        observations: list[_Observation] = []
        for page in document.pages:
            signals = page.metadata.get("parser_page_label_signals", [])
            if not isinstance(signals, list):
                continue
            for signal in signals:
                if not isinstance(signal, dict):
                    continue
                label = _numeric_label(str(signal.get("text") or ""))
                bbox_value = signal.get("normalized_bbox")
                normalized_bbox: tuple[float, float, float, float] | None = None
                if isinstance(bbox_value, (list, tuple)) and len(bbox_value) == 4:
                    normalized_bbox = tuple(float(value) for value in bbox_value)
                if label is None:
                    continue
                if normalized_bbox is not None and not self._is_marginal(normalized_bbox):
                    continue
                observations.append(
                    _Observation(
                        page_index=page.page_index,
                        label=label,
                        source=PageLabelSource.PARSER_PAGE_NUMBER,
                        confidence=0.985,
                        source_element_id=None,
                        normalized_bbox=normalized_bbox,
                        evidence={
                            "rule": "parser_page_number_block",
                            "text": signal.get("text"),
                            "source_locator": signal.get("source_locator"),
                        },
                    )
                )
        return observations

    def _element_observations(self, document: Document) -> list[_Observation]:
        observations: list[_Observation] = []
        page_indexes = {page.page_id: page.page_index for page in document.pages}
        for element in document.elements:
            label = _numeric_label(element.text)
            if label is None or element.bbox is None:
                continue
            normalized_bbox = element.bbox.normalized
            parser_type = str(element.metadata.get("mineru_type") or "").casefold()
            if not self._is_marginal(normalized_bbox) and parser_type != "page_number":
                continue
            page_index = page_indexes.get(element.page_id)
            if page_index is None:
                continue
            observations.append(
                _Observation(
                    page_index=page_index,
                    label=label,
                    source=PageLabelSource.MARGINAL_ELEMENT,
                    confidence=0.98 if parser_type == "page_number" else 0.94,
                    source_element_id=element.element_id,
                    normalized_bbox=normalized_bbox,
                    evidence={
                        "rule": "numeric_marginal_element",
                        "text": element.text,
                        "mineru_type": parser_type or None,
                    },
                )
            )
        return observations

    def _pdf_observations(
        self,
        source_pdf: Path,
        pages: dict[int, Page],
    ) -> list[_Observation]:
        observations: list[_Observation] = []
        pdf = pdfium.PdfDocument(source_pdf)
        try:
            for page_index in range(min(len(pdf), len(pages))):
                label = _numeric_label(pdf.get_page_label(page_index))
                if label is not None:
                    observations.append(
                        _Observation(
                            page_index=page_index,
                            label=label,
                            source=PageLabelSource.PDF_PAGE_LABEL,
                            confidence=0.995,
                            source_element_id=None,
                            normalized_bbox=None,
                            evidence={"rule": "pdf_page_label_dictionary"},
                        )
                    )
        finally:
            pdf.close()

        for page_index, lines in extract_pdf_text_lines(source_pdf).items():
            if page_index not in pages:
                continue
            for line in lines:
                label = _numeric_label(line.text)
                normalized_bbox = _normalized_pdf_bbox(line)
                if (
                    label is None
                    or normalized_bbox is None
                    or not self._is_marginal(normalized_bbox)
                ):
                    continue
                observations.append(
                    _Observation(
                        page_index=page_index,
                        label=label,
                        source=PageLabelSource.PDF_TEXT_LAYER,
                        confidence=0.93,
                        source_element_id=None,
                        normalized_bbox=normalized_bbox,
                        evidence={
                            "rule": "numeric_marginal_pdf_text",
                            "text": line.text,
                            "line_index": line.line_index,
                        },
                    )
                )
        return observations

    def _is_marginal(
        self,
        bbox: tuple[float, float, float, float],
    ) -> bool:
        _, y1, _, y2 = bbox
        return y2 <= self.top_boundary or y1 >= self.bottom_boundary

    @staticmethod
    def _deduplicate(observations: list[_Observation]) -> list[_Observation]:
        by_key: dict[tuple[object, ...], _Observation] = {}
        for observation in observations:
            bbox_key = (
                tuple(round(value, 4) for value in observation.normalized_bbox)
                if observation.normalized_bbox is not None
                else None
            )
            key = (
                observation.page_index,
                observation.label,
                observation.source,
                observation.source_element_id,
                bbox_key,
            )
            current = by_key.get(key)
            if current is None or observation.confidence > current.confidence:
                by_key[key] = observation
        return sorted(
            by_key.values(),
            key=lambda item: (item.page_index, item.label, item.source.value),
        )

    @staticmethod
    def _has_sequence_support(
        candidate: _Observation,
        observations: list[_Observation],
    ) -> bool:
        for other in observations:
            if other is candidate:
                continue
            page_delta = other.page_index - candidate.page_index
            label_delta = other.label - candidate.label
            if page_delta == 0 and abs(label_delta) == 1:
                if (
                    candidate.normalized_bbox is not None
                    and other.normalized_bbox is not None
                ):
                    candidate_x, candidate_y = _bbox_center(
                        candidate.normalized_bbox
                    )
                    other_x, other_y = _bbox_center(other.normalized_bbox)
                    # A true two-page spread normally places the two labels at
                    # opposite outer corners on the same baseline.  Nearby
                    # list numbers or chart ticks must not validate each other.
                    if abs(candidate_x - other_x) >= 0.45 and abs(candidate_y - other_y) <= 0.025:
                        return True
                continue
            if page_delta == 0 or label_delta == 0:
                continue
            if page_delta * label_delta < 0:
                continue
            physical_gap = abs(page_delta)
            label_gap = abs(label_delta)
            if (
                candidate.normalized_bbox is not None
                and other.normalized_bbox is not None
            ):
                candidate_x, candidate_y = _bbox_center(
                    candidate.normalized_bbox
                )
                other_x, other_y = _bbox_center(other.normalized_bbox)
                alternating_corner_sequence = (
                    label_gap == physical_gap
                    and abs(candidate_y - other_y) <= 0.07
                )
                if (
                    abs(candidate_x - other_x) > 0.20
                    and not alternating_corner_sequence
                ):
                    continue
            if physical_gap <= 2 and physical_gap <= label_gap <= physical_gap * 2:
                return True
        return False

    def _bounded_interpolation(
        self,
        observations: list[_Observation],
        confirmed: dict[int, set[int]],
        confirmation_evidence: dict[tuple[int, int], list[str]],
    ) -> list[_Observation]:
        inferred: list[_Observation] = []
        observed_by_page: dict[int, set[int]] = defaultdict(set)
        for observation in observations:
            observed_by_page[observation.page_index].add(observation.label)
        observed_pages = sorted(observed_by_page)
        for left_page, right_page in zip(observed_pages, observed_pages[1:]):
            physical_gap = right_page - left_page - 1
            if physical_gap <= 0 or physical_gap > self.maximum_bounded_gap:
                continue
            left_label = max(observed_by_page[left_page])
            right_label = min(observed_by_page[right_page])
            missing_label_count = right_label - left_label - 1
            if missing_label_count <= 0 or missing_label_count % physical_gap:
                continue
            labels_per_page = missing_label_count // physical_gap
            if labels_per_page not in {1, 2}:
                continue
            confirmed[left_page].update(observed_by_page[left_page])
            confirmed[right_page].update(observed_by_page[right_page])
            for label in observed_by_page[left_page]:
                confirmation_evidence[(left_page, label)].append(
                    "bounded_sequence_endpoint"
                )
            for label in observed_by_page[right_page]:
                confirmation_evidence[(right_page, label)].append(
                    "bounded_sequence_endpoint"
                )
            next_label = left_label + 1
            for page_index in range(left_page + 1, right_page):
                labels = list(range(next_label, next_label + labels_per_page))
                next_label += labels_per_page
                for label in labels:
                    if label in confirmed.get(page_index, set()):
                        continue
                    confirmed[page_index].add(label)
                    confirmation_evidence[(page_index, label)].append(
                        "bounded_sequence_interpolation"
                    )
                    inferred.append(
                        _Observation(
                            page_index=page_index,
                            label=label,
                            source=PageLabelSource.SEQUENCE_INFERENCE,
                            confidence=0.87,
                            source_element_id=None,
                            normalized_bbox=None,
                            evidence={
                                "rule": "bounded_sequence_interpolation",
                                "left_page_index": left_page,
                                "right_page_index": right_page,
                                "labels_per_physical_page": labels_per_page,
                            },
                        )
                    )
        return inferred

    def _backward_sequence_inference(
        self,
        pages: dict[int, Page],
        confirmed: dict[int, set[int]],
        confirmation_evidence: dict[tuple[int, int], list[str]],
    ) -> list[_Observation]:
        inferred: list[_Observation] = []
        singletons = {
            page_index: next(iter(labels))
            for page_index, labels in confirmed.items()
            if len(labels) == 1
        }
        ordered = sorted(singletons)
        runs: list[list[int]] = []
        current: list[int] = []
        for page_index in ordered:
            if (
                current
                and page_index == current[-1] + 1
                and singletons[page_index] == singletons[current[-1]] + 1
            ):
                current.append(page_index)
            else:
                if current:
                    runs.append(current)
                current = [page_index]
        if current:
            runs.append(current)

        for run in runs:
            if len(run) < self.minimum_extrapolation_run:
                continue
            first_page = run[0]
            first_label = singletons[first_page]
            page_index = first_page - 1
            label = first_label - 1
            while page_index in pages and label >= 1:
                if confirmed.get(page_index):
                    break
                confirmed[page_index].add(label)
                confirmation_evidence[(page_index, label)].append(
                    "backward_monotonic_sequence"
                )
                inferred.append(
                    _Observation(
                        page_index=page_index,
                        label=label,
                        source=PageLabelSource.SEQUENCE_INFERENCE,
                        confidence=0.82,
                        source_element_id=None,
                        normalized_bbox=None,
                        evidence={
                            "rule": "backward_monotonic_sequence",
                            "anchor_run_page_indexes": run,
                            "anchor_run_labels": [singletons[item] for item in run],
                        },
                    )
                )
                page_index -= 1
                label -= 1
        return inferred

    @staticmethod
    def _missing_spread_alias_inference(
        observations: list[_Observation],
        confirmed: dict[int, set[int]],
        confirmation_evidence: dict[tuple[int, int], list[str]],
    ) -> list[_Observation]:
        """Fill an unprinted half of a locally established two-page spread.

        A gap of two between adjacent physical pages is not sufficient on its
        own: a nearby physical page must explicitly contain two opposite-side
        labels.  This prevents an ordinary document with a genuinely skipped
        printed number from being reinterpreted as a spread.
        """

        inferred: list[_Observation] = []
        observation_by_key: dict[tuple[int, int], _Observation] = {}
        for observation in observations:
            key = (observation.page_index, observation.label)
            current = observation_by_key.get(key)
            if current is None or observation.confidence > current.confidence:
                observation_by_key[key] = observation

        def has_explicit_double_near(page_index: int) -> bool:
            for nearby in range(page_index - 2, page_index + 3):
                labels = {
                    observation.label
                    for observation in observations
                    if observation.page_index == nearby
                }
                if len(labels) >= 2 and max(labels) - min(labels) == 1:
                    return True
            return False

        changed = True
        while changed:
            changed = False
            for page_index in sorted(confirmed):
                labels = confirmed[page_index]
                if len(labels) != 1 or not has_explicit_double_near(page_index):
                    continue
                label = next(iter(labels))
                observation = observation_by_key.get((page_index, label))
                if observation is None or observation.normalized_bbox is None:
                    continue
                center_x, _ = _bbox_center(observation.normalized_bbox)
                inferred_label: int | None = None
                evidence_page: int | None = None
                if center_x >= 0.70:
                    for nearby in range(page_index - 1, page_index - 4, -1):
                        previous = confirmed.get(nearby, set())
                        distance = page_index - nearby
                        if (
                            len(previous) >= 2
                            and max(previous) - min(previous) == 1
                            and max(previous) + (2 * distance) == label
                        ):
                            inferred_label = label - 1
                            evidence_page = nearby
                            break
                    if inferred_label is None:
                        for nearby in range(page_index + 1, page_index + 4):
                            following = confirmed.get(nearby, set())
                            distance = nearby - page_index
                            if (
                                len(following) >= 2
                                and max(following) - min(following) == 1
                                and max(following) - (2 * distance) == label
                            ):
                                inferred_label = label - 1
                                evidence_page = nearby
                                break
                elif center_x <= 0.30:
                    for nearby in range(page_index + 1, page_index + 4):
                        following = confirmed.get(nearby, set())
                        distance = nearby - page_index
                        if (
                            len(following) >= 2
                            and max(following) - min(following) == 1
                            and min(following) - (2 * distance) == label
                        ):
                            inferred_label = label + 1
                            evidence_page = nearby
                            break
                    if inferred_label is None:
                        for nearby in range(page_index - 1, page_index - 4, -1):
                            previous = confirmed.get(nearby, set())
                            distance = page_index - nearby
                            if (
                                len(previous) >= 2
                                and max(previous) - min(previous) == 1
                                and min(previous) + (2 * distance) == label
                            ):
                                inferred_label = label + 1
                                evidence_page = nearby
                                break
                if inferred_label is None or inferred_label < 1:
                    continue
                confirmed[page_index].add(inferred_label)
                confirmation_evidence[(page_index, inferred_label)].append(
                    "local_two_page_spread_gap"
                )
                inferred.append(
                    _Observation(
                        page_index=page_index,
                        label=inferred_label,
                        source=PageLabelSource.SEQUENCE_INFERENCE,
                        confidence=0.84,
                        source_element_id=None,
                        normalized_bbox=None,
                        evidence={
                            "rule": "local_two_page_spread_gap",
                            "observed_label": label,
                            "supporting_page_index": evidence_page,
                        },
                    )
                )
                changed = True
        return inferred

    @staticmethod
    def _candidate(
        document: Document,
        observation: _Observation,
        *,
        status: PageLabelStatus,
        confidence: float,
    ) -> PageLabelCandidate:
        bbox_key = (
            ":".join(f"{value:.4f}" for value in observation.normalized_bbox)
            if observation.normalized_bbox is not None
            else "none"
        )
        candidate_id = "page-label:" + stable_digest(
            document.document_id,
            str(observation.page_index),
            str(observation.label),
            observation.source.value,
            observation.source_element_id or "none",
            bbox_key,
        )
        return PageLabelCandidate(
            candidate_id=candidate_id,
            label=str(observation.label),
            source=observation.source,
            status=status,
            confidence=confidence,
            source_element_id=observation.source_element_id,
            normalized_bbox=observation.normalized_bbox,
            evidence=observation.evidence,
        )


def _numeric_label(text: str | None) -> int | None:
    if text is None:
        return None
    match = _LABEL_TEXT.fullmatch(text) or _CHINESE_LABEL_TEXT.fullmatch(text)
    if match is None:
        return None
    value = int(match.group("label"))
    return value if value >= 1 else None


def _normalized_pdf_bbox(
    line: PdfTextLine,
) -> tuple[float, float, float, float] | None:
    width = line.source_page_width
    height = line.source_page_height
    if not width or not height or width <= 0 or height <= 0:
        return None
    x1, bottom, x2, top = line.bbox
    normalized = (
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, 1.0 - top / height)),
        max(0.0, min(1.0, x2 / width)),
        max(0.0, min(1.0, 1.0 - bottom / height)),
    )
    if normalized[0] >= normalized[2] or normalized[1] >= normalized[3]:
        return None
    return normalized


def _bbox_center(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
