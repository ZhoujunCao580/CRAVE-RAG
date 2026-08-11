"""Transparent BM25Okapi ranking over retrieval-only SearchUnits."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from softdoc.retrieval.models import (
    BM25Config,
    BM25ElementCandidate,
    BM25SearchResult,
    BM25TraceEntry,
    MatchedOffset,
    SearchUnit,
    SearchUnitBuildResult,
    SubQuestionInput,
)
from softdoc.retrieval.tokenization import bm25_token_spans


@dataclass(frozen=True)
class _IndexedUnit:
    unit: SearchUnit
    terms: tuple[str, ...]
    frequencies: Counter[str]


@dataclass(frozen=True)
class _ScoredUnit:
    indexed: _IndexedUnit
    score: float
    matched_terms: tuple[str, ...]


class BM25Index:
    """One-document BM25 index with deterministic Element-level merging."""

    def __init__(
        self,
        build_result: SearchUnitBuildResult,
        config: BM25Config | None = None,
    ) -> None:
        self.build_result = build_result
        self.config = config or BM25Config()
        self._units = [self._index_unit(unit) for unit in build_result.units]
        self._document_frequency: Counter[str] = Counter()
        for indexed in self._units:
            self._document_frequency.update(set(indexed.terms))
        self._average_length = (
            sum(len(indexed.terms) for indexed in self._units) / len(self._units)
            if self._units
            else 0.0
        )

    def search(self, subquestion: SubQuestionInput) -> BM25SearchResult:
        query_terms = _unique_terms(subquestion.text)
        trace: list[BM25TraceEntry] = []
        if not query_terms:
            trace.append(
                BM25TraceEntry(
                    code="empty_query_terms",
                    description="The SubQuestion produced no lexical BM25 terms.",
                )
            )
            return self._result(subquestion, [], trace)

        scored: list[_ScoredUnit] = []
        for indexed in self._units:
            matched = tuple(
                term for term in query_terms if indexed.frequencies.get(term, 0) > 0
            )
            if not matched:
                continue
            score = self._score(indexed, matched)
            if score > 0.0:
                scored.append(
                    _ScoredUnit(
                        indexed=indexed,
                        score=score,
                        matched_terms=matched,
                    )
                )

        if not scored:
            trace.append(
                BM25TraceEntry(
                    code="no_lexical_match",
                    description="No SearchUnit contains any normalized query term.",
                    data={"query_terms": list(query_terms)},
                )
            )
            return self._result(subquestion, [], trace)

        best_by_element: dict[str, _ScoredUnit] = {}
        for item in scored:
            element_id = item.indexed.unit.element_id
            current = best_by_element.get(element_id)
            if current is None or _part_preference(item) < _part_preference(current):
                best_by_element[element_id] = item

        ordered = sorted(best_by_element.values(), key=_element_preference)
        candidates = [
            _candidate(item, rank)
            for rank, item in enumerate(ordered, start=1)
        ]
        trace.append(
            BM25TraceEntry(
                code="bm25_ranked",
                description=(
                    "BM25 scored SearchUnits and merged them to one candidate per Element."
                ),
                data={
                    "query_terms": list(query_terms),
                    "matched_search_units": len(scored),
                    "ranked_elements": len(candidates),
                    "merge_rule": "maximum_part_score",
                    "tie_break": (
                        "score_desc,page_index,reading_order,element_id"
                    ),
                },
            )
        )
        return self._result(subquestion, candidates, trace)

    def _score(
        self,
        indexed: _IndexedUnit,
        matched_terms: tuple[str, ...],
    ) -> float:
        if not self._units or self._average_length <= 0.0:
            return 0.0
        length = len(indexed.terms)
        score = 0.0
        for term in matched_terms:
            frequency = indexed.frequencies[term]
            document_frequency = self._document_frequency[term]
            inverse_document_frequency = math.log(
                1.0
                + (
                    len(self._units) - document_frequency + 0.5
                )
                / (document_frequency + 0.5)
            )
            denominator = frequency + self.config.k1 * (
                1.0
                - self.config.b
                + self.config.b * length / self._average_length
            )
            score += inverse_document_frequency * (
                frequency * (self.config.k1 + 1.0) / denominator
            )
        return score

    @staticmethod
    def _index_unit(unit: SearchUnit) -> _IndexedUnit:
        terms = tuple(span.term for span in bm25_token_spans(unit.search_text))
        return _IndexedUnit(
            unit=unit,
            terms=terms,
            frequencies=Counter(terms),
        )

    def _result(
        self,
        subquestion: SubQuestionInput,
        candidates: list[BM25ElementCandidate],
        trace: list[BM25TraceEntry],
    ) -> BM25SearchResult:
        return BM25SearchResult(
            subquestion_id=subquestion.subquestion_id,
            document_id=self.build_result.document_id,
            index_version=self.build_result.index_version,
            total_search_units=len(self._units),
            total_candidates=len(candidates),
            candidates=candidates,
            trace=trace,
        )


def _unique_terms(text: str) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for span in bm25_token_spans(text):
        if span.term not in seen:
            seen.add(span.term)
            result.append(span.term)
    return tuple(result)


def _part_preference(item: _ScoredUnit) -> tuple[float, int, str]:
    unit = item.indexed.unit
    return (-item.score, unit.part_index, unit.search_unit_id)


def _element_preference(item: _ScoredUnit) -> tuple[float, int, int, str]:
    unit = item.indexed.unit
    return (-item.score, unit.page_index, unit.reading_order, unit.element_id)


def _candidate(item: _ScoredUnit, rank: int) -> BM25ElementCandidate:
    unit = item.indexed.unit
    matched = set(item.matched_terms)
    offsets = [
        MatchedOffset(term=span.term, start=span.start, end=span.end)
        for span in bm25_token_spans(unit.search_text)
        if span.term in matched
    ]
    return BM25ElementCandidate(
        element_id=unit.element_id,
        bm25_score=item.score,
        bm25_rank=rank,
        matched_search_unit_id=unit.search_unit_id,
        matched_part=unit.part_index,
        matched_terms=list(item.matched_terms),
        matched_offsets=offsets,
        page_id=unit.page_id,
        page_number=unit.page_number,
        section_id=unit.section_id,
        element_type=unit.element_type,
    )
