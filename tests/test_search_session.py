from __future__ import annotations

import pytest

from softdoc.models import ContentAvailability, ElementType
from softdoc.retrieval import (
    AnchorKind,
    AnchorResolution,
    AnchorResolutionStatus,
    AnchorTargetType,
    BM25ElementCandidate,
    BM25SearchResult,
    DenseElementCandidate,
    DenseSearchResult,
    ExactAnchorMatch,
    ExactLookupResult,
    MatchedOffset,
    RetrievalSource,
    SearchSession,
    SearchSessionBuilder,
    SearchSessionConfig,
    SearchSessionNavigator,
    SearchUnit,
    SearchUnitBuildResult,
    SearchUnitConfig,
    SubQuestionInput,
)


DOCUMENT_ID = "doc:test-session"
INDEX_VERSION = "session-test-v1"


def _search_units(count: int = 12) -> SearchUnitBuildResult:
    units = []
    for index in range(count):
        text = f"Methods\nCandidate {index} contains needle-{index} and evidence."
        units.append(
            SearchUnit(
                search_unit_id=f"unit:{index:02d}",
                document_id=DOCUMENT_ID,
                element_id=f"element:{index:02d}",
                part_index=0,
                part_count=1,
                search_text=text,
                content_text=text.split("\n", 1)[1],
                source_char_start=0,
                source_char_end=len(text.split("\n", 1)[1]),
                page_id=f"page:{index // 4:02d}",
                page_index=index // 4,
                page_number=index // 4 + 1,
                reading_order=index % 4,
                section_id="section:methods",
                section_path=["Methods"],
                element_type=ElementType.PARAGRAPH,
                content_availability=ContentAvailability.TEXT_ONLY,
                index_version=INDEX_VERSION,
            )
        )
    return SearchUnitBuildResult(
        document_id=DOCUMENT_ID,
        index_version=INDEX_VERSION,
        config=SearchUnitConfig(index_version=INDEX_VERSION),
        units=units,
    )


def _bm25(units: SearchUnitBuildResult) -> BM25SearchResult:
    candidates = []
    for rank, unit in enumerate(units.units, start=1):
        start = unit.search_text.index("needle")
        candidates.append(
            BM25ElementCandidate(
                element_id=unit.element_id,
                bm25_score=20.0 / rank,
                bm25_rank=rank,
                matched_search_unit_id=unit.search_unit_id,
                matched_part=0,
                matched_terms=["needle"],
                matched_offsets=[
                    MatchedOffset(term="needle", start=start, end=start + 6)
                ],
                page_id=unit.page_id,
                page_number=unit.page_number,
                section_id=unit.section_id,
                element_type=unit.element_type,
            )
        )
    return BM25SearchResult(
        subquestion_id="Q1",
        document_id=DOCUMENT_ID,
        index_version=INDEX_VERSION,
        total_search_units=len(units.units),
        total_candidates=len(candidates),
        candidates=candidates,
    )


def _dense(units: SearchUnitBuildResult) -> DenseSearchResult:
    order = [5, 0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 11]
    candidates = []
    for rank, index in enumerate(order[: len(units.units)], start=1):
        unit = units.units[index]
        start = unit.search_text.index("Candidate")
        candidates.append(
            DenseElementCandidate(
                element_id=unit.element_id,
                dense_score=1.0 - rank / 100.0,
                dense_rank=rank,
                matched_search_unit_id=unit.search_unit_id,
                matched_part=0,
                matched_dense_segment_id=f"segment:{index:02d}",
                matched_segment_index=0,
                matched_text_char_start=start,
                matched_text_char_end=len(unit.search_text),
                page_id=unit.page_id,
                page_number=unit.page_number,
                section_id=unit.section_id,
                element_type=unit.element_type,
            )
        )
    return DenseSearchResult(
        subquestion_id="Q1",
        document_id=DOCUMENT_ID,
        index_version=INDEX_VERSION,
        model_name="mock-e5",
        total_search_units=len(units.units),
        total_dense_segments=len(units.units),
        total_candidates=len(candidates),
        candidates=candidates,
    )


def _exact() -> ExactLookupResult:
    return ExactLookupResult(
        subquestion_id="Q1",
        document_id=DOCUMENT_ID,
        exact_anchor_matches=[
            ExactAnchorMatch(
                resolution_id="resolution:figure3",
                anchor_text="Figure 3",
                anchor_kind=AnchorKind.FIGURE,
                normalized_label="3",
                target_id="element:00",
                target_type=AnchorTargetType.PARAGRAPH,
                page_id="page:00",
                page_number=1,
                section_id="section:methods",
                resolution_method="fixture",
            )
        ],
        anchor_resolutions=[
            AnchorResolution(
                resolution_id="resolution:table99",
                anchor_text="Table 99",
                anchor_kind=AnchorKind.TABLE,
                normalized_label="99",
                source_span=(10, 18),
                status=AnchorResolutionStatus.UNRESOLVED,
                reason="target_not_found",
            )
        ],
    )


def _question() -> SubQuestionInput:
    return SubQuestionInput(subquestion_id="Q1", text="Find the supporting evidence")


def test_session_keeps_exact_separate_and_merges_retrieval_sources() -> None:
    units = _search_units()
    session = SearchSessionBuilder().create(
        subquestion=_question(),
        search_units=units,
        exact=_exact(),
        bm25=_bm25(units),
        dense=_dense(units),
    )

    assert [item.target_id for item in session.exact_anchor_matches] == [
        "element:00"
    ]
    assert "element:00" not in session.ranked_candidate_ids
    assert len(session.ranked_candidate_ids) == 11
    assert len(set(session.ranked_candidate_ids)) == 11
    assert session.ranked_candidate_ids[:5] == [
        "element:05",
        "element:01",
        "element:02",
        "element:06",
        "element:03",
    ]
    merged = next(
        item for item in session.candidate_catalog if item.element_id == "element:01"
    )
    assert merged.matched_by == [RetrievalSource.BM25, RetrievalSource.DENSE]
    assert merged.bm25_rank == 2
    assert merged.dense_rank == 4
    assert session.unresolved_anchors[0].normalized_label == "99"


def test_batches_are_resumable_and_old_previews_remain_available() -> None:
    units = _search_units()
    session = SearchSessionBuilder().create(
        subquestion=_question(),
        search_units=units,
        bm25=_bm25(units),
    )
    navigator = SearchSessionNavigator(units)

    after_first, first = navigator.next_batch(session)
    after_second, second = navigator.next_batch(after_first)
    after_third, third = navigator.next_batch(after_second)

    assert [item.element_id for item in first.candidate_previews] == [
        f"element:{index:02d}" for index in range(5)
    ]
    assert [item.element_id for item in second.candidate_previews] == [
        f"element:{index:02d}" for index in range(5, 10)
    ]
    assert [item.element_id for item in third.candidate_previews] == [
        "element:10",
        "element:11",
    ]
    assert first.next_cursor == 5 and not first.exhausted
    assert second.next_cursor == 10 and not second.exhausted
    assert third.next_cursor == 12 and third.exhausted
    assert after_third.exhausted
    assert len(navigator.preview_history(after_third)) == 12
    assert navigator.get_preview(after_third, "element:00") == (
        first.candidate_previews[0]
    )


def test_session_round_trip_can_resume_with_the_same_search_index() -> None:
    units = _search_units()
    builder = SearchSessionBuilder()
    navigator = SearchSessionNavigator(units)
    original = builder.create(
        subquestion=_question(), search_units=units, bm25=_bm25(units)
    )
    first_state, _ = navigator.next_batch(original)

    restored = SearchSession.model_validate_json(first_state.model_dump_json())
    second_state, second = SearchSessionNavigator(units).next_batch(restored)

    assert second.candidate_previews[0].element_id == "element:05"
    assert second_state.cursor == 10
    assert second_state.search_session_id == original.search_session_id


def test_opened_candidates_are_tracked_idempotently() -> None:
    units = _search_units()
    navigator = SearchSessionNavigator(units)
    session = SearchSessionBuilder().create(
        subquestion=_question(), search_units=units, bm25=_bm25(units)
    )
    shown, _ = navigator.next_batch(session)

    opened = navigator.mark_opened(shown, "element:02")
    opened_again = navigator.mark_opened(opened, "element:02")

    assert opened.opened_candidate_ids == ["element:02"]
    assert opened_again == opened
    with pytest.raises(ValueError, match="shown"):
        navigator.mark_opened(shown, "element:08")


def test_preview_is_a_bounded_original_text_window_without_relations() -> None:
    units = _search_units(1)
    long_text = "prefix " * 70 + "needle critical evidence " + "suffix " * 70
    unit = units.units[0].model_copy(
        update={
            "search_text": long_text,
            "content_text": long_text,
            "source_char_end": len(long_text),
        }
    )
    units = units.model_copy(update={"units": [unit]})
    start = long_text.index("needle")
    bm25 = _bm25(units)
    bm25.candidates[0].matched_offsets = [
        MatchedOffset(term="needle", start=start, end=start + 6)
    ]
    session = SearchSessionBuilder(
        SearchSessionConfig(
            batch_size=5,
            snippet_max_chars=100,
            snippet_context_chars=20,
        )
    ).create(subquestion=_question(), search_units=units, bm25=bm25)

    _, batch = SearchSessionNavigator(units).next_batch(session)
    preview = batch.candidate_previews[0]
    payload = preview.model_dump(mode="json")

    assert "needle" in preview.matched_snippet
    assert len(preview.matched_snippet) <= 100
    assert preview.snippet_truncated
    assert "relations" not in payload
    assert "text" not in payload
    assert "html" not in payload
    assert "image_path" not in payload


def test_dense_only_preview_uses_the_best_dense_search_unit() -> None:
    units = _search_units()
    session = SearchSessionBuilder().create(
        subquestion=_question(),
        search_units=units,
        dense=_dense(units),
    )

    _, batch = SearchSessionNavigator(units).next_batch(session, batch_size=1)
    preview = batch.candidate_previews[0]

    assert preview.element_id == "element:05"
    assert preview.matched_by == [RetrievalSource.DENSE]
    assert preview.dense_rank == 1
    assert preview.bm25_rank is None
    assert "Candidate 5" in preview.matched_snippet


def test_session_identity_is_stable_and_input_mismatches_fail() -> None:
    units = _search_units()
    builder = SearchSessionBuilder()
    first = builder.create(
        subquestion=_question(), search_units=units, bm25=_bm25(units)
    )
    second = builder.create(
        subquestion=_question(), search_units=units, bm25=_bm25(units)
    )
    assert first == second

    wrong = _bm25(units).model_copy(update={"document_id": "doc:wrong"})
    with pytest.raises(ValueError, match="another Document"):
        builder.create(
            subquestion=_question(), search_units=units, bm25=wrong
        )


def test_empty_ranking_is_immediately_exhausted() -> None:
    units = _search_units(0)
    session = SearchSessionBuilder().create(
        subquestion=_question(), search_units=units
    )
    updated, batch = SearchSessionNavigator(units).next_batch(session)

    assert updated.exhausted
    assert batch.exhausted
    assert batch.candidate_previews == []
    assert batch.next_cursor == 0
