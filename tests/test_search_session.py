from __future__ import annotations

import pytest

from softdoc.models import ContentAvailability, ElementType
from softdoc.retrieval import (
    AnchorKind,
    AnchorResolution,
    AnchorResolutionStatus,
    AnchorTargetType,
    BM25Index,
    BM25ElementCandidate,
    BM25SearchResult,
    CandidateMergePolicy,
    CandidateSelectionRoute,
    DenseElementCandidate,
    DenseSearchResult,
    ExactAnchorMatch,
    ExactLookupResult,
    MatchedOffset,
    PreviewMatchScope,
    RetrievalSource,
    SearchSession,
    SearchSessionBuilder,
    SearchSessionConfig,
    SearchSessionNavigator,
    SearchUnit,
    SearchUnitBuildResult,
    SearchUnitConfig,
    SnippetSource,
    SubQuestionInput,
    VisualElementCandidate,
    VisualSearchResult,
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


def _visual() -> VisualSearchResult:
    candidates = []
    for rank, index in enumerate(range(12, 19), start=1):
        candidates.append(
            VisualElementCandidate(
                element_id=f"element:{index:02d}",
                visual_asset_id=f"visual:{index:02d}",
                visual_score=1.0 - rank / 100.0,
                visual_rank=rank,
                page_id=f"page:{index // 4:02d}",
                page_number=index // 4 + 1,
                section_id="section:methods",
                section_path=["Methods"],
                display_label=f"Figure {index}",
                preview_text=f"Visual figure about needle-{index} evidence.",
                element_type=ElementType.FIGURE,
                content_availability=ContentAvailability.VISUAL_ONLY,
            )
        )
    return VisualSearchResult(
        subquestion_id="Q1",
        document_id=DOCUMENT_ID,
        index_fingerprint="visual-index-v1",
        model_name="mock-visual",
        total_visual_assets=len(candidates),
        total_candidates=len(candidates),
        candidates=candidates,
    )


def _table_search_units() -> SearchUnitBuildResult:
    content = (
        "Sitemap #\tType\tIssue\tDescription\tIssues count\n"
        "videositemap.xml\tWarnings\tURLs not accessible\tHTTP status error\t159\n"
        "videositemap.xml\tWarnings\tURLs unreachable\tConnection error\t1732\n"
        "videositemap.xml\tWarnings\tURLs timedout\tRequest timeout\t504"
    )
    unit = SearchUnit(
        search_unit_id="unit:table",
        document_id=DOCUMENT_ID,
        element_id="element:table",
        part_index=0,
        part_count=1,
        search_text=content,
        content_text=content,
        content_search_char_start=0,
        content_search_char_end=len(content),
        source_char_start=0,
        source_char_end=len(content),
        page_id="page:table",
        page_index=0,
        page_number=1,
        reading_order=0,
        element_type=ElementType.TABLE,
        content_availability=ContentAvailability.MIXED,
        table_header_cells=[
            "Sitemap #",
            "Type",
            "Issue",
            "Description",
            "Issues count",
        ],
        table_header_source_element_id="element:table",
        index_version=INDEX_VERSION,
    )
    return SearchUnitBuildResult(
        document_id=DOCUMENT_ID,
        index_version=INDEX_VERSION,
        config=SearchUnitConfig(index_version=INDEX_VERSION),
        units=[unit],
    )


def test_table_preview_keeps_headers_and_selects_query_relevant_row() -> None:
    units = _table_search_units()
    unit = units.units[0]
    start = unit.search_text.index("Issues")
    bm25 = BM25SearchResult(
        subquestion_id="Q1",
        document_id=DOCUMENT_ID,
        index_version=INDEX_VERSION,
        total_search_units=1,
        total_candidates=1,
        candidates=[
            BM25ElementCandidate(
                element_id=unit.element_id,
                bm25_score=1.0,
                bm25_rank=1,
                matched_search_unit_id=unit.search_unit_id,
                matched_part=0,
                matched_terms=["issues"],
                matched_offsets=[
                    MatchedOffset(term="issues", start=start, end=start + 6)
                ],
                page_id=unit.page_id,
                page_number=unit.page_number,
                element_type=unit.element_type,
            )
        ],
    )
    session = SearchSessionBuilder().create(
        subquestion=SubQuestionInput(
            subquestion_id="Q1",
            text="How many questions are there about URL timeout issues?",
        ),
        search_units=units,
        bm25=bm25,
    )

    _, batch = SearchSessionNavigator(units).next_batch(session)
    preview = batch.candidate_previews[0]

    assert preview.snippet_source == SnippetSource.TABLE_PREVIEW
    assert "Columns: Sitemap # | Type | Issue | Description | Issues count" in (
        preview.matched_snippet
    )
    assert "URLs timedout" in preview.matched_snippet
    assert "504" in preview.matched_snippet


def test_table_preview_preserves_rightmost_value_after_long_description() -> None:
    units = _table_search_units()
    unit = units.units[0]
    long_content = unit.content_text.replace(
        "Request timeout",
        "When we tested a sample of the URLs, some were not accessible to "
        "Googlebot because repeated network timeouts prevented retrieval for a "
        "long explanatory reason that should be compacted",
    )
    units = units.model_copy(
        update={
            "units": [
                unit.model_copy(
                    update={
                        "search_text": long_content,
                        "content_text": long_content,
                        "content_search_char_end": len(long_content),
                        "source_char_end": len(long_content),
                    }
                )
            ]
        }
    )
    bm25 = BM25Index(units).search(
        SubQuestionInput(
            subquestion_id="Q1",
            text="How many questions are there about URL timeout issues?",
        )
    )
    session = SearchSessionBuilder().create(
        subquestion=SubQuestionInput(
            subquestion_id="Q1",
            text="How many questions are there about URL timeout issues?",
        ),
        search_units=units,
        bm25=bm25,
    )

    _, batch = SearchSessionNavigator(units).next_batch(session)
    preview = batch.candidate_previews[0]

    assert "URLs timedout" in preview.matched_snippet
    assert "Issues count=504" in preview.matched_snippet
    assert len(preview.matched_snippet) <= 640


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
    exact_match = session.exact_anchor_matches[0]
    assert exact_match.matched_by == [RetrievalSource.BM25, RetrievalSource.DENSE]
    assert exact_match.bm25_rank == 1
    assert exact_match.dense_rank == 2
    assert exact_match.rrf_score is not None
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
    assert merged.rrf_score is not None
    assert session.unresolved_anchors[0].normalized_label == "99"


def test_round_robin_remains_an_explicit_reproducible_baseline() -> None:
    units = _search_units()
    session = SearchSessionBuilder(
        SearchSessionConfig(
            merge_policy=CandidateMergePolicy.ROUND_ROBIN_BM25_FIRST
        )
    ).create(
        subquestion=_question(),
        search_units=units,
        exact=_exact(),
        bm25=_bm25(units),
        dense=_dense(units),
    )

    assert session.ranked_candidate_ids[:5] == [
        "element:05",
        "element:01",
        "element:02",
        "element:06",
        "element:03",
    ]
    assert all(item.rrf_score is None for item in session.candidate_catalog)


def test_fixed_quota_builds_three_bm25_then_two_dense_with_unique_backfill() -> None:
    units = _search_units()
    session = SearchSessionBuilder(
        SearchSessionConfig(
            merge_policy=CandidateMergePolicy.FIXED_QUOTA,
            batch_size=5,
            bm25_quota=3,
            dense_quota=2,
        )
    ).create(
        subquestion=_question(),
        search_units=units,
        exact=_exact(),
        bm25=_bm25(units),
        dense=_dense(units),
    )

    assert session.ranked_candidate_ids[:5] == [
        "element:01",
        "element:02",
        "element:03",
        "element:05",
        "element:06",
    ]
    assert session.ranked_candidate_ids[5:10] == [
        "element:04",
        "element:07",
        "element:08",
        "element:09",
        "element:10",
    ]
    assert len(session.ranked_candidate_ids) == len(set(session.ranked_candidate_ids))
    assert all(item.rrf_score is None for item in session.candidate_catalog)


def test_fixed_quota_requires_quotas_to_equal_batch_size() -> None:
    with pytest.raises(ValueError, match="must sum to batch_size"):
        SearchSessionConfig(
            merge_policy=CandidateMergePolicy.FIXED_QUOTA,
            batch_size=5,
            bm25_quota=2,
            dense_quota=2,
        )


def test_fixed_text_visual_quota_mixes_three_text_and_two_visual_candidates() -> None:
    units = _search_units()
    config = SearchSessionConfig(
        merge_policy=CandidateMergePolicy.FIXED_TEXT_VISUAL_QUOTA,
        batch_size=5,
        text_quota=3,
        visual_quota=2,
    )
    builder = SearchSessionBuilder(config)
    first_session = builder.create(
        subquestion=_question(),
        search_units=units,
        bm25=_bm25(units),
        dense=_dense(units),
        visual=_visual(),
    )
    repeated_session = builder.create(
        subquestion=_question(),
        search_units=units,
        bm25=_bm25(units),
        dense=_dense(units),
        visual=_visual(),
    )

    assert first_session.ranked_candidate_ids == repeated_session.ranked_candidate_ids
    first_five = first_session.candidate_catalog[:5]
    assert sum(
        item.selection_route == CandidateSelectionRoute.TEXT for item in first_five
    ) == 3
    assert sum(
        item.selection_route == CandidateSelectionRoute.VISUAL for item in first_five
    ) == 2
    assert len(first_session.ranked_candidate_ids) == len(
        set(first_session.ranked_candidate_ids)
    )

    _, batch = SearchSessionNavigator(units).next_batch(first_session)
    visual_previews = [
        item
        for item in batch.candidate_previews
        if item.preview_source == RetrievalSource.VISUAL_DENSE
    ]
    assert len(visual_previews) == 2
    assert all(item.snippet_source == SnippetSource.VISUAL_METADATA for item in visual_previews)
    assert all(item.matched_search_unit_id is None for item in visual_previews)
    assert all(item.visual_asset_id for item in visual_previews)


def test_fixed_text_visual_quota_requires_visual_result() -> None:
    units = _search_units()
    builder = SearchSessionBuilder(
        SearchSessionConfig(
            merge_policy=CandidateMergePolicy.FIXED_TEXT_VISUAL_QUOTA
        )
    )
    with pytest.raises(ValueError, match="requires a Visual search result"):
        builder.create(
            subquestion=_question(),
            search_units=units,
            bm25=_bm25(units),
            dense=_dense(units),
        )


def test_default_candidate_policy_is_the_frozen_weighted_rrf_configuration() -> None:
    config = SearchSessionConfig()

    assert config.merge_policy == CandidateMergePolicy.WEIGHTED_RRF
    assert config.rrf_k == 20
    assert config.bm25_weight == 1.0
    assert config.dense_weight == 1.25
    assert config.batch_size == 5


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
    assert preview.preview_source == RetrievalSource.DENSE
    assert preview.match_scope == PreviewMatchScope.CONTENT
    assert preview.snippet_source == SnippetSource.SEARCH_UNIT_TEXT
    assert preview.snippet_source_id == preview.matched_search_unit_id
    assert "Candidate 5" in preview.matched_snippet


def test_rrf_preview_uses_the_source_with_the_larger_rank_contribution() -> None:
    units = _search_units(1)
    original = units.units[0]
    dense_unit = original.model_copy(
        update={
            "search_unit_id": "unit:00:dense",
            "search_text": "Methods\nDense-specific semantic evidence.",
            "content_text": "Dense-specific semantic evidence.",
            "source_char_end": len("Dense-specific semantic evidence."),
        }
    )
    units = units.model_copy(update={"units": [original, dense_unit]})
    bm25 = _bm25(_search_units(1))
    dense = DenseSearchResult(
        subquestion_id="Q1",
        document_id=DOCUMENT_ID,
        index_version=INDEX_VERSION,
        model_name="mock-e5",
        total_search_units=2,
        total_dense_segments=2,
        total_candidates=1,
        candidates=[
            DenseElementCandidate(
                element_id=original.element_id,
                dense_score=0.99,
                dense_rank=1,
                matched_search_unit_id=dense_unit.search_unit_id,
                matched_part=0,
                matched_dense_segment_id="segment:dense",
                matched_segment_index=0,
                matched_text_char_start=0,
                matched_text_char_end=len(dense_unit.search_text),
                page_id=original.page_id,
                page_number=original.page_number,
                section_id=original.section_id,
                element_type=original.element_type,
            )
        ],
    )
    session = SearchSessionBuilder().create(
        subquestion=_question(), search_units=units, bm25=bm25, dense=dense
    )

    _, batch = SearchSessionNavigator(units).next_batch(session)
    preview = batch.candidate_previews[0]

    assert preview.matched_search_unit_id == dense_unit.search_unit_id
    assert preview.preview_source == RetrievalSource.DENSE
    assert preview.match_scope == PreviewMatchScope.MIXED
    assert "Dense-specific" in preview.matched_snippet


def test_metadata_only_bm25_hit_does_not_hide_dense_content_preview() -> None:
    units = _search_units(1)
    unit = units.units[0]
    bm25 = BM25SearchResult(
        subquestion_id="Q1",
        document_id=DOCUMENT_ID,
        index_version=INDEX_VERSION,
        total_search_units=1,
        total_candidates=1,
        candidates=[
            BM25ElementCandidate(
                element_id=unit.element_id,
                bm25_score=10.0,
                bm25_rank=1,
                matched_search_unit_id=unit.search_unit_id,
                matched_part=0,
                matched_terms=["Methods"],
                matched_offsets=[MatchedOffset(term="Methods", start=0, end=7)],
                page_id=unit.page_id,
                page_number=unit.page_number,
                section_id=unit.section_id,
                element_type=unit.element_type,
            )
        ],
    )
    dense = DenseSearchResult(
        subquestion_id="Q1",
        document_id=DOCUMENT_ID,
        index_version=INDEX_VERSION,
        model_name="mock-e5",
        total_search_units=1,
        total_dense_segments=1,
        total_candidates=1,
        candidates=[
            DenseElementCandidate(
                element_id=unit.element_id,
                dense_score=0.9,
                dense_rank=1,
                matched_search_unit_id=unit.search_unit_id,
                matched_part=0,
                matched_dense_segment_id="segment:content",
                matched_segment_index=0,
                matched_text_char_start=unit.search_text.index("Candidate"),
                matched_text_char_end=len(unit.search_text),
                page_id=unit.page_id,
                page_number=unit.page_number,
                section_id=unit.section_id,
                element_type=unit.element_type,
            )
        ],
    )
    session = SearchSessionBuilder(
        SearchSessionConfig(bm25_weight=3.0, dense_weight=1.0)
    ).create(
        subquestion=_question(),
        search_units=units,
        bm25=bm25,
        dense=dense,
    )

    _, batch = SearchSessionNavigator(units).next_batch(session)
    preview = batch.candidate_previews[0]

    assert preview.preview_source == RetrievalSource.DENSE
    assert preview.match_scope == PreviewMatchScope.CONTENT
    assert "Candidate 0" in preview.matched_snippet
    scope_audit = next(
        item
        for item in session.retrieval_trace
        if item.code == "search_metadata_scope_audit"
    )
    assert scope_audit.data["bm25_metadata_only"] == 1
    assert scope_audit.data["dense_metadata_only"] == 0


def test_labeled_visual_candidate_exposes_display_label() -> None:
    original = _search_units(1).units[0]
    search_text = "Experiments\nFigure 5"
    content_text = "Figure 5"
    content_start = search_text.index(content_text)
    visual = original.model_copy(
        update={
            "search_text": search_text,
            "content_text": content_text,
            "display_label": "Figure 5",
            "content_search_char_start": content_start,
            "content_search_char_end": len(search_text),
            "source_char_end": len(content_text),
            "element_type": ElementType.FIGURE,
            "content_availability": ContentAvailability.VISUAL_ONLY,
        }
    )
    units = _search_units(1).model_copy(update={"units": [visual]})
    bm25 = BM25SearchResult(
        subquestion_id="Q1",
        document_id=DOCUMENT_ID,
        index_version=INDEX_VERSION,
        total_search_units=1,
        total_candidates=1,
        candidates=[
            BM25ElementCandidate(
                element_id=visual.element_id,
                bm25_score=5.0,
                bm25_rank=1,
                matched_search_unit_id=visual.search_unit_id,
                matched_part=0,
                matched_terms=["Figure", "5"],
                matched_offsets=[
                    MatchedOffset(
                        term="Figure",
                        start=content_start,
                        end=content_start + len("Figure"),
                    )
                ],
                page_id=visual.page_id,
                page_number=visual.page_number,
                section_id=visual.section_id,
                element_type=visual.element_type,
            )
        ],
    )
    session = SearchSessionBuilder().create(
        subquestion=_question(), search_units=units, bm25=bm25
    )

    updated, batch = SearchSessionNavigator(units).next_batch(session)
    restored = SearchSession.model_validate_json(updated.model_dump_json())
    preview = batch.candidate_previews[0]

    assert preview.display_label == "Figure 5"
    assert preview.preview_source == RetrievalSource.BM25
    assert preview.match_scope == PreviewMatchScope.CONTENT
    assert preview.content_availability == ContentAvailability.VISUAL_ONLY
    assert restored.candidate_catalog[0].display_label == "Figure 5"


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
