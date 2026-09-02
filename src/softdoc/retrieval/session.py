"""Deterministic candidate previews and resumable retrieval sessions."""

from __future__ import annotations

from collections.abc import Iterable

from softdoc.ids import stable_digest
from softdoc.retrieval.models import (
    AnchorResolutionStatus,
    BM25ElementCandidate,
    BM25SearchResult,
    CandidateMergePolicy,
    CandidatePreview,
    CandidateSelectionRoute,
    DenseElementCandidate,
    DenseSearchResult,
    ExactAnchorMatch,
    ExactLookupResult,
    PreviewMatchScope,
    RetrievalSource,
    SearchBatch,
    SearchSession,
    SearchSessionConfig,
    SearchSessionTraceEntry,
    SearchUnit,
    SearchUnitBuildResult,
    SessionCandidate,
    SnippetSource,
    SubQuestionInput,
    VisualElementCandidate,
    VisualSearchResult,
)


class SearchSessionBuilder:
    """Combine independent Element ranks without mixing raw score scales."""

    def __init__(self, config: SearchSessionConfig | None = None) -> None:
        self.config = config or SearchSessionConfig()

    def create(
        self,
        *,
        subquestion: SubQuestionInput,
        search_units: SearchUnitBuildResult,
        exact: ExactLookupResult | None = None,
        bm25: BM25SearchResult | None = None,
        dense: DenseSearchResult | None = None,
        visual: VisualSearchResult | None = None,
    ) -> SearchSession:
        _validate_result_identity(
            subquestion=subquestion,
            search_units=search_units,
            exact=exact,
            bm25=bm25,
            dense=dense,
            visual=visual,
        )
        if (
            self.config.merge_policy
            == CandidateMergePolicy.FIXED_TEXT_VISUAL_QUOTA
            and visual is None
        ):
            raise ValueError(
                "Fixed Text/Visual quota requires a Visual search result"
            )
        units_by_id = {unit.search_unit_id: unit for unit in search_units.units}
        bm25_by_id = {
            candidate.element_id: candidate
            for candidate in (bm25.candidates if bm25 is not None else [])
        }
        dense_by_id = {
            candidate.element_id: candidate
            for candidate in (dense.candidates if dense is not None else [])
        }
        visual_by_id = {
            candidate.element_id: candidate
            for candidate in (visual.candidates if visual is not None else [])
        }
        exact_matches = _deduplicate_exact_matches(
            exact.exact_anchor_matches if exact is not None else []
        )
        exact_element_ids = {
            match.target_id
            for match in exact_matches
            if match.target_type.value not in {"page", "section"}
        }
        rank_bm25 = bm25.candidates if bm25 is not None else []
        rank_dense = dense.candidates if dense is not None else []
        rank_visual = visual.candidates if visual is not None else []
        if self.config.merge_policy in {
            CandidateMergePolicy.FIXED_QUOTA,
            CandidateMergePolicy.FIXED_TEXT_VISUAL_QUOTA,
        }:
            # Exact targets are a separate upstream result.  Remove them before
            # quota assignment so each preview batch still contains 3+2 unique
            # non-Exact candidates instead of developing a quota hole.
            rank_bm25 = [
                candidate
                for candidate in rank_bm25
                if candidate.element_id not in exact_element_ids
            ]
            rank_dense = [
                candidate
                for candidate in rank_dense
                if candidate.element_id not in exact_element_ids
            ]
            rank_visual = [
                candidate
                for candidate in rank_visual
                if candidate.element_id not in exact_element_ids
            ]
        selection_routes: dict[str, CandidateSelectionRoute] = {}
        if (
            self.config.merge_policy
            == CandidateMergePolicy.FIXED_TEXT_VISUAL_QUOTA
        ):
            text_ids, rrf_scores = _weighted_rrf_ids(
                bm25=rank_bm25,
                dense=rank_dense,
                config=self.config,
            )
            selections = _fixed_text_visual_ids(
                text_ids=text_ids,
                visual=rank_visual,
                config=self.config,
                presentation_key=(
                    search_units.document_id,
                    subquestion.subquestion_id,
                    subquestion.text,
                    search_units.index_version,
                ),
            )
            merged_ids = [element_id for element_id, _ in selections]
            selection_routes = dict(selections)
        else:
            merged_ids, rrf_scores = _rank_candidate_ids(
                bm25=rank_bm25,
                dense=rank_dense,
                config=self.config,
            )
        _, exact_rrf_scores = _weighted_rrf_ids(
            bm25=bm25.candidates if bm25 is not None else [],
            dense=dense.candidates if dense is not None else [],
            config=self.config,
        )
        exact_matches = [
            _enrich_exact_match(
                match,
                bm25=bm25_by_id.get(match.target_id),
                dense=dense_by_id.get(match.target_id),
                rrf_score=exact_rrf_scores.get(match.target_id),
            )
            for match in exact_matches
        ]
        ranked_ids = [
            element_id
            for element_id in merged_ids
            if element_id not in exact_element_ids
        ]
        catalog = [
            _session_candidate(
                element_id=element_id,
                bm25=bm25_by_id.get(element_id),
                dense=dense_by_id.get(element_id),
                visual=visual_by_id.get(element_id),
                units_by_id=units_by_id,
                rrf_score=rrf_scores.get(element_id),
                selection_route=selection_routes.get(element_id),
            )
            for element_id in ranked_ids
        ]

        unresolved = (
            [
                resolution
                for resolution in exact.anchor_resolutions
                if resolution.status != AnchorResolutionStatus.UNIQUE
            ]
            if exact is not None
            else []
        )
        trace = _session_trace(
            exact=exact,
            bm25=bm25,
            dense=dense,
            visual=visual,
            exact_element_ids=exact_element_ids,
            ranked_count=len(catalog),
            config=self.config,
            units_by_id=units_by_id,
        )
        session_id = "search-session:" + stable_digest(
            search_units.document_id,
            subquestion.subquestion_id,
            subquestion.text,
            search_units.index_version,
            [match.model_dump(mode="json") for match in exact_matches],
            [
                {
                    "element_id": element_id,
                    "selection_route": (
                        selection_routes[element_id].value
                        if element_id in selection_routes
                        else None
                    ),
                }
                for element_id in ranked_ids
            ],
            self.config.model_dump(mode="json"),
        )
        return SearchSession(
            search_session_id=session_id,
            subquestion_id=subquestion.subquestion_id,
            document_id=search_units.document_id,
            config=self.config,
            exact_anchor_matches=exact_matches,
            unresolved_anchors=unresolved,
            ranked_candidate_ids=ranked_ids,
            candidate_catalog=catalog,
            shown_candidate_ids=[],
            opened_candidate_ids=[],
            cursor=0,
            exhausted=not catalog,
            retrieval_trace=trace,
        )


class SearchSessionNavigator:
    """Advance, revisit, and mark candidates without rerunning retrieval."""

    def __init__(self, search_units: SearchUnitBuildResult) -> None:
        self.document_id = search_units.document_id
        self._units_by_id = {
            unit.search_unit_id: unit for unit in search_units.units
        }

    def next_batch(
        self,
        session: SearchSession,
        *,
        batch_size: int | None = None,
    ) -> tuple[SearchSession, SearchBatch]:
        self._validate_document(session)
        size = batch_size or session.config.batch_size
        if size < 1:
            raise ValueError("SearchSession batch_size must be positive")
        start = session.cursor
        end = min(len(session.candidate_catalog), start + size)
        previews = [
            _candidate_preview(candidate, session.config, self._units_by_id)
            for candidate in session.candidate_catalog[start:end]
        ]
        updated = _replace_session(
            session,
            shown_candidate_ids=session.ranked_candidate_ids[:end],
            cursor=end,
            exhausted=end >= len(session.ranked_candidate_ids),
        )
        return updated, SearchBatch(
            search_session_id=updated.search_session_id,
            exact_anchor_matches=updated.exact_anchor_matches,
            unresolved_anchors=updated.unresolved_anchors,
            candidate_previews=previews,
            next_cursor=updated.cursor,
            exhausted=updated.exhausted,
            retrieval_trace=updated.retrieval_trace,
        )

    def get_preview(
        self,
        session: SearchSession,
        element_id: str,
    ) -> CandidatePreview:
        self._validate_document(session)
        if element_id not in session.shown_candidate_ids:
            raise KeyError(
                f"Candidate has not been shown in this SearchSession: {element_id}"
            )
        candidate = _candidate_by_id(session, element_id)
        return _candidate_preview(candidate, session.config, self._units_by_id)

    def preview_history(self, session: SearchSession) -> list[CandidatePreview]:
        return [
            self.get_preview(session, element_id)
            for element_id in session.shown_candidate_ids
        ]

    def mark_opened(
        self,
        session: SearchSession,
        element_id: str,
    ) -> SearchSession:
        self._validate_document(session)
        if element_id not in session.shown_candidate_ids:
            raise ValueError("A candidate must be shown before it can be opened")
        if element_id in session.opened_candidate_ids:
            return session
        return _replace_session(
            session,
            opened_candidate_ids=[*session.opened_candidate_ids, element_id],
        )

    def _validate_document(self, session: SearchSession) -> None:
        if session.document_id != self.document_id:
            raise ValueError("SearchSessionNavigator uses another Document index")


def _validate_result_identity(
    *,
    subquestion: SubQuestionInput,
    search_units: SearchUnitBuildResult,
    exact: ExactLookupResult | None,
    bm25: BM25SearchResult | None,
    dense: DenseSearchResult | None,
    visual: VisualSearchResult | None,
) -> None:
    for name, result in (
        ("exact", exact),
        ("bm25", bm25),
        ("dense", dense),
        ("visual", visual),
    ):
        if result is None:
            continue
        if result.subquestion_id != subquestion.subquestion_id:
            raise ValueError(f"{name} result belongs to another SubQuestion")
        if result.document_id != search_units.document_id:
            raise ValueError(f"{name} result belongs to another Document")
    for name, result in (("bm25", bm25), ("dense", dense)):
        if result is not None and result.index_version != search_units.index_version:
            raise ValueError(f"{name} result uses another SearchUnit index version")


def _deduplicate_exact_matches(
    matches: Iterable[ExactAnchorMatch],
) -> list[ExactAnchorMatch]:
    result: list[ExactAnchorMatch] = []
    seen: set[str] = set()
    for match in matches:
        if match.target_id in seen:
            continue
        seen.add(match.target_id)
        result.append(match)
    return result


def _enrich_exact_match(
    match: ExactAnchorMatch,
    *,
    bm25: BM25ElementCandidate | None,
    dense: DenseElementCandidate | None,
    rrf_score: float | None,
) -> ExactAnchorMatch:
    matched_by: list[RetrievalSource] = []
    if bm25 is not None:
        matched_by.append(RetrievalSource.BM25)
    if dense is not None:
        matched_by.append(RetrievalSource.DENSE)
    return match.model_copy(
        update={
            "matched_by": matched_by,
            "bm25_rank": bm25.bm25_rank if bm25 is not None else None,
            "dense_rank": dense.dense_rank if dense is not None else None,
            "rrf_score": rrf_score,
        }
    )


def _round_robin_ids(
    bm25: list[BM25ElementCandidate],
    dense: list[DenseElementCandidate],
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for index in range(max(len(bm25), len(dense))):
        for candidates in (bm25, dense):
            if index >= len(candidates):
                continue
            element_id = candidates[index].element_id
            if element_id in seen:
                continue
            seen.add(element_id)
            result.append(element_id)
    return result


def _fixed_quota_ids(
    bm25: list[BM25ElementCandidate],
    dense: list[DenseElementCandidate],
    config: SearchSessionConfig,
) -> list[str]:
    """Build deterministic batches with source quotas and unique backfill."""

    streams: tuple[tuple[list[object], int], ...] = (
        (list(bm25), config.bm25_quota),
        (list(dense), config.dense_quota),
    )
    cursors = [0, 0]
    result: list[str] = []
    seen: set[str] = set()

    def take(stream_index: int, count: int) -> int:
        candidates = streams[stream_index][0]
        added = 0
        while cursors[stream_index] < len(candidates) and added < count:
            candidate = candidates[cursors[stream_index]]
            cursors[stream_index] += 1
            element_id = str(getattr(candidate, "element_id"))
            if element_id in seen:
                continue
            seen.add(element_id)
            result.append(element_id)
            added += 1
        return added

    while True:
        batch_start = len(result)
        for stream_index, (_, quota) in enumerate(streams):
            take(stream_index, quota)
        while len(result) - batch_start < config.batch_size:
            before = len(result)
            for stream_index in range(len(streams)):
                take(stream_index, 1)
                if len(result) - batch_start >= config.batch_size:
                    break
            if len(result) == before:
                break
        if len(result) == batch_start:
            break
    return result


def _rank_candidate_ids(
    *,
    bm25: list[BM25ElementCandidate],
    dense: list[DenseElementCandidate],
    config: SearchSessionConfig,
) -> tuple[list[str], dict[str, float]]:
    if config.merge_policy == CandidateMergePolicy.ROUND_ROBIN_BM25_FIRST:
        return _round_robin_ids(bm25, dense), {}
    if config.merge_policy == CandidateMergePolicy.FIXED_QUOTA:
        return _fixed_quota_ids(bm25, dense, config), {}
    if config.merge_policy == CandidateMergePolicy.WEIGHTED_RRF:
        return _weighted_rrf_ids(bm25=bm25, dense=dense, config=config)
    raise ValueError(f"Unsupported candidate merge policy: {config.merge_policy}")


def _weighted_rrf_ids(
    *,
    bm25: list[BM25ElementCandidate],
    dense: list[DenseElementCandidate],
    config: SearchSessionConfig,
) -> tuple[list[str], dict[str, float]]:
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    page_number: dict[str, int] = {}
    for candidates, weight in (
        (bm25, config.bm25_weight),
        (dense, config.dense_weight),
    ):
        for candidate in candidates:
            rank = (
                candidate.bm25_rank
                if isinstance(candidate, BM25ElementCandidate)
                else candidate.dense_rank
            )
            element_id = candidate.element_id
            scores[element_id] = scores.get(element_id, 0.0) + weight / (
                config.rrf_k + rank
            )
            best_rank[element_id] = min(best_rank.get(element_id, rank), rank)
            page_number[element_id] = candidate.page_number
    ordered = sorted(
        scores,
        key=lambda element_id: (
            -scores[element_id],
            best_rank[element_id],
            page_number[element_id],
            element_id,
        ),
    )
    return ordered, scores


def _fixed_text_visual_ids(
    *,
    text_ids: list[str],
    visual: list[VisualElementCandidate],
    config: SearchSessionConfig,
    presentation_key: tuple[str, str, str, str],
) -> list[tuple[str, CandidateSelectionRoute]]:
    """Create fixed 3+2 batches, then deterministically mix each batch.

    Text candidates are the BM25/Dense weighted-RRF ranking. Visual candidates
    are ranked independently. An Element can enter a session only once; when
    one route runs short, the other route backfills the batch. The final order
    is stable for the same question and index but does not expose route slots.
    """

    streams: tuple[tuple[list[str], CandidateSelectionRoute, int], ...] = (
        (list(text_ids), CandidateSelectionRoute.TEXT, config.text_quota),
        (
            [candidate.element_id for candidate in visual],
            CandidateSelectionRoute.VISUAL,
            config.visual_quota,
        ),
    )
    cursors = [0, 0]
    selected: list[tuple[str, CandidateSelectionRoute]] = []
    seen: set[str] = set()
    batch_index = 0

    def take(stream_index: int, count: int) -> list[tuple[str, CandidateSelectionRoute]]:
        ids, route, _ = streams[stream_index]
        added: list[tuple[str, CandidateSelectionRoute]] = []
        while cursors[stream_index] < len(ids) and len(added) < count:
            element_id = ids[cursors[stream_index]]
            cursors[stream_index] += 1
            if element_id in seen:
                continue
            seen.add(element_id)
            added.append((element_id, route))
        return added

    while True:
        batch: list[tuple[str, CandidateSelectionRoute]] = []
        for stream_index, (_, _, quota) in enumerate(streams):
            batch.extend(take(stream_index, quota))
        while len(batch) < config.batch_size:
            before = len(batch)
            for stream_index in range(len(streams)):
                batch.extend(take(stream_index, 1))
                if len(batch) >= config.batch_size:
                    break
            if len(batch) == before:
                break
        if not batch:
            break
        batch.sort(
            key=lambda item: stable_digest(
                *presentation_key,
                batch_index,
                item[0],
                item[1].value,
            )
        )
        selected.extend(batch)
        batch_index += 1
    return selected


def _session_candidate(
    *,
    element_id: str,
    bm25: BM25ElementCandidate | None,
    dense: DenseElementCandidate | None,
    visual: VisualElementCandidate | None,
    units_by_id: dict[str, SearchUnit],
    rrf_score: float | None,
    selection_route: CandidateSelectionRoute | None,
) -> SessionCandidate:
    source_unit_id = (
        bm25.matched_search_unit_id if bm25 is not None
        else dense.matched_search_unit_id if dense is not None
        else None
    )
    source_unit = units_by_id.get(source_unit_id) if source_unit_id else None
    if source_unit is None and visual is None:
        raise ValueError(f"Candidate has no available source metadata: {element_id}")
    if source_unit is not None and source_unit.element_id != element_id:
        raise ValueError("Candidate SearchUnit belongs to another Element")
    for candidate in (bm25, dense):
        if candidate is None:
            continue
        unit = units_by_id.get(candidate.matched_search_unit_id)
        if unit is None:
            raise ValueError(
                f"Candidate refers to missing SearchUnit: "
                f"{candidate.matched_search_unit_id}"
            )
        if unit.element_id != element_id:
            raise ValueError("Merged candidate SearchUnits disagree on Element ID")

    if visual is not None:
        if visual.element_id != element_id:
            raise ValueError("Visual candidate belongs to another Element")
        if source_unit is not None and (
            visual.page_id != source_unit.page_id
            or visual.page_number != source_unit.page_number
            or visual.element_type != source_unit.element_type
        ):
            raise ValueError("Text and Visual candidate metadata disagree")

    matched_by = []
    if bm25 is not None:
        matched_by.append(RetrievalSource.BM25)
    if dense is not None:
        matched_by.append(RetrievalSource.DENSE)
    if visual is not None:
        matched_by.append(RetrievalSource.VISUAL_DENSE)
    if source_unit is not None:
        element_type = source_unit.element_type
        page_id = source_unit.page_id
        page_number = source_unit.page_number
        section_id = source_unit.section_id
        section_path = source_unit.section_path
        display_label = source_unit.display_label
        content_availability = source_unit.content_availability
    else:
        assert visual is not None
        element_type = visual.element_type
        page_id = visual.page_id
        page_number = visual.page_number
        section_id = visual.section_id
        section_path = visual.section_path
        display_label = visual.display_label
        content_availability = visual.content_availability
    return SessionCandidate(
        element_id=element_id,
        element_type=element_type,
        page_id=page_id,
        page_number=page_number,
        section_id=section_id,
        section_path=section_path,
        display_label=display_label,
        content_availability=content_availability,
        matched_by=matched_by,
        bm25_rank=bm25.bm25_rank if bm25 is not None else None,
        bm25_score=bm25.bm25_score if bm25 is not None else None,
        bm25_search_unit_id=(
            bm25.matched_search_unit_id if bm25 is not None else None
        ),
        bm25_matched_terms=bm25.matched_terms if bm25 is not None else [],
        bm25_matched_offsets=bm25.matched_offsets if bm25 is not None else [],
        dense_rank=dense.dense_rank if dense is not None else None,
        dense_score=dense.dense_score if dense is not None else None,
        dense_search_unit_id=(
            dense.matched_search_unit_id if dense is not None else None
        ),
        dense_match_start=(
            dense.matched_text_char_start if dense is not None else None
        ),
        dense_match_end=(
            dense.matched_text_char_end if dense is not None else None
        ),
        rrf_score=rrf_score,
        visual_asset_id=(visual.visual_asset_id if visual is not None else None),
        visual_rank=visual.visual_rank if visual is not None else None,
        visual_score=visual.visual_score if visual is not None else None,
        visual_preview_text=(visual.preview_text if visual is not None else None),
        selection_route=selection_route,
    )


def _session_trace(
    *,
    exact: ExactLookupResult | None,
    bm25: BM25SearchResult | None,
    dense: DenseSearchResult | None,
    visual: VisualSearchResult | None,
    exact_element_ids: set[str],
    ranked_count: int,
    config: SearchSessionConfig,
    units_by_id: dict[str, SearchUnit],
) -> list[SearchSessionTraceEntry]:
    trace = [
        SearchSessionTraceEntry(
            code="candidate_ranking_created",
            description=(
                "Text and optional Visual Element rankings were merged with "
                "stable Element deduplication."
            ),
            data={
                "policy": config.merge_policy.value,
                "bm25_candidates": len(bm25.candidates) if bm25 else 0,
                "dense_candidates": len(dense.candidates) if dense else 0,
                "visual_candidates": len(visual.candidates) if visual else 0,
                "ranked_candidates": ranked_count,
                "exact_element_targets_excluded": len(exact_element_ids),
                "fixed_final_top_k": False,
                "rrf_k": config.rrf_k,
                "bm25_weight": config.bm25_weight,
                "dense_weight": config.dense_weight,
                "bm25_quota": config.bm25_quota,
                "dense_quota": config.dense_quota,
                "text_quota": config.text_quota,
                "visual_quota": config.visual_quota,
                "stable_mixed_presentation": (
                    config.merge_policy
                    == CandidateMergePolicy.FIXED_TEXT_VISUAL_QUOTA
                ),
                "visual_index_fingerprint": (
                    visual.index_fingerprint if visual is not None else None
                ),
                "visual_model": visual.model_name if visual is not None else None,
            },
        )
    ]
    trace.append(
        SearchSessionTraceEntry(
            code="search_metadata_scope_audit",
            description=(
                "Count metadata-only matches in each source's first five "
                "candidates without changing their ranking."
            ),
            data=_metadata_scope_audit(bm25, dense, units_by_id, limit=5),
        )
    )
    if exact is not None:
        trace.extend(
            SearchSessionTraceEntry(
                code=f"exact_{item.code}",
                description=item.description,
                data=item.data,
            )
            for item in exact.trace
        )
    if bm25 is not None:
        trace.extend(
            SearchSessionTraceEntry(
                code=f"bm25_{item.code}",
                description=item.description,
                data=item.data,
            )
            for item in bm25.trace
        )
    if dense is not None:
        trace.extend(
            SearchSessionTraceEntry(
                code=f"dense_{item.code}",
                description=item.description,
                data=item.data,
            )
            for item in dense.trace
        )
    return trace


def _metadata_scope_audit(
    bm25: BM25SearchResult | None,
    dense: DenseSearchResult | None,
    units_by_id: dict[str, SearchUnit],
    *,
    limit: int,
) -> dict[str, int]:
    bm25_scopes = [
        _ranges_scope(
            units_by_id[candidate.matched_search_unit_id],
            [(offset.start, offset.end) for offset in candidate.matched_offsets],
        )
        for candidate in (bm25.candidates[:limit] if bm25 is not None else [])
        if candidate.matched_search_unit_id in units_by_id
    ]
    dense_scopes = [
        _ranges_scope(
            units_by_id[candidate.matched_search_unit_id],
            [
                (
                    candidate.matched_text_char_start,
                    candidate.matched_text_char_end,
                )
            ],
        )
        for candidate in (dense.candidates[:limit] if dense is not None else [])
        if candidate.matched_search_unit_id in units_by_id
    ]
    return {
        "limit": limit,
        "bm25_evaluated": len(bm25_scopes),
        "bm25_metadata_only": sum(
            scope == PreviewMatchScope.METADATA for scope in bm25_scopes
        ),
        "dense_evaluated": len(dense_scopes),
        "dense_metadata_only": sum(
            scope == PreviewMatchScope.METADATA for scope in dense_scopes
        ),
    }


def _candidate_preview(
    candidate: SessionCandidate,
    config: SearchSessionConfig,
    units_by_id: dict[str, SearchUnit],
) -> CandidatePreview:
    preview_source, match_scope = _preview_choice(
        candidate,
        config,
        units_by_id,
    )
    if preview_source == RetrievalSource.VISUAL_DENSE:
        text = candidate.visual_preview_text or ""
        start, end = _snippet_window(
            text=text,
            anchor_start=0,
            anchor_end=min(len(text), config.snippet_max_chars),
            max_chars=config.snippet_max_chars,
            context_chars=config.snippet_context_chars,
        )
        if not candidate.visual_asset_id:
            raise ValueError("Visual CandidatePreview requires a visual asset")
        return CandidatePreview(
            element_id=candidate.element_id,
            element_type=candidate.element_type,
            page_id=candidate.page_id,
            page_number=candidate.page_number,
            section_path=candidate.section_path,
            display_label=candidate.display_label,
            matched_snippet=text[start:end],
            snippet_char_start=start,
            snippet_char_end=end,
            snippet_truncated=start > 0 or end < len(text),
            snippet_source=SnippetSource.VISUAL_METADATA,
            snippet_source_id=candidate.visual_asset_id,
            matched_search_unit_id=None,
            visual_asset_id=candidate.visual_asset_id,
            matched_by=candidate.matched_by,
            preview_source=RetrievalSource.VISUAL_DENSE,
            match_scope=PreviewMatchScope.UNKNOWN,
            bm25_rank=candidate.bm25_rank,
            dense_rank=candidate.dense_rank,
            visual_rank=candidate.visual_rank,
            rrf_score=candidate.rrf_score,
            content_availability=candidate.content_availability,
        )
    if preview_source == RetrievalSource.BM25:
        unit_id = candidate.bm25_search_unit_id
        offsets = candidate.bm25_matched_offsets
        anchor_start = min((offset.start for offset in offsets), default=0)
        anchor_end = max((offset.end for offset in offsets), default=anchor_start)
    else:
        unit_id = candidate.dense_search_unit_id
        anchor_start = candidate.dense_match_start or 0
        anchor_end = candidate.dense_match_end or anchor_start
    if not unit_id:
        raise ValueError("CandidatePreview requires a matched SearchUnit")
    unit = units_by_id.get(unit_id)
    if unit is None or unit.element_id != candidate.element_id:
        raise ValueError("CandidatePreview refers to an unavailable SearchUnit")
    text = unit.search_text
    start, end = _snippet_window(
        text=text,
        anchor_start=anchor_start,
        anchor_end=anchor_end,
        max_chars=config.snippet_max_chars,
        context_chars=config.snippet_context_chars,
    )
    return CandidatePreview(
        element_id=candidate.element_id,
        element_type=candidate.element_type,
        page_id=candidate.page_id,
        page_number=candidate.page_number,
        section_path=candidate.section_path,
        display_label=candidate.display_label,
        matched_snippet=text[start:end],
        snippet_char_start=start,
        snippet_char_end=end,
        snippet_truncated=start > 0 or end < len(text),
        snippet_source=SnippetSource.SEARCH_UNIT_TEXT,
        snippet_source_id=unit_id,
        matched_search_unit_id=unit_id,
        matched_by=candidate.matched_by,
        preview_source=preview_source,
        match_scope=match_scope,
        bm25_rank=candidate.bm25_rank,
        dense_rank=candidate.dense_rank,
        visual_rank=candidate.visual_rank,
        rrf_score=candidate.rrf_score,
        content_availability=candidate.content_availability,
    )


def _preview_choice(
    candidate: SessionCandidate,
    config: SearchSessionConfig,
    units_by_id: dict[str, SearchUnit],
) -> tuple[RetrievalSource, PreviewMatchScope]:
    if candidate.selection_route == CandidateSelectionRoute.VISUAL:
        return RetrievalSource.VISUAL_DENSE, PreviewMatchScope.UNKNOWN
    has_bm25 = RetrievalSource.BM25 in candidate.matched_by
    has_dense = RetrievalSource.DENSE in candidate.matched_by
    if not has_bm25 and not has_dense:
        if RetrievalSource.VISUAL_DENSE in candidate.matched_by:
            return RetrievalSource.VISUAL_DENSE, PreviewMatchScope.UNKNOWN
        raise ValueError("Candidate has no previewable retrieval source")
    if not has_bm25:
        return RetrievalSource.DENSE, _dense_match_scope(candidate, units_by_id)
    if not has_dense:
        return RetrievalSource.BM25, _bm25_match_scope(candidate, units_by_id)

    bm25_scope = _bm25_match_scope(candidate, units_by_id)
    dense_scope = _dense_match_scope(candidate, units_by_id)
    content_scopes = {PreviewMatchScope.CONTENT, PreviewMatchScope.MIXED}
    if bm25_scope == PreviewMatchScope.METADATA and dense_scope in content_scopes:
        return RetrievalSource.DENSE, dense_scope
    if dense_scope == PreviewMatchScope.METADATA and bm25_scope in content_scopes:
        return RetrievalSource.BM25, bm25_scope
    if config.merge_policy != CandidateMergePolicy.WEIGHTED_RRF:
        return RetrievalSource.BM25, bm25_scope
    assert candidate.bm25_rank is not None
    assert candidate.dense_rank is not None
    bm25_contribution = config.bm25_weight / (config.rrf_k + candidate.bm25_rank)
    dense_contribution = config.dense_weight / (config.rrf_k + candidate.dense_rank)
    if bm25_contribution >= dense_contribution:
        return RetrievalSource.BM25, bm25_scope
    return RetrievalSource.DENSE, dense_scope


def _bm25_match_scope(
    candidate: SessionCandidate,
    units_by_id: dict[str, SearchUnit],
) -> PreviewMatchScope:
    unit = _candidate_unit(candidate.bm25_search_unit_id, candidate, units_by_id)
    ranges = [(offset.start, offset.end) for offset in candidate.bm25_matched_offsets]
    return _ranges_scope(unit, ranges)


def _dense_match_scope(
    candidate: SessionCandidate,
    units_by_id: dict[str, SearchUnit],
) -> PreviewMatchScope:
    unit = _candidate_unit(candidate.dense_search_unit_id, candidate, units_by_id)
    if candidate.dense_match_start is None or candidate.dense_match_end is None:
        return PreviewMatchScope.UNKNOWN
    return _ranges_scope(
        unit,
        [(candidate.dense_match_start, candidate.dense_match_end)],
    )


def _candidate_unit(
    unit_id: str | None,
    candidate: SessionCandidate,
    units_by_id: dict[str, SearchUnit],
) -> SearchUnit:
    if not unit_id:
        raise ValueError("Candidate source requires a matched SearchUnit")
    unit = units_by_id.get(unit_id)
    if unit is None or unit.element_id != candidate.element_id:
        raise ValueError("Candidate source refers to an unavailable SearchUnit")
    return unit


def _ranges_scope(
    unit: SearchUnit,
    ranges: list[tuple[int, int]],
) -> PreviewMatchScope:
    if not ranges:
        return PreviewMatchScope.UNKNOWN
    content_range = _content_search_range(unit)
    if content_range is None:
        return PreviewMatchScope.UNKNOWN
    content_start, content_end = content_range
    touches_content = False
    touches_metadata = False
    for start, end in ranges:
        if start < content_end and end > content_start:
            touches_content = True
        if start < content_start or end > content_end:
            touches_metadata = True
    if touches_content and touches_metadata:
        return PreviewMatchScope.MIXED
    if touches_content:
        return PreviewMatchScope.CONTENT
    if touches_metadata:
        return PreviewMatchScope.METADATA
    return PreviewMatchScope.UNKNOWN


def _content_search_range(unit: SearchUnit) -> tuple[int, int] | None:
    if (
        unit.content_search_char_start is not None
        and unit.content_search_char_end is not None
    ):
        return unit.content_search_char_start, unit.content_search_char_end
    start = unit.search_text.rfind(unit.content_text)
    if start < 0:
        return None
    return start, start + len(unit.content_text)


def _snippet_window(
    *,
    text: str,
    anchor_start: int,
    anchor_end: int,
    max_chars: int,
    context_chars: int,
) -> tuple[int, int]:
    if not text:
        return 0, 0
    if len(text) <= max_chars:
        return 0, len(text)
    anchor_start = max(0, min(anchor_start, len(text)))
    anchor_end = max(anchor_start, min(anchor_end, len(text)))
    start = max(0, anchor_start - context_chars)
    end = min(len(text), max(anchor_end + context_chars, start + max_chars))
    if end - start > max_chars:
        end = start + max_chars
    if end - start < max_chars:
        start = max(0, end - max_chars)
    return start, end


def _candidate_by_id(
    session: SearchSession,
    element_id: str,
) -> SessionCandidate:
    for candidate in session.candidate_catalog:
        if candidate.element_id == element_id:
            return candidate
    raise KeyError(f"Unknown SearchSession candidate: {element_id}")


def _replace_session(session: SearchSession, **changes: object) -> SearchSession:
    data = session.model_dump(mode="python")
    data.update(changes)
    return SearchSession.model_validate(data)
