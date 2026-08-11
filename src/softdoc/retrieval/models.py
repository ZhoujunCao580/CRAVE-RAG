"""Pydantic runtime models for retrieval over a finalized SoftDoc."""

from __future__ import annotations

from enum import Enum
from typing import Any, Self

from pydantic import Field, field_validator, model_validator

from softdoc.models import ContentAvailability, ElementType, SoftDocModel


class AnchorKind(str, Enum):
    PAGE = "page"
    FIGURE = "figure"
    TABLE = "table"
    SECTION = "section"


class AnchorResolutionStatus(str, Enum):
    UNIQUE = "unique"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"


class AnchorTargetType(str, Enum):
    PAGE = "page"
    SECTION = "section"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    FIGURE = "figure"
    CHART = "chart"
    TABLE = "table"
    CODE = "code"
    ALGORITHM = "algorithm"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    LIST = "list"
    EQUATION = "equation"


class RetrievalSource(str, Enum):
    BM25 = "bm25"
    DENSE = "dense"


class SnippetSource(str, Enum):
    SEARCH_UNIT_TEXT = "search_unit.search_text"


class PreviewMatchScope(str, Enum):
    CONTENT = "content"
    METADATA = "metadata"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class SubQuestionInput(SoftDocModel):
    subquestion_id: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @field_validator("subquestion_id", "text")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("SubQuestion fields must not be blank")
        return stripped


class AnchorTargetHandle(SoftDocModel):
    target_id: str
    target_type: AnchorTargetType
    page_id: str
    page_number: int = Field(ge=1)
    section_id: str | None = None
    resolution_method: str
    label_source_id: str | None = None


class AnchorResolution(SoftDocModel):
    resolution_id: str
    anchor_text: str
    anchor_kind: AnchorKind
    normalized_label: str
    source_span: tuple[int, int]
    status: AnchorResolutionStatus
    matches: list[AnchorTargetHandle] = Field(default_factory=list)
    reason: str | None = None

    @model_validator(mode="after")
    def validate_status_cardinality(self) -> Self:
        count = len(self.matches)
        if self.status == AnchorResolutionStatus.UNIQUE and count != 1:
            raise ValueError("A unique Anchor resolution requires one match")
        if self.status == AnchorResolutionStatus.AMBIGUOUS and count < 2:
            raise ValueError("An ambiguous Anchor resolution requires multiple matches")
        if self.status == AnchorResolutionStatus.UNRESOLVED and count != 0:
            raise ValueError("An unresolved Anchor resolution cannot contain matches")
        return self


class ExactAnchorMatch(SoftDocModel):
    resolution_id: str
    anchor_text: str
    anchor_kind: AnchorKind
    normalized_label: str
    target_id: str
    target_type: AnchorTargetType
    page_id: str
    page_number: int = Field(ge=1)
    section_id: str | None = None
    resolution_method: str
    matched_by: list[RetrievalSource] = Field(default_factory=list)
    bm25_rank: int | None = Field(default=None, ge=1)
    dense_rank: int | None = Field(default=None, ge=1)
    rrf_score: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def validate_retrieval_metadata(self) -> Self:
        if len(set(self.matched_by)) != len(self.matched_by):
            raise ValueError("Exact Anchor retrieval sources must be unique")
        if (self.bm25_rank is not None) != (RetrievalSource.BM25 in self.matched_by):
            raise ValueError("Exact Anchor BM25 rank and source must appear together")
        if (self.dense_rank is not None) != (RetrievalSource.DENSE in self.matched_by):
            raise ValueError("Exact Anchor Dense rank and source must appear together")
        if self.rrf_score is not None and not self.matched_by:
            raise ValueError("Exact Anchor RRF score requires a retrieval source")
        return self


class ExactLookupTraceEntry(SoftDocModel):
    code: str
    description: str
    resolution_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class ExactLookupResult(SoftDocModel):
    subquestion_id: str
    document_id: str
    anchor_resolutions: list[AnchorResolution] = Field(default_factory=list)
    exact_anchor_matches: list[ExactAnchorMatch] = Field(default_factory=list)
    trace: list[ExactLookupTraceEntry] = Field(default_factory=list)


class SearchUnitConfig(SoftDocModel):
    split_threshold_tokens: int = Field(default=256, ge=1)
    part_size_tokens: int = Field(default=256, ge=1)
    overlap_tokens: int = Field(default=32, ge=0)
    index_version: str = Field(default="search-unit-v1", min_length=1)

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if self.overlap_tokens >= self.part_size_tokens:
            raise ValueError("SearchUnit overlap must be smaller than part size")
        return self


class SearchUnit(SoftDocModel):
    search_unit_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    element_id: str = Field(min_length=1)
    part_index: int = Field(ge=0)
    part_count: int = Field(ge=1)
    search_text: str = Field(min_length=1)
    content_text: str = Field(min_length=1)
    display_label: str | None = Field(default=None, min_length=1)
    content_search_char_start: int | None = Field(default=None, ge=0)
    content_search_char_end: int | None = Field(default=None, gt=0)
    source_char_start: int = Field(ge=0)
    source_char_end: int = Field(gt=0)
    page_id: str = Field(min_length=1)
    page_index: int = Field(ge=0)
    page_number: int = Field(ge=1)
    reading_order: int = Field(ge=0)
    section_id: str | None = None
    section_path: list[str] = Field(default_factory=list)
    element_type: ElementType
    content_availability: ContentAvailability
    index_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_part(self) -> Self:
        if self.part_index >= self.part_count:
            raise ValueError("SearchUnit part_index must be smaller than part_count")
        if self.source_char_start >= self.source_char_end:
            raise ValueError("SearchUnit source range must be non-empty")
        if (self.content_search_char_start is None) != (
            self.content_search_char_end is None
        ):
            raise ValueError("SearchUnit content search bounds must appear together")
        if (
            self.content_search_char_start is not None
            and self.content_search_char_end is not None
        ):
            if self.content_search_char_start >= self.content_search_char_end:
                raise ValueError("SearchUnit content search range must be non-empty")
            if self.content_search_char_end > len(self.search_text):
                raise ValueError("SearchUnit content search range exceeds search_text")
            if (
                self.search_text[
                    self.content_search_char_start : self.content_search_char_end
                ]
                != self.content_text
            ):
                raise ValueError(
                    "SearchUnit content search range must select content_text"
                )
        return self


class SkippedSearchElement(SoftDocModel):
    element_id: str = Field(min_length=1)
    element_type: ElementType
    reason: str = Field(min_length=1)
    details: dict[str, Any] = Field(default_factory=dict)


class SearchUnitBuildResult(SoftDocModel):
    document_id: str = Field(min_length=1)
    index_version: str = Field(min_length=1)
    config: SearchUnitConfig
    units: list[SearchUnit] = Field(default_factory=list)
    skipped_elements: list[SkippedSearchElement] = Field(default_factory=list)


class BM25Config(SoftDocModel):
    k1: float = Field(default=1.2, gt=0.0)
    b: float = Field(default=0.75, ge=0.0, le=1.0)


class MatchedOffset(SoftDocModel):
    term: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_offset(self) -> Self:
        if self.start >= self.end:
            raise ValueError("Matched offset requires start < end")
        return self


class BM25ElementCandidate(SoftDocModel):
    element_id: str = Field(min_length=1)
    bm25_score: float = Field(gt=0.0)
    bm25_rank: int = Field(ge=1)
    matched_search_unit_id: str = Field(min_length=1)
    matched_part: int = Field(ge=0)
    matched_terms: list[str] = Field(default_factory=list)
    matched_offsets: list[MatchedOffset] = Field(default_factory=list)
    page_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    section_id: str | None = None
    element_type: ElementType


class BM25TraceEntry(SoftDocModel):
    code: str = Field(min_length=1)
    description: str = Field(min_length=1)
    data: dict[str, Any] = Field(default_factory=dict)


class BM25SearchResult(SoftDocModel):
    subquestion_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    index_version: str = Field(min_length=1)
    total_search_units: int = Field(ge=0)
    total_candidates: int = Field(ge=0)
    candidates: list[BM25ElementCandidate] = Field(default_factory=list)
    trace: list[BM25TraceEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ranks(self) -> Self:
        if self.total_candidates != len(self.candidates):
            raise ValueError("BM25 total_candidates must match candidates")
        if [item.bm25_rank for item in self.candidates] != list(
            range(1, len(self.candidates) + 1)
        ):
            raise ValueError("BM25 candidate ranks must be contiguous")
        return self


class EncoderInputType(str, Enum):
    QUERY = "query"
    PASSAGE = "passage"


class DenseDevice(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class EncoderFingerprint(SoftDocModel):
    model_name: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    tokenizer_revision: str = Field(min_length=1)
    embedding_dimension: int = Field(ge=1)
    max_length: int = Field(ge=1)
    pooling_method: str = Field(default="attention_mask_mean", min_length=1)
    normalize_embeddings: bool = True
    dtype: str = Field(default="float32", min_length=1)


class EncoderTokenSpan(SoftDocModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        if self.start >= self.end:
            raise ValueError("Encoder token span requires start < end")
        return self


class DenseConfig(SoftDocModel):
    max_length: int = Field(default=512, ge=8)
    segment_token_budget: int = Field(default=480, ge=1)
    segment_overlap_tokens: int = Field(default=48, ge=0)
    batch_size: int = Field(default=16, ge=1)
    cache_schema_version: str = Field(default="dense-cache-v1", min_length=1)
    segment_index_version: str = Field(default="dense-segment-v1", min_length=1)

    @model_validator(mode="after")
    def validate_segment_window(self) -> Self:
        if self.segment_token_budget >= self.max_length:
            raise ValueError("Dense segment token budget must be below max_length")
        if self.segment_overlap_tokens >= self.segment_token_budget:
            raise ValueError("Dense segment overlap must be below token budget")
        return self


class DenseSegment(SoftDocModel):
    dense_segment_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    search_unit_id: str = Field(min_length=1)
    element_id: str = Field(min_length=1)
    segment_index: int = Field(ge=0)
    segment_count: int = Field(ge=1)
    segment_text: str = Field(min_length=1)
    search_text_char_start: int = Field(ge=0)
    search_text_char_end: int = Field(gt=0)
    model_token_count: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_segment(self) -> Self:
        if self.segment_index >= self.segment_count:
            raise ValueError("Dense segment_index must be smaller than segment_count")
        if self.search_text_char_start >= self.search_text_char_end:
            raise ValueError("Dense segment source range must be non-empty")
        return self


class EmbeddingCacheKey(SoftDocModel):
    cache_schema_version: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    search_unit_id: str = Field(min_length=1)
    dense_segment_id: str = Field(min_length=1)
    search_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_version: str = Field(min_length=1)
    segment_index_version: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    tokenizer_revision: str = Field(min_length=1)
    pooling_method: str = Field(min_length=1)
    max_length: int = Field(ge=1)
    normalize_embeddings: bool
    embedding_dimension: int = Field(ge=1)
    dtype: str = Field(min_length=1)


class EmbeddingCacheRecord(SoftDocModel):
    key: EmbeddingCacheKey
    vector: list[float] = Field(min_length=1)

    @field_validator("vector")
    @classmethod
    def validate_finite_vector(cls, value: list[float]) -> list[float]:
        if not all(float("-inf") < item < float("inf") for item in value):
            raise ValueError("Cached embedding values must be finite")
        return value

    @model_validator(mode="after")
    def validate_dimension(self) -> Self:
        if len(self.vector) != self.key.embedding_dimension:
            raise ValueError("Cached embedding dimension does not match its key")
        return self


class DenseElementCandidate(SoftDocModel):
    element_id: str = Field(min_length=1)
    dense_score: float = Field(ge=-1.0, le=1.0)
    dense_rank: int = Field(ge=1)
    matched_search_unit_id: str = Field(min_length=1)
    matched_part: int = Field(ge=0)
    matched_dense_segment_id: str = Field(min_length=1)
    matched_segment_index: int = Field(ge=0)
    matched_text_char_start: int = Field(ge=0)
    matched_text_char_end: int = Field(gt=0)
    page_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    section_id: str | None = None
    element_type: ElementType

    @model_validator(mode="after")
    def validate_match_range(self) -> Self:
        if self.matched_text_char_start >= self.matched_text_char_end:
            raise ValueError("Dense match range must be non-empty")
        return self


class DenseTraceEntry(SoftDocModel):
    code: str = Field(min_length=1)
    description: str = Field(min_length=1)
    data: dict[str, Any] = Field(default_factory=dict)


class DenseSearchResult(SoftDocModel):
    subquestion_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    index_version: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    total_search_units: int = Field(ge=0)
    total_dense_segments: int = Field(ge=0)
    total_candidates: int = Field(ge=0)
    candidates: list[DenseElementCandidate] = Field(default_factory=list)
    trace: list[DenseTraceEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ranks(self) -> Self:
        if self.total_candidates != len(self.candidates):
            raise ValueError("Dense total_candidates must match candidates")
        if [item.dense_rank for item in self.candidates] != list(
            range(1, len(self.candidates) + 1)
        ):
            raise ValueError("Dense candidate ranks must be contiguous")
        return self


class CandidateMergePolicy(str, Enum):
    WEIGHTED_RRF = "weighted_rrf"
    ROUND_ROBIN_BM25_FIRST = "round_robin_bm25_first"


class SearchSessionConfig(SoftDocModel):
    batch_size: int = Field(default=5, ge=1)
    snippet_max_chars: int = Field(default=320, ge=40)
    snippet_context_chars: int = Field(default=80, ge=0)
    merge_policy: CandidateMergePolicy = CandidateMergePolicy.WEIGHTED_RRF
    rrf_k: int = Field(default=20, ge=1)
    bm25_weight: float = Field(default=1.0, gt=0.0)
    dense_weight: float = Field(default=1.25, gt=0.0)

    @model_validator(mode="after")
    def validate_snippet_window(self) -> Self:
        if self.snippet_context_chars * 2 >= self.snippet_max_chars:
            raise ValueError(
                "CandidatePreview context must leave room for matched content"
            )
        return self


class SessionCandidate(SoftDocModel):
    """Persisted retrieval metadata for one deduplicated Element candidate."""

    element_id: str = Field(min_length=1)
    element_type: ElementType
    page_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    section_id: str | None = None
    section_path: list[str] = Field(default_factory=list)
    display_label: str | None = Field(default=None, min_length=1)
    content_availability: ContentAvailability
    matched_by: list[RetrievalSource] = Field(min_length=1)
    bm25_rank: int | None = Field(default=None, ge=1)
    bm25_score: float | None = Field(default=None, gt=0.0)
    bm25_search_unit_id: str | None = None
    bm25_matched_terms: list[str] = Field(default_factory=list)
    bm25_matched_offsets: list[MatchedOffset] = Field(default_factory=list)
    dense_rank: int | None = Field(default=None, ge=1)
    dense_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    dense_search_unit_id: str | None = None
    dense_match_start: int | None = Field(default=None, ge=0)
    dense_match_end: int | None = Field(default=None, gt=0)
    rrf_score: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def validate_sources(self) -> Self:
        if len(set(self.matched_by)) != len(self.matched_by):
            raise ValueError("SessionCandidate matched_by sources must be unique")
        if RetrievalSource.BM25 in self.matched_by:
            if self.bm25_rank is None or not self.bm25_search_unit_id:
                raise ValueError("BM25 candidates require rank and SearchUnit")
        elif any(
            value is not None
            for value in (self.bm25_rank, self.bm25_score, self.bm25_search_unit_id)
        ):
            raise ValueError("BM25 metadata requires the bm25 source")
        if RetrievalSource.DENSE in self.matched_by:
            if self.dense_rank is None or not self.dense_search_unit_id:
                raise ValueError("Dense candidates require rank and SearchUnit")
            if self.dense_match_start is None or self.dense_match_end is None:
                raise ValueError("Dense candidates require a matched text range")
            if self.dense_match_start >= self.dense_match_end:
                raise ValueError("Dense candidate match range must be non-empty")
        elif any(
            value is not None
            for value in (
                self.dense_rank,
                self.dense_score,
                self.dense_search_unit_id,
                self.dense_match_start,
                self.dense_match_end,
            )
        ):
            raise ValueError("Dense metadata requires the dense source")
        return self


class CandidatePreview(SoftDocModel):
    """Small deterministic card used to choose what to READ next."""

    element_id: str = Field(min_length=1)
    element_type: ElementType
    page_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    section_path: list[str] = Field(default_factory=list)
    display_label: str | None = Field(default=None, min_length=1)
    matched_snippet: str
    snippet_char_start: int = Field(ge=0)
    snippet_char_end: int = Field(ge=0)
    snippet_truncated: bool
    snippet_source: SnippetSource = SnippetSource.SEARCH_UNIT_TEXT
    snippet_source_id: str = Field(min_length=1)
    matched_search_unit_id: str = Field(min_length=1)
    matched_by: list[RetrievalSource] = Field(min_length=1)
    preview_source: RetrievalSource
    match_scope: PreviewMatchScope
    bm25_rank: int | None = Field(default=None, ge=1)
    dense_rank: int | None = Field(default=None, ge=1)
    rrf_score: float | None = Field(default=None, gt=0.0)
    content_availability: ContentAvailability

    @model_validator(mode="after")
    def validate_snippet_range(self) -> Self:
        if self.snippet_source_id != self.matched_search_unit_id:
            raise ValueError("CandidatePreview snippet source must be its SearchUnit")
        if self.preview_source not in self.matched_by:
            raise ValueError("CandidatePreview source must be a matched source")
        if self.snippet_char_start > self.snippet_char_end:
            raise ValueError("CandidatePreview snippet range is invalid")
        if not self.matched_snippet and (
            self.snippet_char_start != 0 or self.snippet_char_end != 0
        ):
            raise ValueError("An empty CandidatePreview must use range (0, 0)")
        return self


class SearchSessionTraceEntry(SoftDocModel):
    code: str = Field(min_length=1)
    description: str = Field(min_length=1)
    data: dict[str, Any] = Field(default_factory=dict)


class SearchSession(SoftDocModel):
    """Serializable cursor over a complete, deduplicated candidate ranking."""

    search_session_id: str = Field(min_length=1)
    subquestion_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    config: SearchSessionConfig
    exact_anchor_matches: list[ExactAnchorMatch] = Field(default_factory=list)
    unresolved_anchors: list[AnchorResolution] = Field(default_factory=list)
    ranked_candidate_ids: list[str] = Field(default_factory=list)
    candidate_catalog: list[SessionCandidate] = Field(default_factory=list)
    shown_candidate_ids: list[str] = Field(default_factory=list)
    opened_candidate_ids: list[str] = Field(default_factory=list)
    cursor: int = Field(default=0, ge=0)
    exhausted: bool = False
    retrieval_trace: list[SearchSessionTraceEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_session_state(self) -> Self:
        catalog_ids = [item.element_id for item in self.candidate_catalog]
        if len(set(catalog_ids)) != len(catalog_ids):
            raise ValueError("SearchSession candidate IDs must be unique")
        if catalog_ids != self.ranked_candidate_ids:
            raise ValueError(
                "SearchSession ranked_candidate_ids must match candidate catalog order"
            )
        if len(set(self.shown_candidate_ids)) != len(self.shown_candidate_ids):
            raise ValueError("SearchSession shown candidate IDs must be unique")
        if len(set(self.opened_candidate_ids)) != len(self.opened_candidate_ids):
            raise ValueError("SearchSession opened candidate IDs must be unique")
        ranked = set(self.ranked_candidate_ids)
        if not set(self.shown_candidate_ids).issubset(ranked):
            raise ValueError("Shown candidates must belong to the ranking")
        if not set(self.opened_candidate_ids).issubset(
            set(self.shown_candidate_ids)
        ):
            raise ValueError("Opened candidates must have been shown first")
        if self.cursor > len(self.ranked_candidate_ids):
            raise ValueError("SearchSession cursor exceeds candidate count")
        if self.shown_candidate_ids != self.ranked_candidate_ids[: self.cursor]:
            raise ValueError(
                "Shown candidate IDs must be the ranking prefix before the cursor"
            )
        if self.exhausted != (self.cursor >= len(self.ranked_candidate_ids)):
            raise ValueError("SearchSession exhausted state disagrees with cursor")
        exact_ids = [item.target_id for item in self.exact_anchor_matches]
        if len(set(exact_ids)) != len(exact_ids):
            raise ValueError("Exact Anchor targets must be deduplicated")
        if set(exact_ids).intersection(ranked):
            raise ValueError("Exact Element targets must not repeat in normal ranking")
        return self


class SearchBatch(SoftDocModel):
    search_session_id: str = Field(min_length=1)
    exact_anchor_matches: list[ExactAnchorMatch] = Field(default_factory=list)
    unresolved_anchors: list[AnchorResolution] = Field(default_factory=list)
    candidate_previews: list[CandidatePreview] = Field(default_factory=list)
    next_cursor: int = Field(ge=0)
    exhausted: bool
    retrieval_trace: list[SearchSessionTraceEntry] = Field(default_factory=list)
