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
