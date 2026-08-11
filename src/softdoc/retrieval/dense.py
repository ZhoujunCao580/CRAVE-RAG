"""Dense retrieval over SearchUnits with lossless model-specific segmentation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from softdoc.ids import stable_digest
from softdoc.retrieval.cache import EmbeddingCache, MemoryEmbeddingCache
from softdoc.retrieval.encoder import (
    DenseEncoderError,
    DenseInputTooLongError,
    TextEncoder,
)
from softdoc.retrieval.models import (
    DenseConfig,
    DenseElementCandidate,
    DenseSearchResult,
    DenseSegment,
    DenseTraceEntry,
    EmbeddingCacheKey,
    EmbeddingCacheRecord,
    EncoderInputType,
    SearchUnit,
    SearchUnitBuildResult,
    SubQuestionInput,
)


@dataclass(frozen=True)
class _ScoredSegment:
    segment: DenseSegment
    unit: SearchUnit
    score: float


class DenseIndex:
    """Eager single-document matrix index with Element-level best-part merge."""

    def __init__(
        self,
        build_result: SearchUnitBuildResult,
        encoder: TextEncoder,
        *,
        config: DenseConfig | None = None,
        cache: EmbeddingCache | None = None,
    ) -> None:
        self.build_result = build_result
        self.encoder = encoder
        self.config = config or DenseConfig()
        self.cache = cache or MemoryEmbeddingCache()
        if self.config.max_length > encoder.fingerprint.max_length:
            raise ValueError(
                "Dense max_length cannot exceed the encoder's declared limit"
            )

        self._units_by_id = {
            unit.search_unit_id: unit for unit in build_result.units
        }
        self.segments = _build_dense_segments(
            build_result.units,
            encoder,
            self.config,
        )
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_writes = 0
        self._matrix = self._encode_passages()

    def search(self, subquestion: SubQuestionInput) -> DenseSearchResult:
        if not self.segments:
            return self._result(
                subquestion,
                [],
                [
                    DenseTraceEntry(
                        code="empty_dense_index",
                        description="The Document has no Dense SearchUnit segments.",
                    )
                ],
            )

        query_count = self.encoder.prepared_token_count(
            subquestion.text,
            EncoderInputType.QUERY,
        )
        if query_count > self.config.max_length:
            raise DenseInputTooLongError(
                "Refusing to truncate SubQuestion: "
                f"{query_count} tokens exceeds {self.config.max_length}"
            )
        query_vectors = self.encoder.encode(
            [subquestion.text],
            EncoderInputType.QUERY,
        )
        if len(query_vectors) != 1:
            raise DenseEncoderError("Encoder did not return exactly one query vector")
        query = _normalized_vector(
            query_vectors[0],
            self.encoder.fingerprint.embedding_dimension,
        )
        scores = self._matrix @ query

        best_by_element: dict[str, _ScoredSegment] = {}
        for segment, raw_score in zip(self.segments, scores, strict=True):
            unit = self._units_by_id[segment.search_unit_id]
            item = _ScoredSegment(
                segment=segment,
                unit=unit,
                score=float(np.clip(raw_score, -1.0, 1.0)),
            )
            current = best_by_element.get(unit.element_id)
            if current is None or _segment_preference(item) < _segment_preference(
                current
            ):
                best_by_element[unit.element_id] = item

        ordered = sorted(best_by_element.values(), key=_element_preference)
        candidates = [
            _candidate(item, rank) for rank, item in enumerate(ordered, start=1)
        ]
        split_units = len(
            {
                segment.search_unit_id
                for segment in self.segments
                if segment.segment_count > 1
            }
        )
        trace = [
            DenseTraceEntry(
                code="dense_ranked",
                description=(
                    "Dense segments were scored by normalized dot product and "
                    "merged to one candidate per Element."
                ),
                data={
                    "query_token_count": query_count,
                    "split_search_units": split_units,
                    "cache_hits": self.cache_hits,
                    "cache_misses": self.cache_misses,
                    "cache_writes": self.cache_writes,
                    "merge_rule": "maximum_segment_score",
                    "tie_break": (
                        "score_desc,page_index,reading_order,element_id,"
                        "search_unit_id,segment_index"
                    ),
                },
            )
        ]
        return self._result(subquestion, candidates, trace)

    def _encode_passages(self) -> np.ndarray:
        dimension = self.encoder.fingerprint.embedding_dimension
        if not self.segments:
            return np.empty((0, dimension), dtype=np.float32)

        vectors: list[np.ndarray | None] = [None] * len(self.segments)
        missing: list[tuple[int, DenseSegment, EmbeddingCacheKey]] = []
        for index, segment in enumerate(self.segments):
            unit = self._units_by_id[segment.search_unit_id]
            key = self._cache_key(segment, unit)
            cached = self.cache.get(key)
            if cached is None:
                self.cache_misses += 1
                missing.append((index, segment, key))
            else:
                self.cache_hits += 1
                vectors[index] = _normalized_vector(cached, dimension)

        for start in range(0, len(missing), self.config.batch_size):
            batch = missing[start : start + self.config.batch_size]
            encoded = self.encoder.encode(
                [segment.segment_text for _, segment, _ in batch],
                EncoderInputType.PASSAGE,
            )
            if len(encoded) != len(batch):
                raise DenseEncoderError(
                    "Encoder output count does not match passage batch size"
                )
            for (index, _segment, key), vector in zip(batch, encoded, strict=True):
                normalized = _normalized_vector(vector, dimension)
                values = normalized.astype(np.float32).tolist()
                vectors[index] = normalized
                self.cache.put(EmbeddingCacheRecord(key=key, vector=values))
                self.cache_writes += 1

        if any(vector is None for vector in vectors):
            raise DenseEncoderError("Dense passage matrix contains a missing vector")
        return np.stack(vectors).astype(np.float32)  # type: ignore[arg-type]

    def _cache_key(
        self,
        segment: DenseSegment,
        unit: SearchUnit,
    ) -> EmbeddingCacheKey:
        fingerprint = self.encoder.fingerprint
        return EmbeddingCacheKey(
            cache_schema_version=self.config.cache_schema_version,
            document_id=segment.document_id,
            search_unit_id=segment.search_unit_id,
            dense_segment_id=segment.dense_segment_id,
            search_text_sha256=hashlib.sha256(
                segment.segment_text.encode("utf-8")
            ).hexdigest(),
            index_version=unit.index_version,
            segment_index_version=self.config.segment_index_version,
            model_name=fingerprint.model_name,
            model_revision=fingerprint.model_revision,
            tokenizer_revision=fingerprint.tokenizer_revision,
            pooling_method=fingerprint.pooling_method,
            max_length=self.config.max_length,
            normalize_embeddings=True,
            embedding_dimension=fingerprint.embedding_dimension,
            dtype="float32",
        )

    def _result(
        self,
        subquestion: SubQuestionInput,
        candidates: list[DenseElementCandidate],
        trace: list[DenseTraceEntry],
    ) -> DenseSearchResult:
        return DenseSearchResult(
            subquestion_id=subquestion.subquestion_id,
            document_id=self.build_result.document_id,
            index_version=self.build_result.index_version,
            model_name=self.encoder.fingerprint.model_name,
            total_search_units=len(self.build_result.units),
            total_dense_segments=len(self.segments),
            total_candidates=len(candidates),
            candidates=candidates,
            trace=trace,
        )


def _build_dense_segments(
    units: list[SearchUnit],
    encoder: TextEncoder,
    config: DenseConfig,
) -> list[DenseSegment]:
    result: list[DenseSegment] = []
    for unit in units:
        ranges = _safe_segment_ranges(unit.search_text, encoder, config)
        count = len(ranges)
        for index, (start, end, token_count) in enumerate(ranges):
            text = unit.search_text[start:end]
            segment_id = "dense-segment:" + stable_digest(
                unit.search_unit_id,
                index,
                start,
                end,
                text,
                config.segment_index_version,
                config.max_length,
                config.segment_token_budget,
                config.segment_overlap_tokens,
                encoder.fingerprint.model_name,
                encoder.fingerprint.tokenizer_revision,
            )
            result.append(
                DenseSegment(
                    dense_segment_id=segment_id,
                    document_id=unit.document_id,
                    search_unit_id=unit.search_unit_id,
                    element_id=unit.element_id,
                    segment_index=index,
                    segment_count=count,
                    segment_text=text,
                    search_text_char_start=start,
                    search_text_char_end=end,
                    model_token_count=token_count,
                )
            )
    return result


def _safe_segment_ranges(
    text: str,
    encoder: TextEncoder,
    config: DenseConfig,
) -> list[tuple[int, int, int]]:
    full_count = encoder.prepared_token_count(text, EncoderInputType.PASSAGE)
    if full_count <= config.max_length:
        return [(0, len(text), full_count)]

    spans = encoder.content_token_spans(text)
    if not spans:
        raise DenseInputTooLongError(
            "Overlength SearchUnit produced no token offsets for safe segmentation"
        )

    ranges: list[tuple[int, int, int]] = []
    start_index = 0
    while start_index < len(spans):
        end_index = min(
            len(spans),
            start_index + config.segment_token_budget,
        )
        while end_index > start_index:
            start = 0 if start_index == 0 else spans[start_index].start
            end = len(text) if end_index == len(spans) else spans[end_index].start
            segment_text = text[start:end]
            token_count = encoder.prepared_token_count(
                segment_text,
                EncoderInputType.PASSAGE,
            )
            if token_count <= config.max_length:
                break
            end_index -= 1
        else:
            raise DenseInputTooLongError(
                "A single tokenizer unit exceeds the configured Dense input limit"
            )

        if not segment_text:
            raise DenseInputTooLongError("Dense safe segmentation produced empty text")
        ranges.append((start, end, token_count))
        if end_index >= len(spans):
            break
        start_index = max(
            start_index + 1,
            end_index - config.segment_overlap_tokens,
        )

    if not ranges or ranges[0][0] != 0 or ranges[-1][1] != len(text):
        raise DenseEncoderError("Dense segmentation did not cover the complete text")
    if any(left[1] < right[0] for left, right in zip(ranges, ranges[1:])):
        raise DenseEncoderError("Dense segmentation left an uncovered text gap")
    return ranges


def _normalized_vector(values: list[float], dimension: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (dimension,):
        raise DenseEncoderError(
            f"Expected embedding shape ({dimension},), got {vector.shape}"
        )
    if not np.isfinite(vector).all():
        raise DenseEncoderError("Embedding contains NaN or infinite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise DenseEncoderError("Embedding has zero norm")
    return vector / norm


def _segment_preference(
    item: _ScoredSegment,
) -> tuple[float, int, int, str]:
    return (
        -item.score,
        item.unit.part_index,
        item.segment.segment_index,
        item.segment.dense_segment_id,
    )


def _element_preference(
    item: _ScoredSegment,
) -> tuple[float, int, int, str, str, int]:
    return (
        -item.score,
        item.unit.page_index,
        item.unit.reading_order,
        item.unit.element_id,
        item.unit.search_unit_id,
        item.segment.segment_index,
    )


def _candidate(item: _ScoredSegment, rank: int) -> DenseElementCandidate:
    return DenseElementCandidate(
        element_id=item.unit.element_id,
        dense_score=item.score,
        dense_rank=rank,
        matched_search_unit_id=item.unit.search_unit_id,
        matched_part=item.unit.part_index,
        matched_dense_segment_id=item.segment.dense_segment_id,
        matched_segment_index=item.segment.segment_index,
        matched_text_char_start=item.segment.search_text_char_start,
        matched_text_char_end=item.segment.search_text_char_end,
        page_id=item.unit.page_id,
        page_number=item.unit.page_number,
        section_id=item.unit.section_id,
        element_type=item.unit.element_type,
    )
