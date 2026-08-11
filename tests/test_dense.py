from __future__ import annotations

import re
from collections.abc import Sequence

import pytest

from softdoc.models import ElementType
from softdoc.retrieval import (
    DenseConfig,
    DenseEncoderError,
    DenseIndex,
    DenseInputTooLongError,
    DenseModelUnavailableError,
    EncoderFingerprint,
    EncoderInputType,
    EncoderTokenSpan,
    MemoryEmbeddingCache,
    SearchUnitBuilder,
    SearchUnitConfig,
    SubQuestionInput,
    e5_prefixed_text,
    resolve_dense_device,
)


class MockEncoder:
    def __init__(self, *, revision: str = "mock-r1") -> None:
        self._fingerprint = EncoderFingerprint(
            model_name="mock-e5",
            model_revision=revision,
            tokenizer_revision=revision,
            embedding_dimension=3,
            max_length=512,
            pooling_method="attention_mask_mean",
            normalize_embeddings=True,
            dtype="float32",
        )
        self.calls: list[tuple[EncoderInputType, list[str]]] = []

    @property
    def fingerprint(self) -> EncoderFingerprint:
        return self._fingerprint

    def prepared_token_count(
        self,
        text: str,
        input_type: EncoderInputType,
    ) -> int:
        del input_type
        return len(self.content_token_spans(text)) + 3

    def content_token_spans(self, text: str) -> list[EncoderTokenSpan]:
        return [
            EncoderTokenSpan(start=match.start(), end=match.end())
            for match in re.finditer(r"\S+", text)
        ]

    def encode(
        self,
        texts: Sequence[str],
        input_type: EncoderInputType,
    ) -> list[list[float]]:
        self.calls.append((input_type, list(texts)))
        vectors = []
        for text in texts:
            lowered = text.casefold()
            if "needle" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "related" in lowered:
                vectors.append([0.8, 0.2, 0.0])
            else:
                vectors.append([0.0, 1.0, 0.0])
        return vectors

    @property
    def passage_text_count(self) -> int:
        return sum(
            len(texts)
            for input_type, texts in self.calls
            if input_type == EncoderInputType.PASSAGE
        )


def _single_unit_result(document, element_id, *, index_version="test-v1"):
    result = SearchUnitBuilder(
        SearchUnitConfig(
            split_threshold_tokens=10_000,
            part_size_tokens=10_000,
            overlap_tokens=0,
            index_version=index_version,
        )
    ).build(document)
    unit = next(item for item in result.units if item.element_id == element_id)
    return result.model_copy(update={"units": [unit]})


def test_e5_query_and_passage_prefixes_are_explicit() -> None:
    assert e5_prefixed_text("What changed?", EncoderInputType.QUERY) == (
        "query: What changed?"
    )
    assert e5_prefixed_text("The evidence", EncoderInputType.PASSAGE) == (
        "passage: The evidence"
    )


def test_cpu_device_is_honoured_without_consulting_cuda() -> None:
    assert resolve_dense_device("cpu", cuda_available=True) == "cpu"
    assert resolve_dense_device("auto", cuda_available=False) == "cpu"
    with pytest.raises(DenseModelUnavailableError, match="CUDA"):
        resolve_dense_device("cuda", cuda_available=False)


def test_short_search_unit_is_encoded_once_without_extra_segmentation(
    parsed_document,
) -> None:
    target = next(
        item
        for item in parsed_document.elements
        if item.element_type == ElementType.PARAGRAPH
    )
    build_result = _single_unit_result(parsed_document, target.element_id)
    encoder = MockEncoder()

    index = DenseIndex(build_result, encoder)

    assert len(index.segments) == 1
    assert index.segments[0].segment_text == build_result.units[0].search_text
    assert encoder.passage_text_count == 1


def test_overlength_unit_is_losslessly_split_before_encoding(parsed_document) -> None:
    document = parsed_document.model_copy(deep=True)
    target = next(
        item for item in document.elements if item.element_type == ElementType.PARAGRAPH
    )
    target.section_path = None
    target.text = " ".join(f"word{index}" for index in range(25))
    build_result = _single_unit_result(document, target.element_id)
    encoder = MockEncoder()
    config = DenseConfig(
        max_length=12,
        segment_token_budget=8,
        segment_overlap_tokens=2,
        batch_size=2,
    )

    index = DenseIndex(build_result, encoder, config=config)

    assert len(index.segments) > 1
    assert index.segments[0].search_text_char_start == 0
    assert index.segments[-1].search_text_char_end == len(
        build_result.units[0].search_text
    )
    assert all(segment.model_token_count <= 12 for segment in index.segments)
    assert all(
        left.search_text_char_end >= right.search_text_char_start
        for left, right in zip(index.segments, index.segments[1:])
    )
    assert encoder.passage_text_count == len(index.segments)


def test_dense_mock_ranking_merges_to_complete_element_ranking(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    paragraphs = [
        item for item in document.elements if item.element_type == ElementType.PARAGRAPH
    ][:2]
    paragraphs[0].text = "The needle evidence is here."
    paragraphs[1].text = "A related but different discussion."
    build_result = SearchUnitBuilder().build(document)
    index = DenseIndex(build_result, MockEncoder())

    result = index.search(SubQuestionInput(subquestion_id="Q1", text="needle"))

    assert result.candidates[0].element_id == paragraphs[0].element_id
    assert result.total_candidates == len(
        {unit.element_id for unit in build_result.units}
    )
    assert [item.dense_rank for item in result.candidates] == list(
        range(1, result.total_candidates + 1)
    )


def test_multiple_search_unit_parts_merge_using_best_dense_part(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    target = next(
        item for item in document.elements if item.element_type == ElementType.PARAGRAPH
    )
    target.section_path = None
    target.text = "a b c d e f g h i j k l needle m n o p q r s"
    build_result = SearchUnitBuilder(
        SearchUnitConfig(
            split_threshold_tokens=8,
            part_size_tokens=8,
            overlap_tokens=1,
        )
    ).build(document)

    result = DenseIndex(build_result, MockEncoder()).search(
        SubQuestionInput(subquestion_id="Q-parts", text="needle")
    )
    candidate = next(
        item for item in result.candidates if item.element_id == target.element_id
    )

    assert candidate.matched_part > 0
    assert sum(
        item.element_id == target.element_id for item in result.candidates
    ) == 1


def test_embedding_cache_hits_and_index_version_invalidates(parsed_document) -> None:
    target = next(
        item
        for item in parsed_document.elements
        if item.element_type == ElementType.PARAGRAPH
    )
    cache = MemoryEmbeddingCache()
    encoder = MockEncoder()
    first_build = _single_unit_result(
        parsed_document,
        target.element_id,
        index_version="index-v1",
    )

    first = DenseIndex(first_build, encoder, cache=cache)
    encoded_after_first = encoder.passage_text_count
    second = DenseIndex(first_build, encoder, cache=cache)
    second_build = _single_unit_result(
        parsed_document,
        target.element_id,
        index_version="index-v2",
    )
    third = DenseIndex(second_build, encoder, cache=cache)

    assert first.cache_misses == 1 and first.cache_writes == 1
    assert second.cache_hits == 1 and second.cache_misses == 0
    assert encoder.passage_text_count == encoded_after_first + 1
    assert third.cache_misses == 1


def test_overlength_query_is_rejected_instead_of_truncated(parsed_document) -> None:
    target = next(
        item
        for item in parsed_document.elements
        if item.element_type == ElementType.PARAGRAPH
    )
    index = DenseIndex(
        _single_unit_result(parsed_document, target.element_id),
        MockEncoder(),
        config=DenseConfig(
            max_length=12,
            segment_token_budget=8,
            segment_overlap_tokens=2,
        ),
    )

    with pytest.raises(DenseInputTooLongError, match="SubQuestion"):
        index.search(
            SubQuestionInput(
                subquestion_id="Q-long",
                text=" ".join(f"query{index}" for index in range(20)),
            )
        )


def test_invalid_embedding_shape_fails_clearly(parsed_document) -> None:
    class BadEncoder(MockEncoder):
        def encode(self, texts, input_type):
            return [[1.0, 2.0] for _ in texts]

    target = next(
        item
        for item in parsed_document.elements
        if item.element_type == ElementType.PARAGRAPH
    )
    with pytest.raises(DenseEncoderError, match="shape"):
        DenseIndex(
            _single_unit_result(parsed_document, target.element_id),
            BadEncoder(),
        )


def test_dense_result_is_stable_and_json_round_trippable(parsed_document) -> None:
    build_result = SearchUnitBuilder().build(parsed_document)
    index = DenseIndex(build_result, MockEncoder())
    question = SubQuestionInput(subquestion_id="Q-stable", text="needle")

    first = index.search(question)
    second = index.search(question)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert type(first).model_validate_json(first.model_dump_json()) == first
