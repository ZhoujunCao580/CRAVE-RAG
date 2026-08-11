from __future__ import annotations

from softdoc.models import ElementType
from softdoc.pipeline import document_fingerprint
from softdoc.retrieval import (
    BM25Index,
    ExactAnchorLookup,
    SearchUnitBuilder,
    SearchUnitConfig,
    SubQuestionInput,
)


def _search(document, text: str, *, builder=None):
    build_result = (builder or SearchUnitBuilder()).build(document)
    return BM25Index(build_result).search(
        SubQuestionInput(subquestion_id="Q1", text=text)
    ), build_result


def test_bm25_retrieves_a_year_and_saves_offsets(parsed_document) -> None:
    document = parsed_document.model_copy(deep=True)
    target = next(
        item for item in document.elements if item.element_type == ElementType.PARAGRAPH
    )
    target.text = "The Orion programme reported its results in 2021."

    result, units = _search(document, "Orion results 2021")

    assert result.candidates[0].element_id == target.element_id
    assert {"orion", "results", "2021"}.issubset(
        result.candidates[0].matched_terms
    )
    matched_unit = next(
        unit
        for unit in units.units
        if unit.search_unit_id == result.candidates[0].matched_search_unit_id
    )
    assert all(
        matched_unit.search_text[offset.start:offset.end].casefold()
        == offset.term
        for offset in result.candidates[0].matched_offsets
    )


def test_bm25_indexes_figure_and_table_numbers_without_using_exact(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    figure = next(
        item for item in document.elements if item.element_type == ElementType.FIGURE
    )
    table = next(
        item for item in document.elements if item.element_type == ElementType.TABLE
    )
    figure.reference_label = "Figure 77"
    table.reference_label = "Table 88"

    figure_result, _ = _search(document, "Figure 77")
    table_result, _ = _search(document, "Table 88")

    assert figure_result.candidates[0].element_id == figure.element_id
    assert table_result.candidates[0].element_id == table.element_id


def test_bm25_retrieves_a_proper_name(parsed_document) -> None:
    document = parsed_document.model_copy(deep=True)
    target = next(
        item for item in document.elements if item.element_type == ElementType.PARAGRAPH
    )
    target.text = "Professor ZetaQuasar designed the evaluation protocol."

    result, _ = _search(document, "Who is ZetaQuasar?")

    assert result.candidates[0].element_id == target.element_id
    assert "zetaquasar" in result.candidates[0].matched_terms


def test_multiple_parts_merge_to_one_element_using_best_part(parsed_document) -> None:
    document = parsed_document.model_copy(deep=True)
    target = next(
        item for item in document.elements if item.element_type == ElementType.PARAGRAPH
    )
    target.text = (
        "a b c d e f g h needle x "
        "needle needle y z one two three four five six seven eight nine ten"
    )
    builder = SearchUnitBuilder(
        SearchUnitConfig(
            split_threshold_tokens=10,
            part_size_tokens=10,
            overlap_tokens=2,
        )
    )

    result, units = _search(document, "needle", builder=builder)
    target_units = [unit for unit in units.units if unit.element_id == target.element_id]

    assert len(target_units) > 1
    assert result.total_candidates == 1
    assert result.candidates[0].element_id == target.element_id
    assert result.candidates[0].matched_part > 0


def test_zero_match_returns_no_zero_score_elements(parsed_document) -> None:
    result, _ = _search(parsed_document, "termthatdoesnotexistanywhere")

    assert result.candidates == []
    assert result.total_candidates == 0
    assert result.trace[0].code == "no_lexical_match"


def test_equal_scores_use_page_reading_order_and_element_id(parsed_document) -> None:
    document = parsed_document.model_copy(deep=True)
    paragraphs = [
        item for item in document.elements if item.element_type == ElementType.PARAGRAPH
    ][:2]
    assert len(paragraphs) == 2
    for paragraph in paragraphs:
        paragraph.text = "stabletiephrase"
        paragraph.section_path = None

    result, _ = _search(document, "stabletiephrase")

    expected = sorted(
        paragraphs,
        key=lambda item: (item.page_number, item.reading_order, item.element_id),
    )
    assert [item.element_id for item in result.candidates] == [
        item.element_id for item in expected
    ]
    assert result.candidates[0].bm25_score == result.candidates[1].bm25_score
    assert [item.bm25_rank for item in result.candidates] == [1, 2]


def test_bm25_result_is_stable_and_json_round_trippable(parsed_document) -> None:
    build_result = SearchUnitBuilder().build(parsed_document)
    index = BM25Index(build_result)
    query = SubQuestionInput(subquestion_id="Q-stable", text="document representation")

    first = index.search(query)
    second = index.search(query)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert type(first).model_validate_json(first.model_dump_json()) == first


def test_exact_result_and_softdoc_are_not_changed_by_bm25(parsed_document) -> None:
    document = parsed_document.model_copy(deep=True)
    figure = next(
        item for item in document.elements if item.element_type == ElementType.FIGURE
    )
    figure.reference_label = "Figure 3"
    question = SubQuestionInput(subquestion_id="Q-exact", text="Read Figure 3")
    exact_before = ExactAnchorLookup().lookup(question, document)
    fingerprint_before = document_fingerprint(document)

    bm25_result, _ = _search(document, question.text)
    exact_after = ExactAnchorLookup().lookup(question, document)

    assert bm25_result.candidates
    assert exact_after == exact_before
    assert document_fingerprint(document) == fingerprint_before
