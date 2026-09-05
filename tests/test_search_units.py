from __future__ import annotations

from pathlib import Path

from softdoc.models import ElementType
from softdoc.retrieval import (
    SearchUnitBuilder,
    SearchUnitConfig,
    html_to_text,
)
from softdoc.retrieval.tokenization import chunk_token_spans


def _units_for(result, element_id):
    return [unit for unit in result.units if unit.element_id == element_id]


def test_short_paragraph_creates_one_unit(parsed_document) -> None:
    paragraph = next(
        item for item in parsed_document.elements if item.element_type == ElementType.PARAGRAPH
    )

    result = SearchUnitBuilder().build(parsed_document)

    units = _units_for(result, paragraph.element_id)
    assert len(units) == 1
    assert units[0].content_text == paragraph.text
    assert units[0].part_index == 0
    assert units[0].part_count == 1


def test_long_paragraph_creates_stable_overlapping_parts(parsed_document) -> None:
    document = parsed_document.model_copy(deep=True)
    paragraph = next(
        item for item in document.elements if item.element_type == ElementType.PARAGRAPH
    )
    paragraph.text = " ".join(f"token{index}" for index in range(55))
    builder = SearchUnitBuilder(
        SearchUnitConfig(
            split_threshold_tokens=20,
            part_size_tokens=20,
            overlap_tokens=5,
        )
    )

    first = builder.build(document)
    second = builder.build(document)
    units = _units_for(first, paragraph.element_id)

    assert len(units) > 1
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert [unit.part_index for unit in units] == list(range(len(units)))
    assert all(unit.part_count == len(units) for unit in units)
    assert all(len(chunk_token_spans(unit.content_text)) <= 20 for unit in units)
    assert units[0].search_unit_id == second.units[
        first.units.index(units[0])
    ].search_unit_id


def test_table_html_is_flattened_and_split_without_a_cell_graph(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    table = next(
        item for item in document.elements if item.element_type == ElementType.TABLE
    )
    table.reference_label = "Table 77"
    table.html = "<table>" + "".join(
        f"<tr><td>row{index}</td><td>value{index}</td></tr>"
        for index in range(20)
    ) + "</table>"
    result = SearchUnitBuilder(
        SearchUnitConfig(
            split_threshold_tokens=12,
            part_size_tokens=12,
            overlap_tokens=2,
        )
    ).build(document)

    units = _units_for(result, table.element_id)

    assert len(units) > 1
    assert all("<td>" not in unit.search_text for unit in units)
    assert all("Table 77" in unit.search_text for unit in units)
    assert all(unit.display_label == "Table 77" for unit in units)
    assert all(
        unit.search_text[
            unit.content_search_char_start : unit.content_search_char_end
        ]
        == unit.content_text
        for unit in units
    )
    assert "row0\tvalue0" in html_to_text(table.html)


def test_units_map_back_to_existing_elements_in_document_order(
    parsed_document,
) -> None:
    result = SearchUnitBuilder().build(parsed_document)
    element_ids = {element.element_id for element in parsed_document.elements}

    assert result.units
    assert all(unit.element_id in element_ids for unit in result.units)
    assert all(unit.document_id == parsed_document.document_id for unit in result.units)
    assert [
        (unit.page_index, unit.reading_order, unit.element_id, unit.part_index)
        for unit in result.units
    ] == sorted(
        (
            unit.page_index,
            unit.reading_order,
            unit.element_id,
            unit.part_index,
        )
        for unit in result.units
    )


def test_relation_neighbor_text_is_not_copied_into_visual_unit(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    figure = next(
        item for item in document.elements if item.element_type == ElementType.FIGURE
    )
    caption = next(
        item
        for item in document.elements
        if item.element_type == ElementType.CAPTION
        and item.metadata.get("target_element_id") == figure.element_id
    )
    figure.reference_label = "Figure 1"
    caption.text = "NEIGHBOR_SECRET caption text"

    result = SearchUnitBuilder().build(document)
    figure_text = "\n".join(
        unit.search_text for unit in _units_for(result, figure.element_id)
    )

    assert "Figure 1" in figure_text
    assert "NEIGHBOR_SECRET" not in figure_text


def test_builder_is_idempotent_and_does_not_mutate_softdoc(parsed_document) -> None:
    before = parsed_document.model_dump(mode="json")
    builder = SearchUnitBuilder()

    first = builder.build(parsed_document)
    second = builder.build(parsed_document)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert parsed_document.model_dump(mode="json") == before


def test_unlabelled_visual_only_element_is_skipped_with_reason(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    figure = next(
        item for item in document.elements if item.element_type == ElementType.FIGURE
    )
    figure.image_path = Path("visual-only.png")
    figure.text = None
    figure.reference_label = None
    figure.section_path = ["Experiments"]

    result = SearchUnitBuilder().build(document)

    assert not _units_for(result, figure.element_id)
    skipped = next(
        item for item in result.skipped_elements if item.element_id == figure.element_id
    )
    assert skipped.reason == "no_searchable_visual_text"


def test_hybrid_generated_visual_text_is_preserved_but_not_indexed(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    figure = next(
        item for item in document.elements if item.element_type == ElementType.FIGURE
    )
    figure.text = "Invented Category 1 Category 2 values"
    figure.reference_label = None
    figure.metadata["generated_visual_text"] = True
    figure.metadata["retrieval_text_status"] = "unverified"

    result = SearchUnitBuilder().build(document)

    assert figure.text == "Invented Category 1 Category 2 values"
    assert not _units_for(result, figure.element_id)
    skipped = next(
        item for item in result.skipped_elements if item.element_id == figure.element_id
    )
    assert skipped.reason == "unverified_generated_visual_text"


def test_heading_does_not_duplicate_itself_from_section_path(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    heading = next(
        item for item in document.elements if item.element_type == ElementType.HEADING
    )
    heading.text = "2 Experiments"
    heading.section_path = ["Methods", "2 Experiments"]

    unit = _units_for(SearchUnitBuilder().build(document), heading.element_id)[0]

    assert unit.search_text == "Methods\n2 Experiments"
    assert unit.display_label == "2 Experiments"


def test_table_without_html_falls_back_to_own_text(parsed_document) -> None:
    document = parsed_document.model_copy(deep=True)
    table = next(
        item for item in document.elements if item.element_type == ElementType.TABLE
    )
    table.text = "Fallback textual table content"
    table.html = None

    unit = _units_for(SearchUnitBuilder().build(document), table.element_id)[0]

    assert "Fallback textual table content" in unit.search_text


def test_continued_table_without_own_headers_inherits_confirmed_fragment_headers(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    source = next(
        item for item in document.elements if item.element_type == ElementType.TABLE
    )
    source.html = (
        "<table><tr><th>Issue</th><th>Count</th></tr>"
        "<tr><td>URLs not accessible</td><td>159</td></tr></table>"
    )
    source.metadata["cross_page_table_fragment"] = {
        "status": "confirmed",
        "group_id": "table-group:test",
        "fragment_index": 0,
    }
    target_page = document.pages[-1]
    target = source.model_copy(deep=True)
    target.element_id = source.element_id + ":continuation"
    target.page_id = target_page.page_id
    target.page_number = target_page.page_number
    target.html = "<table><tr><td>URLs timed out</td><td>504</td></tr></table>"
    target.metadata["cross_page_table_fragment"] = {
        "status": "confirmed",
        "group_id": "table-group:test",
        "fragment_index": 1,
    }
    document.elements.append(target)

    units = _units_for(SearchUnitBuilder().build(document), target.element_id)

    assert units
    assert all(unit.table_header_cells == ["Issue", "Count"] for unit in units)
    assert all(
        unit.table_header_source_element_id == source.element_id for unit in units
    )
    assert all(unit.table_is_continuation for unit in units)


def test_relations_do_not_merge_distinct_elements(parsed_document) -> None:
    result = SearchUnitBuilder().build(parsed_document)

    assert all(
        unit.element_id == other.element_id
        for unit in result.units
        for other in result.units
        if unit.search_unit_id == other.search_unit_id
    )
    assert len({unit.search_unit_id for unit in result.units}) == len(result.units)
