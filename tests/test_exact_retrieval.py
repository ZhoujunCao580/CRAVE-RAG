from __future__ import annotations

from softdoc.models import ElementType, RelationType
from softdoc.pipeline import document_fingerprint
from softdoc.relations import RelationBuilder
from softdoc.retrieval import (
    AnchorKind,
    AnchorResolutionStatus,
    AnchorTargetType,
    ExactAnchorLookup,
    ExactLookupResult,
    SubQuestionInput,
)


def _lookup(document, text: str, *, subquestion_id: str = "Q1"):
    return ExactAnchorLookup().lookup(
        SubQuestionInput(subquestion_id=subquestion_id, text=text),
        document,
    )


def _with_reference_label(parsed_document, element_type: ElementType, label: str):
    document = parsed_document.model_copy(deep=True)
    target = next(
        element
        for element in document.elements
        if element.element_type == element_type
    )
    target.reference_label = label
    return document, target


def test_exact_figure_variants_are_normalized_and_deduplicated(
    parsed_document,
) -> None:
    document, target = _with_reference_label(
        parsed_document,
        ElementType.FIGURE,
        "Figure 42",
    )

    result = _lookup(document, "Compare Figure 42, Fig. 42, and 图42.")

    assert len(result.anchor_resolutions) == 1
    resolution = result.anchor_resolutions[0]
    assert resolution.anchor_kind == AnchorKind.FIGURE
    assert resolution.normalized_label == "42"
    assert resolution.status == AnchorResolutionStatus.UNIQUE
    assert resolution.matches[0].target_id == target.element_id
    assert resolution.matches[0].target_type == AnchorTargetType.FIGURE
    assert len(result.exact_anchor_matches) == 1
    assert sum(
        item.code == "duplicate_anchor_ignored" for item in result.trace
    ) == 2


def test_exact_table_uses_an_element_handle(parsed_document) -> None:
    document, target = _with_reference_label(
        parsed_document,
        ElementType.TABLE,
        "Table 77",
    )

    result = _lookup(document, "Read Table 77 and 表77.")

    match = result.exact_anchor_matches[0]
    assert match.target_id == target.element_id
    assert match.target_type == AnchorTargetType.TABLE
    assert match.page_id == target.page_id
    assert match.page_number == target.page_number


def test_page_anchor_is_one_based_and_returns_page_handle(
    parsed_document,
) -> None:
    page = next(page for page in parsed_document.pages if page.page_number == 3)

    result = _lookup(parsed_document, "Page 3, page 3, 第3页")

    assert len(result.anchor_resolutions) == 1
    match = result.exact_anchor_matches[0]
    assert match.target_id == page.page_id
    assert match.target_type == AnchorTargetType.PAGE
    assert match.page_number == 3
    assert page.page_index == 2


def test_section_anchor_uses_heading_page(parsed_document) -> None:
    document = parsed_document.model_copy(deep=True)
    section = document.sections[0]
    section.title = "4.1 Methods"
    heading = next(
        element
        for element in document.elements
        if element.element_id == section.heading_element_id
    )

    result = _lookup(document, "See Section 4.1 and 第4.1节.")

    assert len(result.anchor_resolutions) == 1
    match = result.exact_anchor_matches[0]
    assert match.target_id == section.section_id
    assert match.target_type == AnchorTargetType.SECTION
    assert match.section_id == section.section_id
    assert match.page_id == heading.page_id
    assert match.page_number == heading.page_number


def test_multiple_anchor_kinds_are_preserved_in_source_order(
    parsed_document,
) -> None:
    document, figure = _with_reference_label(
        parsed_document,
        ElementType.FIGURE,
        "Figure 42",
    )
    table = next(
        element
        for element in document.elements
        if element.element_type == ElementType.TABLE
    )
    table.reference_label = "Table 77"

    result = _lookup(document, "Compare Table 77 with Figure 42 on Page 3.")

    assert [item.anchor_kind for item in result.anchor_resolutions] == [
        AnchorKind.TABLE,
        AnchorKind.FIGURE,
        AnchorKind.PAGE,
    ]
    assert {item.target_id for item in result.exact_anchor_matches} == {
        table.element_id,
        figure.element_id,
        document.pages[2].page_id,
    }


def test_ambiguous_figure_keeps_all_targets_without_auto_selection(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    figures = [
        element
        for element in document.elements
        if element.element_type == ElementType.FIGURE
    ][:2]
    assert len(figures) == 2
    for figure in figures:
        figure.reference_label = "Figure 88"

    result = _lookup(document, "Inspect Figure 88.")

    resolution = result.anchor_resolutions[0]
    assert resolution.status == AnchorResolutionStatus.AMBIGUOUS
    assert resolution.reason == "multiple_targets"
    assert {match.target_id for match in resolution.matches} == {
        figure.element_id for figure in figures
    }
    assert result.exact_anchor_matches == []


def test_missing_number_is_unresolved_without_failure(parsed_document) -> None:
    result = _lookup(parsed_document, "Inspect Figure 999 and Section 999.1.")

    assert [item.status for item in result.anchor_resolutions] == [
        AnchorResolutionStatus.UNRESOLVED,
        AnchorResolutionStatus.UNRESOLVED,
    ]
    assert all(item.reason == "target_not_found" for item in result.anchor_resolutions)
    assert result.exact_anchor_matches == []


def test_number_boundaries_do_not_turn_figure_30_into_figure_3(
    parsed_document,
) -> None:
    document, _ = _with_reference_label(
        parsed_document,
        ElementType.FIGURE,
        "Figure 3",
    )

    result = _lookup(document, "Inspect Figure 30.")

    assert result.anchor_resolutions[0].normalized_label == "30"
    assert result.anchor_resolutions[0].status == AnchorResolutionStatus.UNRESOLVED


def test_no_supported_anchor_returns_trace_only(parsed_document) -> None:
    result = _lookup(parsed_document, "Which method performs best?")

    assert result.anchor_resolutions == []
    assert result.exact_anchor_matches == []
    assert [item.code for item in result.trace] == ["no_anchor_detected"]


def test_lookup_is_stable_round_trippable_and_does_not_mutate_inputs(
    parsed_document,
) -> None:
    document, _ = _with_reference_label(
        parsed_document,
        ElementType.FIGURE,
        "Figure 42",
    )
    subquestion = SubQuestionInput(
        subquestion_id="stable-Q",
        text="Read Figure 42 on Page 3.",
    )
    before_document = document_fingerprint(document)
    before_subquestion = subquestion.model_dump(mode="json")

    first = ExactAnchorLookup().lookup(subquestion, document)
    second = ExactAnchorLookup().lookup(subquestion, document)
    restored = ExactLookupResult.model_validate_json(first.model_dump_json())

    assert first == second == restored
    assert document_fingerprint(document) == before_document
    assert subquestion.model_dump(mode="json") == before_subquestion


def test_shared_label_builder_preserves_relation_builder_idempotency(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    expected = [
        relation.model_dump(mode="json")
        for relation in document.relations
        if relation.relation_type == RelationType.REFERS_TO
    ]

    RelationBuilder(document).build_all()

    actual = [
        relation.model_dump(mode="json")
        for relation in document.relations
        if relation.relation_type == RelationType.REFERS_TO
    ]
    assert actual == expected


def test_page_anchor_prefers_printed_label_over_physical_number(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    printed_target = document.pages[1]
    physical_page_three = document.pages[2]
    printed_target.page_label_aliases = ["3"]
    printed_target.display_page_label = "3"
    printed_target.display_page_label_confidence = 0.95

    result = _lookup(document, "Read Page 3.")

    match = result.exact_anchor_matches[0]
    assert match.target_id == printed_target.page_id
    assert match.target_id != physical_page_three.page_id
    assert match.resolution_method == "printed_page_label"


def test_word_ordinal_page_uses_physical_order_not_printed_label(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    physical_first = document.pages[0]
    printed_page_one = document.pages[2]
    printed_page_one.page_label_aliases = ["1"]
    printed_page_one.display_page_label = "1"
    printed_page_one.display_page_label_confidence = 0.95

    result = _lookup(document, "Read the first page.")

    match = result.exact_anchor_matches[0]
    assert match.target_id == physical_first.page_id
    assert match.target_id != printed_page_one.page_id
    assert match.resolution_method == "physical_document_order"


def test_numeric_and_word_ordinals_are_normalized_and_deduplicated(
    parsed_document,
) -> None:
    physical_second = parsed_document.pages[1]

    result = _lookup(parsed_document, "Compare the second page with the 2nd page.")

    assert len(result.anchor_resolutions) == 1
    assert len(result.exact_anchor_matches) == 1
    assert result.exact_anchor_matches[0].target_id == physical_second.page_id
    assert result.exact_anchor_matches[0].normalized_label == "2"
    assert result.exact_anchor_matches[0].resolution_method == "physical_document_order"
    assert sum(item.code == "duplicate_anchor_ignored" for item in result.trace) == 1


def test_last_page_uses_final_physical_page(parsed_document) -> None:
    physical_last = max(
        parsed_document.pages,
        key=lambda page: (page.page_index, page.page_id),
    )

    result = _lookup(parsed_document, "Inspect the last page.")

    match = result.exact_anchor_matches[0]
    assert match.target_id == physical_last.page_id
    assert match.normalized_label == "last"
    assert match.resolution_method == "physical_last_page"


def test_printed_page_and_ordinal_page_remain_distinct_when_targets_differ(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    physical_first = document.pages[0]
    printed_page_one = document.pages[2]
    printed_page_one.page_label_aliases = ["1"]
    printed_page_one.display_page_label = "1"
    printed_page_one.display_page_label_confidence = 0.95

    result = _lookup(document, "Compare Page 1 with the first page.")

    assert len(result.anchor_resolutions) == 2
    assert [match.target_id for match in result.exact_anchor_matches] == [
        printed_page_one.page_id,
        physical_first.page_id,
    ]
    assert [match.resolution_method for match in result.exact_anchor_matches] == [
        "printed_page_label",
        "physical_document_order",
    ]


def test_two_printed_labels_can_address_the_same_physical_page(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    spread = document.pages[1]
    spread.page_label_aliases = ["16", "17"]
    spread.display_page_label = "16"
    spread.display_page_label_confidence = 0.91

    result = _lookup(document, "Compare Page 16 with Page 17.")

    assert len(result.exact_anchor_matches) == 2
    assert {match.target_id for match in result.exact_anchor_matches} == {
        spread.page_id
    }
    assert all(
        match.resolution_method == "printed_page_label"
        for match in result.exact_anchor_matches
    )


def test_answer_format_example_anchors_are_ignored(parsed_document) -> None:
    result = _lookup(
        parsed_document,
        'List the relevant pages; format the answer, for example, ["Page 17", "Page 25"].',
    )

    assert result.anchor_resolutions == []
    assert result.exact_anchor_matches == []
    assert sum(item.code == "example_anchor_ignored" for item in result.trace) == 2


def test_numeric_figure_caption_is_not_registered_as_section(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    caption_relation = next(
        relation
        for relation in document.relations
        if relation.relation_type == RelationType.CAPTION_OF
    )
    caption = next(
        element
        for element in document.elements
        if element.element_id == caption_relation.source_id
    )
    caption.text = "9999 features"

    result = _lookup(document, "Inspect Section 9999.")

    assert result.anchor_resolutions[0].status == AnchorResolutionStatus.UNRESOLVED


def test_slide_anchor_uses_physical_page_order(parsed_document) -> None:
    result = _lookup(parsed_document, "Inspect Slide 2.")

    match = result.exact_anchor_matches[0]
    assert match.target_id == parsed_document.pages[1].page_id
    assert match.anchor_kind == AnchorKind.PAGE
    assert match.resolution_method == "physical_document_order"


def test_page_range_expands_to_individual_printed_page_anchors(
    parsed_document,
) -> None:
    result = _lookup(parsed_document, "Inspect Pages 1-3.")

    assert [item.normalized_label for item in result.anchor_resolutions] == [
        "1",
        "2",
        "3",
    ]
    assert [item.page_number for item in result.exact_anchor_matches] == [1, 2, 3]


def test_singular_figure_range_keeps_the_full_range_span(parsed_document) -> None:
    document, target = _with_reference_label(
        parsed_document,
        ElementType.FIGURE,
        "Figure 1",
    )

    result = _lookup(document, "Compare Figure 1-4.")

    assert [item.normalized_label for item in result.anchor_resolutions] == [
        "1",
        "2",
        "3",
        "4",
    ]
    assert result.anchor_resolutions[0].anchor_text == "Figure 1-4"
    assert result.anchor_resolutions[0].status == AnchorResolutionStatus.UNIQUE
    assert result.anchor_resolutions[0].matches[0].target_id == target.element_id


def test_page_parentheses_and_word_cardinal_are_supported(parsed_document) -> None:
    result = _lookup(parsed_document, "Compare page(2) and Page two.")

    assert len(result.exact_anchor_matches) == 1
    assert result.exact_anchor_matches[0].page_number == 2


def test_first_figure_uses_document_reading_order(parsed_document) -> None:
    page_indexes = {page.page_id: page.page_index for page in parsed_document.pages}
    expected = min(
        (
            element
            for element in parsed_document.elements
            if element.element_type in {ElementType.FIGURE, ElementType.CHART}
        ),
        key=lambda element: (
            page_indexes[element.page_id],
            element.reading_order,
            element.element_id,
        ),
    )

    result = _lookup(parsed_document, "Inspect the first figure.")

    assert result.exact_anchor_matches[0].target_id == expected.element_id
    assert result.exact_anchor_matches[0].resolution_method == "element_physical_order"


def test_roman_table_label_is_registered_and_resolved(parsed_document) -> None:
    document, target = _with_reference_label(
        parsed_document,
        ElementType.TABLE,
        "Table II",
    )

    result = _lookup(document, "Read Table II.")

    assert result.exact_anchor_matches[0].target_id == target.element_id
    assert result.exact_anchor_matches[0].normalized_label == "ii"


def test_appendix_letter_resolves_section_title(parsed_document) -> None:
    document = parsed_document.model_copy(deep=True)
    document.sections[0].title = "Appendix C - Supplemental results"

    result = _lookup(document, "Count the items in Appendix C.")

    assert result.exact_anchor_matches[0].target_id == document.sections[0].section_id
    assert result.exact_anchor_matches[0].resolution_method == "appendix_title"


def test_cover_page_resolves_first_physical_page(parsed_document) -> None:
    result = _lookup(parsed_document, "Inspect the cover page.")

    assert result.exact_anchor_matches[0].target_id == parsed_document.pages[0].page_id
    assert result.exact_anchor_matches[0].resolution_method == "physical_document_order"


def test_standalone_cover_resolves_first_physical_page(parsed_document) -> None:
    result = _lookup(parsed_document, "How many people are in the images on the cover?")

    assert result.exact_anchor_matches[0].target_id == parsed_document.pages[0].page_id


def test_cover_of_each_chapter_is_not_misread_as_document_cover(
    parsed_document,
) -> None:
    result = _lookup(parsed_document, "What animal is on the cover of each chapter?")

    assert result.exact_anchor_matches == []
    assert [item.code for item in result.trace] == ["no_anchor_detected"]


def test_named_section_title_resolves_with_simple_plural_variation(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    document.sections[0].title = "Recorded Demonstrations"

    result = _lookup(
        document,
        "How many videos are in the Recorded Demonstration section?",
    )

    assert result.exact_anchor_matches[0].target_id == document.sections[0].section_id
    assert result.exact_anchor_matches[0].resolution_method == "named_section_title"


def test_named_section_title_resolves_unique_multiword_suffix(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    document.sections[0].title = "3.5 Warning and Cautions"

    result = _lookup(document, "What is stated in the Section Warning and Cautions?")

    assert result.exact_anchor_matches[0].target_id == document.sections[0].section_id


def test_named_section_title_normalizes_programme_spelling(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    document.sections[0].title = "LEADERSHIP PROGRAMMES"

    result = _lookup(document, "What is listed in the Leadership program section?")

    assert result.exact_anchor_matches[0].target_id == document.sections[0].section_id


def test_named_section_containment_must_be_unique(parsed_document) -> None:
    document = parsed_document.model_copy(deep=True)
    document.sections[0].title = "5.4 Self-Correction with Model Editing"
    document.sections.append(
        document.sections[0].model_copy(
            update={
                "section_id": "section:second-self-correction",
                "title": "5.5 Multi-modal Self-Correction",
            }
        )
    )

    result = _lookup(document, 'What is listed in the "Self-Correction" section?')

    assert result.anchor_resolutions[0].status == AnchorResolutionStatus.UNRESOLVED
    assert result.exact_anchor_matches == []


def test_generic_section_question_is_not_treated_as_named_anchor(
    parsed_document,
) -> None:
    result = _lookup(parsed_document, "Which section explains the procedure?")

    assert result.exact_anchor_matches == []


def test_oversized_page_range_is_recognized_without_no_anchor_diagnostic(
    parsed_document,
) -> None:
    result = _lookup(parsed_document, "Count figures in Pages 400-640.")

    assert result.exact_anchor_matches == []
    assert [item.code for item in result.trace] == ["anchor_range_too_large"]


def test_list_format_example_introduced_by_like_is_not_an_anchor(
    parsed_document,
) -> None:
    result = _lookup(
        parsed_document,
        "Which chapters are included? Your answer should be formatted as a "
        "list like ['Chapter 1', 'Chapter 3'].",
    )

    assert result.anchor_resolutions == []
    assert sum(item.code == "example_anchor_ignored" for item in result.trace) == 2


def test_relative_slide_offset_is_not_misread_as_slide_number(
    parsed_document,
) -> None:
    result = _lookup(
        parsed_document,
        "Inspect Slide 2 and the slide two positions after it.",
    )

    assert [item.normalized_label for item in result.anchor_resolutions] == ["2"]


def test_figure_preposition_is_not_misread_as_roman_label(
    parsed_document,
) -> None:
    result = _lookup(parsed_document, "Which figure in Page 2 contains the result?")

    assert [item.anchor_kind for item in result.anchor_resolutions] == [AnchorKind.PAGE]
    assert [item.normalized_label for item in result.anchor_resolutions] == ["2"]


def test_compound_figure_parser_does_not_consume_following_prose(
    parsed_document,
) -> None:
    document, target = _with_reference_label(
        parsed_document,
        ElementType.FIGURE,
        "Figure 1",
    )

    result = _lookup(document, "In Figure 1, compared with the baseline, what changed?")

    assert [item.normalized_label for item in result.anchor_resolutions] == ["1"]
    assert result.exact_anchor_matches[0].target_id == target.element_id
