from __future__ import annotations

from softdoc.models import BoundingBox, ElementType, PageLabelStatus
from softdoc.page_labels import PageLabelResolver


def _document_with_labels(parsed_document, page_count, observations):
    document = parsed_document.model_copy(deep=True)
    page_template = document.pages[0]
    element_template = next(
        element
        for element in document.elements
        if element.element_type == ElementType.PARAGRAPH
    )
    document.pages = []
    document.elements = []
    document.sections = []
    document.relations = []
    for page_number in range(1, page_count + 1):
        page_id = f"test:page:{page_number:04d}"
        page = page_template.model_copy(deep=True)
        object.__setattr__(page, "page_id", page_id)
        object.__setattr__(page, "page_index", page_number - 1)
        object.__setattr__(page, "page_number", page_number)
        object.__setattr__(page, "element_ids", [])
        object.__setattr__(page, "reading_order", [])
        object.__setattr__(page, "display_page_label", None)
        object.__setattr__(page, "display_page_label_confidence", None)
        object.__setattr__(page, "page_label_aliases", [])
        object.__setattr__(page, "page_label_candidates", [])
        document.pages.append(page)

    per_page_counts = {}
    for page_number, text, x1 in observations:
        index = per_page_counts.get(page_number, 0)
        per_page_counts[page_number] = index + 1
        page = document.pages[page_number - 1]
        element_id = f"test:page:{page_number:04d}:label:{index}"
        element = element_template.model_copy(deep=True)
        object.__setattr__(element, "element_id", element_id)
        object.__setattr__(element, "page_id", page.page_id)
        object.__setattr__(element, "page_number", page_number)
        object.__setattr__(element, "reading_order", index)
        object.__setattr__(element, "element_type", ElementType.PARAGRAPH)
        object.__setattr__(element, "text", text)
        object.__setattr__(
            element,
            "bbox",
            BoundingBox.from_raw(
                bbox_id=f"bbox:{element_id}",
                raw=(x1, 930, x1 + 20, 950),
                page_width=1000,
                page_height=1000,
            ),
        )
        document.elements.append(element)
        object.__setattr__(page, "element_ids", [*page.element_ids, element_id])
        object.__setattr__(page, "reading_order", [*page.reading_order, element_id])
    return document


def test_front_matter_remains_unlabelled_before_a_delayed_page_one(
    parsed_document,
) -> None:
    document = _document_with_labels(
        parsed_document,
        6,
        [(4, "1", 500), (5, "2", 500), (6, "3", 500)],
    )

    PageLabelResolver().resolve(document)

    assert [page.page_label_aliases for page in document.pages[:3]] == [[], [], []]
    assert [page.display_page_label for page in document.pages[3:]] == ["1", "2", "3"]


def test_bounded_missing_page_label_is_inferred_without_global_offset(
    parsed_document,
) -> None:
    document = _document_with_labels(
        parsed_document,
        4,
        [(2, "1", 500), (4, "3", 500)],
    )

    PageLabelResolver().resolve(document)

    assert document.pages[0].page_label_aliases == []
    assert document.pages[1].page_label_aliases == ["1"]
    assert document.pages[2].page_label_aliases == ["2"]
    assert document.pages[3].page_label_aliases == ["3"]
    inferred = document.pages[2].page_label_candidates
    assert len(inferred) == 1
    assert inferred[0].source.value == "sequence_inference"
    assert inferred[0].status == PageLabelStatus.CONFIRMED


def test_double_page_spread_keeps_two_printed_aliases_per_physical_page(
    parsed_document,
) -> None:
    document = _document_with_labels(
        parsed_document,
        8,
        [
            (3, "3", 500),
            (7, "10", 20),
            (7, "11", 970),
            (8, "12", 20),
            (8, "13", 970),
        ],
    )

    PageLabelResolver().resolve(document)

    assert document.pages[3].page_label_aliases == ["4", "5"]
    assert document.pages[4].page_label_aliases == ["6", "7"]
    assert document.pages[5].page_label_aliases == ["8", "9"]
    assert document.pages[6].page_label_aliases == ["10", "11"]
    assert document.pages[7].page_label_aliases == ["12", "13"]


def test_isolated_marginal_number_remains_candidate_not_addressable_label(
    parsed_document,
) -> None:
    document = _document_with_labels(
        parsed_document,
        3,
        [(2, "80", 120)],
    )

    PageLabelResolver().resolve(document)

    assert document.pages[1].page_label_aliases == []
    assert document.pages[1].page_label_candidates[0].status == PageLabelStatus.CANDIDATE


def test_adjacent_pages_with_a_skipped_printed_number_do_not_invent_it(
    parsed_document,
) -> None:
    document = _document_with_labels(
        parsed_document,
        2,
        [(1, "1", 500), (2, "3", 500)],
    )

    PageLabelResolver().resolve(document)

    assert document.pages[0].page_label_aliases == ["1"]
    assert document.pages[1].page_label_aliases == ["3"]
    assert all(
        candidate.label != "2"
        for page in document.pages
        for candidate in page.page_label_candidates
    )


def test_missing_half_of_established_two_page_spread_is_inferred(
    parsed_document,
) -> None:
    document = _document_with_labels(
        parsed_document,
        4,
        [
            (1, "2", 20),
            (1, "3", 970),
            (2, "5", 970),
            (3, "6", 20),
            (3, "7", 970),
            (4, "9", 970),
        ],
    )

    PageLabelResolver().resolve(document)

    assert document.pages[0].page_label_aliases == ["2", "3"]
    assert document.pages[1].page_label_aliases == ["4", "5"]
    assert document.pages[2].page_label_aliases == ["6", "7"]
    assert document.pages[3].page_label_aliases == ["8", "9"]


def test_missing_full_spread_between_local_anchors_is_inferred(
    parsed_document,
) -> None:
    document = _document_with_labels(
        parsed_document,
        3,
        [
            (1, "16", 20),
            (1, "17", 970),
            (3, "21", 970),
        ],
    )

    PageLabelResolver().resolve(document)

    assert document.pages[0].page_label_aliases == ["16", "17"]
    assert document.pages[1].page_label_aliases == ["18", "19"]
    assert document.pages[2].page_label_aliases == ["20", "21"]


def test_future_explicit_spread_can_validate_missing_left_alias(
    parsed_document,
) -> None:
    document = _document_with_labels(
        parsed_document,
        3,
        [
            (1, "3", 970),
            (2, "5", 970),
            (3, "6", 20),
            (3, "7", 970),
        ],
    )

    PageLabelResolver().resolve(document)

    assert document.pages[0].page_label_aliases == ["2", "3"]
    assert document.pages[1].page_label_aliases == ["4", "5"]
    assert document.pages[2].page_label_aliases == ["6", "7"]


def test_consecutive_labels_may_alternate_between_outer_corners(
    parsed_document,
) -> None:
    document = _document_with_labels(
        parsed_document,
        3,
        [(1, "41", 20), (2, "42", 970), (3, "43", 20)],
    )

    PageLabelResolver().resolve(document)

    assert [page.page_label_aliases for page in document.pages] == [
        ["41"],
        ["42"],
        ["43"],
    ]


def test_page_label_resolution_is_idempotent(parsed_document) -> None:
    document = _document_with_labels(
        parsed_document,
        4,
        [(2, "1", 500), (4, "3", 500)],
    )
    resolver = PageLabelResolver()

    resolver.resolve(document)
    first = document.model_dump(mode="json")
    resolver.resolve(document)

    assert document.model_dump(mode="json") == first
