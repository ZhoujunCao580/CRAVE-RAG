from __future__ import annotations

from pathlib import Path

from softdoc.ids import bbox_id
from softdoc.models import (
    BoundingBox,
    Document,
    Element,
    ElementType,
    Page,
    Provenance,
)
from softdoc.normalization import ElementNormalizer
from softdoc.profiles import DocumentProfile


def _provenance(locator: str, raw_payload: dict | None = None) -> Provenance:
    return Provenance(
        provenance_id=f"prov:test:{locator}",
        adapter="mineru",
        source_path=Path("form.pdf"),
        source_locator=locator,
        raw_payload=raw_payload or {},
    )


def _element(
    element_id: str,
    element_type: ElementType,
    reading_order: int,
    text: str | None,
    *,
    raw_payload: dict | None = None,
) -> Element:
    return Element(
        element_id=element_id,
        document_id="doc:form",
        page_id="page:1",
        page_number=1,
        element_type=element_type,
        reading_order=reading_order,
        bbox=BoundingBox.from_raw(
            bbox_id=bbox_id(element_id),
            raw=(20, 40 + reading_order * 80, 580, 90 + reading_order * 80),
            page_width=600,
            page_height=800,
        ),
        text=text,
        provenance=_provenance(element_id, raw_payload),
        metadata={"mineru_type": element_type.value},
    )


def _document(elements: list[Element]) -> Document:
    page = Page(
        page_id="page:1",
        document_id="doc:form",
        page_index=0,
        page_number=1,
        width=600,
        height=800,
        element_ids=[element.element_id for element in elements],
        reading_order=[element.element_id for element in elements],
        provenance=_provenance("page"),
    )
    return Document(
        document_id="doc:form",
        source_path=Path("form.pdf"),
        pages=[page],
        elements=elements,
        provenance=_provenance("document"),
    )


def test_numbered_form_lists_become_atomic_stable_paragraphs() -> None:
    raw_items = [
        {
            "item_content": [{"content": "1. What is your name?"}],
            "bbox": [20, 40, 580, 90],
            "level": 0,
        },
        {
            "item_content": [{"content": "2. How can we contact you?"}],
            "bbox": [20, 100, 580, 150],
            "level": 0,
        },
    ]
    form_list = _element(
        "form-list",
        ElementType.LIST,
        0,
        "1. What is your name?\n2. How can we contact you?",
        raw_payload={"content": {"list_items": raw_items}},
    )
    third_question = _element(
        "third-question",
        ElementType.PARAGRAPH,
        1,
        "3. What is your preferred contact method?",
    )
    actual_list = _element(
        "actual-list",
        ElementType.LIST,
        2,
        "1. Prepare materials\n2. Submit the application",
        raw_payload={
            "content": {
                "list_items": [
                    {"item_content": [{"content": "1. Prepare materials"}]},
                    {"item_content": [{"content": "2. Submit the application"}]},
                ]
            }
        },
    )
    document = _document([form_list, third_question, actual_list])

    decisions = ElementNormalizer().normalize(document, DocumentProfile.REPORT)

    questions = [item for item in document.elements if item.metadata.get("form_question")]
    assert [item.text for item in questions] == [
        "1. What is your name?",
        "2. How can we contact you?",
        "3. What is your preferred contact method?",
    ]
    assert all(item.element_type == ElementType.PARAGRAPH for item in questions)
    assert all(item.metadata["form_question_marker"] for item in questions)
    assert questions[0].metadata["raw_list_item"]["level"] == 0
    assert questions[0].provenance.raw_payload["list_item"] == raw_items[0]
    assert all("form-question" in item.element_id for item in questions[:2])
    assert actual_list.element_type == ElementType.LIST
    assert actual_list in document.elements
    page = document.pages[0]
    assert page.element_ids == page.reading_order
    assert page.element_ids == [item.element_id for item in document.elements]
    assert [item.reading_order for item in document.elements] == list(
        range(len(document.elements))
    )
    assert any(
        decision.rule == "numbered_form_question_to_paragraph"
        for decision in decisions
    )


def test_consecutive_table_component_paragraphs_become_footnotes() -> None:
    table = _element("table", ElementType.TABLE, 0, "Name | Status | Score")
    note_a = _element(
        "note-a", ElementType.PARAGRAPH, 1, "a. The Name field identifies the respondent."
    )
    note_b = _element(
        "note-b", ElementType.PARAGRAPH, 2, "b. The Status field records the current state."
    )
    next_table = _element("next-table", ElementType.TABLE, 3, "Region | Total")
    document = _document([table, note_a, note_b, next_table])

    decisions = ElementNormalizer().normalize(document, DocumentProfile.REPORT)

    assert note_a.element_type == ElementType.FOOTNOTE
    assert note_b.element_type == ElementType.FOOTNOTE
    assert note_a.metadata["target_element_id"] == table.element_id
    assert note_b.metadata["target_element_id"] == table.element_id
    assert next_table.element_type == ElementType.TABLE
    assert {decision.element_id for decision in decisions if decision.rule == "table_component_paragraph_to_footnote"} == {
        note_a.element_id,
        note_b.element_id,
    }


def test_wrapped_regular_list_uses_layout_lines_for_atomic_question_boxes() -> None:
    form_list = _element(
        "wrapped-list",
        ElementType.LIST,
        0,
        "5. Unique Project Identifier: ABC-123\n"
        "6. What kind of investment will this be?\n"
        "Continue the explanatory note on this line.",
        raw_payload={
            "content": {
                "list_items": [
                    {
                        "item_content": [
                            {
                                "content": "5. Unique Project Identifier: ABC-123\n"
                                "6. What kind of investment will this be?\n"
                                "Continue the explanatory note on this line."
                            }
                        ]
                    }
                ]
            }
        },
    )
    form_list.provenance.metadata["layout_payload"] = {
        "lines": [
            {"bbox": [30, 100, 560, 130]},
            {"bbox": [30, 150, 560, 180]},
            {"bbox": [30, 181, 560, 205]},
        ]
    }
    document = _document([form_list])

    ElementNormalizer().normalize(document, DocumentProfile.REPORT)

    questions = document.elements
    assert [question.text for question in questions] == [
        "5. Unique Project Identifier: ABC-123",
        "6. What kind of investment will this be? Continue the explanatory note on this line.",
    ]
    assert questions[0].bbox is not None
    assert questions[1].bbox is not None
    assert questions[0].bbox.raw == (30.0, 100.0, 560.0, 130.0)
    assert questions[1].bbox.raw == (30.0, 150.0, 560.0, 205.0)
    assert questions[0].metadata["layout_line_index"] == 0
    assert questions[1].metadata["layout_line_index"] == 1


def test_isolated_numbered_question_list_becomes_paragraph() -> None:
    question = _element(
        "question-14",
        ElementType.LIST,
        0,
        "14. Does this investment support a program assessed using No\n"
        "the Program Assessment Rating Tool (PART)?",
        raw_payload={
            "content": {
                "list_items": [
                    {"item_content": [{"content": "14. Does this investment support a program assessed using No"}]},
                    {"item_content": [{"content": "the Program Assessment Rating Tool (PART)?"}]},
                ]
            }
        },
    )
    document = _document([question])

    ElementNormalizer().normalize(document, DocumentProfile.REPORT)

    assert len(document.elements) == 1
    assert document.elements[0].element_type == ElementType.PARAGRAPH
    assert document.elements[0].text == (
        "14. Does this investment support a program assessed using No "
        "the Program Assessment Rating Tool (PART)?"
    )
    assert document.elements[0].metadata["form_question_marker"] == "14."


def test_unrelated_alpha_paragraphs_after_table_are_preserved() -> None:
    table = _element("table", ElementType.TABLE, 0, "Name | Status")
    note_a = _element("note-a", ElementType.PARAGRAPH, 1, "a. This approach is cheaper.")
    note_b = _element("note-b", ElementType.PARAGRAPH, 2, "b. The result is easier to explain.")
    document = _document([table, note_a, note_b])

    ElementNormalizer().normalize(document, DocumentProfile.REPORT)

    assert note_a.element_type == ElementType.PARAGRAPH
    assert note_b.element_type == ElementType.PARAGRAPH
    assert "target_element_id" not in note_a.metadata


def test_empty_list_is_excluded_from_semantic_relations() -> None:
    empty = _element(
        "empty-list",
        ElementType.LIST,
        0,
        None,
        raw_payload={"content": {"list_items": []}},
    )
    document = _document([empty])

    decisions = ElementNormalizer().normalize(document, DocumentProfile.REPORT)

    assert empty.metadata["excluded_from_relations"] is True
    assert empty.metadata["empty_list"] is True
    assert decisions[0].rule == "empty_list_excluded_from_semantic_relations"
