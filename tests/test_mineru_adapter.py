from __future__ import annotations

import json
from pathlib import Path

import pytest

from softdoc.adapters import MinerUAdapter
from softdoc.models import ElementType, RelationType
from softdoc.parser import DocumentParser


def test_adapter_satisfies_parser_protocol() -> None:
    assert isinstance(MinerUAdapter(), DocumentParser)


def test_stable_ids(mineru_fixture_dir: Path, tmp_path: Path) -> None:
    first = MinerUAdapter().parse(mineru_fixture_dir, tmp_path / "first")
    second = MinerUAdapter().parse(mineru_fixture_dir, tmp_path / "second")
    assert first.document_id == second.document_id
    assert [page.page_id for page in first.pages] == [page.page_id for page in second.pages]
    assert [element.element_id for element in first.elements] == [element.element_id for element in second.elements]
    assert [relation.relation_id for relation in first.relations] == [relation.relation_id for relation in second.relations]


def test_adapter_preserves_structure_and_raw_payload(parsed_document) -> None:
    assert len(parsed_document.pages) == 3
    assert any(element.element_type == ElementType.HEADING for element in parsed_document.elements)
    assert sum(element.element_type == ElementType.PARAGRAPH for element in parsed_document.elements) == 3
    assert any(element.element_type == ElementType.TABLE and element.html for element in parsed_document.elements)
    assert parsed_document.metadata["adapter_warnings"]
    paragraph = next(
        element
        for element in parsed_document.elements
        if element.provenance.raw_payload.get("experimental_field")
    )
    assert paragraph.provenance.raw_payload["experimental_field"].startswith("retained")


def test_caption_footnote_and_reading_order(parsed_document) -> None:
    caption_relations = [
        relation for relation in parsed_document.relations if relation.relation_type == RelationType.CAPTION_OF
    ]
    footnote_relations = [
        relation for relation in parsed_document.relations if relation.relation_type == RelationType.FOOTNOTE_OF
    ]
    assert len(caption_relations) == 3
    assert len(footnote_relations) == 1
    elements = {element.element_id: element for element in parsed_document.elements}
    assert all(elements[relation.source_id].element_type == ElementType.CAPTION for relation in caption_relations)
    assert all(
        elements[relation.target_id].element_type in {ElementType.FIGURE, ElementType.TABLE}
        for relation in caption_relations
    )
    for page in parsed_document.pages:
        assert page.reading_order == page.element_ids
        orders = [
            element.reading_order
            for element in parsed_document.elements
            if element.page_id == page.page_id
        ]
        assert orders == list(range(len(orders)))


def test_section_membership_crosses_pages(parsed_document) -> None:
    assert len(parsed_document.sections) == 1
    section = parsed_document.sections[0]
    assert len(section.page_ids) == 3
    assert all(element.section_id == section.section_id for element in parsed_document.elements)
    assert any(
        relation.relation_type == RelationType.BELONGS_TO_SECTION
        for relation in parsed_document.relations
    )


def test_document_page_and_page_element_relations(parsed_document) -> None:
    contains = [
        relation for relation in parsed_document.relations if relation.relation_type == RelationType.CONTAINS
    ]
    next_pages = [
        relation for relation in parsed_document.relations if relation.relation_type == RelationType.NEXT_PAGE
    ]
    assert sum(relation.source_id == parsed_document.document_id for relation in contains) == 3
    assert len(next_pages) == 2
    assert next_pages[0].source_id == parsed_document.pages[0].page_id
    assert next_pages[0].target_id == parsed_document.pages[1].page_id


def test_mineru_34_middle_schema_without_layout_json(tmp_path: Path) -> None:
    input_dir = tmp_path / "mineru_34"
    input_dir.mkdir()
    middle = {
        "_version_name": "3.4.4",
        "pdf_info": [
            {
                "page_idx": 0,
                "page_size": [595, 841],
                "para_blocks": [
                    {"type": "title", "bbox": [95, 67, 499, 85]},
                    {
                        "type": "image",
                        "bbox": [70, 72, 524, 266],
                        "blocks": [
                            {"type": "image_body", "bbox": [70, 72, 524, 266]},
                            {"type": "image_caption", "bbox": [67, 277, 526, 315]},
                        ],
                    },
                    {
                        "type": "code",
                        "bbox": [90, 330, 495, 480],
                        "blocks": [
                            {"type": "code_body", "bbox": [90, 330, 495, 480]},
                            {"type": "code_caption", "bbox": [67, 490, 527, 520]},
                        ],
                    },
                ],
                "discarded_blocks": [
                    {"type": "page_footnote", "bbox": [67, 751, 291, 774]}
                ],
            }
        ],
    }
    content_v2 = [
        [
            {
                "type": "title",
                "bbox": [159, 79, 838, 101],
                "content": {
                    "title_content": [{"type": "text", "content": "Test Paper"}],
                    "level": 1,
                },
            },
            {
                "type": "image",
                "bbox": [117, 85, 880, 316],
                "content": {
                    "content": "A diagram.",
                    "image_caption": [
                        {"type": "text", "content": "Figure 1: Test diagram."}
                    ],
                    "image_footnote": [],
                },
            },
            {
                "type": "algorithm",
                "bbox": [151, 392, 831, 571],
                "content": {
                    "algorithm_content": [
                        {"type": "text", "content": "STEP(x) -> y"}
                    ],
                    "algorithm_caption": [
                        {"type": "text", "content": "Figure 2: Test algorithm."}
                    ],
                    "algorithm_footnote": [],
                },
            },
            {
                "type": "page_footnote",
                "bbox": [112, 892, 489, 920],
                "content": {
                    "page_footnote_content": [
                        {"type": "text", "content": "1 Repository URL"}
                    ]
                },
            },
        ]
    ]
    (input_dir / "paper_middle.json").write_text(
        json.dumps(middle), encoding="utf-8"
    )
    (input_dir / "paper_content_list_v2.json").write_text(
        json.dumps(content_v2), encoding="utf-8"
    )

    document = MinerUAdapter().parse(input_dir, tmp_path / "output")

    assert document.document_id.startswith("doc:paper:")
    assert document.title == "Test Paper"
    assert document.provenance.parser_version == "3.4.4"
    assert (document.pages[0].width, document.pages[0].height) == (595, 841)
    figure = next(
        element
        for element in document.elements
        if element.element_type == ElementType.FIGURE
    )
    assert figure.bbox is not None
    assert figure.bbox.raw == (70.0, 72.0, 524.0, 266.0)
    assert figure.bbox.normalized[0] == pytest.approx(70 / 595)
    captions = [
        element
        for element in document.elements
        if element.element_type == ElementType.CAPTION
    ]
    assert len(captions) == 2
    assert all(caption.bbox is not None for caption in captions)
    assert any(
        element.element_type == ElementType.ALGORITHM
        for element in document.elements
    )
    page_footnote = next(
        element
        for element in document.elements
        if element.metadata.get("mineru_type") == "page_footnote"
    )
    assert page_footnote.text == "1 Repository URL"
    assert sum(
        relation.relation_type == RelationType.CAPTION_OF
        for relation in document.relations
    ) == 2
    assert not any(
        relation.relation_type == RelationType.FOOTNOTE_OF
        and relation.source_id == page_footnote.element_id
        for relation in document.relations
    )
