from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from softdoc.adapters import MinerUAdapter
from softdoc.models import (
    ContentAvailability,
    Document,
    ElementParseStatus,
    ElementType,
    RelationType,
)
from softdoc.normalization import ElementNormalizer
from softdoc.parser import DocumentParser
from softdoc.pipeline import SoftDocPipeline
from softdoc.serialization import load_document, write_document
from softdoc.store import DocumentStore


def _parse_mineru(input_dir: Path, output_dir: Path) -> Document:
    return SoftDocPipeline(MinerUAdapter()).parse(input_dir, output_dir)


def test_adapter_satisfies_parser_protocol() -> None:
    assert isinstance(MinerUAdapter(), DocumentParser)


def test_adapter_returns_raw_document_without_postprocessing(
    mineru_fixture_dir: Path,
    tmp_path: Path,
) -> None:
    document = MinerUAdapter().parse(
        mineru_fixture_dir,
        tmp_path / "raw_adapter_output",
    )

    assert document.sections == []
    assert document.relations == []
    assert "document_profile" not in document.metadata
    assert "heading_decisions" not in document.metadata
    assert "section_resolution_decisions" not in document.metadata


def test_slide_ocr_year_rule_does_not_change_contractions() -> None:
    corrected, corrections = ElementNormalizer._correct_slide_ocr(
        "we'll compare results from 'II, Oct 'I2, and in15"
    )
    assert corrected == "we'll compare results from '11, Oct '12, and in '15"
    assert [item["rule"] for item in corrections] == [
        "two_digit_year",
        "mixed_digit_year",
        "missing_year_apostrophe",
    ]


def test_stable_ids(mineru_fixture_dir: Path, tmp_path: Path) -> None:
    first = _parse_mineru(mineru_fixture_dir, tmp_path / "first")
    second = _parse_mineru(mineru_fixture_dir, tmp_path / "second")
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


def test_parser_declared_header_footer_are_preserved_and_excluded(tmp_path: Path) -> None:
    input_dir = tmp_path / "mineru_headers"
    input_dir.mkdir()
    layout = {
        "pdf_info": [
            {
                "page_idx": 0,
                "page_size": [1000, 1000],
                "para_blocks": [
                    {"type": "page_header", "bbox": [100, 20, 900, 45]},
                    {"type": "title", "bbox": [100, 150, 900, 190]},
                    {"type": "page_header", "bbox": [100, 300, 300, 340]},
                    {"type": "page_footer", "bbox": [100, 940, 900, 970]},
                ],
            }
        ]
    }
    content = [
        [
            {
                "type": "page_header",
                "bbox": [100, 20, 900, 45],
                "content": {
                    "page_header_content": [
                        {"type": "text", "content": "Running header"}
                    ]
                },
            },
            {
                "type": "title",
                "bbox": [100, 150, 900, 190],
                "content": {
                    "title_content": [
                        {"type": "text", "content": "1 Introduction"}
                    ],
                    "level": 1,
                },
            },
            {
                "type": "page_header",
                "bbox": [100, 300, 300, 340],
                "content": {
                    "page_header_content": [
                        {"type": "text", "content": "Local chart label"}
                    ]
                },
            },
            {
                "type": "page_footer",
                "bbox": [100, 940, 900, 970],
                "content": {
                    "page_footer_content": [
                        {"type": "text", "content": "Confidential"}
                    ]
                },
            },
        ]
    ]
    (input_dir / "layout.json").write_text(
        json.dumps(layout), encoding="utf-8"
    )
    (input_dir / "fixture_content_list_v2.json").write_text(
        json.dumps(content), encoding="utf-8"
    )

    document = _parse_mineru(input_dir, tmp_path / "output")
    marginal = [
        element
        for element in document.elements
        if element.metadata.get("repeated_region")
    ]

    assert [element.text for element in marginal] == [
        "Running header",
        "Confidential",
    ]
    assert all(element.element_type == ElementType.PARAGRAPH for element in marginal)
    assert all(element.section_id is None for element in marginal)
    local_label = next(
        element
        for element in document.elements
        if element.text == "Local chart label"
    )
    assert local_label.metadata.get("repeated_region") is None
    assert len(document.sections) == 1
    assert len(document.metadata["repeated_header_footer_decisions"]) == 2


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

    document = _parse_mineru(input_dir, tmp_path / "output")

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


def test_empty_headings_are_skipped_without_empty_sections(
    mineru_degraded_fixture_dir: Path,
    tmp_path: Path,
) -> None:
    document = _parse_mineru(
        mineru_degraded_fixture_dir,
        tmp_path / "output",
    )

    assert all(
        (element.text or "").strip()
        for element in document.elements
        if element.element_type == ElementType.HEADING
    )
    assert all(section.title.strip() for section in document.sections)
    assert any(
        element.text == "Content after empty headings survives."
        for element in document.elements
    )
    warnings = document.metadata["adapter_warnings"]
    skipped = [item for item in warnings if item["code"] == "empty_heading_skipped"]
    assert len(skipped) == 2
    assert {
        item["payload"]["raw_payload"]["metadata"]["source_document"]
        for item in skipped
    } == {
        "2024.ug.eprospectus.pdf",
        "catvsdogdlpycon15se-150512122612-lva1-app6891_95.pdf",
    }


def test_empty_tables_are_preserved_as_degraded_fallback_crops(
    mineru_degraded_fixture_dir: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    document = _parse_mineru(mineru_degraded_fixture_dir, output_dir)
    tables = [
        element
        for element in document.elements
        if element.element_type == ElementType.TABLE
    ]

    assert len(tables) == 7
    assert all(table.parse_status == ElementParseStatus.DEGRADED for table in tables)
    visual_tables = [
        table
        for table in tables
        if table.content_availability == ContentAvailability.VISUAL_ONLY
    ]
    unavailable_tables = [
        table
        for table in tables
        if table.content_availability == ContentAvailability.UNAVAILABLE
    ]
    assert len(visual_tables) == 6
    assert len(unavailable_tables) == 1
    assert unavailable_tables[0].bbox is None
    for table in visual_tables:
        assert table.crop_image_path is not None
        assert (output_dir / table.crop_image_path).is_file()
        assert table.provenance.raw_payload["metadata"]["source_case"]
    assert {
        table.provenance.raw_payload["metadata"]["source_document"]
        for table in visual_tables
    } == {
        "936c0e2c2e6c8e0c07c51bfaf7fd0a83.pdf",
        "afe620b9beac86c1027b96d31d396407.pdf",
        "DSA-278777.pdf",
        "e79deb02a0c0e87511080836c5d4347b.pdf",
        "Macbook_air.pdf",
    }
    full_page = next(
        table
        for table in visual_tables
        if table.metadata.get("source_case") == "full_page_empty_table"
        or table.provenance.raw_payload.get("metadata", {}).get("source_case")
        == "full_page_empty_table"
    )
    with Image.open(output_dir / full_page.crop_image_path) as crop:
        assert crop.width > 150
        assert crop.height > 150
    warnings = document.metadata["adapter_warnings"]
    assert sum(item["code"] == "degraded_table" for item in warnings) == 7
    assert any(
        item["code"] == "missing_asset"
        and item["payload"]["owner_id"] == full_page.element_id
        for item in warnings
    )


def test_block_conversion_error_is_isolated_and_round_trips(
    mineru_degraded_fixture_dir: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    document = _parse_mineru(mineru_degraded_fixture_dir, output_dir)

    assert any(
        element.text == "A later block still converts after an exception."
        for element in document.elements
    )
    failed = [
        item
        for item in document.metadata["adapter_warnings"]
        if item["code"] == "block_conversion_failed"
    ]
    assert len(failed) == 1
    assert failed[0]["payload"]["raw_payload"]["index"] == "not-an-integer"
    DocumentStore(document).validate_references(raise_on_error=True)

    artifact_dir = tmp_path / "artifact"
    write_document(document, artifact_dir, render_overlays=False)
    restored = load_document(artifact_dir)
    restored_tables = [
        element
        for element in restored.elements
        if element.element_type == ElementType.TABLE
    ]
    assert len(restored.pages) == 8
    assert len(restored_tables) == 7
    assert {
        (table.parse_status, table.content_availability)
        for table in restored_tables
    } == {
        (ElementParseStatus.DEGRADED, ContentAvailability.VISUAL_ONLY),
        (ElementParseStatus.DEGRADED, ContentAvailability.UNAVAILABLE),
    }


def test_slide_index_block_and_external_chart_title_are_preserved(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "mineru_slides"
    input_dir.mkdir()
    pages = [
        {
            "page_idx": page_index,
            "page_size": [1000, 600],
            "para_blocks": [],
            "discarded_blocks": [],
        }
        for page_index in range(4)
    ]
    pages[0]["para_blocks"] = [
        {
            "type": "index",
            "bbox": [0, 0, 1000, 600],
            "lines": [
                {"bbox": [50, 280, 360, 320]},
                {"bbox": [30, 540, 300, 570]},
            ],
        }
    ]
    pages[1]["para_blocks"] = [
        {"type": "title", "bbox": [50, 20, 850, 70]},
        {
            "type": "chart",
            "bbox": [70, 180, 930, 550],
            "blocks": [
                {"type": "chart_caption", "bbox": [250, 120, 750, 160]},
                {"type": "chart_body", "bbox": [70, 180, 930, 550]},
            ],
        },
    ]
    pages[2]["para_blocks"] = [
        {"type": "title", "bbox": [50, 20, 850, 70]},
        {"type": "paragraph", "bbox": [220, 110, 780, 150]},
        {"type": "chart", "bbox": [70, 170, 930, 550]},
    ]
    pages[3]["para_blocks"] = [
        {"type": "title", "bbox": [50, 20, 850, 70]},
        {
            "type": "chart",
            "bbox": [70, 170, 930, 550],
            "blocks": [
                {"type": "chart_body", "bbox": [70, 170, 930, 550]},
                {"type": "chart_caption", "bbox": [200, 560, 800, 585]},
            ],
        },
    ]
    content = [
        [
            {
                "type": "index",
                "bbox": [0, 0, 1000, 1000],
                "content": {
                    "list_type": "text_list",
                    "list_items": [
                        {
                            "item_type": "text",
                            "item_content": [
                                {"type": "text", "content": "MARKET GROWTH"}
                            ],
                        },
                        {
                            "item_type": "text",
                            "item_content": [
                                {
                                    "type": "text",
                                    "content": "THEBIGDATAGROUP.COM",
                                }
                            ],
                        },
                    ],
                },
            }
        ],
        [
            {
                "type": "title",
                "bbox": [50, 30, 850, 110],
                "content": {
                    "title_content": [
                        {"type": "text", "content": "Overall Revenue"}
                    ],
                    "level": 2,
                },
            },
            {
                "type": "chart",
                "bbox": [70, 300, 930, 920],
                "content": {
                    "content": "bar chart",
                    "chart_caption": [
                        {
                            "type": "text",
                            "content": "Vendors With Revenue Over $100M",
                        }
                    ],
                    "chart_footnote": [],
                },
            },
        ],
        [
            {
                "type": "title",
                "bbox": [50, 30, 850, 110],
                "content": {
                    "title_content": [
                        {"type": "text", "content": "Next Slide"}
                    ],
                    "level": 2,
                },
            },
            {
                "type": "paragraph",
                "bbox": [220, 183, 780, 250],
                "content": {
                    "paragraph_content": [
                        {
                            "type": "text",
                            "content": "Yearly Vendor Revenue",
                        }
                    ]
                },
            },
            {
                "type": "chart",
                "bbox": [70, 283, 930, 917],
                "content": {
                    "content": "second bar chart",
                    "chart_caption": [],
                    "chart_footnote": [],
                },
            },
        ],
        [
            {
                "type": "title",
                "bbox": [50, 30, 850, 110],
                "content": {
                    "title_content": [
                        {"type": "text", "content": "Final Slide"}
                    ],
                    "level": 2,
                },
            },
            {
                "type": "chart",
                "bbox": [70, 283, 930, 917],
                "content": {
                    "content": "third bar chart",
                    "chart_caption": [],
                    "chart_footnote": [
                        {
                            "type": "text",
                            "content": "Example.orgsource: Example Research",
                        }
                    ],
                },
            },
        ],
    ]
    (input_dir / "slides_middle.json").write_text(
        json.dumps({"pdf_info": pages}),
        encoding="utf-8",
    )
    (input_dir / "slides_content_list_v2.json").write_text(
        json.dumps(content),
        encoding="utf-8",
    )

    document = _parse_mineru(input_dir, tmp_path / "output")
    index_element = next(
        element
        for element in document.elements
        if element.metadata.get("mineru_type") == "index"
    )
    assert index_element.element_type == ElementType.HEADING
    assert index_element.text == "MARKET GROWTH"
    assert index_element.heading_level == 1
    assert (
        index_element.metadata["element_normalization"]["rule"]
        == "slide_grouped_divider_title_recovered"
    )
    assert index_element.metadata["grouped_auxiliary_items"][0]["text"] == (
        "THEBIGDATAGROUP.COM"
    )
    assert not any(
        warning["code"] == "unsupported_block_type"
        and warning["payload"].get("type") == "index"
        for warning in document.metadata["adapter_warnings"]
    )

    chart = next(
        element
        for element in document.elements
        if element.element_type == ElementType.CHART
    )
    caption = next(
        element
        for element in document.elements
        if element.text == "Vendors With Revenue Over $100M"
    )
    assert caption.element_type == ElementType.CAPTION
    assert (
        caption.metadata["element_normalization"]["rule"]
        == "external_visual_title_preserved"
    )
    assert any(
        relation.relation_type == RelationType.CAPTION_OF
        and relation.source_id == caption.element_id
        and relation.target_id == chart.element_id
        for relation in document.relations
    )
    promoted_title = next(
        element
        for element in document.elements
        if element.text == "Yearly Vendor Revenue"
    )
    assert promoted_title.element_type == ElementType.CAPTION
    assert (
        promoted_title.metadata["element_normalization"]["rule"]
        == "slide_paragraph_chart_title_to_caption"
    )
    assert any(
        relation.relation_type == RelationType.CAPTION_OF
        and relation.source_id == promoted_title.element_id
        for relation in document.relations
    )
    attribution = next(
        element
        for element in document.elements
        if element.text == "Example.orgsource: Example Research"
    )
    assert attribution.element_type == ElementType.FOOTNOTE
    assert (
        attribution.metadata["element_normalization"]["rule"]
        == "visual_attribution_preserved"
    )
    assert not any(
        relation.relation_type == RelationType.CAPTION_OF
        and relation.source_id == attribution.element_id
        for relation in document.relations
    )


def test_repeated_slide_header_style_becomes_primary_heading(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "repeated_slide_titles"
    input_dir.mkdir()
    title_texts = [
        "Revenue $5.IB in 'II",
        "DATA SENSED PER YEAR",
        "BIG DATA LAW #2",
        "Contact",
    ]
    pages = []
    content = []
    for page_index, title_text in enumerate(title_texts):
        body_type = "title" if page_index == 2 else "paragraph"
        body_key = (
            "title_content"
            if body_type == "title"
            else "paragraph_content"
        )
        pages.append(
            {
                "page_idx": page_index,
                "page_size": [1000, 600],
                "para_blocks": [
                    {"type": "header", "bbox": [40, 20, 430, 56]},
                    {
                        "type": body_type,
                        "bbox": [140, 260, 860, 390],
                    },
                ],
                "discarded_blocks": [],
            }
        )
        content.append(
            [
                {
                    "type": "page_header",
                    "bbox": [40, 33, 430, 93],
                    "content": {
                        "page_header_content": [
                            {"type": "text", "content": title_text}
                        ]
                    },
                },
                {
                    "type": body_type,
                    "bbox": [140, 433, 860, 650],
                    "content": {
                        body_key: [
                            {
                                "type": "text",
                                "content": (
                                    "This is explanatory slide body text "
                                    "rather than another section heading."
                                ),
                            }
                        ],
                        **({"level": 2} if body_type == "title" else {}),
                    },
                },
            ]
        )
    (input_dir / "slides_middle.json").write_text(
        json.dumps({"pdf_info": pages}),
        encoding="utf-8",
    )
    (input_dir / "slides_content_list_v2.json").write_text(
        json.dumps(content),
        encoding="utf-8",
    )

    document = _parse_mineru(input_dir, tmp_path / "output")
    headings = [
        element
        for element in document.elements
        if element.element_type == ElementType.HEADING
    ]
    assert {element.text for element in headings} == {
        "Revenue $5.1B in '11",
        "DATA SENSED PER YEAR",
        "BIG DATA LAW #2",
        "Contact",
    }
    assert all(element.heading_level == 1 for element in headings)
    assert all(element.reading_order == 0 for element in headings)
    body = next(
        element
        for element in document.elements
        if element.page_number == 3
        and element.element_type == ElementType.PARAGRAPH
        and "explanatory" in (element.text or "")
    )
    page_heading = next(
        element for element in headings if element.page_number == 3
    )
    assert body.section_id == page_heading.section_id
    assert (
        body.metadata["element_normalization"]["rule"]
        == "slide_body_text_demoted_below_primary_title"
    )


def test_sparse_central_slide_title_is_promoted(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "sparse_divider_slides"
    input_dir.mkdir()
    pages = []
    content = []
    for page_index in range(5):
        if page_index == 1:
            pages.append(
                {
                    "page_idx": page_index,
                    "page_size": [1000, 600],
                    "para_blocks": [
                        {
                            "type": "paragraph",
                            "bbox": [310, 280, 690, 320],
                        },
                        {
                            "type": "footer",
                            "bbox": [30, 560, 300, 585],
                        },
                    ],
                    "discarded_blocks": [],
                }
            )
            content.append(
                [
                    {
                        "type": "paragraph",
                        "bbox": [310, 467, 690, 533],
                        "content": {
                            "paragraph_content": [
                                {
                                    "type": "text",
                                    "content": "WHAT IS BIG DATA?",
                                }
                            ]
                        },
                    },
                    {
                        "type": "page_footer",
                        "bbox": [30, 933, 300, 975],
                        "content": {
                            "page_footer_content": [
                                {
                                    "type": "text",
                                    "content": "EXAMPLE.COM",
                                }
                            ]
                        },
                    },
                ]
            )
        else:
            pages.append(
                {
                    "page_idx": page_index,
                    "page_size": [1000, 600],
                    "para_blocks": [
                        {"type": "image", "bbox": [0, 0, 1000, 600]}
                    ],
                    "discarded_blocks": [],
                }
            )
            content.append(
                [
                    {
                        "type": "image",
                        "bbox": [0, 0, 1000, 1000],
                        "content": {
                            "content": "full-page example",
                            "image_caption": [],
                            "image_footnote": [],
                        },
                    }
                ]
            )
    (input_dir / "slides_middle.json").write_text(
        json.dumps({"pdf_info": pages}),
        encoding="utf-8",
    )
    (input_dir / "slides_content_list_v2.json").write_text(
        json.dumps(content),
        encoding="utf-8",
    )

    document = _parse_mineru(input_dir, tmp_path / "output")
    title = next(
        element
        for element in document.elements
        if element.text == "WHAT IS BIG DATA?"
    )
    assert title.element_type == ElementType.HEADING
    assert title.heading_level == 1
    assert title.section_id is not None
    assert (
        title.metadata["element_normalization"]["rule"]
        == "sparse_slide_divider_paragraph_promoted"
    )


def test_spatially_distinct_caption_array_is_not_concatenated(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "split_captions"
    input_dir.mkdir()
    layout = {
        "pdf_info": [
            {
                "page_idx": 0,
                "page_size": [1000, 600],
                "para_blocks": [
                    {
                        "type": "image",
                        "bbox": [20, 250, 450, 500],
                        "blocks": [
                            {
                                "type": "image_caption",
                                "bbox": [150, 190, 300, 225],
                            },
                            {
                                "type": "image_body",
                                "bbox": [20, 250, 450, 500],
                            },
                            {
                                "type": "image_caption",
                                "bbox": [150, 520, 300, 555],
                            },
                        ],
                    }
                ],
                "discarded_blocks": [],
            }
        ]
    }
    content = [
        [
            {
                "type": "image",
                "bbox": [20, 417, 450, 833],
                "content": {
                    "content": "visual",
                    "image_caption": [
                        {"type": "text", "content": "1 feature"},
                        {"type": "text", "content": "2 features"},
                    ],
                    "image_footnote": [],
                },
            }
        ]
    ]
    (input_dir / "layout.json").write_text(
        json.dumps(layout),
        encoding="utf-8",
    )
    (input_dir / "fixture_content_list_v2.json").write_text(
        json.dumps(content),
        encoding="utf-8",
    )

    document = _parse_mineru(input_dir, tmp_path / "output")
    captions = [
        element
        for element in document.elements
        if element.element_type == ElementType.CAPTION
    ]
    assert [caption.text for caption in captions] == [
        "1 feature",
        "2 features",
    ]
    assert len({caption.element_id for caption in captions}) == 2


def test_progressive_slide_repairs_missing_upper_visual_target(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "progressive_slides"
    input_dir.mkdir()
    (input_dir / "pages").mkdir()
    for page_index in range(4):
        Image.new("RGB", (1000, 600), "white").save(
            input_dir / "pages" / f"page_{page_index}.png"
        )
    pages = []
    content = []
    for page_index in range(5):
        if page_index == 0:
            pages.append(
                {
                    "page_idx": page_index,
                    "page_size": [1000, 600],
                    "para_blocks": [
                        {
                            "type": "paragraph",
                            "bbox": [150, 190, 300, 225],
                        }
                    ],
                    "discarded_blocks": [],
                }
            )
            content.append(
                [
                    {
                        "type": "paragraph",
                        "bbox": [150, 317, 300, 375],
                        "content": {
                            "paragraph_content": [
                                {"type": "text", "content": "I feature"}
                            ]
                        },
                    }
                ]
            )
            continue
        if page_index == 1:
            body_bbox = [20, 260, 450, 500]
            blocks = [
                {"type": "image_caption", "bbox": [150, 190, 300, 225]},
                {"type": "image_body", "bbox": body_bbox},
                {"type": "image_caption", "bbox": [150, 520, 300, 555]},
            ]
            captions = [
                {"type": "text", "content": "I feature"},
                {"type": "text", "content": "2 features"},
            ]
        else:
            body_bbox = [50, 0, 470, 180]
            blocks = [
                {"type": "image_body", "bbox": body_bbox},
                {"type": "image_caption", "bbox": [150, 190, 300, 225]},
            ]
            captions = [{"type": "text", "content": "I feature"}]
        pages.append(
            {
                "page_idx": page_index,
                "page_size": [1000, 600],
                "para_blocks": [
                    {
                        "type": "image",
                        "bbox": body_bbox,
                        "blocks": blocks,
                    }
                ],
                "discarded_blocks": [],
            }
        )
        content.append(
            [
                {
                    "type": "image",
                    "bbox": [
                        body_bbox[0],
                        round(body_bbox[1] / 0.6),
                        body_bbox[2],
                        round(body_bbox[3] / 0.6),
                    ],
                    "content": {
                        "content": "visual",
                        "image_caption": captions,
                        "image_footnote": [],
                    },
                }
            ]
        )
    (input_dir / "slides_middle.json").write_text(
        json.dumps({"pdf_info": pages}),
        encoding="utf-8",
    )
    (input_dir / "slides_content_list_v2.json").write_text(
        json.dumps(content),
        encoding="utf-8",
    )

    document = _parse_mineru(input_dir, tmp_path / "output")
    paragraph_page = next(
        page for page in document.pages if page.page_number == 1
    )
    paragraph_elements = [
        element
        for element in document.elements
        if element.page_id == paragraph_page.page_id
    ]
    paragraph_label = next(
        element
        for element in paragraph_elements
        if element.text == "1 feature"
    )
    paragraph_visual = next(
        element
        for element in paragraph_elements
        if element.metadata.get("recovered_visual_region")
    )
    assert paragraph_label.element_type == ElementType.CAPTION
    assert any(
        relation.relation_type == RelationType.CAPTION_OF
        and relation.source_id == paragraph_label.element_id
        and relation.target_id == paragraph_visual.element_id
        for relation in document.relations
    )

    first_page = next(page for page in document.pages if page.page_number == 2)
    first_elements = [
        element
        for element in document.elements
        if element.page_id == first_page.page_id
    ]
    recovered = next(
        element
        for element in first_elements
        if element.metadata.get("recovered_visual_region")
    )
    captions = {
        element.text: element
        for element in first_elements
        if element.element_type == ElementType.CAPTION
    }
    caption_relations = [
        relation
        for relation in document.relations
        if relation.relation_type == RelationType.CAPTION_OF
        and relation.source_id
        in {caption.element_id for caption in captions.values()}
    ]
    targets = {
        captions[text].text: next(
            relation.target_id
            for relation in caption_relations
            if relation.source_id == captions[text].element_id
        )
        for text in captions
    }
    original_visual = next(
        element
        for element in first_elements
        if element.element_type == ElementType.FIGURE
        and element.element_id != recovered.element_id
    )
    assert targets["1 feature"] == recovered.element_id
    assert targets["2 features"] == original_visual.element_id
