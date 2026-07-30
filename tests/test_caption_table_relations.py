from __future__ import annotations

from pathlib import Path

from softdoc.ids import bbox_id, element_id, provenance_id
from softdoc.models import (
    BoundingBox,
    Document,
    Element,
    ElementType,
    Page,
    Provenance,
)
from softdoc.relations import RelationBuilder


def _element(
    document,
    page,
    order: int,
    element_type: ElementType,
    name: str,
    *,
    text: str | None = None,
    html: str | None = None,
    bbox: tuple[float, float, float, float] = (100, 100, 900, 300),
    metadata: dict | None = None,
) -> Element:
    item_id = element_id(
        document.document_id,
        page.page_index,
        order,
        element_type.value,
        name,
    )
    return Element(
        element_id=item_id,
        document_id=document.document_id,
        page_id=page.page_id,
        page_number=page.page_number,
        element_type=element_type,
        reading_order=order,
        bbox=BoundingBox.from_raw(
            bbox_id=bbox_id(item_id),
            raw=bbox,
            page_width=1000,
            page_height=1400,
        ),
        text=text,
        html=html,
        provenance=Provenance(
            provenance_id=provenance_id("test", "caption-table", item_id),
            adapter="test",
            source_path=document.source_path,
            source_locator=item_id,
        ),
        metadata=metadata or {},
    )


def _replace_page_content(document, page, elements: list[Element]) -> None:
    document.pages[page.page_index] = page.model_copy(
        update={
            "element_ids": [element.element_id for element in elements],
            "reading_order": [element.element_id for element in elements],
        }
    )


def _document(*, page_count: int = 2) -> Document:
    document_id = "caption-table-relations"
    provenance = Provenance(
        provenance_id=provenance_id("test", "caption-table", document_id),
        adapter="test",
        source_path=Path("fixture"),
        source_locator=document_id,
    )
    return Document(
        document_id=document_id,
        source_path=Path("fixture"),
        provenance=provenance,
        pages=[
            Page(
                page_id=f"page-{index}",
                document_id=document_id,
                page_index=index,
                page_number=index + 1,
                width=1000,
                height=1400,
                provenance=provenance.model_copy(deep=True),
            )
            for index in range(page_count)
        ],
    )


def test_table_continuation_ignores_running_header_with_multiple_tables(
) -> None:
    document = _document()
    source_page, target_page = document.pages[:2]
    source = _element(
        document,
        source_page,
        1,
        ElementType.TABLE,
        "source",
        html=(
            "<table><tr><th>Register</th><th>Value</th></tr>"
            "<tr><td>A</td><td>1</td></tr></table>"
        ),
        bbox=(100, 850, 900, 1390),
    )
    source_other = _element(
        document,
        source_page,
        0,
        ElementType.TABLE,
        "source-other",
        html="<table><tr><th>Unrelated</th></tr><tr><td>x</td></tr></table>",
        bbox=(100, 100, 900, 400),
    )
    running_header = _element(
        document,
        target_page,
        0,
        ElementType.TABLE,
        "running-header",
        text="Product Technical Manual",
        html="<table><tr><td>Product Technical Manual</td></tr></table>",
        bbox=(100, 20, 900, 50),
        metadata={"mineru_type": "page_header"},
    )
    target = _element(
        document,
        target_page,
        1,
        ElementType.TABLE,
        "target",
        html=(
            "<table><tr><th>Register</th><th>Value</th></tr>"
            "<tr><td>B</td><td>2</td></tr></table>"
        ),
        bbox=(100, 60, 900, 700),
    )
    target_other = _element(
        document,
        target_page,
        2,
        ElementType.TABLE,
        "target-other",
        html="<table><tr><th>Different</th></tr><tr><td>x</td></tr></table>",
        bbox=(100, 760, 900, 1200),
    )
    _replace_page_content(document, source_page, [source_other, source])
    _replace_page_content(
        document, target_page, [running_header, target, target_other]
    )
    document.pages = document.pages[:2]
    document.elements = [source_other, source, running_header, target, target_other]
    document.sections = []
    document.relations = []

    relations = RelationBuilder(document).build_cross_page_continuation_candidates()

    relation = next(item for item in relations if item.source_id == source.element_id)
    assert relation.target_id == target.element_id
    assert relation.status.value == "candidate"
    assert relation.evidence[0].data["table_headers_match"] is True


def test_repeated_page_header_promoted_to_caption_has_no_caption_relation(
) -> None:
    document = _document()
    first_page, second_page = document.pages[:2]
    first_header = _element(
        document,
        first_page,
        0,
        ElementType.PARAGRAPH,
        "first-header",
        text="Product Technical Manual",
        metadata={"mineru_type": "page_header"},
    )
    table = _element(
        document,
        first_page,
        1,
        ElementType.TABLE,
        "table",
        html="<table><tr><th>Value</th></tr><tr><td>1</td></tr></table>",
    )
    promoted_caption = _element(
        document,
        first_page,
        2,
        ElementType.CAPTION,
        "promoted-header",
        text="Product Technical Manual",
        metadata={"target_element_id": table.element_id},
    )
    second_header = _element(
        document,
        second_page,
        0,
        ElementType.PARAGRAPH,
        "second-header",
        text="Product Technical Manual",
        metadata={"mineru_type": "page_header"},
    )
    _replace_page_content(document, first_page, [first_header, table, promoted_caption])
    _replace_page_content(document, second_page, [second_header])
    document.pages = document.pages[:2]
    document.elements = [first_header, table, promoted_caption, second_header]
    document.sections = []
    document.relations = []

    relations = RelationBuilder(document).build_caption_relations()

    assert not any(item.source_id == promoted_caption.element_id for item in relations)


def test_conflicting_numbered_caption_rebinds_to_matching_same_page_table(
) -> None:
    document = _document(page_count=1)
    page = document.pages[0]
    table_four = _element(
        document,
        page,
        0,
        ElementType.TABLE,
        "table-four",
        html=(
            "<table><caption>Table 4 Earlier material</caption>"
            "<tr><th>Value</th></tr><tr><td>4</td></tr></table>"
        ),
        bbox=(100, 100, 900, 450),
    )
    caption = _element(
        document,
        page,
        1,
        ElementType.CAPTION,
        "table-five-caption",
        text="Table 5 Technical Reference data",
        bbox=(100, 480, 900, 520),
        metadata={"target_element_id": table_four.element_id},
    )
    table_five = _element(
        document,
        page,
        2,
        ElementType.TABLE,
        "table-five",
        html=(
            "<table><caption>Table 5 Technical Reference</caption>"
            "<tr><th>Value</th></tr><tr><td>5</td></tr></table>"
        ),
        bbox=(100, 550, 900, 1200),
    )
    _replace_page_content(document, page, [table_four, caption, table_five])
    document.pages = [document.pages[0]]
    document.elements = [table_four, caption, table_five]
    document.sections = []
    document.relations = []

    relation = RelationBuilder(document).build_caption_relations()[0]

    assert relation.source_id == caption.element_id
    assert relation.target_id == table_five.element_id
    assert relation.evidence[0].rule == "caption_label_table_target_repaired"


def test_form_style_numbered_table_title_rebinds_to_following_table() -> None:
    document = _document(page_count=1)
    page = document.pages[0]
    srm_table = _element(
        document,
        page,
        0,
        ElementType.TABLE,
        "srm-table",
        html=(
            "<table><tr><td>4. Service Component Reference Model (SRM) Table:</td></tr>"
            "<tr><td>Service Domain</td></tr></table>"
        ),
        bbox=(100, 100, 900, 450),
    )
    trm_caption = _element(
        document,
        page,
        1,
        ElementType.CAPTION,
        "trm-caption",
        text="5. Technical Reference Model (TRM) Table:",
        bbox=(100, 480, 900, 520),
        metadata={"target_element_id": srm_table.element_id},
    )
    trm_table = _element(
        document,
        page,
        2,
        ElementType.TABLE,
        "trm-table",
        html="<table><tr><th>Service Area</th><th>Standard</th></tr></table>",
        bbox=(100, 550, 900, 1200),
    )
    _replace_page_content(document, page, [srm_table, trm_caption, trm_table])
    document.elements = [srm_table, trm_caption, trm_table]

    relation = RelationBuilder(document).build_caption_relations()[0]

    assert relation.source_id == trm_caption.element_id
    assert relation.target_id == trm_table.element_id
    assert relation.evidence[0].rule == "caption_label_table_target_repaired"
