from __future__ import annotations

from pathlib import Path

from softdoc.models import (
    Document,
    Element,
    ElementType,
    Page,
    Provenance,
    RelationStatus,
    RelationType,
)
from softdoc.pipeline import PassContext, TableFragmentReconciliationPass
from softdoc.relations import RelationBuilder
from softdoc.table_fragments import (
    FRAGMENT_METADATA_KEY,
    TableReconciliationStatus,
    reconcile_mineru_aggregate_tables,
)


def _provenance(locator: str, *, html: str = "") -> Provenance:
    return Provenance(
        provenance_id=f"prov:{locator}",
        adapter="mineru",
        source_path=Path("content_list_v2.json"),
        source_locator=locator,
        raw_payload={"type": "table", "content": {"html": html}},
    )


def _document(aggregate_html: str, local_htmls: list[str]) -> Document:
    document_id = "doc:tables"
    elements: list[Element] = []
    pages: list[Page] = []
    all_html = [aggregate_html, *local_htmls]
    for index, html in enumerate(all_html):
        page_number = index + 1
        element_id = f"table:{page_number}"
        metadata = (
            {}
            if index == 0
            else {
                "html_recovery": {
                    "source": "page_preproc_blocks",
                    "bbox_iou": 1.0,
                }
            }
        )
        element = Element(
            element_id=element_id,
            document_id=document_id,
            page_id=f"page:{page_number}",
            page_number=page_number,
            element_type=ElementType.TABLE,
            reading_order=0,
            html=html,
            provenance=_provenance(
                f"page[{index}].block[0]",
                html=aggregate_html if index == 0 else "",
            ),
            metadata=metadata,
        )
        page = Page(
            page_id=element.page_id,
            document_id=document_id,
            page_index=index,
            page_number=page_number,
            width=100.0,
            height=100.0,
            element_ids=[element_id],
            reading_order=[element_id],
            provenance=_provenance(f"page[{index}]"),
        )
        elements.append(element)
        pages.append(page)
    return Document(
        document_id=document_id,
        source_path=Path("fixture.pdf"),
        pages=pages,
        elements=elements,
        provenance=_provenance("document"),
    )


def _table(*rows: str) -> str:
    return "<table>" + "".join(f"<tr><td>{row}</td></tr>" for row in rows) + "</table>"


def test_reconciliation_splits_unique_suffix_and_confirms_relation() -> None:
    aggregate_html = _table("Header", "page-one", "page-two")
    document = _document(
        aggregate_html,
        [_table("page-two")],
    )

    result = reconcile_mineru_aggregate_tables(document)
    RelationBuilder(document).build_all()

    assert len(result.confirmed_decisions) == 1
    assert "page-two" not in (document.elements[0].html or "")
    assert document.elements[1].html == _table("page-two")
    assert (
        document.elements[0].provenance.raw_payload["content"]["html"]
        == aggregate_html
    )
    relation = next(
        relation
        for relation in document.relations
        if relation.relation_type == RelationType.CONTINUED_ON
    )
    assert relation.source_id == "table:1"
    assert relation.target_id == "table:2"
    assert relation.status == RelationStatus.CONFIRMED
    assert relation.confidence == 1.0


def test_confirmed_split_preserves_page_local_structure_and_original_html() -> None:
    aggregate_html = (
        '<table class="financial"><tr><th rowspan="2">Company</th>'
        '<th colspan="2">Revenue</th></tr>'
        "<tr><th>2022</th><th>2023</th></tr>"
        '<tr><td data-role="row-key">A Company</td><td>10</td><td>12</td></tr>'
        '<tr><td data-role="row-key">B Company</td><td>15</td><td>18</td></tr>'
        "</table>"
    )
    continuation_html = (
        '<table><tr><td data-role="row-key">B Company</td>'
        "<td>15</td><td>18</td></tr></table>"
    )
    document = _document(aggregate_html, [continuation_html])

    result = reconcile_mineru_aggregate_tables(document)
    RelationBuilder(document).build_all()

    assert len(result.confirmed_decisions) == 1
    source, continuation = document.elements
    assert 'rowspan="2"' in (source.html or "")
    assert 'colspan="2"' in (source.html or "")
    assert 'data-role="row-key">A Company' in (source.html or "")
    assert "B Company" not in (source.html or "")
    assert continuation.html == continuation_html
    assert 'data-role="row-key">B Company' in (continuation.html or "")
    assert source.provenance.raw_payload["content"]["html"] == aggregate_html
    relation = next(
        relation
        for relation in document.relations
        if relation.relation_type == RelationType.CONTINUED_ON
        and relation.status == RelationStatus.CONFIRMED
    )
    assert relation.source_id == source.element_id
    assert relation.target_id == continuation.element_id
    assert relation.evidence[0].data["aggregate_html_preserved_in_provenance"]


def test_reconciliation_ignores_repeated_page_header_rows() -> None:
    document = _document(
        _table("Title", "Columns", "A", "B", "C", "D"),
        [_table("Title", "Columns", "C", "D")],
    )

    decision = reconcile_mineru_aggregate_tables(document).confirmed_decisions[0]

    assert decision.assignments[1].repeated_header_rows_ignored == 2
    assert document.elements[0].html == _table("Title", "Columns", "A", "B")
    assert document.elements[1].html == _table("Title", "Columns", "C", "D")


def test_ambiguous_rows_leave_entire_group_unchanged() -> None:
    aggregate = _table("Header", "same", "same")
    document = _document(aggregate, [_table("same")])
    before = document.model_dump(mode="json")

    result = reconcile_mineru_aggregate_tables(document)

    assert result.decisions[0].status == TableReconciliationStatus.SKIPPED
    # Only the external audit metadata is added; no Element is touched.
    after_without_audit = document.model_copy(deep=True)
    after_without_audit.metadata.pop("cross_page_table_reconciliation")
    assert after_without_audit.model_dump(mode="json") == before
    assert FRAGMENT_METADATA_KEY not in document.elements[0].metadata


def test_skipped_reconciliation_does_not_create_confirmed_continuation() -> None:
    aggregate = _table("Header", "same", "same")
    document = _document(aggregate, [_table("same")])

    result = reconcile_mineru_aggregate_tables(document)
    RelationBuilder(document).build_all()

    assert result.decisions[0].status == TableReconciliationStatus.SKIPPED
    assert all(
        not (
            relation.relation_type == RelationType.CONTINUED_ON
            and relation.status == RelationStatus.CONFIRMED
        )
        for relation in document.relations
    )
    assert all(
        FRAGMENT_METADATA_KEY not in element.metadata for element in document.elements
    )
    assert document.elements[0].html == aggregate
    assert document.elements[0].provenance.raw_payload["content"]["html"] == aggregate


def test_rowspan_crossing_boundary_is_not_split() -> None:
    aggregate = (
        '<table><tr><td rowspan="2">shared</td><td>A</td></tr>'
        "<tr><td>B</td></tr></table>"
    )
    document = _document(aggregate, [_table("B")])
    before_html = document.elements[0].html

    result = reconcile_mineru_aggregate_tables(document)

    assert result.decisions[0].status == TableReconciliationStatus.SKIPPED
    assert result.decisions[0].reason == "rowspan_crosses_page_boundary"
    assert document.elements[0].html == before_html


def test_table_fragment_pass_is_idempotent(tmp_path: Path) -> None:
    document = _document(
        _table("Header", "page-one", "page-two"),
        [_table("page-two")],
    )
    context = PassContext(
        input_path=tmp_path,
        output_dir=tmp_path,
        available={"visual_assets_recovered"},
    )
    document_pass = TableFragmentReconciliationPass()

    first = document_pass.apply(document, context)
    after_first = document.model_dump(mode="json")
    second = document_pass.apply(document, context)

    assert first.changed
    assert second.skipped
    assert not second.changed
    assert document.model_dump(mode="json") == after_first
