from pathlib import Path

import pytest

from softdoc.models import Element, ElementType, Provenance
from softdoc.table_view import (
    TableMaterializer,
    TableView,
    TableViewIssueCode,
)


def _element(
    html: str | None,
    *,
    element_type: ElementType = ElementType.TABLE,
    image_path: Path | None = None,
) -> Element:
    return Element(
        element_id="doc:test:page:0000:element:0000:table:main",
        document_id="doc:test",
        page_id="doc:test:page:0000",
        page_number=1,
        element_type=element_type,
        reading_order=0,
        html=html,
        image_path=image_path,
        provenance=Provenance(
            provenance_id="prov:test",
            adapter="fixture",
            source_path=Path("fixture.json"),
            source_locator="block[0]",
        ),
    )


def test_rowspan_serializes_only_anchor_cells_and_runtime_lookup_resolves_span(
    tmp_path: Path,
) -> None:
    html = (
        "<table>"
        '<tr><td rowspan="2">Asia</td><td>2022</td></tr>'
        "<tr><td>2023</td></tr>"
        "</table>"
    )
    materializer = TableMaterializer()

    result = materializer.materialize(_element(html), document_root=tmp_path)

    assert [(cell.row, cell.column, cell.rowspan, cell.text) for cell in result.view.cells] == [
        (0, 0, 2, "Asia"),
        (0, 1, 1, "2022"),
        (1, 1, 1, "2023"),
    ]
    assert not any(cell.row == 1 and cell.column == 0 for cell in result.view.cells)
    assert materializer.get_cell_at(result.view, 1, 0) is result.view.cells[0]
    assert materializer.get_cell_at(result.view, 1, 1) is result.view.cells[2]


def test_colspan_occupies_coordinates_without_duplicate_cells(tmp_path: Path) -> None:
    result = TableMaterializer().materialize(
        _element(
            "<table><tr><td colspan='2'>Revenue</td></tr>"
            "<tr><td>2022</td><td>2023</td></tr></table>"
        ),
        document_root=tmp_path,
    )

    assert result.view.row_count == 2
    assert result.view.column_count == 2
    assert len(result.view.cells) == 3
    assert TableMaterializer().get_cell_at(result.view, 0, 1).text == "Revenue"


def test_colspan_skips_complete_range_occupied_by_prior_rowspan(tmp_path: Path) -> None:
    result = TableMaterializer().materialize(
        _element(
            "<table>"
            "<tr><td>A</td><td rowspan='2'>B</td></tr>"
            "<tr><td colspan='2'>C</td></tr>"
            "</table>"
        ),
        document_root=tmp_path,
    )

    assert [(cell.text, cell.row, cell.column, cell.colspan) for cell in result.view.cells] == [
        ("A", 0, 0, 1),
        ("B", 0, 1, 1),
        ("C", 1, 2, 2),
    ]
    assert TableMaterializer().get_cell_at(result.view, 1, 1).text == "B"
    assert TableMaterializer().get_cell_at(result.view, 1, 2).text == "C"


def test_only_existing_html_images_become_visual_assets(tmp_path: Path) -> None:
    asset = tmp_path / "assets" / "elements" / "existing.jpg"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"fixture")
    html = (
        "<table><tr>"
        '<td><img src="assets/elements/existing.jpg"></td>'
        '<td><img src="assets/elements/missing.jpg"></td>'
        "</tr></table>"
    )

    result = TableMaterializer().materialize(
        _element(html),
        document_root=tmp_path,
    )

    assert len(result.view.visual_assets) == 1
    assert result.view.visual_assets[0].path == Path("assets/elements/existing.jpg")
    assert result.view.cells[0].visual_asset_ids == [
        result.view.visual_assets[0].visual_asset_id
    ]
    assert result.view.cells[1].visual_asset_ids == []
    assert TableViewIssueCode.MISSING_INTERNAL_VISUAL in {
        issue.issue_code for issue in result.issues
    }


def test_empty_html_produces_empty_view_without_failing(tmp_path: Path) -> None:
    result = TableMaterializer().materialize(
        _element(None, image_path=Path("missing.jpg")),
        document_root=tmp_path,
    )

    assert result.view.row_count == 0
    assert result.view.column_count == 0
    assert result.view.cells == []
    assert TableViewIssueCode.NO_HTML in {issue.issue_code for issue in result.issues}


def test_table_view_is_stable_and_round_trips(tmp_path: Path) -> None:
    html = "<table><tr><td>A</td><td></td></tr></table>"
    materializer = TableMaterializer()

    first = materializer.materialize(_element(html), document_root=tmp_path)
    second = materializer.materialize(_element(html), document_root=tmp_path)
    restored = TableView.model_validate_json(first.view.model_dump_json())

    assert first == second
    assert restored == first.view
    assert first.view.cells[1].text is None
    assert first.view.cells[1].visual_asset_ids == []


def test_materializer_rejects_non_table_element(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only accepts table"):
        TableMaterializer().materialize(
            _element("<table></table>", element_type=ElementType.PARAGRAPH),
            document_root=tmp_path,
        )
