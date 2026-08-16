from pathlib import Path

from scripts.audit_representative_tables import (
    _enclosing_html_cell,
    _html_for_review,
    _html_rows_for_placeholder,
    _resolve_review_html_images,
    parse_table_html,
)


def test_embedded_image_mapping_returns_only_its_exact_cell() -> None:
    html = (
        "<table><tr>"
        '<td>A<img src="a.jpg"></td>'
        '<td>B<img src="b.jpg"></td>'
        '<td>C<img src="c.jpg"></td>'
        "</tr></table>"
    )
    match = html.index('src="b.jpg"')

    cell = _enclosing_html_cell(html, match, match + len('src="b.jpg"'))

    assert 'src="b.jpg"' in cell
    assert 'src="a.jpg"' not in cell
    assert 'src="c.jpg"' not in cell
    assert cell.startswith("<td>")
    assert cell.endswith("</td>")


def test_placeholder_rows_are_selected_by_label_not_all_placeholders() -> None:
    html = (
        "<table>"
        "<tr><td>&lt;image 1&gt; first</td></tr>"
        "<tr><td>&lt;image 2&gt; second</td></tr>"
        "<tr><td>&lt;image 1&gt; repeated</td></tr>"
        "</table>"
    )

    rows = _html_rows_for_placeholder(html, "image 1")

    assert len(rows) == 2
    assert all("image 1" in row for row in rows)
    assert all("image 2" not in row for row in rows)


def test_table_html_parser_preserves_spans_and_anchor_coordinates() -> None:
    html = """
    <table>
      <tr><th rowspan="2">Region</th><th colspan="2">Revenue</th></tr>
      <tr><th>2022</th><th>2023</th></tr>
      <tr><td>US</td><td>10</td><td>12</td></tr>
    </table>
    """

    cells, row_count, column_count, issues = parse_table_html(html)

    assert row_count == 3
    assert column_count == 3
    assert issues == []
    assert [
        (cell.row, cell.column, cell.rowspan, cell.colspan, cell.text)
        for cell in cells
    ] == [
        (0, 0, 2, 1, "Region"),
        (0, 1, 1, 2, "Revenue"),
        (1, 1, 1, 1, "2022"),
        (1, 2, 1, 1, "2023"),
        (2, 0, 1, 1, "US"),
        (2, 1, 1, 1, "10"),
        (2, 2, 1, 1, "12"),
    ]


def test_table_html_parser_reports_implicit_missing_cells() -> None:
    cells, row_count, column_count, issues = parse_table_html(
        "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td></tr></table>"
    )

    assert [(cell.row, cell.column, cell.text) for cell in cells] == [
        (0, 0, "A"),
        (0, 1, "B"),
        (1, 0, "C"),
    ]
    assert row_count == 2
    assert column_count == 2
    assert issues == ["ragged_row:1:missing=1"]


def test_table_html_parser_degrades_empty_html_without_raising() -> None:
    cells, row_count, column_count, issues = parse_table_html("")

    assert cells == []
    assert row_count == 0
    assert column_count == 0
    assert issues == ["no_rows"]


def test_review_html_replaces_unportable_embedded_images() -> None:
    rendered = _html_for_review(
        '<table><tr><td>A<img src="images/cell.jpg"/></td></tr></table>'
    )

    assert '<img src="images/cell.jpg"' not in rendered
    assert "[embedded image omitted: cell.jpg]" in rendered


def test_embedded_image_review_resolves_real_local_images(tmp_path: Path) -> None:
    document_dir = tmp_path / "softdoc" / "example"
    output_dir = tmp_path / "review"
    asset = document_dir / "assets" / "elements" / "cell.jpg"
    asset.parent.mkdir(parents=True)
    output_dir.mkdir()
    asset.write_bytes(b"image fixture")

    rendered, images = _resolve_review_html_images(
        '<table><tr><td><img src="assets/elements/cell.jpg"></td></tr></table>',
        document_dir,
        output_dir,
    )

    assert len(images) == 1
    assert images[0]["name"] == "cell.jpg"
    assert images[0]["uri"] in rendered
    assert (output_dir / images[0]["uri"]).resolve() == asset.resolve()
