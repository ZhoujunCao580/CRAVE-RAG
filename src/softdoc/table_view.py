"""Deterministic, parser-neutral table views derived from SoftDoc HTML.

Only real HTML ``td``/``th`` nodes are serialized as cells.  Coordinates
covered by rowspan/colspan are resolved through a runtime occupancy grid and
never become duplicate cells.
"""

from __future__ import annotations

from enum import Enum
from html.parser import HTMLParser
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from softdoc.ids import stable_digest
from softdoc.models import Element, ElementType, SoftDocModel


class TableVisualAssetSource(str, Enum):
    HTML_IMG_SRC = "html_img_src"


class TableViewIssueCode(str, Enum):
    NO_HTML = "no_html"
    NO_ROWS = "no_rows"
    NESTED_TABLE = "nested_table"
    INVALID_SPAN = "invalid_span"
    MISSING_INTERNAL_VISUAL = "missing_internal_visual"
    MISSING_OUTER_VISUAL = "missing_outer_visual"


class TableVisualAsset(SoftDocModel):
    visual_asset_id: str
    path: Path
    source: TableVisualAssetSource = TableVisualAssetSource.HTML_IMG_SRC


class TableCell(SoftDocModel):
    cell_id: str
    row: int = Field(ge=0)
    column: int = Field(ge=0)
    rowspan: int = Field(default=1, ge=1)
    colspan: int = Field(default=1, ge=1)
    text: str | None = None
    visual_asset_ids: list[str] = Field(default_factory=list)


class TableView(SoftDocModel):
    table_view_id: str
    document_id: str
    element_id: str
    page_id: str
    page_number: int = Field(ge=1)
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    cells: list[TableCell] = Field(default_factory=list)
    visual_assets: list[TableVisualAsset] = Field(default_factory=list)
    outer_visual_path: Path | None = None

    @model_validator(mode="after")
    def validate_grid(self) -> Self:
        if not self.cells and (self.row_count or self.column_count):
            raise ValueError("An empty TableView must have a 0 x 0 grid")
        cell_ids = [cell.cell_id for cell in self.cells]
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("TableView cell IDs must be unique")
        asset_ids = [asset.visual_asset_id for asset in self.visual_assets]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("TableView visual asset IDs must be unique")
        known_assets = set(asset_ids)
        occupied: dict[tuple[int, int], str] = {}
        for cell in self.cells:
            if cell.row + cell.rowspan > self.row_count:
                raise ValueError("TableView cell rowspan exceeds row_count")
            if cell.column + cell.colspan > self.column_count:
                raise ValueError("TableView cell colspan exceeds column_count")
            missing_assets = set(cell.visual_asset_ids) - known_assets
            if missing_assets:
                raise ValueError(
                    f"TableView cell refers to unknown visual assets: {missing_assets}"
                )
            for row in range(cell.row, cell.row + cell.rowspan):
                for column in range(cell.column, cell.column + cell.colspan):
                    coordinate = (row, column)
                    if coordinate in occupied:
                        raise ValueError(
                            "TableView anchor spans overlap at "
                            f"r{row}c{column}: {occupied[coordinate]} and {cell.cell_id}"
                        )
                    occupied[coordinate] = cell.cell_id
        return self


class TableViewIssue(SoftDocModel):
    issue_code: TableViewIssueCode
    element_id: str
    detail: str


class TableViewBuildResult(SoftDocModel):
    view: TableView
    issues: list[TableViewIssue] = Field(default_factory=list)


class _SourceCell:
    def __init__(self, *, rowspan: int, colspan: int) -> None:
        self.rowspan = rowspan
        self.colspan = colspan
        self.text_parts: list[str] = []
        self.image_sources: list[str] = []


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[_SourceCell]] = []
        self.invalid_spans: list[str] = []
        self.nested_tables = 0
        self._table_depth = 0
        self._row: list[_SourceCell] | None = None
        self._cell: _SourceCell | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        tag = tag.lower()
        attributes = dict(attrs)
        if tag == "table":
            self._table_depth += 1
            if self._table_depth > 1:
                self.nested_tables += 1
            return
        if self._table_depth != 1:
            return
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            rowspan = self._span(attributes.get("rowspan"), "rowspan")
            colspan = self._span(attributes.get("colspan"), "colspan")
            self._cell = _SourceCell(rowspan=rowspan, colspan=colspan)
        elif tag == "img" and self._cell is not None:
            source = (attributes.get("src") or "").strip()
            if source:
                self._cell.image_sources.append(source)
        elif tag == "br" and self._cell is not None:
            self._cell.text_parts.append(" ")

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if (
            tag in {"td", "th"}
            and self._table_depth == 1
            and self._cell is not None
            and self._row is not None
        ):
            self._row.append(self._cell)
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table_depth == 1:
            self.rows.append(self._row)
            self._row = None
        elif tag == "table":
            self._table_depth = max(0, self._table_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._table_depth == 1 and self._cell is not None:
            self._cell.text_parts.append(data)

    def _span(self, raw_value: str | None, name: str) -> int:
        if raw_value is None:
            return 1
        try:
            value = int(raw_value)
        except ValueError:
            self.invalid_spans.append(f"{name}={raw_value!r}")
            return 1
        if value < 1:
            self.invalid_spans.append(f"{name}={raw_value!r}")
            return 1
        return value


class TableMaterializer:
    """Build TableView and resolve span-covered coordinates at runtime."""

    def materialize(
        self,
        element: Element,
        *,
        document_root: Path,
    ) -> TableViewBuildResult:
        if element.element_type != ElementType.TABLE:
            raise ValueError("TableMaterializer only accepts table Elements")
        document_root = Path(document_root)
        issues: list[TableViewIssue] = []
        outer_visual_path = self._existing_outer_visual(
            element, document_root=document_root, issues=issues
        )
        raw_html = (element.html or "").strip()
        if not raw_html:
            issues.append(
                TableViewIssue(
                    issue_code=TableViewIssueCode.NO_HTML,
                    element_id=element.element_id,
                    detail="The Table Element has no HTML to materialize.",
                )
            )
            return TableViewBuildResult(
                view=self._view(
                    element,
                    row_count=0,
                    column_count=0,
                    cells=[],
                    visual_assets=[],
                    outer_visual_path=outer_visual_path,
                ),
                issues=issues,
            )

        parser = _TableHTMLParser()
        parser.feed(raw_html)
        parser.close()
        if parser.nested_tables:
            issues.append(
                TableViewIssue(
                    issue_code=TableViewIssueCode.NESTED_TABLE,
                    element_id=element.element_id,
                    detail=f"Ignored {parser.nested_tables} nested table(s).",
                )
            )
        for detail in parser.invalid_spans:
            issues.append(
                TableViewIssue(
                    issue_code=TableViewIssueCode.INVALID_SPAN,
                    element_id=element.element_id,
                    detail=detail,
                )
            )
        if not parser.rows:
            issues.append(
                TableViewIssue(
                    issue_code=TableViewIssueCode.NO_ROWS,
                    element_id=element.element_id,
                    detail="The HTML contains no top-level table rows.",
                )
            )

        occupied: dict[tuple[int, int], TableCell] = {}
        cells: list[TableCell] = []
        assets_by_source: dict[str, TableVisualAsset] = {}
        for row_index, source_row in enumerate(parser.rows):
            column_index = 0
            for source_cell in source_row:
                while any(
                    (row_index, column) in occupied
                    for column in range(
                        column_index, column_index + source_cell.colspan
                    )
                ):
                    column_index += 1
                visual_asset_ids: list[str] = []
                for source in source_cell.image_sources:
                    normalized_source = source.replace("\\", "/")
                    source_path = Path(normalized_source)
                    resolved = (
                        source_path
                        if source_path.is_absolute()
                        else document_root / source_path
                    )
                    if not resolved.is_file():
                        issues.append(
                            TableViewIssue(
                                issue_code=(
                                    TableViewIssueCode.MISSING_INTERNAL_VISUAL
                                ),
                                element_id=element.element_id,
                                detail=f"HTML img src is unavailable: {source}",
                            )
                        )
                        continue
                    asset = assets_by_source.get(normalized_source)
                    if asset is None:
                        asset = TableVisualAsset(
                            visual_asset_id=(
                                "visual:"
                                + stable_digest(
                                    element.element_id,
                                    normalized_source,
                                    length=12,
                                )
                            ),
                            path=source_path,
                        )
                        assets_by_source[normalized_source] = asset
                    if asset.visual_asset_id not in visual_asset_ids:
                        visual_asset_ids.append(asset.visual_asset_id)
                text = " ".join("".join(source_cell.text_parts).split()) or None
                cell = TableCell(
                    cell_id=f"{element.element_id}#r{row_index}c{column_index}",
                    row=row_index,
                    column=column_index,
                    rowspan=source_cell.rowspan,
                    colspan=source_cell.colspan,
                    text=text,
                    visual_asset_ids=visual_asset_ids,
                )
                cells.append(cell)
                for row in range(row_index, row_index + source_cell.rowspan):
                    for column in range(
                        column_index, column_index + source_cell.colspan
                    ):
                        occupied[(row, column)] = cell
                column_index += source_cell.colspan

        row_count = max((row for row, _ in occupied), default=-1) + 1
        column_count = max((column for _, column in occupied), default=-1) + 1
        return TableViewBuildResult(
            view=self._view(
                element,
                row_count=row_count,
                column_count=column_count,
                cells=cells,
                visual_assets=list(assets_by_source.values()),
                outer_visual_path=outer_visual_path,
            ),
            issues=issues,
        )

    def get_cell_at(
        self,
        view: TableView,
        row: int,
        column: int,
    ) -> TableCell | None:
        if row < 0 or column < 0:
            return None
        return self._occupancy_grid(view).get((row, column))

    @staticmethod
    def _occupancy_grid(view: TableView) -> dict[tuple[int, int], TableCell]:
        occupied: dict[tuple[int, int], TableCell] = {}
        for cell in view.cells:
            for row in range(cell.row, cell.row + cell.rowspan):
                for column in range(cell.column, cell.column + cell.colspan):
                    occupied[(row, column)] = cell
        return occupied

    @staticmethod
    def _view(
        element: Element,
        *,
        row_count: int,
        column_count: int,
        cells: list[TableCell],
        visual_assets: list[TableVisualAsset],
        outer_visual_path: Path | None,
    ) -> TableView:
        return TableView(
            table_view_id=f"table-view:{stable_digest(element.element_id, length=16)}",
            document_id=element.document_id,
            element_id=element.element_id,
            page_id=element.page_id,
            page_number=element.page_number,
            row_count=row_count,
            column_count=column_count,
            cells=cells,
            visual_assets=visual_assets,
            outer_visual_path=outer_visual_path,
        )

    @staticmethod
    def _existing_outer_visual(
        element: Element,
        *,
        document_root: Path,
        issues: list[TableViewIssue],
    ) -> Path | None:
        visual_path = element.visual_asset_path
        if visual_path is None:
            issues.append(
                TableViewIssue(
                    issue_code=TableViewIssueCode.MISSING_OUTER_VISUAL,
                    element_id=element.element_id,
                    detail="The Table Element has no outer visual asset.",
                )
            )
            return None
        resolved = visual_path if visual_path.is_absolute() else document_root / visual_path
        if not resolved.is_file():
            issues.append(
                TableViewIssue(
                    issue_code=TableViewIssueCode.MISSING_OUTER_VISUAL,
                    element_id=element.element_id,
                    detail=f"Outer visual asset is unavailable: {visual_path}",
                )
            )
            return None
        return visual_path
