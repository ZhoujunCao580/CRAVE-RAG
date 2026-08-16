"""Materialize and audit TableViews for a retained SoftDoc corpus."""

from __future__ import annotations

import argparse
import html as html_module
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from audit_representative_tables import parse_table_html
from softdoc.models import Element, ElementType
from softdoc.serialization import load_document
from softdoc.table_view import (
    TableMaterializer,
    TableView,
    TableViewBuildResult,
)


def materialize_corpus(corpus_dir: Path, output_dir: Path) -> dict[str, Any]:
    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir)
    softdoc_root = corpus_dir / "softdoc"
    if not softdoc_root.is_dir():
        raise FileNotFoundError(f"SoftDoc corpus is unavailable: {softdoc_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    review_dir = output_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    materializer = TableMaterializer()
    documents: list[dict[str, Any]] = []
    issue_counts: Counter[str] = Counter()
    total_tables = 0
    total_cells = 0
    total_visual_assets = 0
    automatic_failures: list[dict[str, str]] = []
    category_counts: Counter[str] = Counter()

    for document_root in sorted(path for path in softdoc_root.iterdir() if path.is_dir()):
        document = load_document(document_root)
        tables = [
            element
            for element in document.elements
            if element.element_type == ElementType.TABLE
        ]
        results = [
            materializer.materialize(element, document_root=document_root)
            for element in tables
        ]
        checks = [
            _automated_check(
                element=element,
                result=result,
                document_root=document_root,
                materializer=materializer,
            )
            for element, result in zip(tables, results, strict=True)
        ]
        for result in results:
            issue_counts.update(issue.issue_code.value for issue in result.issues)
            total_cells += len(result.view.cells)
            total_visual_assets += len(result.view.visual_assets)
        for element, result in zip(tables, results, strict=True):
            category_counts[_table_category(element, result)] += 1
        for check in checks:
            automatic_failures.extend(check["failures"])

        document_output = output_dir / "documents" / document_root.name
        document_output.mkdir(parents=True, exist_ok=True)
        _write_jsonl(
            document_output / "table_views.jsonl",
            [result.view.model_dump(mode="json") for result in results],
        )
        _write_jsonl(
            document_output / "table_view_issues.jsonl",
            [
                issue.model_dump(mode="json")
                for result in results
                for issue in result.issues
            ],
        )
        _write_json(document_output / "automated_checks.json", checks)
        review_path = review_dir / "documents" / f"{document_root.name}.html"
        _write_document_review(
            review_path=review_path,
            document_name=document_root.name,
            document_root=document_root,
            tables=tables,
            results=results,
        )
        documents.append(
            {
                "document": document_root.name,
                "table_count": len(tables),
                "cell_count": sum(len(result.view.cells) for result in results),
                "visual_asset_count": sum(
                    len(result.view.visual_assets) for result in results
                ),
                "issue_count": sum(len(result.issues) for result in results),
                "automatic_failure_count": sum(
                    len(check["failures"]) for check in checks
                ),
                "review_path": _relative_uri(review_path, review_dir),
            }
        )
        total_tables += len(tables)

    summary = {
        "document_count": len(documents),
        "table_count": total_tables,
        "table_view_count": total_tables,
        "cell_count": total_cells,
        "resolved_internal_visual_asset_count": total_visual_assets,
        "categories": dict(sorted(category_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "automatic_check_failure_count": len(automatic_failures),
        "automatic_checks": [
            "all Table Elements produce one TableView",
            "independent HTML parser agrees on anchor cells and coordinates",
            "all serialized visual assets exist",
            "rowspan/colspan runtime occupancy resolves to anchor cells",
            "materialization is deterministic",
            "TableView JSON round-trip is lossless",
        ],
        "automatic_check_limit": (
            "These checks do not prove that MinerU recognized every visual row, "
            "column, or value correctly; that requires visual review."
        ),
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "documents.json", documents)
    _write_json(output_dir / "automatic_failures.json", automatic_failures)
    _write_index(review_dir / "index.html", summary, documents)
    return summary


def _automated_check(
    *,
    element: Element,
    result: TableViewBuildResult,
    document_root: Path,
    materializer: TableMaterializer,
) -> dict[str, Any]:
    failures: list[dict[str, str]] = []

    def fail(code: str, detail: str) -> None:
        failures.append(
            {"element_id": element.element_id, "check": code, "detail": detail}
        )

    audit_cells, audit_rows, audit_columns, audit_issues = parse_table_html(
        element.html or ""
    )
    expected = [
        (cell.row, cell.column, cell.rowspan, cell.colspan, cell.text or None)
        for cell in audit_cells
    ]
    actual = [
        (cell.row, cell.column, cell.rowspan, cell.colspan, cell.text)
        for cell in result.view.cells
    ]
    if expected != actual:
        fail("independent_anchor_cells", "Core and audit HTML parsers disagree.")
    if (audit_rows, audit_columns) != (
        result.view.row_count,
        result.view.column_count,
    ):
        fail(
            "independent_grid_size",
            f"audit={audit_rows}x{audit_columns}, "
            f"view={result.view.row_count}x{result.view.column_count}",
        )
    for asset in result.view.visual_assets:
        resolved = asset.path if asset.path.is_absolute() else document_root / asset.path
        if not resolved.is_file():
            fail("visual_asset_exists", f"Missing {asset.path}")
    for cell in result.view.cells:
        for row in range(cell.row, cell.row + cell.rowspan):
            for column in range(cell.column, cell.column + cell.colspan):
                resolved = materializer.get_cell_at(result.view, row, column)
                if resolved is None or resolved.cell_id != cell.cell_id:
                    fail(
                        "span_occupancy",
                        f"r{row}c{column} does not resolve to {cell.cell_id}",
                    )
    repeated = materializer.materialize(element, document_root=document_root)
    if repeated != result:
        fail("deterministic_rebuild", "Repeated materialization changed the result.")
    restored = TableView.model_validate_json(result.view.model_dump_json())
    if restored != result.view:
        fail("json_round_trip", "TableView changed after JSON round-trip.")
    return {
        "element_id": element.element_id,
        "audit_parser_issues": audit_issues,
        "passed": not failures,
        "failures": failures,
    }


def _table_category(element: Element, result: TableViewBuildResult) -> str:
    if not (element.html or "").strip():
        return "no_html"
    if result.view.visual_assets:
        return "resolved_internal_images"
    if re.search(r"(?:&lt;|<)image\s*\d+(?:&gt;|>)", element.html or "", re.I):
        return "unresolved_image_placeholder"
    return "structured_text_only"


def _write_document_review(
    *,
    review_path: Path,
    document_name: str,
    document_root: Path,
    tables: list[Element],
    results: list[TableViewBuildResult],
) -> None:
    review_path.parent.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    for index, (element, result) in enumerate(zip(tables, results, strict=True), start=1):
        outer = _asset_html(
            document_root=document_root,
            asset_path=result.view.outer_visual_path,
            review_path=review_path,
            alt="outer Table crop",
        )
        original = _resolved_source_html(
            element.html or "",
            document_root=document_root,
            review_path=review_path,
        )
        reconstructed = _table_view_html(
            result.view,
            document_root=document_root,
            review_path=review_path,
        )
        issue_text = ", ".join(issue.issue_code.value for issue in result.issues) or "none"
        cards.append(
            f'''<article class="card">
<h2>{index}. PDF page {element.page_number} | {result.view.row_count} x {result.view.column_count}</h2>
<p><code>{html_module.escape(element.element_id)}</code></p>
<p>anchor cells: {len(result.view.cells)} | resolved internal images: {len(result.view.visual_assets)} | issues: {html_module.escape(issue_text)}</p>
<div class="views">
  <section><h3>1. Actual outer Element crop</h3>{outer}</section>
  <section><h3>2. Original MinerU HTML</h3><div class="table-wrap">{original or '<em>No HTML</em>'}</div></section>
  <section><h3>3. Materialized TableView</h3><div class="table-wrap">{reconstructed or '<em>Empty TableView</em>'}</div></section>
</div>
</article>'''
        )
    page = f'''<!doctype html><html><head><meta charset="utf-8"><title>{html_module.escape(document_name)} TableViews</title>
<style>{_review_css()}</style></head><body>
<p><a href="../index.html">Back to corpus index</a></p>
<h1>{html_module.escape(document_name)}</h1>
<p>{len(tables)} Table Elements. This page supports visual review; it is not itself a human verdict.</p>
{''.join(cards)}
</body></html>'''
    review_path.write_text(page, encoding="utf-8")


def _table_view_html(
    view: TableView,
    *,
    document_root: Path,
    review_path: Path,
) -> str:
    if not view.cells:
        return ""
    assets = {asset.visual_asset_id: asset for asset in view.visual_assets}
    cells_by_row: dict[int, list[Any]] = {}
    for cell in view.cells:
        cells_by_row.setdefault(cell.row, []).append(cell)
    rows: list[str] = []
    for row_index in range(view.row_count):
        rendered_cells: list[str] = []
        for cell in sorted(cells_by_row.get(row_index, []), key=lambda item: item.column):
            content = html_module.escape(cell.text or "")
            for asset_id in cell.visual_asset_ids:
                asset = assets[asset_id]
                content += _asset_html(
                    document_root=document_root,
                    asset_path=asset.path,
                    review_path=review_path,
                    alt=asset_id,
                )
            rendered_cells.append(
                '<td rowspan="{rowspan}" colspan="{colspan}" '
                'data-coordinate="r{row}c{column}">'
                '<small>r{row}c{column}</small>{content}</td>'.format(
                    rowspan=cell.rowspan,
                    colspan=cell.colspan,
                    row=cell.row,
                    column=cell.column,
                    content=content,
                )
            )
        rows.append("<tr>" + "".join(rendered_cells) + "</tr>")
    return "<table>" + "".join(rows) + "</table>"


def _resolved_source_html(
    raw_html: str,
    *,
    document_root: Path,
    review_path: Path,
) -> str:
    pattern = re.compile(
        r"(?P<prefix><img\b[^>]*?\bsrc\s*=\s*)(?P<quote>['\"])(?P<source>.*?)(?P=quote)",
        re.IGNORECASE,
    )

    def replace(match: re.Match[str]) -> str:
        source = Path(match.group("source").replace("\\", "/"))
        resolved = source if source.is_absolute() else document_root / source
        if not resolved.is_file():
            return '<span class="missing">[missing img src]</span>'
        uri = _relative_uri(resolved, review_path.parent)
        return f'{match.group("prefix")}"{uri}"'

    return pattern.sub(replace, raw_html)


def _asset_html(
    *,
    document_root: Path,
    asset_path: Path | None,
    review_path: Path,
    alt: str,
) -> str:
    if asset_path is None:
        return '<em class="missing">No visual asset</em>'
    resolved = asset_path if asset_path.is_absolute() else document_root / asset_path
    if not resolved.is_file():
        return '<em class="missing">Missing visual asset</em>'
    return (
        f'<img src="{_relative_uri(resolved, review_path.parent)}" '
        f'alt="{html_module.escape(alt)}">'
    )


def _write_index(
    path: Path,
    summary: dict[str, Any],
    documents: list[dict[str, Any]],
) -> None:
    rows = "".join(
        '<tr><td><a href="{path}">{name}</a></td><td>{tables}</td>'
        '<td>{cells}</td><td>{visuals}</td><td>{issues}</td><td>{failures}</td></tr>'.format(
            path=html_module.escape(item["review_path"]),
            name=html_module.escape(item["document"]),
            tables=item["table_count"],
            cells=item["cell_count"],
            visuals=item["visual_asset_count"],
            issues=item["issue_count"],
            failures=item["automatic_failure_count"],
        )
        for item in documents
    )
    page = f'''<!doctype html><html><head><meta charset="utf-8"><title>Representative-28 TableViews</title>
<style>{_review_css()}</style></head><body>
<h1>Representative-28 TableView Review</h1>
<pre>{html_module.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>
<table><tr><th>Document</th><th>Tables</th><th>Cells</th><th>Internal images</th><th>Issues</th><th>Automatic failures</th></tr>{rows}</table>
</body></html>'''
    path.write_text(page, encoding="utf-8")


def _review_css() -> str:
    return """
body{font-family:Segoe UI,Arial,sans-serif;background:#111827;color:#e5e7eb;margin:20px}
a{color:#93c5fd}.card{background:#1f2937;padding:18px;margin:18px 0;border-radius:12px}
.views{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
section{background:#fff;color:#111;padding:10px;overflow:auto;min-height:180px}
img{max-width:100%;max-height:700px;object-fit:contain;display:block;margin:4px auto}
.table-wrap table,body>table{border-collapse:collapse;width:100%}
.table-wrap td,.table-wrap th,body>table td,body>table th{border:1px solid #64748b;padding:5px;vertical-align:top}
.table-wrap small{display:block;color:#b91c1c;font:11px monospace}.table-wrap img{max-width:220px;max-height:160px}
.missing{color:#b91c1c;font-weight:700}code,pre{white-space:pre-wrap;word-break:break-all}
@media(max-width:1300px){.views{grid-template-columns:1fr}}
"""


def _relative_uri(path: Path, base_dir: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base_dir.resolve())).as_posix()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/processed/representative_28"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/representative_28/table_views"),
    )
    args = parser.parse_args()
    print(json.dumps(materialize_corpus(args.corpus, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
