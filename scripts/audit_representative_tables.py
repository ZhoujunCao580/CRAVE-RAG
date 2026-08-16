"""Audit table extraction for the retained representative PDF corpus.

This is an offline diagnostic utility.  It does not mutate SoftDoc documents
or implement the future online TableReader/TableView API.
"""

from __future__ import annotations

import argparse
import html as html_module
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
from PIL import Image
from PIL import ImageDraw, ImageFont

from softdoc.models import ElementType
from softdoc.serialization import load_document


# Manually reviewed MinerU type-error candidates from representative-28.
# This list is audit evidence, not a parser rule and never mutates SoftDoc.
TABLE_TYPE_ERROR_REVIEW: dict[str, tuple[str, str]] = {
    "doc:0e94b4197b10096b1f4c699701570fbf:26022465:page:0008:element:0002:table:main": (
        "figure",
        "Historical route/profile graphic; the outer visual is not a row-column table.",
    ),
    "doc:2311.16502v3:6f18f973:page:0001:element:0000:table:main": (
        "figure",
        "Composite benchmark overview panel containing text and several embedded visuals.",
    ),
    "doc:2311.16502v3:6f18f973:page:0026:element:0000:table:main": (
        "figure",
        "Multimodal QA example panel; any embedded table is only a subregion.",
    ),
    "doc:2311.16502v3:6f18f973:page:0046:element:0000:table:main": (
        "figure",
        "Multimodal QA example panel rather than a single table.",
    ),
    "doc:2311.16502v3:6f18f973:page:0047:element:0000:table:main": (
        "figure",
        "Multimodal QA example panel rather than a single table.",
    ),
    "doc:2311.16502v3:6f18f973:page:0051:element:0000:table:main": (
        "figure",
        "Multimodal QA example panel rather than a single table.",
    ),
    "doc:2311.16502v3:6f18f973:page:0056:element:0000:table:main": (
        "figure",
        "Multimodal QA example panel rather than a single table.",
    ),
    "doc:2311.16502v3:6f18f973:page:0061:element:0000:table:main": (
        "figure",
        "Multimodal QA example panel rather than a single table.",
    ),
    "doc:2311.16502v3:6f18f973:page:0067:element:0000:table:main": (
        "figure",
        "Multimodal QA example panel rather than a single table.",
    ),
    "doc:2311.16502v3:6f18f973:page:0071:element:0000:table:main": (
        "figure",
        "Multimodal QA example panel rather than a single table.",
    ),
    "doc:2311.16502v3:6f18f973:page:0074:element:0000:table:main": (
        "figure",
        "Multimodal QA example panel rather than a single table.",
    ),
    "doc:2311.16502v3:6f18f973:page:0077:element:0000:table:main": (
        "figure",
        "Full QA example with an embedded table; HTML represents only part of the outer visual.",
    ),
    "doc:2311.16502v3:6f18f973:page:0078:element:0000:table:main": (
        "figure",
        "Full QA example with an embedded table; HTML represents only part of the outer visual.",
    ),
    "doc:2311.16502v3:6f18f973:page:0079:element:0000:table:main": (
        "figure",
        "Multimodal QA example panel rather than a single table.",
    ),
    "doc:2311.16502v3:6f18f973:page:0081:element:0000:table:main": (
        "figure",
        "Multimodal QA example panel rather than a single table.",
    ),
    "doc:2311.16502v3:6f18f973:page:0098:element:0000:table:main": (
        "figure",
        "Multimodal QA example panel rather than a single table.",
    ),
    "doc:2311.16502v3:6f18f973:page:0100:element:0000:table:main": (
        "figure",
        "Multimodal QA example panel rather than a single table.",
    ),
    "doc:2311.16502v3:6f18f973:page:0101:element:0000:table:main": (
        "figure",
        "Multimodal QA example panel rather than a single table.",
    ),
    "doc:91521110100m_4k_uhd_display_user_manual_v1.1:f6c0957e:page:0011:element:0000:table:main": (
        "figure",
        "OSD menu screenshot; visual layout is a UI, not a table.",
    ),
    "doc:91521110100m_4k_uhd_display_user_manual_v1.1:f6c0957e:page:0016:element:0002:table:main": (
        "figure",
        "OSD information screenshot; visual layout is a UI, not a table.",
    ),
    "doc:bigdatatrends-120723191058-phpapp02_95:fd254a4c:page:0028:element:0003:table:main": (
        "code",
        "Log screenshot with repeated lines, not row-column tabular data.",
    ),
    "doc:disciplined-agile-business-analysis-160218012713_95:6c5549a3:page:0007:element:0001:table:main": (
        "figure",
        "Process/stack diagram whose boxes were mistaken for table cells.",
    ),
}


# Review-only regions for unresolved ``<image N>`` placeholders. MinerU did not
# emit an inner-image bbox for these objects, so these coordinates were checked
# against the exact outer Element crop by a human. They are deliberately kept
# in this audit script and never enter SoftDoc or the online pipeline.
# Coordinates are normalized within the outer Element crop.
UNRESOLVED_PLACEHOLDER_REVIEW_REGIONS: dict[
    str, tuple[tuple[str, tuple[float, float, float, float]], ...]
] = {
    "doc:2311.16502v3:6f18f973:page:0026:element:0000:table:main": (
        ("image 1", (0.010, 0.270, 0.395, 0.625)),
    ),
    "doc:2311.16502v3:6f18f973:page:0046:element:0000:table:main": (
        ("image 1", (0.018, 0.225, 0.355, 0.500)),
    ),
    "doc:2311.16502v3:6f18f973:page:0047:element:0000:table:main": (
        ("image 1", (0.018, 0.215, 0.355, 0.435)),
    ),
    "doc:2311.16502v3:6f18f973:page:0051:element:0000:table:main": (
        ("image 1", (0.018, 0.220, 0.880, 0.400)),
    ),
    "doc:2311.16502v3:6f18f973:page:0061:element:0000:table:main": (
        ("image 1", (0.015, 0.280, 0.355, 0.490)),
    ),
    "doc:2311.16502v3:6f18f973:page:0071:element:0000:table:main": (
        ("image 1", (0.015, 0.268, 0.285, 0.485)),
    ),
    "doc:2311.16502v3:6f18f973:page:0074:element:0000:table:main": (
        ("image 1", (0.008, 0.212, 0.425, 0.360)),
        ("image 2", (0.425, 0.212, 0.750, 0.360)),
    ),
    "doc:2311.16502v3:6f18f973:page:0078:element:0000:table:main": (
        ("image 1", (0.010, 0.260, 0.805, 0.425)),
    ),
    "doc:2311.16502v3:6f18f973:page:0104:element:0000:table:main": (
        ("image 1", (0.015, 0.330, 0.470, 0.615)),
    ),
}


@dataclass(frozen=True)
class ParsedCell:
    row: int
    column: int
    rowspan: int
    colspan: int
    text: str


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, int, int]]] = []
        self._row: list[tuple[str, int, int]] | None = None
        self._cell_parts: list[str] | None = None
        self._rowspan = 1
        self._colspan = 1
        self.table_depth = 0
        self.nested_tables = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attributes = dict(attrs)
        if tag == "table":
            self.table_depth += 1
            if self.table_depth > 1:
                self.nested_tables += 1
        elif tag == "tr" and self.table_depth == 1:
            self._row = []
        elif tag in {"td", "th"} and self.table_depth == 1 and self._row is not None:
            self._cell_parts = []
            self._rowspan = _positive_int(attributes.get("rowspan"), default=1)
            self._colspan = _positive_int(attributes.get("colspan"), default=1)
        elif tag == "br" and self._cell_parts is not None:
            self._cell_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._cell_parts is not None and self._row is not None:
            text = " ".join("".join(self._cell_parts).split())
            self._row.append((text, self._rowspan, self._colspan))
            self._cell_parts = None
        elif tag == "tr" and self._row is not None and self.table_depth == 1:
            self.rows.append(self._row)
            self._row = None
        elif tag == "table":
            self.table_depth = max(0, self.table_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)


def parse_table_html(source: str) -> tuple[list[ParsedCell], int, int, list[str]]:
    parser = _TableHTMLParser()
    issues: list[str] = []
    try:
        parser.feed(source)
        parser.close()
    except Exception as error:  # pragma: no cover - HTMLParser is deliberately tolerant
        return [], 0, 0, [f"html_parse_error:{type(error).__name__}"]
    if parser.nested_tables:
        issues.append(f"nested_tables:{parser.nested_tables}")
    if not parser.rows:
        return [], 0, 0, issues + ["no_rows"]

    occupied: dict[tuple[int, int], ParsedCell] = {}
    anchors: list[ParsedCell] = []
    for row_index, source_row in enumerate(parser.rows):
        column_index = 0
        for text, rowspan, colspan in source_row:
            while any(
                (row_index, column) in occupied
                for column in range(column_index, column_index + colspan)
            ):
                column_index += 1
            cell = ParsedCell(row_index, column_index, rowspan, colspan, text)
            anchors.append(cell)
            for row_offset in range(rowspan):
                for column_offset in range(colspan):
                    coordinate = (row_index + row_offset, column_index + column_offset)
                    if coordinate in occupied:
                        issues.append(f"span_overlap:r{coordinate[0]}c{coordinate[1]}")
                    occupied[coordinate] = cell
            column_index += colspan

    row_count = max((coordinate[0] for coordinate in occupied), default=-1) + 1
    column_count = max((coordinate[1] for coordinate in occupied), default=-1) + 1
    for row_index in range(row_count):
        missing = [
            column_index
            for column_index in range(column_count)
            if (row_index, column_index) not in occupied
        ]
        if missing:
            issues.append(f"ragged_row:{row_index}:missing={','.join(map(str, missing))}")
    return anchors, row_count, column_count, sorted(set(issues))


def _positive_int(value: str | None, *, default: int) -> int:
    try:
        parsed = int(value or default)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _normalize_text(value: str) -> str:
    value = html_module.unescape(unicodedata.normalize("NFKC", value)).casefold()
    return "".join(character for character in value if character.isalnum())


def _tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", html_module.unescape(value)).casefold()
    return [
        token
        for token in re.findall(r"[\w%.$€£¥+\-]+", normalized)
        if len(token) > 1
    ]


def _cell_text_score(cell_text: str, source_text: str) -> tuple[bool, float]:
    cell_normalized = _normalize_text(cell_text)
    source_normalized = _normalize_text(source_text)
    if len(cell_normalized) < 2:
        return False, math.nan
    exact = cell_normalized in source_normalized
    tokens = _tokens(cell_text)
    if not tokens:
        return exact, 1.0 if exact else 0.0
    source_tokens = _tokens(source_text)
    source_joined = " ".join(source_tokens)
    present = sum(_normalize_text(token) in _normalize_text(source_joined) for token in tokens)
    return exact, present / len(tokens)


def _bbox_iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _resolve_asset(document_dir: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    return candidate if candidate.is_absolute() else document_dir / candidate


def _first_html(payload: Any) -> str | None:
    if isinstance(payload, dict):
        html_value = payload.get("html")
        if isinstance(html_value, str) and html_value.strip():
            return html_value
        for value in payload.values():
            found = _first_html(value)
            if found:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _first_html(value)
            if found:
                return found
    return None


def _page_text_in_bbox(text_page: Any, bbox: list[float], page_height: float) -> str:
    left, top, right, bottom = bbox
    return text_page.get_text_bounded(left, page_height - bottom, right, page_height - top)


def _relative_uri(path: Path | None, output_dir: Path) -> str:
    if path is None:
        return ""
    return Path("..", Path(path).resolve().relative_to(output_dir.parent.resolve())).as_posix()


def audit(
    corpus_dir: Path,
    output_dir: Path,
    *,
    scan_visual_misses: bool = False,
) -> dict[str, Any]:
    softdoc_root = corpus_dir / "softdoc"
    pdf_root = corpus_dir / "pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_rows: list[dict[str, Any]] = []
    document_rows: list[dict[str, Any]] = []
    unmatched_preproc: list[dict[str, Any]] = []
    possible_visual_misses: list[dict[str, Any]] = []

    for document_dir in sorted(path for path in softdoc_root.iterdir() if path.is_dir()):
        document = load_document(document_dir)
        pdf_path = pdf_root / f"{document_dir.name}.pdf"
        pdf = pdfium.PdfDocument(pdf_path)
        page_by_id = {page.page_id: page for page in document.pages}
        tables_by_page: dict[str, list[Any]] = defaultdict(list)
        for element in document.elements:
            if element.element_type == ElementType.TABLE:
                tables_by_page[element.page_id].append(element)

        doc_counts = Counter()
        for page in document.pages:
            preproc_tables: list[dict[str, Any]] = []
            for index, block in enumerate(page.provenance.raw_payload.get("preproc_blocks", [])):
                if not isinstance(block, dict) or str(block.get("type", "")).lower() != "table":
                    continue
                bbox = block.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                preproc_tables.append({"index": index, "bbox": [float(x) for x in bbox], "html": _first_html(block)})
            page_tables = tables_by_page.get(page.page_id, [])
            for preproc in preproc_tables:
                matches = [
                    table.element_id
                    for table in page_tables
                    if table.bbox and _bbox_iou(preproc["bbox"], list(table.bbox.raw)) >= 0.85
                ]
                if not matches:
                    unmatched_preproc.append(
                        {
                            "document": document_dir.name,
                            "page_number": page.page_number,
                            "preproc_index": preproc["index"],
                            "bbox": preproc["bbox"],
                            "has_html": bool(preproc["html"]),
                        }
                    )
            page_asset = _resolve_asset(document_dir, page.image_path)
            if scan_visual_misses and page_asset and page_asset.is_file():
                for candidate_bbox in _detect_ruled_table_regions(page_asset, page.width, page.height):
                    overlaps = [
                        _bbox_iou(candidate_bbox, list(table.bbox.raw))
                        for table in page_tables
                        if table.bbox
                    ]
                    if max(overlaps, default=0.0) < 0.35:
                        possible_visual_misses.append(
                            {
                                "document": document_dir.name,
                                "page_number": page.page_number,
                                "page_id": page.page_id,
                                "bbox": [round(value, 2) for value in candidate_bbox],
                                "page_width": page.width,
                                "page_height": page.height,
                                "page_asset": str(page_asset),
                                "maximum_table_overlap": round(max(overlaps, default=0.0), 4),
                                "detector": "ruled_grid_cv_heuristic",
                            }
                        )

        for table in [element for element in document.elements if element.element_type == ElementType.TABLE]:
            doc_counts["tables"] += 1
            page = page_by_id[table.page_id]
            raw_bbox = list(table.bbox.raw) if table.bbox else None
            html_source = table.html or ""
            cells, row_count, column_count, grid_issues = parse_table_html(html_source)
            if not html_source:
                doc_counts["missing_html"] += 1
            if grid_issues:
                doc_counts["grid_issues"] += 1

            page_asset = _resolve_asset(document_dir, page.image_path)
            table_asset = _resolve_asset(document_dir, table.visual_asset_path)
            asset_exists = bool(table_asset and table_asset.is_file())
            page_asset_exists = bool(page_asset and page_asset.is_file())
            crop_ratio_error: float | None = None
            if asset_exists and raw_bbox:
                with Image.open(table_asset) as image:
                    asset_ratio = image.width / max(1, image.height)
                bbox_ratio = (raw_bbox[2] - raw_bbox[0]) / max(1.0, raw_bbox[3] - raw_bbox[1])
                crop_ratio_error = abs(asset_ratio / bbox_ratio - 1.0)

            bbox_valid = bool(
                raw_bbox
                and 0 <= raw_bbox[0] < raw_bbox[2] <= page.width
                and 0 <= raw_bbox[1] < raw_bbox[3] <= page.height
            )
            bounded_text = ""
            if raw_bbox and page.page_index < len(pdf):
                pdf_page = pdf[page.page_index]
                text_page = pdf_page.get_textpage()
                bounded_text = _page_text_in_bbox(text_page, raw_bbox, page.height)
                text_page.close()
                pdf_page.close()

            eligible = 0
            exact_matches = 0
            token_scores: list[float] = []
            cell_audit: list[dict[str, Any]] = []
            for cell in cells:
                exact, token_score = _cell_text_score(cell.text, bounded_text)
                if not math.isnan(token_score):
                    eligible += 1
                    exact_matches += int(exact)
                    token_scores.append(token_score)
                cell_audit.append(
                    {
                        "cell_id": f"r{cell.row}c{cell.column}",
                        "rowspan": cell.rowspan,
                        "colspan": cell.colspan,
                        "text": cell.text,
                        "exact_in_pdf_bbox_text": exact if not math.isnan(token_score) else None,
                        "token_coverage": None if math.isnan(token_score) else round(token_score, 4),
                    }
                )
            exact_ratio = exact_matches / eligible if eligible else None
            token_coverage = sum(token_scores) / len(token_scores) if token_scores else None
            has_pdf_text = bool(_normalize_text(bounded_text))
            risk_reasons: list[str] = []
            if not html_source:
                risk_reasons.append("missing_html")
            if grid_issues:
                risk_reasons.append("grid_structure_issue")
            if not asset_exists:
                risk_reasons.append("missing_table_asset")
            if not page_asset_exists:
                risk_reasons.append("missing_page_asset")
            if not bbox_valid:
                risk_reasons.append("invalid_or_missing_bbox")
            if crop_ratio_error is not None and crop_ratio_error > 0.20:
                risk_reasons.append("crop_aspect_mismatch")
            if has_pdf_text and eligible >= 3 and (exact_ratio or 0.0) < 0.65 and (token_coverage or 0.0) < 0.85:
                risk_reasons.append("low_cell_text_agreement")
            if not has_pdf_text:
                risk_reasons.append("no_pdf_text_layer_for_independent_check")

            if risk_reasons:
                doc_counts["risk_tables"] += 1
            if table.metadata.get("html_recovery"):
                doc_counts["recovered_html"] += 1
            table_rows.append(
                {
                    "document": document_dir.name,
                    "document_id": document.document_id,
                    "element_id": table.element_id,
                    "page_id": table.page_id,
                    "page_number": table.page_number,
                    "bbox": raw_bbox,
                    "page_width": page.width,
                    "page_height": page.height,
                    "page_asset": str(page_asset) if page_asset else None,
                    "table_asset": str(table_asset) if table_asset else None,
                    "html": html_source,
                    "row_count": row_count,
                    "column_count": column_count,
                    "anchor_cell_count": len(cells),
                    "grid_issues": grid_issues,
                    "asset_exists": asset_exists,
                    "page_asset_exists": page_asset_exists,
                    "bbox_valid": bbox_valid,
                    "crop_aspect_relative_error": None if crop_ratio_error is None else round(crop_ratio_error, 4),
                    "pdf_bbox_text": bounded_text,
                    "has_pdf_text_layer": has_pdf_text,
                    "eligible_cell_count": eligible,
                    "exact_cell_match_count": exact_matches,
                    "exact_cell_match_ratio": None if exact_ratio is None else round(exact_ratio, 4),
                    "mean_cell_token_coverage": None if token_coverage is None else round(token_coverage, 4),
                    "risk_reasons": risk_reasons,
                    "html_recovery": table.metadata.get("html_recovery"),
                    "cells": cell_audit,
                }
            )
        pdf.close()
        document_rows.append({"document": document_dir.name, **dict(doc_counts)})

    summary = {
        "documents": len(document_rows),
        "tables": len(table_rows),
        "tables_with_html": sum(bool(row["html"]) for row in table_rows),
        "tables_without_html": sum(not row["html"] for row in table_rows),
        "tables_with_assets": sum(row["asset_exists"] for row in table_rows),
        "tables_with_valid_bbox": sum(row["bbox_valid"] for row in table_rows),
        "tables_with_pdf_text_layer": sum(row["has_pdf_text_layer"] for row in table_rows),
        "tables_without_pdf_text_layer": sum(not row["has_pdf_text_layer"] for row in table_rows),
        "tables_with_grid_issues": sum(bool(row["grid_issues"]) for row in table_rows),
        "tables_with_low_cell_text_agreement": sum("low_cell_text_agreement" in row["risk_reasons"] for row in table_rows),
        "tables_with_crop_aspect_mismatch": sum("crop_aspect_mismatch" in row["risk_reasons"] for row in table_rows),
        "tables_recovered_from_preproc": sum(bool(row["html_recovery"]) for row in table_rows),
        "unmatched_preproc_table_blocks": len(unmatched_preproc),
        "possible_visual_table_misses": (
            len(possible_visual_misses) if scan_visual_misses else None
        ),
        "risk_reason_counts": dict(Counter(reason for row in table_rows for reason in row["risk_reasons"])),
        "audit_scope_note": (
            "Cell text is independently compared with the PDF text layer inside the table bbox. "
            "This validates content presence, not exact visual cell boundaries. Image-only tables "
            "and MinerU-level visual misses require visual review. The optional ruled-grid "
            "scan is diagnostic only and is disabled by default because it is not a reliable "
            "table detector."
        ),
    }
    payload = {
        "summary": summary,
        "documents": document_rows,
        "unmatched_preproc_tables": unmatched_preproc,
        "possible_visual_table_misses": possible_visual_misses,
        "tables": table_rows,
    }
    (output_dir / "table_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_markdown(
        output_dir / "table_audit.md",
        summary,
        document_rows,
        table_rows,
        unmatched_preproc,
        possible_visual_misses if scan_visual_misses else None,
    )
    _write_gallery(output_dir / "index.html", output_dir, table_rows, summary)
    _write_risk_contact_sheets(output_dir / "contact_sheets", table_rows)
    return payload


def _write_markdown(
    path: Path,
    summary: dict[str, Any],
    documents: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    unmatched: list[dict[str, Any]],
    possible_visual_misses: list[dict[str, Any]] | None,
) -> None:
    lines = [
        "# Representative-28 table audit",
        "",
        "## Summary",
        "",
        *(f"- {key}: {value}" for key, value in summary.items() if key != "audit_scope_note"),
        "",
        f"> {summary['audit_scope_note']}",
        "",
        "## Per-document counts",
        "",
        "| Document | Tables | Missing HTML | Recovered HTML | Grid issues | Risk tables |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in documents:
        lines.append(
            f"| {row['document']} | {row.get('tables', 0)} | {row.get('missing_html', 0)} | "
            f"{row.get('recovered_html', 0)} | {row.get('grid_issues', 0)} | {row.get('risk_tables', 0)} |"
        )
    lines += ["", "## High-risk tables", ""]
    risky = [row for row in tables if row["risk_reasons"]]
    if not risky:
        lines.append("No automatically detected risks.")
    else:
        lines += [
            "| Document | Page | Element | Rows x columns | Exact cell ratio | Token coverage | Reasons |",
            "|---|---:|---|---:|---:|---:|---|",
        ]
        for row in risky:
            lines.append(
                f"| {row['document']} | {row['page_number']} | `{row['element_id']}` | "
                f"{row['row_count']}x{row['column_count']} | {row['exact_cell_match_ratio']} | "
                f"{row['mean_cell_token_coverage']} | {', '.join(row['risk_reasons'])} |"
            )
    lines += ["", "## Unmatched MinerU preproc table blocks", ""]
    if unmatched:
        for row in unmatched:
            lines.append(f"- {row}")
    else:
        lines.append("None. Every MinerU preproc table block maps to a SoftDoc Table element.")
    lines += ["", "## MinerU-level visual miss check", ""]
    if possible_visual_misses is None:
        lines.append(
            "The noisy ruled-grid detector was not run. The audit proves that every MinerU "
            "preproc table was preserved by SoftDoc, but it does not claim that MinerU found "
            "every visually apparent table in every PDF."
        )
    else:
        lines.append(
            "These are ruled-grid CV candidates, not confirmed misses. Borderless tables "
            "cannot be certified by this detector."
        )
        if possible_visual_misses:
            for row in possible_visual_misses:
                lines.append(
                    f"- {row['document']} page {row['page_number']}, bbox={row['bbox']}, "
                    f"max overlap={row['maximum_table_overlap']}"
                )
        else:
            lines.append("No unmatched ruled-grid candidates were detected.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _detect_ruled_table_regions(image_path: Path, page_width: float, page_height: float) -> list[list[float]]:
    """Return conservative page-coordinate bboxes for ruled-grid regions.

    The heuristic is only a guard against obvious missed bordered tables.  It
    deliberately does not claim recall for borderless financial tables.
    """

    # OpenCV is intentionally an optional audit-only dependency. The default
    # audit does not import or require it.
    import cv2
    import numpy as np

    with Image.open(image_path) as source:
        grayscale = np.asarray(source.convert("L"))
    pixel_height, pixel_width = grayscale.shape
    if max(pixel_width, pixel_height) > 1400:
        scale = 1400 / max(pixel_width, pixel_height)
        grayscale = cv2.resize(grayscale, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    height, width = grayscale.shape
    binary = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, width // 18), 1)),
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(18, height // 28))),
    )
    grid = cv2.dilate(
        cv2.bitwise_or(horizontal, vertical),
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[list[float]] = []
    for contour in contours:
        x, y, candidate_width, candidate_height = cv2.boundingRect(contour)
        if candidate_width < width * 0.20 or candidate_height < height * 0.035:
            continue
        if candidate_width * candidate_height < width * height * 0.008:
            continue
        roi_h = horizontal[y : y + candidate_height, x : x + candidate_width]
        roi_v = vertical[y : y + candidate_height, x : x + candidate_width]
        horizontal_components = _long_component_count(roi_h, horizontal=True)
        vertical_components = _long_component_count(roi_v, horizontal=False)
        if horizontal_components < 3 or vertical_components < 2:
            continue
        candidates.append(
            [
                x / width * page_width,
                y / height * page_height,
                (x + candidate_width) / width * page_width,
                (y + candidate_height) / height * page_height,
            ]
        )
    return candidates


def _long_component_count(mask: Any, *, horizontal: bool) -> int:
    import cv2

    count = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape
    for contour in contours:
        _, _, component_width, component_height = cv2.boundingRect(contour)
        if horizontal and component_width >= width * 0.30:
            count += 1
        elif not horizontal and component_height >= height * 0.30:
            count += 1
    return count


def _write_gallery(path: Path, output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    cards: list[str] = []
    for index, row in enumerate(rows, start=1):
        page_uri = _relative_uri(Path(row["page_asset"]), output_dir) if row["page_asset"] else ""
        table_uri = _relative_uri(Path(row["table_asset"]), output_dir) if row["table_asset"] else ""
        bbox = row["bbox"] or [0, 0, 1, 1]
        cells_html = "".join(
            f'<td rowspan="{cell["rowspan"]}" colspan="{cell["colspan"]}">'
            f'<small>{html_module.escape(cell["cell_id"])}</small>{html_module.escape(cell["text"])}</td>'
            for cell in row["cells"]
        )
        # Rebuild rows from anchor coordinates for readable diagnostic rendering.
        by_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for cell in row["cells"]:
            row_number = int(cell["cell_id"].split("c", 1)[0][1:])
            by_row[row_number].append(cell)
        matrix = "".join(
            "<tr>" + "".join(
                f'<td rowspan="{cell["rowspan"]}" colspan="{cell["colspan"]}">'
                f'<small>{html_module.escape(cell["cell_id"])}</small>{html_module.escape(cell["text"])}</td>'
                for cell in sorted(cells, key=lambda item: int(item["cell_id"].split("c", 1)[1]))
            ) + "</tr>"
            for _, cells in sorted(by_row.items())
        )
        risks = ", ".join(row["risk_reasons"]) or "none"
        cards.append(
            f'''<article class="card" data-risk="{html_module.escape(risks)}">
<h2>{index}. {html_module.escape(row['document'])} - page {row['page_number']}</h2>
<p><code>{html_module.escape(row['element_id'])}</code></p>
<p>Grid {row['row_count']} x {row['column_count']}; exact cell text ratio: {row['exact_cell_match_ratio']};
token coverage: {row['mean_cell_token_coverage']}; risks: <strong>{html_module.escape(risks)}</strong></p>
<div class="views">
  <section><h3>1. Page image with table bbox</h3><div class="page-wrap"><img src="{page_uri}" alt="page"><span class="bbox" data-x1="{bbox[0]}" data-y1="{bbox[1]}" data-x2="{bbox[2]}" data-y2="{bbox[3]}" data-w="{row['page_width']}" data-h="{row['page_height']}"></span></div></section>
  <section><h3>2. MinerU table image asset</h3><img class="table-image" src="{table_uri}" alt="table crop"></section>
  <section><h3>3. MinerU HTML rendered by browser</h3><div class="rendered">{row['html']}</div></section>
  <section><h3>4. Addressable logical cell matrix (diagnostic only)</h3><table class="matrix">{matrix}</table></section>
</div>
<details><summary>PDF text inside bbox</summary><pre>{html_module.escape(row['pdf_bbox_text'])}</pre></details>
<details><summary>Raw MinerU HTML</summary><pre>{html_module.escape(row['html'])}</pre></details>
</article>'''
        )
    document = f'''<!doctype html><html><head><meta charset="utf-8"><title>Representative-28 Table Audit</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#111827;color:#e5e7eb;margin:20px}}code,pre{{white-space:pre-wrap;word-break:break-all}}button{{margin-right:8px;padding:8px}}.card{{background:#1f2937;padding:18px;margin:18px 0;border-radius:12px}}.views{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}section{{background:#fff;color:#111;padding:12px;overflow:auto;min-height:180px}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #64748b;padding:5px}}small{{display:block;color:#dc2626}}.page-wrap{{position:relative;display:inline-block;line-height:0}}.page-wrap img{{display:block}}.bbox{{position:absolute;border:3px solid #ef4444;box-sizing:border-box;pointer-events:none}}.hidden{{display:none}}@media(max-width:1100px){{.views{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Representative-28 Table Audit</h1>
<p>This page shows four views of the same table, not four different tables. Views 1 and 2 are images; views 3 and 4 are HTML representations.</p>
<p>{html_module.escape(json.dumps(summary, ensure_ascii=False))}</p>
<button onclick="filterCards(false)">Show all</button><button onclick="filterCards(true)">Show risk only</button>
{''.join(cards)}
<script>
for (const box of document.querySelectorAll('.bbox')) {{
  box.style.left=(100*box.dataset.x1/box.dataset.w)+'%'; box.style.top=(100*box.dataset.y1/box.dataset.h)+'%';
  box.style.width=(100*(box.dataset.x2-box.dataset.x1)/box.dataset.w)+'%'; box.style.height=(100*(box.dataset.y2-box.dataset.y1)/box.dataset.h)+'%';
}}
function filterCards(riskOnly) {{ for (const card of document.querySelectorAll('.card')) card.classList.toggle('hidden', riskOnly && card.dataset.risk==='none'); }}
</script></body></html>'''
    path.write_text(document, encoding="utf-8")


def write_table_type_error_review(
    audit_path: Path,
    output_dir: Path,
    *,
    contains_embedded_images: bool | None = None,
) -> dict[str, Any]:
    """Write a focused, non-mutating review for known type-error candidates."""

    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    all_rows = payload.get("tables", [])
    selected = [
        row for row in all_rows if row.get("element_id") in TABLE_TYPE_ERROR_REVIEW
    ]
    if contains_embedded_images is not None:
        selected = [
            row
            for row in selected
            if bool(re.search(r"<img\b", row.get("html") or "", re.IGNORECASE))
            is contains_embedded_images
        ]
    selected_ids = {row["element_id"] for row in selected}
    missing = sorted(set(TABLE_TYPE_ERROR_REVIEW) - selected_ids)
    if contains_embedded_images is None and missing:
        raise ValueError(f"Missing type-error review elements: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    review_rows: list[dict[str, Any]] = []
    for row in selected:
        suggested_type, reason = TABLE_TYPE_ERROR_REVIEW[row["element_id"]]
        reviewed = dict(row)
        reviewed["audit_suggested_type"] = suggested_type
        reviewed["audit_reason"] = reason
        review_rows.append(reviewed)

    summary = {
        "review_scope": (
            "manually identified MinerU table type-error candidates"
            if contains_embedded_images is None
            else (
                "type-error candidates with embedded HTML images"
                if contains_embedded_images
                else "type-error candidates without embedded HTML images"
            )
        ),
        "candidate_count": len(review_rows),
        "mutates_softdoc": False,
        "warning": (
            "A plausible HTML table can still be wrong for the outer visual when the crop "
            "is a composite figure containing only a small embedded table."
        ),
    }
    _write_type_error_gallery(
        output_dir / "index.html",
        output_dir,
        review_rows,
        summary,
    )
    compact_rows = [
        {
            "document": row["document"],
            "page_number": row["page_number"],
            "element_id": row["element_id"],
            "current_type": "table",
            "suggested_type": row["audit_suggested_type"],
            "reason": row["audit_reason"],
            "row_count": row["row_count"],
            "column_count": row["column_count"],
            "html_present": bool(row["html"]),
            "table_asset": row["table_asset"],
        }
        for row in review_rows
    ]
    result = {"summary": summary, "candidates": compact_rows}
    (output_dir / "review.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def _write_type_error_gallery(
    path: Path,
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    cards: list[str] = []
    for index, row in enumerate(rows, start=1):
        page_uri = _relative_uri(Path(row["page_asset"]), output_dir)
        table_uri = _relative_uri(Path(row["table_asset"]), output_dir)
        bbox = row["bbox"]
        by_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for cell in row["cells"]:
            row_number = int(cell["cell_id"].split("c", 1)[0][1:])
            by_row[row_number].append(cell)
        matrix = "".join(
            "<tr>"
            + "".join(
                f'<td rowspan="{cell["rowspan"]}" colspan="{cell["colspan"]}">'
                f'<small>{html_module.escape(cell["cell_id"])}</small>'
                f'{html_module.escape(cell["text"])}</td>'
                for cell in sorted(
                    cells,
                    key=lambda item: int(item["cell_id"].split("c", 1)[1]),
                )
            )
            + "</tr>"
            for _, cells in sorted(by_row.items())
        )
        table_asset = Path(row["table_asset"])
        document_dir = table_asset.parents[2]
        rendered_html, embedded_images = _resolve_review_html_images(
            row["html"],
            document_dir,
            output_dir,
        )
        embedded_gallery = "".join(
            '<figure><img src="{uri}" alt="embedded image">'
            '<figcaption>{index}. {name}</figcaption></figure>'.format(
                uri=image["uri"],
                index=image_index,
                name=html_module.escape(image["name"]),
            )
            for image_index, image in enumerate(embedded_images, start=1)
        )
        if not embedded_gallery:
            embedded_gallery = (
                '<p class="no-images">This HTML contains no embedded image files.</p>'
            )
        cards.append(
            f'''<article class="card">
<h2>{index}. {html_module.escape(row['document'])} - page {row['page_number']}</h2>
<p><code>{html_module.escape(row['element_id'])}</code></p>
<p class="judgment"><strong>Audit judgment:</strong> current <code>table</code> -> likely <code>{html_module.escape(row['audit_suggested_type'])}</code><br>
<strong>Reason:</strong> {html_module.escape(row['audit_reason'])}</p>
<p>MinerU HTML grid: {row['row_count']} x {row['column_count']}. HTML present: {bool(row['html'])}.</p>
<div class="views">
  <section><h3>1. Original page + detected bbox</h3><div class="page-wrap"><img src="{page_uri}" alt="page"><span class="bbox" data-x1="{bbox[0]}" data-y1="{bbox[1]}" data-x2="{bbox[2]}" data-y2="{bbox[3]}" data-w="{row['page_width']}" data-h="{row['page_height']}"></span></div></section>
  <section><h3>2. MinerU crop currently typed as Table</h3><img src="{table_uri}" alt="candidate crop"></section>
  <section><h3>3. MinerU HTML interpretation ({len(embedded_images)} actual embedded images)</h3><div class="rendered">{rendered_html}</div></section>
  <section><h3>4. HTML cell coordinates (not a corrected TableView)</h3><table>{matrix}</table></section>
  <section class="wide"><h3>5. Embedded image files</h3><div class="gallery">{embedded_gallery}</div></section>
</div>
<details><summary>Raw MinerU HTML</summary><pre>{html_module.escape(row['html'])}</pre></details>
</article>'''
        )

    document = f'''<!doctype html><html><head><meta charset="utf-8"><title>Table Type Error Review</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#111827;color:#e5e7eb;margin:20px}}code,pre{{white-space:pre-wrap;word-break:break-all}}.card{{background:#1f2937;padding:18px;margin:18px 0;border-radius:12px}}.judgment{{background:#3f2b20;padding:12px;border-left:5px solid #f59e0b}}.views{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}section{{background:#fff;color:#111;padding:12px;overflow:auto;min-height:180px}}section.wide{{grid-column:1/-1}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #64748b;padding:5px}}.rendered img{{max-width:260px;max-height:220px;object-fit:contain}}.gallery{{display:flex;flex-wrap:wrap;gap:12px}}figure{{margin:0;padding:8px;border:1px solid #94a3b8;max-width:300px}}figure img{{display:block;max-width:280px;max-height:240px;object-fit:contain}}figcaption{{margin-top:6px;word-break:break-all}}.no-images{{color:#475569}}small{{display:block;color:#dc2626}}.page-wrap{{position:relative;display:inline-block;line-height:0}}.page-wrap img{{display:block}}.bbox{{position:absolute;border:3px solid #ef4444;box-sizing:border-box;pointer-events:none}}@media(max-width:1100px){{.views{{grid-template-columns:1fr}}section.wide{{grid-column:auto}}}}
</style></head><body>
<h1>{len(rows)} MinerU Table Type-Error Candidates</h1>
<p>This review is intentionally limited to these {len(rows)} candidates. It does not change SoftDoc.</p>
<p>Important: a candidate can have plausible HTML because MinerU extracted an embedded table or arranged visible text into rows. That does not make the entire visual a Table.</p>
<p>{html_module.escape(json.dumps(summary, ensure_ascii=False))}</p>
{''.join(cards)}
<script>for(const box of document.querySelectorAll('.bbox')){{box.style.left=(100*box.dataset.x1/box.dataset.w)+'%';box.style.top=(100*box.dataset.y1/box.dataset.h)+'%';box.style.width=(100*(box.dataset.x2-box.dataset.x1)/box.dataset.w)+'%';box.style.height=(100*(box.dataset.y2-box.dataset.y1)/box.dataset.h)+'%';}}</script>
</body></html>'''
    path.write_text(document, encoding="utf-8")


def _html_for_review(raw_html: str) -> str:
    """Render table markup without broken relative image references.

    MinerU can embed ``images/...`` paths inside table HTML.  Those paths are
    relative to the MinerU output, not to this portable review.  Keep the raw
    HTML below the card, but replace embedded images in the rendered preview
    with an explicit marker so the review never shows misleading broken icons.
    """

    def replace_image(match: re.Match[str]) -> str:
        tag = match.group(0)
        source_match = re.search(r"\bsrc\s*=\s*(['\"])(.*?)\1", tag, re.IGNORECASE)
        source = source_match.group(2) if source_match else "unknown"
        return (
            '<span class="embedded-image-placeholder">'
            f"[embedded image omitted: {html_module.escape(Path(source).name)}]"
            "</span>"
        )

    return re.sub(r"<img\b[^>]*>", replace_image, raw_html, flags=re.IGNORECASE)


def write_embedded_image_table_review(
    corpus_dir: Path,
    output_dir: Path,
) -> dict[str, int]:
    """Render every SoftDoc table whose HTML embeds one or more real images."""

    softdoc_root = corpus_dir / "softdoc"
    output_dir.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    table_count = 0
    image_count = 0
    candidate_count = 0

    for document_dir in sorted(path for path in softdoc_root.iterdir() if path.is_dir()):
        document = load_document(document_dir)
        pages = {page.page_id: page for page in document.pages}
        for element in document.elements:
            if element.element_type != ElementType.TABLE or not element.html:
                continue
            rendered_html, embedded = _resolve_review_html_images(
                element.html,
                document_dir,
                output_dir,
            )
            if not embedded:
                continue

            table_count += 1
            image_count += len(embedded)
            page = pages[element.page_id]
            page_asset = _resolve_asset(document_dir, page.image_path)
            outer_asset = _resolve_asset(
                document_dir,
                element.image_path or element.crop_image_path,
            )
            page_uri = _relative_uri(page_asset, output_dir)
            outer_uri = _relative_uri(outer_asset, output_dir)
            bbox = element.bbox.normalized
            review = TABLE_TYPE_ERROR_REVIEW.get(element.element_id)
            if review is None:
                verdict = '<span class="badge valid">legitimate image-containing table</span>'
                explanation = (
                    "The outer object has stable row-column semantics; images are cell content."
                )
            else:
                candidate_count += 1
                suggested_type, reason = review
                verdict = (
                    '<span class="badge candidate">type-error candidate: table → '
                    f'{html_module.escape(suggested_type)}</span>'
                )
                explanation = reason

            image_gallery = "".join(
                '<figure><img src="{uri}" alt="embedded image">'
                '<figcaption>{index}. {name}</figcaption></figure>'.format(
                    uri=entry["uri"],
                    index=index,
                    name=html_module.escape(entry["name"]),
                )
                for index, entry in enumerate(embedded, start=1)
            )
            cards.append(
                f'''<article class="card">
<h2>{table_count}. {html_module.escape(document_dir.name)} — PDF page {element.page_number}</h2>
<p>{verdict}</p><p>{html_module.escape(explanation)}</p>
<p><code>{html_module.escape(element.element_id)}</code> · embedded images: {len(embedded)}</p>
<div class="views">
  <section><h3>1. Original page + bbox</h3><div class="page-wrap"><img src="{page_uri}" alt="page"><span class="bbox" style="left:{bbox[0]*100}%;top:{bbox[1]*100}%;width:{(bbox[2]-bbox[0])*100}%;height:{(bbox[3]-bbox[1])*100}%"></span></div></section>
  <section><h3>2. Whole Element crop</h3><img src="{outer_uri}" alt="whole table crop"></section>
  <section class="wide"><h3>3. SoftDoc HTML rendered with actual images</h3><div class="rendered">{rendered_html}</div></section>
  <section class="wide"><h3>4. Embedded image files ({len(embedded)})</h3><div class="gallery">{image_gallery}</div></section>
</div>
<details><summary>Current SoftDoc HTML</summary><pre>{html_module.escape(element.html)}</pre></details>
</article>'''
            )

    summary = {
        "tables": table_count,
        "embedded_images": image_count,
        "type_error_candidates": candidate_count,
        "legitimate_tables": table_count - candidate_count,
    }
    document_html = f'''<!doctype html><html><head><meta charset="utf-8"><title>Embedded-image Table Review</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#111827;color:#e5e7eb;margin:20px}}code,pre{{white-space:pre-wrap;word-break:break-all}}.card{{background:#1f2937;padding:18px;margin:18px 0;border-radius:12px}}.views{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}section{{background:#fff;color:#111;padding:12px;overflow:auto;min-height:180px}}section.wide{{grid-column:1/-1}}img{{max-width:100%;height:auto}}.rendered table{{border-collapse:collapse;width:100%}}.rendered td,.rendered th{{border:1px solid #64748b;padding:6px;vertical-align:top}}.rendered img{{max-width:260px;max-height:220px;object-fit:contain}}.gallery{{display:flex;flex-wrap:wrap;gap:12px}}figure{{margin:0;padding:8px;border:1px solid #94a3b8;max-width:300px}}figure img{{display:block;max-width:280px;max-height:240px;object-fit:contain}}figcaption{{margin-top:6px;word-break:break-all}}.badge{{display:inline-block;padding:5px 9px;border-radius:999px;font-weight:700}}.valid{{background:#166534}}.candidate{{background:#92400e}}.page-wrap{{position:relative;display:inline-block;line-height:0}}.page-wrap>img{{display:block}}.bbox{{position:absolute;border:4px solid #ef4444;box-sizing:border-box;pointer-events:none}}@media(max-width:1100px){{.views{{grid-template-columns:1fr}}section.wide{{grid-column:auto}}}}
</style></head><body>
<h1>16 SoftDoc Tables with Embedded HTML Images</h1>
<p>This review renders the actual 55 local image files. It does not replace images with src text or placeholders.</p>
<p><strong>{html_module.escape(json.dumps(summary, ensure_ascii=False))}</strong></p>
{''.join(cards)}
</body></html>'''
    (output_dir / "index.html").write_text(document_html, encoding="utf-8")
    return summary


def write_unresolved_visual_table_review(
    corpus_dir: Path,
    output_dir: Path,
) -> dict[str, int]:
    """Review tables with no HTML image src but evidence of visual content."""

    softdoc_root = corpus_dir / "softdoc"
    output_dir.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    placeholder_candidates = 0
    manual_visual_candidates = 0

    for document_dir in sorted(path for path in softdoc_root.iterdir() if path.is_dir()):
        document = load_document(document_dir)
        pages = {page.page_id: page for page in document.pages}
        for element in document.elements:
            if element.element_type != ElementType.TABLE or not element.html:
                continue
            if re.search(r"<img\b", element.html, re.IGNORECASE):
                continue
            placeholders = sorted(
                set(
                    re.findall(
                        r"(?:&lt;|<)(image\s*\d+)(?:&gt;|>)",
                        element.html,
                        re.IGNORECASE,
                    )
                )
            )
            manual_review = TABLE_TYPE_ERROR_REVIEW.get(element.element_id)
            if not placeholders and manual_review is None:
                continue

            if placeholders:
                placeholder_candidates += 1
                signal = (
                    f"HTML contains {len(placeholders)} unresolved visual placeholder(s) "
                    "but zero <img src> resources."
                )
            else:
                manual_visual_candidates += 1
                signal = (
                    "No HTML image or placeholder; retained because visual review found "
                    "the whole object is a non-tabular graphic/UI/code image."
                )
            judgment = (
                manual_review[1]
                if manual_review is not None
                else "Not in the previous 22-item audit; discovered by the placeholder scan."
            )
            page = pages[element.page_id]
            page_asset = _resolve_asset(document_dir, page.image_path)
            outer_asset = _write_exact_bbox_review_crop(
                page_asset=page_asset,
                normalized_bbox=element.bbox.normalized,
                destination=output_dir / "assets" / f"candidate_{len(cards) + 1:03d}.png",
            )
            outer_uri = _relative_uri(outer_asset, output_dir)
            verified_regions = UNRESOLVED_PLACEHOLDER_REVIEW_REGIONS.get(
                element.element_id, ()
            )
            region_cards: list[str] = []
            for region_index, (label, region_bbox) in enumerate(
                verified_regions, start=1
            ):
                region_asset = _write_normalized_crop(
                    source_asset=outer_asset,
                    normalized_bbox=region_bbox,
                    destination=(
                        output_dir
                        / "assets"
                        / f"candidate_{len(cards) + 1:03d}_{region_index}.png"
                    ),
                )
                label_rows = _html_rows_for_placeholder(element.html, label)
                rendered_references = _render_unresolved_placeholder_rows(
                    label_rows, label
                )
                region_cards.append(
                    f'''<div class="mapping">
<section><h4>Verified visual region: {html_module.escape(label)}</h4>
<img src="{_relative_uri(region_asset, output_dir)}" alt="verified visual region"></section>
<section><h4>Every HTML reference to {html_module.escape(label)}</h4>
<div class="rendered">{rendered_references}</div></section>
</div>'''
                )

            if region_cards:
                mapping_html = "".join(region_cards)
                mapping_status = (
                    "The inner region was manually verified for this audit. "
                    "It is not an automatic MinerU bbox."
                )
            else:
                mapping_html = (
                    '<p class="unmapped">No internal image mapping is available. '
                    "Only the exact outer Element crop can be shown.</p>"
                )
                mapping_status = (
                    "No inner image bbox or img src exists; an exact internal crop "
                    "cannot be claimed."
                )
            index = len(cards) + 1
            cards.append(
                f'''<article class="card">
<h2>{index}. {html_module.escape(document_dir.name)} — PDF page {element.page_number}</h2>
<p><code>{html_module.escape(element.element_id)}</code></p>
<p class="signal"><strong>Why included:</strong> {html_module.escape(signal)}</p>
<p><strong>Audit note:</strong> {html_module.escape(judgment)}</p>
<p>HTML <code>&lt;img src&gt;</code>: <strong>0</strong> · unresolved placeholders: <strong>{len(placeholders)}</strong></p>
<p class="mapping-status"><strong>Mapping status:</strong> {html_module.escape(mapping_status)}</p>
<section class="outer"><h3>1. Exact outer Element crop (context only)</h3><img src="{outer_uri}" alt="exact outer Element bbox crop"></section>
<h3>2. Placeholder-to-visual audit mapping</h3>
{mapping_html}
<details><summary>Current SoftDoc HTML</summary><pre>{html_module.escape(element.html)}</pre></details>
</article>'''
            )

    summary = {
        "tables_without_img_src_but_with_visual_evidence": len(cards),
        "with_unresolved_placeholders": placeholder_candidates,
        "manual_visual_candidates_without_placeholders": manual_visual_candidates,
    }
    page = f'''<!doctype html><html><head><meta charset="utf-8"><title>Unresolved Visual Table Review</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#111827;color:#e5e7eb;margin:20px}}code,pre{{white-space:pre-wrap;word-break:break-all}}.card{{background:#1f2937;padding:18px;margin:18px 0;border-radius:12px}}.signal{{background:#3f2b20;padding:12px;border-left:5px solid #f59e0b}}.mapping-status{{background:#172554;padding:12px;border-left:5px solid #60a5fa}}section{{background:#fff;color:#111;padding:12px;overflow:auto;min-height:120px}}section.outer{{margin-bottom:16px}}section.outer img{{max-height:850px}}.mapping{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:14px 0}}img{{max-width:100%;height:auto}}.rendered table{{border-collapse:collapse;width:100%}}.rendered td,.rendered th{{border:1px solid #64748b;padding:6px;vertical-align:top}}mark.missing{{background:#fecaca;color:#991b1b;font-weight:800;padding:2px 5px}}.unmapped{{background:#7f1d1d;padding:14px;font-weight:700}}@media(max-width:1100px){{.mapping{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Tables with no HTML img src but retained visual content</h1>
<p>The outer crop is context only. When MinerU omitted the inner image bbox, this review shows a separately verified audit crop and every HTML reference to the same placeholder. It never labels the outer Element as an exact inner-image crop.</p>
<p><strong>{html_module.escape(json.dumps(summary, ensure_ascii=False))}</strong></p>
{''.join(cards)}
</body></html>'''
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    return summary


def write_type_error_embedded_image_mapping_review(
    corpus_dir: Path,
    output_dir: Path,
) -> dict[str, int]:
    """Map each real HTML img src to the exact cell that references it."""

    softdoc_root = corpus_dir / "softdoc"
    output_dir.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    parent_ids: set[str] = set()
    pattern = re.compile(
        r"(?P<prefix><img\b[^>]*?\bsrc\s*=\s*)(?P<quote>['\"])(?P<source>.*?)(?P=quote)",
        re.IGNORECASE,
    )

    for document_dir in sorted(path for path in softdoc_root.iterdir() if path.is_dir()):
        document = load_document(document_dir)
        for element in document.elements:
            if element.element_id not in TABLE_TYPE_ERROR_REVIEW or not element.html:
                continue
            matches = list(pattern.finditer(element.html))
            if not matches:
                continue
            parent_ids.add(element.element_id)
            for match in matches:
                source = match.group("source")
                source_path = Path(source.replace("\\", "/"))
                asset = source_path if source_path.is_absolute() else document_dir / source_path
                if not asset.is_file():
                    raise FileNotFoundError(
                        f"Embedded image is unavailable for mapping review: {source}"
                    )
                image_uri = _relative_uri(asset, output_dir)
                cell_html = _enclosing_html_cell(element.html, match.start(), match.end())
                rendered_cell, _ = _resolve_review_html_images(
                    "<table><tr>" + cell_html + "</tr></table>",
                    document_dir,
                    output_dir,
                )
                index = len(cards) + 1
                cards.append(
                    f'''<article class="card">
<h2>{index}. {html_module.escape(document_dir.name)} — PDF page {element.page_number}</h2>
<p><code>{html_module.escape(element.element_id)}</code></p>
<p><strong>Exact src:</strong> <code>{html_module.escape(source)}</code></p>
<div class="views">
  <section><h3>1. Actual file referenced by this src</h3><img class="source-image" src="{image_uri}" alt="referenced embedded image"></section>
  <section><h3>2. Exact HTML cell containing this src</h3><div class="rendered">{rendered_cell}</div></section>
</div>
</article>'''
                )

    summary = {
        "type_error_parent_elements": len(parent_ids),
        "mapped_img_src": len(cards),
    }
    page = f'''<!doctype html><html><head><meta charset="utf-8"><title>Type-error img src mapping</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#111827;color:#e5e7eb;margin:20px}}code{{white-space:pre-wrap;word-break:break-all}}.card{{background:#1f2937;padding:18px;margin:18px 0;border-radius:12px}}.views{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}section{{background:#fff;color:#111;padding:12px;overflow:auto;min-height:180px}}img.source-image{{display:block;max-width:100%;max-height:650px;object-fit:contain;margin:auto}}.rendered table{{border-collapse:collapse;width:100%}}.rendered td,.rendered th{{border:1px solid #64748b;padding:6px;vertical-align:top}}.rendered img{{max-width:320px;max-height:320px;object-fit:contain}}@media(max-width:1100px){{.views{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Exact mapping for img src inside the 22 type-error candidates</h1>
<p>Each card represents one actual img src. No parent Element crop is shown, so unrelated content cannot be mistaken for the referenced image.</p>
<p><strong>{html_module.escape(json.dumps(summary, ensure_ascii=False))}</strong></p>
{''.join(cards)}
</body></html>'''
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    return summary


def _resolve_review_html_images(
    raw_html: str,
    document_dir: Path,
    output_dir: Path,
) -> tuple[str, list[dict[str, str]]]:
    resolved: list[dict[str, str]] = []

    def replace(match: re.Match[str]) -> str:
        prefix, quote, source = match.group("prefix", "quote", "source")
        source_path = Path(source.replace("\\", "/"))
        asset = source_path if source_path.is_absolute() else document_dir / source_path
        if not asset.is_file():
            raise FileNotFoundError(
                f"Embedded image is unavailable for review: {document_dir.name}: {source}"
            )
        uri = _relative_uri(asset, output_dir)
        resolved.append({"name": asset.name, "uri": uri})
        return f"{prefix}{quote}{uri}{quote}"

    pattern = re.compile(
        r"(?P<prefix><img\b[^>]*?\bsrc\s*=\s*)(?P<quote>['\"])(?P<source>.*?)(?P=quote)",
        re.IGNORECASE,
    )
    return pattern.sub(replace, raw_html), resolved


def _html_rows_with_visual_placeholders(raw_html: str) -> list[str]:
    return [
        row
        for row in re.findall(
            r"<tr\b[^>]*>.*?</tr>",
            raw_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if re.search(
            r"(?:&lt;|<)image\s*\d+(?:&gt;|>)",
            row,
            re.IGNORECASE,
        )
    ]


def _html_rows_for_placeholder(raw_html: str, label: str) -> list[str]:
    token = re.compile(
        rf"(?:&lt;|<){re.escape(label)}(?:&gt;|>)",
        re.IGNORECASE,
    )
    return [
        row
        for row in re.findall(
            r"<tr\b[^>]*>.*?</tr>",
            raw_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if token.search(row)
    ]


def _render_unresolved_placeholder_rows(rows: list[str], label: str) -> str:
    if not rows:
        return '<p class="unmapped">No matching HTML reference was found.</p>'
    context_html = "<table>" + "".join(rows) + "</table>"
    return re.sub(
        rf"(?:&lt;|<){re.escape(label)}(?:&gt;|>)",
        f'<mark class="missing">[UNRESOLVED {html_module.escape(label)}]</mark>',
        context_html,
        flags=re.IGNORECASE,
    )


def _enclosing_html_cell(raw_html: str, start: int, end: int) -> str:
    """Return the smallest td/th containing the match, never the whole row."""

    lowered = raw_html.lower()
    candidates: list[tuple[int, str]] = []
    for tag in ("td", "th"):
        cell_start = lowered.rfind(f"<{tag}", 0, start)
        if cell_start >= 0:
            candidates.append((cell_start, tag))
    if not candidates:
        return raw_html[max(0, start - 200) : min(len(raw_html), end + 200)]
    cell_start, tag = max(candidates)
    closing = f"</{tag}>"
    cell_end = lowered.find(closing, end)
    if cell_end < 0:
        return raw_html[max(0, start - 200) : min(len(raw_html), end + 200)]
    return raw_html[cell_start : cell_end + len(closing)]


def _enclosing_html_row(raw_html: str, start: int, end: int) -> str:
    row_start = raw_html.lower().rfind("<tr", 0, start)
    row_end = raw_html.lower().find("</tr>", end)
    if row_start < 0 or row_end < 0:
        return raw_html[max(0, start - 400) : min(len(raw_html), end + 400)]
    return raw_html[row_start : row_end + len("</tr>")]


def _write_exact_bbox_review_crop(
    *,
    page_asset: Path | None,
    normalized_bbox: tuple[float, float, float, float],
    destination: Path,
) -> Path:
    if page_asset is None or not page_asset.is_file():
        raise FileNotFoundError(f"Page asset is unavailable: {page_asset}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(page_asset) as source:
        width, height = source.size
        x1, y1, x2, y2 = normalized_bbox
        box = (
            max(0, min(width - 1, round(x1 * width))),
            max(0, min(height - 1, round(y1 * height))),
            max(1, min(width, round(x2 * width))),
            max(1, min(height, round(y2 * height))),
        )
        source.crop(box).save(destination)
    return destination


def _write_normalized_crop(
    *,
    source_asset: Path,
    normalized_bbox: tuple[float, float, float, float],
    destination: Path,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_asset) as source:
        width, height = source.size
        x1, y1, x2, y2 = normalized_bbox
        box = (
            max(0, min(width - 1, round(x1 * width))),
            max(0, min(height - 1, round(y1 * height))),
            max(1, min(width, round(x2 * width))),
            max(1, min(height, round(y2 * height))),
        )
        source.crop(box).save(destination)
    return destination


def _write_risk_contact_sheets(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    risky = [row for row in rows if row["risk_reasons"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    tile_width, tile_height = 520, 380
    image_height = 325
    columns, page_size = 3, 12
    for sheet_index in range(0, len(risky), page_size):
        batch = risky[sheet_index : sheet_index + page_size]
        rows_count = math.ceil(len(batch) / columns)
        sheet = Image.new("RGB", (columns * tile_width, rows_count * tile_height), "white")
        draw = ImageDraw.Draw(sheet)
        for tile_index, row in enumerate(batch):
            x = (tile_index % columns) * tile_width
            y = (tile_index // columns) * tile_height
            asset_path = Path(row["table_asset"]) if row["table_asset"] else None
            if asset_path and asset_path.is_file():
                with Image.open(asset_path) as source:
                    source = source.convert("RGB")
                    source.thumbnail((tile_width - 20, image_height - 10))
                    sheet.paste(source, (x + 10, y + 5))
            label = f"{sheet_index + tile_index + 1}: {row['document']} p{row['page_number']}"
            reason = ", ".join(row["risk_reasons"])
            metrics = f"grid={row['row_count']}x{row['column_count']} exact={row['exact_cell_match_ratio']} token={row['mean_cell_token_coverage']}"
            draw.text((x + 8, y + image_height), label[:82], fill="black", font=font)
            draw.text((x + 8, y + image_height + 15), reason[:82], fill="#b91c1c", font=font)
            draw.text((x + 8, y + image_height + 30), metrics[:82], fill="black", font=font)
        sheet.save(output_dir / f"risk_tables_{sheet_index // page_size + 1:02d}.jpg", quality=88)


def write_possible_miss_contact_sheets(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Write cropped CV candidates for human diagnosis (not called by default)."""

    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    tile_width, tile_height = 520, 380
    image_height = 325
    columns, page_size = 3, 12
    for sheet_index in range(0, len(rows), page_size):
        batch = rows[sheet_index : sheet_index + page_size]
        rows_count = math.ceil(len(batch) / columns)
        sheet = Image.new("RGB", (columns * tile_width, rows_count * tile_height), "white")
        draw = ImageDraw.Draw(sheet)
        for tile_index, row in enumerate(batch):
            x = (tile_index % columns) * tile_width
            y = (tile_index // columns) * tile_height
            page_asset = Path(row["page_asset"])
            if page_asset.is_file():
                with Image.open(page_asset) as source:
                    source = source.convert("RGB")
                    page_width, page_height = source.size
                    # Candidate coordinates use the PDF page coordinate scale;
                    # normalized ratios remain the same in the rendered page.
                    bbox = row["bbox"]
                    document_width = row.get("page_width") or page_width
                    document_height = row.get("page_height") or page_height
                    crop_box = (
                        int(bbox[0] / document_width * page_width),
                        int(bbox[1] / document_height * page_height),
                        int(bbox[2] / document_width * page_width),
                        int(bbox[3] / document_height * page_height),
                    )
                    source = source.crop(crop_box)
                    source.thumbnail((tile_width - 20, image_height - 10))
                    sheet.paste(source, (x + 10, y + 5))
            label = f"{sheet_index + tile_index + 1}: {row['document']} p{row['page_number']}"
            draw.text((x + 8, y + image_height), label[:82], fill="black", font=font)
            draw.text((x + 8, y + image_height + 15), str(row["bbox"])[:82], fill="#b91c1c", font=font)
        sheet.save(output_dir / f"possible_misses_{sheet_index // page_size + 1:02d}.jpg", quality=88)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/representative_28"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/representative_28/table_audit"))
    parser.add_argument(
        "--embedded-image-review-output",
        type=Path,
        help="Only generate the focused review for tables containing HTML images.",
    )
    parser.add_argument(
        "--unresolved-visual-review-output",
        type=Path,
        help="Only review tables with visual evidence but no HTML img src.",
    )
    parser.add_argument(
        "--type-error-img-mapping-output",
        type=Path,
        help="Map each img src in the type-error candidates to its exact HTML cell.",
    )
    parser.add_argument(
        "--scan-visual-misses",
        action="store_true",
        help=(
            "Run an experimental ruled-grid detector. Results are noisy diagnostic "
            "candidates, not confirmed missed tables."
        ),
    )
    args = parser.parse_args()
    if args.embedded_image_review_output is not None:
        summary = write_embedded_image_table_review(
            args.corpus,
            args.embedded_image_review_output,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.unresolved_visual_review_output is not None:
        summary = write_unresolved_visual_table_review(
            args.corpus,
            args.unresolved_visual_review_output,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.type_error_img_mapping_output is not None:
        summary = write_type_error_embedded_image_mapping_review(
            args.corpus,
            args.type_error_img_mapping_output,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    payload = audit(
        args.corpus,
        args.output,
        scan_visual_misses=args.scan_visual_misses,
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
