"""Summarize cold PDF-to-retrieval throughput without mixing online QA time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import pypdfium2 as pdfium


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "representative_28.json"
DEFAULT_PDF_ROOT = ROOT / "data" / "raw" / "mmlongbench_doc" / "documents"
DEFAULT_SOFTDOC_SUMMARY = (
    ROOT
    / "data"
    / "processed"
    / "representative_28_extension"
    / "softdoc_final"
    / "build_summary.json"
)
DEFAULT_RETRIEVAL_SUMMARY = (
    ROOT
    / "data"
    / "processed"
    / "representative_28_extension"
    / "retrieval_dense_e5"
    / "run_summary.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "representative_28_extension"
    / "pipeline_performance"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mineru-seconds", type=float, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pdf-root", type=Path, default=DEFAULT_PDF_ROOT)
    parser.add_argument(
        "--softdoc-summary", type=Path, default=DEFAULT_SOFTDOC_SUMMARY
    )
    parser.add_argument(
        "--retrieval-summary", type=Path, default=DEFAULT_RETRIEVAL_SUMMARY
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = _load_json(args.manifest)
    softdoc = _load_json(args.softdoc_summary)
    retrieval = _load_json(args.retrieval_summary)
    documents = [str(row["doc_id"]) for row in manifest["extension_documents"]]
    pdf_rows = [_pdf_info(args.pdf_root / name) for name in documents]
    document_count = len(pdf_rows)
    page_count = sum(int(row["pages"]) for row in pdf_rows)

    softdoc_seconds = float(softdoc["timing"]["total_seconds"])
    encoder_load_seconds = float(retrieval["encoder_load_seconds"])
    retrieval_seconds = float(retrieval["total_seconds"])
    offline_seconds = (
        args.mineru_seconds
        + softdoc_seconds
        + encoder_load_seconds
        + retrieval_seconds
    )
    report = {
        "scope": (
            "Cold offline preparation of the 14-document extension: raw PDF "
            "through MinerU, SoftDoc serialization without overlays, "
            "SearchUnit/BM25 construction, and E5 Dense indexing. Download "
            "time and question answering are excluded."
        ),
        "corpus": manifest["name"],
        "extension_documents": document_count,
        "extension_pages": page_count,
        "extension_pdf_bytes": sum(int(row["bytes"]) for row in pdf_rows),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "dense_device": retrieval["device"],
            "dense_model": retrieval["model"]["model_name"],
        },
        "stage_seconds": {
            "mineru_batch": round(args.mineru_seconds, 3),
            "softdoc_and_serialization": round(softdoc_seconds, 3),
            "dense_model_load_shared": round(encoder_load_seconds, 3),
            "search_units_bm25_dense_index": round(retrieval_seconds, 3),
            "offline_total": round(offline_seconds, 3),
        },
        "throughput": {
            "offline_average_seconds_per_document": round(
                offline_seconds / document_count, 3
            ),
            "offline_average_seconds_per_page": round(
                offline_seconds / page_count, 3
            ),
            "mineru_average_seconds_per_document": round(
                args.mineru_seconds / document_count, 3
            ),
            "mineru_average_seconds_per_page": round(
                args.mineru_seconds / page_count, 3
            ),
            "softdoc_average_seconds_per_document": round(
                softdoc_seconds / document_count, 3
            ),
            "retrieval_average_seconds_per_document": round(
                (encoder_load_seconds + retrieval_seconds) / document_count, 3
            ),
        },
        "retrieval_stage_breakdown_seconds": retrieval[
            "stage_totals_seconds"
        ],
        "documents": pdf_rows,
    }

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "pipeline_performance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "pipeline_performance.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.resolve().read_text(encoding="utf-8-sig"))


def _pdf_info(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    document = pdfium.PdfDocument(path)
    try:
        pages = len(document)
    finally:
        document.close()
    return {"document": path.name, "pages": pages, "bytes": path.stat().st_size}


def _markdown(report: dict[str, object]) -> str:
    stages = report["stage_seconds"]
    throughput = report["throughput"]
    breakdown = report["retrieval_stage_breakdown_seconds"]
    return "\n".join(
        [
            "# PDF-to-Retrieval Performance",
            "",
            str(report["scope"]),
            "",
            f"- Documents: {report['extension_documents']}",
            f"- Pages: {report['extension_pages']}",
            f"- Offline total: {stages['offline_total']:.3f} s",
            f"- Average/PDF: "
            f"{throughput['offline_average_seconds_per_document']:.3f} s",
            f"- Average/page: "
            f"{throughput['offline_average_seconds_per_page']:.3f} s",
            "",
            "| Stage | Seconds |",
            "|---|---:|",
            f"| MinerU batch | {stages['mineru_batch']:.3f} |",
            f"| SoftDoc + serialization | "
            f"{stages['softdoc_and_serialization']:.3f} |",
            f"| E5 model load (shared) | "
            f"{stages['dense_model_load_shared']:.3f} |",
            f"| SearchUnit + BM25 + Dense index | "
            f"{stages['search_units_bm25_dense_index']:.3f} |",
            "",
            "## Retrieval index breakdown",
            "",
            f"- SoftDoc JSON load: {breakdown['softdoc_load']} s",
            f"- SearchUnit build: {breakdown['search_unit_build']} s",
            f"- BM25 build: {breakdown['bm25_build']} s",
            f"- Dense index: {breakdown['dense_index']} s",
            "",
            "This is offline preparation time. It must not be compared directly "
            "with an online per-question latency that starts from a prebuilt index.",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
