"""Build a representative corpus and record SoftDoc conversion timings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from softdoc.adapters import MinerUAdapter
from softdoc.models import Document
from softdoc.pipeline import SoftDocPipeline
from softdoc.rule_audit import write_rule_coverage_reports
from softdoc.serialization import load_document, write_document
from softdoc.store import DocumentStore


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "data" / "processed" / "representative_14" / "mineru"
)
DEFAULT_OUTPUT = (
    ROOT / "data" / "processed" / "representative_14" / "softdoc_final"
)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-overlays",
        action="store_true",
        help="Skip debug overlay rendering.",
    )
    args = parser.parse_args(argv)

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    built_documents: list[Document] = []
    for document_dir in sorted(
        path for path in input_root.iterdir() if path.is_dir()
    ):
        document_started = time.perf_counter()
        artifact_dir = next(
            (
                candidate
                for candidate in (
                    document_dir / "hybrid_auto",
                    document_dir / "auto",
                    document_dir,
                )
                if candidate.is_dir()
                and any(candidate.glob("*_content_list_v2.json"))
            ),
            document_dir,
        )
        destination = output_root / document_dir.name
        reusable_page_assets: dict[int, Path] = {}
        if (destination / "document.json").is_file():
            previous = load_document(destination)
            for page in previous.pages:
                if page.image_path is None:
                    continue
                asset = destination / page.image_path
                if asset.is_file():
                    reusable_page_assets[page.page_index] = (
                        page.image_path
                    )
        adapter = MinerUAdapter()
        pipeline_started = time.perf_counter()
        pipeline_result = SoftDocPipeline(adapter).run(
            artifact_dir,
            destination,
        )
        pipeline_seconds = time.perf_counter() - pipeline_started
        document = pipeline_result.document
        for page in document.pages:
            if page.image_path is None:
                page.image_path = reusable_page_assets.get(page.page_index)
        missing_page_images = [
            page.page_number
            for page in document.pages
            if page.image_path is None
            or not (
                page.image_path
                if page.image_path.is_absolute()
                else destination / page.image_path
            ).is_file()
        ]
        if missing_page_images and not args.no_overlays:
            raise RuntimeError(
                f"{document_dir.name}: cannot create trustworthy overlays; "
                f"page images are missing for pages {missing_page_images}. "
                "Run this script in the project environment with pypdfium2 "
                "installed, or use --no-overlays explicitly."
            )
        serialization_started = time.perf_counter()
        write_document(
            document,
            destination,
            render_overlays=not args.no_overlays,
        )
        serialization_seconds = time.perf_counter() - serialization_started
        validation_errors = DocumentStore(document).validate_references()
        rows.append(
            {
                "document": document_dir.name,
                "pages": len(document.pages),
                "elements": len(document.elements),
                "sections": len(document.sections),
                "relations": len(document.relations),
                "warnings": len(
                    document.metadata.get("adapter_warnings", [])
                ),
                "passes": [
                    report.model_dump(mode="json")
                    for report in pipeline_result.pass_reports
                ],
                "validation_errors": validation_errors,
                "profile": document.metadata.get(
                    "document_profile", {}
                ).get("profile"),
                "title": document.title,
                "pipeline_seconds": round(pipeline_seconds, 6),
                "serialization_seconds": round(serialization_seconds, 6),
                "total_seconds": round(
                    time.perf_counter() - document_started, 6
                ),
            }
        )
        print(
            f"{document_dir.name}: pages={len(document.pages)} "
            f"elements={len(document.elements)} "
            f"errors={len(validation_errors)}"
        )
        built_documents.append(document)

    summary = {
        "documents": len(rows),
        "successful": sum(not row["validation_errors"] for row in rows),
        "pages": sum(int(row["pages"]) for row in rows),
        "elements": sum(int(row["elements"]) for row in rows),
        "timing": {
            "pipeline_seconds": round(
                sum(float(row["pipeline_seconds"]) for row in rows), 3
            ),
            "serialization_seconds": round(
                sum(float(row["serialization_seconds"]) for row in rows), 3
            ),
            "total_seconds": round(
                sum(float(row["total_seconds"]) for row in rows), 3
            ),
            "average_seconds_per_document": round(
                sum(float(row["total_seconds"]) for row in rows) / len(rows), 3
            )
            if rows
            else 0.0,
        },
        "rows": rows,
    }
    (output_root / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_rule_coverage_reports(built_documents, output_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["successful"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
