"""Rebuild the 14-document representative corpus with the current pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from softdoc.adapters import MinerUAdapter
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
    for document_dir in sorted(
        path for path in input_root.iterdir() if path.is_dir()
    ):
        artifact_dir = (
            document_dir / "auto"
            if (document_dir / "auto").is_dir()
            else document_dir
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
        document = adapter.parse(artifact_dir, destination)
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
        write_document(
            document,
            destination,
            render_overlays=not args.no_overlays,
        )
        validation_errors = DocumentStore(document).validate_references()
        rows.append(
            {
                "document": document_dir.name,
                "pages": len(document.pages),
                "elements": len(document.elements),
                "sections": len(document.sections),
                "relations": len(document.relations),
                "warnings": len(adapter.warnings),
                "validation_errors": validation_errors,
                "profile": document.metadata.get(
                    "document_profile", {}
                ).get("profile"),
                "title": document.title,
            }
        )
        print(
            f"{document_dir.name}: pages={len(document.pages)} "
            f"elements={len(document.elements)} "
            f"errors={len(validation_errors)}"
        )

    summary = {
        "documents": len(rows),
        "successful": sum(not row["validation_errors"] for row in rows),
        "pages": sum(int(row["pages"]) for row in rows),
        "elements": sum(int(row["elements"]) for row in rows),
        "rows": rows,
    }
    (output_root / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["successful"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
