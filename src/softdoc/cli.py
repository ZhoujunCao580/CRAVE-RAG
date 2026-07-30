"""Command-line interface for milestone-one parsing and validation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from softdoc.adapters import MinerUAdapter
from softdoc.pipeline import SoftDocPipeline
from softdoc.rule_audit import write_rule_coverage_reports
from softdoc.serialization import load_document, write_document
from softdoc.store import DocumentStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="softdoc", description="Soft document structure utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_mineru = subparsers.add_parser("parse-mineru", help="Convert a MinerU output directory")
    parse_mineru.add_argument("input_dir", type=Path)
    parse_mineru.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser("validate", help="Validate a serialized soft document")
    validate.add_argument("output_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    if args.command == "parse-mineru":
        adapter = MinerUAdapter()
        document = SoftDocPipeline(adapter).parse(
            args.input_dir,
            args.output,
        )
        write_document(document, args.output)
        write_rule_coverage_reports([document], args.output)
        print(
            f"Parsed {document.document_id}: {len(document.pages)} pages, "
            f"{len(document.elements)} elements, {len(document.relations)} relations"
        )
        warning_count = len(document.metadata.get("adapter_warnings", []))
        if warning_count:
            print(
                f"Recorded {warning_count} adapter warnings in "
                "debug/adapter_warnings.json"
            )
        return 0
    if args.command == "validate":
        document = load_document(args.output_dir)
        errors = DocumentStore(document).validate_references()
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            return 1
        print(
            f"Valid {document.document_id}: {len(document.pages)} pages, "
            f"{len(document.elements)} elements, {len(document.relations)} relations"
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
