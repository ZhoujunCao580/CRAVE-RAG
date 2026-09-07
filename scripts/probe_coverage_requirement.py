"""Resolve one coverage requirement against a persisted SoftDoc.

This is intentionally retrieval- and model-free.  It is the audit/recovery
entrypoint for checking printed-page versus physical-page interpretations
before a coverage run is allowed to claim completeness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from softdoc.coverage_reasoning import (
    CoverageOperator,
    CoverageRequirement,
    CoverageScopeResolver,
    CoverageSourceType,
    PageNumberNamespace,
    QuestionCoveragePlan,
    build_coverage_inventory,
)
from softdoc.serialization import load_document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document-dir", type=Path, required=True)
    parser.add_argument("--question-id", required=True)
    parser.add_argument(
        "--operator",
        choices=[item.value for item in CoverageOperator],
        required=True,
    )
    parser.add_argument("--scope-text", required=True)
    parser.add_argument("--item-type", required=True)
    parser.add_argument(
        "--source-type",
        choices=[item.value for item in CoverageSourceType],
        required=True,
    )
    parser.add_argument("--predicate")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--page-namespace",
        choices=[
            PageNumberNamespace.PRINTED_PAGE_LABEL.value,
            PageNumberNamespace.PHYSICAL_PAGE_ORDER.value,
        ],
        help="Explicit audit correction; omitted means use the deterministic resolver.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = load_document(args.document_dir)
    requirement = CoverageRequirement(
        operator=args.operator,
        scope_text=args.scope_text,
        item_type=args.item_type,
        source_type=args.source_type,
        predicate=args.predicate,
        limit=args.limit,
    )
    namespace = (
        PageNumberNamespace(args.page_namespace)
        if args.page_namespace is not None
        else None
    )
    resolution = CoverageScopeResolver().resolve(
        requirement,
        document,
        namespace_override=namespace,
    )
    plan = QuestionCoveragePlan(
        question_id=args.question_id,
        requirement=requirement,
        scope_resolution=resolution,
        inventory=build_coverage_inventory(requirement, resolution, document),
    )
    payload = plan.model_dump(mode="json")
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
