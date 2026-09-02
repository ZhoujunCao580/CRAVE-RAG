"""Validate a frozen evaluation protocol and create one immutable run snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from softdoc.evaluation_protocol import (
    build_experiment_snapshot,
    load_evaluation_protocol,
    protocol_sha256,
    write_experiment_snapshot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--bindings",
        type=Path,
        help="JSON object containing every required experiment binding",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = load_evaluation_protocol(args.protocol)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "valid",
                    "protocol_id": protocol.protocol_id,
                    "protocol_sha256": protocol_sha256(protocol),
                    "benchmark_role": protocol.benchmark.dataset_role,
                    "clean_holdout": protocol.benchmark.clean_holdout,
                    "document_count": protocol.benchmark.document_count,
                    "question_count": protocol.benchmark.question_count,
                },
                indent=2,
            )
        )
        return 0
    if args.bindings is None or args.output_root is None:
        raise SystemExit("snapshot creation requires --bindings and --output-root")
    bindings = json.loads(args.bindings.read_text(encoding="utf-8"))
    if not isinstance(bindings, dict):
        raise SystemExit("--bindings must contain one JSON object")
    snapshot = build_experiment_snapshot(protocol, bindings)
    output = write_experiment_snapshot(snapshot, args.output_root)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
