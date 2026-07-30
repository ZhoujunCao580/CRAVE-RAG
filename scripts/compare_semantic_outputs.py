"""Compare two 14-document SoftDoc output roots semantically."""

from __future__ import annotations

import argparse
from pathlib import Path

from softdoc.semantic_diff import (
    compare_softdoc_roots,
    write_semantic_diff_reports,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_root", type=Path)
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_softdoc_roots(
        args.baseline_root,
        args.candidate_root,
    )
    write_semantic_diff_reports(report, args.output)
    print(
        f"Compared {report.document_count} documents: "
        f"equal={report.equal_document_count}, "
        f"changed={report.changed_document_count}"
    )
    return 0 if report.semantically_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
