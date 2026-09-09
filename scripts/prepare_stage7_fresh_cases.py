"""Freeze the old Stage-7 case IDs into a fresh-from-step-zero case manifest.

Only the case selection is reused. No checkpoint, action, observation, evidence,
search cursor, or model output is copied into the new experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def prepare(
    *,
    all_cases_path: Path,
    old_manifests: list[Path],
    output_path: Path,
    expected_count: int,
) -> dict[str, Any]:
    all_cases = _jsonl(all_cases_path)
    by_id = {str(row["case_id"]): row for row in all_cases}
    selected_ids: list[str] = []
    for manifest_path in old_manifests:
        for row in _json(manifest_path).get("cases", []):
            case_id = str(row["case_id"])
            if case_id not in selected_ids:
                selected_ids.append(case_id)
    if len(selected_ids) != expected_count:
        raise ValueError(
            f"Expected {expected_count} selected cases, found {len(selected_ids)}"
        )
    missing = sorted(set(selected_ids) - by_id.keys())
    if missing:
        raise ValueError(f"Selected cases missing from all-cases manifest: {missing}")

    selected = [by_id[case_id] for case_id in selected_ids]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in selected
        ),
        encoding="utf-8",
    )
    selection = {
        "schema_version": "stage7-fresh-selection-v0.1",
        "selection_only_from_old_runs": True,
        "restored_runtime_state": False,
        "case_count": len(selected),
        "case_ids": selected_ids,
        "source_all_cases": str(all_cases_path),
        "source_selection_manifests": [str(path) for path in old_manifests],
        "output_cases": str(output_path),
    }
    selection_path = output_path.with_suffix(".selection.json")
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return selection


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-cases", type=Path, required=True)
    parser.add_argument(
        "--old-manifest", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=199)
    args = parser.parse_args(argv)
    result = prepare(
        all_cases_path=args.all_cases,
        old_manifests=args.old_manifest,
        output_path=args.output,
        expected_count=args.expected_count,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
