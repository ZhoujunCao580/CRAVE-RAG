"""Validate raw component outputs without exposing reviewer-only Gold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from softdoc.answering import AnswerResult
from softdoc.controller import ControllerAction
from softdoc.planning.models import PlannerDraft
from softdoc.reading_state import EvidenceCheckDecision
from softdoc.visual_reading import VisualReadResult


OUTPUT_ADAPTERS: dict[str, TypeAdapter[Any]] = {
    "planner": TypeAdapter(PlannerDraft),
    "controller": TypeAdapter(ControllerAction),
    "visual_reader": TypeAdapter(VisualReadResult),
    "checker": TypeAdapter(EvidenceCheckDecision),
    "answerer": TypeAdapter(AnswerResult),
}


def validate_run(component: str, input_path: Path, output_dir: Path) -> dict[str, int]:
    adapter = OUTPUT_ADAPTERS[component]
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        input_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if set(record) != {"case_id", "raw_output"}:
            raise ValueError(
                f"Line {line_number} must contain exactly case_id and raw_output"
            )
        case_id = str(record["case_id"])
        if case_id in seen:
            raise ValueError(f"Duplicate case_id: {case_id}")
        seen.add(case_id)
        raw_output = record["raw_output"]
        try:
            payload = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
            parsed = adapter.validate_python(payload)
            records.append(
                {
                    "case_id": case_id,
                    "schema_valid": True,
                    "parsed_output": adapter.dump_python(parsed, mode="json"),
                    "error": None,
                }
            )
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            records.append(
                {
                    "case_id": case_id,
                    "schema_valid": False,
                    "parsed_output": None,
                    "error": str(exc),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "schema_validation.jsonl"
    result_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    valid_count = sum(bool(item["schema_valid"]) for item in records)
    summary = {"case_count": len(records), "schema_valid": valid_count}
    (output_dir / "schema_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=sorted(OUTPUT_ADAPTERS), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(validate_run(args.component, args.input, args.output))


if __name__ == "__main__":
    main()
