"""Summarize a fresh Stage 7 model batch without inspecting gold answers."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "batch_manifest.json")
    status_counts: Counter[str] = Counter()
    diagnostic_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    ready_by_step: Counter[int] = Counter()
    cases_by_status: dict[str, list[str]] = {}
    cases_by_diagnostic: dict[str, list[str]] = {}
    elapsed: list[float] = []
    model_elapsed: list[float] = []
    descriptor_calls = 0
    descriptor_errors = 0
    placeholder_case_ids: list[str] = []
    incomplete_case_summaries: list[dict[str, Any]] = []

    for case in manifest.get("cases", []):
        if case.get("status") != "succeeded":
            continue
        case_id = case["case_id"]
        case_dir = root / case_id
        run = _read_json(case_dir / "reading_run.json")
        trace = _read_json(case_dir / "action_trace.json").get("entries", [])
        run_manifest = _read_json(case_dir / "run_manifest.json")
        status = run["status"]
        status_counts[status] += 1
        cases_by_status.setdefault(status, []).append(case_id)
        elapsed.append(float(case.get("elapsed_seconds") or 0.0))
        model_elapsed.append(float(run_manifest.get("model_call_elapsed_seconds") or 0.0))
        for entry in trace:
            action_counts[str(entry.get("action_name") or "unknown")] += 1
        for diagnostic in run.get("diagnostics", []):
            code = str(diagnostic.get("code") or "unknown")
            diagnostic_counts[code] += 1
            cases_by_diagnostic.setdefault(code, []).append(case_id)
        if status == "ready":
            ready_by_step[len(trace)] += 1
        else:
            incomplete_case_summaries.append(
                {
                    "case_id": case_id,
                    "status": status,
                    "action_count": len(trace),
                    "last_action": (
                        trace[-1].get("action_name") if trace else None
                    ),
                    "diagnostic_codes": [
                        item.get("code") for item in run.get("diagnostics", [])
                    ],
                    "evidence_count": len(
                        run.get("evidence_memory", {}).get("evidence", [])
                    ),
                    "observation_count": len(
                        run.get("observation_store", {}).get("observations", [])
                    ),
                }
            )
        calls = _read_jsonl(case_dir / "visual_descriptor_calls.jsonl")
        descriptor_calls += len(calls)
        descriptor_errors += sum(bool(item.get("error")) for item in calls)
        batches = case_dir / "candidate_batches.jsonl"
        if batches.exists() and "candidate matched from visual content" in batches.read_text(
            encoding="utf-8"
        ):
            placeholder_case_ids.append(case_id)

    total_ready = sum(ready_by_step.values())
    ready_at_most_7 = sum(value for step, value in ready_by_step.items() if step <= 7)
    ready_8_to_12 = sum(value for step, value in ready_by_step.items() if 8 <= step <= 12)
    ready_13_to_16 = sum(value for step, value in ready_by_step.items() if 13 <= step <= 16)
    processed = sum(status_counts.values())
    result = {
        "schema_version": "stage7-fresh-summary-v0.1",
        "root": str(root),
        "batch": {
            "status": manifest.get("status"),
            "case_count": manifest.get("case_count"),
            "succeeded": manifest.get("succeeded"),
            "failed": manifest.get("failed"),
            "processed_reading_runs": processed,
        },
        "reading_status_counts": dict(status_counts),
        "ready_curve": {
            "ready_total": total_ready,
            "ready_at_most_step_7": ready_at_most_7,
            "new_ready_steps_8_to_12": ready_8_to_12,
            "new_ready_steps_13_to_16": ready_13_to_16,
            "first_ready_by_step": {
                str(step): ready_by_step[step] for step in sorted(ready_by_step)
            },
        },
        "efficiency": {
            "elapsed_seconds_mean": round(statistics.mean(elapsed), 3) if elapsed else None,
            "elapsed_seconds_median": round(statistics.median(elapsed), 3) if elapsed else None,
            "elapsed_seconds_max": round(max(elapsed), 3) if elapsed else None,
            "model_call_seconds_mean": (
                round(statistics.mean(model_elapsed), 3) if model_elapsed else None
            ),
        },
        "action_counts": dict(action_counts.most_common()),
        "diagnostic_counts": dict(diagnostic_counts.most_common()),
        "cases_by_diagnostic": {
            key: sorted(set(value)) for key, value in cases_by_diagnostic.items()
        },
        "visual_descriptors": {
            "live_call_count": descriptor_calls,
            "error_count": descriptor_errors,
            "placeholder_case_count": len(placeholder_case_ids),
            "placeholder_case_ids": sorted(placeholder_case_ids),
        },
        "cases_by_status": {
            key: sorted(value) for key, value in cases_by_status.items()
        },
        "incomplete_cases": incomplete_case_summaries,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.root.resolve())
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
