"""Summarize persisted Stage 3-7 server runs without judging Gold answers."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from typing import Any, Iterable


VISUAL_TYPES = {"figure", "chart", "table"}
VISUAL_PLACEHOLDER_SUFFIX = "candidate matched from visual content"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return []
    return (
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return round(ordered[index], 3)


def _timing(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 3) if values else None,
        "median": round(statistics.median(values), 3) if values else None,
        "p90": _percentile(values, 0.9),
        "max": round(max(values), 3) if values else None,
    }


def _case_metrics(output_dir: Path) -> dict[str, Any]:
    run = _load_json(output_dir / "reading_run.json")
    manifest = _load_json(output_dir / "run_manifest.json")
    entries = run.get("action_trace", {}).get("entries", [])
    action_counts = Counter(item.get("action_name") for item in entries)

    state_only_calls = 0
    recalled_observations = 0
    reused_evidence = 0
    checker_validation_attempts: list[int] = []
    checker_repair_calls = 0
    checker_finish_reasons: Counter[str] = Counter()
    for call in _jsonl(output_dir / "checker_calls.jsonl"):
        checker_repair_calls += bool(
            (call.get("metadata") or {}).get("rejected_attempts")
        )
        payload = call.get("input") or {}
        observations = payload.get("observations") or []
        limitations = payload.get("limitations") or []
        recalled = payload.get("recalled_observations") or []
        if not observations and not limitations:
            state_only_calls += 1
        recalled_observations += len(recalled)
        result = call.get("output") or {}
        reused_evidence += len(result.get("reused_evidence_ids") or [])
        trace = result.get("checker_trace") or {}
        metadata = trace.get("metadata") or {}
        if isinstance(metadata.get("validation_attempts"), int):
            checker_validation_attempts.append(metadata["validation_attempts"])
        finish_reason = metadata.get("finish_reason")
        if finish_reason:
            checker_finish_reasons[str(finish_reason)] += 1

    controller_validation_attempts: list[int] = []
    controller_repair_calls = 0
    for call in _jsonl(output_dir / "controller_calls.jsonl"):
        controller_repair_calls += bool(
            (call.get("metadata") or {}).get("rejected_attempts")
            or ((call.get("metadata") or {}).get("generation_metadata") or {}).get(
                "rejected_attempts"
            )
        )
        result = call.get("output") or {}
        trace = result.get("controller_trace") or {}
        metadata = trace.get("metadata") or {}
        if isinstance(metadata.get("validation_attempts"), int):
            controller_validation_attempts.append(metadata["validation_attempts"])

    answerer_repair_calls = sum(
        bool((call.get("metadata") or {}).get("rejected_attempts"))
        for call in _jsonl(output_dir / "answerer_calls.jsonl")
    )

    visual_candidates: dict[str, dict[str, Any]] = {}
    for batch in _jsonl(output_dir / "candidate_batches.jsonl"):
        view = batch.get("visible_search_view") or {}
        for item in view.get("candidate_previews") or []:
            if item.get("element_type") not in VISUAL_TYPES:
                continue
            visual_candidates.setdefault(item.get("element_id", ""), item)
    placeholders = [
        item
        for item in visual_candidates.values()
        if not str(item.get("matched_snippet") or "").strip()
        or str(item.get("matched_snippet") or "")
        .strip()
        .casefold()
        .endswith(VISUAL_PLACEHOLDER_SUFFIX)
    ]

    required = {
        "planner": "planner_calls.jsonl",
        "controller": "controller_calls.jsonl",
        "reader": "reader_calls.jsonl",
        "checker": "checker_calls.jsonl",
        "answerer": "answerer_calls.jsonl",
        "coverage_checker": "coverage_checker_calls.jsonl",
        "candidates": "candidate_batches.jsonl",
        "actions": "action_trace.json",
        "evidence_deltas": "evidence_deltas.jsonl",
    }
    missing_files = [name for name, filename in required.items() if not (output_dir / filename).is_file()]
    coverage_plans = run.get("coverage_plans") or []
    return {
        "run_status": run.get("status"),
        "action_count": len(entries),
        "action_counts": dict(sorted(action_counts.items())),
        "call_counts": manifest.get("call_counts") or {},
        "model_call_elapsed_seconds": manifest.get("model_call_elapsed_seconds"),
        "state_only_checker_calls": state_only_calls,
        "recalled_observations": recalled_observations,
        "reused_evidence_ids": reused_evidence,
        "checker_repair_calls": checker_repair_calls
        + sum(value > 1 for value in checker_validation_attempts),
        "controller_repair_calls": controller_repair_calls
        + sum(value > 1 for value in controller_validation_attempts),
        "answerer_repair_calls": answerer_repair_calls,
        "checker_finish_reasons": dict(checker_finish_reasons),
        "coverage_plan_count": len(coverage_plans),
        "coverage_statuses": dict(
            Counter(
                ((item.get("execution") or {}).get("status") or (item.get("inventory") or {}).get("status") or "unknown")
                for item in coverage_plans
            )
        ),
        "visual_candidate_count": len(visual_candidates),
        "visual_placeholder_count": len(placeholders),
        "visual_placeholder_ids": [item.get("element_id") for item in placeholders],
        "missing_artifact_files": missing_files,
    }


def _aggregate_cases(rows: list[dict[str, Any]]) -> dict[str, Any]:
    succeeded = [row for row in rows if row.get("batch_status") == "succeeded"]
    elapsed = [float(row["elapsed_seconds"]) for row in succeeded if row.get("elapsed_seconds") is not None]
    model_elapsed = [
        float(row["model_call_elapsed_seconds"])
        for row in succeeded
        if row.get("model_call_elapsed_seconds") is not None
    ]
    return {
        "cases": len(rows),
        "batch_status_counts": dict(Counter(row.get("batch_status") for row in rows)),
        "run_status_counts": dict(Counter(row.get("run_status") for row in succeeded)),
        "elapsed_seconds": _timing(elapsed),
        "model_call_elapsed_seconds": _timing(model_elapsed),
        "peak_gpu_memory_mib": max(
            (row.get("peak_gpu_memory_mib") or 0 for row in rows), default=0
        ),
        "total_actions": sum(row.get("action_count") or 0 for row in succeeded),
        "state_only_checker_calls": sum(row.get("state_only_checker_calls") or 0 for row in succeeded),
        "recalled_observations": sum(row.get("recalled_observations") or 0 for row in succeeded),
        "reused_evidence_ids": sum(row.get("reused_evidence_ids") or 0 for row in succeeded),
        "checker_repair_calls": sum(row.get("checker_repair_calls") or 0 for row in succeeded),
        "controller_repair_calls": sum(row.get("controller_repair_calls") or 0 for row in succeeded),
        "answerer_repair_calls": sum(row.get("answerer_repair_calls") or 0 for row in succeeded),
        "coverage_plan_cases": sum(bool(row.get("coverage_plan_count")) for row in succeeded),
        "visual_candidates": sum(row.get("visual_candidate_count") or 0 for row in succeeded),
        "visual_placeholders": sum(row.get("visual_placeholder_count") or 0 for row in succeeded),
        "cases_with_visual_placeholders": sum(bool(row.get("visual_placeholder_count")) for row in succeeded),
        "cases_with_missing_artifacts": sum(bool(row.get("missing_artifact_files")) for row in succeeded),
    }


def summarize_stage36(
    root: Path,
    groups_path: Path,
    supplements: list[Path],
) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    manifests = []
    for attempt_root in [root, *supplements]:
        manifest = _load_json(attempt_root / "outputs" / "batch_manifest.json")
        manifests.append(
            {
                "root": str(attempt_root),
                **{
                    key: manifest.get(key)
                    for key in (
                        "status",
                        "case_count",
                        "succeeded",
                        "failed",
                        "started_at",
                        "finished_at",
                    )
                },
            }
        )
        for item in manifest.get("cases", []):
            row = {
                "case_id": item.get("case_id"),
                "batch_status": item.get("status"),
                "elapsed_seconds": item.get("elapsed_seconds"),
                "peak_gpu_memory_mib": item.get("peak_gpu_memory_mib"),
                "error": item.get("error"),
                "attempt_root": str(attempt_root),
            }
            output_dir = Path(
                item.get("output_dir")
                or attempt_root / "outputs" / str(item.get("case_id"))
            )
            if item.get("status") == "succeeded" and (
                output_dir / "reading_run.json"
            ).is_file():
                row.update(_case_metrics(output_dir))
            previous = rows.get(str(item.get("case_id")))
            if previous is None or row["batch_status"] == "succeeded":
                rows[str(item.get("case_id"))] = row

    groups_payload = _load_json(groups_path)
    groups = groups_payload["groups"]
    grouped: dict[str, Any] = {}
    for stage_name in ("stage3", "stage4"):
        grouped[stage_name] = {
            name: _aggregate_cases([rows[qid] for qid in case_ids if qid in rows])
            for name, case_ids in groups[stage_name].items()
        }
    grouped["stage5"] = _aggregate_cases(
        [rows[qid] for qid in groups["stage5"] if qid in rows]
    )
    grouped["stage6"] = {
        name: _aggregate_cases([rows[qid] for qid in case_ids if qid in rows])
        for name, case_ids in groups["stage6"].items()
    }
    return {
        "manifests": manifests,
        "selection": {key: groups_payload.get(key) for key in ("requested_unique", "selected_unique", "missing")},
        "overall": _aggregate_cases(list(rows.values())),
        "groups": grouped,
        "failures": [row for row in rows.values() if row.get("batch_status") != "succeeded"],
        "cases": rows,
    }


def summarize_stage7(root: Path, supplements: list[Path]) -> dict[str, Any]:
    manifests = []
    by_case: dict[str, dict[str, Any]] = {}
    for attempt_root in [root, *supplements]:
        manifest = _load_json(attempt_root / "outputs" / "resume_manifest.json")
        manifests.append(
            {
                "root": str(attempt_root),
                **{
                    key: manifest.get(key)
                    for key in (
                        "status",
                        "case_count",
                        "completed",
                        "succeeded",
                        "failed",
                        "started_at",
                        "finished_at",
                        "total_step_limit",
                        "workers",
                    )
                },
            }
        )
        for row in manifest.get("cases", []):
            previous = by_case.get(str(row.get("case_id")))
            if previous is None or row.get("status") == "succeeded":
                by_case[str(row.get("case_id"))] = row
    rows = list(by_case.values())
    succeeded = [row for row in rows if row.get("status") == "succeeded"]
    histogram = Counter(
        str(row.get("first_ready_step")) if row.get("first_ready_step") is not None else "never"
        for row in succeeded
    )
    return {
        "manifests": manifests,
        "unique_cases": len(rows),
        "first_ready_step": dict(sorted(histogram.items())),
        "new_status_counts": dict(Counter(row.get("new_status") for row in succeeded)),
        "additional_elapsed_seconds": _timing(
            [float(row["additional_elapsed_seconds"]) for row in succeeded]
        ),
        "failures": [row for row in rows if row.get("status") != "succeeded"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage36-root", type=Path, required=True)
    parser.add_argument("--stage7-root", type=Path, required=True)
    parser.add_argument("--stage36-supplement", type=Path, action="append", default=[])
    parser.add_argument("--stage7-supplement", type=Path, action="append", default=[])
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema_version": "stage1-to7-experiment-summary-v0.1",
        "stage3_to6": summarize_stage36(
            args.stage36_root,
            args.groups,
            args.stage36_supplement,
        ),
        "stage7": summarize_stage7(args.stage7_root, args.stage7_supplement),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "stage36_manifests": payload["stage3_to6"]["manifests"],
                "stage7_manifests": payload["stage7"]["manifests"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
