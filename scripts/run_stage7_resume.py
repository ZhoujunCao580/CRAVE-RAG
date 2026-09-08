"""Resume frozen step-7 budget checkpoints without replaying earlier actions.

This is intentionally a narrow experiment runner.  It loads the complete
model-pipeline packet for every old ``budget_exhausted`` case, restores the
canonical ReadingEnvironment state, and permits actions only until total step
12.  Outputs are written to a new directory and the old baseline is read-only.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
import traceback

from scripts.run_model_batch import _PersistentRuntime
from softdoc.model_runner import (
    ModelPipelineRun,
    _RecordingAnswerer,
    _RecordingChecker,
    _RecordingController,
    _RecordingReader,
    _bind_controller_action_ids,
    load_model_pipeline_run,
    write_model_pipeline_run,
)
from softdoc.prompt_registry import prompt_manifest
from softdoc.reading_environment import ReadingEnvironment, ReadingEnvironmentConfig


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _case_map(path: Path) -> dict[str, dict]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[row["case_id"]] = row
    return result


def _budget_case_ids(old_root: Path) -> list[str]:
    result = []
    for path in old_root.glob("Q*/reading_run.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") == "budget_exhausted":
            result.append(path.parent.name)
    return sorted(result, key=lambda value: int(value[1:]))


def _resume_one(
    *,
    case_id: str,
    case: dict,
    old_root: Path,
    output_root: Path,
    runtime: _PersistentRuntime,
    total_step_limit: int,
) -> dict:
    started = time.perf_counter()
    old = load_model_pipeline_run(old_root / case_id)
    old_steps = len(old.reading_run.action_trace.entries)
    additional = total_step_limit - old_steps
    if additional < 1:
        raise ValueError(
            f"{case_id}: checkpoint already has {old_steps} actions, "
            f"not below limit {total_step_limit}"
        )
    document, search_service = runtime._search_resources(case["document_dir"])
    runner = runtime._runner()
    records = list(old.stage_calls)
    environment = ReadingEnvironment(
        document,
        asset_root=Path(case["document_dir"]),
        controller=_RecordingController(runner.controller, records),
        reader=_RecordingReader(runner.reader, records),
        checker=_RecordingChecker(runner.checker, records),
        answerer=_RecordingAnswerer(runner.answerer, records),
        search_service=search_service,
        config=ReadingEnvironmentConfig(action_budget=total_step_limit),
    )
    resumed = environment.resume(
        old.reading_run,
        additional_action_budget=additional,
    )
    _bind_controller_action_ids(records, resumed.action_trace)
    combined = ModelPipelineRun(
        document_id=old.document_id,
        question=old.question,
        plan=old.plan,
        reading_run=resumed,
        prompt_bindings=prompt_manifest(),
        stage_calls=records,
    )
    write_model_pipeline_run(combined, output_root / case_id)
    final_steps = len(resumed.action_trace.entries)
    return {
        "case_id": case_id,
        "status": "succeeded",
        "old_status": old.reading_run.status.value,
        "new_status": resumed.status.value,
        "old_steps": old_steps,
        "final_steps": final_steps,
        "first_ready_step": final_steps if resumed.status.value == "ready" else None,
        "additional_elapsed_seconds": round(time.perf_counter() - started, 3),
        "answer": (
            resumed.answer.model_dump(mode="json")
            if resumed.answer is not None
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--total-step-limit", type=int, default=12)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--expected-count", type=int, default=199)
    parser.add_argument(
        "--allow-case-subset",
        action="store_true",
        help=(
            "Resume only budget checkpoints named by --cases. This is intended "
            "for non-overwriting recovery runs after the full batch; without "
            "this flag the frozen manifest must still contain every checkpoint."
        ),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--text-model", required=True)
    parser.add_argument("--visual-model", required=True)
    parser.add_argument("--dense-model-path", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--visual-search-index", type=Path, required=True)
    parser.add_argument("--visual-descriptor-cache", type=Path)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    case_map = _case_map(args.cases)
    all_case_ids = _budget_case_ids(args.old_root)
    case_ids = (
        [case_id for case_id in all_case_ids if case_id in case_map]
        if args.allow_case_subset
        else all_case_ids
    )
    if len(case_ids) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} budget checkpoints, found {len(case_ids)}"
        )
    missing = sorted(set(case_ids).difference(case_map))
    if missing:
        raise ValueError(f"Cases missing from frozen manifest: {missing[:10]}")

    runtime_args = argparse.Namespace(
        base_url=args.base_url,
        inference_backend="vllm",
        text_model=args.text_model,
        visual_model=args.visual_model,
        timeout=180.0,
        context_length=32768,
        action_budget=args.total_step_limit,
        planner_max_tokens=768,
        controller_max_tokens=512,
        reader_max_tokens=1536,
        checker_max_tokens=1536,
        answerer_max_tokens=768,
        dense=True,
        dense_model="intfloat/multilingual-e5-small",
        dense_model_path=args.dense_model_path,
        dense_device="cpu",
        embedding_cache=args.embedding_cache,
        visual_search_index=args.visual_search_index,
        visual_search_model="vidore/colSmol-500M",
        visual_search_device="cuda",
        visual_similarity_chunk_elements=16_000_000,
        visual_descriptor_cache=args.visual_descriptor_cache,
        multimodal_table_reader=True,
    )
    manifest = {
        "schema_version": "budget-resume-v0.1",
        "started_at": _utc_now(),
        "finished_at": None,
        "status": "initializing",
        "old_root": str(args.old_root.resolve()),
        "case_count": len(case_ids),
        "total_step_limit": args.total_step_limit,
        "workers": args.workers,
        "completed": 0,
        "succeeded": 0,
        "failed": 0,
        "cases": [],
    }
    manifest_path = output_root / "resume_manifest.json"
    _write_json(manifest_path, manifest)
    runtime = _PersistentRuntime(runtime_args)
    manifest["status"] = "running"
    _write_json(manifest_path, manifest)
    lock = threading.Lock()
    results: dict[str, dict] = {}

    def execute(case_id: str) -> dict:
        try:
            return _resume_one(
                case_id=case_id,
                case=case_map[case_id],
                old_root=args.old_root,
                output_root=output_root,
                runtime=runtime,
                total_step_limit=args.total_step_limit,
            )
        except Exception as exc:
            return {
                "case_id": case_id,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, case_id): case_id for case_id in case_ids}
        for future in as_completed(futures):
            row = future.result()
            with lock:
                results[row["case_id"]] = row
                ordered = [results[qid] for qid in case_ids if qid in results]
                manifest["cases"] = ordered
                manifest["completed"] = len(ordered)
                manifest["succeeded"] = sum(
                    item["status"] == "succeeded" for item in ordered
                )
                manifest["failed"] = len(ordered) - manifest["succeeded"]
                _write_json(manifest_path, manifest)

    manifest["finished_at"] = _utc_now()
    manifest["status"] = (
        "completed" if manifest["failed"] == 0 else "completed_with_errors"
    )
    ready_histogram = {str(step): 0 for step in range(8, args.total_step_limit + 1)}
    ready_histogram["never"] = 0
    for row in manifest["cases"]:
        if row["status"] != "succeeded" or row.get("first_ready_step") is None:
            ready_histogram["never"] += 1
        else:
            ready_histogram[str(row["first_ready_step"])] += 1
    manifest["first_ready_histogram"] = ready_histogram
    _write_json(manifest_path, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "completed": manifest["completed"],
        "succeeded": manifest["succeeded"],
        "failed": manifest["failed"],
        "first_ready_histogram": ready_histogram,
    }, ensure_ascii=False))
    return 0 if manifest["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
