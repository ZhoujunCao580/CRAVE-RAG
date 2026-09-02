"""Run independent model-backed questions without aborting the whole batch.

The batch manifest contains no Gold answer or Gold evidence fields. Each case
invokes the canonical ``softdoc run-model`` entry point in a separate process,
so a malformed model response or one document failure cannot corrupt later
cases.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
CASE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _output_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def load_cases(path: Path, *, path_root: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: each case must be a JSON object")
        unknown = set(row) - {"case_id", "document_dir", "question", "question_id", "run_key"}
        if unknown:
            raise ValueError(
                f"{path}:{line_number}: unsupported fields: {sorted(unknown)}"
            )
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or CASE_ID_PATTERN.fullmatch(case_id) is None:
            raise ValueError(
                f"{path}:{line_number}: case_id must match {CASE_ID_PATTERN.pattern!r}"
            )
        if case_id in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate case_id {case_id!r}")
        seen_ids.add(case_id)
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{path}:{line_number}: question must not be blank")
        document_value = row.get("document_dir")
        if not isinstance(document_value, str) or not document_value.strip():
            raise ValueError(f"{path}:{line_number}: document_dir must not be blank")
        document_dir = Path(document_value)
        if not document_dir.is_absolute():
            document_dir = path_root / document_dir
        normalized = dict(row)
        normalized["document_dir"] = str(document_dir.resolve())
        normalized["question"] = question.strip()
        cases.append(normalized)
    if not cases:
        raise ValueError(f"{path}: no cases found")
    return cases


def build_case_command(case: dict[str, Any], args: argparse.Namespace, output: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "softdoc.cli",
        "run-model",
        case["document_dir"],
        "--question",
        case["question"],
        "--output",
        str(output),
        "--base-url",
        args.base_url,
        "--text-model",
        args.text_model,
        "--visual-model",
        args.visual_model,
        "--timeout",
        str(args.timeout),
        "--context-length",
        str(args.context_length),
        "--action-budget",
        str(args.action_budget),
        "--run-key",
        case.get("run_key") or f"{args.run_key_prefix}-{case['case_id']}",
    ]
    if case.get("question_id"):
        command.extend(["--question-id", case["question_id"]])
    if args.dense:
        command.extend(
            [
                "--dense",
                "--dense-model",
                args.dense_model,
                "--dense-device",
                args.dense_device,
            ]
        )
        if args.dense_model_path is not None:
            command.extend(["--dense-model-path", str(args.dense_model_path)])
        if args.embedding_cache is not None:
            command.extend(["--embedding-cache", str(args.embedding_cache)])
        if args.visual_search_index is not None:
            command.extend(
                [
                    "--visual-search-index",
                    str(args.visual_search_index),
                    "--visual-search-device",
                    args.visual_search_device,
                    "--visual-similarity-chunk-elements",
                    str(args.visual_similarity_chunk_elements),
                ]
            )
            if args.visual_search_model is not None:
                command.extend(
                    ["--visual-search-model", args.visual_search_model]
                )
    return command


def run_batch(
    *,
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
    executor: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Batch output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "batch_manifest.json"
    log_root = output_root / "_logs"
    log_root.mkdir()
    manifest: dict[str, Any] = {
        "schema_version": "model-batch-v0.1",
        "started_at": _utc_now(),
        "finished_at": None,
        "status": "running",
        "case_count": len(cases),
        "succeeded": 0,
        "failed": 0,
        "runtime": {
            "base_url": args.base_url,
            "text_model": args.text_model,
            "visual_model": args.visual_model,
            "context_length": args.context_length,
            "action_budget": args.action_budget,
            "dense": args.dense,
            "visual_search_index": (
                str(args.visual_search_index)
                if args.visual_search_index is not None
                else None
            ),
            "visual_search_model": args.visual_search_model,
        },
        "cases": [],
    }
    _write_json(manifest_path, manifest)
    environment = dict(os.environ)
    source_root = str(ROOT / "src")
    environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")

    for case in cases:
        case_output = output_root / case["case_id"]
        command = build_case_command(case, args, case_output)
        started_at = _utc_now()
        try:
            completed = executor(
                command,
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=args.case_timeout,
            )
            return_code = completed.returncode
            error = None if return_code == 0 else f"run-model exited with code {return_code}"
            stdout = _output_text(completed.stdout)
            stderr = _output_text(completed.stderr)
        except subprocess.TimeoutExpired as exc:
            return_code = None
            error = f"case timed out after {args.case_timeout} seconds"
            stdout = _output_text(exc.stdout)
            stderr = _output_text(exc.stderr)
        except Exception as exc:  # keep later independent cases runnable
            return_code = None
            error = f"{type(exc).__name__}: {exc}"
            stdout = ""
            stderr = ""
        succeeded = return_code == 0
        stdout_log = log_root / f"{case['case_id']}.stdout.log"
        stderr_log = log_root / f"{case['case_id']}.stderr.log"
        stdout_log.write_text(stdout, encoding="utf-8")
        stderr_log.write_text(stderr, encoding="utf-8")
        manifest["cases"].append(
            {
                "case_id": case["case_id"],
                "question_id": case.get("question_id"),
                "document_dir": case["document_dir"],
                "output_dir": str(case_output),
                "started_at": started_at,
                "finished_at": _utc_now(),
                "status": "succeeded" if succeeded else "failed",
                "return_code": return_code,
                "error": error,
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
            }
        )
        key = "succeeded" if succeeded else "failed"
        manifest[key] += 1
        _write_json(manifest_path, manifest)

    manifest["finished_at"] = _utc_now()
    manifest["status"] = "completed" if manifest["failed"] == 0 else "completed_with_errors"
    _write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="UTF-8 JSONL case manifest")
    parser.add_argument("--path-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--text-model", default="qwen3:8b")
    parser.add_argument("--visual-model", default="qwen3-vl:4b")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--case-timeout", type=float, default=None)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--action-budget", type=int, default=7)
    parser.add_argument("--run-key-prefix", default="batch-v0")
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--dense-model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--dense-model-path", type=Path)
    parser.add_argument("--dense-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--visual-search-index", type=Path)
    parser.add_argument("--visual-search-model")
    parser.add_argument(
        "--visual-search-device", choices=("cpu", "cuda"), default="cuda"
    )
    parser.add_argument(
        "--visual-similarity-chunk-elements",
        type=int,
        default=16_000_000,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.visual_search_index is not None and not args.dense:
        raise ValueError("--visual-search-index requires --dense")
    cases = load_cases(args.cases, path_root=args.path_root.resolve())
    manifest = run_batch(cases=cases, args=args)
    print(
        f"Batch {manifest['status']}: {manifest['succeeded']} succeeded, "
        f"{manifest['failed']} failed; artifacts: {args.output_root.resolve()}"
    )
    return 0 if manifest["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
