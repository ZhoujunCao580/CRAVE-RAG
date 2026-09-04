"""Run model-backed questions without aborting the whole batch.

The batch manifest contains no Gold answer or Gold evidence fields. Each case
can either invoke the canonical ``softdoc run-model`` entry point in an isolated
process or use a persistent in-process runtime.  Persistent mode loads text and
visual retrieval models once, caches one search service per document, and can
run several independent trajectories concurrently.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed


ROOT = Path(__file__).resolve().parents[1]
CASE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _batch_lock_path(
    cases: list[dict[str, Any]], args: argparse.Namespace
) -> Path:
    identity = {
        "base_url": args.base_url,
        "inference_backend": args.inference_backend,
        "text_model": args.text_model,
        "visual_model": args.visual_model,
        "cases": [
            {
                "case_id": item["case_id"],
                "question_id": item.get("question_id"),
                "document_dir": item["document_dir"],
            }
            for item in cases
        ],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    return Path(tempfile.gettempdir()) / f"softdoc-model-batch-{digest}.lock"


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class _BatchRunLock:
    """Prevent accidental duplicate batches from sharing one model service."""

    def __init__(self, cases: list[dict[str, Any]], args: argparse.Namespace) -> None:
        self.path = _batch_lock_path(cases, args)
        self.output_root = str(args.output_root.resolve())
        self._fd: int | None = None

    def __enter__(self) -> "_BatchRunLock":
        owner = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "output_root": self.output_root,
            "started_at": _utc_now(),
        }
        for _ in range(2):
            try:
                self._fd = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError:
                try:
                    existing = json.loads(self.path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    existing = {}
                existing_pid = existing.get("pid")
                same_host = existing.get("hostname") == socket.gethostname()
                if (
                    same_host
                    and isinstance(existing_pid, int)
                    and not _pid_is_alive(existing_pid)
                ):
                    self.path.unlink(missing_ok=True)
                    continue
                detail = (
                    f"PID {existing_pid} on {existing.get('hostname')}"
                    if existing
                    else "an unknown owner"
                )
                raise RuntimeError(
                    "A matching model batch is already running under "
                    f"{detail}; output={existing.get('output_root')!r}."
                )
            else:
                assert self._fd is not None
                os.write(
                    self._fd,
                    (json.dumps(owner, ensure_ascii=False) + "\n").encode("utf-8"),
                )
                return self
        raise RuntimeError(f"Could not acquire model batch lock: {self.path}")

    def __exit__(self, *_: object) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self.path.unlink(missing_ok=True)


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
        "--inference-backend",
        args.inference_backend,
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
        "--planner-max-tokens",
        str(getattr(args, "planner_max_tokens", 768)),
        "--controller-max-tokens",
        str(getattr(args, "controller_max_tokens", 512)),
        "--reader-max-tokens",
        str(getattr(args, "reader_max_tokens", 1536)),
        "--checker-max-tokens",
        str(getattr(args, "checker_max_tokens", 1536)),
        "--answerer-max-tokens",
        str(getattr(args, "answerer_max_tokens", 768)),
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
    with _BatchRunLock(cases, args):
        if getattr(args, "execution_mode", "subprocess") == "persistent":
            if executor is not subprocess.run:
                raise ValueError("Persistent mode does not accept a subprocess executor")
            return _run_persistent_batch_unlocked(cases=cases, args=args)
        return _run_batch_unlocked(cases=cases, args=args, executor=executor)


def _group_cases_by_document(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep first-seen document order while placing its questions together."""

    groups: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        groups.setdefault(case["document_dir"], []).append(case)
    return [case for group in groups.values() for case in group]


class _LockedVisualModel:
    """Serialize calls into one shared ColSmol model while reusing its weights."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self._lock = threading.Lock()

    def encode_query(self, sentences: list[str], **kwargs: Any) -> Any:
        with self._lock:
            return self._model.encode_query(sentences, **kwargs)

    def similarity(self, queries: list[Any], documents: list[Any], **kwargs: Any) -> Any:
        with self._lock:
            return self._model.similarity(queries, documents, **kwargs)


class _PersistentRuntime:
    """Reusable model clients and retrieval objects for one batch process."""

    def __init__(self, args: argparse.Namespace) -> None:
        from softdoc.retrieval import FileEmbeddingCache, HuggingFaceE5Encoder

        self.args = args
        self._service_lock = threading.Lock()
        self._services: dict[str, tuple[Any, Any]] = {}
        self._dense_encoder = None
        self._embedding_cache = None
        self._visual_model = None
        if args.dense:
            self._dense_encoder = HuggingFaceE5Encoder(
                model_name=args.dense_model,
                model_path=args.dense_model_path,
                device=args.dense_device,
                local_files_only=args.dense_model_path is not None,
            )
            if args.embedding_cache is not None:
                self._embedding_cache = FileEmbeddingCache(args.embedding_cache)
        if args.visual_search_index is not None:
            from sentence_transformers import MultiVectorEncoder

            model_name = args.visual_search_model
            if not model_name:
                config_path = Path(args.visual_search_index) / "config.json"
                model_name = json.loads(config_path.read_text(encoding="utf-8"))["model"]
            self._visual_model = _LockedVisualModel(
                MultiVectorEncoder(model_name, device=args.visual_search_device)
            )

    def _search_resources(self, document_dir: str) -> tuple[Any, Any]:
        from softdoc.reading_environment import DocumentSearchService
        from softdoc.retrieval import (
            CandidateMergePolicy,
            DenseIndex,
            SearchSessionConfig,
            SearchUnitBuilder,
            VisualDenseIndex,
        )
        from softdoc.serialization import load_document

        with self._service_lock:
            cached = self._services.get(document_dir)
            if cached is not None:
                return cached
            document = load_document(Path(document_dir))
            search_service = None
            if self.args.dense:
                search_units = SearchUnitBuilder().build(document)
                dense_index = DenseIndex(
                    search_units,
                    self._dense_encoder,
                    cache=self._embedding_cache,
                )
                visual_index = (
                    VisualDenseIndex(
                        document,
                        self.args.visual_search_index,
                        model_name=self.args.visual_search_model,
                        device=self.args.visual_search_device,
                        similarity_chunk_elements=(
                            self.args.visual_similarity_chunk_elements
                        ),
                        model=self._visual_model,
                    )
                    if self.args.visual_search_index is not None
                    else None
                )
                config = (
                    SearchSessionConfig(
                        merge_policy=CandidateMergePolicy.FIXED_TEXT_VISUAL_QUOTA,
                        batch_size=5,
                        text_quota=3,
                        visual_quota=2,
                    )
                    if visual_index is not None
                    else None
                )
                search_service = DocumentSearchService(
                    document,
                    search_units=search_units,
                    dense_backend=dense_index,
                    visual_backend=visual_index,
                    config=config,
                )
            resources = (document, search_service)
            self._services[document_dir] = resources
            return resources

    def _runner(self) -> Any:
        from softdoc.controller_ollama import VLLMControllerBackend
        from softdoc.model_backends import (
            ModelBackedReader,
            OllamaAnswererBackend,
            OllamaEvidenceCheckerBackend,
            OllamaVisualReaderBackend,
        )
        from softdoc.model_runner import ModelBackedRunner
        from softdoc.openai_compatible import (
            OpenAICompatibleConfig,
            OpenAICompatibleStructuredClient,
        )
        from softdoc.planning import InitialPlanner, VLLMPlannerBackend
        from softdoc.reading_environment import ReadingEnvironmentConfig

        if self.args.inference_backend != "vllm":
            raise ValueError("Persistent mode currently requires --inference-backend vllm")

        def config(max_tokens: int) -> OpenAICompatibleConfig:
            return OpenAICompatibleConfig(
                model=self.args.text_model,
                base_url=self.args.base_url,
                timeout_seconds=self.args.timeout,
                max_tokens=max_tokens,
            )

        planner_config = config(self.args.planner_max_tokens)
        controller_config = config(self.args.controller_max_tokens)
        reader_client = OpenAICompatibleStructuredClient(
            config(self.args.reader_max_tokens)
        )
        checker_client = OpenAICompatibleStructuredClient(
            config(self.args.checker_max_tokens)
        )
        answerer_client = OpenAICompatibleStructuredClient(
            config(self.args.answerer_max_tokens)
        )
        return ModelBackedRunner(
            planner=InitialPlanner(VLLMPlannerBackend(planner_config)),
            controller=VLLMControllerBackend(controller_config),
            reader=ModelBackedReader(OllamaVisualReaderBackend(reader_client)),
            checker=OllamaEvidenceCheckerBackend(checker_client),
            answerer=OllamaAnswererBackend(answerer_client),
            environment_config=ReadingEnvironmentConfig(
                action_budget=self.args.action_budget
            ),
        )

    def run_case(self, case: dict[str, Any], output: Path) -> None:
        from softdoc.model_runner import write_model_pipeline_run

        document, search_service = self._search_resources(case["document_dir"])
        result = self._runner().run(
            document=document,
            asset_root=Path(case["document_dir"]),
            question=case["question"],
            run_key=case.get("run_key") or f"{self.args.run_key_prefix}-{case['case_id']}",
            question_id=case.get("question_id"),
            search_service=search_service,
        )
        write_model_pipeline_run(result, output)


def _run_callable_with_peak_vram(call: Callable[[], None]) -> int | None:
    stop = threading.Event()
    samples: list[int] = []

    def sample() -> None:
        while not stop.is_set():
            value = _query_gpu_memory_mib()
            if value is not None:
                samples.append(value)
            stop.wait(1.0)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        call()
    finally:
        stop.set()
        sampler.join(timeout=6)
    return max(samples) if samples else None


def _run_persistent_batch_unlocked(
    *, cases: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Batch output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    log_root = output_root / "_logs"
    log_root.mkdir()
    manifest_path = output_root / "batch_manifest.json"
    ordered_cases = _group_cases_by_document(cases)
    manifest: dict[str, Any] = {
        "schema_version": "model-batch-v0.2",
        "started_at": _utc_now(),
        "finished_at": None,
        "status": "initializing",
        "case_count": len(cases),
        "succeeded": 0,
        "failed": 0,
        "runtime": {
            "execution_mode": "persistent",
            "workers": args.workers,
            "grouped_by_document": True,
            "base_url": args.base_url,
            "inference_backend": args.inference_backend,
            "text_model": args.text_model,
            "visual_model": args.visual_model,
            "context_length": args.context_length,
            "action_budget": args.action_budget,
            "dense": args.dense,
            "visual_search_index": str(args.visual_search_index) if args.visual_search_index else None,
            "visual_search_model": args.visual_search_model,
            "max_tokens": {
                "planner": args.planner_max_tokens,
                "controller": args.controller_max_tokens,
                "reader": args.reader_max_tokens,
                "checker": args.checker_max_tokens,
                "answerer": args.answerer_max_tokens,
            },
        },
        "cases": [],
    }
    _write_json(manifest_path, manifest)
    initialization_started = time.perf_counter()
    runtime = _PersistentRuntime(args)
    manifest["runtime"]["initialization_seconds"] = round(
        time.perf_counter() - initialization_started, 3
    )
    manifest["status"] = "running"
    _write_json(manifest_path, manifest)

    def execute(case: dict[str, Any]) -> dict[str, Any]:
        case_output = output_root / case["case_id"]
        started_at = _utc_now()
        started = time.perf_counter()
        error = None
        stderr = ""
        peak_vram_mib = None
        try:
            peak_vram_mib = _run_callable_with_peak_vram(
                lambda: runtime.run_case(case, case_output)
            )
            succeeded = True
        except Exception as exc:
            succeeded = False
            error = f"{type(exc).__name__}: {exc}"
            stderr = traceback.format_exc()
        stdout_log = log_root / f"{case['case_id']}.stdout.log"
        stderr_log = log_root / f"{case['case_id']}.stderr.log"
        stdout_log.write_text("", encoding="utf-8")
        stderr_log.write_text(stderr, encoding="utf-8")
        return {
            "case_id": case["case_id"],
            "question_id": case.get("question_id"),
            "document_dir": case["document_dir"],
            "output_dir": str(case_output),
            "started_at": started_at,
            "finished_at": _utc_now(),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "peak_gpu_memory_mib": peak_vram_mib,
            "status": "succeeded" if succeeded else "failed",
            "return_code": 0 if succeeded else None,
            "error": error,
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
        }

    completed_by_id: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, case): case for case in ordered_cases}
        for future in as_completed(futures):
            record = future.result()
            completed_by_id[record["case_id"]] = record
            manifest["cases"] = [
                completed_by_id[case["case_id"]]
                for case in ordered_cases
                if case["case_id"] in completed_by_id
            ]
            manifest["succeeded"] = sum(
                item["status"] == "succeeded" for item in manifest["cases"]
            )
            manifest["failed"] = len(manifest["cases"]) - manifest["succeeded"]
            _write_json(manifest_path, manifest)

    manifest["finished_at"] = _utc_now()
    manifest["status"] = "completed" if manifest["failed"] == 0 else "completed_with_errors"
    _write_json(manifest_path, manifest)
    return manifest


def _run_batch_unlocked(
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
            "inference_backend": args.inference_backend,
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
        case_started = time.perf_counter()
        peak_vram_mib: int | None = None
        try:
            if executor is subprocess.run:
                completed, peak_vram_mib = _run_with_peak_vram(
                    command,
                    cwd=ROOT,
                    env=environment,
                    timeout=args.case_timeout,
                )
            else:
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
                "elapsed_seconds": round(time.perf_counter() - case_started, 3),
                "peak_gpu_memory_mib": peak_vram_mib,
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
    parser.add_argument(
        "--inference-backend",
        choices=("ollama", "vllm"),
        default="ollama",
    )
    parser.add_argument("--text-model", default="qwen3:8b")
    parser.add_argument("--visual-model", default="qwen3-vl:4b")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--case-timeout", type=float, default=None)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--action-budget", type=int, default=7)
    parser.add_argument("--run-key-prefix", default="batch-v0")
    parser.add_argument(
        "--execution-mode",
        choices=("subprocess", "persistent"),
        default="subprocess",
        help="Persistent mode reuses retrieval models and supports concurrent trajectories.",
    )
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--planner-max-tokens", type=int, default=768)
    parser.add_argument("--controller-max-tokens", type=int, default=512)
    parser.add_argument("--reader-max-tokens", type=int, default=1536)
    parser.add_argument("--checker-max-tokens", type=int, default=1536)
    parser.add_argument("--answerer-max-tokens", type=int, default=768)
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


def _query_gpu_memory_mib() -> int | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    values: list[int] = []
    for line in completed.stdout.splitlines():
        try:
            values.append(int(line.strip()))
        except ValueError:
            continue
    return max(values) if values else None


def _run_with_peak_vram(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float | None,
) -> tuple[subprocess.CompletedProcess[str], int | None]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stop = threading.Event()
    samples: list[int] = []

    def sample() -> None:
        while not stop.is_set():
            value = _query_gpu_memory_mib()
            if value is not None:
                samples.append(value)
            stop.wait(0.5)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr)
    finally:
        stop.set()
        sampler.join(timeout=6)
    return (
        subprocess.CompletedProcess(command, process.returncode, stdout, stderr),
        max(samples) if samples else None,
    )


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
