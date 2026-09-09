"""Run model-backed questions without aborting the whole batch.

The batch manifest contains no Gold answer or Gold evidence fields. Each case
can either invoke the low-level ``softdoc run-model`` entry point in an isolated
process or use a persistent in-process runtime.  The named baseline profile is
the canonical full-architecture entry point: it loads retrieval models once,
caches one search service per document, and can run several independent
trajectories concurrently.
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
BASELINE_RUNTIME_PROFILE = "crave-baseline-v1"


def _validate_runtime_profile(args: argparse.Namespace) -> None:
    """Fail fast when a named experiment profile would omit a core module."""

    profile = getattr(args, "runtime_profile", None)
    if profile is None:
        return
    if profile != BASELINE_RUNTIME_PROFILE:
        raise ValueError(f"Unknown runtime profile: {profile}")

    errors: list[str] = []
    if args.execution_mode != "persistent":
        errors.append("--execution-mode persistent")
    if args.inference_backend != "vllm":
        errors.append("--inference-backend vllm")
    if not args.dense:
        errors.append("--dense")
    if args.visual_search_index is None:
        errors.append("--visual-search-index PATH")
    if args.visual_descriptor_cache is None:
        errors.append("--visual-descriptor-cache PATH")
    if not args.visual_descriptor_on_demand:
        errors.append("--visual-descriptor-on-demand")
    if not args.multimodal_table_reader:
        errors.append("--multimodal-table-reader")
    if errors:
        raise ValueError(
            f"Runtime profile {profile!r} is incomplete; required settings: "
            + ", ".join(errors)
        )

    index_root = Path(args.visual_search_index)
    required_index_entries = ("config.json", "state.json", "assets.jsonl", "shards")
    missing_index_entries = [
        name for name in required_index_entries if not (index_root / name).exists()
    ]
    if missing_index_entries:
        raise ValueError(
            f"Runtime profile {profile!r} visual index is incomplete at "
            f"{index_root}: missing {missing_index_entries}"
        )
    try:
        index_config = json.loads(
            (index_root / "config.json").read_text(encoding="utf-8")
        )
        index_state = json.loads(
            (index_root / "state.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Runtime profile {profile!r} cannot read visual index metadata: {exc}"
        ) from exc
    if index_state.get("state") != "completed" or index_state.get(
        "pending_image_count"
    ) not in (None, 0):
        raise ValueError(
            f"Runtime profile {profile!r} requires a completed visual index"
        )
    configured_model = index_config.get("model")
    if args.visual_search_model and configured_model != args.visual_search_model:
        raise ValueError(
            "--visual-search-model does not match visual index config.json: "
            f"{args.visual_search_model!r} != {configured_model!r}"
        )
    if not any((index_root / "shards").glob("*.npz")):
        raise ValueError(
            f"Runtime profile {profile!r} visual index has no embedding shards"
        )
    cache_path = Path(args.visual_descriptor_cache)
    if cache_path.exists() and not cache_path.is_file():
        raise ValueError("--visual-descriptor-cache must be a JSONL file path")


def _validate_batch_args(args: argparse.Namespace) -> None:
    """Validate cross-option contracts for CLI and programmatic callers."""

    _validate_runtime_profile(args)
    if args.visual_search_index is not None and not args.dense:
        raise ValueError("--visual-search-index requires --dense")
    if args.visual_descriptor_cache is not None and args.execution_mode != "persistent":
        raise ValueError(
            "--visual-descriptor-cache requires --execution-mode persistent; "
            "the canonical full architecture is the persistent batch runner, "
            "including for one-question smoke tests"
        )
    if args.visual_descriptor_on_demand and args.execution_mode != "persistent":
        raise ValueError(
            "--visual-descriptor-on-demand requires --execution-mode persistent"
        )
    if args.visual_descriptor_on_demand and args.visual_search_index is None:
        raise ValueError(
            "--visual-descriptor-on-demand requires --visual-search-index"
        )


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically persist diagnostic rows, including failed QA trajectories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
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
    if getattr(args, "disable_answerer_thinking", False):
        command.append("--disable-answerer-thinking")
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
    if getattr(args, "multimodal_table_reader", False):
        command.append("--multimodal-table-reader")
    return command


def run_batch(
    *,
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
    executor: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    _validate_batch_args(args)
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
        self._visual_descriptor_backend = None
        self._visual_descriptor_lock = threading.RLock()
        self._case_context = threading.local()
        self._visual_descriptor_records: dict[
            tuple[str, str, str, str, str], dict[str, Any]
        ] = {}
        descriptor_cache = getattr(args, "visual_descriptor_cache", None)
        if descriptor_cache is not None and Path(descriptor_cache).is_file():
            from softdoc.visual_retrieval import VisualRetrievalDescriptor

            for line_number, line in enumerate(
                Path(descriptor_cache).read_text(encoding="utf-8").splitlines(), 1
            ):
                if not line.strip():
                    continue
                row = json.loads(line)
                document_id = row.get("document_id")
                if not isinstance(document_id, str) or not document_id:
                    raise ValueError(
                        f"{descriptor_cache}:{line_number}: missing document_id"
                    )
                descriptor = VisualRetrievalDescriptor.model_validate(
                    {
                        name: row.get(name)
                        for name in VisualRetrievalDescriptor.model_fields
                    }
                )
                key = (
                    document_id,
                    descriptor.element_id,
                    descriptor.visual_asset_sha256,
                    descriptor.generator_model,
                    descriptor.prompt_version,
                )
                self._visual_descriptor_records[key] = row
        if getattr(args, "visual_descriptor_on_demand", False):
            if descriptor_cache is None:
                raise ValueError(
                    "--visual-descriptor-on-demand requires "
                    "--visual-descriptor-cache"
                )
            if args.inference_backend != "vllm":
                raise ValueError(
                    "On-demand visual descriptions currently require the "
                    "OpenAI-compatible vLLM backend"
                )
            from softdoc.model_backends import OllamaVisualRetrievalBackend
            from softdoc.openai_compatible import (
                OpenAICompatibleConfig,
                OpenAICompatibleStructuredClient,
            )

            self._visual_descriptor_backend = OllamaVisualRetrievalBackend(
                OpenAICompatibleStructuredClient(
                    OpenAICompatibleConfig(
                        model=args.visual_model,
                        base_url=args.base_url,
                        timeout_seconds=args.timeout,
                        max_tokens=args.visual_descriptor_max_tokens,
                    )
                )
            )
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
                    batch_enricher=(
                        lambda batch: self._enrich_visual_candidate_batch(
                            document, Path(document_dir), batch
                        )
                        if (
                            self._visual_descriptor_backend is not None
                            or getattr(self.args, "visual_descriptor_cache", None)
                            is not None
                        )
                        else batch
                    ),
                )
            resources = (document, search_service)
            self._services[document_dir] = resources
            return resources

    def _enrich_visual_candidate_batch(
        self, document: Any, document_dir: Path, batch: Any
    ) -> Any:
        """Describe only the already-frozen visual slots and never rerank them."""

        from softdoc.retrieval import (
            CandidatePreview,
            RetrievalSource,
            SearchBatch,
            SnippetSource,
        )
        from softdoc.visual_retrieval import (
            VISUAL_RETRIEVAL_PROMPT_VERSION,
            VisualRetrievalDescriptor,
            VisualRetrievalResult,
            build_visual_retrieval_request,
            materialize_visual_retrieval_descriptors,
        )

        visual_previews = [
            item
            for item in batch.candidate_previews
            if item.preview_source == RetrievalSource.VISUAL_DENSE
            and item.visual_asset_id is not None
            and item.snippet_source == SnippetSource.VISUAL_METADATA
        ]
        if not visual_previews:
            return batch

        with self._visual_descriptor_lock:
            requested_ids = {item.element_id for item in visual_previews}
            request = build_visual_retrieval_request(
                document, document_dir, element_ids=requested_ids
            )
            request_ids = {item.element_id for item in request.visual_inputs}
            for missing_id in sorted(requested_ids - request_ids):
                self._record_visual_descriptor_call(
                    {
                        "component": "visual_retrieval",
                        "source": "environment",
                        "input": {"element_id": missing_id},
                        "output": None,
                        "error": "visual_asset_unavailable",
                        "elapsed_seconds": 0.0,
                    }
                )

            prompt_version = (
                self._visual_descriptor_backend.prompt_version
                if self._visual_descriptor_backend is not None
                else VISUAL_RETRIEVAL_PROMPT_VERSION
            )
            descriptors_by_element: dict[str, VisualRetrievalDescriptor] = {}
            for request_item in request.visual_inputs:
                key = (
                    document.document_id,
                    request_item.element_id,
                    request_item.visual_asset_sha256,
                    self.args.visual_model,
                    prompt_version,
                )
                cached_row = self._visual_descriptor_records.get(key)
                if cached_row is not None:
                    descriptor = VisualRetrievalDescriptor.model_validate(
                        {
                            name: cached_row.get(name)
                            for name in VisualRetrievalDescriptor.model_fields
                        }
                    )
                    descriptors_by_element[descriptor.element_id] = descriptor
                    continue
                if self._visual_descriptor_backend is None:
                    continue

                single_request = request.model_copy(
                    update={"visual_inputs": [request_item]}
                )
                started = time.perf_counter()
                try:
                    generated = self._visual_descriptor_backend.describe(
                        single_request
                    )
                    if {
                        item.input_id for item in generated.descriptors
                    } != {request_item.input_id}:
                        raise ValueError(
                            "Visual descriptor response must cover its one input exactly"
                        )
                    descriptor = materialize_visual_retrieval_descriptors(
                        document,
                        single_request,
                        VisualRetrievalResult(
                            descriptors=list(generated.descriptors)
                        ),
                        generator_model=self.args.visual_model,
                        prompt_version=prompt_version,
                    )[0]
                except Exception as exc:
                    self._record_visual_descriptor_call(
                        {
                            "component": "visual_retrieval",
                            "source": "model",
                            "input": single_request.model_dump(mode="json"),
                            "output": None,
                            "error": f"{type(exc).__name__}: {exc}",
                            "elapsed_seconds": round(
                                time.perf_counter() - started, 3
                            ),
                        }
                    )
                    # Preview enrichment is optional metadata.  Preserve the
                    # frozen candidate and let the QA trajectory continue.
                    continue

                self._record_visual_descriptor_call(
                    {
                        "component": "visual_retrieval",
                        "source": "model",
                        "input": single_request.model_dump(mode="json"),
                        "output": generated.model_dump(mode="json"),
                        "error": None,
                        "elapsed_seconds": round(
                            time.perf_counter() - started, 3
                        ),
                    }
                )
                descriptors_by_element[descriptor.element_id] = descriptor
                row = descriptor.model_dump(mode="json")
                row["document_id"] = document.document_id
                row["softdoc_dir"] = str(document_dir)
                self._visual_descriptor_records[key] = row
                cache_path = Path(self.args.visual_descriptor_cache)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())

            updated = []
            for preview in batch.candidate_previews:
                descriptor = descriptors_by_element.get(preview.element_id)
                if (
                    descriptor is None
                    or preview.snippet_source != SnippetSource.VISUAL_METADATA
                ):
                    updated.append(preview)
                    continue
                text = " ".join(descriptor.search_summary.split())
                payload = preview.model_dump(mode="json")
                payload.update(
                    {
                        "matched_snippet": text,
                        "snippet_char_start": 0,
                        "snippet_char_end": len(text),
                        "snippet_truncated": False,
                    }
                )
                updated.append(CandidatePreview.model_validate(payload))
            return SearchBatch.model_validate(
                {
                    **batch.model_dump(mode="json"),
                    "candidate_previews": [
                        item.model_dump(mode="json") for item in updated
                    ],
                }
            )

    def _record_visual_descriptor_call(self, record: dict[str, Any]) -> None:
        records = getattr(self._case_context, "visual_descriptor_calls", None)
        if records is not None:
            records.append(record)

    def _runner(self) -> Any:
        from softdoc.controller_ollama import VLLMControllerBackend
        from softdoc.model_backends import (
            ModelBackedReader,
            MultimodalTableReaderBackend,
            OllamaAnswererBackend,
            OllamaEvidenceCheckerBackend,
            OllamaVisualReaderBackend,
            OllamaVisualScanBackend,
        )
        from softdoc.model_runner import ModelBackedRunner
        from softdoc.openai_compatible import (
            OpenAICompatibleConfig,
            OpenAICompatibleStructuredClient,
        )
        from softdoc.planning import InitialPlanner, PlannerConfig, VLLMPlannerBackend
        from softdoc.reading_environment import ReadingEnvironmentConfig

        if self.args.inference_backend != "vllm":
            raise ValueError("Persistent mode currently requires --inference-backend vllm")

        def config(
            max_tokens: int, *, enable_thinking: bool | None = None
        ) -> OpenAICompatibleConfig:
            return OpenAICompatibleConfig(
                model=self.args.text_model,
                base_url=self.args.base_url,
                timeout_seconds=self.args.timeout,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
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
            config(
                self.args.answerer_max_tokens,
                enable_thinking=(
                    False
                    if getattr(self.args, "disable_answerer_thinking", False)
                    else None
                ),
            )
        )
        return ModelBackedRunner(
            planner=InitialPlanner(
                VLLMPlannerBackend(planner_config),
                PlannerConfig(fallback_to_root_on_limit=True),
            ),
            controller=VLLMControllerBackend(controller_config),
            reader=ModelBackedReader(
                OllamaVisualReaderBackend(reader_client),
                table_reader=(
                    MultimodalTableReaderBackend(reader_client)
                    if getattr(self.args, "multimodal_table_reader", False)
                    else None
                ),
            ),
            checker=OllamaEvidenceCheckerBackend(checker_client),
            answerer=OllamaAnswererBackend(answerer_client),
            visual_scanner=OllamaVisualScanBackend(reader_client),
            environment_config=ReadingEnvironmentConfig(
                action_budget=self.args.action_budget
            ),
        )

    def run_case(self, case: dict[str, Any], output: Path) -> None:
        from softdoc.model_runner import write_model_pipeline_run

        self._case_context.visual_descriptor_calls = []
        try:
            document, search_service = self._search_resources(case["document_dir"])
            result = self._runner().run(
                document=document,
                asset_root=Path(case["document_dir"]),
                question=case["question"],
                run_key=(
                    case.get("run_key")
                    or f"{self.args.run_key_prefix}-{case['case_id']}"
                ),
                question_id=case.get("question_id"),
                search_service=search_service,
            )
            write_model_pipeline_run(result, output)
        finally:
            records = list(self._case_context.visual_descriptor_calls)
            if records:
                _write_jsonl(output / "visual_descriptor_calls.jsonl", records)
            self._case_context.visual_descriptor_calls = None


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
            "runtime_profile": getattr(args, "runtime_profile", None),
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
            "visual_descriptor_cache": (
                str(args.visual_descriptor_cache)
                if getattr(args, "visual_descriptor_cache", None)
                else None
            ),
            "visual_descriptor_on_demand": getattr(
                args, "visual_descriptor_on_demand", False
            ),
            "multimodal_table_reader": getattr(
                args, "multimodal_table_reader", False
            ),
            "max_tokens": {
                "planner": args.planner_max_tokens,
                "controller": args.controller_max_tokens,
                "reader": args.reader_max_tokens,
                "checker": args.checker_max_tokens,
                "answerer": args.answerer_max_tokens,
            },
            "disable_answerer_thinking": getattr(
                args, "disable_answerer_thinking", False
            ),
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
            "runtime_profile": getattr(args, "runtime_profile", None),
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
        "--runtime-profile",
        choices=(BASELINE_RUNTIME_PROFILE,),
        help=(
            "Named fail-fast architecture contract. crave-baseline-v1 requires "
            "persistent vLLM, Dense + Visual Dense, preview-only descriptor "
            "cache/on-demand generation, and the multimodal Table Reader."
        ),
    )
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
    parser.add_argument(
        "--disable-answerer-thinking",
        action="store_true",
        help=(
            "Send chat_template_kwargs.enable_thinking=false only for the "
            "Answerer. Planner, Controller, Reader and Checker keep the model "
            "default."
        ),
    )
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--dense-model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--dense-model-path", type=Path)
    parser.add_argument("--dense-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--visual-search-index", type=Path)
    parser.add_argument("--visual-search-model")
    parser.add_argument(
        "--visual-descriptor-cache",
        type=Path,
        help=(
            "Validated JSONL cache of Controller-preview-only visual "
            "descriptions; never BM25/Dense corpus text."
        ),
    )
    parser.add_argument(
        "--visual-descriptor-on-demand",
        action="store_true",
        help=(
            "After each 3-text+2-visual batch is frozen, generate and cache "
            "descriptions only for its visible visual candidates."
        ),
    )
    parser.add_argument("--visual-descriptor-max-tokens", type=int, default=256)
    parser.add_argument("--multimodal-table-reader", action="store_true")
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
    _validate_batch_args(args)
    cases = load_cases(args.cases, path_root=args.path_root.resolve())
    manifest = run_batch(cases=cases, args=args)
    print(
        f"Batch {manifest['status']}: {manifest['succeeded']} succeeded, "
        f"{manifest['failed']} failed; artifacts: {args.output_root.resolve()}"
    )
    return 0 if manifest["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
