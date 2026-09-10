from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess

import pytest
from PIL import Image

from scripts.run_model_batch import (
    BASELINE_RUNTIME_PROFILE,
    _PersistentRuntime,
    _batch_lock_path,
    _group_cases_by_document,
    _validate_batch_args,
    _validate_runtime_profile,
    build_case_command,
    load_cases,
    run_batch,
)
from softdoc.models import ElementType
from softdoc.retrieval import (
    CandidatePreview,
    PreviewMatchScope,
    RetrievalSource,
    SearchBatch,
    SnippetSource,
)
from softdoc.visual_retrieval import (
    VISUAL_RETRIEVAL_METADATA_KEY,
    VisualRetrievalDraft,
    VisualRetrievalResult,
    build_visual_retrieval_request,
)


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_root=tmp_path / "batch",
        base_url="http://127.0.0.1:11434",
        inference_backend="ollama",
        text_model="text-model",
        controller_model=None,
        visual_model="visual-model",
        timeout=30.0,
        case_timeout=120.0,
        context_length=4096,
        action_budget=5,
        run_key_prefix="pilot",
        execution_mode="subprocess",
        workers=1,
        planner_max_tokens=768,
        disable_planner_thinking=False,
        controller_max_tokens=512,
        reader_max_tokens=1536,
        checker_max_tokens=1536,
        answerer_max_tokens=768,
        disable_answerer_thinking=False,
        dense=False,
        dense_model="dense-model",
        dense_model_path=None,
        dense_device="cpu",
        embedding_cache=None,
        visual_search_index=None,
        visual_search_model=None,
        visual_descriptor_cache=None,
        visual_descriptor_on_demand=False,
        visual_descriptor_max_tokens=256,
        multimodal_table_reader=False,
        visual_search_device="cuda",
        visual_similarity_chunk_elements=16_000_000,
        runtime_profile=None,
    )


def test_load_cases_validates_ids_and_resolves_document_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "cases.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "case_id": "Q-1",
                "question_id": "benchmark:Q-1",
                "document_dir": "documents/doc-1",
                "question": "  What is reported?  ",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_cases(manifest, path_root=tmp_path)

    assert cases[0]["question"] == "What is reported?"
    assert Path(cases[0]["document_dir"]) == (tmp_path / "documents" / "doc-1").resolve()

    manifest.write_text(
        '{"case_id":"../escape","document_dir":"doc","question":"Q"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="case_id"):
        load_cases(manifest, path_root=tmp_path)


def test_case_command_can_select_controller_only_model(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.inference_backend = "vllm"
    args.controller_model = "controller-sft"
    case = {
        "case_id": "Q1",
        "document_dir": str(tmp_path / "doc1"),
        "question": "Question one?",
    }

    command = build_case_command(case, args, tmp_path / "output")

    assert command[command.index("--text-model") + 1] == "text-model"
    assert command[command.index("--controller-model") + 1] == "controller-sft"


def test_run_batch_keeps_running_after_one_case_fails(tmp_path: Path) -> None:
    args = _args(tmp_path)
    cases = [
        {
            "case_id": "Q1",
            "question_id": "benchmark:Q1",
            "document_dir": str(tmp_path / "doc1"),
            "question": "Question one?",
        },
        {
            "case_id": "Q2",
            "document_dir": str(tmp_path / "doc2"),
            "question": "Question two?",
        },
    ]
    commands: list[list[str]] = []

    def fake_executor(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            7 if len(commands) == 1 else 0,
            stdout="second completed" if len(commands) == 2 else "",
            stderr="first failed" if len(commands) == 1 else "",
        )

    manifest = run_batch(cases=cases, args=args, executor=fake_executor)

    assert len(commands) == 2
    assert manifest["status"] == "completed_with_errors"
    assert manifest["succeeded"] == 1
    assert manifest["failed"] == 1
    assert [item["status"] for item in manifest["cases"]] == ["failed", "succeeded"]
    assert "--question-id" in commands[0]
    assert "--question-id" not in commands[1]
    assert (args.output_root / "_logs" / "Q1.stderr.log").read_text(
        encoding="utf-8"
    ) == "first failed"
    assert (args.output_root / "_logs" / "Q2.stdout.log").read_text(
        encoding="utf-8"
    ) == "second completed"
    written = json.loads(
        (args.output_root / "batch_manifest.json").read_text(encoding="utf-8")
    )
    assert written == manifest


def test_batch_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.output_root.mkdir(parents=True)
    (args.output_root / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        run_batch(
            cases=[
                {
                    "case_id": "Q1",
                    "document_dir": str(tmp_path / "doc"),
                    "question": "Question?",
                }
            ],
            args=args,
        )


def test_batch_refuses_duplicate_live_runner(tmp_path: Path) -> None:
    args = _args(tmp_path)
    cases = [
        {
            "case_id": "Q1",
            "document_dir": str(tmp_path / "doc"),
            "question": "Question?",
        }
    ]
    lock_path = _batch_lock_path(cases, args)
    lock_path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "output_root": "another-output",
            }
        ),
        encoding="utf-8",
    )
    try:
        with pytest.raises(RuntimeError, match="already running"):
            run_batch(cases=cases, args=args)
    finally:
        lock_path.unlink(missing_ok=True)


def test_dense_case_command_preserves_runtime_options(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.dense = True
    args.dense_model_path = tmp_path / "model"
    args.embedding_cache = tmp_path / "cache"
    command = build_case_command(
        {
            "case_id": "Q1",
            "document_dir": str(tmp_path / "doc"),
            "question": "Question?",
        },
        args,
        tmp_path / "output",
    )
    assert "--dense" in command
    assert command[command.index("--dense-device") + 1] == "cpu"
    assert command[command.index("--dense-model-path") + 1] == str(tmp_path / "model")


def test_vllm_case_command_selects_openai_compatible_backend(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.inference_backend = "vllm"
    args.base_url = "http://127.0.0.1:8000/v1"
    command = build_case_command(
        {
            "case_id": "Q1",
            "document_dir": str(tmp_path / "doc"),
            "question": "Question?",
        },
        args,
        tmp_path / "output",
    )
    assert command[command.index("--inference-backend") + 1] == "vllm"
    assert command[command.index("--base-url") + 1] == args.base_url
    assert command[command.index("--controller-max-tokens") + 1] == "512"


def test_visual_case_command_preserves_runtime_options(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.dense = True
    args.visual_search_index = tmp_path / "visual-index"
    args.visual_search_model = "visual-retriever"
    args.multimodal_table_reader = True
    command = build_case_command(
        {
            "case_id": "Q1",
            "document_dir": str(tmp_path / "doc"),
            "question": "Question?",
        },
        args,
        tmp_path / "output",
    )
    assert command[command.index("--visual-search-index") + 1] == str(
        tmp_path / "visual-index"
    )
    assert command[command.index("--visual-search-model") + 1] == (
        "visual-retriever"
    )
    assert "--multimodal-table-reader" in command


def test_group_cases_by_document_is_stable() -> None:
    cases = [
        {"case_id": "A1", "document_dir": "doc-a"},
        {"case_id": "B1", "document_dir": "doc-b"},
        {"case_id": "A2", "document_dir": "doc-a"},
    ]

    grouped = _group_cases_by_document(cases)

    assert [item["case_id"] for item in grouped] == ["A1", "A2", "B1"]


def test_on_demand_descriptor_enriches_only_frozen_visual_preview_and_caches(
    parsed_document, tmp_path: Path
) -> None:
    document = parsed_document.model_copy(deep=True)
    figure = next(
        item for item in document.elements if item.element_type == ElementType.FIGURE
    )
    image_path = tmp_path / "figure.png"
    Image.new("RGB", (80, 60), color=(220, 230, 240)).save(image_path)
    figure.image_path = image_path
    figure.crop_image_path = None
    figure.summary = None
    figure.keywords = []
    figure.metadata.pop(VISUAL_RETRIEVAL_METADATA_KEY, None)
    request = build_visual_retrieval_request(
        document, tmp_path, element_ids={figure.element_id}
    )
    visual_input = request.visual_inputs[0]
    page = next(item for item in document.pages if item.page_id == figure.page_id)
    placeholder = "figure candidate matched from visual content"
    preview = CandidatePreview(
        element_id=figure.element_id,
        element_type=figure.element_type,
        page_id=figure.page_id,
        page_number=page.page_index + 1,
        section_path=list(figure.section_path),
        matched_snippet=placeholder,
        snippet_char_start=0,
        snippet_char_end=len(placeholder),
        snippet_truncated=False,
        snippet_source=SnippetSource.VISUAL_METADATA,
        snippet_source_id=visual_input.visual_asset_id,
        visual_asset_id=visual_input.visual_asset_id,
        matched_by=[RetrievalSource.VISUAL_DENSE],
        preview_source=RetrievalSource.VISUAL_DENSE,
        match_scope=PreviewMatchScope.UNKNOWN,
        visual_rank=1,
        content_availability=figure.content_availability,
    )
    batch = SearchBatch(
        search_session_id="search:test",
        candidate_previews=[preview],
        next_cursor=1,
        exhausted=True,
    )

    class FakeBackend:
        prompt_version = "visual-retrieval-v0.1"

        def __init__(self) -> None:
            self.calls = 0

        def describe(self, current_request):
            self.calls += 1
            return VisualRetrievalResult(
                descriptors=[
                    VisualRetrievalDraft(
                        input_id=current_request.visual_inputs[0].input_id,
                        search_summary=(
                            "A waterfront map labels several piers along the bay."
                        ),
                        keywords=["waterfront", "piers", "bay map"],
                    )
                ]
            )

    backend = FakeBackend()
    runtime = _PersistentRuntime.__new__(_PersistentRuntime)
    runtime.args = argparse.Namespace(
        visual_model="mock-vlm",
        visual_descriptor_cache=tmp_path / "descriptors.jsonl",
    )
    runtime._visual_descriptor_backend = backend
    runtime._visual_descriptor_lock = __import__("threading").RLock()
    runtime._case_context = __import__("threading").local()
    runtime._case_context.visual_descriptor_calls = []
    runtime._visual_descriptor_records = {}

    enriched = runtime._enrich_visual_candidate_batch(document, tmp_path, batch)
    cached = runtime._enrich_visual_candidate_batch(document, tmp_path, batch)

    assert [item.element_id for item in enriched.candidate_previews] == [
        figure.element_id
    ]
    assert "piers" in enriched.candidate_previews[0].matched_snippet
    assert "piers" in cached.candidate_previews[0].matched_snippet
    assert backend.calls == 1
    assert len(runtime._case_context.visual_descriptor_calls) == 1
    assert runtime._case_context.visual_descriptor_calls[0]["error"] is None
    rows = (tmp_path / "descriptors.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["element_id"] == figure.element_id
    assert figure.summary is None
    assert figure.keywords == []
    assert VISUAL_RETRIEVAL_METADATA_KEY not in figure.metadata


def test_descriptor_failure_keeps_frozen_preview_and_records_error(
    parsed_document, tmp_path: Path
) -> None:
    document = parsed_document.model_copy(deep=True)
    figure = next(
        item for item in document.elements if item.element_type == ElementType.FIGURE
    )
    image_path = tmp_path / "figure.png"
    Image.new("RGB", (80, 60), color=(220, 230, 240)).save(image_path)
    figure.image_path = image_path
    figure.crop_image_path = None
    request = build_visual_retrieval_request(
        document, tmp_path, element_ids={figure.element_id}
    )
    visual_input = request.visual_inputs[0]
    page = next(item for item in document.pages if item.page_id == figure.page_id)
    placeholder = "figure candidate matched from visual content"
    batch = SearchBatch(
        search_session_id="search:test",
        candidate_previews=[
            CandidatePreview(
                element_id=figure.element_id,
                element_type=figure.element_type,
                page_id=figure.page_id,
                page_number=page.page_index + 1,
                section_path=list(figure.section_path),
                matched_snippet=placeholder,
                snippet_char_start=0,
                snippet_char_end=len(placeholder),
                snippet_truncated=False,
                snippet_source=SnippetSource.VISUAL_METADATA,
                snippet_source_id=visual_input.visual_asset_id,
                visual_asset_id=visual_input.visual_asset_id,
                matched_by=[RetrievalSource.VISUAL_DENSE],
                preview_source=RetrievalSource.VISUAL_DENSE,
                match_scope=PreviewMatchScope.UNKNOWN,
                visual_rank=1,
                content_availability=figure.content_availability,
            )
        ],
        next_cursor=1,
        exhausted=True,
    )

    class FailingBackend:
        prompt_version = "visual-retrieval-v0.1"

        def describe(self, _):
            raise RuntimeError("temporary VLM failure")

    runtime = _PersistentRuntime.__new__(_PersistentRuntime)
    runtime.args = argparse.Namespace(
        visual_model="mock-vlm",
        visual_descriptor_cache=tmp_path / "descriptors.jsonl",
    )
    runtime._visual_descriptor_backend = FailingBackend()
    runtime._visual_descriptor_lock = __import__("threading").RLock()
    runtime._case_context = __import__("threading").local()
    runtime._case_context.visual_descriptor_calls = []
    runtime._visual_descriptor_records = {}

    enriched = runtime._enrich_visual_candidate_batch(document, tmp_path, batch)

    assert enriched.candidate_previews[0].matched_snippet == placeholder
    assert "temporary VLM failure" in runtime._case_context.visual_descriptor_calls[0][
        "error"
    ]
    assert not (tmp_path / "descriptors.jsonl").exists()


def test_descriptor_diagnostics_survive_later_case_failure(tmp_path: Path) -> None:
    runtime = _PersistentRuntime.__new__(_PersistentRuntime)
    runtime.args = argparse.Namespace(run_key_prefix="test")
    runtime._case_context = __import__("threading").local()

    def fake_search_resources(_document_dir: str):
        runtime._record_visual_descriptor_call(
            {
                "component": "visual_retrieval",
                "error": "temporary VLM failure",
            }
        )
        return object(), object()

    class FailingRunner:
        def run(self, **_):
            raise RuntimeError("later QA failure")

    runtime._search_resources = fake_search_resources
    runtime._runner = lambda: FailingRunner()
    output = tmp_path / "failed-case"

    with pytest.raises(RuntimeError, match="later QA failure"):
        runtime.run_case(
            {
                "case_id": "Q-test",
                "document_dir": str(tmp_path / "document"),
                "question": "What happened?",
            },
            output,
        )

    rows = (output / "visual_descriptor_calls.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["error"] == "temporary VLM failure"


def test_baseline_runtime_profile_rejects_missing_modules(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.runtime_profile = BASELINE_RUNTIME_PROFILE

    with pytest.raises(ValueError, match="profile.*incomplete"):
        _validate_runtime_profile(args)


def test_baseline_runtime_profile_accepts_complete_architecture(tmp_path: Path) -> None:
    args = _args(tmp_path)
    index = tmp_path / "visual-index"
    (index / "shards").mkdir(parents=True)
    (index / "config.json").write_text(
        json.dumps({"model": "visual-index-model"}), encoding="utf-8"
    )
    (index / "state.json").write_text(
        json.dumps({"state": "completed", "pending_image_count": 0}),
        encoding="utf-8",
    )
    (index / "assets.jsonl").write_text("{}\n", encoding="utf-8")
    (index / "shards" / "shard-00000.npz").write_bytes(b"fixture")
    args.runtime_profile = BASELINE_RUNTIME_PROFILE
    args.execution_mode = "persistent"
    args.inference_backend = "vllm"
    args.dense = True
    args.visual_search_index = index
    args.visual_descriptor_cache = tmp_path / "descriptors.jsonl"
    args.visual_descriptor_on_demand = True
    args.multimodal_table_reader = True

    _validate_runtime_profile(args)


def test_descriptor_cache_cannot_be_silently_ignored_in_subprocess_mode(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.visual_descriptor_cache = tmp_path / "descriptors.jsonl"

    with pytest.raises(ValueError, match="requires --execution-mode persistent"):
        _validate_batch_args(args)


def test_persistent_batch_reuses_one_runtime_and_writes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.execution_mode = "persistent"
    args.inference_backend = "vllm"
    args.workers = 2
    cases = [
        {
            "case_id": "A1",
            "document_dir": str(tmp_path / "doc-a"),
            "question": "Question A1?",
        },
        {
            "case_id": "B1",
            "document_dir": str(tmp_path / "doc-b"),
            "question": "Question B1?",
        },
        {
            "case_id": "A2",
            "document_dir": str(tmp_path / "doc-a"),
            "question": "Question A2?",
        },
    ]
    constructed: list[object] = []

    class FakeRuntime:
        def __init__(self, _: argparse.Namespace) -> None:
            constructed.append(self)

        def run_case(self, case: dict[str, object], output: Path) -> None:
            output.mkdir(parents=True)
            (output / "run_manifest.json").write_text(
                json.dumps({"case_id": case["case_id"]}), encoding="utf-8"
            )

    monkeypatch.setattr("scripts.run_model_batch._PersistentRuntime", FakeRuntime)
    monkeypatch.setattr(
        "scripts.run_model_batch._run_callable_with_peak_vram",
        lambda call: (call(), 123)[1],
    )

    manifest = run_batch(cases=cases, args=args)

    assert len(constructed) == 1
    assert manifest["status"] == "completed"
    assert manifest["succeeded"] == 3
    assert [item["case_id"] for item in manifest["cases"]] == ["A1", "A2", "B1"]
    assert all(item["peak_gpu_memory_mib"] == 123 for item in manifest["cases"])
