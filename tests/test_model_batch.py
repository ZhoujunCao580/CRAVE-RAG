from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess

import pytest

from scripts.run_model_batch import (
    _batch_lock_path,
    _group_cases_by_document,
    build_case_command,
    load_cases,
    run_batch,
)


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_root=tmp_path / "batch",
        base_url="http://127.0.0.1:11434",
        inference_backend="ollama",
        text_model="text-model",
        visual_model="visual-model",
        timeout=30.0,
        case_timeout=120.0,
        context_length=4096,
        action_budget=5,
        run_key_prefix="pilot",
        execution_mode="subprocess",
        workers=1,
        planner_max_tokens=768,
        controller_max_tokens=512,
        reader_max_tokens=1536,
        checker_max_tokens=1536,
        answerer_max_tokens=768,
        dense=False,
        dense_model="dense-model",
        dense_model_path=None,
        dense_device="cpu",
        embedding_cache=None,
        visual_search_index=None,
        visual_search_model=None,
        visual_search_device="cuda",
        visual_similarity_chunk_elements=16_000_000,
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


def test_group_cases_by_document_is_stable() -> None:
    cases = [
        {"case_id": "A1", "document_dir": "doc-a"},
        {"case_id": "B1", "document_dir": "doc-b"},
        {"case_id": "A2", "document_dir": "doc-a"},
    ]

    grouped = _group_cases_by_document(cases)

    assert [item["case_id"] for item in grouped] == ["A1", "A2", "B1"]


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
