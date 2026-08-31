from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import pytest

from scripts.run_model_batch import build_case_command, load_cases, run_batch


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_root=tmp_path / "batch",
        base_url="http://127.0.0.1:11434",
        text_model="text-model",
        visual_model="visual-model",
        timeout=30.0,
        case_timeout=120.0,
        context_length=4096,
        action_budget=5,
        run_key_prefix="pilot",
        dense=False,
        dense_model="dense-model",
        dense_model_path=None,
        dense_device="cpu",
        embedding_cache=None,
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
