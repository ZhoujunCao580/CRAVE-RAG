"""Unified launcher for all frozen prompt evaluations.

The component evaluators retain their specialized cases and validators.  This
launcher gives local Windows and remote Linux environments one stable command,
records the exact prompt manifest, and fails immediately when a child evaluator
fails.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from softdoc.prompt_registry import prompt_manifest


TEXT_COMPONENTS = ("planner", "checker", "answerer", "controller")
ALL_COMPONENTS = (*TEXT_COMPONENTS, "visual_reader")


def _script_command(
    component: str,
    *,
    text_model: str,
    visual_model: str,
    base_url: str,
    output_root: Path,
    passes: int,
    visual_config: Path | None,
    visual_corpus: Path | None,
) -> list[str]:
    python = sys.executable
    if component == "planner":
        return [
            python,
            str(ROOT / "scripts" / "evaluate_planner_mock.py"),
            "--model",
            text_model,
            "--base-url",
            base_url,
            "--passes",
            str(passes),
            "--output",
            str(output_root / "planner" / "results.json"),
        ]
    if component == "checker":
        return [
            python,
            str(ROOT / "scripts" / "evaluate_checker_mock.py"),
            "--model",
            text_model,
            "--base-url",
            base_url,
            "--output",
            str(output_root / "checker"),
        ]
    if component == "answerer":
        return [
            python,
            str(ROOT / "scripts" / "evaluate_answerer_mock.py"),
            "--model",
            text_model,
            "--base-url",
            base_url,
            "--output",
            str(output_root / "answerer" / "results.json"),
        ]
    if component == "controller":
        return [
            python,
            str(ROOT / "scripts" / "evaluate_controller_mock.py"),
            "--model",
            text_model,
            "--base-url",
            base_url,
            "--passes",
            str(passes),
            "--output",
            str(output_root / "controller"),
        ]
    if component == "visual_reader":
        if visual_config is None or visual_corpus is None:
            raise ValueError(
                "visual_reader evaluation requires --visual-config and --visual-corpus"
            )
        return [
            python,
            str(ROOT / "scripts" / "run_visual_reader_probe.py"),
            "--model",
            visual_model,
            "--config",
            str(visual_config),
            "--corpus-root",
            str(visual_corpus),
            "--output-dir",
            str(output_root / "visual_reader"),
        ]
    raise ValueError(f"Unsupported component: {component}")


def _expand_components(values: Sequence[str]) -> list[str]:
    expanded: list[str] = []
    for value in values:
        additions = TEXT_COMPONENTS if value == "all_text" else ALL_COMPONENTS if value == "all" else (value,)
        for component in additions:
            if component not in expanded:
                expanded.append(component)
    return expanded


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run reproducible CRAVE-RAG prompt evaluations"
    )
    parser.add_argument(
        "--component",
        action="append",
        choices=(*ALL_COMPONENTS, "all_text", "all"),
        default=[],
        help="Repeat to evaluate multiple components; default: all_text",
    )
    parser.add_argument("--text-model", default="qwen3:8b")
    parser.add_argument("--visual-model", default="qwen3-vl:4b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--passes", type=int, choices=(1, 2), default=2)
    parser.add_argument("--visual-config", type=Path)
    parser.add_argument("--visual-corpus", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path(".runlogs/prompt_eval"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    components = _expand_components(args.component or ["all_text"])
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    commands = [
        _script_command(
            component,
            text_model=args.text_model,
            visual_model=args.visual_model,
            base_url=args.base_url,
            output_root=output_root,
            passes=args.passes,
            visual_config=args.visual_config.resolve() if args.visual_config else None,
            visual_corpus=args.visual_corpus.resolve() if args.visual_corpus else None,
        )
        for component in components
    ]
    run_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "components": components,
        "text_model": args.text_model,
        "visual_model": args.visual_model,
        "base_url": args.base_url,
        "passes": args.passes,
        "prompts": prompt_manifest(),
        "commands": commands,
        "dry_run": args.dry_run,
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    environment = dict(os.environ)
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")
    for component, command in zip(components, commands, strict=True):
        print(f"[{component}] {' '.join(command)}", flush=True)
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        if completed.returncode != 0:
            print(f"{component} failed with exit code {completed.returncode}", file=sys.stderr)
            return completed.returncode
    print(f"Evaluation artifacts: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
