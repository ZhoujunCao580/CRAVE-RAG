"""Command-line interface for milestone-one parsing and validation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from softdoc.adapters import MinerUAdapter
from softdoc.controller_ollama import OllamaControllerBackend, OllamaControllerConfig
from softdoc.model_backends import (
    ModelBackedReader,
    OllamaAnswererBackend,
    OllamaEvidenceCheckerBackend,
    OllamaModelConfig,
    OllamaStructuredClient,
    OllamaVisualReaderBackend,
)
from softdoc.model_runner import ModelBackedRunner, write_model_pipeline_run
from softdoc.planning import InitialPlanner, OllamaPlannerBackend, OllamaPlannerConfig
from softdoc.pipeline import SoftDocPipeline
from softdoc.prompt_registry import PromptComponent, get_prompt, prompt_manifest
from softdoc.rule_audit import write_rule_coverage_reports
from softdoc.server_readiness import check_server_readiness, readiness_install_hint
from softdoc.serialization import load_document, write_document
from softdoc.store import DocumentStore
from softdoc.reading_environment import (
    DocumentSearchService,
    ReadingEnvironmentConfig,
)
from softdoc.retrieval import (
    DenseIndex,
    FileEmbeddingCache,
    HuggingFaceE5Encoder,
    SearchUnitBuilder,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="softdoc", description="Soft document structure utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_mineru = subparsers.add_parser("parse-mineru", help="Convert a MinerU output directory")
    parse_mineru.add_argument("input_dir", type=Path)
    parse_mineru.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser("validate", help="Validate a serialized soft document")
    validate.add_argument("output_dir", type=Path)

    prompts = subparsers.add_parser(
        "prompts", help="List, inspect, or export frozen model prompts"
    )
    prompt_subparsers = prompts.add_subparsers(dest="prompt_command", required=True)
    prompt_subparsers.add_parser("list", help="Print the prompt manifest")
    prompt_show = prompt_subparsers.add_parser("show", help="Print one prompt")
    prompt_show.add_argument("component", choices=[item.value for item in PromptComponent])
    prompt_show.add_argument(
        "--question",
        help="Root question required when rendering the dynamic Planner prompt",
    )
    prompt_export = prompt_subparsers.add_parser(
        "export", help="Export the complete prompt manifest and prompt text"
    )
    prompt_export.add_argument("--output", type=Path, required=True)

    doctor = subparsers.add_parser(
        "doctor", help="Check a fresh machine for runtime or training readiness"
    )
    doctor.add_argument("--profile", choices=("core", "eval", "train"), default="core")
    doctor.add_argument("--json", action="store_true", dest="as_json")

    run_model = subparsers.add_parser(
        "run-model",
        help="Run Planner, retrieval, Controller, Reader, Checker, and Answerer",
    )
    run_model.add_argument("document_dir", type=Path, help="Serialized SoftDoc directory")
    question_group = run_model.add_mutually_exclusive_group(required=True)
    question_group.add_argument("--question")
    question_group.add_argument("--question-file", type=Path)
    run_model.add_argument("--output", type=Path, required=True)
    run_model.add_argument("--base-url", default="http://localhost:11434")
    run_model.add_argument("--text-model", default="qwen3:8b")
    run_model.add_argument("--visual-model", default="qwen3-vl:4b")
    run_model.add_argument("--timeout", type=float, default=180.0)
    run_model.add_argument("--context-length", type=int, default=8192)
    run_model.add_argument("--action-budget", type=int, default=7)
    run_model.add_argument("--run-key", default="model-v0")
    run_model.add_argument(
        "--dense",
        action="store_true",
        help="Enable multilingual-E5 Dense retrieval in addition to BM25",
    )
    run_model.add_argument("--dense-model", default="intfloat/multilingual-e5-small")
    run_model.add_argument("--dense-model-path", type=Path)
    run_model.add_argument("--dense-device", choices=("auto", "cpu", "cuda"), default="auto")
    run_model.add_argument("--embedding-cache", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    if args.command == "parse-mineru":
        adapter = MinerUAdapter()
        document = SoftDocPipeline(adapter).parse(
            args.input_dir,
            args.output,
        )
        write_document(document, args.output)
        write_rule_coverage_reports([document], args.output)
        print(
            f"Parsed {document.document_id}: {len(document.pages)} pages, "
            f"{len(document.elements)} elements, {len(document.relations)} relations"
        )
        warning_count = len(document.metadata.get("adapter_warnings", []))
        if warning_count:
            print(
                f"Recorded {warning_count} adapter warnings in "
                "debug/adapter_warnings.json"
            )
        return 0
    if args.command == "validate":
        document = load_document(args.output_dir)
        errors = DocumentStore(document).validate_references()
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            return 1
        print(
            f"Valid {document.document_id}: {len(document.pages)} pages, "
            f"{len(document.elements)} elements, {len(document.relations)} relations"
        )
        return 0
    if args.command == "prompts":
        if args.prompt_command == "list":
            print(json.dumps(prompt_manifest(), indent=2, ensure_ascii=False))
            return 0
        if args.prompt_command == "show":
            spec = get_prompt(args.component)
            if spec.component == PromptComponent.PLANNER:
                if not args.question:
                    raise SystemExit("planner prompt requires --question")
                print(spec.render(args.question))
            else:
                print(spec.render())
            return 0
        if args.prompt_command == "export":
            args.output.mkdir(parents=True, exist_ok=True)
            manifest = prompt_manifest()
            for item in PromptComponent:
                spec = get_prompt(item)
                text = (
                    spec.render("<ROOT_QUESTION>")
                    if item == PromptComponent.PLANNER
                    else spec.render()
                )
                (args.output / f"{item.value}__{spec.version}.txt").write_text(
                    text + ("" if text.endswith("\n") else "\n"),
                    encoding="utf-8",
                )
            (args.output / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"Exported {len(manifest)} prompts to {args.output}")
            return 0
    if args.command == "doctor":
        report = check_server_readiness(args.profile)
        payload = report.model_dump()
        if args.as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(f"Profile: {report.profile}")
            print(f"Python: {report.python_version}")
            print(f"Platform: {report.platform}")
            for dependency in report.dependencies:
                marker = "OK" if dependency.available else "MISSING"
                suffix = f" {dependency.version}" if dependency.version else ""
                print(f"[{marker}] {dependency.name}{suffix}")
                if dependency.detail:
                    print(f"  {dependency.detail}")
            if report.cuda_available is not None:
                print(f"CUDA: {report.cuda_available} {report.cuda_device or ''}".rstrip())
            print(f"Ollama executable: {report.ollama_executable or 'not found'}")
            print(f"Ready: {report.ready}")
            if not report.ready:
                print(f"Install hint: {readiness_install_hint(args.profile)}")
        return 0 if report.ready else 1
    if args.command == "run-model":
        if args.output.exists():
            if not args.output.is_dir():
                raise SystemExit(f"run output path is not a directory: {args.output}")
            if any(args.output.iterdir()):
                raise SystemExit(f"run output directory is not empty: {args.output}")
        question = (
            args.question
            if args.question is not None
            else args.question_file.read_text(encoding="utf-8")
        )
        document = load_document(args.document_dir)
        common = {
            "base_url": args.base_url,
            "timeout_seconds": args.timeout,
        }
        planner = InitialPlanner(
            OllamaPlannerBackend(
                OllamaPlannerConfig(model=args.text_model, **common)
            )
        )
        controller = OllamaControllerBackend(
            OllamaControllerConfig(
                model=args.text_model,
                context_length=args.context_length,
                **common,
            )
        )
        text_client = OllamaStructuredClient(
            OllamaModelConfig(
                model=args.text_model,
                context_length=args.context_length,
                **common,
            )
        )
        visual_client = OllamaStructuredClient(
            OllamaModelConfig(
                model=args.visual_model,
                context_length=args.context_length,
                **common,
            )
        )
        reader = ModelBackedReader(
            OllamaVisualReaderBackend(visual_client)
        )
        search_service = None
        if args.dense:
            search_units = SearchUnitBuilder().build(document)
            encoder = HuggingFaceE5Encoder(
                model_name=args.dense_model,
                model_path=args.dense_model_path,
                device=args.dense_device,
                local_files_only=args.dense_model_path is not None,
            )
            cache = (
                FileEmbeddingCache(args.embedding_cache)
                if args.embedding_cache is not None
                else None
            )
            dense_index = DenseIndex(search_units, encoder, cache=cache)
            search_service = DocumentSearchService(
                document,
                search_units=search_units,
                dense_backend=dense_index,
            )
        runner = ModelBackedRunner(
            planner=planner,
            controller=controller,
            reader=reader,
            checker=OllamaEvidenceCheckerBackend(text_client),
            answerer=OllamaAnswererBackend(text_client),
            environment_config=ReadingEnvironmentConfig(
                action_budget=args.action_budget
            ),
        )
        result = runner.run(
            document=document,
            asset_root=args.document_dir,
            question=question,
            run_key=args.run_key,
            search_service=search_service,
        )
        write_model_pipeline_run(result, args.output)
        print(
            f"Run {result.reading_run.status.value}: "
            f"{len(result.reading_run.action_trace.entries)} actions; "
            f"artifacts written to {args.output}"
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
