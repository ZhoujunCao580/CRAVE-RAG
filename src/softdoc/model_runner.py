"""Model-backed end-to-end orchestration and auditable run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from pydantic import Field, model_validator

from softdoc.answering import AnswerInput, AnswerResult
from softdoc.controller import ControllerAction, ControllerInput
from softdoc.ids import root_question_id
from softdoc.models import Document, SoftDocModel
from softdoc.planning.models import InitialPlan
from softdoc.prompt_registry import prompt_manifest
from softdoc.reading_environment import (
    AnswererBackend,
    ControllerBackend,
    DocumentSearchService,
    EvidenceCheckerBackend,
    ReaderBackend,
    ReaderContext,
    ReaderOutput,
    ReadingEnvironment,
    ReadingEnvironmentConfig,
    ReadingRunResult,
)
from softdoc.reading_state import ActionTrace, EvidenceCheckInput, EvidenceCheckResult


MODEL_PIPELINE_VERSION = "model-pipeline-v0.1"


class Planner(Protocol):
    def create_plan(self, question: str) -> InitialPlan: ...


class StageCallRecord(SoftDocModel):
    """One model-facing stage input and its validated output."""

    component: str
    call_index: int = Field(ge=0)
    input: dict[str, Any]
    output: dict[str, Any]
    succeeded: bool = True
    action_id: str | None = Field(default=None, min_length=1)


class ModelPipelineRun(SoftDocModel):
    """In-memory complete run; the writer stores large stages separately."""

    pipeline_version: str = MODEL_PIPELINE_VERSION
    document_id: str
    question: str
    plan: InitialPlan
    reading_run: ReadingRunResult
    prompt_bindings: list[dict[str, str | int]] = Field(default_factory=prompt_manifest)
    stage_calls: list[StageCallRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_stage_calls(self) -> "ModelPipelineRun":
        keys = [(item.component, item.call_index) for item in self.stage_calls]
        if len(keys) != len(set(keys)):
            raise ValueError("Stage call component/index pairs must be unique")
        for record in self.stage_calls:
            if record.component in {"controller", "reader", "checker"} and (
                record.action_id is None
            ):
                raise ValueError(
                    f"{record.component} stage call {record.call_index} requires action_id"
                )
        controller_action_ids = [
            item.action_id
            for item in self.stage_calls
            if item.component == "controller"
        ]
        if len(controller_action_ids) != len(set(controller_action_ids)):
            raise ValueError("Controller stage calls must reference unique action IDs")
        return self


class _RecordingController:
    def __init__(self, backend: ControllerBackend, records: list[StageCallRecord]) -> None:
        self.backend = backend
        self.records = records

    def decide(self, controller_input: ControllerInput) -> ControllerAction | dict[str, Any]:
        output = self.backend.decide(controller_input)
        output_dict = (
            output.model_dump(mode="json")
            if isinstance(output, SoftDocModel)
            else dict(output)
        )
        self.records.append(
            StageCallRecord(
                component="controller",
                call_index=_next_index(self.records, "controller"),
                input=controller_input.model_dump(mode="json"),
                output=output_dict,
            )
        )
        return output


class _RecordingReader:
    def __init__(self, backend: ReaderBackend, records: list[StageCallRecord]) -> None:
        self.backend = backend
        self.records = records

    def read(self, context: ReaderContext) -> ReaderOutput:
        input_payload = {
            "action_id": context.action_id,
            "question_id": context.question_id,
            "local_problem": context.local_problem,
            "document_id": context.document.document_id,
            "inputs": [item.model_dump(mode="json") for item in context.inputs],
        }
        try:
            output = self.backend.read(context)
        except Exception as exc:
            self.records.append(
                StageCallRecord(
                    component="reader",
                    call_index=_next_index(self.records, "reader"),
                    input=input_payload,
                    output={"error_type": type(exc).__name__, "error": str(exc)},
                    succeeded=False,
                    action_id=context.action_id,
                )
            )
            raise
        self.records.append(
            StageCallRecord(
                component="reader",
                call_index=_next_index(self.records, "reader"),
                input=input_payload,
                output=output.model_dump(mode="json"),
                action_id=context.action_id,
            )
        )
        return output


class _RecordingChecker:
    def __init__(
        self,
        backend: EvidenceCheckerBackend,
        records: list[StageCallRecord],
    ) -> None:
        self.backend = backend
        self.records = records

    def check(self, checker_input: EvidenceCheckInput) -> EvidenceCheckResult:
        input_payload = checker_input.model_dump(mode="json")
        try:
            output = self.backend.check(checker_input)
        except Exception as exc:
            self.records.append(
                StageCallRecord(
                    component="checker",
                    call_index=_next_index(self.records, "checker"),
                    input=input_payload,
                    output={"error_type": type(exc).__name__, "error": str(exc)},
                    succeeded=False,
                    action_id=checker_input.action_id,
                )
            )
            raise
        self.records.append(
            StageCallRecord(
                component="checker",
                call_index=_next_index(self.records, "checker"),
                input=input_payload,
                output=output.model_dump(mode="json"),
                action_id=checker_input.action_id,
            )
        )
        return output


class _RecordingAnswerer:
    def __init__(self, backend: AnswererBackend, records: list[StageCallRecord]) -> None:
        self.backend = backend
        self.records = records

    def answer(self, answer_input: AnswerInput) -> AnswerResult:
        input_payload = answer_input.model_dump(mode="json")
        try:
            output = self.backend.answer(answer_input)
        except Exception as exc:
            self.records.append(
                StageCallRecord(
                    component="answerer",
                    call_index=_next_index(self.records, "answerer"),
                    input=input_payload,
                    output={"error_type": type(exc).__name__, "error": str(exc)},
                    succeeded=False,
                )
            )
            raise
        self.records.append(
            StageCallRecord(
                component="answerer",
                call_index=_next_index(self.records, "answerer"),
                input=input_payload,
                output=output.model_dump(mode="json"),
            )
        )
        return output


class ModelBackedRunner:
    """Join Planner and ReadingEnvironment without duplicating their contracts."""

    def __init__(
        self,
        *,
        planner: Planner,
        controller: ControllerBackend,
        reader: ReaderBackend,
        checker: EvidenceCheckerBackend,
        answerer: AnswererBackend,
        environment_config: ReadingEnvironmentConfig | None = None,
    ) -> None:
        self.planner = planner
        self.controller = controller
        self.reader = reader
        self.checker = checker
        self.answerer = answerer
        self.environment_config = environment_config or ReadingEnvironmentConfig()

    def run(
        self,
        *,
        document: Document,
        asset_root: Path,
        question: str,
        run_key: str = "model-v0",
        question_id: str | None = None,
        search_service: DocumentSearchService | None = None,
    ) -> ModelPipelineRun:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("The model-backed run question must not be blank")
        plan = self.planner.create_plan(normalized_question)
        records = [
            StageCallRecord(
                component="planner",
                call_index=0,
                input={"question": normalized_question},
                output=plan.model_dump(mode="json"),
            )
        ]
        environment = ReadingEnvironment(
            document,
            asset_root=Path(asset_root),
            controller=_RecordingController(self.controller, records),
            reader=_RecordingReader(self.reader, records),
            checker=_RecordingChecker(self.checker, records),
            answerer=_RecordingAnswerer(self.answerer, records),
            search_service=search_service,
            config=self.environment_config,
        )
        result = environment.run_with_plan(
            root_question_id=(
                question_id or root_question_id(normalized_question)
            ),
            plan=plan,
            run_key=run_key,
        )
        _bind_controller_action_ids(records, result.action_trace)
        return ModelPipelineRun(
            document_id=document.document_id,
            question=normalized_question,
            plan=plan,
            reading_run=result,
            stage_calls=records,
        )


def write_model_pipeline_run(run: ModelPipelineRun, output_dir: Path) -> None:
    """Write a non-overwriting, module-by-module audit packet."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"Run output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(f"Run output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    calls_by_component: dict[str, list[StageCallRecord]] = {}
    for record in run.stage_calls:
        calls_by_component.setdefault(record.component, []).append(record)

    _write_json(output_dir / "planner.json", run.plan.model_dump(mode="json"))
    _write_json(
        output_dir / "reading_run.json",
        run.reading_run.model_dump(mode="json"),
    )
    for component in ("controller", "reader", "checker", "answerer"):
        _write_jsonl(
            output_dir / f"{component}_calls.jsonl",
            [item.model_dump(mode="json") for item in calls_by_component.get(component, [])],
        )

    manifest = {
        "pipeline_version": run.pipeline_version,
        "reading_session_id": run.reading_run.evidence_memory.reading_session_id,
        "root_question_id": run.reading_run.root_question.question_id,
        "document_id": run.document_id,
        "question": run.question,
        "status": run.reading_run.status.value,
        "prompts": run.prompt_bindings,
        "call_counts": {
            component: len(records)
            for component, records in sorted(calls_by_component.items())
        },
        "stage_call_order": [
            {"component": record.component, "call_index": record.call_index}
            for record in run.stage_calls
        ],
        "answer": (
            run.reading_run.answer.model_dump(mode="json")
            if run.reading_run.answer is not None
            else None
        ),
    }
    _write_json(output_dir / "run_manifest.json", manifest)


def load_model_pipeline_run(input_dir: Path) -> ModelPipelineRun:
    """Load one audit packet written by :func:`write_model_pipeline_run`."""

    input_dir = Path(input_dir)
    manifest = json.loads((input_dir / "run_manifest.json").read_text(encoding="utf-8"))
    plan = InitialPlan.model_validate_json(
        (input_dir / "planner.json").read_text(encoding="utf-8")
    )
    reading_run = ReadingRunResult.model_validate_json(
        (input_dir / "reading_run.json").read_text(encoding="utf-8")
    )
    loaded_records = [
        StageCallRecord(
            component="planner",
            call_index=0,
            input={"question": manifest["question"]},
            output=plan.model_dump(mode="json"),
        )
    ]
    for component in ("controller", "reader", "checker", "answerer"):
        path = input_dir / f"{component}_calls.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing model call log: {path}")
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                loaded_records.append(StageCallRecord.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: invalid stage call: {exc}") from exc
    order = manifest.get("stage_call_order")
    if order is None:
        records = loaded_records
    else:
        by_key = {
            (record.component, record.call_index): record
            for record in loaded_records
        }
        requested_keys = [
            (item["component"], item["call_index"])
            for item in order
        ]
        if len(requested_keys) != len(set(requested_keys)):
            raise ValueError("run_manifest stage_call_order contains duplicate keys")
        if set(requested_keys) != set(by_key):
            raise ValueError(
                "run_manifest stage_call_order does not cover every model call"
            )
        records = [by_key[key] for key in requested_keys]
    return ModelPipelineRun(
        pipeline_version=manifest["pipeline_version"],
        document_id=manifest["document_id"],
        question=manifest["question"],
        plan=plan,
        reading_run=reading_run,
        prompt_bindings=manifest["prompts"],
        stage_calls=records,
    )


def _bind_controller_action_ids(
    records: list[StageCallRecord], action_trace: ActionTrace
) -> None:
    """Bind model decisions to executed actions, excluding automatic Exact reads."""

    controller_records = [item for item in records if item.component == "controller"]
    controller_entries = [
        item
        for item in action_trace.entries
        if item.metadata.get("trigger") != "exact_anchor"
    ]
    if len(controller_records) != len(controller_entries):
        raise ValueError(
            "Controller call count does not match non-automatic ActionTrace entries"
        )
    for record, entry in zip(controller_records, controller_entries, strict=True):
        proposed_name = record.output.get("action")
        if proposed_name != entry.action_name:
            raise ValueError(
                "Controller output/action trace mismatch at call "
                f"{record.call_index}: {proposed_name!r} != {entry.action_name!r}"
            )
        record.action_id = entry.action_id


def _next_index(records: list[StageCallRecord], component: str) -> int:
    return sum(item.component == component for item in records)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
