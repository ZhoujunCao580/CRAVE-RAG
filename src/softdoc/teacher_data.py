"""Review complete reading runs and export audited Controller SFT records."""

from __future__ import annotations

from enum import StrEnum
import json
from pathlib import Path
from typing import Literal, Sequence

from pydantic import Field, field_validator, model_validator

from softdoc.controller import ControllerInput, validate_controller_action
from softdoc.model_runner import ModelPipelineRun, StageCallRecord, load_model_pipeline_run
from softdoc.models import SoftDocModel
from softdoc.prompt_registry import PromptComponent, get_prompt
from softdoc.reading_state import (
    EvidenceCheckInput,
    EvidenceCheckResult,
    apply_evidence_check_result,
)
from softdoc.training_data import OpenAIMessagesSFTRecord, SFTExample


TEACHER_REVIEW_SCHEMA_VERSION = "teacher-review-v0"
CONTROLLER_SFT_DATASET_VERSION = "controller-sft-dataset-v0"
CHECKER_REVIEW_SCHEMA_VERSION = "checker-review-v0"
CHECKER_SFT_DATASET_VERSION = "checker-sft-dataset-v0"
TEACHER_GENERATION_PROTOCOL = "teacher-no-gold-v0"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ControllerStepReview(SoftDocModel):
    """A human/Teacher judgment about one executed Controller decision."""

    controller_call_index: int = Field(ge=0)
    action_id: str = Field(min_length=1)
    training_label_status: ReviewStatus = ReviewStatus.PENDING
    review_note: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def require_rejection_note(self) -> "ControllerStepReview":
        if (
            self.training_label_status == ReviewStatus.REJECTED
            and self.review_note is None
        ):
            raise ValueError("A rejected Controller step requires review_note")
        return self


class TeacherReview(SoftDocModel):
    """Thin labels over one ModelPipelineRun; it does not duplicate the run."""

    schema_version: Literal[TEACHER_REVIEW_SCHEMA_VERSION] = (
        TEACHER_REVIEW_SCHEMA_VERSION
    )
    reading_session_id: str = Field(min_length=1)
    episode_status: ReviewStatus = ReviewStatus.PENDING
    controller_steps: list[ControllerStepReview] = Field(default_factory=list)
    first_corrupted_action_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_review_state(self) -> "TeacherReview":
        call_indexes = [item.controller_call_index for item in self.controller_steps]
        action_ids = [item.action_id for item in self.controller_steps]
        if len(call_indexes) != len(set(call_indexes)):
            raise ValueError("Controller review call indexes must be unique")
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("Controller review action IDs must be unique")

        if self.episode_status != ReviewStatus.PENDING and any(
            item.training_label_status == ReviewStatus.PENDING
            for item in self.controller_steps
        ):
            raise ValueError("A finalized episode cannot contain pending step reviews")
        if self.episode_status == ReviewStatus.ACCEPTED and any(
            item.training_label_status != ReviewStatus.ACCEPTED
            for item in self.controller_steps
        ):
            raise ValueError("An accepted episode requires every Controller step accepted")

        if self.first_corrupted_action_id is not None:
            matching = [
                item
                for item in self.controller_steps
                if item.action_id == self.first_corrupted_action_id
            ]
            if not matching:
                raise ValueError("first_corrupted_action_id is not a reviewed action")
            if matching[0].training_label_status != ReviewStatus.REJECTED:
                raise ValueError("first_corrupted_action_id must identify a rejected step")
        if (
            self.episode_status == ReviewStatus.ACCEPTED
            and self.first_corrupted_action_id is not None
        ):
            raise ValueError("An accepted episode cannot have a corrupted action")
        return self


class CheckerStepReview(SoftDocModel):
    """A review label for one recorded Checker call.

    ``replacement_output`` lets a reviewer preserve the immutable raw run while
    supplying the corrected Teacher delta that should become the SFT target.
    """

    checker_call_index: int = Field(ge=0)
    action_id: str = Field(min_length=1)
    training_label_status: ReviewStatus = ReviewStatus.PENDING
    review_note: str | None = Field(default=None, min_length=1)
    replacement_output: EvidenceCheckResult | None = None

    @model_validator(mode="after")
    def validate_step_review(self) -> "CheckerStepReview":
        if self.training_label_status == ReviewStatus.REJECTED and self.review_note is None:
            raise ValueError("A rejected Checker step requires review_note")
        if self.training_label_status == ReviewStatus.PENDING and self.replacement_output:
            raise ValueError("A pending Checker step cannot have replacement_output")
        if self.replacement_output is not None and (
            self.replacement_output.action_id != self.action_id
        ):
            raise ValueError("Checker replacement_output action_id must match the step")
        return self


class CheckerReview(SoftDocModel):
    """Thin review/correction labels over all Checker calls in one run."""

    schema_version: Literal[CHECKER_REVIEW_SCHEMA_VERSION] = (
        CHECKER_REVIEW_SCHEMA_VERSION
    )
    reading_session_id: str = Field(min_length=1)
    episode_status: ReviewStatus = ReviewStatus.PENDING
    checker_steps: list[CheckerStepReview] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_review_state(self) -> "CheckerReview":
        indexes = [item.checker_call_index for item in self.checker_steps]
        if len(indexes) != len(set(indexes)):
            raise ValueError("Checker review call indexes must be unique")
        if self.episode_status != ReviewStatus.PENDING and any(
            item.training_label_status == ReviewStatus.PENDING
            for item in self.checker_steps
        ):
            raise ValueError("A finalized Checker review cannot contain pending steps")
        if self.episode_status == ReviewStatus.ACCEPTED and any(
            item.training_label_status != ReviewStatus.ACCEPTED
            for item in self.checker_steps
        ):
            raise ValueError("An accepted episode requires every Checker step accepted")
        return self


class ControllerSFTSourceRun(SoftDocModel):
    reading_session_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    root_question_id: str = Field(min_length=1)
    episode_status: ReviewStatus
    exported_example_ids: list[str] = Field(min_length=1)

    @field_validator("exported_example_ids")
    @classmethod
    def validate_example_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Source-run example IDs must be unique")
        return value


class CheckerSFTSourceRun(ControllerSFTSourceRun):
    """One source run contributing reviewed Checker examples."""


class ControllerSFTDatasetManifest(SoftDocModel):
    """Lineage and version binding for one exported Controller dataset."""

    schema_version: Literal[CONTROLLER_SFT_DATASET_VERSION] = (
        CONTROLLER_SFT_DATASET_VERSION
    )
    generation_protocol: Literal[TEACHER_GENERATION_PROTOCOL] = (
        TEACHER_GENERATION_PROTOCOL
    )
    component: Literal[PromptComponent.CONTROLLER] = PromptComponent.CONTROLLER
    prompt_version: str = Field(min_length=1)
    prompt_sha256: str = Field(min_length=64, max_length=64)
    source_pipeline_versions: list[str] = Field(min_length=1)
    source_runs: list[ControllerSFTSourceRun] = Field(min_length=1)
    example_count: int = Field(ge=1)
    internal_records_file: Literal["controller_sft.jsonl"] = "controller_sft.jsonl"
    llama_factory_messages_file: Literal["controller_sft_messages.jsonl"] = (
        "controller_sft_messages.jsonl"
    )
    llama_factory_dataset_info_file: Literal["dataset_info.json"] = (
        "dataset_info.json"
    )

    @model_validator(mode="after")
    def validate_dataset_counts(self) -> "ControllerSFTDatasetManifest":
        session_ids = [item.reading_session_id for item in self.source_runs]
        if len(session_ids) != len(set(session_ids)):
            raise ValueError("Dataset source reading sessions must be unique")
        example_ids = [
            example_id
            for source_run in self.source_runs
            for example_id in source_run.exported_example_ids
        ]
        if len(example_ids) != len(set(example_ids)):
            raise ValueError("Dataset example IDs must be unique")
        if self.example_count != len(example_ids):
            raise ValueError("example_count does not match exported example IDs")
        if len(self.source_pipeline_versions) != len(set(self.source_pipeline_versions)):
            raise ValueError("source_pipeline_versions must be unique")
        return self


class CheckerSFTDatasetManifest(SoftDocModel):
    """Lineage and version binding for one exported Checker dataset."""

    schema_version: Literal[CHECKER_SFT_DATASET_VERSION] = CHECKER_SFT_DATASET_VERSION
    generation_protocol: Literal[TEACHER_GENERATION_PROTOCOL] = (
        TEACHER_GENERATION_PROTOCOL
    )
    component: Literal[PromptComponent.CHECKER] = PromptComponent.CHECKER
    prompt_version: str = Field(min_length=1)
    prompt_sha256: str = Field(min_length=64, max_length=64)
    source_pipeline_versions: list[str] = Field(min_length=1)
    source_runs: list[CheckerSFTSourceRun] = Field(min_length=1)
    example_count: int = Field(ge=1)
    internal_records_file: Literal["checker_sft.jsonl"] = "checker_sft.jsonl"
    llama_factory_messages_file: Literal["checker_sft_messages.jsonl"] = (
        "checker_sft_messages.jsonl"
    )
    llama_factory_dataset_info_file: Literal["dataset_info.json"] = (
        "dataset_info.json"
    )

    @model_validator(mode="after")
    def validate_dataset_counts(self) -> "CheckerSFTDatasetManifest":
        session_ids = [item.reading_session_id for item in self.source_runs]
        if len(session_ids) != len(set(session_ids)):
            raise ValueError("Dataset source reading sessions must be unique")
        example_ids = [
            example_id
            for source_run in self.source_runs
            for example_id in source_run.exported_example_ids
        ]
        if len(example_ids) != len(set(example_ids)):
            raise ValueError("Dataset example IDs must be unique")
        if self.example_count != len(example_ids):
            raise ValueError("example_count does not match exported example IDs")
        if len(self.source_pipeline_versions) != len(set(self.source_pipeline_versions)):
            raise ValueError("source_pipeline_versions must be unique")
        return self


def build_teacher_review_template(run: ModelPipelineRun) -> TeacherReview:
    """Create a pending review with one label slot per Controller call."""

    return TeacherReview(
        reading_session_id=_reading_session_id(run),
        controller_steps=[
            ControllerStepReview(
                controller_call_index=record.call_index,
                action_id=_required_action_id(record),
            )
            for record in _controller_calls(run)
        ],
    )


def build_checker_review_template(run: ModelPipelineRun) -> CheckerReview:
    """Create a pending review with one label slot per recorded Checker call."""

    return CheckerReview(
        reading_session_id=_reading_session_id(run),
        checker_steps=[
            CheckerStepReview(
                checker_call_index=record.call_index,
                action_id=_required_action_id(record),
            )
            for record in _checker_calls(run)
        ],
    )


def validate_teacher_review(run: ModelPipelineRun, review: TeacherReview) -> None:
    """Require the review to cover exactly the Controller calls in the run."""

    if review.reading_session_id != _reading_session_id(run):
        raise ValueError("Teacher review reading_session_id does not match the run")
    expected = [
        (item.call_index, _required_action_id(item)) for item in _controller_calls(run)
    ]
    actual = [
        (item.controller_call_index, item.action_id)
        for item in sorted(review.controller_steps, key=lambda item: item.controller_call_index)
    ]
    if actual != expected:
        raise ValueError("Teacher review must cover every Controller call exactly once")


def validate_checker_review(run: ModelPipelineRun, review: CheckerReview) -> None:
    """Require the review to cover exactly the Checker calls in the run."""

    if review.reading_session_id != _reading_session_id(run):
        raise ValueError("Checker review reading_session_id does not match the run")
    expected = [
        (item.call_index, _required_action_id(item)) for item in _checker_calls(run)
    ]
    actual = [
        (item.checker_call_index, item.action_id)
        for item in sorted(review.checker_steps, key=lambda item: item.checker_call_index)
    ]
    if actual != expected:
        raise ValueError("Checker review must cover every Checker call exactly once")


def build_controller_sft_examples(
    run: ModelPipelineRun,
    review: TeacherReview,
) -> list[SFTExample]:
    """Export only explicitly accepted Controller decisions from one reviewed run."""

    validate_teacher_review(run, review)
    if review.episode_status == ReviewStatus.PENDING:
        raise ValueError("A pending Teacher review cannot be exported")

    controller_prompt = _bound_controller_prompt(run)
    current_prompt = get_prompt(PromptComponent.CONTROLLER)
    if controller_prompt["version"] != current_prompt.version:
        raise ValueError(
            "Run used Controller prompt "
            f"{controller_prompt['version']!r}, but the active prompt is "
            f"{current_prompt.version!r}"
        )
    if controller_prompt["sha256"] != current_prompt.sha256:
        raise ValueError("Run Controller prompt hash does not match the active prompt")

    review_by_index = {
        item.controller_call_index: item for item in review.controller_steps
    }
    examples: list[SFTExample] = []
    for record in _controller_calls(run):
        step_review = review_by_index[record.call_index]
        if step_review.training_label_status != ReviewStatus.ACCEPTED:
            continue
        if not record.succeeded:
            raise ValueError("A failed Controller call cannot be accepted for SFT")
        controller_input = ControllerInput.model_validate(record.input)
        action = validate_controller_action(record.output, controller_input)
        examples.append(
            SFTExample(
                example_id=(
                    f"{review.reading_session_id}:controller:{record.call_index:03d}"
                ),
                component=PromptComponent.CONTROLLER,
                prompt_version=current_prompt.version,
                input_text=json.dumps(
                    controller_input.model_dump(mode="json"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                target=action.model_dump(mode="json"),
            )
        )
    return examples


def build_checker_sft_examples(
    run: ModelPipelineRun,
    review: CheckerReview,
) -> list[SFTExample]:
    """Export accepted Checker decisions, validating each delta atomically."""

    validate_checker_review(run, review)
    if review.episode_status == ReviewStatus.PENDING:
        raise ValueError("A pending Checker review cannot be exported")

    checker_prompt = _bound_prompt(run, PromptComponent.CHECKER)
    current_prompt = get_prompt(PromptComponent.CHECKER)
    if checker_prompt["version"] != current_prompt.version:
        raise ValueError(
            "Run used Checker prompt "
            f"{checker_prompt['version']!r}, but the active prompt is "
            f"{current_prompt.version!r}"
        )
    if checker_prompt["sha256"] != current_prompt.sha256:
        raise ValueError("Run Checker prompt hash does not match the active prompt")

    review_by_index = {item.checker_call_index: item for item in review.checker_steps}
    examples: list[SFTExample] = []
    for record in _checker_calls(run):
        step_review = review_by_index[record.call_index]
        if step_review.training_label_status != ReviewStatus.ACCEPTED:
            continue
        checker_input = EvidenceCheckInput.model_validate(record.input)
        if step_review.replacement_output is not None:
            result = step_review.replacement_output
        else:
            if not record.succeeded:
                raise ValueError(
                    "An accepted failed Checker call requires replacement_output"
                )
            result = EvidenceCheckResult.model_validate(record.output)
        apply_evidence_check_result(checker_input, result)
        examples.append(
            SFTExample(
                example_id=(
                    f"{review.reading_session_id}:checker:{record.call_index:03d}"
                ),
                component=PromptComponent.CHECKER,
                prompt_version=current_prompt.version,
                input_text=json.dumps(
                    checker_input.model_dump(mode="json"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                target=result.model_dump(mode="json"),
            )
        )
    return examples


def build_controller_sft_dataset(
    reviewed_runs: Sequence[tuple[ModelPipelineRun, TeacherReview]],
) -> tuple[ControllerSFTDatasetManifest, list[SFTExample]]:
    """Combine reviewed runs into one deterministic Controller-only dataset."""

    if not reviewed_runs:
        raise ValueError("At least one reviewed run is required")
    all_examples: list[SFTExample] = []
    source_runs: list[ControllerSFTSourceRun] = []
    pipeline_versions: set[str] = set()
    for run, review in reviewed_runs:
        examples = build_controller_sft_examples(run, review)
        if not examples:
            continue
        all_examples.extend(examples)
        pipeline_versions.add(run.pipeline_version)
        source_runs.append(
            ControllerSFTSourceRun(
                reading_session_id=review.reading_session_id,
                document_id=run.document_id,
                root_question_id=run.reading_run.root_question.question_id,
                episode_status=review.episode_status,
                exported_example_ids=[item.example_id for item in examples],
            )
        )
    if not all_examples:
        raise ValueError("Reviewed runs contain no accepted Controller decisions")
    example_ids = [item.example_id for item in all_examples]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("Controller SFT example IDs must be unique across runs")

    prompt = get_prompt(PromptComponent.CONTROLLER)
    manifest = ControllerSFTDatasetManifest(
        prompt_version=prompt.version,
        prompt_sha256=prompt.sha256,
        source_pipeline_versions=sorted(pipeline_versions),
        source_runs=source_runs,
        example_count=len(all_examples),
    )
    return manifest, all_examples


def build_checker_sft_dataset(
    reviewed_runs: Sequence[tuple[ModelPipelineRun, CheckerReview]],
) -> tuple[CheckerSFTDatasetManifest, list[SFTExample]]:
    """Combine reviewed runs into one deterministic Checker-only dataset."""

    if not reviewed_runs:
        raise ValueError("At least one reviewed run is required")
    all_examples: list[SFTExample] = []
    source_runs: list[CheckerSFTSourceRun] = []
    pipeline_versions: set[str] = set()
    for run, review in reviewed_runs:
        examples = build_checker_sft_examples(run, review)
        if not examples:
            continue
        all_examples.extend(examples)
        pipeline_versions.add(run.pipeline_version)
        source_runs.append(
            CheckerSFTSourceRun(
                reading_session_id=review.reading_session_id,
                document_id=run.document_id,
                root_question_id=run.reading_run.root_question.question_id,
                episode_status=review.episode_status,
                exported_example_ids=[item.example_id for item in examples],
            )
        )
    if not all_examples:
        raise ValueError("Reviewed runs contain no accepted Checker decisions")
    example_ids = [item.example_id for item in all_examples]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("Checker SFT example IDs must be unique across runs")

    prompt = get_prompt(PromptComponent.CHECKER)
    manifest = CheckerSFTDatasetManifest(
        prompt_version=prompt.version,
        prompt_sha256=prompt.sha256,
        source_pipeline_versions=sorted(pipeline_versions),
        source_runs=source_runs,
        example_count=len(all_examples),
    )
    return manifest, all_examples


def write_teacher_review(review: TeacherReview, path: Path) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Teacher review already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, review.model_dump(mode="json"))


def load_teacher_review(path: Path) -> TeacherReview:
    return TeacherReview.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_checker_review(review: CheckerReview, path: Path) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Checker review already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, review.model_dump(mode="json"))


def load_checker_review(path: Path) -> CheckerReview:
    return CheckerReview.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_controller_sft_dataset(
    reviewed_runs: Sequence[tuple[ModelPipelineRun, TeacherReview]],
    output_dir: Path,
) -> ControllerSFTDatasetManifest:
    manifest, examples = build_controller_sft_dataset(reviewed_runs)
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"Dataset output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(f"Dataset output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "controller_sft.jsonl").write_text(
        "".join(
            json.dumps(item.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
            + "\n"
            for item in examples
        ),
        encoding="utf-8",
    )
    message_records = [
        OpenAIMessagesSFTRecord.model_validate(
            {"messages": item.training_messages()}
        )
        for item in examples
    ]
    (output_dir / "controller_sft_messages.jsonl").write_text(
        "".join(
            json.dumps(
                item.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for item in message_records
        ),
        encoding="utf-8",
    )
    _write_json(
        output_dir / "dataset_info.json",
        {
            "crave_controller_sft": {
                "file_name": "controller_sft_messages.jsonl",
                "formatting": "sharegpt",
                "columns": {"messages": "messages"},
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                    "system_tag": "system",
                },
            }
        },
    )
    _write_json(output_dir / "dataset_manifest.json", manifest.model_dump(mode="json"))
    return manifest


def write_checker_sft_dataset(
    reviewed_runs: Sequence[tuple[ModelPipelineRun, CheckerReview]],
    output_dir: Path,
) -> CheckerSFTDatasetManifest:
    manifest, examples = build_checker_sft_dataset(reviewed_runs)
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"Dataset output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(f"Dataset output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checker_sft.jsonl").write_text(
        "".join(
            json.dumps(item.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
            + "\n"
            for item in examples
        ),
        encoding="utf-8",
    )
    message_records = [
        OpenAIMessagesSFTRecord.model_validate({"messages": item.training_messages()})
        for item in examples
    ]
    (output_dir / "checker_sft_messages.jsonl").write_text(
        "".join(
            json.dumps(
                item.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for item in message_records
        ),
        encoding="utf-8",
    )
    _write_json(
        output_dir / "dataset_info.json",
        {
            "crave_checker_sft": {
                "file_name": "checker_sft_messages.jsonl",
                "formatting": "sharegpt",
                "columns": {"messages": "messages"},
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                    "system_tag": "system",
                },
            }
        },
    )
    _write_json(output_dir / "dataset_manifest.json", manifest.model_dump(mode="json"))
    return manifest


def load_reviewed_run(run_dir: Path) -> tuple[ModelPipelineRun, TeacherReview]:
    run_dir = Path(run_dir)
    run = load_model_pipeline_run(run_dir)
    review = load_teacher_review(run_dir / "teacher_review.json")
    validate_teacher_review(run, review)
    return run, review


def load_checker_reviewed_run(
    run_dir: Path,
) -> tuple[ModelPipelineRun, CheckerReview]:
    run_dir = Path(run_dir)
    run = load_model_pipeline_run(run_dir)
    review = load_checker_review(run_dir / "checker_review.json")
    validate_checker_review(run, review)
    return run, review


def _controller_calls(run: ModelPipelineRun) -> list[StageCallRecord]:
    return sorted(
        (item for item in run.stage_calls if item.component == "controller"),
        key=lambda item: item.call_index,
    )


def _checker_calls(run: ModelPipelineRun) -> list[StageCallRecord]:
    return sorted(
        (item for item in run.stage_calls if item.component == "checker"),
        key=lambda item: item.call_index,
    )


def _required_action_id(record: StageCallRecord) -> str:
    if record.action_id is None:
        raise ValueError(
            f"{record.component.capitalize()} call {record.call_index} has no action_id"
        )
    return record.action_id


def _reading_session_id(run: ModelPipelineRun) -> str:
    return run.reading_run.evidence_memory.reading_session_id


def _bound_controller_prompt(run: ModelPipelineRun) -> dict[str, str | int]:
    return _bound_prompt(run, PromptComponent.CONTROLLER)


def _bound_prompt(
    run: ModelPipelineRun,
    component: PromptComponent,
) -> dict[str, str | int]:
    matches = [
        item
        for item in run.prompt_bindings
        if item.get("component") == component.value
    ]
    if len(matches) != 1:
        raise ValueError(f"Run must bind exactly one {component.value} prompt")
    binding = matches[0]
    if not isinstance(binding.get("version"), str) or not isinstance(
        binding.get("sha256"), str
    ):
        raise ValueError(f"Run {component.value} prompt binding is incomplete")
    return binding


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
