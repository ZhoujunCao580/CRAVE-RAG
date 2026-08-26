"""Canonical logs and derived views for evidence-directed document reading.

This module deliberately contains no Controller, Checker model, recall, or
model client. It freezes the state contracts and the deterministic application
of a validated Checker delta:

* ``ObservationStore`` is the canonical, append-only-by-contract read history.
* ``EvidenceMemory`` is the minimal Checker-owned evidence state.
* ``ActionTrace`` is the canonical action history.
* ``ExplorationState`` is a derived Controller working view, never a third log.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Self

from pydantic import Field, field_validator, model_validator

from softdoc.ids import evidence_id
from softdoc.models import Relation, RelationStatus, RelationType, SoftDocModel
from softdoc.retrieval.models import SearchSession


NormalizedRegion = tuple[float, float, float, float]


def _validate_normalized_region(value: NormalizedRegion) -> NormalizedRegion:
    x1, y1, x2, y2 = value
    if not 0.0 <= x1 < x2 <= 1.0 or not 0.0 <= y1 < y2 <= 1.0:
        raise ValueError(
            "A normalized region must satisfy 0 <= x1 < x2 <= 1 and "
            "0 <= y1 < y2 <= 1"
        )
    return value


def _unique_nonblank(values: list[str], *, label: str) -> list[str]:
    if any(not value.strip() for value in values):
        raise ValueError(f"{label} must not contain blank IDs")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")
    return values


class ReaderKind(str, Enum):
    TEXT = "text"
    TABLE = "table"
    VISUAL = "visual"
    PAGE = "page"


class ReadingSourceType(str, Enum):
    PAGE = "page"
    SECTION = "section"
    ELEMENT = "element"
    TABLE_VIEW = "table_view"
    TABLE_CELL = "table_cell"
    REGION = "region"


class ReadRepresentation(str, Enum):
    ELEMENT_TEXT = "element_text"
    TABLE_VIEW = "table_view"
    ELEMENT_VISUAL = "element_visual"
    PAGE_VISUAL = "page_visual"
    REGION_CROP = "region_crop"


class ReadInput(SoftDocModel):
    """One concrete input supplied to a Reader in one Controller action."""

    input_id: str = Field(min_length=1, pattern=r"^I[1-9][0-9]*$")
    source_id: str = Field(min_length=1)
    source_type: ReadingSourceType
    representation: ReadRepresentation
    document_id: str = Field(min_length=1)
    page_id: str = Field(min_length=1)
    element_id: str | None = Field(default=None, min_length=1)
    table_view_id: str | None = Field(default=None, min_length=1)
    cell_id: str | None = Field(default=None, min_length=1)
    visual_asset_id: str | None = Field(default=None, min_length=1)
    bbox: NormalizedRegion | None = None
    visual_asset_path: Path | None = None

    @field_validator("bbox")
    @classmethod
    def validate_bbox(
        cls, value: NormalizedRegion | None
    ) -> NormalizedRegion | None:
        return None if value is None else _validate_normalized_region(value)

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        element_sources = {
            ReadingSourceType.ELEMENT,
            ReadingSourceType.TABLE_VIEW,
            ReadingSourceType.TABLE_CELL,
        }
        table_sources = {
            ReadingSourceType.TABLE_VIEW,
            ReadingSourceType.TABLE_CELL,
        }
        if self.source_type in element_sources and self.element_id is None:
            raise ValueError(f"A {self.source_type.value} source requires element_id")
        if self.source_type in table_sources and self.table_view_id is None:
            raise ValueError(f"A {self.source_type.value} source requires table_view_id")
        if self.source_type == ReadingSourceType.TABLE_CELL and self.cell_id is None:
            raise ValueError("A table_cell source requires cell_id")
        if self.source_type == ReadingSourceType.REGION:
            if (
                self.visual_asset_id is None
                or self.bbox is None
            ):
                raise ValueError(
                    "A region source requires visual_asset_id and bbox"
                )

        visual_representations = {
            ReadRepresentation.ELEMENT_VISUAL,
            ReadRepresentation.PAGE_VISUAL,
            ReadRepresentation.REGION_CROP,
        }
        if self.representation in visual_representations and (
            self.visual_asset_id is None
            or self.visual_asset_path is None
        ):
            raise ValueError(
                "A visual representation requires visual_asset_id and "
                "visual_asset_path"
            )
        if (
            self.representation == ReadRepresentation.ELEMENT_TEXT
            and self.element_id is None
        ):
            raise ValueError("element_text requires element_id")
        if (
            self.representation == ReadRepresentation.TABLE_VIEW
            and self.table_view_id is None
        ):
            raise ValueError("table_view representation requires table_view_id")
        if (
            self.representation == ReadRepresentation.REGION_CROP
            and self.bbox is None
        ):
            raise ValueError("region_crop requires bbox")
        return self


class ObservationSourceRef(SoftDocModel):
    """A precise grounding inside an input actually supplied to the Reader."""

    input_id: str = Field(min_length=1, pattern=r"^I[1-9][0-9]*$")
    cell_id: str | None = Field(default=None, min_length=1)
    bbox: NormalizedRegion | None = None

    @field_validator("bbox")
    @classmethod
    def validate_bbox(
        cls, value: NormalizedRegion | None
    ) -> NormalizedRegion | None:
        return None if value is None else _validate_normalized_region(value)

class ObservationLimitation(SoftDocModel):
    description: str = Field(min_length=1)
    input_ids: list[str] = Field(default_factory=list)

    @field_validator("input_ids")
    @classmethod
    def validate_input_ids(cls, value: list[str]) -> list[str]:
        _unique_nonblank(value, label="Limitation input IDs")
        if any(not item.startswith("I") for item in value):
            raise ValueError("Limitation input IDs must use local I1/I2 aliases")
        return value


class StoredObservation(SoftDocModel):
    """One concrete Reader fact before Checker evidence promotion."""

    observation_id: str = Field(min_length=1)
    action_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    sources: list[ObservationSourceRef] = Field(min_length=1)

    @field_validator("sources")
    @classmethod
    def validate_unique_sources(
        cls, value: list[ObservationSourceRef]
    ) -> list[ObservationSourceRef]:
        ids = [item.input_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("Observation source IDs must be unique")
        return value


class ReadRecord(SoftDocModel):
    """Canonical record of one read, including reads with no Observation."""

    action_id: str = Field(min_length=1)
    reader_kind: ReaderKind
    document_id: str = Field(min_length=1)
    subquestion_id: str | None = Field(default=None, min_length=1)
    local_problem: str = Field(min_length=1)
    inputs: list[ReadInput] = Field(min_length=1)
    observation_ids: list[str] = Field(default_factory=list)
    limitations: list[ObservationLimitation] = Field(default_factory=list)

    @field_validator("observation_ids")
    @classmethod
    def validate_observation_ids(cls, value: list[str]) -> list[str]:
        return _unique_nonblank(value, label="ReadRecord observation IDs")

    @model_validator(mode="after")
    def validate_inputs(self) -> Self:
        input_ids = [item.input_id for item in self.inputs]
        _unique_nonblank(input_ids, label="ReadRecord input IDs")
        visual_asset_ids = [
            item.visual_asset_id
            for item in self.inputs
            if item.visual_asset_id is not None
        ]
        _unique_nonblank(visual_asset_ids, label="ReadRecord visual_asset_ids")
        if any(item.document_id != self.document_id for item in self.inputs):
            raise ValueError("Read inputs must belong to the ReadRecord document")
        known_inputs = set(input_ids)
        for limitation in self.limitations:
            if not set(limitation.input_ids).issubset(known_inputs):
                raise ValueError("Limitation references an unknown read input")
        return self


class ObservationStore(SoftDocModel):
    """Append-only-by-contract source of truth for completed read requests."""

    reading_session_id: str = Field(min_length=1)
    root_question_id: str = Field(min_length=1)
    read_records: list[ReadRecord] = Field(default_factory=list)
    observations: list[StoredObservation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        record_ids = [record.action_id for record in self.read_records]
        _unique_nonblank(record_ids, label="ObservationStore ReadRecord action IDs")
        observations_by_id = {
            observation.observation_id: observation for observation in self.observations
        }
        if len(observations_by_id) != len(self.observations):
            raise ValueError("ObservationStore observation_ids must be unique")

        referenced_ids: list[str] = []
        for record in self.read_records:
            inputs_by_id = {item.input_id: item for item in record.inputs}
            for observation_id in record.observation_ids:
                observation = observations_by_id.get(observation_id)
                if observation is None:
                    raise ValueError(
                        f"ReadRecord references unknown Observation: {observation_id}"
                    )
                if observation.action_id != record.action_id:
                    raise ValueError("Observation action_id must match its ReadRecord")
                for source in observation.sources:
                    requested = inputs_by_id.get(source.input_id)
                    if requested is None:
                        raise ValueError(
                            "Observation grounding references an unknown read input"
                        )
                    if source.cell_id is not None and (
                        requested.source_type != ReadingSourceType.TABLE_VIEW
                        and requested.source_type != ReadingSourceType.TABLE_CELL
                    ):
                        raise ValueError("Only table inputs may ground a cell_id")
                referenced_ids.append(observation_id)

        if set(referenced_ids) != set(observations_by_id):
            raise ValueError(
                "Every stored Observation must be referenced by one ReadRecord"
            )
        if len(referenced_ids) != len(set(referenced_ids)):
            raise ValueError(
                "A stored Observation must belong to exactly one ReadRecord"
            )
        return self


class EvidenceStatus(str, Enum):
    INCOMPLETE = "incomplete"
    READY = "ready"


class QuestionStatus(str, Enum):
    INCOMPLETE = "incomplete"
    SATISFIED = "satisfied"


class QuestionState(SoftDocModel):
    """One Planner question plus its runtime evidence status."""

    question_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    status: QuestionStatus = QuestionStatus.INCOMPLETE

    @field_validator("depends_on")
    @classmethod
    def validate_dependencies(cls, value: list[str]) -> list[str]:
        return _unique_nonblank(value, label="Question dependency IDs")


class EvidenceItem(SoftDocModel):
    evidence_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    observation_ids: list[str] = Field(min_length=1)
    supports_question_ids: list[str] = Field(min_length=1)

    @field_validator("observation_ids", "supports_question_ids")
    @classmethod
    def validate_id_lists(cls, value: list[str], info: object) -> list[str]:
        label = (
            "Evidence observation IDs"
            if getattr(info, "field_name", None) == "observation_ids"
            else "Evidence supported question IDs"
        )
        return _unique_nonblank(value, label=label)


class CurrentTarget(SoftDocModel):
    """The only question currently pursued and its remaining evidence gap."""

    question_id: str = Field(min_length=1)
    gap_description: str = Field(min_length=1)


def _validate_question_dag(questions: list[QuestionState]) -> None:
    """Reject dependency cycles while preserving Planner order elsewhere."""

    dependencies = {
        question.question_id: question.depends_on for question in questions
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(question_id: str) -> None:
        if question_id in visited:
            return
        if question_id in visiting:
            raise ValueError("Question dependencies must form an acyclic graph")
        visiting.add(question_id)
        for dependency_id in dependencies[question_id]:
            visit(dependency_id)
        visiting.remove(question_id)
        visited.add(question_id)

    for question in questions:
        visit(question.question_id)


def select_next_runnable_question(
    questions: list[QuestionState],
) -> QuestionState | None:
    """Return the first unfinished dependency-ready question in stable order."""

    questions_by_id = {question.question_id: question for question in questions}
    for question in questions:
        if question.status == QuestionStatus.SATISFIED:
            continue
        if all(
            questions_by_id[dependency_id].status == QuestionStatus.SATISFIED
            for dependency_id in question.depends_on
        ):
            return question
    return None


class EvidenceMemory(SoftDocModel):
    """Root-scoped evidence shared by all current and deferred subquestions."""

    reading_session_id: str = Field(min_length=1)
    root_question_id: str = Field(min_length=1)
    root_status: EvidenceStatus = EvidenceStatus.INCOMPLETE
    questions: list[QuestionState] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    current_target: CurrentTarget | None = None

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        question_ids = [item.question_id for item in self.questions]
        _unique_nonblank(question_ids, label="Evidence question IDs")
        if self.root_question_id in question_ids:
            raise ValueError("Root must not be duplicated in the question list")
        questions_by_id = {item.question_id: item for item in self.questions}
        for question in self.questions:
            if question.question_id in question.depends_on:
                raise ValueError("A question cannot depend on itself")
            unknown_dependencies = set(question.depends_on).difference(questions_by_id)
            if unknown_dependencies:
                raise ValueError(
                    f"Question {question.question_id} has unknown dependencies: "
                    + ", ".join(sorted(unknown_dependencies))
                )
        _validate_question_dag(self.questions)
        evidence_ids = [item.evidence_id for item in self.evidence]
        _unique_nonblank(evidence_ids, label="Evidence IDs")
        known_question_ids = {self.root_question_id, *question_ids}
        for item in self.evidence:
            unknown = set(item.supports_question_ids).difference(known_question_ids)
            if unknown:
                raise ValueError(
                    f"Evidence {item.evidence_id} supports unknown questions: "
                    + ", ".join(sorted(unknown))
                )
        if self.current_target is not None:
            target_id = self.current_target.question_id
            if target_id not in known_question_ids:
                raise ValueError("current_target references an unknown question_id")
            if target_id != self.root_question_id:
                target = questions_by_id[target_id]
                if target.status == QuestionStatus.SATISFIED:
                    raise ValueError("current_target cannot reference a satisfied question")
                unsatisfied_dependencies = [
                    dependency_id
                    for dependency_id in target.depends_on
                    if questions_by_id[dependency_id].status
                    != QuestionStatus.SATISFIED
                ]
                if unsatisfied_dependencies:
                    raise ValueError(
                        "current_target has unsatisfied dependencies: "
                        + ", ".join(unsatisfied_dependencies)
                    )
        if self.root_status == EvidenceStatus.READY and self.current_target is not None:
            raise ValueError("Ready EvidenceMemory cannot have a current_target")
        if self.root_status == EvidenceStatus.INCOMPLETE and self.current_target is None:
            raise ValueError("Incomplete EvidenceMemory requires one current_target")
        return self


def initialize_evidence_memory(
    *,
    reading_session_id: str,
    root_question_id: str,
    root_question_text: str,
    questions: list[QuestionState],
) -> EvidenceMemory:
    """Create runtime memory and deterministically select its first target."""

    normalized_questions = [
        question.model_copy(update={"status": QuestionStatus.INCOMPLETE})
        for question in questions
    ]
    next_question = select_next_runnable_question(normalized_questions)
    current_target = CurrentTarget(
        question_id=(
            next_question.question_id if next_question is not None else root_question_id
        ),
        gap_description=(
            next_question.text if next_question is not None else root_question_text
        ),
    )
    return EvidenceMemory(
        reading_session_id=reading_session_id,
        root_question_id=root_question_id,
        questions=normalized_questions,
        current_target=current_target,
    )


def register_deferred_question(
    memory: EvidenceMemory,
    *,
    question_id: str,
    text: str,
    depends_on: list[str] | None = None,
    max_questions: int = 6,
    max_depth: int = 4,
) -> EvidenceMemory:
    """Register one program-approved Deferred Planner proposal.

    Deferred questions may only be registered after the current static plan is
    exhausted and Root remains the current target. The environment, not the
    model, supplies the globally unique ``question_id`` and validates the DAG.
    """

    if max_questions < 1 or max_depth < 1:
        raise ValueError("Deferred Planner limits must be positive")
    if memory.root_status != EvidenceStatus.INCOMPLETE:
        raise ValueError("Cannot register a question after Root is ready")
    if (
        memory.current_target is None
        or memory.current_target.question_id != memory.root_question_id
    ):
        raise ValueError(
            "Deferred questions may be registered only when Root is the current target"
        )
    if any(
        question.status == QuestionStatus.INCOMPLETE for question in memory.questions
    ):
        raise ValueError("The existing plan must be exhausted before deferring")
    if len(memory.questions) >= max_questions:
        raise ValueError("Deferred question would exceed max_questions")

    new_question = QuestionState(
        question_id=question_id,
        text=text,
        depends_on=depends_on or [],
    )
    next_questions = [*memory.questions, new_question]
    candidate = memory.model_copy(
        update={
            "questions": next_questions,
            "current_target": CurrentTarget(
                question_id=question_id,
                gap_description=text,
            ),
        }
    )
    candidate = EvidenceMemory.model_validate(candidate.model_dump(mode="python"))

    depths: dict[str, int] = {}
    questions_by_id = {
        question.question_id: question for question in candidate.questions
    }

    def depth(current_id: str) -> int:
        if current_id in depths:
            return depths[current_id]
        current = questions_by_id[current_id]
        value = 1 + max((depth(item) for item in current.depends_on), default=0)
        depths[current_id] = value
        return value

    if max((depth(item.question_id) for item in candidate.questions), default=0) > max_depth:
        raise ValueError("Deferred question would exceed max_depth")
    return candidate


class RootQuestion(SoftDocModel):
    """Root objective supplied to the Checker without duplicating current target."""

    question_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class EvidenceCheckInput(SoftDocModel):
    """Complete state visible to one Evidence Checker invocation.

    The Checker receives the full current EvidenceMemory.  Only its output is a
    delta, so using a delta never hides existing Evidence from the Checker.
    """

    action_id: str = Field(min_length=1)
    root_question: RootQuestion
    evidence_memory: EvidenceMemory
    observations: list[StoredObservation] = Field(min_length=1)
    limitations: list[ObservationLimitation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_context(self) -> Self:
        if self.evidence_memory.root_question_id != self.root_question.question_id:
            raise ValueError("Checker Root question must match EvidenceMemory")
        observation_ids = [item.observation_id for item in self.observations]
        _unique_nonblank(observation_ids, label="Checker Observation IDs")
        if any(item.action_id != self.action_id for item in self.observations):
            raise ValueError("Checker Observations must belong to its action_id")
        return self


class ObservationAssessment(SoftDocModel):
    observation_id: str = Field(min_length=1)
    used_for_evidence: bool
    assessment: str = Field(min_length=1)


class EvidenceAddition(SoftDocModel):
    """New Evidence content; the program assigns its stable evidence_id."""

    statement: str = Field(min_length=1)
    observation_ids: list[str] = Field(min_length=1)
    supports_question_ids: list[str] = Field(min_length=1)

    @field_validator("observation_ids", "supports_question_ids")
    @classmethod
    def validate_id_lists(cls, value: list[str], info: object) -> list[str]:
        label = (
            "Evidence addition Observation IDs"
            if getattr(info, "field_name", None) == "observation_ids"
            else "Evidence addition supported question IDs"
        )
        return _unique_nonblank(value, label=label)


class EvidenceReplacement(EvidenceAddition):
    """Complete replacement content for one existing Evidence item."""

    evidence_id: str = Field(min_length=1)


class EvidenceRemoval(SoftDocModel):
    evidence_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class EvidenceUpdates(SoftDocModel):
    add: list[EvidenceAddition] = Field(default_factory=list)
    replace: list[EvidenceReplacement] = Field(default_factory=list)
    remove: list[EvidenceRemoval] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_targets(self) -> Self:
        replace_ids = [item.evidence_id for item in self.replace]
        remove_ids = [item.evidence_id for item in self.remove]
        _unique_nonblank(replace_ids, label="Evidence replacement IDs")
        _unique_nonblank(remove_ids, label="Evidence removal IDs")
        overlap = set(replace_ids).intersection(remove_ids)
        if overlap:
            raise ValueError(
                "Evidence cannot be both replaced and removed: "
                + ", ".join(sorted(overlap))
            )
        return self


class EvidenceCheckResult(SoftDocModel):
    """Validated Checker delta; this is not another canonical store."""

    action_id: str = Field(min_length=1)
    observation_assessments: list[ObservationAssessment] = Field(min_length=1)
    evidence_updates: EvidenceUpdates = Field(default_factory=EvidenceUpdates)
    current_target_status: QuestionStatus
    root_status: EvidenceStatus
    remaining_gap_description: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        assessment_ids = [item.observation_id for item in self.observation_assessments]
        _unique_nonblank(assessment_ids, label="Checker assessment Observation IDs")
        if self.current_target_status == QuestionStatus.INCOMPLETE:
            if self.remaining_gap_description is None:
                raise ValueError(
                    "An incomplete current target requires a remaining gap description"
                )
        elif self.remaining_gap_description is not None:
            raise ValueError(
                "A satisfied current target must not describe another gap"
            )
        if (
            self.root_status == EvidenceStatus.READY
            and self.current_target_status != QuestionStatus.SATISFIED
        ):
            raise ValueError("A ready Root requires a satisfied current target")
        return self


def apply_evidence_check_result(
    checker_input: EvidenceCheckInput,
    result: EvidenceCheckResult,
) -> EvidenceMemory:
    """Apply one Checker delta to a copy and return a fully validated memory.

    The input memory is never mutated.  Any bad reference or invalid final state
    raises before the caller can replace the canonical EvidenceMemory, giving
    the caller an all-or-nothing commit boundary.
    """

    if result.action_id != checker_input.action_id:
        raise ValueError("Checker result action_id does not match its input")

    new_observation_ids = {
        observation.observation_id for observation in checker_input.observations
    }
    assessment_ids = {
        assessment.observation_id
        for assessment in result.observation_assessments
    }
    if assessment_ids != new_observation_ids:
        raise ValueError(
            "Checker must assess every new Observation exactly once and no others"
        )

    existing_items = list(checker_input.evidence_memory.evidence)
    existing_by_id = {item.evidence_id: item for item in existing_items}
    replace_by_id = {
        item.evidence_id: item for item in result.evidence_updates.replace
    }
    remove_ids = {item.evidence_id for item in result.evidence_updates.remove}
    targeted_ids = set(replace_by_id).union(remove_ids)
    missing_targets = targeted_ids.difference(existing_by_id)
    if missing_targets:
        raise ValueError(
            "Checker delta references missing Evidence: "
            + ", ".join(sorted(missing_targets))
        )

    known_observation_ids = set(new_observation_ids)
    for item in existing_items:
        known_observation_ids.update(item.observation_ids)
    proposed_observation_ids = {
        observation_id
        for item in [
            *result.evidence_updates.add,
            *result.evidence_updates.replace,
        ]
        for observation_id in item.observation_ids
    }
    unknown_observation_ids = proposed_observation_ids.difference(
        known_observation_ids
    )
    if unknown_observation_ids:
        raise ValueError(
            "Checker delta references unavailable Observations: "
            + ", ".join(sorted(unknown_observation_ids))
        )

    next_items: list[EvidenceItem] = []
    for item in existing_items:
        if item.evidence_id in remove_ids:
            continue
        replacement = replace_by_id.get(item.evidence_id)
        if replacement is None:
            next_items.append(item)
        else:
            next_items.append(
                EvidenceItem(
                    evidence_id=item.evidence_id,
                    statement=replacement.statement,
                    observation_ids=replacement.observation_ids,
                    supports_question_ids=replacement.supports_question_ids,
                )
            )

    existing_ids = {item.evidence_id for item in next_items}
    for index, addition in enumerate(result.evidence_updates.add):
        new_id = evidence_id(result.action_id, index)
        if new_id in existing_ids:
            raise ValueError(f"Checker addition would duplicate Evidence ID {new_id}")
        existing_ids.add(new_id)
        next_items.append(
            EvidenceItem(
                evidence_id=new_id,
                statement=addition.statement,
                observation_ids=addition.observation_ids,
                supports_question_ids=addition.supports_question_ids,
            )
        )

    current_target = checker_input.evidence_memory.current_target
    if current_target is None:
        raise ValueError("Checker input requires an active current_target")
    target_question_id = current_target.question_id
    for update in [
        *result.evidence_updates.add,
        *result.evidence_updates.replace,
    ]:
        if update.supports_question_ids != [target_question_id]:
            raise ValueError(
                "Checker Evidence updates may support only the current target "
                f"{target_question_id} in v0"
            )
    root_question_id = checker_input.evidence_memory.root_question_id
    questions = list(checker_input.evidence_memory.questions)
    questions_by_id = {item.question_id: item for item in questions}
    if target_question_id == root_question_id:
        expected_root_status = (
            EvidenceStatus.READY
            if result.current_target_status == QuestionStatus.SATISFIED
            else EvidenceStatus.INCOMPLETE
        )
        if result.root_status != expected_root_status:
            raise ValueError(
                "When current target is Root, current_target_status must "
                "agree with root_status"
            )
    else:
        current_question = questions_by_id[target_question_id]
        questions_by_id[target_question_id] = current_question.model_copy(
            update={"status": result.current_target_status}
        )

    next_questions = [questions_by_id[item.question_id] for item in questions]
    next_target: CurrentTarget | None
    if result.root_status == EvidenceStatus.READY:
        next_target = None
    elif result.current_target_status == QuestionStatus.INCOMPLETE:
        if result.remaining_gap_description is None:
            raise ValueError("An incomplete current target requires a gap description")
        next_target = CurrentTarget(
            question_id=target_question_id,
            gap_description=result.remaining_gap_description,
        )
    else:
        next_question = select_next_runnable_question(next_questions)
        if next_question is not None:
            if result.remaining_gap_description is not None:
                raise ValueError(
                    "Checker must not choose the next planned question or its gap"
                )
            next_target = CurrentTarget(
                question_id=next_question.question_id,
                gap_description=next_question.text,
            )
        else:
            next_target = CurrentTarget(
                question_id=root_question_id,
                gap_description=(
                    "Determine what evidence is still missing to answer the Root "
                    f"Question: {checker_input.root_question.text}"
                ),
            )

    next_memory = EvidenceMemory(
        reading_session_id=checker_input.evidence_memory.reading_session_id,
        root_question_id=checker_input.evidence_memory.root_question_id,
        root_status=result.root_status,
        questions=next_questions,
        evidence=next_items,
        current_target=next_target,
    )

    used_observation_ids = {
        observation_id
        for item in next_memory.evidence
        for observation_id in item.observation_ids
    }
    for assessment in result.observation_assessments:
        expected = assessment.observation_id in used_observation_ids
        if assessment.used_for_evidence != expected:
            raise ValueError(
                "used_for_evidence must agree with the resulting EvidenceMemory "
                f"for {assessment.observation_id}"
            )
    return next_memory


class ActionExecutionStatus(str, Enum):
    """Whether the Environment executed an action normally.

    This is not an assessment of an Observation's factual correctness or
    usefulness as Evidence.
    """

    SUCCEEDED = "succeeded"
    DEGRADED = "degraded"
    FAILED = "failed"


class ExplorationSourceHandle(SoftDocModel):
    """Typed source that may become the Controller's current focus."""

    source_id: str = Field(min_length=1)
    source_type: ReadingSourceType
    document_id: str = Field(min_length=1)
    page_id: str | None = Field(default=None, min_length=1)
    element_id: str | None = Field(default=None, min_length=1)
    table_view_id: str | None = Field(default=None, min_length=1)
    cell_id: str | None = Field(default=None, min_length=1)
    visual_asset_id: str | None = Field(default=None, min_length=1)
    bbox: NormalizedRegion | None = None

    @field_validator("bbox")
    @classmethod
    def validate_bbox(
        cls, value: NormalizedRegion | None
    ) -> NormalizedRegion | None:
        return None if value is None else _validate_normalized_region(value)

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        element_sources = {
            ReadingSourceType.ELEMENT,
            ReadingSourceType.TABLE_VIEW,
            ReadingSourceType.TABLE_CELL,
        }
        table_sources = {
            ReadingSourceType.TABLE_VIEW,
            ReadingSourceType.TABLE_CELL,
        }
        if self.source_type == ReadingSourceType.PAGE and self.page_id is None:
            raise ValueError("A page focus requires page_id")
        if self.source_type in element_sources and self.element_id is None:
            raise ValueError(f"A {self.source_type.value} focus requires element_id")
        if self.source_type in table_sources and self.table_view_id is None:
            raise ValueError(f"A {self.source_type.value} focus requires table_view_id")
        if self.source_type == ReadingSourceType.TABLE_CELL and self.cell_id is None:
            raise ValueError("A table_cell focus requires cell_id")
        if self.source_type == ReadingSourceType.REGION:
            if self.visual_asset_id is None or self.bbox is None:
                raise ValueError("A region focus requires visual_asset_id and bbox")
        return self


class ActionTraceEntry(SoftDocModel):
    """Canonical record of one Controller-selected environment action."""

    step_index: int = Field(ge=0)
    action_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    action_name: str = Field(min_length=1, pattern=r"^[A-Z][A-Z0-9_]*$")
    target_ids: list[str] = Field(default_factory=list)
    primary_target: ExplorationSourceHandle | None = None
    query: str | None = Field(default=None, min_length=1)
    execution_status: ActionExecutionStatus
    observation_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("target_ids", "observation_ids")
    @classmethod
    def validate_ids(cls, value: list[str]) -> list[str]:
        return _unique_nonblank(value, label="ActionTrace IDs")

    @model_validator(mode="after")
    def validate_primary_target(self) -> Self:
        if (
            self.primary_target is not None
            and self.primary_target.source_id not in self.target_ids
        ):
            raise ValueError("Action primary_target must be included in target_ids")
        return self


class ActionTrace(SoftDocModel):
    """Append-only-by-contract action source of truth for one reading run."""

    reading_session_id: str = Field(min_length=1)
    root_question_id: str = Field(min_length=1)
    entries: list[ActionTraceEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_entries(self) -> Self:
        action_ids = [entry.action_id for entry in self.entries]
        _unique_nonblank(action_ids, label="ActionTrace action IDs")
        indexes = [entry.step_index for entry in self.entries]
        if indexes != list(range(len(indexes))):
            raise ValueError("ActionTrace step indexes must be contiguous from zero")
        return self


class ConfirmedRelationHandle(SoftDocModel):
    relation_id: str = Field(min_length=1)
    relation_type: RelationType
    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)


class CandidateNavigationHint(SoftDocModel):
    """A target worth inspecting; it does not assert that the Relation is true."""

    relation_id: str = Field(min_length=1)
    relation_type: RelationType
    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class RecentActionSummary(SoftDocModel):
    action_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    action_name: str = Field(min_length=1)
    target_ids: list[str] = Field(default_factory=list)
    execution_status: ActionExecutionStatus
    observation_ids: list[str] = Field(default_factory=list)

    @field_validator("target_ids", "observation_ids")
    @classmethod
    def validate_ids(cls, value: list[str]) -> list[str]:
        return _unique_nonblank(value, label="Recent action IDs")


class ExplorationState(SoftDocModel):
    """Derived Controller view; canonical history lives in logs and sessions."""

    reading_session_id: str = Field(min_length=1)
    root_question_id: str = Field(min_length=1)
    current_focus: ExplorationSourceHandle | None = None
    attempted_source_ids: list[str] = Field(default_factory=list)
    attempted_search_queries: list[str] = Field(default_factory=list)
    active_search_session_ids: list[str] = Field(default_factory=list)
    confirmed_relation_handles: list[ConfirmedRelationHandle] = Field(
        default_factory=list
    )
    candidate_navigation_hints: list[CandidateNavigationHint] = Field(
        default_factory=list
    )
    recent_actions: list[RecentActionSummary] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_entries(self) -> Self:
        collections: list[tuple[str, list[str]]] = [
            ("attempted source IDs", self.attempted_source_ids),
            ("attempted search queries", self.attempted_search_queries),
            ("active search session IDs", self.active_search_session_ids),
            (
                "confirmed relation IDs",
                [item.relation_id for item in self.confirmed_relation_handles],
            ),
            (
                "candidate relation IDs",
                [item.relation_id for item in self.candidate_navigation_hints],
            ),
            ("recent action IDs", [item.action_id for item in self.recent_actions]),
        ]
        for label, values in collections:
            _unique_nonblank(values, label=f"ExplorationState {label}")
        confirmed_ids = {
            item.relation_id for item in self.confirmed_relation_handles
        }
        candidate_ids = {item.relation_id for item in self.candidate_navigation_hints}
        if confirmed_ids.intersection(candidate_ids):
            raise ValueError(
                "A Relation cannot be both confirmed and a candidate navigation hint"
            )
        return self


class ExplorationStateBuilder:
    """Build a compact view without creating another canonical history."""

    def build(
        self,
        *,
        observation_store: ObservationStore,
        action_trace: ActionTrace,
        active_search_sessions: Iterable[SearchSession] = (),
        available_relations: Iterable[Relation] = (),
        recent_action_limit: int = 5,
        attempted_search_limit: int = 10,
    ) -> ExplorationState:
        if recent_action_limit < 0:
            raise ValueError("recent_action_limit must be non-negative")
        if attempted_search_limit < 0:
            raise ValueError("attempted_search_limit must be non-negative")
        if (
            observation_store.reading_session_id != action_trace.reading_session_id
            or observation_store.root_question_id != action_trace.root_question_id
        ):
            raise ValueError(
                "ObservationStore and ActionTrace must belong to the same reading run"
            )

        attempted_source_ids = _ordered_unique(
            source.source_id
            for record in observation_store.read_records
            for source in record.inputs
        )
        all_attempted_search_queries = _ordered_unique(
            entry.query.strip()
            for entry in action_trace.entries
            if "SEARCH" in entry.action_name
            and entry.query is not None
            and entry.query.strip()
        )
        attempted_search_queries = (
            all_attempted_search_queries[-attempted_search_limit:]
            if attempted_search_limit
            else []
        )
        active_search_session_ids = _ordered_unique(
            session.search_session_id for session in active_search_sessions
        )

        current_focus = next(
            (
                entry.primary_target
                for entry in reversed(action_trace.entries)
                if entry.primary_target is not None
                and entry.execution_status in {
                    ActionExecutionStatus.SUCCEEDED,
                    ActionExecutionStatus.DEGRADED,
                }
            ),
            None,
        )

        confirmed: list[ConfirmedRelationHandle] = []
        candidates: list[CandidateNavigationHint] = []
        seen_relations: set[str] = set()
        for relation in available_relations:
            if current_focus is None or current_focus.source_id not in {
                relation.source_id,
                relation.target_id,
            }:
                continue
            if relation.relation_id in seen_relations:
                continue
            seen_relations.add(relation.relation_id)
            if relation.status == RelationStatus.CONFIRMED:
                confirmed.append(
                    ConfirmedRelationHandle(
                        relation_id=relation.relation_id,
                        relation_type=relation.relation_type,
                        source_id=relation.source_id,
                        target_id=relation.target_id,
                    )
                )
            elif relation.status == RelationStatus.CANDIDATE:
                candidates.append(
                    CandidateNavigationHint(
                        relation_id=relation.relation_id,
                        relation_type=relation.relation_type,
                        source_id=relation.source_id,
                        target_id=relation.target_id,
                        confidence=relation.confidence,
                    )
                )

        recent_entries = (
            action_trace.entries[-recent_action_limit:]
            if recent_action_limit
            else []
        )
        recent_actions = [
            RecentActionSummary(
                action_id=entry.action_id,
                question_id=entry.question_id,
                action_name=entry.action_name,
                target_ids=list(entry.target_ids),
                execution_status=entry.execution_status,
                observation_ids=list(entry.observation_ids),
            )
            for entry in recent_entries
        ]
        return ExplorationState(
            reading_session_id=observation_store.reading_session_id,
            root_question_id=observation_store.root_question_id,
            current_focus=current_focus,
            attempted_source_ids=attempted_source_ids,
            attempted_search_queries=attempted_search_queries,
            active_search_session_ids=active_search_session_ids,
            confirmed_relation_handles=confirmed,
            candidate_navigation_hints=candidates,
            recent_actions=recent_actions,
        )


def _ordered_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
