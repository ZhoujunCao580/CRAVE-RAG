"""Frozen Controller-input contract.

The Controller receives a compact projection of canonical reading state.  It
does not receive full retrieval rankings, Reader records, Checker deltas, or
the legacy ExplorationState object.
"""

from __future__ import annotations

from enum import Enum
from collections.abc import Iterable
from typing import Annotated, Any, Literal, Self, TypeAlias

from pydantic import Field, TypeAdapter, field_validator, model_validator

from softdoc.models import (
    ContentAvailability,
    ElementType,
    RelationType,
    SoftDocModel,
)
from softdoc.reading_state import (
    ActionTrace,
    ActionExecutionStatus,
    EvidenceStatus,
    EvidenceMemory,
    ObservationStore,
    QuestionStatus,
    ReadingSourceType,
    RootQuestion,
)
from softdoc.retrieval.models import AnchorTargetType, SearchBatch, SearchSession


CONTROLLER_INPUT_VERSION = "controller-input-v0.1"
CONTROLLER_ACTION_VERSION = "controller-action-v0.1"


def _unique_nonblank(values: list[str], *, label: str) -> list[str]:
    if any(not value.strip() for value in values):
        raise ValueError(f"{label} must not contain blank values")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")
    return values


class ControllerSubQuestion(SoftDocModel):
    question_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    status: QuestionStatus

    @field_validator("depends_on")
    @classmethod
    def validate_dependencies(cls, value: list[str]) -> list[str]:
        return _unique_nonblank(value, label="SubQuestion dependency IDs")


class ControllerEvidence(SoftDocModel):
    evidence_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    supports_question_ids: list[str] = Field(min_length=1)

    @field_validator("supports_question_ids")
    @classmethod
    def validate_supports(cls, value: list[str]) -> list[str]:
        return _unique_nonblank(value, label="Evidence supported question IDs")


class ControllerGap(SoftDocModel):
    question_id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class ControllerReadingLocation(SoftDocModel):
    source_id: str = Field(min_length=1)
    source_type: ReadingSourceType
    page_id: str = Field(min_length=1)


class ControllerLimitation(SoftDocModel):
    description: str = Field(min_length=1)
    source_ids: list[str] = Field(default_factory=list)

    @field_validator("source_ids")
    @classmethod
    def validate_sources(cls, value: list[str]) -> list[str]:
        return _unique_nonblank(value, label="Limitation source IDs")


class ControllerObservationAssessment(SoftDocModel):
    observation_id: str = Field(min_length=1)
    used_for_evidence: bool
    assessment: str = Field(min_length=1)


class ControllerActionFeedback(SoftDocModel):
    limitations: list[ControllerLimitation] = Field(default_factory=list)
    observation_assessments: list[ControllerObservationAssessment] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_nonempty(self) -> Self:
        if not self.limitations and not self.observation_assessments:
            raise ValueError("Action feedback must contain a limitation or assessment")
        return self


class ControllerRecentAction(SoftDocModel):
    action_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    action_name: str = Field(min_length=1, pattern=r"^[A-Z][A-Z0-9_]*$")
    target_ids: list[str] = Field(default_factory=list)
    execution_status: ActionExecutionStatus
    observation_ids: list[str] = Field(default_factory=list)
    feedback: ControllerActionFeedback | None = None

    @field_validator("target_ids", "observation_ids")
    @classmethod
    def validate_ids(cls, value: list[str], info: object) -> list[str]:
        label = getattr(info, "field_name", "Action IDs")
        return _unique_nonblank(value, label=label)


class ControllerConfirmedRelation(SoftDocModel):
    relation_id: str = Field(min_length=1)
    relation_type: RelationType
    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)


class ControllerCandidateRelation(ControllerConfirmedRelation):
    confidence: float = Field(ge=0.0, le=1.0)


class ControllerSearchTab(SoftDocModel):
    search_session_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    has_more: bool


class ControllerExactAnchorMatch(SoftDocModel):
    anchor_text: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    target_type: AnchorTargetType
    page_id: str = Field(min_length=1)
    resolution_method: str = Field(min_length=1)


class ControllerCandidatePreview(SoftDocModel):
    """Small deterministic reading-entry card, never Evidence."""

    element_id: str = Field(min_length=1)
    element_type: ElementType
    page_id: str = Field(min_length=1)
    section_path: list[str] = Field(default_factory=list)
    matched_snippet: str
    content_availability: ContentAvailability

    @field_validator("section_path")
    @classmethod
    def validate_section_path(cls, value: list[str]) -> list[str]:
        if any(not component.strip() for component in value):
            raise ValueError("Section path must not contain blank components")
        return value


class ControllerVisibleSearchView(SoftDocModel):
    search_session_id: str = Field(min_length=1)
    exact_anchor_matches: list[ControllerExactAnchorMatch] = Field(
        default_factory=list
    )
    candidate_previews: list[ControllerCandidatePreview] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_candidates(self) -> Self:
        exact_ids = [item.target_id for item in self.exact_anchor_matches]
        candidate_ids = [item.element_id for item in self.candidate_previews]
        _unique_nonblank(exact_ids, label="Exact Anchor target IDs")
        _unique_nonblank(candidate_ids, label="CandidatePreview Element IDs")
        overlap = set(exact_ids).intersection(candidate_ids)
        if overlap:
            raise ValueError(
                "Exact Anchor targets must not repeat as normal candidates: "
                + ", ".join(sorted(overlap))
            )
        return self


class ControllerInput(SoftDocModel):
    """Frozen input visible to one Controller decision."""

    reading_session_id: str = Field(min_length=1)
    root_question: RootQuestion
    root_status: EvidenceStatus
    subquestions: list[ControllerSubQuestion] = Field(default_factory=list)
    evidence: list[ControllerEvidence] = Field(default_factory=list)
    current_gap: ControllerGap | None = None
    reading_locations: list[ControllerReadingLocation] = Field(default_factory=list)
    recent_actions: list[ControllerRecentAction] = Field(default_factory=list)
    confirmed_relations: list[ControllerConfirmedRelation] = Field(
        default_factory=list
    )
    candidate_relations: list[ControllerCandidateRelation] = Field(
        default_factory=list
    )
    search_tabs: list[ControllerSearchTab] = Field(default_factory=list)
    visible_search_view: ControllerVisibleSearchView | None = None
    remaining_action_budget: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        question_ids = [item.question_id for item in self.subquestions]
        _unique_nonblank(question_ids, label="Controller SubQuestion IDs")
        if self.root_question.question_id in question_ids:
            raise ValueError("Root Question must not be duplicated in subquestions")
        questions_by_id = {item.question_id: item for item in self.subquestions}
        for question in self.subquestions:
            unknown = set(question.depends_on).difference(questions_by_id)
            if unknown:
                raise ValueError(
                    f"SubQuestion {question.question_id} has unknown dependencies: "
                    + ", ".join(sorted(unknown))
                )
            if question.question_id in question.depends_on:
                raise ValueError("A SubQuestion cannot depend on itself")
        self._validate_question_dag(questions_by_id)

        if self.root_status == EvidenceStatus.READY:
            if self.current_gap is not None:
                raise ValueError("A ready Root must not have a current_gap")
        elif self.current_gap is None:
            raise ValueError("An incomplete Root requires a current_gap")

        known_question_ids = {self.root_question.question_id, *question_ids}
        if self.current_gap is not None:
            if self.current_gap.question_id not in known_question_ids:
                raise ValueError("current_gap references an unknown question")
            if self.current_gap.question_id == self.root_question.question_id:
                unfinished = [
                    item.question_id
                    for item in self.subquestions
                    if item.status != QuestionStatus.SATISFIED
                ]
                if unfinished:
                    raise ValueError(
                        "Root may become current_gap only after all SubQuestions are "
                        "satisfied"
                    )
            else:
                target = questions_by_id[self.current_gap.question_id]
                if target.status != QuestionStatus.INCOMPLETE:
                    raise ValueError("current_gap SubQuestion must be incomplete")
                blocked = [
                    dependency_id
                    for dependency_id in target.depends_on
                    if questions_by_id[dependency_id].status
                    != QuestionStatus.SATISFIED
                ]
                if blocked:
                    raise ValueError(
                        "current_gap SubQuestion has unsatisfied dependencies: "
                        + ", ".join(blocked)
                    )

        evidence_ids = [item.evidence_id for item in self.evidence]
        _unique_nonblank(evidence_ids, label="Controller Evidence IDs")
        for item in self.evidence:
            unknown = set(item.supports_question_ids).difference(known_question_ids)
            if unknown:
                raise ValueError(
                    f"Evidence {item.evidence_id} supports unknown questions: "
                    + ", ".join(sorted(unknown))
                )

        location_ids = [item.source_id for item in self.reading_locations]
        _unique_nonblank(location_ids, label="Controller reading-location source IDs")
        action_ids = [item.action_id for item in self.recent_actions]
        _unique_nonblank(action_ids, label="Controller recent action IDs")
        for action in self.recent_actions:
            if action.question_id not in known_question_ids:
                raise ValueError(
                    f"Action {action.action_id} references an unknown question"
                )

        confirmed_ids = [item.relation_id for item in self.confirmed_relations]
        candidate_ids = [item.relation_id for item in self.candidate_relations]
        _unique_nonblank(confirmed_ids, label="Confirmed Relation IDs")
        _unique_nonblank(candidate_ids, label="Candidate Relation IDs")
        overlap = set(confirmed_ids).intersection(candidate_ids)
        if overlap:
            raise ValueError(
                "A Relation cannot be both confirmed and candidate: "
                + ", ".join(sorted(overlap))
            )

        session_ids = [item.search_session_id for item in self.search_tabs]
        _unique_nonblank(session_ids, label="Controller SearchSession IDs")
        if (
            self.visible_search_view is not None
            and self.visible_search_view.search_session_id not in set(session_ids)
        ):
            raise ValueError(
                "visible_search_view must reference a SearchSession in search_tabs"
            )
        return self

    @staticmethod
    def _validate_question_dag(
        questions_by_id: dict[str, ControllerSubQuestion],
    ) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(question_id: str) -> None:
            if question_id in visited:
                return
            if question_id in visiting:
                raise ValueError("Controller SubQuestion dependencies must form a DAG")
            visiting.add(question_id)
            for dependency_id in questions_by_id[question_id].depends_on:
                visit(dependency_id)
            visiting.remove(question_id)
            visited.add(question_id)

        for question_id in questions_by_id:
            visit(question_id)


class ControllerActionName(str, Enum):
    SEARCH = "SEARCH"
    READ_SOURCE = "READ_SOURCE"
    FOLLOW_RELATION = "FOLLOW_RELATION"
    EXPLORE_CANDIDATE_RELATION = "EXPLORE_CANDIDATE_RELATION"
    READ_ADJACENT_PAGE = "READ_ADJACENT_PAGE"


class ControllerSearchOperation(str, Enum):
    NEW = "new"
    NEXT = "next"
    SWITCH = "switch"


class AdjacentPageDirection(str, Enum):
    NEXT = "next"
    PREVIOUS = "previous"


class ControllerSearchAction(SoftDocModel):
    action: Literal[ControllerActionName.SEARCH]
    operation: ControllerSearchOperation
    query: str | None = Field(default=None, min_length=1)
    search_session_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_operation_arguments(self) -> Self:
        if self.operation == ControllerSearchOperation.NEW:
            if self.query is None or not self.query.strip():
                raise ValueError("SEARCH new requires a nonblank query")
            if self.search_session_id is not None:
                raise ValueError("SEARCH new must not include search_session_id")
        else:
            if self.search_session_id is None:
                raise ValueError(
                    f"SEARCH {self.operation.value} requires search_session_id"
                )
            if self.query is not None:
                raise ValueError(
                    f"SEARCH {self.operation.value} must not include query"
                )
        return self


class ControllerReadSourceAction(SoftDocModel):
    action: Literal[ControllerActionName.READ_SOURCE]
    source_ids: list[str] = Field(min_length=1)
    local_problem: str = Field(min_length=1)

    @field_validator("source_ids")
    @classmethod
    def validate_sources(cls, value: list[str]) -> list[str]:
        return _unique_nonblank(value, label="READ_SOURCE source IDs")


class ControllerFollowRelationAction(SoftDocModel):
    action: Literal[ControllerActionName.FOLLOW_RELATION]
    relation_id: str = Field(min_length=1)
    local_problem: str = Field(min_length=1)


class ControllerExploreCandidateRelationAction(SoftDocModel):
    action: Literal[ControllerActionName.EXPLORE_CANDIDATE_RELATION]
    relation_id: str = Field(min_length=1)
    local_problem: str = Field(min_length=1)


class ControllerReadAdjacentPageAction(SoftDocModel):
    action: Literal[ControllerActionName.READ_ADJACENT_PAGE]
    from_page_id: str = Field(min_length=1)
    direction: AdjacentPageDirection
    local_problem: str = Field(min_length=1)


ControllerAction: TypeAlias = Annotated[
    ControllerSearchAction
    | ControllerReadSourceAction
    | ControllerFollowRelationAction
    | ControllerExploreCandidateRelationAction
    | ControllerReadAdjacentPageAction,
    Field(discriminator="action"),
]


_CONTROLLER_ACTION_ADAPTER = TypeAdapter(ControllerAction)


class ControllerInputBuilder:
    """Project canonical state into the small view visible to the Controller.

    Full Reader records, Checker deltas, retrieval rankings, and parser payloads
    remain in their owning stores.  This builder exposes only actionable handles
    and compact feedback from recent actions.
    """

    def build(
        self,
        *,
        root_question: RootQuestion,
        evidence_memory: EvidenceMemory,
        observation_store: ObservationStore,
        action_trace: ActionTrace,
        relations: Iterable[Any] = (),
        search_sessions: Iterable[SearchSession] = (),
        visible_search_batch: SearchBatch | None = None,
        remaining_action_budget: int,
        recent_action_limit: int = 5,
    ) -> ControllerInput:
        if recent_action_limit < 0:
            raise ValueError("recent_action_limit must be non-negative")
        if evidence_memory.root_question_id != root_question.question_id:
            raise ValueError("Controller Root Question must match EvidenceMemory")
        if not (
            observation_store.reading_session_id
            == evidence_memory.reading_session_id
            == action_trace.reading_session_id
        ):
            raise ValueError("Controller stores must belong to one reading session")

        sessions = list(search_sessions)
        sessions_by_id = {item.search_session_id: item for item in sessions}
        if len(sessions_by_id) != len(sessions):
            raise ValueError("Controller SearchSession IDs must be unique")
        if (
            visible_search_batch is not None
            and visible_search_batch.search_session_id not in sessions_by_id
        ):
            raise ValueError("Visible SearchBatch has no canonical SearchSession")

        records_by_action = {
            record.action_id: record for record in observation_store.read_records
        }
        locations: list[ControllerReadingLocation] = []
        seen_locations: set[str] = set()
        for entry in reversed(action_trace.entries):
            if entry.execution_status not in {
                ActionExecutionStatus.SUCCEEDED,
                ActionExecutionStatus.DEGRADED,
            }:
                continue
            handles = []
            if entry.primary_target is not None:
                handles.append(entry.primary_target)
            record = records_by_action.get(entry.action_id)
            if record is not None:
                handles.extend(record.inputs)
            for handle in handles:
                source_id = handle.source_id
                page_id = handle.page_id
                source_type = handle.source_type
                if source_id in seen_locations or page_id is None:
                    continue
                seen_locations.add(source_id)
                locations.append(
                    ControllerReadingLocation(
                        source_id=source_id,
                        source_type=source_type,
                        page_id=page_id,
                    )
                )

        focus = next(
            (
                entry.primary_target
                for entry in reversed(action_trace.entries)
                if entry.primary_target is not None
                and entry.execution_status
                in {ActionExecutionStatus.SUCCEEDED, ActionExecutionStatus.DEGRADED}
            ),
            None,
        )
        confirmed: list[ControllerConfirmedRelation] = []
        candidates: list[ControllerCandidateRelation] = []
        if focus is not None:
            for relation in relations:
                if focus.source_id not in {relation.source_id, relation.target_id}:
                    continue
                if relation.status.value == "confirmed":
                    confirmed.append(
                        ControllerConfirmedRelation(
                            relation_id=relation.relation_id,
                            relation_type=relation.relation_type,
                            source_id=relation.source_id,
                            target_id=relation.target_id,
                        )
                    )
                elif relation.status.value == "candidate":
                    candidates.append(
                        ControllerCandidateRelation(
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
        recent_actions: list[ControllerRecentAction] = []
        for entry in recent_entries:
            raw_feedback = entry.metadata.get("controller_feedback")
            feedback = (
                ControllerActionFeedback.model_validate(raw_feedback)
                if raw_feedback is not None
                else None
            )
            recent_actions.append(
                ControllerRecentAction(
                    action_id=entry.action_id,
                    question_id=entry.question_id,
                    action_name=entry.action_name,
                    target_ids=entry.target_ids,
                    execution_status=entry.execution_status,
                    observation_ids=entry.observation_ids,
                    feedback=feedback,
                )
            )

        visible_view = None
        if visible_search_batch is not None:
            visible_view = ControllerVisibleSearchView(
                search_session_id=visible_search_batch.search_session_id,
                exact_anchor_matches=[
                    ControllerExactAnchorMatch(
                        anchor_text=item.anchor_text,
                        target_id=item.target_id,
                        target_type=item.target_type,
                        page_id=item.page_id,
                        resolution_method=item.resolution_method,
                    )
                    for item in visible_search_batch.exact_anchor_matches
                ],
                candidate_previews=[
                    ControllerCandidatePreview(
                        element_id=item.element_id,
                        element_type=item.element_type,
                        page_id=item.page_id,
                        section_path=item.section_path,
                        matched_snippet=item.matched_snippet,
                        content_availability=item.content_availability,
                    )
                    for item in visible_search_batch.candidate_previews
                ],
            )

        return ControllerInput(
            reading_session_id=evidence_memory.reading_session_id,
            root_question=root_question,
            root_status=evidence_memory.root_status,
            subquestions=[
                ControllerSubQuestion(
                    question_id=item.question_id,
                    text=item.text,
                    depends_on=item.depends_on,
                    status=item.status,
                )
                for item in evidence_memory.questions
            ],
            evidence=[
                ControllerEvidence(
                    evidence_id=item.evidence_id,
                    statement=item.statement,
                    supports_question_ids=item.supports_question_ids,
                )
                for item in evidence_memory.evidence
            ],
            current_gap=(
                ControllerGap(
                    question_id=evidence_memory.current_target.question_id,
                    description=evidence_memory.current_target.gap_description,
                )
                if evidence_memory.current_target is not None
                else None
            ),
            reading_locations=locations,
            recent_actions=recent_actions,
            confirmed_relations=confirmed,
            candidate_relations=candidates,
            search_tabs=[
                ControllerSearchTab(
                    search_session_id=item.search_session_id,
                    query=_session_query(item, action_trace),
                    has_more=not item.exhausted,
                )
                for item in sessions
            ],
            visible_search_view=visible_view,
            remaining_action_budget=remaining_action_budget,
        )


def _session_query(session: SearchSession, action_trace: ActionTrace) -> str:
    for entry in reversed(action_trace.entries):
        if (
            entry.metadata.get("search_session_id") == session.search_session_id
            and entry.query
        ):
            return entry.query
    raise ValueError(
        f"SearchSession {session.search_session_id} has no query in ActionTrace"
    )


def validate_controller_action(
    value: ControllerAction | dict[str, Any],
    controller_input: ControllerInput,
) -> ControllerAction:
    """Validate one action against only the handles visible to the Controller."""

    action = _CONTROLLER_ACTION_ADAPTER.validate_python(value)

    if isinstance(action, ControllerSearchAction):
        if action.operation != ControllerSearchOperation.NEW:
            tabs = {
                item.search_session_id: item for item in controller_input.search_tabs
            }
            tab = tabs.get(action.search_session_id or "")
            if tab is None:
                raise ValueError("SEARCH references a SearchSession not visible here")
            if action.operation == ControllerSearchOperation.NEXT and not tab.has_more:
                raise ValueError("SEARCH next requires a SearchSession with has_more")
        return action

    if isinstance(action, ControllerReadSourceAction):
        visible_sources = {item.source_id for item in controller_input.reading_locations}
        if controller_input.visible_search_view is not None:
            visible_sources.update(
                item.target_id
                for item in controller_input.visible_search_view.exact_anchor_matches
            )
            visible_sources.update(
                item.element_id
                for item in controller_input.visible_search_view.candidate_previews
            )
        unknown = set(action.source_ids).difference(visible_sources)
        if unknown:
            raise ValueError(
                "READ_SOURCE references sources not visible here: "
                + ", ".join(sorted(unknown))
            )
        return action

    if isinstance(action, ControllerFollowRelationAction):
        visible_ids = {
            item.relation_id for item in controller_input.confirmed_relations
        }
        if action.relation_id not in visible_ids:
            raise ValueError("FOLLOW_RELATION requires a visible confirmed Relation")
        return action

    if isinstance(action, ControllerExploreCandidateRelationAction):
        visible_ids = {
            item.relation_id for item in controller_input.candidate_relations
        }
        if action.relation_id not in visible_ids:
            raise ValueError(
                "EXPLORE_CANDIDATE_RELATION requires a visible candidate Relation"
            )
        return action

    visible_page_ids = {item.page_id for item in controller_input.reading_locations}
    if action.from_page_id not in visible_page_ids:
        raise ValueError(
            "READ_ADJACENT_PAGE requires a page visible in reading_locations"
        )
    return action
