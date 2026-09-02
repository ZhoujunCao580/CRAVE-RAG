"""Executable v0 loop joining retrieval, reading, evidence checking, and answering.

The environment owns orchestration and deterministic state transitions.  Model
capability remains behind injectable interfaces, so unit tests and local audits
can use scripted teacher outputs without pretending they came from a production
LLM/VLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, Self

from pydantic import Field, model_validator

from softdoc.answering import (
    AnswerInput,
    AnswerInputBuilder,
    AnswerResult,
    validate_answer_result,
)
from softdoc.controller import (
    AdjacentPageDirection,
    ControllerAction,
    ControllerActionFeedback,
    ControllerExploreCandidateRelationAction,
    ControllerFollowRelationAction,
    ControllerInput,
    ControllerInputBuilder,
    ControllerLimitation,
    ControllerObservationAssessment,
    ControllerReadAdjacentPageAction,
    ControllerReadSourceAction,
    ControllerSearchAction,
    ControllerSearchOperation,
    ControllerStopAction,
    validate_controller_action,
)
from softdoc.ids import (
    action_id as make_action_id,
    observation_id as make_observation_id,
    read_input_id,
    reading_session_id as make_reading_session_id,
    stable_digest,
)
from softdoc.models import (
    Document,
    Element,
    ElementType,
    Page,
    Relation,
    RelationStatus,
    SoftDocModel,
)
from softdoc.planning.models import InitialPlan
from softdoc.reading_state import (
    ActionExecutionStatus,
    ActionTrace,
    ActionTraceEntry,
    EvidenceCheckInput,
    EvidenceCheckResult,
    EvidenceMemory,
    EvidenceStatus,
    ExplorationSourceHandle,
    ObservationLimitation,
    ObservationSourceRef,
    ObservationStore,
    QuestionState,
    ReadInput,
    ReadRecord,
    ReaderKind,
    ReadingSourceType,
    ReadRepresentation,
    RootQuestion,
    StoredObservation,
    apply_evidence_check_result,
    initialize_evidence_memory,
)
from softdoc.reading_state_validation import ReadingStateReferenceValidator
from softdoc.retrieval import (
    BM25Index,
    DenseSearchResult,
    ExactAnchorLookup,
    ExactLookupResult,
    SearchBatch,
    SearchSession,
    SearchSessionBuilder,
    SearchSessionConfig,
    SearchSessionNavigator,
    SearchUnitBuildResult,
    SearchUnitBuilder,
    SubQuestionInput,
    VisualSearchResult,
    html_to_text,
)
from softdoc.store import DocumentStore
from softdoc.table_view import TableMaterializer, TableView


READING_ENVIRONMENT_VERSION = "reading-environment-v0.2"


class ReaderObservationDraft(SoftDocModel):
    """Reader-owned fact before the Environment assigns a global ID."""

    text: str = Field(min_length=1)
    sources: list[ObservationSourceRef] = Field(min_length=1)


class ReaderOutput(SoftDocModel):
    """Common output accepted from text, table, page, or visual Readers."""

    reader_kind: ReaderKind
    observations: list[ReaderObservationDraft] = Field(default_factory=list)
    limitations: list[ObservationLimitation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_nonempty(self) -> Self:
        if not self.observations and not self.limitations:
            raise ValueError("Reader output requires an Observation or limitation")
        return self


@dataclass(frozen=True)
class ReaderContext:
    action_id: str
    question_id: str
    local_problem: str
    document: Document
    inputs: tuple[ReadInput, ...]
    elements_by_id: dict[str, Element]
    pages_by_id: dict[str, Page]
    table_views_by_id: dict[str, TableView]


class ControllerBackend(Protocol):
    def decide(self, controller_input: ControllerInput) -> ControllerAction | dict[str, Any]:
        """Choose one frozen Controller action."""


class ReaderBackend(Protocol):
    def read(self, context: ReaderContext) -> ReaderOutput:
        """Return grounded observations and limitations for one read action."""


class EvidenceCheckerBackend(Protocol):
    def check(self, checker_input: EvidenceCheckInput) -> EvidenceCheckResult:
        """Assess new observations and return an EvidenceMemory delta."""


class AnswererBackend(Protocol):
    def answer(self, answer_input: AnswerInput) -> AnswerResult:
        """Synthesize the final answer from ready Evidence only."""


class DenseSearchBackend(Protocol):
    def search(self, subquestion: SubQuestionInput) -> DenseSearchResult:
        """Optional Dense retriever already bound to this Document index."""


class VisualSearchBackend(Protocol):
    def search(self, subquestion: SubQuestionInput) -> VisualSearchResult:
        """Optional visual retriever already bound to this Document index."""


class ReadingRunStatus(str, Enum):
    READY = "ready"
    STOPPED_INCOMPLETE = "stopped_incomplete"
    BUDGET_EXHAUSTED = "budget_exhausted"


class EnvironmentDiagnostic(SoftDocModel):
    code: str = Field(min_length=1)
    description: str = Field(min_length=1)
    action_id: str | None = None
    question_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReadingEnvironmentConfig(SoftDocModel):
    action_budget: int = Field(default=7, ge=1, le=100)
    recent_action_limit: int = Field(default=5, ge=0)
    search: SearchSessionConfig = Field(default_factory=SearchSessionConfig)


class ReadingRunResult(SoftDocModel):
    environment_version: str = READING_ENVIRONMENT_VERSION
    status: ReadingRunStatus
    root_question: RootQuestion
    evidence_memory: EvidenceMemory
    observation_store: ObservationStore
    action_trace: ActionTrace
    search_sessions: list[SearchSession] = Field(default_factory=list)
    controller_input_history: list[ControllerInput] = Field(default_factory=list)
    exact_lookup_results: list[ExactLookupResult] = Field(default_factory=list)
    diagnostics: list[EnvironmentDiagnostic] = Field(default_factory=list)
    answer: AnswerResult | None = None


class DocumentSearchService:
    """One-document Exact + BM25 + optional Dense/Visual retrieval service."""

    def __init__(
        self,
        document: Document,
        *,
        search_units: SearchUnitBuildResult | None = None,
        dense_backend: DenseSearchBackend | None = None,
        visual_backend: VisualSearchBackend | None = None,
        config: SearchSessionConfig | None = None,
    ) -> None:
        self.document = document
        self.search_units = search_units or SearchUnitBuilder().build(document)
        self.exact_lookup = ExactAnchorLookup()
        self.bm25 = BM25Index(self.search_units)
        self.dense_backend = dense_backend
        self.visual_backend = visual_backend
        self.builder = SearchSessionBuilder(config)
        self.navigator = SearchSessionNavigator(self.search_units)

    def lookup_exact(self, question: SubQuestionInput) -> ExactLookupResult:
        return self.exact_lookup.lookup(question, self.document)

    def start(
        self,
        *,
        question_id: str,
        query: str,
    ) -> tuple[SearchSession, SearchBatch]:
        request = SubQuestionInput(subquestion_id=question_id, text=query)
        bm25 = self.bm25.search(request)
        dense = self.dense_backend.search(request) if self.dense_backend else None
        visual = self.visual_backend.search(request) if self.visual_backend else None
        session = self.builder.create(
            subquestion=request,
            search_units=self.search_units,
            bm25=bm25,
            dense=dense,
            visual=visual,
        )
        return self.navigator.next_batch(session)

    def next_batch(self, session: SearchSession) -> tuple[SearchSession, SearchBatch]:
        return self.navigator.next_batch(session)

    def mark_opened(self, session: SearchSession, element_id: str) -> SearchSession:
        return self.navigator.mark_opened(session, element_id)


class DeterministicContentReader:
    """Safe local baseline for text and materialized tables.

    It does not answer the local problem and does not inspect pixels.  Text and
    table content are returned verbatim as observations; visual inputs return a
    limitation so a Controller can choose another representation or a VLM.
    """

    def read(self, context: ReaderContext) -> ReaderOutput:
        observations: list[ReaderObservationDraft] = []
        limitations: list[ObservationLimitation] = []
        kinds: set[ReaderKind] = set()
        for item in context.inputs:
            if item.representation == ReadRepresentation.ELEMENT_TEXT:
                element = context.elements_by_id[item.element_id or ""]
                content = (element.text or html_to_text(element.html or "")).strip()
                if content:
                    observations.append(
                        ReaderObservationDraft(
                            text=content,
                            sources=[ObservationSourceRef(input_id=item.input_id)],
                        )
                    )
                else:
                    limitations.append(
                        ObservationLimitation(
                            description="The Element contains no readable text.",
                            input_ids=[item.input_id],
                        )
                    )
                kinds.add(ReaderKind.TEXT)
                continue

            if item.representation == ReadRepresentation.TABLE_VIEW:
                view = context.table_views_by_id[item.table_view_id or ""]
                cell_text = [
                    f"r{cell.row}c{cell.column}: {cell.text}"
                    for cell in view.cells
                    if cell.text
                ]
                if cell_text:
                    observations.append(
                        ReaderObservationDraft(
                            text="\n".join(cell_text),
                            sources=[ObservationSourceRef(input_id=item.input_id)],
                        )
                    )
                else:
                    limitations.append(
                        ObservationLimitation(
                            description=(
                                "The TableView has no readable structured cell text."
                            ),
                            input_ids=[item.input_id],
                        )
                    )
                kinds.add(ReaderKind.TABLE)
                continue

            limitations.append(
                ObservationLimitation(
                    description=(
                        "This deterministic baseline cannot inspect visual pixels; "
                        "a visual Reader is required."
                    ),
                    input_ids=[item.input_id],
                )
            )
            kinds.add(
                ReaderKind.PAGE
                if item.representation == ReadRepresentation.PAGE_VISUAL
                else ReaderKind.VISUAL
            )

        reader_kind = next(iter(kinds)) if len(kinds) == 1 else ReaderKind.VISUAL
        return ReaderOutput(
            reader_kind=reader_kind,
            observations=observations,
            limitations=limitations,
        )


class ReadingEnvironment:
    """Run the frozen Controller action space against canonical reading state."""

    def __init__(
        self,
        document: Document,
        *,
        asset_root: Path,
        controller: ControllerBackend,
        reader: ReaderBackend,
        checker: EvidenceCheckerBackend,
        answerer: AnswererBackend,
        search_service: DocumentSearchService | None = None,
        config: ReadingEnvironmentConfig | None = None,
    ) -> None:
        self.document = document
        self.store = DocumentStore(document)
        self.asset_root = Path(asset_root)
        self.controller = controller
        self.reader = reader
        self.checker = checker
        self.answerer = answerer
        self.config = config or ReadingEnvironmentConfig()
        self.search = search_service or DocumentSearchService(
            document, config=self.config.search
        )
        self._input_builder = ControllerInputBuilder()
        self._reference_validator = ReadingStateReferenceValidator()
        self._answer_builder = AnswerInputBuilder()
        self._table_materializer = TableMaterializer()
        self._table_views: dict[str, TableView] = {}
        self._sessions: dict[str, SearchSession] = {}
        self._visible_batches: dict[str, SearchBatch] = {}
        self._visible_session_id: str | None = None
        self._activated_question_ids: set[str] = set()
        self._diagnostics: list[EnvironmentDiagnostic] = []
        self._controller_inputs: list[ControllerInput] = []
        self._exact_results: list[ExactLookupResult] = []
        self._stop_reason: str | None = None

    def run(
        self,
        *,
        root_question: RootQuestion,
        questions: list[QuestionState] | None = None,
        run_key: str = "v0",
    ) -> ReadingRunResult:
        # A ReadingEnvironment may be reused in tests or services.  Canonical
        # Document/search indexes are reusable; per-run state is not.
        self._table_views = {}
        self._sessions = {}
        self._visible_batches = {}
        self._visible_session_id = None
        self._activated_question_ids = set()
        self._diagnostics = []
        self._controller_inputs = []
        self._exact_results = []
        self._stop_reason = None
        session_id = make_reading_session_id(root_question.question_id, run_key)
        memory = initialize_evidence_memory(
            reading_session_id=session_id,
            root_question_id=root_question.question_id,
            root_question_text=root_question.text,
            questions=questions or [],
        )
        observations = ObservationStore(
            reading_session_id=session_id,
            root_question_id=root_question.question_id,
        )
        trace = ActionTrace(
            reading_session_id=session_id,
            root_question_id=root_question.question_id,
        )

        while (
            memory.root_status != EvidenceStatus.READY
            and self._stop_reason is None
            and len(trace.entries) < self.config.action_budget
        ):
            target = memory.current_target
            if target is None:
                raise ValueError("Incomplete reading state lost its current target")
            if target.question_id not in self._activated_question_ids:
                self._activated_question_ids.add(target.question_id)
                observations, memory, trace, routed = self._route_exact_anchors(
                    root_question=root_question,
                    memory=memory,
                    observations=observations,
                    trace=trace,
                )
                if routed:
                    self._validate_state(observations, memory, trace)
                    continue

            controller_input = self._build_controller_input(
                root_question=root_question,
                memory=memory,
                observations=observations,
                trace=trace,
            )
            self._controller_inputs.append(controller_input)
            proposed = self.controller.decide(controller_input)
            action = validate_controller_action(proposed, controller_input)
            observations, memory, trace = self._execute_controller_action(
                action=action,
                root_question=root_question,
                memory=memory,
                observations=observations,
                trace=trace,
            )
            self._validate_state(observations, memory, trace)

        answer = None
        if memory.root_status == EvidenceStatus.READY:
            answer_input = self._answer_builder.build(
                root_question=root_question,
                evidence_memory=memory,
            )
            answer = validate_answer_result(
                answer_input,
                self.answerer.answer(answer_input),
            )
            status = ReadingRunStatus.READY
        elif self._stop_reason is not None:
            status = ReadingRunStatus.STOPPED_INCOMPLETE
        else:
            status = ReadingRunStatus.BUDGET_EXHAUSTED
            self._diagnostics.append(
                EnvironmentDiagnostic(
                    code="action_budget_exhausted",
                    description=(
                        "The run stopped with incomplete Evidence because the v0 "
                        "action budget was exhausted."
                    ),
                    question_id=(
                        memory.current_target.question_id
                        if memory.current_target is not None
                        else None
                    ),
                )
            )

        return ReadingRunResult(
            status=status,
            root_question=root_question,
            evidence_memory=memory,
            observation_store=observations,
            action_trace=trace,
            search_sessions=list(self._sessions.values()),
            controller_input_history=self._controller_inputs,
            exact_lookup_results=self._exact_results,
            diagnostics=self._diagnostics,
            answer=answer,
        )

    def run_with_plan(
        self,
        *,
        root_question_id: str,
        plan: InitialPlan,
        run_key: str = "v0",
    ) -> ReadingRunResult:
        """Execute one already validated Planner result.

        Planning remains outside the environment: this method only maps the
        canonical Planner schema into runtime question state. It cannot
        silently alter, add, or reorder SubQuestions.
        """

        return self.run(
            root_question=RootQuestion(
                question_id=root_question_id,
                text=plan.original_question,
            ),
            questions=[
                QuestionState(
                    question_id=item.subquestion_id,
                    text=item.text,
                    depends_on=list(item.depends_on),
                )
                for item in plan.subquestions
            ],
            run_key=run_key,
        )

    def _route_exact_anchors(
        self,
        *,
        root_question: RootQuestion,
        memory: EvidenceMemory,
        observations: ObservationStore,
        trace: ActionTrace,
    ) -> tuple[ObservationStore, EvidenceMemory, ActionTrace, bool]:
        target = memory.current_target
        assert target is not None
        text = (
            root_question.text
            if target.question_id == root_question.question_id
            else next(
                item.text
                for item in memory.questions
                if item.question_id == target.question_id
            )
        )
        result = self.search.lookup_exact(
            SubQuestionInput(subquestion_id=target.question_id, text=text)
        )
        self._exact_results.append(result)
        direct_ids = [
            item.target_id
            for item in result.exact_anchor_matches
            if item.target_type.value != "section"
            and not self._source_was_attempted(item.target_id, observations)
        ]
        if direct_ids:
            observations, memory, trace = self._execute_read(
                source_ids=direct_ids,
                local_problem=target.gap_description,
                action_name="READ_SOURCE",
                root_question=root_question,
                memory=memory,
                observations=observations,
                trace=trace,
                metadata={"trigger": "exact_anchor"},
            )
            return observations, memory, trace, True

        section_matches = [
            item
            for item in result.exact_anchor_matches
            if item.target_type.value == "section"
        ]
        for match in section_matches:
            self._diagnostics.append(
                EnvironmentDiagnostic(
                    code="exact_section_scope_not_implemented",
                    description=(
                        "The Section Anchor was resolved, but v0 has no scoped "
                        "Section-read primitive; the Controller may still SEARCH."
                    ),
                    question_id=target.question_id,
                    metadata={"section_id": match.target_id},
                )
            )

        for resolution in result.anchor_resolutions:
            if resolution.status.value != "unique":
                self._diagnostics.append(
                    EnvironmentDiagnostic(
                        code=f"exact_anchor_{resolution.status.value}",
                        description=(
                            f"Exact Anchor {resolution.anchor_text!r} was not "
                            "automatically read."
                        ),
                        question_id=target.question_id,
                        metadata={"reason": resolution.reason},
                    )
                )
        return observations, memory, trace, False

    @staticmethod
    def _source_was_attempted(
        source_id: str, observation_store: ObservationStore
    ) -> bool:
        return any(
            item.source_id == source_id
            for record in observation_store.read_records
            for item in record.inputs
        ) or any(
            record.inputs
            and any(item.element_id == source_id for item in record.inputs)
            for record in observation_store.read_records
        )

    def _execute_controller_action(
        self,
        *,
        action: ControllerAction,
        root_question: RootQuestion,
        memory: EvidenceMemory,
        observations: ObservationStore,
        trace: ActionTrace,
    ) -> tuple[ObservationStore, EvidenceMemory, ActionTrace]:
        if isinstance(action, ControllerSearchAction):
            return observations, memory, self._execute_search(action, memory, trace)
        if isinstance(action, ControllerStopAction):
            self._stop_reason = action.reason
            target = memory.current_target
            assert target is not None
            entry = ActionTraceEntry(
                step_index=len(trace.entries),
                action_id=make_action_id(trace.reading_session_id, len(trace.entries)),
                question_id=target.question_id,
                action_name=action.action.value,
                execution_status=ActionExecutionStatus.SUCCEEDED,
                metadata={"reason": action.reason},
            )
            self._diagnostics.append(
                EnvironmentDiagnostic(
                    code="controller_stopped_incomplete",
                    description=action.reason,
                    action_id=entry.action_id,
                    question_id=target.question_id,
                )
            )
            return (
                observations,
                memory,
                ActionTrace(
                    reading_session_id=trace.reading_session_id,
                    root_question_id=trace.root_question_id,
                    entries=[*trace.entries, entry],
                ),
            )
        if isinstance(action, ControllerReadSourceAction):
            return self._execute_read(
                source_ids=action.source_ids,
                local_problem=action.local_problem,
                action_name=action.action.value,
                root_question=root_question,
                memory=memory,
                observations=observations,
                trace=trace,
            )
        if isinstance(
            action,
            (ControllerFollowRelationAction, ControllerExploreCandidateRelationAction),
        ):
            relation = self._relation(action.relation_id)
            target_id = self._other_relation_endpoint(relation, trace)
            return self._execute_read(
                source_ids=[target_id],
                local_problem=action.local_problem,
                action_name=action.action.value,
                root_question=root_question,
                memory=memory,
                observations=observations,
                trace=trace,
                metadata={"relation_id": relation.relation_id},
            )
        assert isinstance(action, ControllerReadAdjacentPageAction)
        adjacent = self._adjacent_page(action.from_page_id, action.direction)
        if adjacent is None:
            return observations, memory, self._append_failed_action(
                trace=trace,
                memory=memory,
                action_name=action.action.value,
                target_ids=[action.from_page_id],
                description="The requested adjacent page does not exist.",
            )
        return self._execute_read(
            source_ids=[adjacent.page_id],
            local_problem=action.local_problem,
            action_name=action.action.value,
            root_question=root_question,
            memory=memory,
            observations=observations,
            trace=trace,
            metadata={
                "from_page_id": action.from_page_id,
                "direction": action.direction.value,
            },
        )

    def _execute_search(
        self,
        action: ControllerSearchAction,
        memory: EvidenceMemory,
        trace: ActionTrace,
    ) -> ActionTrace:
        target = memory.current_target
        assert target is not None
        if action.operation == ControllerSearchOperation.NEW:
            assert action.query is not None
            session, batch = self.search.start(
                question_id=target.question_id,
                query=action.query,
            )
            if session.search_session_id in self._sessions:
                session = self._sessions[session.search_session_id]
                batch = self._visible_batches[session.search_session_id]
            else:
                self._sessions[session.search_session_id] = session
                self._visible_batches[session.search_session_id] = batch
            self._visible_session_id = session.search_session_id
            query = action.query
        else:
            assert action.search_session_id is not None
            session = self._sessions[action.search_session_id]
            if action.operation == ControllerSearchOperation.NEXT:
                session, batch = self.search.next_batch(session)
                self._sessions[session.search_session_id] = session
                self._visible_batches[session.search_session_id] = batch
            else:
                batch = self._visible_batches[session.search_session_id]
            self._visible_session_id = session.search_session_id
            query = None

        entry = ActionTraceEntry(
            step_index=len(trace.entries),
            action_id=make_action_id(trace.reading_session_id, len(trace.entries)),
            question_id=target.question_id,
            action_name="SEARCH",
            target_ids=[item.element_id for item in batch.candidate_previews],
            query=query,
            execution_status=ActionExecutionStatus.SUCCEEDED,
            metadata={
                "operation": action.operation.value,
                "search_session_id": session.search_session_id,
            },
        )
        return ActionTrace(
            reading_session_id=trace.reading_session_id,
            root_question_id=trace.root_question_id,
            entries=[*trace.entries, entry],
        )

    def _execute_read(
        self,
        *,
        source_ids: list[str],
        local_problem: str,
        action_name: str,
        root_question: RootQuestion,
        memory: EvidenceMemory,
        observations: ObservationStore,
        trace: ActionTrace,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[ObservationStore, EvidenceMemory, ActionTrace]:
        target = memory.current_target
        assert target is not None
        step = len(trace.entries)
        current_action_id = make_action_id(trace.reading_session_id, step)
        try:
            resolved = [
                self._read_input(source_id, index)
                for index, source_id in enumerate(source_ids)
            ]
        except (KeyError, ValueError) as exc:
            trace = self._append_failed_action(
                trace=trace,
                memory=memory,
                action_name=action_name,
                target_ids=source_ids,
                description=str(exc),
                metadata=metadata,
            )
            return observations, memory, trace

        primary = self._primary_handle(source_ids[0]) if len(source_ids) == 1 else None
        context = ReaderContext(
            action_id=current_action_id,
            question_id=target.question_id,
            local_problem=local_problem,
            document=self.document,
            inputs=tuple(resolved),
            elements_by_id={item.element_id: item for item in self.document.elements},
            pages_by_id={item.page_id: item for item in self.document.pages},
            table_views_by_id=dict(self._table_views),
        )
        try:
            reader_output = self.reader.read(context)
        except Exception as exc:  # model/client failures must not lose the action
            reader_output = ReaderOutput(
                reader_kind=ReaderKind.VISUAL,
                limitations=[
                    ObservationLimitation(
                        description=f"Reader backend failed: {exc}",
                        input_ids=[item.input_id for item in resolved],
                    )
                ],
            )

        input_ids = {item.input_id for item in resolved}
        for draft in reader_output.observations:
            unknown = {item.input_id for item in draft.sources}.difference(input_ids)
            if unknown:
                raise ValueError(
                    "Reader Observation references unavailable inputs: "
                    + ", ".join(sorted(unknown))
                )

        stored = [
            StoredObservation(
                observation_id=make_observation_id(current_action_id, index),
                action_id=current_action_id,
                text=draft.text,
                sources=draft.sources,
            )
            for index, draft in enumerate(reader_output.observations)
        ]
        record = ReadRecord(
            action_id=current_action_id,
            reader_kind=reader_output.reader_kind,
            document_id=self.document.document_id,
            subquestion_id=(
                target.question_id
                if target.question_id != root_question.question_id
                else None
            ),
            local_problem=local_problem,
            inputs=resolved,
            observation_ids=[item.observation_id for item in stored],
            limitations=reader_output.limitations,
        )
        next_observations = ObservationStore(
            reading_session_id=observations.reading_session_id,
            root_question_id=observations.root_question_id,
            read_records=[*observations.read_records, record],
            observations=[*observations.observations, *stored],
        )

        feedback = self._feedback(reader_output.limitations, [], record)
        status = (
            ActionExecutionStatus.DEGRADED
            if reader_output.limitations
            else ActionExecutionStatus.SUCCEEDED
        )
        action_metadata = dict(metadata or {})
        action_metadata["requested_action"] = action_name
        if feedback is not None:
            action_metadata["controller_feedback"] = feedback.model_dump(mode="json")
        entry = ActionTraceEntry(
            step_index=step,
            action_id=current_action_id,
            question_id=target.question_id,
            action_name=action_name,
            target_ids=source_ids,
            primary_target=primary,
            execution_status=status,
            observation_ids=[item.observation_id for item in stored],
            metadata=action_metadata,
        )
        next_trace = ActionTrace(
            reading_session_id=trace.reading_session_id,
            root_question_id=trace.root_question_id,
            entries=[*trace.entries, entry],
        )

        for source_id in source_ids:
            for session_id, session in list(self._sessions.items()):
                if source_id in session.shown_candidate_ids:
                    self._sessions[session_id] = self.search.mark_opened(
                        session, source_id
                    )

        next_memory = memory
        if stored:
            checker_input = EvidenceCheckInput(
                action_id=current_action_id,
                root_question=root_question,
                evidence_memory=memory,
                observations=stored,
                limitations=reader_output.limitations,
            )
            try:
                check_result = self.checker.check(checker_input)
                next_memory = apply_evidence_check_result(checker_input, check_result)
                feedback = self._feedback(
                    reader_output.limitations,
                    check_result.observation_assessments,
                    record,
                )
            except Exception as exc:
                self._diagnostics.append(
                    EnvironmentDiagnostic(
                        code="checker_result_rejected",
                        description=str(exc),
                        action_id=current_action_id,
                        question_id=target.question_id,
                    )
                )
                feedback = ControllerActionFeedback(
                    limitations=self._controller_limitations(
                        reader_output.limitations, record
                    ),
                    observation_assessments=[
                        ControllerObservationAssessment(
                            observation_id=item.observation_id,
                            used_for_evidence=False,
                            assessment=(
                                "The Checker result was rejected by validation; "
                                "EvidenceMemory was not changed."
                            ),
                        )
                        for item in stored
                    ],
                )
                entry = entry.model_copy(
                    update={"execution_status": ActionExecutionStatus.DEGRADED}
                )

            metadata_with_feedback = dict(entry.metadata)
            if feedback is not None:
                metadata_with_feedback["controller_feedback"] = feedback.model_dump(
                    mode="json"
                )
            entry = entry.model_copy(update={"metadata": metadata_with_feedback})
            next_trace = ActionTrace(
                reading_session_id=trace.reading_session_id,
                root_question_id=trace.root_question_id,
                entries=[*trace.entries, entry],
            )

        return next_observations, next_memory, next_trace

    def _feedback(
        self,
        limitations: list[ObservationLimitation],
        assessments: list[Any],
        record: ReadRecord,
    ) -> ControllerActionFeedback | None:
        controller_limitations = self._controller_limitations(limitations, record)
        controller_assessments = [
            ControllerObservationAssessment(
                observation_id=item.observation_id,
                used_for_evidence=item.used_for_evidence,
                assessment=item.assessment,
            )
            for item in assessments
        ]
        if not controller_limitations and not controller_assessments:
            return None
        return ControllerActionFeedback(
            limitations=controller_limitations,
            observation_assessments=controller_assessments,
        )

    @staticmethod
    def _controller_limitations(
        limitations: list[ObservationLimitation], record: ReadRecord
    ) -> list[ControllerLimitation]:
        inputs_by_id = {item.input_id: item for item in record.inputs}
        return [
            ControllerLimitation(
                description=item.description,
                source_ids=[
                    inputs_by_id[input_id].source_id
                    for input_id in item.input_ids
                    if input_id in inputs_by_id
                ],
            )
            for item in limitations
        ]

    def _read_input(self, source_id: str, index: int) -> ReadInput:
        input_id = read_input_id(index)
        try:
            page = self.store.get_page(source_id)
        except KeyError:
            page = None
        if page is not None:
            path = self._asset_path(page.image_path)
            if path is None:
                raise ValueError(f"Page has no visual asset: {source_id}")
            return ReadInput(
                input_id=input_id,
                source_id=page.page_id,
                source_type=ReadingSourceType.PAGE,
                representation=ReadRepresentation.PAGE_VISUAL,
                document_id=self.document.document_id,
                page_id=page.page_id,
                visual_asset_id=(
                    "visual:"
                    + stable_digest(
                        self.document.document_id,
                        page.page_id,
                        "page_visual",
                    )
                ),
                visual_asset_path=path,
            )

        element = self.store.get_element(source_id)
        if element.element_type == ElementType.TABLE and element.html:
            built = self._table_materializer.materialize(
                element, document_root=self.asset_root
            )
            view = built.view
            self._table_views[view.table_view_id] = view
            return ReadInput(
                input_id=input_id,
                source_id=view.table_view_id,
                source_type=ReadingSourceType.TABLE_VIEW,
                representation=ReadRepresentation.TABLE_VIEW,
                document_id=self.document.document_id,
                page_id=element.page_id,
                element_id=element.element_id,
                table_view_id=view.table_view_id,
            )

        visual_path = self._asset_path(element.visual_asset_path)
        if visual_path is not None:
            return ReadInput(
                input_id=input_id,
                source_id=element.element_id,
                source_type=ReadingSourceType.ELEMENT,
                representation=ReadRepresentation.ELEMENT_VISUAL,
                document_id=self.document.document_id,
                page_id=element.page_id,
                element_id=element.element_id,
                visual_asset_id=(
                    "visual:"
                    + stable_digest(
                        self.document.document_id,
                        element.element_id,
                        "element_visual",
                    )
                ),
                visual_asset_path=visual_path,
            )

        if (element.text or element.html or "").strip():
            return ReadInput(
                input_id=input_id,
                source_id=element.element_id,
                source_type=ReadingSourceType.ELEMENT,
                representation=ReadRepresentation.ELEMENT_TEXT,
                document_id=self.document.document_id,
                page_id=element.page_id,
                element_id=element.element_id,
            )
        raise ValueError(f"Element has no readable representation: {source_id}")

    def _asset_path(self, path: Path | None) -> Path | None:
        if path is None:
            return None
        # SoftDocs may be authored on Windows and consumed on Linux. Relative
        # asset references retain the source platform's path separator.
        path = Path(str(path).replace("\\", "/"))
        resolved = path if path.is_absolute() else self.asset_root / path
        return resolved if resolved.is_file() else None

    def _primary_handle(self, source_id: str) -> ExplorationSourceHandle:
        try:
            page = self.store.get_page(source_id)
            return ExplorationSourceHandle(
                source_id=page.page_id,
                source_type=ReadingSourceType.PAGE,
                document_id=self.document.document_id,
                page_id=page.page_id,
            )
        except KeyError:
            element = self.store.get_element(source_id)
            return ExplorationSourceHandle(
                source_id=element.element_id,
                source_type=ReadingSourceType.ELEMENT,
                document_id=self.document.document_id,
                page_id=element.page_id,
                element_id=element.element_id,
            )

    def _relation(self, relation_id: str) -> Relation:
        for relation in self.document.relations:
            if relation.relation_id == relation_id:
                return relation
        raise KeyError(f"Unknown Relation: {relation_id}")

    @staticmethod
    def _other_relation_endpoint(
        relation: Relation, trace: ActionTrace
    ) -> str:
        focus = next(
            (
                item.primary_target.source_id
                for item in reversed(trace.entries)
                if item.primary_target is not None
                and item.execution_status
                in {ActionExecutionStatus.SUCCEEDED, ActionExecutionStatus.DEGRADED}
            ),
            None,
        )
        if focus == relation.source_id:
            return relation.target_id
        if focus == relation.target_id:
            return relation.source_id
        raise ValueError("Relation is not connected to the current reading focus")

    def _adjacent_page(
        self, page_id: str, direction: AdjacentPageDirection
    ) -> Page | None:
        page = self.store.get_page(page_id)
        offset = 1 if direction == AdjacentPageDirection.NEXT else -1
        target_index = page.page_index + offset
        return next(
            (item for item in self.document.pages if item.page_index == target_index),
            None,
        )

    def _append_failed_action(
        self,
        *,
        trace: ActionTrace,
        memory: EvidenceMemory,
        action_name: str,
        target_ids: list[str],
        description: str,
        metadata: dict[str, Any] | None = None,
    ) -> ActionTrace:
        target = memory.current_target
        assert target is not None
        current_action_id = make_action_id(trace.reading_session_id, len(trace.entries))
        action_metadata = dict(metadata or {})
        action_metadata["failure"] = description
        self._diagnostics.append(
            EnvironmentDiagnostic(
                code="action_execution_failed",
                description=description,
                action_id=current_action_id,
                question_id=target.question_id,
            )
        )
        entry = ActionTraceEntry(
            step_index=len(trace.entries),
            action_id=current_action_id,
            question_id=target.question_id,
            action_name=action_name,
            target_ids=target_ids,
            execution_status=ActionExecutionStatus.FAILED,
            metadata=action_metadata,
        )
        return ActionTrace(
            reading_session_id=trace.reading_session_id,
            root_question_id=trace.root_question_id,
            entries=[*trace.entries, entry],
        )

    def _build_controller_input(
        self,
        *,
        root_question: RootQuestion,
        memory: EvidenceMemory,
        observations: ObservationStore,
        trace: ActionTrace,
    ) -> ControllerInput:
        visible_batch = (
            self._visible_batches.get(self._visible_session_id)
            if self._visible_session_id is not None
            else None
        )
        return self._input_builder.build(
            root_question=root_question,
            evidence_memory=memory,
            observation_store=observations,
            action_trace=trace,
            relations=self.document.relations,
            relation_sources=[*self.document.pages, *self.document.elements],
            readable_source_ids=[
                *[page.page_id for page in self.document.pages],
                *[element.element_id for element in self.document.elements],
            ],
            search_sessions=self._sessions.values(),
            visible_search_batch=visible_batch,
            remaining_action_budget=(
                self.config.action_budget - len(trace.entries)
            ),
            recent_action_limit=self.config.recent_action_limit,
        )

    def _validate_state(
        self,
        observations: ObservationStore,
        memory: EvidenceMemory,
        trace: ActionTrace,
    ) -> None:
        self._reference_validator.validate(
            observation_store=observations,
            evidence_memory=memory,
            action_trace=trace,
            search_sessions=self._sessions.values(),
            relations=self.document.relations,
            raise_on_error=True,
        )
