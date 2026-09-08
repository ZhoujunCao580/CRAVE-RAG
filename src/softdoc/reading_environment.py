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
import re
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
    ControllerReadPageContextAction,
    ControllerReadSourceAction,
    ControllerSearchAction,
    ControllerSearchOperation,
    ControllerStopAction,
    validate_controller_action,
)
from softdoc.coverage_reasoning import (
    CoverageBatchCheckInput,
    CoverageBatchCheckResult,
    CoverageExecutionProgress,
    CoverageExecutionStatus,
    CoverageInventoryStatus,
    CoverageItemVerdict,
    CoverageLimitation,
    CoverageObservation,
    CoverageOperator,
    CoverageRequirement,
    CoverageScopeResolver,
    CoverageScopeStatus,
    PageNumberNamespace,
    QuestionCoveragePlan,
    apply_coverage_batch_result,
    build_coverage_inventory,
    validate_coverage_batch_result,
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
    EvidenceAddition,
    EvidenceCheckInput,
    EvidenceCheckResult,
    EvidenceMemory,
    EvidenceStatus,
    EvidenceUpdates,
    ExplorationSourceHandle,
    ObservationAssessment,
    ObservationLimitation,
    ObservationSourceRef,
    ObservationStore,
    QuestionState,
    QuestionStatus,
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
    CandidatePreview,
    CandidateSelectionRoute,
    DenseSearchResult,
    ExactAnchorLookup,
    ExactLookupResult,
    PreviewMatchScope,
    RetrievalSource,
    SearchBatch,
    SearchSession,
    SearchSessionBuilder,
    SearchSessionConfig,
    SearchSessionNavigator,
    SearchUnitBuildResult,
    SearchUnitBuilder,
    SnippetSource,
    SubQuestionInput,
    VisualSearchResult,
    html_to_text,
)
from softdoc.store import DocumentStore
from softdoc.table_view import TableMaterializer, TableView


READING_ENVIRONMENT_VERSION = "reading-environment-v0.6"


_RECALL_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "which",
        "with",
    }
)


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


class SemanticCoverageCheckerBackend(Protocol):
    def check_coverage(
        self, checker_input: CoverageBatchCheckInput
    ) -> CoverageBatchCheckResult:
        """Assess every canonical inventory item in one bounded batch."""


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
    observation_recall_limit: int = Field(default=3, ge=0, le=10)
    coverage_batch_size: int = Field(default=5, ge=1, le=16)
    search: SearchSessionConfig = Field(default_factory=SearchSessionConfig)
    coverage_namespace_overrides: dict[str, PageNumberNamespace] = Field(
        default_factory=dict
    )


class ReadingRunResult(SoftDocModel):
    environment_version: str = READING_ENVIRONMENT_VERSION
    status: ReadingRunStatus
    root_question: RootQuestion
    evidence_memory: EvidenceMemory
    observation_store: ObservationStore
    action_trace: ActionTrace
    search_sessions: list[SearchSession] = Field(default_factory=list)
    visible_search_batches: list[SearchBatch] = Field(default_factory=list)
    visible_search_session_id: str | None = Field(default=None, min_length=1)
    activated_question_ids: list[str] = Field(default_factory=list)
    controller_input_history: list[ControllerInput] = Field(default_factory=list)
    exact_lookup_results: list[ExactLookupResult] = Field(default_factory=list)
    coverage_plans: list[QuestionCoveragePlan] = Field(default_factory=list)
    diagnostics: list[EnvironmentDiagnostic] = Field(default_factory=list)
    answer: AnswerResult | None = None

    @model_validator(mode="after")
    def validate_resume_state(self) -> Self:
        session_ids = [item.search_session_id for item in self.search_sessions]
        if len(session_ids) != len(set(session_ids)):
            raise ValueError("ReadingRunResult SearchSession IDs must be unique")
        batch_ids = [
            item.search_session_id for item in self.visible_search_batches
        ]
        if len(batch_ids) != len(set(batch_ids)):
            raise ValueError("ReadingRunResult visible SearchBatch IDs must be unique")
        unknown_batches = set(batch_ids).difference(session_ids)
        if unknown_batches:
            raise ValueError(
                "ReadingRunResult visible batches reference missing sessions: "
                + ", ".join(sorted(unknown_batches))
            )
        if self.visible_search_session_id is not None:
            if self.visible_search_session_id not in session_ids:
                raise ValueError("Visible SearchSession is missing from the registry")
            if (
                self.environment_version == READING_ENVIRONMENT_VERSION
                and self.visible_search_session_id not in batch_ids
            ):
                raise ValueError("Visible SearchSession requires a visible SearchBatch")
        if len(self.activated_question_ids) != len(
            set(self.activated_question_ids)
        ):
            raise ValueError("Activated question IDs must be unique")
        known_question_ids = {
            self.root_question.question_id,
            *[item.question_id for item in self.evidence_memory.questions],
        }
        unknown_questions = set(self.activated_question_ids).difference(
            known_question_ids
        )
        if unknown_questions:
            raise ValueError(
                "Activated question IDs are not registered: "
                + ", ".join(sorted(unknown_questions))
            )
        return self


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
        self._coverage_plans: list[QuestionCoveragePlan] = []
        self._stop_reason: str | None = None
        self._action_limit = self.config.action_budget

    def run(
        self,
        *,
        root_question: RootQuestion,
        questions: list[QuestionState] | None = None,
        coverage_requirements: dict[str, CoverageRequirement] | None = None,
        run_key: str = "v0",
    ) -> ReadingRunResult:
        # A ReadingEnvironment may be reused in tests or services.  Canonical
        # Document/search indexes are reusable; per-run state is not.
        self._reset_runtime_state()
        session_id = make_reading_session_id(root_question.question_id, run_key)
        memory = initialize_evidence_memory(
            reading_session_id=session_id,
            root_question_id=root_question.question_id,
            root_question_text=root_question.text,
            questions=questions or [],
        )
        known_question_ids = {
            root_question.question_id,
            *[item.question_id for item in memory.questions],
        }
        unknown_coverage_questions = set(coverage_requirements or {}).difference(
            known_question_ids
        )
        if unknown_coverage_questions:
            raise ValueError(
                "Coverage requirements reference unknown questions: "
                + ", ".join(sorted(unknown_coverage_questions))
            )
        observations = ObservationStore(
            reading_session_id=session_id,
            root_question_id=root_question.question_id,
        )
        trace = ActionTrace(
            reading_session_id=session_id,
            root_question_id=root_question.question_id,
        )
        self._prepare_coverage_plans(coverage_requirements or {})
        return self._continue_run(
            root_question=root_question,
            memory=memory,
            observations=observations,
            trace=trace,
            action_limit=self.config.action_budget,
        )

    def resume(
        self,
        previous: ReadingRunResult,
        *,
        additional_action_budget: int,
    ) -> ReadingRunResult:
        """Continue one budget-exhausted run without replaying prior model calls.

        Model backends are stateless at this boundary: their next inputs are
        rebuilt from the restored canonical reading state.  Legacy v0.2 runs
        did not persist the small private visibility registry, so it is
        reconstructed deterministically from SearchSession and ActionTrace.
        """

        if additional_action_budget < 1:
            raise ValueError("additional_action_budget must be positive")
        coverage_blocked = any(
            item.code
            in {
                "coverage_scope_execution_blocked",
                "coverage_semantic_execution_not_implemented",
                "coverage_semantic_checker_unavailable",
                "coverage_semantic_check_failed",
                "coverage_semantic_items_unresolved",
                "coverage_operator_not_implemented",
            }
            for item in previous.diagnostics
        )
        if previous.status != ReadingRunStatus.BUDGET_EXHAUSTED and not (
            previous.status == ReadingRunStatus.STOPPED_INCOMPLETE
            and coverage_blocked
        ):
            raise ValueError(
                "Only budget_exhausted runs or coverage-blocked checkpoints "
                "may be resumed"
            )
        if previous.answer is not None and not (
            previous.answer.answer == "Not answerable"
            and previous.answer.used_evidence_ids == []
        ):
            raise ValueError("A resumable run must not already contain an answer")
        if previous.root_question.question_id != previous.evidence_memory.root_question_id:
            raise ValueError("Resume Root Question does not match EvidenceMemory")
        if previous.evidence_memory.root_status == EvidenceStatus.READY:
            raise ValueError("A ready EvidenceMemory must not be resumed")
        action_limit = len(previous.action_trace.entries) + additional_action_budget
        if action_limit > 100:
            raise ValueError("A resumed run may contain at most 100 actions")

        self._restore_runtime_state(previous)
        memory = previous.evidence_memory.model_copy(deep=True)
        observations = previous.observation_store.model_copy(deep=True)
        trace = previous.action_trace.model_copy(deep=True)
        self._validate_state(observations, memory, trace)
        return self._continue_run(
            root_question=previous.root_question.model_copy(deep=True),
            memory=memory,
            observations=observations,
            trace=trace,
            action_limit=action_limit,
        )

    def _reset_runtime_state(self) -> None:
        self._table_views = {}
        self._sessions = {}
        self._visible_batches = {}
        self._visible_session_id = None
        self._activated_question_ids = set()
        self._diagnostics = []
        self._controller_inputs = []
        self._exact_results = []
        self._coverage_plans = []
        self._stop_reason = None
        self._action_limit = self.config.action_budget

    def _restore_runtime_state(self, previous: ReadingRunResult) -> None:
        self._reset_runtime_state()
        if any(
            session.document_id != self.document.document_id
            for session in previous.search_sessions
        ):
            raise ValueError("Resume SearchSession belongs to another Document")
        self._sessions = {
            session.search_session_id: session.model_copy(deep=True)
            for session in previous.search_sessions
        }
        if len(self._sessions) != len(previous.search_sessions):
            raise ValueError("Resume SearchSession IDs must be unique")

        if previous.visible_search_batches:
            self._visible_batches = {
                batch.search_session_id: batch.model_copy(deep=True)
                for batch in previous.visible_search_batches
            }
            if len(self._visible_batches) != len(previous.visible_search_batches):
                raise ValueError("Resume visible SearchBatch IDs must be unique")
        else:
            self._visible_batches = {}
            for session_id, session in self._sessions.items():
                if session.cursor <= 0:
                    continue
                session = self._remap_legacy_search_unit_ids(session)
                self._sessions[session_id] = session
                try:
                    batch = self._reconstruct_visible_batch(session)
                except ValueError as exc:
                    if "unavailable SearchUnit" not in str(exc):
                        raise
                    batch = self._restore_legacy_visible_batch(
                        previous,
                        session,
                    )
                self._visible_batches[session_id] = batch

        visible_session_id = previous.visible_search_session_id
        if visible_session_id is None:
            visible_session_id = self._infer_visible_session_id(previous.action_trace)
        if visible_session_id is not None:
            if visible_session_id not in self._sessions:
                raise ValueError("Resume visible SearchSession is missing")
            if visible_session_id not in self._visible_batches:
                raise ValueError("Resume visible SearchBatch is missing")
        self._visible_session_id = visible_session_id

        activated = previous.activated_question_ids or self._infer_activated_questions(
            previous
        )
        known_question_ids = {
            previous.root_question.question_id,
            *[
                question.question_id
                for question in previous.evidence_memory.questions
            ],
        }
        unknown_activated = set(activated).difference(known_question_ids)
        if unknown_activated:
            raise ValueError(
                "Resume activated question IDs are unknown: "
                + ", ".join(sorted(unknown_activated))
            )
        self._activated_question_ids = set(activated)
        coverage_diagnostic_codes = {
            "coverage_scope_not_resolved",
            "coverage_page_namespace_review_recommended",
            "coverage_scope_execution_blocked",
            "coverage_semantic_execution_not_implemented",
            "coverage_semantic_checker_unavailable",
            "coverage_semantic_check_failed",
            "coverage_semantic_items_unresolved",
            "coverage_operator_not_implemented",
        }
        self._diagnostics = [
            item.model_copy(deep=True)
            for item in previous.diagnostics
            if item.code != "action_budget_exhausted"
            and item.code not in coverage_diagnostic_codes
        ]
        self._controller_inputs = [
            item.model_copy(deep=True)
            for item in previous.controller_input_history
        ]
        self._exact_results = [
            item.model_copy(deep=True)
            for item in previous.exact_lookup_results
        ]
        self._prepare_coverage_plans(
            {
                item.question_id: item.requirement
                for item in previous.coverage_plans
            }
        )
        previous_plans = {item.question_id: item for item in previous.coverage_plans}
        self._coverage_plans = [
            plan.model_copy(
                update={
                    "execution": self._resumable_coverage_execution(
                        previous_plans[plan.question_id].execution
                    )
                }
            )
            if plan.question_id in previous_plans
            and previous_plans[plan.question_id].requirement == plan.requirement
            and previous_plans[plan.question_id].scope_resolution
            == plan.scope_resolution
            and previous_plans[plan.question_id].inventory == plan.inventory
            else plan
            for plan in self._coverage_plans
        ]

    @staticmethod
    def _resumable_coverage_execution(
        execution: CoverageExecutionProgress,
    ) -> CoverageExecutionProgress:
        """Retry only unresolved items while preserving resolved batch work."""

        if execution.status != CoverageExecutionStatus.BLOCKED:
            return execution.model_copy(deep=True)
        resolved = [
            item
            for item in execution.assessments
            if item.verdict != CoverageItemVerdict.UNRESOLVED
        ]
        return CoverageExecutionProgress(
            status=(
                CoverageExecutionStatus.IN_PROGRESS
                if resolved
                else CoverageExecutionStatus.PENDING
            ),
            assessments=resolved,
            completed_batch_count=execution.completed_batch_count,
            matched_count=None,
            matched_values=list(execution.matched_values),
            completion_reason="retrying_previously_unresolved_items",
        )

    def _prepare_coverage_plans(
        self,
        requirements: dict[str, CoverageRequirement],
    ) -> None:
        resolver = CoverageScopeResolver()
        self._coverage_plans = []
        for question_id, requirement in requirements.items():
            resolution = resolver.resolve(
                requirement,
                self.document,
                namespace_override=self.config.coverage_namespace_overrides.get(
                    question_id
                ),
            )
            inventory = build_coverage_inventory(
                requirement,
                resolution,
                self.document,
            )
            self._coverage_plans.append(
                QuestionCoveragePlan(
                    question_id=question_id,
                    requirement=requirement,
                    scope_resolution=resolution,
                    inventory=inventory,
                )
            )
            if resolution.status != CoverageScopeStatus.RESOLVED:
                self._diagnostics.append(
                    EnvironmentDiagnostic(
                        code="coverage_scope_not_resolved",
                        description=(
                            "Coverage scope was not fully resolved; deterministic "
                            "coverage completion is blocked and no partial count "
                            "may be treated as complete."
                        ),
                        question_id=question_id,
                        metadata={
                            "raw_scope_text": resolution.raw_scope_text,
                            "chosen_namespace": (
                                resolution.chosen_namespace.value
                                if resolution.chosen_namespace is not None
                                else None
                            ),
                            "status": resolution.status.value,
                            "missing_labels": resolution.missing_labels,
                            "ambiguous_labels": resolution.ambiguous_labels,
                        },
                    )
                )
            elif resolution.requires_review:
                self._diagnostics.append(
                    EnvironmentDiagnostic(
                        code="coverage_page_namespace_review_recommended",
                        description=(
                            "Both printed-label and physical-page interpretations "
                            "resolve but select different pages. The recorded choice "
                            "can be replaced with an explicit namespace override."
                        ),
                        question_id=question_id,
                        metadata={
                            "raw_scope_text": resolution.raw_scope_text,
                            "chosen_namespace": resolution.chosen_namespace.value,
                        },
                    )
                )

    def _reconstruct_visible_batch(self, session: SearchSession) -> SearchBatch:
        size = session.config.batch_size
        start = ((session.cursor - 1) // size) * size
        visible_ids = session.shown_candidate_ids[start : session.cursor]
        return SearchBatch(
            search_session_id=session.search_session_id,
            exact_anchor_matches=session.exact_anchor_matches,
            unresolved_anchors=session.unresolved_anchors,
            candidate_previews=[
                self.search.navigator.get_preview(session, element_id)
                for element_id in visible_ids
            ],
            next_cursor=session.cursor,
            exhausted=session.exhausted,
            retrieval_trace=session.retrieval_trace,
        )

    def _remap_legacy_search_unit_ids(self, session: SearchSession) -> SearchSession:
        """Keep an old ranking usable after deterministic SearchUnit upgrades.

        SearchSession ranks stable Element IDs, but early checkpoints also
        persisted versioned SearchUnit IDs inside each candidate.  When a newer
        builder changes those derived IDs, remap only the source handle to a
        current unit for the same Element.  Ranking, cursor, shown/opened IDs,
        and all prior actions remain unchanged.
        """

        units_by_id = {
            item.search_unit_id: item for item in self.search.search_units.units
        }
        units_by_element: dict[str, list[Any]] = {}
        for unit in self.search.search_units.units:
            units_by_element.setdefault(unit.element_id, []).append(unit)
        changed = False
        candidates = []
        for candidate in session.candidate_catalog:
            referenced = [
                candidate.bm25_search_unit_id,
                candidate.dense_search_unit_id,
                candidate.table_preview_search_unit_id,
            ]
            if all(item is None or item in units_by_id for item in referenced):
                candidates.append(candidate)
                continue
            current_units = sorted(
                units_by_element.get(candidate.element_id, []),
                key=lambda item: (item.part_index, item.search_unit_id),
            )
            if not current_units:
                candidates.append(candidate)
                continue
            unit = current_units[0]
            update: dict[str, Any] = {}
            if candidate.bm25_search_unit_id is not None:
                update["bm25_search_unit_id"] = unit.search_unit_id
                update["bm25_matched_offsets"] = []
            if candidate.dense_search_unit_id is not None:
                update["dense_search_unit_id"] = unit.search_unit_id
                update["dense_match_start"] = 0
                update["dense_match_end"] = max(1, min(len(unit.search_text), 1))
            if candidate.table_preview_search_unit_id is not None:
                update["table_preview_search_unit_id"] = unit.search_unit_id
            candidates.append(candidate.model_copy(update=update))
            changed = True
        return (
            session.model_copy(update={"candidate_catalog": candidates})
            if changed
            else session
        )

    @staticmethod
    def _restore_legacy_visible_batch(
        previous: ReadingRunResult,
        session: SearchSession,
    ) -> SearchBatch:
        """Restore the exact card text saved before SearchUnit IDs changed.

        Legacy checkpoints did not persist ``SearchBatch``.  Their Controller
        history still contains the actual cards shown to the model, so a resume
        must prefer that immutable history over regenerating cards from a newer
        SearchUnit index.  The recovered provenance is used for display only;
        source actions remain validated against stable Element IDs.
        """

        view = next(
            (
                item.visible_search_view
                for item in reversed(previous.controller_input_history)
                if item.visible_search_view is not None
                and item.visible_search_view.search_session_id
                == session.search_session_id
            ),
            None,
        )
        if view is None:
            raise ValueError(
                "Legacy SearchSession cannot be restored because its shown "
                "CandidatePreview history is missing"
            )
        catalog = {item.element_id: item for item in session.candidate_catalog}
        previews: list[CandidatePreview] = []
        for shown in view.candidate_previews:
            candidate = catalog.get(shown.element_id)
            if candidate is None:
                raise ValueError(
                    "Legacy CandidatePreview refers to a missing SessionCandidate"
                )
            if (
                candidate.selection_route == CandidateSelectionRoute.VISUAL
                or not any(
                    source in candidate.matched_by
                    for source in (RetrievalSource.BM25, RetrievalSource.DENSE)
                )
            ):
                preview_source = RetrievalSource.VISUAL_DENSE
                snippet_source = SnippetSource.VISUAL_METADATA
                snippet_source_id = candidate.visual_asset_id
                matched_search_unit_id = None
            elif candidate.table_preview_search_unit_id is not None:
                preview_source = (
                    RetrievalSource.BM25
                    if RetrievalSource.BM25 in candidate.matched_by
                    else RetrievalSource.DENSE
                )
                snippet_source = SnippetSource.TABLE_PREVIEW
                snippet_source_id = candidate.table_preview_search_unit_id
                matched_search_unit_id = candidate.table_preview_search_unit_id
            else:
                preview_source = (
                    RetrievalSource.BM25
                    if RetrievalSource.BM25 in candidate.matched_by
                    else RetrievalSource.DENSE
                )
                snippet_source = SnippetSource.SEARCH_UNIT_TEXT
                snippet_source_id = (
                    candidate.bm25_search_unit_id
                    if preview_source == RetrievalSource.BM25
                    else candidate.dense_search_unit_id
                )
                matched_search_unit_id = snippet_source_id
            if snippet_source_id is None:
                raise ValueError("Legacy CandidatePreview has no persisted source ID")
            previews.append(
                CandidatePreview(
                    element_id=shown.element_id,
                    element_type=shown.element_type,
                    page_id=shown.page_id,
                    page_number=candidate.page_number,
                    section_path=shown.section_path,
                    display_label=candidate.display_label,
                    matched_snippet=shown.matched_snippet,
                    snippet_char_start=0,
                    snippet_char_end=len(shown.matched_snippet),
                    snippet_truncated=False,
                    snippet_source=snippet_source,
                    snippet_source_id=snippet_source_id,
                    matched_search_unit_id=matched_search_unit_id,
                    visual_asset_id=candidate.visual_asset_id,
                    matched_by=candidate.matched_by,
                    preview_source=preview_source,
                    match_scope=PreviewMatchScope.UNKNOWN,
                    bm25_rank=candidate.bm25_rank,
                    dense_rank=candidate.dense_rank,
                    visual_rank=candidate.visual_rank,
                    rrf_score=candidate.rrf_score,
                    content_availability=shown.content_availability,
                )
            )
        return SearchBatch(
            search_session_id=session.search_session_id,
            exact_anchor_matches=session.exact_anchor_matches,
            unresolved_anchors=session.unresolved_anchors,
            candidate_previews=previews,
            next_cursor=session.cursor,
            exhausted=session.exhausted,
            retrieval_trace=session.retrieval_trace,
        )

    @staticmethod
    def _infer_visible_session_id(trace: ActionTrace) -> str | None:
        return next(
            (
                str(entry.metadata["search_session_id"])
                for entry in reversed(trace.entries)
                if entry.action_name == "SEARCH"
                and entry.metadata.get("search_session_id")
            ),
            None,
        )

    @staticmethod
    def _infer_activated_questions(previous: ReadingRunResult) -> list[str]:
        activated: list[str] = []
        for question_id in [
            *[entry.question_id for entry in previous.action_trace.entries],
            *[
                item.current_gap.question_id
                for item in previous.controller_input_history
                if item.current_gap is not None
            ],
        ]:
            if question_id not in activated:
                activated.append(question_id)
        return activated

    def _continue_run(
        self,
        *,
        root_question: RootQuestion,
        memory: EvidenceMemory,
        observations: ObservationStore,
        trace: ActionTrace,
        action_limit: int,
    ) -> ReadingRunResult:
        self._action_limit = action_limit

        while (
            memory.root_status != EvidenceStatus.READY
            and self._stop_reason is None
            and len(trace.entries) < action_limit
        ):
            target = memory.current_target
            if target is None:
                raise ValueError("Incomplete reading state lost its current target")
            observations, memory, trace, counted = self._route_structural_coverage(
                root_question=root_question,
                memory=memory,
                observations=observations,
                trace=trace,
            )
            if counted:
                self._validate_state(observations, memory, trace)
                continue
            observations, memory, trace, inspected = self._route_semantic_coverage(
                root_question=root_question,
                memory=memory,
                observations=observations,
                trace=trace,
            )
            if inspected:
                self._validate_state(observations, memory, trace)
                continue
            if self._block_unavailable_coverage(target.question_id):
                break
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
            answer = AnswerResult(
                answer="Not answerable",
                used_evidence_ids=[],
            )
        else:
            status = ReadingRunStatus.BUDGET_EXHAUSTED
            answer = AnswerResult(
                answer="Not answerable",
                used_evidence_ids=[],
            )
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
            visible_search_batches=list(self._visible_batches.values()),
            visible_search_session_id=self._visible_session_id,
            activated_question_ids=sorted(self._activated_question_ids),
            controller_input_history=self._controller_inputs,
            exact_lookup_results=self._exact_results,
            coverage_plans=self._coverage_plans,
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
            coverage_requirements={
                **(
                    {root_question_id: plan.coverage_requirement}
                    if plan.coverage_requirement is not None
                    else {}
                ),
                **{
                    item.subquestion_id: item.coverage_requirement
                    for item in plan.subquestions
                    if item.coverage_requirement is not None
                },
            },
            run_key=run_key,
        )

    def _route_structural_coverage(
        self,
        *,
        root_question: RootQuestion,
        memory: EvidenceMemory,
        observations: ObservationStore,
        trace: ActionTrace,
    ) -> tuple[ObservationStore, EvidenceMemory, ActionTrace, bool]:
        """Materialize an exact count only when canonical inventory is complete.

        This is an Environment-owned operation, not a model inference.  It is
        intentionally unavailable for semantic predicates or when the counted
        item differs from the enumerated source (for example, people inside
        figures).  Persisting it as one structural Observation keeps the final
        answer grounded in the exact canonical source set and makes resume and
        auditing use the same stores as ordinary Reader output.
        """

        target = memory.current_target
        if target is None:
            return observations, memory, trace, False
        plan = next(
            (
                item
                for item in self._coverage_plans
                if item.question_id == target.question_id
            ),
            None,
        )
        if plan is None or plan.inventory.structural_count is None:
            return observations, memory, trace, False
        if any(
            entry.action_name == "COUNT_INVENTORY"
            and entry.question_id == target.question_id
            for entry in trace.entries
        ):
            return observations, memory, trace, False

        step = len(trace.entries)
        current_action_id = make_action_id(trace.reading_session_id, step)
        pages_by_id = {page.page_id: page for page in self.document.pages}
        elements_by_id = {
            element.element_id: element for element in self.document.elements
        }
        ordered_source_ids: list[str] = []
        source_page_ids: dict[str, str] = {}
        for item in plan.inventory.items:
            for index, source_id in enumerate(item.source_ids):
                if source_id not in source_page_ids:
                    element = elements_by_id.get(source_id)
                    if element is not None:
                        source_page_ids[source_id] = element.page_id
                    else:
                        page_index = min(index, len(item.page_ids) - 1)
                        source_page_ids[source_id] = item.page_ids[page_index]
                    ordered_source_ids.append(source_id)
        # A valid zero count still needs provenance.  Ground it in every page
        # of the exhaustively resolved scope rather than inventing a source.
        if not ordered_source_ids:
            for page in plan.scope_resolution.resolved_pages:
                ordered_source_ids.append(page.page_id)
                source_page_ids[page.page_id] = page.page_id

        inputs: list[ReadInput] = []
        source_refs: list[ObservationSourceRef] = []
        for index, source_id in enumerate(ordered_source_ids):
            page_id = source_page_ids[source_id]
            page = pages_by_id[page_id]
            element = elements_by_id.get(source_id)
            input_id = read_input_id(index)
            read_input = ReadInput(
                input_id=input_id,
                source_id=source_id,
                source_type=(
                    ReadingSourceType.ELEMENT
                    if element is not None
                    else ReadingSourceType.PAGE
                ),
                representation=ReadRepresentation.STRUCTURAL_METADATA,
                document_id=self.document.document_id,
                page_id=page_id,
                element_id=element.element_id if element is not None else None,
            )
            inputs.append(read_input)
            source_refs.append(
                ObservationSourceRef(
                    input_id=input_id,
                    source_id=source_id,
                    page_id=page_id,
                    physical_page_number=page.page_number,
                    display_page_label=page.display_page_label,
                    element_id=(
                        element.element_id if element is not None else None
                    ),
                )
            )

        count = plan.inventory.structural_count
        observation = StoredObservation(
            observation_id=make_observation_id(current_action_id, 0),
            action_id=current_action_id,
            text=(
                f"The exhaustively resolved scope {plan.requirement.scope_text!r} "
                f"contains exactly {count} canonical "
                f"{plan.requirement.source_type.value} source(s), after confirmed "
                "cross-page deduplication."
            ),
            sources=source_refs,
        )
        record = ReadRecord(
            action_id=current_action_id,
            reader_kind="coverage",
            document_id=self.document.document_id,
            subquestion_id=(
                target.question_id
                if target.question_id != root_question.question_id
                else None
            ),
            local_problem=target.gap_description,
            inputs=inputs,
            observation_ids=[observation.observation_id],
        )
        next_observations = ObservationStore(
            reading_session_id=observations.reading_session_id,
            root_question_id=observations.root_question_id,
            read_records=[*observations.read_records, record],
            observations=[*observations.observations, observation],
        )
        checker_input = EvidenceCheckInput(
            action_id=current_action_id,
            root_question=root_question,
            evidence_memory=memory,
            observations=[observation],
        )
        check_result = EvidenceCheckResult(
            action_id=current_action_id,
            observation_assessments=[
                ObservationAssessment(
                    observation_id=observation.observation_id,
                    used_for_evidence=True,
                    assessment=(
                        "Accepted deterministic Environment inventory with "
                        "complete resolved scope."
                    ),
                )
            ],
            evidence_updates=EvidenceUpdates(
                add=[
                    EvidenceAddition(
                        statement=observation.text,
                        observation_ids=[observation.observation_id],
                        supports_question_ids=[target.question_id],
                    )
                ]
            ),
            current_target_status=QuestionStatus.SATISFIED,
            root_status=(
                EvidenceStatus.READY
                if target.question_id == root_question.question_id
                else EvidenceStatus.INCOMPLETE
            ),
        )
        next_memory = apply_evidence_check_result(checker_input, check_result)
        entry = ActionTraceEntry(
            step_index=step,
            action_id=current_action_id,
            question_id=target.question_id,
            action_name="COUNT_INVENTORY",
            target_ids=[item.inventory_id for item in plan.inventory.items],
            execution_status=ActionExecutionStatus.SUCCEEDED,
            observation_ids=[observation.observation_id],
            metadata={
                "scope_text": plan.requirement.scope_text,
                "page_number_namespace": plan.scope_resolution.chosen_namespace.value,
                "resolved_page_ids": plan.scope_resolution.resolved_page_ids,
                "source_type": plan.requirement.source_type.value,
                "structural_count": count,
            },
        )
        next_trace = ActionTrace(
            reading_session_id=trace.reading_session_id,
            root_question_id=trace.root_question_id,
            entries=[*trace.entries, entry],
        )
        if next_memory.root_status != EvidenceStatus.READY:
            next_memory = self._advance_state_only_checks(
                root_question=root_question,
                previous_memory=memory,
                memory=next_memory,
                observations=next_observations,
                triggering_action_id=current_action_id,
            )
        return next_observations, next_memory, next_trace, True

    def _route_semantic_coverage(
        self,
        *,
        root_question: RootQuestion,
        memory: EvidenceMemory,
        observations: ObservationStore,
        trace: ActionTrace,
    ) -> tuple[ObservationStore, EvidenceMemory, ActionTrace, bool]:
        """Read and judge one bounded batch from a semantic count inventory."""

        target = memory.current_target
        if target is None:
            return observations, memory, trace, False
        plan = next(
            (item for item in self._coverage_plans if item.question_id == target.question_id),
            None,
        )
        if (
            plan is None
            or plan.inventory.status != CoverageInventoryStatus.COMPLETE
            or plan.inventory.structural_count is not None
            or plan.requirement.operator != CoverageOperator.COUNT
            or plan.execution.status
            in {CoverageExecutionStatus.COMPLETE, CoverageExecutionStatus.BLOCKED}
        ):
            return observations, memory, trace, False

        checker_method = getattr(self.checker, "check_coverage", None)
        if checker_method is None:
            return observations, memory, trace, False

        assessed_ids = plan.execution.assessed_inventory_ids
        pending = [
            item for item in plan.inventory.items if item.inventory_id not in assessed_ids
        ]
        if not pending:
            return observations, memory, trace, False
        batch_items = pending[: self.config.coverage_batch_size]
        step = len(trace.entries)
        current_action_id = make_action_id(trace.reading_session_id, step)
        question_text = (
            root_question.text
            if target.question_id == root_question.question_id
            else next(
                item.text
                for item in memory.questions
                if item.question_id == target.question_id
            )
        )
        local_problem = self._coverage_local_problem(question_text, plan.requirement)

        read_inputs: list[ReadInput] = []
        readable_inputs: list[ReadInput] = []
        input_to_inventory: dict[str, str] = {}
        reader_limitations: list[ObservationLimitation] = []
        for inventory_item in batch_items:
            for source_id in inventory_item.source_ids:
                input_index = len(read_inputs)
                try:
                    read_input = self._read_input(source_id, input_index)
                    readable_inputs.append(read_input)
                except (KeyError, ValueError) as exc:
                    read_input = self._coverage_metadata_input(
                        inventory_item=inventory_item,
                        source_id=source_id,
                        index=input_index,
                    )
                    reader_limitations.append(
                        ObservationLimitation(
                            code="coverage_source_unreadable",
                            description=str(exc),
                            input_ids=[read_input.input_id],
                        )
                    )
                read_inputs.append(read_input)
                input_to_inventory[read_input.input_id] = inventory_item.inventory_id

        if readable_inputs:
            context = ReaderContext(
                action_id=current_action_id,
                question_id=target.question_id,
                local_problem=local_problem,
                document=self.document,
                inputs=tuple(readable_inputs),
                elements_by_id={item.element_id: item for item in self.document.elements},
                pages_by_id={item.page_id: item for item in self.document.pages},
                table_views_by_id=dict(self._table_views),
            )
            try:
                reader_output = self.reader.read(context)
            except Exception as exc:
                reader_output = ReaderOutput(
                    reader_kind=ReaderKind.VISUAL,
                    limitations=[
                        ObservationLimitation(
                            code="coverage_reader_failed",
                            description=f"Reader backend failed: {exc}",
                            input_ids=[item.input_id for item in readable_inputs],
                        )
                    ],
                )
            reader_limitations.extend(reader_output.limitations)
            reader_kind: ReaderKind | str = reader_output.reader_kind
            drafts = reader_output.observations
        else:
            reader_kind = "coverage"
            drafts = []

        inputs_by_id = {item.input_id: item for item in read_inputs}
        pages_by_id = {item.page_id: item for item in self.document.pages}
        for draft in drafts:
            unknown = {item.input_id for item in draft.sources}.difference(inputs_by_id)
            if unknown:
                raise ValueError(
                    "Reader Observation references unavailable coverage inputs: "
                    + ", ".join(sorted(unknown))
                )
        stored = [
            StoredObservation(
                observation_id=make_observation_id(current_action_id, index),
                action_id=current_action_id,
                text=draft.text,
                sources=[
                    source.model_copy(
                        update={
                            "source_id": inputs_by_id[source.input_id].source_id,
                            "page_id": inputs_by_id[source.input_id].page_id,
                            "physical_page_number": pages_by_id[
                                inputs_by_id[source.input_id].page_id
                            ].page_number,
                            "display_page_label": pages_by_id[
                                inputs_by_id[source.input_id].page_id
                            ].display_page_label,
                            "element_id": inputs_by_id[source.input_id].element_id,
                        }
                    )
                    for source in draft.sources
                ],
            )
            for index, draft in enumerate(drafts)
        ]

        coverage_observations = [
            CoverageObservation(
                observation_id=item.observation_id,
                text=item.text,
                source_ids=[
                    source.source_id
                    for source in item.sources
                    if source.source_id is not None
                ],
                page_ids=list(
                    dict.fromkeys(
                        source.page_id
                        for source in item.sources
                        if source.page_id is not None
                    )
                ),
            )
            for item in stored
        ]
        coverage_limitations: list[CoverageLimitation] = []
        all_batch_ids = [item.inventory_id for item in batch_items]
        for limitation in reader_limitations:
            limitation_items = list(
                dict.fromkeys(
                    input_to_inventory[input_id]
                    for input_id in limitation.input_ids
                    if input_id in input_to_inventory
                )
            )
            if not limitation.input_ids:
                limitation_items = all_batch_ids
            if limitation_items:
                coverage_limitations.append(
                    CoverageLimitation(
                        description=limitation.description,
                        inventory_ids=limitation_items,
                    )
                )
        checker_input = CoverageBatchCheckInput(
            action_id=current_action_id,
            question_id=target.question_id,
            question_text=question_text,
            requirement=plan.requirement,
            inventory_items=batch_items,
            observations=coverage_observations,
            limitations=coverage_limitations,
        )

        check_error: Exception | None = None
        try:
            batch_result = validate_coverage_batch_result(
                checker_input, checker_method(checker_input)
            )
            updated_plan = apply_coverage_batch_result(plan, batch_result)
            self._replace_coverage_plan(updated_plan)
        except Exception as exc:
            check_error = exc
            batch_result = None
            updated_plan = plan

        if (
            batch_result is not None
            and updated_plan.execution.status == CoverageExecutionStatus.COMPLETE
        ):
            read_inputs, summary_sources = self._append_coverage_provenance_inputs(
                read_inputs=read_inputs,
                plan=updated_plan,
            )
            matched_count = updated_plan.execution.matched_count
            assert matched_count is not None
            summary = StoredObservation(
                observation_id=make_observation_id(current_action_id, len(stored)),
                action_id=current_action_id,
                text=self._coverage_completion_statement(updated_plan),
                sources=summary_sources,
            )
            stored.append(summary)
            checker_state_input = EvidenceCheckInput(
                action_id=current_action_id,
                root_question=root_question,
                evidence_memory=memory,
                observations=[summary],
            )
            deterministic_result = EvidenceCheckResult(
                action_id=current_action_id,
                observation_assessments=[
                    ObservationAssessment(
                        observation_id=summary.observation_id,
                        used_for_evidence=True,
                        assessment=(
                            "Accepted exhaustive semantic inventory after every "
                            "canonical item received a resolved verdict."
                        ),
                    )
                ],
                evidence_updates=EvidenceUpdates(
                    add=[
                        EvidenceAddition(
                            statement=summary.text,
                            observation_ids=[summary.observation_id],
                            supports_question_ids=[target.question_id],
                        )
                    ]
                ),
                current_target_status=QuestionStatus.SATISFIED,
                root_status=(
                    EvidenceStatus.READY
                    if target.question_id == root_question.question_id
                    else EvidenceStatus.INCOMPLETE
                ),
            )
            next_memory = apply_evidence_check_result(
                checker_state_input, deterministic_result
            )
        else:
            next_memory = memory

        record = ReadRecord(
            action_id=current_action_id,
            reader_kind=reader_kind,
            document_id=self.document.document_id,
            subquestion_id=(
                target.question_id
                if target.question_id != root_question.question_id
                else None
            ),
            local_problem=local_problem,
            inputs=read_inputs,
            observation_ids=[item.observation_id for item in stored],
            limitations=reader_limitations,
        )
        next_observations = ObservationStore(
            reading_session_id=observations.reading_session_id,
            root_question_id=observations.root_question_id,
            read_records=[*observations.read_records, record],
            observations=[*observations.observations, *stored],
        )

        verdicts = (
            {
                item.inventory_id: item.verdict.value
                for item in batch_result.assessments
            }
            if batch_result is not None
            else {}
        )
        execution_status = (
            ActionExecutionStatus.DEGRADED
            if reader_limitations
            or check_error is not None
            or any(value == CoverageItemVerdict.UNRESOLVED.value for value in verdicts.values())
            else ActionExecutionStatus.SUCCEEDED
        )
        entry = ActionTraceEntry(
            step_index=step,
            action_id=current_action_id,
            question_id=target.question_id,
            action_name="INSPECT_COVERAGE_BATCH",
            target_ids=[item.inventory_id for item in batch_items],
            execution_status=execution_status,
            observation_ids=[item.observation_id for item in stored],
            metadata={
                "scope_text": plan.requirement.scope_text,
                "item_type": plan.requirement.item_type,
                "source_type": plan.requirement.source_type.value,
                "batch_size": len(batch_items),
                "verdicts": verdicts,
                "coverage_status": updated_plan.execution.status.value,
                "assessed_count": len(updated_plan.execution.assessments),
                "inventory_count": len(updated_plan.inventory.items),
                "matched_count": updated_plan.execution.matched_count,
            },
        )
        next_trace = ActionTrace(
            reading_session_id=trace.reading_session_id,
            root_question_id=trace.root_question_id,
            entries=[*trace.entries, entry],
        )

        if check_error is not None:
            description = f"Semantic Coverage Checker failed: {check_error}"
            self._stop_reason = description
            self._diagnostics.append(
                EnvironmentDiagnostic(
                    code="coverage_semantic_check_failed",
                    description=description,
                    action_id=current_action_id,
                    question_id=target.question_id,
                    metadata={
                        "inventory_ids": [item.inventory_id for item in batch_items]
                    },
                )
            )
        elif updated_plan.execution.status == CoverageExecutionStatus.BLOCKED:
            description = (
                "Semantic coverage remains incomplete because one or more "
                "canonical inventory items could not be resolved."
            )
            self._stop_reason = description
            self._diagnostics.append(
                EnvironmentDiagnostic(
                    code="coverage_semantic_items_unresolved",
                    description=description,
                    action_id=current_action_id,
                    question_id=target.question_id,
                    metadata={
                        "unresolved_inventory_ids": (
                            updated_plan.execution.unresolved_inventory_ids
                        )
                    },
                )
            )
        elif updated_plan.execution.status == CoverageExecutionStatus.COMPLETE:
            next_memory = self._advance_state_only_checks(
                root_question=root_question,
                previous_memory=memory,
                memory=next_memory,
                observations=next_observations,
                triggering_action_id=current_action_id,
            )
        return next_observations, next_memory, next_trace, True

    def _block_unavailable_coverage(self, question_id: str) -> bool:
        """Block only unsafe or unsupported coverage paths."""

        plan = next(
            (item for item in self._coverage_plans if item.question_id == question_id),
            None,
        )
        if plan is None or plan.inventory.structural_count is not None:
            return False
        if plan.inventory.status == CoverageInventoryStatus.BLOCKED:
            code = "coverage_scope_execution_blocked"
            description = (
                "Coverage execution stopped because the requested scope is "
                "unresolved or requires a page-namespace review. Apply an "
                "explicit namespace override before resuming."
            )
        elif plan.requirement.operator != CoverageOperator.COUNT:
            code = "coverage_operator_not_implemented"
            description = (
                "This Coverage version executes semantic counts only; the "
                f"operator {plan.requirement.operator.value!r} remains unsupported."
            )
        elif plan.execution.status == CoverageExecutionStatus.BLOCKED:
            code = "coverage_semantic_items_unresolved"
            description = (
                "Semantic coverage remains incomplete because canonical items "
                "still have unresolved verdicts."
            )
        elif getattr(self.checker, "check_coverage", None) is None:
            code = "coverage_semantic_checker_unavailable"
            description = (
                "The configured Checker backend does not implement the semantic "
                "Coverage Checker contract."
            )
        else:
            return False
        self._stop_reason = description
        self._diagnostics.append(
            EnvironmentDiagnostic(
                code=code,
                description=description,
                question_id=question_id,
                metadata={
                    "scope_text": plan.requirement.scope_text,
                    "operator": plan.requirement.operator.value,
                    "item_type": plan.requirement.item_type,
                    "source_type": plan.requirement.source_type.value,
                    "scope_status": plan.scope_resolution.status.value,
                    "requires_review": plan.scope_resolution.requires_review,
                    "coverage_status": plan.execution.status.value,
                },
            )
        )
        return True

    @staticmethod
    def _coverage_local_problem(
        question_text: str, requirement: CoverageRequirement
    ) -> str:
        predicate = requirement.predicate or "the requested item type"
        return (
            f"Coverage inspection for: {question_text}\n"
            f"Inspect each supplied canonical {requirement.source_type.value} source "
            f"for {requirement.item_type!r}; criterion: {predicate}. Report only "
            "facts visible in the supplied source and any limitation."
        )

    def _coverage_metadata_input(
        self,
        *,
        inventory_item: Any,
        source_id: str,
        index: int,
    ) -> ReadInput:
        element = next(
            (item for item in self.document.elements if item.element_id == source_id),
            None,
        )
        page_id = (
            element.page_id
            if element is not None
            else inventory_item.page_ids[
                min(inventory_item.source_ids.index(source_id), len(inventory_item.page_ids) - 1)
            ]
        )
        return ReadInput(
            input_id=read_input_id(index),
            source_id=source_id,
            source_type=(
                ReadingSourceType.ELEMENT
                if element is not None
                else ReadingSourceType.PAGE
            ),
            representation=ReadRepresentation.STRUCTURAL_METADATA,
            document_id=self.document.document_id,
            page_id=page_id,
            element_id=element.element_id if element is not None else None,
        )

    def _append_coverage_provenance_inputs(
        self,
        *,
        read_inputs: list[ReadInput],
        plan: QuestionCoveragePlan,
    ) -> tuple[list[ReadInput], list[ObservationSourceRef]]:
        inputs = list(read_inputs)
        by_source = {item.source_id: item for item in inputs}
        for inventory_item in plan.inventory.items:
            for source_id in inventory_item.source_ids:
                if source_id not in by_source:
                    item = self._coverage_metadata_input(
                        inventory_item=inventory_item,
                        source_id=source_id,
                        index=len(inputs),
                    )
                    inputs.append(item)
                    by_source[source_id] = item
        pages_by_id = {item.page_id: item for item in self.document.pages}
        sources = [
            ObservationSourceRef(
                input_id=item.input_id,
                source_id=item.source_id,
                page_id=item.page_id,
                physical_page_number=pages_by_id[item.page_id].page_number,
                display_page_label=pages_by_id[item.page_id].display_page_label,
                element_id=item.element_id,
            )
            for item in inputs
        ]
        return inputs, sources

    @staticmethod
    def _coverage_completion_statement(plan: QuestionCoveragePlan) -> str:
        count = plan.execution.matched_count
        values = plan.execution.matched_values
        predicate = plan.requirement.predicate or "no additional predicate"
        value_suffix = (
            " Matched values: " + "; ".join(values) + "." if values else ""
        )
        return (
            f"Exhaustively inspected all {len(plan.inventory.items)} canonical "
            f"{plan.requirement.source_type.value} source(s) in resolved scope "
            f"{plan.requirement.scope_text!r}. The exact semantic count of "
            f"{plan.requirement.item_type!r} under criterion {predicate!r} is "
            f"{count}. Every inventory item has a matched or not_matched verdict; "
            f"zero items remain unresolved.{value_suffix}"
        )

    def _replace_coverage_plan(self, updated: QuestionCoveragePlan) -> None:
        self._coverage_plans = [
            updated if item.question_id == updated.question_id else item
            for item in self._coverage_plans
        ]

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
        if isinstance(action, ControllerReadPageContextAction):
            target_page = self._page_at_offset(action.base_page_id, action.offset)
            if target_page is None:
                return observations, memory, self._append_failed_action(
                    trace=trace,
                    memory=memory,
                    action_name=action.action.value,
                    target_ids=[action.base_page_id],
                    description="The requested page-context offset does not exist.",
                    metadata={
                        "base_page_id": action.base_page_id,
                        "offset": action.offset,
                    },
                )
            source_ids = [target_page.page_id]
            if action.offset == 0:
                focused_source_id = self._latest_opened_element_on_page(
                    trace, action.base_page_id
                )
                if focused_source_id is not None:
                    source_ids.append(focused_source_id)
            return self._execute_read(
                source_ids=source_ids,
                local_problem=action.local_problem,
                action_name=action.action.value,
                root_question=root_question,
                memory=memory,
                observations=observations,
                trace=trace,
                metadata={
                    "base_page_id": action.base_page_id,
                    "offset": action.offset,
                    "focused_source_id": (
                        source_ids[1] if len(source_ids) == 2 else None
                    ),
                },
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
        inputs_by_id = {item.input_id: item for item in resolved}
        pages_by_id = {item.page_id: item for item in self.document.pages}
        for draft in reader_output.observations:
            unknown = {item.input_id for item in draft.sources}.difference(input_ids)
            if unknown:
                raise ValueError(
                    "Reader Observation references unavailable inputs: "
                    + ", ".join(sorted(unknown))
                )

        # I1/I2 are deliberately local to one Reader call.  Expand them to
        # authoritative, stable document identities before persistence so a
        # later Checker/Recall pass cannot mistake I1 from two reads for the
        # same source.  Reader-authored stable fields, if any, are ignored.
        expanded_sources = []
        for draft in reader_output.observations:
            # A model may cite the same call-local input more than once while
            # describing several facts from one table/image.  StoredObservation
            # deliberately permits only one grounding per input_id, so collapse
            # those duplicate citations at the deterministic Environment
            # boundary instead of rejecting an otherwise valid READ result.
            seen_input_ids: set[str] = set()
            sources: list[ObservationSourceRef] = []
            for source in draft.sources:
                if source.input_id in seen_input_ids:
                    continue
                seen_input_ids.add(source.input_id)
                read_input = inputs_by_id[source.input_id]
                page = pages_by_id[read_input.page_id]
                sources.append(
                    source.model_copy(
                        update={
                            "source_id": read_input.source_id,
                            "page_id": read_input.page_id,
                            "physical_page_number": page.page_number,
                            "display_page_label": page.display_page_label,
                            "element_id": read_input.element_id,
                        }
                    )
                )
            expanded_sources.append(sources)
        stored = [
            StoredObservation(
                observation_id=make_observation_id(current_action_id, index),
                action_id=current_action_id,
                text=draft.text,
                sources=expanded_sources[index],
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
        check_succeeded = False
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
                check_succeeded = True
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

            if check_succeeded:
                next_memory = self._advance_state_only_checks(
                    root_question=root_question,
                    previous_memory=memory,
                    memory=next_memory,
                    observations=next_observations,
                    triggering_action_id=current_action_id,
                )

        return next_observations, next_memory, next_trace

    def _advance_state_only_checks(
        self,
        *,
        root_question: RootQuestion,
        previous_memory: EvidenceMemory,
        memory: EvidenceMemory,
        observations: ObservationStore,
        triggering_action_id: str,
    ) -> EvidenceMemory:
        """Recheck a newly selected target from Evidence, then recalled facts."""

        previous_target = previous_memory.current_target
        previous_target_id = (
            previous_target.question_id if previous_target is not None else None
        )
        current = memory
        for _ in range(len(current.questions) + 1):
            target = current.current_target
            if (
                target is None
                or target.question_id == previous_target_id
                or current.root_status == EvidenceStatus.READY
            ):
                break

            if target.question_id == root_question.question_id:
                if not self._is_root_finalization_state(
                    memory=current,
                    root_question=root_question,
                ):
                    break
                updated = self._finalize_root_from_evidence(
                    root_question=root_question,
                    memory=current,
                    triggering_action_id=triggering_action_id,
                )
            else:
                updated = current
                if current.evidence:
                    updated = self._recheck_target_from_evidence(
                        root_question=root_question,
                        memory=current,
                        triggering_action_id=triggering_action_id,
                    )
                updated_target = updated.current_target
                if (
                    updated_target is not None
                    and updated_target.question_id == target.question_id
                    and self.config.observation_recall_limit
                ):
                    recalled = self._select_recalled_observations(
                        memory=updated,
                        observations=observations,
                    )
                    if recalled:
                        updated = self._recheck_target_from_observations(
                            root_question=root_question,
                            memory=updated,
                            recalled_observations=recalled,
                            triggering_action_id=triggering_action_id,
                        )

            previous_target_id = target.question_id
            if updated == current:
                break
            current = updated

        return current

    def _select_recalled_observations(
        self,
        *,
        memory: EvidenceMemory,
        observations: ObservationStore,
    ) -> list[StoredObservation]:
        """Return a small, deduplicated lexical view of unaccepted history.

        Recall is deliberately target-switch-only and does not expose the full
        ObservationStore to the Controller.  Stable source identity and the
        normalized claim text suppress repeated reads of the same content.
        """

        target = memory.current_target
        if target is None or target.question_id == memory.root_question_id:
            return []
        target_text = next(
            item.text for item in memory.questions
            if item.question_id == target.question_id
        )
        query_terms = self._recall_terms(
            f"{target_text} {target.gap_description}"
        )
        if not query_terms:
            return []

        accepted_observation_ids = {
            observation_id
            for item in memory.evidence
            for observation_id in item.observation_ids
        }
        records_by_action = {
            record.action_id: record for record in observations.read_records
        }
        ranked: list[tuple[int, int, str, StoredObservation]] = []
        for index, observation in enumerate(observations.observations):
            if observation.observation_id in accepted_observation_ids:
                continue
            record = records_by_action.get(observation.action_id)
            if record is None or record.subquestion_id == target.question_id:
                continue
            observation_terms = self._recall_terms(observation.text)
            overlap = query_terms.intersection(observation_terms)
            if not overlap:
                continue
            numeric_overlap = sum(
                token[0].isdigit() for token in overlap if token
            )
            score = len(overlap) * 4 + numeric_overlap * 3
            ranked.append((score, -index, observation.observation_id, observation))

        selected: list[StoredObservation] = []
        seen_claims: set[tuple[str, tuple[str, ...]]] = set()
        for _score, _index, _observation_id, observation in sorted(
            ranked,
            key=lambda item: (-item[0], -item[1], item[2]),
        ):
            source_ids = tuple(
                sorted(
                    source.source_id or source.input_id
                    for source in observation.sources
                )
            )
            key = (" ".join(observation.text.split()).casefold(), source_ids)
            if key in seen_claims:
                continue
            seen_claims.add(key)
            selected.append(observation)
            if len(selected) >= self.config.observation_recall_limit:
                break
        return selected

    @staticmethod
    def _recall_terms(value: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?%?", value.casefold())
            if token not in _RECALL_STOPWORDS and len(token) > 1
        }

    @staticmethod
    def _is_root_finalization_state(
        *,
        memory: EvidenceMemory,
        root_question: RootQuestion,
    ) -> bool:
        target = memory.current_target
        return (
            memory.root_status == EvidenceStatus.INCOMPLETE
            and target is not None
            and target.question_id == root_question.question_id
            and target.gap_description == root_question.text
            and bool(memory.questions)
            and all(
                item.status == QuestionStatus.SATISFIED
                for item in memory.questions
            )
        )

    def _recheck_target_from_evidence(
        self,
        *,
        root_question: RootQuestion,
        memory: EvidenceMemory,
        triggering_action_id: str,
    ) -> EvidenceMemory:
        """Judge one newly selected SubQuestion without fabricating another read."""

        target = memory.current_target
        assert target is not None
        recheck_action_id = (
            f"{triggering_action_id}:target-recheck:{target.question_id}"
        )
        checker_input = EvidenceCheckInput(
            action_id=recheck_action_id,
            root_question=root_question,
            evidence_memory=memory,
            observations=[],
            limitations=[],
        )
        try:
            check_result = self.checker.check(checker_input)
            return apply_evidence_check_result(checker_input, check_result)
        except Exception as exc:
            self._diagnostics.append(
                EnvironmentDiagnostic(
                    code="target_recheck_rejected",
                    description=str(exc),
                    action_id=recheck_action_id,
                    question_id=target.question_id,
                )
            )
            return memory

    def _recheck_target_from_observations(
        self,
        *,
        root_question: RootQuestion,
        memory: EvidenceMemory,
        recalled_observations: list[StoredObservation],
        triggering_action_id: str,
    ) -> EvidenceMemory:
        """Reassess a bounded historical Observation view for a new target."""

        target = memory.current_target
        assert target is not None
        recall_action_id = (
            f"{triggering_action_id}:observation-recall:{target.question_id}"
        )
        checker_input = EvidenceCheckInput(
            action_id=recall_action_id,
            root_question=root_question,
            evidence_memory=memory,
            observations=[],
            recalled_observations=recalled_observations,
            limitations=[],
        )
        self._diagnostics.append(
            EnvironmentDiagnostic(
                code="observation_recall_selected",
                description=(
                    "Historical Observations were selected for a target-switch "
                    "Checker recheck."
                ),
                action_id=recall_action_id,
                question_id=target.question_id,
                metadata={
                    "observation_ids": [
                        item.observation_id for item in recalled_observations
                    ]
                },
            )
        )
        try:
            check_result = self.checker.check(checker_input)
            return apply_evidence_check_result(checker_input, check_result)
        except Exception as exc:
            self._diagnostics.append(
                EnvironmentDiagnostic(
                    code="observation_recall_rejected",
                    description=str(exc),
                    action_id=recall_action_id,
                    question_id=target.question_id,
                )
            )
            return memory

    def _finalize_root_from_evidence(
        self,
        *,
        root_question: RootQuestion,
        memory: EvidenceMemory,
        triggering_action_id: str,
    ) -> EvidenceMemory:
        """Ask the Checker to judge Root sufficiency without inventing a read."""

        finalization_action_id = f"{triggering_action_id}:root-finalization"
        checker_input = EvidenceCheckInput(
            action_id=finalization_action_id,
            root_question=root_question,
            evidence_memory=memory,
            observations=[],
            limitations=[],
        )
        try:
            check_result = self.checker.check(checker_input)
            return apply_evidence_check_result(checker_input, check_result)
        except Exception as exc:
            self._diagnostics.append(
                EnvironmentDiagnostic(
                    code="root_finalization_rejected",
                    description=str(exc),
                    action_id=finalization_action_id,
                    question_id=root_question.question_id,
                )
            )
            return memory

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
                code=item.code,
                description=item.description,
                source_ids=[
                    (
                        inputs_by_id[input_id].element_id
                        or inputs_by_id[input_id].source_id
                    )
                    for input_id in item.input_ids
                    if input_id in inputs_by_id
                ],
                relevant_visible_content=item.relevant_visible_content,
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
            visual_path = self._asset_path(view.outer_visual_path)
            return ReadInput(
                input_id=input_id,
                source_id=view.table_view_id,
                source_type=ReadingSourceType.TABLE_VIEW,
                representation=ReadRepresentation.TABLE_VIEW,
                document_id=self.document.document_id,
                page_id=element.page_id,
                element_id=element.element_id,
                table_view_id=view.table_view_id,
                visual_asset_id=(
                    "visual:"
                    + stable_digest(
                        self.document.document_id,
                        element.element_id,
                        "table_visual",
                    )
                    if visual_path is not None
                    else None
                ),
                visual_asset_path=visual_path,
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

    def _page_at_offset(self, page_id: str, offset: int) -> Page | None:
        page = self.store.get_page(page_id)
        target_index = page.page_index + offset
        return next(
            (item for item in self.document.pages if item.page_index == target_index),
            None,
        )

    @staticmethod
    def _latest_opened_element_on_page(
        trace: ActionTrace, page_id: str
    ) -> str | None:
        for entry in reversed(trace.entries):
            handle = entry.primary_target
            if (
                handle is not None
                and handle.page_id == page_id
                and handle.element_id is not None
                and entry.execution_status
                in {ActionExecutionStatus.SUCCEEDED, ActionExecutionStatus.DEGRADED}
            ):
                return handle.element_id
        return None

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
            relation_search_units=self.search.search_units.units,
            readable_source_ids=[
                *[page.page_id for page in self.document.pages],
                *[element.element_id for element in self.document.elements],
            ],
            search_sessions=self._sessions.values(),
            visible_search_batch=visible_batch,
            remaining_action_budget=(
                self._action_limit - len(trace.entries)
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
