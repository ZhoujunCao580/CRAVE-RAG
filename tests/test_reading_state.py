import pytest
from pydantic import ValidationError

from softdoc.ids import action_id, evidence_id, observation_id, read_input_id
from softdoc.models import Relation, RelationEvidence, RelationSource, RelationStatus, RelationType
from softdoc.reading_state import (
    ActionOutcome, ActionTrace, ActionTraceEntry, CurrentTarget,
    EvidenceAddition, EvidenceCheckInput, EvidenceCheckResult, EvidenceItem,
    EvidenceMemory, EvidenceRemoval, EvidenceReplacement, EvidenceStatus,
    EvidenceUpdates, ExplorationSourceHandle, ExplorationStateBuilder,
    ObservationAssessment, ObservationLimitation, ObservationSourceRef,
    ObservationStore, QuestionState, QuestionStatus, ReadInput, ReadRecord,
    ReaderKind, ReadingSourceType, ReadRepresentation, RootQuestion,
    StoredObservation, apply_evidence_check_result, initialize_evidence_memory,
    register_deferred_question, select_next_runnable_question,
)


def _input(input_id: str = "I1", source_id: str = "element:figure:1") -> ReadInput:
    return ReadInput(
        input_id=input_id, source_id=source_id,
        source_type=ReadingSourceType.ELEMENT,
        representation=ReadRepresentation.ELEMENT_VISUAL,
        document_id="doc:1", page_id="page:1", element_id=source_id,
        visual_asset_id=f"visual:{input_id}",
        visual_asset_path=f"assets/{input_id}.png",
    )


def _observation(
    observation_id_value: str = "obs:1", *,
    action_id_value: str = "action:read", input_id: str = "I1",
    text: str = "Revenue in 2023 was 12 million.",
) -> StoredObservation:
    return StoredObservation(
        observation_id=observation_id_value, action_id=action_id_value, text=text,
        sources=[ObservationSourceRef(input_id=input_id)],
    )


def _store(*, subquestion_id: str | None = "Q2") -> ObservationStore:
    return ObservationStore(
        reading_session_id="reading:1", root_question_id="root:1",
        read_records=[ReadRecord(
            action_id="action:read", reader_kind=ReaderKind.VISUAL,
            document_id="doc:1", subquestion_id=subquestion_id,
            local_problem="What was the revenue in 2023?", inputs=[_input()],
            observation_ids=["obs:1"],
        )],
        observations=[_observation()],
    )


def _memory() -> EvidenceMemory:
    return EvidenceMemory(
        reading_session_id="reading:1", root_question_id="root:1",
        questions=[
            QuestionState(question_id="Q1", text="What was the revenue in 2022?",
                          status=QuestionStatus.SATISFIED),
            QuestionState(question_id="Q2", text="What was the revenue in 2023?"),
        ],
        evidence=[EvidenceItem(
            evidence_id="evidence:old", statement="Revenue in 2022 was 10 million.",
            observation_ids=["obs:old"], supports_question_ids=["Q1"],
        )],
        current_target=CurrentTarget(
            question_id="Q2", gap_description="The 2023 revenue is unknown."
        ),
    )


def _checker_input(memory: EvidenceMemory | None = None) -> EvidenceCheckInput:
    return EvidenceCheckInput(
        action_id="action:read",
        root_question=RootQuestion(
            question_id="root:1", text="How did revenue change from 2022 to 2023?"
        ),
        evidence_memory=memory or _memory(), observations=[_observation()],
    )


def _assessment(*, used: bool) -> ObservationAssessment:
    return ObservationAssessment(
        observation_id="obs:1", used_for_evidence=used,
        assessment=("The observation supplies the requested value."
                    if used else "The observation does not resolve the current target."),
    )


def test_runtime_ids_are_stable_and_reject_negative_indexes():
    current_action = action_id("reading:1", 2)
    assert read_input_id(0) == "I1"
    assert observation_id(current_action, 0) == observation_id(current_action, 0)
    assert evidence_id(current_action, 0) == evidence_id(current_action, 0)
    for function, arguments in [
        (action_id, ("reading:1", -1)), (read_input_id, (-1,)),
        (observation_id, ("action:1", -1)), (evidence_id, ("action:1", -1)),
    ]:
        with pytest.raises(ValueError, match="non-negative"):
            function(*arguments)


def test_observation_store_round_trip_and_joint_grounding():
    store = ObservationStore(
        reading_session_id="reading:1", root_question_id="root:1",
        read_records=[ReadRecord(
            action_id="action:joint", reader_kind=ReaderKind.VISUAL,
            document_id="doc:1", subquestion_id="Q1",
            local_problem="Do the charts show the same trend?",
            inputs=[_input("I1"), _input("I2", "element:figure:2")],
            observation_ids=["obs:joint"],
        )],
        observations=[StoredObservation(
            observation_id="obs:joint", action_id="action:joint",
            text="Both charts rise from left to right.",
            sources=[ObservationSourceRef(input_id="I1"),
                     ObservationSourceRef(input_id="I2")],
        )],
    )
    assert ObservationStore.model_validate_json(store.model_dump_json()) == store
    assert [source.input_id for source in store.observations[0].sources] == ["I1", "I2"]


def test_limitation_and_observation_must_reference_same_action_input():
    with pytest.raises(ValidationError, match="unknown read input"):
        ReadRecord(
            action_id="action:read", reader_kind=ReaderKind.VISUAL,
            document_id="doc:1", local_problem="Read labels.", inputs=[_input()],
            limitations=[ObservationLimitation(
                description="The label is too small.", input_ids=["I2"]
            )],
        )
    payload = _store().model_dump(mode="json")
    payload["observations"][0]["sources"][0]["input_id"] = "I9"
    with pytest.raises(ValidationError, match="unknown read input"):
        ObservationStore.model_validate(payload)


def test_initialize_memory_selects_first_dependency_ready_question():
    memory = initialize_evidence_memory(
        reading_session_id="reading:1", root_question_id="root:1",
        root_question_text="Compare A and B.",
        questions=[
            QuestionState(question_id="Q1", text="Find A."),
            QuestionState(question_id="Q2", text="Find B.", depends_on=["Q1"]),
        ],
    )
    assert memory.current_target == CurrentTarget(
        question_id="Q1", gap_description="Find A."
    )
    assert select_next_runnable_question(memory.questions).question_id == "Q1"


def test_memory_rejects_unknown_cycle_and_blocked_current_target():
    with pytest.raises(ValidationError, match="unknown dependencies"):
        EvidenceMemory(
            reading_session_id="reading:1", root_question_id="root:1",
            questions=[QuestionState(
                question_id="Q1", text="Find A.", depends_on=["Q9"]
            )],
            current_target=CurrentTarget(question_id="Q1", gap_description="Find A."),
        )
    with pytest.raises(ValidationError, match="acyclic"):
        EvidenceMemory(
            reading_session_id="reading:1", root_question_id="root:1",
            questions=[
                QuestionState(question_id="Q1", text="A", depends_on=["Q2"]),
                QuestionState(question_id="Q2", text="B", depends_on=["Q1"]),
            ], current_target=CurrentTarget(question_id="root:1", gap_description="Root"),
        )
    with pytest.raises(ValidationError, match="unsatisfied dependencies"):
        EvidenceMemory(
            reading_session_id="reading:1", root_question_id="root:1",
            questions=[
                QuestionState(question_id="Q1", text="Find A."),
                QuestionState(question_id="Q2", text="Find B.", depends_on=["Q1"]),
            ], current_target=CurrentTarget(question_id="Q2", gap_description="Find B."),
        )


def test_checker_incomplete_result_keeps_same_target_and_refines_gap():
    result = EvidenceCheckResult(
        action_id="action:read", observation_assessments=[_assessment(used=False)],
        current_target_status=QuestionStatus.INCOMPLETE,
        root_status=EvidenceStatus.INCOMPLETE,
        remaining_gap_description="The 2023 revenue still needs a reliable value.",
    )
    updated = apply_evidence_check_result(_checker_input(), result)
    assert updated.current_target == CurrentTarget(
        question_id="Q2",
        gap_description="The 2023 revenue still needs a reliable value.",
    )
    assert updated.evidence == _memory().evidence


def test_checker_satisfied_result_adds_evidence_and_can_finish_root():
    result = EvidenceCheckResult(
        action_id="action:read", observation_assessments=[_assessment(used=True)],
        evidence_updates=EvidenceUpdates(add=[EvidenceAddition(
            statement="Revenue in 2023 was 12 million.", observation_ids=["obs:1"],
            supports_question_ids=["Q2"],
        )]),
        current_target_status=QuestionStatus.SATISFIED,
        root_status=EvidenceStatus.READY,
    )
    original = _memory().model_copy(deep=True)
    checker_input = _checker_input()
    updated = apply_evidence_check_result(checker_input, result)
    assert checker_input.evidence_memory == original
    assert updated.root_status == EvidenceStatus.READY
    assert updated.current_target is None
    assert [question.status for question in updated.questions] == [
        QuestionStatus.SATISFIED, QuestionStatus.SATISFIED,
    ]


def test_checker_cannot_silently_update_multiple_questions_in_one_round():
    result = EvidenceCheckResult(
        action_id="action:read", observation_assessments=[_assessment(used=True)],
        evidence_updates=EvidenceUpdates(add=[EvidenceAddition(
            statement="The read appears relevant to both years.",
            observation_ids=["obs:1"], supports_question_ids=["Q1", "Q2"],
        )]), current_target_status=QuestionStatus.SATISFIED,
        root_status=EvidenceStatus.READY,
    )
    with pytest.raises(ValueError, match="only the current target Q2"):
        apply_evidence_check_result(_checker_input(), result)


def test_program_not_checker_selects_next_dependency_ready_question():
    memory = EvidenceMemory(
        reading_session_id="reading:1", root_question_id="root:1",
        questions=[
            QuestionState(question_id="Q1", text="Find A."),
            QuestionState(question_id="Q2", text="Find B.", depends_on=["Q1"]),
        ], current_target=CurrentTarget(question_id="Q1", gap_description="Find A."),
    )
    result = EvidenceCheckResult(
        action_id="action:read", observation_assessments=[_assessment(used=True)],
        evidence_updates=EvidenceUpdates(add=[EvidenceAddition(
            statement="A is 72%.", observation_ids=["obs:1"],
            supports_question_ids=["Q1"],
        )]), current_target_status=QuestionStatus.SATISFIED,
        root_status=EvidenceStatus.INCOMPLETE,
    )
    updated = apply_evidence_check_result(_checker_input(memory), result)
    assert updated.current_target == CurrentTarget(
        question_id="Q2", gap_description="Find B."
    )


def test_exhausted_plan_returns_to_root_for_deferred_planning_or_direct_reading():
    result = EvidenceCheckResult(
        action_id="action:read", observation_assessments=[_assessment(used=True)],
        evidence_updates=EvidenceUpdates(add=[EvidenceAddition(
            statement="Revenue in 2023 was 12 million.", observation_ids=["obs:1"],
            supports_question_ids=["Q2"],
        )]), current_target_status=QuestionStatus.SATISFIED,
        root_status=EvidenceStatus.INCOMPLETE,
        remaining_gap_description="The cause of the change is still unknown.",
    )
    updated = apply_evidence_check_result(_checker_input(), result)
    assert updated.current_target == CurrentTarget(
        question_id="root:1", gap_description="The cause of the change is still unknown."
    )


def test_deferred_question_is_registered_only_after_plan_exhaustion():
    exhausted = EvidenceMemory(
        reading_session_id="reading:1", root_question_id="root:1",
        questions=[QuestionState(
            question_id="Q1", text="Find values.", status=QuestionStatus.SATISFIED
        )],
        current_target=CurrentTarget(
            question_id="root:1", gap_description="The explanation is missing."
        ),
    )
    updated = register_deferred_question(
        exhausted, question_id="Q2", text="What explains the increase?",
        depends_on=["Q1"],
    )
    assert updated.current_target == CurrentTarget(
        question_id="Q2", gap_description="What explains the increase?"
    )
    with pytest.raises(ValueError, match="Root is the current target"):
        register_deferred_question(_memory(), question_id="Q3", text="Why?")


def test_checker_delta_replace_remove_and_bad_references_are_atomic():
    memory = _memory().model_copy(update={"evidence": [
        *_memory().evidence,
        EvidenceItem(evidence_id="evidence:bad", statement="An unreliable value.",
                     observation_ids=["obs:bad"], supports_question_ids=["Q1"]),
    ]})
    result = EvidenceCheckResult(
        action_id="action:read", observation_assessments=[_assessment(used=True)],
        evidence_updates=EvidenceUpdates(
            replace=[EvidenceReplacement(
                evidence_id="evidence:old", statement="Revenue in 2023 was 12 million.",
                observation_ids=["obs:1"], supports_question_ids=["Q2"],
            )],
            remove=[EvidenceRemoval(evidence_id="evidence:bad", reason="Contradicted")],
        ), current_target_status=QuestionStatus.INCOMPLETE,
        root_status=EvidenceStatus.INCOMPLETE,
        remaining_gap_description="The 2023 revenue is unknown.",
    )
    updated = apply_evidence_check_result(_checker_input(memory), result)
    assert [item.evidence_id for item in updated.evidence] == ["evidence:old"]
    assert len(memory.evidence) == 2
    bad = result.model_copy(update={
        "evidence_updates": EvidenceUpdates(remove=[
            EvidenceRemoval(evidence_id="missing", reason="Bad ID")
        ]), "observation_assessments": [_assessment(used=False)],
    })
    with pytest.raises(ValueError, match="missing Evidence"):
        apply_evidence_check_result(_checker_input(memory), bad)
    assert len(memory.evidence) == 2


def _relation(relation_id: str, status: RelationStatus, relation_type: RelationType,
              *, source_id: str = "element:source") -> Relation:
    return Relation(
        relation_id=relation_id, source_id=source_id, target_id="element:target",
        relation_type=relation_type, confidence=0.8, status=status,
        created_by=RelationSource.DETERMINISTIC_RULE,
        evidence=[RelationEvidence(rule="test_rule", description="test")],
    )


def test_exploration_state_is_derived_and_exposes_only_local_relations():
    focus = ExplorationSourceHandle(
        source_id="element:source", source_type=ReadingSourceType.ELEMENT,
        document_id="doc:1", page_id="page:1", element_id="element:source",
    )
    trace = ActionTrace(
        reading_session_id="reading:1", root_question_id="root:1", entries=[
            ActionTraceEntry(step_index=0, action_id="action:search", question_id="Q2",
                             action_name="SEARCH", query="2023 revenue",
                             outcome=ActionOutcome.SUCCEEDED),
            ActionTraceEntry(step_index=1, action_id="action:read", question_id="Q2",
                             action_name="READ_ELEMENT", target_ids=["element:source"],
                             primary_target=focus, outcome=ActionOutcome.SUCCEEDED,
                             observation_ids=["obs:1"],
                             result_summary="The observation narrowed the gap."),
        ],
    )
    state = ExplorationStateBuilder().build(
        observation_store=_store(), action_trace=trace,
        available_relations=[
            _relation("relation:confirmed", RelationStatus.CONFIRMED, RelationType.CAPTION_OF),
            _relation("relation:candidate", RelationStatus.CANDIDATE, RelationType.CONTINUED_ON),
            _relation("relation:unrelated", RelationStatus.CONFIRMED,
                      RelationType.CAPTION_OF, source_id="element:elsewhere"),
        ],
    )
    assert state.current_focus == focus
    assert state.attempted_source_ids == ["element:figure:1"]
    assert state.attempted_search_queries == ["2023 revenue"]
    assert [item.relation_id for item in state.confirmed_relation_handles] == ["relation:confirmed"]
    assert [item.relation_id for item in state.candidate_navigation_hints] == ["relation:candidate"]
    assert state.recent_actions[-1].question_id == "Q2"


def test_action_trace_requires_contiguous_steps():
    with pytest.raises(ValidationError, match="contiguous"):
        ActionTrace(
            reading_session_id="reading:1", root_question_id="root:1",
            entries=[ActionTraceEntry(
                step_index=1, action_id="action:1", question_id="root:1",
                action_name="SEARCH", outcome=ActionOutcome.SUCCEEDED,
            )],
        )
