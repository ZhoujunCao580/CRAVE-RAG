import pytest

from softdoc.reading_state import (
    ActionOutcome,
    ActionTrace,
    ActionTraceEntry,
    EvidenceItem,
    EvidenceMemory,
    EvidenceStatus,
    ObservationSourceRef,
    ObservationStore,
    QuestionState,
    QuestionStatus,
    ReadInput,
    ReadRecord,
    ReaderKind,
    ReadingSourceType,
    ReadRepresentation,
    StoredObservation,
)
from softdoc.reading_state_validation import (
    ReadingStateReferenceError,
    ReadingStateReferenceValidator,
)


def _valid_bundle() -> tuple[ObservationStore, EvidenceMemory, ActionTrace]:
    store = ObservationStore(
        reading_session_id="reading:1",
        root_question_id="root:1",
        read_records=[
            ReadRecord(
                action_id="action:read",
                reader_kind=ReaderKind.VISUAL,
                document_id="doc:1",
                subquestion_id="Q1",
                local_problem="Which method has the tallest bar?",
                inputs=[
                    ReadInput(
                        input_id="I1",
                        source_id="element:figure:1",
                        source_type=ReadingSourceType.ELEMENT,
                        representation=ReadRepresentation.ELEMENT_VISUAL,
                        document_id="doc:1",
                        page_id="page:1",
                        element_id="element:figure:1",
                        visual_asset_id="visual:1",
                        visual_asset_path="assets/figure_1.png",
                    )
                ],
                observation_ids=["obs:1"],
            )
        ],
        observations=[
            StoredObservation(
                observation_id="obs:1",
                action_id="action:read",
                text="Method B has the tallest bar.",
                sources=[ObservationSourceRef(input_id="I1")],
            )
        ],
    )
    memory = EvidenceMemory(
        reading_session_id="reading:1",
        root_question_id="root:1",
        root_status=EvidenceStatus.READY,
        questions=[
            QuestionState(
                question_id="Q1",
                text="Which method has the tallest bar?",
                status=QuestionStatus.SATISFIED,
            )
        ],
        evidence=[
            EvidenceItem(
                evidence_id="evidence:1",
                statement="Method B has the tallest bar.",
                observation_ids=["obs:1"],
                supports_question_ids=["Q1"],
            )
        ],
    )
    trace = ActionTrace(
        reading_session_id="reading:1",
        root_question_id="root:1",
        entries=[
            ActionTraceEntry(
                step_index=0,
                action_id="action:read",
                question_id="Q1",
                action_name="READ_ELEMENT",
                target_ids=["element:figure:1"],
                outcome=ActionOutcome.SUCCEEDED,
                observation_ids=["obs:1"],
            )
        ],
    )
    return store, memory, trace


def test_cross_store_reference_validator_accepts_consistent_run():
    store, memory, trace = _valid_bundle()
    assert ReadingStateReferenceValidator().validate(
        observation_store=store, evidence_memory=memory, action_trace=trace
    ) == []


def test_validator_reports_dangling_evidence_observation():
    store, memory, trace = _valid_bundle()
    memory = memory.model_copy(
        update={
            "evidence": [
                EvidenceItem(
                    evidence_id="evidence:missing",
                    statement="Unsupported statement.",
                    observation_ids=["obs:missing"],
                    supports_question_ids=["Q1"],
                )
            ]
        }
    )
    errors = ReadingStateReferenceValidator().validate(
        observation_store=store, evidence_memory=memory, action_trace=trace
    )
    assert errors == [
        "Evidence evidence:missing references missing Observation obs:missing"
    ]


def test_validator_checks_action_ownership_and_question_identity():
    store, memory, _ = _valid_bundle()
    wrong_action = ActionTrace(
        reading_session_id="reading:1",
        root_question_id="root:1",
        entries=[
            ActionTraceEntry(
                step_index=0,
                action_id="action:other",
                question_id="Q1",
                action_name="READ_ELEMENT",
                outcome=ActionOutcome.SUCCEEDED,
                observation_ids=["obs:1"],
            )
        ],
    )
    errors = ReadingStateReferenceValidator().validate(
        observation_store=store, evidence_memory=memory, action_trace=wrong_action
    )
    assert any("references missing Action action:read" in item for item in errors)
    assert any("belongs to Action action:read" in item for item in errors)

    _, _, trace = _valid_bundle()
    trace = trace.model_copy(
        update={
            "entries": [trace.entries[0].model_copy(update={"question_id": "Q-other"})]
        }
    )
    errors = ReadingStateReferenceValidator().validate(
        observation_store=store, evidence_memory=memory, action_trace=trace
    )
    assert any("does not match ReadRecord question Q1" in item for item in errors)
    assert any("references missing Question Q-other" in item for item in errors)


def test_validator_can_raise_aggregated_error():
    store, memory, trace = _valid_bundle()
    memory = memory.model_copy(
        update={
            "evidence": [
                EvidenceItem(
                    evidence_id="evidence:missing",
                    statement="Unsupported statement.",
                    observation_ids=["obs:missing"],
                    supports_question_ids=["Q1"],
                )
            ]
        }
    )
    with pytest.raises(ReadingStateReferenceError, match="obs:missing"):
        ReadingStateReferenceValidator().validate(
            observation_store=store,
            evidence_memory=memory,
            action_trace=trace,
            raise_on_error=True,
        )
