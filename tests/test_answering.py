import pytest
from pydantic import ValidationError

from softdoc.answering import (
    ANSWERER_PROMPT_VERSION,
    ANSWERER_SYSTEM_PROMPT,
    AnswerInput,
    AnswerInputBuilder,
    AnswerResult,
    answerer_user_prompt,
    validate_answer_result,
)
from softdoc.reading_state import (
    EvidenceItem,
    EvidenceMemory,
    EvidenceStatus,
    QuestionState,
    QuestionStatus,
    RootQuestion,
)


def _ready_memory() -> EvidenceMemory:
    return EvidenceMemory(
        reading_session_id="reading:1",
        root_question_id="root:1",
        root_status=EvidenceStatus.READY,
        questions=[
            QuestionState(
                question_id="Q1",
                text="What was revenue in 2022?",
                status=QuestionStatus.SATISFIED,
            ),
            QuestionState(
                question_id="Q2",
                text="What was revenue in 2023?",
                status=QuestionStatus.SATISFIED,
            ),
        ],
        evidence=[
            EvidenceItem(
                evidence_id="evidence:1",
                statement="Revenue in 2022 was $10 million.",
                observation_ids=["observation:1"],
                supports_question_ids=["Q1"],
            ),
            EvidenceItem(
                evidence_id="evidence:2",
                statement="Revenue in 2023 was $12 million.",
                observation_ids=["observation:2"],
                supports_question_ids=["Q2"],
            ),
        ],
        current_target=None,
    )


def _answer_input() -> AnswerInput:
    return AnswerInputBuilder().build(
        root_question=RootQuestion(
            question_id="root:1",
            text="How did revenue change from 2022 to 2023?",
        ),
        evidence_memory=_ready_memory(),
    )


def test_builder_materializes_only_question_graph_and_evidence_statements():
    answer_input = _answer_input()

    assert answer_input.question_graph[0].question_id == "Q1"
    assert answer_input.evidence[0].statement == "Revenue in 2022 was $10 million."
    dumped = answer_input.model_dump()
    assert "observation_ids" not in dumped["evidence"][0]
    assert "current_target" not in dumped
    assert "root_status" not in dumped


def test_builder_accepts_root_direct_evidence_with_an_empty_question_graph():
    root = RootQuestion(question_id="root:direct", text="What is Figure 3 titled?")
    memory = EvidenceMemory(
        reading_session_id="reading:direct",
        root_question_id=root.question_id,
        root_status=EvidenceStatus.READY,
        questions=[],
        evidence=[
            EvidenceItem(
                evidence_id="evidence:direct",
                statement="Figure 3 is titled System Architecture.",
                observation_ids=["observation:direct"],
                supports_question_ids=[root.question_id],
            )
        ],
        current_target=None,
    )

    answer_input = AnswerInputBuilder().build(
        root_question=root,
        evidence_memory=memory,
    )

    assert answer_input.question_graph == []
    assert answer_input.evidence[0].supports_question_ids == [root.question_id]


def test_builder_rejects_incomplete_memory():
    memory = _ready_memory().model_copy(
        update={
            "root_status": EvidenceStatus.INCOMPLETE,
            "current_target": {"question_id": "Q1", "gap_description": "Missing."},
        }
    )

    with pytest.raises(ValueError, match="only when Root Evidence is ready"):
        AnswerInputBuilder().build(
            root_question=RootQuestion(question_id="root:1", text="Question?"),
            evidence_memory=memory,
        )


def test_answer_input_rejects_unknown_question_support():
    payload = _answer_input().model_dump()
    payload["evidence"][0]["supports_question_ids"] = ["Q9"]

    with pytest.raises(ValidationError, match="supports unknown questions"):
        AnswerInput.model_validate(payload)


def test_answer_input_rejects_cyclic_question_graph():
    payload = _answer_input().model_dump()
    payload["question_graph"][0]["depends_on"] = ["Q2"]
    payload["question_graph"][1]["depends_on"] = ["Q1"]

    with pytest.raises(ValidationError, match="must form a DAG"):
        AnswerInput.model_validate(payload)


def test_answer_result_must_reference_available_evidence():
    result = AnswerResult(answer="It increased by 20%.", used_evidence_ids=["missing"])

    with pytest.raises(ValueError, match="unavailable Evidence"):
        validate_answer_result(_answer_input(), result)


def test_answer_result_accepts_multiple_calculation_inputs():
    result = AnswerResult(
        answer="Revenue increased by $2 million, or 20%.",
        used_evidence_ids=["evidence:1", "evidence:2"],
    )

    assert validate_answer_result(_answer_input(), result) is result


def test_answer_result_allows_evidence_free_not_answerable_fallback():
    result = AnswerResult(answer="Not answerable", used_evidence_ids=[])

    assert result.answer == "Not answerable"
    assert result.used_evidence_ids == []


def test_answer_result_rejects_evidence_free_substantive_answer():
    with pytest.raises(ValidationError, match="must reference at least one Evidence"):
        AnswerResult(answer="42", used_evidence_ids=[])


def test_answerer_prompt_freezes_minimal_boundary():
    assert ANSWERER_PROMPT_VERSION == "answerer-v0.8"
    assert "question_graph only describes" in ANSWERER_SYSTEM_PROMPT
    assert "Use only the supplied Evidence statements" in ANSWERER_SYSTEM_PROMPT
    assert "directly supports every part" in ANSWERER_SYSTEM_PROMPT
    assert "mutually consistent" not in ANSWERER_SYSTEM_PROMPT
    assert "missing or conflicting" in ANSWERER_SYSTEM_PROMPT
    assert "does not by itself establish its cause" in ANSWERER_SYSTEM_PROMPT
    assert "Do not create citations or source locations" in ANSWERER_SYSTEM_PROMPT
    assert "including inputs to any calculation" in ANSWERER_SYSTEM_PROMPT
    assert "every compared alternative needed to establish the result" in (
        ANSWERER_SYSTEM_PROMPT
    )
    assert "shortest self-contained final answer" in ANSWERER_SYSTEM_PROMPT
    assert 'return only "Yes" or "No"' in ANSWERER_SYSTEM_PROMPT
    assert "compact JSON-style" in ANSWERER_SYSTEM_PROMPT

    prompt = answerer_user_prompt(_answer_input())
    output_shape = prompt.split("Return exactly this JSON shape:", 1)[1]
    assert '"answer"' in output_shape
    assert '"used_evidence_ids"' in output_shape
    assert '"citations"' not in output_shape
