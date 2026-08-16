from __future__ import annotations

import json
from hashlib import sha256

import pytest
from pydantic import ValidationError

from softdoc.planning import (
    InitialPlanner,
    INITIAL_PLANNER_PROMPT_VERSION,
    OllamaPlannerBackend,
    OllamaPlannerConfig,
    PlannerBackendResponse,
    PlannerConfig,
    PlannerDraft,
    PlannerOutputError,
    build_initial_planner_prompt,
)


class MockPlannerBackend:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    @property
    def backend_name(self) -> str:
        return "mock"

    def generate(self, prompt: str) -> PlannerBackendResponse:
        self.prompts.append(prompt)
        return PlannerBackendResponse(
            content=json.dumps(self.payload, ensure_ascii=False),
            model="mock-planner",
            metadata={"fixture": True},
        )


class FakeOllamaTransport:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object], float]] = []

    def post_json(
        self,
        url: str,
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        self.calls.append((url, payload, timeout_seconds))
        return self.response


class SequencePlannerBackend:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = payloads
        self.prompts: list[str] = []

    @property
    def backend_name(self) -> str:
        return "sequence-mock"

    def generate(self, prompt: str) -> PlannerBackendResponse:
        self.prompts.append(prompt)
        payload = self.payloads.pop(0)
        return PlannerBackendResponse(
            content=json.dumps(payload),
            model="sequence-model",
        )


def _subquestion(
    subquestion_id: str,
    text: str,
    *,
    depends_on: list[str] | None = None,
) -> dict[str, object]:
    return {
        "subquestion_id": subquestion_id,
        "text": text,
        "depends_on": depends_on or [],
    }


def _plan(question: str, subquestions: list[dict[str, object]]) -> dict[str, object]:
    return {"original_question": question, "subquestions": subquestions}


def test_simple_fact_remains_one_subquestion() -> None:
    question = "What is the title of Figure 3?"
    backend = MockPlannerBackend(
        _plan(
            question,
            [_subquestion("Q1", question)],
        )
    )

    result = InitialPlanner(backend).create_plan(question)

    assert len(result.subquestions) == 1
    assert not hasattr(result.subquestions[0], "explicit_anchors")
    assert result.planner_trace.backend_name == "mock"
    assert result.planner_trace.model == "mock-planner"
    assert result.planner_trace.metadata == {
        "fixture": True,
        "validation_attempts": 1,
    }


def test_independent_facts_are_parallel_not_artificially_sequential() -> None:
    question = "What were the revenues in 2022 and 2023?"
    backend = MockPlannerBackend(
        _plan(
            question,
            [
                _subquestion("Q1", "What was the revenue in 2022?"),
                _subquestion("Q2", "What was the revenue in 2023?"),
            ],
        )
    )

    result = InitialPlanner(backend).create_plan(question)

    assert [item.depends_on for item in result.subquestions] == [[], []]


def test_comparison_can_depend_on_two_parallel_facts() -> None:
    question = "Compare the revenues in 2022 and 2023."
    backend = MockPlannerBackend(
        _plan(
            question,
            [
                _subquestion("Q1", "What was the revenue in 2022?"),
                _subquestion("Q2", "What was the revenue in 2023?"),
                _subquestion(
                    "Q3",
                    "How did revenue change between 2022 and 2023?",
                    depends_on=["Q1", "Q2"],
                ),
            ],
        )
    )

    result = InitialPlanner(backend).create_plan(question)

    assert result.subquestions[2].depends_on == ["Q1", "Q2"]


def test_model_facing_schema_excludes_explicit_anchors() -> None:
    subquestion_schema = PlannerDraft.model_json_schema()["$defs"][
        "PlannedSubQuestion"
    ]
    assert "explicit_anchors" not in subquestion_schema["properties"]


def test_removed_explicit_anchors_field_is_rejected() -> None:
    question = "What is shown in Figure 3?"
    payload = _subquestion("Q1", question)
    payload["explicit_anchors"] = ["Figure 3"]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PlannerDraft.model_validate(_plan(question, [payload]))


def test_unknown_dependency_is_rejected() -> None:
    with pytest.raises(ValidationError, match="existing IDs"):
        PlannerDraft.model_validate(
            _plan("Question", [_subquestion("Q1", "Need one fact?", depends_on=["Q9"])])
        )


def test_cyclic_dependencies_are_rejected() -> None:
    with pytest.raises(ValidationError, match="must form a DAG"):
        PlannerDraft.model_validate(
            _plan(
                "Question",
                [
                    _subquestion("Q1", "Need the first fact?", depends_on=["Q2"]),
                    _subquestion("Q2", "Need the second fact?", depends_on=["Q1"]),
                ],
            )
        )


@pytest.mark.parametrize(
    ("extra_field", "value"),
    [("next_action", "SEARCH"), ("answer", "The answer is 42")],
)
def test_tool_actions_and_early_answer_fields_are_rejected(
    extra_field: str,
    value: str,
) -> None:
    question = "What is the revenue?"
    payload = _plan(question, [_subquestion("Q1", question)])
    payload[extra_field] = value
    backend = MockPlannerBackend(payload)

    with pytest.raises(PlannerOutputError, match="invalid"):
        InitialPlanner(backend).create_plan(question)


def test_original_question_must_be_preserved_verbatim() -> None:
    backend = MockPlannerBackend(
        _plan("A rewritten question", [_subquestion("Q1", "What is required?")])
    )

    with pytest.raises(PlannerOutputError, match="verbatim"):
        InitialPlanner(backend).create_plan("The original question")


def test_configured_subquestion_limit_is_enforced() -> None:
    question = "Question"
    backend = MockPlannerBackend(
        _plan(
            question,
            [_subquestion(f"Q{index}", f"Need fact {index}?") for index in range(1, 4)],
        )
    )

    with pytest.raises(PlannerOutputError, match="SubQuestion limit"):
        InitialPlanner(backend, PlannerConfig(max_subquestions=2)).create_plan(question)


def test_configured_depth_counts_original_question_as_root() -> None:
    question = "Question"
    backend = MockPlannerBackend(
        _plan(
            question,
            [
                _subquestion("Q1", "Need fact one?"),
                _subquestion("Q2", "Need fact two?", depends_on=["Q1"]),
                _subquestion("Q3", "Need fact three?", depends_on=["Q2"]),
            ],
        )
    )

    with pytest.raises(PlannerOutputError, match="DAG depth"):
        InitialPlanner(backend, PlannerConfig(max_depth=3)).create_plan(question)


def test_removed_answer_requirements_field_is_rejected() -> None:
    payload = _subquestion("Q1", "What was the revenue in 2023?")
    payload["answer_requirements"] = ["the revenue amount for 2023"]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PlannerDraft.model_validate(
            _plan(
                "What was the revenue in 2023?",
                [payload],
            )
        )


def test_invalid_first_response_is_corrected_once_and_traced() -> None:
    question = "Compare the revenues in 2022 and 2023."
    backend = SequencePlannerBackend(
        [
            _plan(
                question,
                [
                    _subquestion("Q1", "What was the revenue in 2022?"),
                    _subquestion("Q1", "What was the revenue in 2023?"),
                ],
            ),
            _plan(
                question,
                [
                    _subquestion("Q1", "What was the revenue in 2022?"),
                    _subquestion("Q2", "What was the revenue in 2023?"),
                ],
            ),
        ]
    )

    plan = InitialPlanner(backend).create_plan(question)

    assert len(backend.prompts) == 2
    assert "previous response was rejected" in backend.prompts[1]
    assert plan.planner_trace.metadata["validation_attempts"] == 2
    assert plan.planner_trace.warnings[0].code == "planner_validation_retry"


def test_prompt_defines_smallest_evidence_plan_and_implicit_root() -> None:
    prompt = build_initial_planner_prompt(
        "Which company is higher and by how much in Figure 3?"
    )
    compact_prompt = " ".join(prompt.split())
    assert "smallest sufficient set" in prompt
    assert "original question is an implicit Root" in prompt
    assert "is not a SubQuestion" in prompt
    assert "is not an additional Agent action" in prompt
    assert "Every SubQuestion must request new evidence" in prompt
    assert "The Root performs that final operation" in prompt
    assert "same local piece of document evidence" in prompt
    assert "recombine facts already requested separately" in prompt
    assert "true dependency through an unknown entity" in prompt
    assert "is required to instantiate the later evidence need" in compact_prompt
    assert "condition, category, value, time period, or search phrase" in compact_prompt
    assert "searched independently from the original question alone" in compact_prompt
    assert "keep the complete original question as one SubQuestion" in prompt
    assert "does not mean planning failed" in prompt
    assert "explicit_anchors" not in prompt
    assert "percentage change" not in prompt
    assert "answer_requirements" not in prompt
    assert "BBA" not in prompt
    assert "500 MHz" not in prompt
    assert "Figure 6" not in prompt


def test_planner_v014_prompt_is_frozen() -> None:
    prompt = build_initial_planner_prompt("FROZEN PLANNER PROMPT SNAPSHOT")
    assert INITIAL_PLANNER_PROMPT_VERSION == "planner-v0.14"
    assert sha256(prompt.encode("utf-8")).hexdigest() == (
        "d4ff7c777b8a673d9f74efeef8018f9cd4fc56ef7c8b23eb4b0b321e06149e94"
    )


def test_ollama_backend_uses_local_chat_api_and_json_schema() -> None:
    question = "What is the revenue?"
    content = json.dumps(_plan(question, [_subquestion("Q1", question)]))
    transport = FakeOllamaTransport(
        {
            "model": "qwen-test",
            "message": {"role": "assistant", "content": content},
            "eval_count": 42,
        }
    )
    backend = OllamaPlannerBackend(
        OllamaPlannerConfig(
            base_url="http://localhost:11434/",
            model="qwen-test",
            timeout_seconds=12,
        ),
        transport,
    )

    response = backend.generate("planner prompt")

    assert response.content == content
    assert response.metadata == {"eval_count": 42}
    url, payload, timeout = transport.calls[0]
    assert url == "http://localhost:11434/api/chat"
    assert timeout == 12
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["format"] == PlannerDraft.model_json_schema()
    assert payload["messages"] == [{"role": "user", "content": "planner prompt"}]


def test_prompt_and_plan_serialization_are_stable() -> None:
    question = "图3中哪个方法表现最好？"
    payload = _plan(
        question,
        [_subquestion("Q1", question)],
    )
    backend = MockPlannerBackend(payload)
    planner = InitialPlanner(backend)

    first = planner.create_plan(question)
    second = planner.create_plan(question)

    assert first.model_dump_json() == second.model_dump_json()
    assert backend.prompts[0] == backend.prompts[1]
    assert build_initial_planner_prompt(question) == backend.prompts[0]
    assert _round_trip(first) == first


def _round_trip(plan):
    return type(plan).model_validate_json(plan.model_dump_json())
