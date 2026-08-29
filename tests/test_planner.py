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
    build_initial_planner_system_prompt,
    build_initial_planner_user_prompt,
)


class MockPlannerBackend:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    @property
    def backend_name(self) -> str:
        return "mock"

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> PlannerBackendResponse:
        self.calls.append((system_prompt, user_prompt))
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
        self.calls: list[tuple[str, str]] = []

    @property
    def backend_name(self) -> str:
        return "sequence-mock"

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> PlannerBackendResponse:
        self.calls.append((system_prompt, user_prompt))
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


def test_simple_fact_can_use_empty_plan() -> None:
    question = "What is the title of Figure 3?"
    backend = MockPlannerBackend(_plan(question, []))

    result = InitialPlanner(backend).create_plan(question)

    assert result.subquestions == []
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


def test_comparison_inputs_can_remain_parallel() -> None:
    question = "Compare the revenues in 2022 and 2023."
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


def test_model_facing_schema_excludes_explicit_anchors() -> None:
    subquestion_schema = PlannerDraft.model_json_schema()["$defs"][
        "PlannedSubQuestion"
    ]
    assert "explicit_anchors" not in subquestion_schema["properties"]


def test_model_facing_schema_requires_subquestions_but_allows_empty() -> None:
    schema = PlannerDraft.model_json_schema()
    assert "subquestions" in schema["required"]
    assert "minItems" not in schema["properties"]["subquestions"]
    assert PlannerDraft.model_validate(_plan("What is Figure 3?", [])).subquestions == []


def test_missing_subquestions_is_not_confused_with_empty_plan() -> None:
    with pytest.raises(ValidationError, match="Field required"):
        PlannerDraft.model_validate({"original_question": "What is Figure 3?"})


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


def test_zero_limits_allow_only_an_empty_plan() -> None:
    question = "What is the title of Figure 3?"
    result = InitialPlanner(
        MockPlannerBackend(_plan(question, [])),
        PlannerConfig(max_subquestions=0, max_depth=1),
    ).create_plan(question)
    assert result.subquestions == []

    with pytest.raises(PlannerOutputError, match="SubQuestion limit"):
        InitialPlanner(
            MockPlannerBackend(_plan(question, [_subquestion("Q1", question)])),
            PlannerConfig(
                max_subquestions=0,
                max_depth=1,
                max_validation_attempts=1,
            ),
        ).create_plan(question)


def test_depth_one_rejects_a_nonempty_plan() -> None:
    question = "What is the title of Figure 3?"
    with pytest.raises(PlannerOutputError, match="DAG depth"):
        InitialPlanner(
            MockPlannerBackend(_plan(question, [_subquestion("Q1", question)])),
            PlannerConfig(max_depth=1, max_validation_attempts=1),
        ).create_plan(question)


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

    assert len(backend.calls) == 2
    assert backend.calls[0][0] == backend.calls[1][0]
    assert "previous response was rejected" in backend.calls[1][1]
    assert plan.planner_trace.metadata["validation_attempts"] == 2
    assert plan.planner_trace.warnings[0].code == "planner_validation_retry"


def test_prompt_defines_empty_parallel_and_dependent_plans() -> None:
    prompt = build_initial_planner_prompt(
        "Which company is higher and by how much in Figure 3?"
    )
    compact_prompt = " ".join(prompt.split())
    assert "# Role" in prompt
    assert "# Goal" in prompt
    assert "The Root is the complete original user question" in prompt
    assert "It is not emitted as a SubQuestion" in compact_prompt
    assert "Choose one of these plan shapes" in compact_prompt
    assert "Empty plan" in prompt
    assert "Parallel plan" in prompt
    assert "Dependent plan" in prompt
    assert "A factual result is an answerable value" in compact_prompt
    assert (
        "Identify only the factual results required by the wording of the Root"
        in compact_prompt
    )
    assert "return an empty subquestions list" in compact_prompt
    assert "does not require a separately answerable intermediate" in compact_prompt
    assert "two or more separately named factual results" in compact_prompt
    assert "Do not create one SubQuestion per member" in compact_prompt
    assert "single set-valued or list-valued answer" in compact_prompt
    assert "Use depends_on only when" in prompt
    assert "cannot be fully written until an earlier answer" in compact_prompt
    assert "derived quantity from separately specified values" in compact_prompt
    assert "selected entity is needed to request a different fact" in compact_prompt
    assert "Do not create a SubQuestion that repeats or paraphrases" in compact_prompt
    assert "Counting the Root as depth 1" in compact_prompt
    assert "no dependency path may exceed depth" in compact_prompt
    assert "whether facts share a page or table" not in compact_prompt
    assert "A SubQuestion with no dependencies is at depth 2" not in compact_prompt
    assert "downstream" not in prompt
    assert "Evidence" not in prompt
    assert "Answerer" not in prompt
    assert "evidence need" not in prompt
    assert "Root-direct" not in prompt
    assert "explicit_anchors" not in prompt
    assert "percentage change" not in prompt
    assert "answer_requirements" not in prompt
    assert "BBA" not in prompt
    assert "500 MHz" not in prompt
    assert "Figure 6" not in prompt


def test_planner_v020_prompt_is_frozen() -> None:
    prompt = build_initial_planner_prompt("FROZEN PLANNER PROMPT SNAPSHOT")
    assert INITIAL_PLANNER_PROMPT_VERSION == "planner-v0.20"
    assert sha256(prompt.encode("utf-8")).hexdigest() == (
        "ee98333712b028c93ff6d37759a0736d4b6c37b7c3cb511fae0c58a7e78e5a70"
    )


def test_ollama_backend_uses_local_chat_api_and_json_schema() -> None:
    question = "What is the revenue?"
    content = json.dumps(_plan(question, []))
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

    response = backend.generate(
        system_prompt="planner system prompt",
        user_prompt="planner user prompt",
    )

    assert response.content == content
    assert response.metadata == {"eval_count": 42}
    url, payload, timeout = transport.calls[0]
    assert url == "http://localhost:11434/api/chat"
    assert timeout == 12
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["format"] == PlannerDraft.model_json_schema()
    assert payload["messages"] == [
        {"role": "system", "content": "planner system prompt"},
        {"role": "user", "content": "planner user prompt"},
    ]


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
    assert backend.calls[0] == backend.calls[1]
    assert backend.calls[0] == (
        build_initial_planner_system_prompt(),
        build_initial_planner_user_prompt(question),
    )
    assert question not in backend.calls[0][0]
    assert backend.calls[0][1].count(question) == 1
    assert build_initial_planner_prompt(question).endswith(backend.calls[0][1])
    assert _round_trip(first) == first


def _round_trip(plan):
    return type(plan).model_validate_json(plan.model_dump_json())
