import json
from pathlib import Path

from scripts.evaluate_controller_mock import build_cases, matches_expected
from softdoc.controller import (
    ControllerGap,
    ControllerInput,
)
from softdoc.controller_prompt import (
    CONTROLLER_PROMPT_VERSION,
    CONTROLLER_SYSTEM_PROMPT,
    build_controller_user_prompt,
)
from softdoc.reading_state import EvidenceStatus, RootQuestion


def test_controller_prompt_is_frozen_to_current_action_contract() -> None:
    assert CONTROLLER_PROMPT_VERSION == "controller-policy-v0.8"
    assert "Treat ControllerInput as read-only" in CONTROLLER_SYSTEM_PROMPT
    assert "using only IDs supplied\nin ControllerInput" in CONTROLLER_SYSTEM_PROMPT
    assert "Return only the action JSON matching the provided Schema" in (
        CONTROLLER_SYSTEM_PROMPT
    )
    for action_name in (
        "SEARCH",
        "READ_SOURCE",
        "FOLLOW_RELATION",
        "EXPLORE_CANDIDATE_RELATION",
        "READ_ADJACENT_PAGE",
        "STOP",
    ):
        assert action_name in CONTROLLER_SYSTEM_PROMPT
    assert '"action": "STOP"' in CONTROLLER_SYSTEM_PROMPT
    assert CONTROLLER_SYSTEM_PROMPT.count("# Output") == 1
    assert "one of the following action shapes" in CONTROLLER_SYSTEM_PROMPT
    assert "INSPECT_REGION" not in CONTROLLER_SYSTEM_PROMPT


def test_controller_prompt_uses_flexible_route_selection_and_scoped_recheck() -> None:
    assert "Do not follow a fixed action priority" in CONTROLLER_SYSTEM_PROMPT
    assert "conflict,\n  uncertainty, incomplete grounding, or failed reading" in (
        CONTROLLER_SYSTEM_PROMPT
    )
    assert "Any verification must\n  remain focused on the current gap" in (
        CONTROLLER_SYSTEM_PROMPT
    )
    assert (
        "Confirmed status establishes that\n"
        "   the link is accepted; it does not establish that the related source is\n"
        "   relevant to the current gap."
        in CONTROLLER_SYSTEM_PROMPT
    )
    assert "inspect relation_type and\n   related_source_preview" in (
        CONTROLLER_SYSTEM_PROMPT
    )
    assert "label_or_snippet" in CONTROLLER_SYSTEM_PROMPT
    assert "navigation clue, not\n  Evidence" in CONTROLLER_SYSTEM_PROMPT
    assert "existing routes are irrelevant, exhausted" in CONTROLLER_SYSTEM_PROMPT
    assert "specific recoverable failure" in CONTROLLER_SYSTEM_PROMPT


def test_controller_user_prompt_is_only_validated_input_json() -> None:
    controller_input = ControllerInput(
        reading_session_id="reading:1",
        root_question=RootQuestion(
            question_id="root:1",
            text="What value is reported?",
        ),
        root_status=EvidenceStatus.INCOMPLETE,
        current_gap=ControllerGap(
            question_id="root:1",
            description="The reported value is unknown.",
        ),
        remaining_action_budget=3,
    )

    serialized = build_controller_user_prompt(controller_input)

    assert ControllerInput.model_validate_json(serialized) == controller_input
    assert "Choose exactly one" not in serialized


def test_controller_prompt_maps_preview_handles_without_exposing_page_as_source() -> None:
    assert "copy its element_id into READ_SOURCE.source_ids" in CONTROLLER_SYSTEM_PROMPT
    assert "CandidatePreview page_id is location metadata" in CONTROLLER_SYSTEM_PROMPT
    assert (
        "A visual candidate may be promising when the required information is likely"
        in CONTROLLER_SYSTEM_PROMPT
    )
    assert (
        "Candidate position does not\n  guarantee relevance or source type."
        in CONTROLLER_SYSTEM_PROMPT
    )


def test_controller_policy_suite_covers_v06_relation_preview_boundaries() -> None:
    cases = {case.case_id: case for case in build_cases()}

    assert len(cases) == 20
    assert cases["K16"].category == "irrelevant_confirmed_relation"
    assert cases["K17"].category == "avoid_exact_repeat"
    assert cases["K18"].category == "targeted_reread_after_feedback"
    assert cases["K19"].category == "decomposed_current_gap"
    assert cases["K20"].category == "stop_no_justified_route"
    assert matches_expected(
        {
            "action": "READ_SOURCE",
            "source_ids": ["element:multi-year-table"],
            "local_problem": "Read the 2023 revenue row.",
        },
        cases["K18"].expected,
    )
    assert not matches_expected(
        {
            "action": "READ_SOURCE",
            "source_ids": ["element:multi-year-table"],
            "local_problem": "Read the 2022 revenue row.",
        },
        cases["K18"].expected,
    )


def test_materialized_controller_suite_separates_inputs_and_gold() -> None:
    root = Path("evals/prompts/controller")
    model_rows = [
        json.loads(line)
        for line in (root / "model_inputs/controller_cases_v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    gold_rows = [
        json.loads(line)
        for line in (root / "review_only/controller_gold_v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert len(model_rows) == len(gold_rows) == 28
    assert {row["case_id"] for row in model_rows} == {
        row["case_id"] for row in gold_rows
    }
    assert all(set(row) == {"case_id", "controller_input"} for row in model_rows)
    assert all("expected_action" not in row for row in model_rows)
    for row in model_rows:
        ControllerInput.model_validate(row["controller_input"])

    relation_case = next(row for row in model_rows if row["case_id"] == "K16")
    relation = relation_case["controller_input"]["confirmed_relations"][0]
    assert relation["current_endpoint_id"] == "element:methods-caption"
    assert relation["related_source_preview"]["source_id"] == "element:methods-figure"
    assert "training methodology" in relation["related_source_preview"]["label_or_snippet"]
