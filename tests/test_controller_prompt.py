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
    assert CONTROLLER_PROMPT_VERSION == "controller-policy-v0.2"
    for action_name in (
        "SEARCH",
        "READ_SOURCE",
        "FOLLOW_RELATION",
        "EXPLORE_CANDIDATE_RELATION",
        "READ_ADJACENT_PAGE",
    ):
        assert action_name in CONTROLLER_SYSTEM_PROMPT
    assert '"action": "STOP"' not in CONTROLLER_SYSTEM_PROMPT
    assert "INSPECT_REGION" not in CONTROLLER_SYSTEM_PROMPT


def test_controller_prompt_uses_flexible_route_selection_and_scoped_recheck() -> None:
    assert "Do not follow a fixed action priority" in CONTROLLER_SYSTEM_PROMPT
    assert "conflict,\n  uncertainty, incomplete grounding, or failed reading" in (
        CONTROLLER_SYSTEM_PROMPT
    )
    assert "Any verification must\n  remain focused on the current gap" in (
        CONTROLLER_SYSTEM_PROMPT
    )


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
