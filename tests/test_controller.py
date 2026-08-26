import pytest
from pydantic import ValidationError

from softdoc.controller import (
    CONTROLLER_ACTION_VERSION,
    CONTROLLER_INPUT_VERSION,
    AdjacentPageDirection,
    ControllerActionName,
    ControllerCandidatePreview,
    ControllerCandidateRelation,
    ControllerConfirmedRelation,
    ControllerEvidence,
    ControllerExploreCandidateRelationAction,
    ControllerFollowRelationAction,
    ControllerGap,
    ControllerInput,
    ControllerReadAdjacentPageAction,
    ControllerReadSourceAction,
    ControllerReadingLocation,
    ControllerRecentAction,
    ControllerSearchAction,
    ControllerSearchOperation,
    ControllerSearchTab,
    ControllerSubQuestion,
    ControllerVisibleSearchView,
    validate_controller_action,
)
from softdoc.models import ContentAvailability, ElementType, RelationType
from softdoc.reading_state import (
    ActionExecutionStatus,
    EvidenceStatus,
    QuestionStatus,
    ReadingSourceType,
    RootQuestion,
)


def _input(**changes: object) -> ControllerInput:
    data: dict[str, object] = {
        "reading_session_id": "reading:1",
        "root_question": RootQuestion(
            question_id="root:1",
            text="How did revenue change?",
        ),
        "root_status": EvidenceStatus.INCOMPLETE,
        "subquestions": [
            ControllerSubQuestion(
                question_id="Q1",
                text="What was 2022 revenue?",
                depends_on=[],
                status=QuestionStatus.SATISFIED,
            ),
            ControllerSubQuestion(
                question_id="Q2",
                text="What was 2023 revenue?",
                depends_on=[],
                status=QuestionStatus.INCOMPLETE,
            ),
        ],
        "evidence": [
            ControllerEvidence(
                evidence_id="evidence:1",
                statement="Revenue in 2022 was 10 million.",
                supports_question_ids=["Q1"],
            )
        ],
        "current_gap": ControllerGap(
            question_id="Q2",
            description="The 2023 revenue is unknown.",
        ),
        "reading_locations": [
            ControllerReadingLocation(
                source_id="element:table:1",
                source_type=ReadingSourceType.ELEMENT,
                page_id="page:opaque",
            )
        ],
        "recent_actions": [
            ControllerRecentAction(
                action_id="action:1",
                question_id="Q2",
                action_name="READ_SOURCE",
                target_ids=["element:table:1"],
                execution_status=ActionExecutionStatus.SUCCEEDED,
                observation_ids=["observation:1"],
            )
        ],
        "search_tabs": [
            ControllerSearchTab(
                search_session_id="search:1",
                query="2023 revenue",
                has_more=True,
            )
        ],
        "visible_search_view": ControllerVisibleSearchView(
            search_session_id="search:1",
            candidate_previews=[
                ControllerCandidatePreview(
                    element_id="element:table:1",
                    element_type=ElementType.TABLE,
                    page_id="page:opaque",
                    section_path=["Financial Results"],
                    matched_snippet="Revenue ... 2022 ... 2023 ...",
                    content_availability=ContentAvailability.STRUCTURED,
                )
            ],
        ),
        "remaining_action_budget": 4,
    }
    data.update(changes)
    return ControllerInput.model_validate(data)


def test_controller_input_version_and_round_trip() -> None:
    value = _input()
    assert CONTROLLER_INPUT_VERSION == "controller-input-v0.1"
    assert CONTROLLER_ACTION_VERSION == "controller-action-v0.1"
    restored = ControllerInput.model_validate_json(value.model_dump_json())
    assert restored == value
    assert restored.recent_actions[0].execution_status == "succeeded"


def test_ready_root_requires_null_gap() -> None:
    with pytest.raises(ValidationError, match="ready Root must not have"):
        _input(root_status=EvidenceStatus.READY)

    value = _input(root_status=EvidenceStatus.READY, current_gap=None)
    assert value.current_gap is None


def test_incomplete_root_requires_gap() -> None:
    with pytest.raises(ValidationError, match="incomplete Root requires"):
        _input(current_gap=None)


def test_visible_search_view_must_reference_tab() -> None:
    with pytest.raises(ValidationError, match="must reference a SearchSession"):
        _input(
            visible_search_view=ControllerVisibleSearchView(
                search_session_id="search:missing"
            )
        )


def test_outcome_is_not_a_controller_field() -> None:
    payload = _input().model_dump(mode="json")
    payload["recent_actions"][0]["outcome"] = "succeeded"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ControllerInput.model_validate(payload)


def test_frozen_action_names_exclude_deferred_region_and_plan_actions() -> None:
    assert {item.value for item in ControllerActionName} == {
        "SEARCH",
        "READ_SOURCE",
        "FOLLOW_RELATION",
        "EXPLORE_CANDIDATE_RELATION",
        "READ_ADJACENT_PAGE",
    }


def test_search_new_requires_only_a_query() -> None:
    action = validate_controller_action(
        {"action": "SEARCH", "operation": "new", "query": "2023 revenue"},
        _input(),
    )
    assert isinstance(action, ControllerSearchAction)
    assert action.operation == ControllerSearchOperation.NEW

    with pytest.raises(ValidationError, match="requires a nonblank query"):
        validate_controller_action(
            {"action": "SEARCH", "operation": "new"},
            _input(),
        )


def test_search_next_and_switch_require_visible_sessions() -> None:
    next_action = validate_controller_action(
        {
            "action": "SEARCH",
            "operation": "next",
            "search_session_id": "search:1",
        },
        _input(),
    )
    assert isinstance(next_action, ControllerSearchAction)

    with pytest.raises(ValueError, match="not visible"):
        validate_controller_action(
            {
                "action": "SEARCH",
                "operation": "switch",
                "search_session_id": "search:missing",
            },
            _input(),
        )

    with pytest.raises(ValueError, match="has_more"):
        validate_controller_action(
            {
                "action": "SEARCH",
                "operation": "next",
                "search_session_id": "search:1",
            },
            _input(
                search_tabs=[
                    ControllerSearchTab(
                        search_session_id="search:1",
                        query="2023 revenue",
                        has_more=False,
                    )
                ]
            ),
        )


def test_read_source_selects_only_visible_sources() -> None:
    action = validate_controller_action(
        {
            "action": "READ_SOURCE",
            "source_ids": ["element:table:1"],
            "local_problem": "Read the 2023 revenue.",
        },
        _input(),
    )
    assert isinstance(action, ControllerReadSourceAction)

    with pytest.raises(ValueError, match="not visible"):
        validate_controller_action(
            {
                "action": "READ_SOURCE",
                "source_ids": ["element:hidden"],
                "local_problem": "Read the hidden source.",
            },
            _input(),
        )


def test_confirmed_and_candidate_relations_use_different_actions() -> None:
    state = _input(
        confirmed_relations=[
            ControllerConfirmedRelation(
                relation_id="relation:confirmed",
                relation_type=RelationType.CAPTION_OF,
                source_id="element:caption:1",
                target_id="element:figure:1",
            )
        ],
        candidate_relations=[
            ControllerCandidateRelation(
                relation_id="relation:candidate",
                relation_type=RelationType.CONTINUED_ON,
                source_id="element:table:1",
                target_id="element:table:2",
                confidence=0.8,
            )
        ],
    )
    followed = validate_controller_action(
        {
            "action": "FOLLOW_RELATION",
            "relation_id": "relation:confirmed",
            "local_problem": "Read the figure belonging to this caption.",
        },
        state,
    )
    explored = validate_controller_action(
        {
            "action": "EXPLORE_CANDIDATE_RELATION",
            "relation_id": "relation:candidate",
            "local_problem": "Check whether the table continues.",
        },
        state,
    )
    assert isinstance(followed, ControllerFollowRelationAction)
    assert isinstance(explored, ControllerExploreCandidateRelationAction)

    with pytest.raises(ValueError, match="confirmed Relation"):
        validate_controller_action(
            {
                "action": "FOLLOW_RELATION",
                "relation_id": "relation:candidate",
                "local_problem": "Follow it.",
            },
            state,
        )


def test_read_adjacent_page_uses_one_action_with_a_direction() -> None:
    action = validate_controller_action(
        {
            "action": "READ_ADJACENT_PAGE",
            "from_page_id": "page:opaque",
            "direction": "previous",
            "local_problem": "Recover the missing table header.",
        },
        _input(),
    )
    assert isinstance(action, ControllerReadAdjacentPageAction)
    assert action.direction == AdjacentPageDirection.PREVIOUS

    with pytest.raises(ValueError, match="visible in reading_locations"):
        validate_controller_action(
            {
                "action": "READ_ADJACENT_PAGE",
                "from_page_id": "page:hidden",
                "direction": "next",
                "local_problem": "Check the next page.",
            },
            _input(),
        )


def test_unfrozen_action_names_are_rejected() -> None:
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        validate_controller_action(
            {"action": "INSPECT_REGION", "region_id": "region:1"},
            _input(),
        )
