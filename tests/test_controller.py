from pathlib import Path

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
    ControllerRelationEndpointPreview,
    ControllerRecentAction,
    ControllerSearchAction,
    ControllerSearchOperation,
    ControllerSearchTab,
    ControllerStopAction,
    ControllerSubQuestion,
    ControllerVisibleSearchView,
    build_controller_relation_endpoint_preview,
    validate_controller_action,
)
from softdoc.models import (
    ContentAvailability,
    Element,
    ElementType,
    Page,
    Provenance,
    RelationType,
)
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


def _endpoint(
    source_id: str,
    *,
    element_type: ElementType,
    text: str,
    page_id: str = "page:relation",
    availability: ContentAvailability = ContentAvailability.TEXT_ONLY,
) -> ControllerRelationEndpointPreview:
    return ControllerRelationEndpointPreview(
        source_id=source_id,
        source_type=ReadingSourceType.ELEMENT,
        page_id=page_id,
        element_type=element_type,
        label_or_snippet=text,
        content_availability=availability,
    )


def test_controller_input_version_and_round_trip() -> None:
    value = _input()
    assert CONTROLLER_INPUT_VERSION == "controller-input-v0.3"
    assert CONTROLLER_ACTION_VERSION == "controller-action-v0.2"
    restored = ControllerInput.model_validate_json(value.model_dump_json())
    assert restored == value
    assert restored.recent_actions[0].execution_status == "succeeded"


def test_relation_endpoint_preview_is_deterministic_and_uses_table_content() -> None:
    table = Element(
        element_id="element:table:2",
        document_id="doc:1",
        page_id="page:2",
        page_number=2,
        element_type=ElementType.TABLE,
        reading_order=3,
        section_path=["Financial Results"],
        reference_label="Table 2",
        html=(
            "<table><tr><th>Year</th><th>Revenue</th></tr>"
            "<tr><td>2023</td><td>12 million</td></tr></table>"
        ),
        provenance=Provenance(
            provenance_id="prov:table:2",
            adapter="test",
            source_path=Path("fixture.json"),
            source_locator="table:2",
        ),
    )

    first = build_controller_relation_endpoint_preview(table)
    second = build_controller_relation_endpoint_preview(table)

    assert first == second
    assert first.source_id == table.element_id
    assert first.element_type == ElementType.TABLE
    assert first.section_path == ["Financial Results"]
    assert first.label_or_snippet == "Table 2 — Year Revenue 2023 12 million"
    assert len(first.label_or_snippet) <= 240


def test_page_relation_endpoint_preview_keeps_page_handle_opaque() -> None:
    page = Page(
        page_id="page:opaque",
        document_id="doc:1",
        page_index=17,
        page_number=18,
        width=612,
        height=792,
        display_page_label="12",
        page_label_aliases=["12"],
        provenance=Provenance(
            provenance_id="prov:page:opaque",
            adapter="test",
            source_path=Path("fixture.json"),
            source_locator="page:18",
        ),
    )

    preview = build_controller_relation_endpoint_preview(page)

    assert preview.source_id == "page:opaque"
    assert preview.page_id == "page:opaque"
    assert preview.label_or_snippet == ""
    assert preview.element_type is None
    assert preview.content_availability is None


def test_relation_view_requires_preview_of_the_opposite_endpoint() -> None:
    with pytest.raises(ValidationError, match="opposite Relation endpoint"):
        ControllerConfirmedRelation(
            relation_id="relation:bad-preview",
            relation_type=RelationType.CAPTION_OF,
            source_id="element:caption:1",
            target_id="element:figure:1",
            current_endpoint_id="element:caption:1",
            related_source_preview=_endpoint(
                "element:another-figure",
                element_type=ElementType.FIGURE,
                text="Wrong endpoint",
                availability=ContentAvailability.VISUAL_ONLY,
            ),
        )


def test_ready_root_requires_null_gap() -> None:
    with pytest.raises(ValidationError, match="ready Root must not have"):
        _input(root_status=EvidenceStatus.READY)

    value = _input(
        root_status=EvidenceStatus.READY,
        current_gap=None,
        subquestions=[
            ControllerSubQuestion(
                question_id="Q1",
                text="What was 2022 revenue?",
                status=QuestionStatus.SATISFIED,
            ),
            ControllerSubQuestion(
                question_id="Q2",
                text="What was 2023 revenue?",
                status=QuestionStatus.SATISFIED,
            ),
        ],
    )
    assert value.current_gap is None

    with pytest.raises(ValidationError, match="every SubQuestion"):
        _input(root_status=EvidenceStatus.READY, current_gap=None)


def test_incomplete_root_requires_gap() -> None:
    with pytest.raises(ValidationError, match="incomplete Root requires"):
        _input(current_gap=None)


def test_root_direct_controller_input_accepts_empty_subquestions_and_root_evidence() -> None:
    value = _input(
        subquestions=[],
        evidence=[
            ControllerEvidence(
                evidence_id="evidence:root",
                statement="Figure 3 is titled System Architecture.",
                supports_question_ids=["root:1"],
            )
        ],
        current_gap=ControllerGap(
            question_id="root:1",
            description="Verify the complete title of Figure 3.",
        ),
        recent_actions=[],
    )

    assert value.subquestions == []
    assert value.current_gap.question_id == value.root_question.question_id
    assert value.evidence[0].supports_question_ids == ["root:1"]


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
        "STOP",
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
                current_endpoint_id="element:caption:1",
                related_source_preview=_endpoint(
                    "element:figure:1",
                    element_type=ElementType.FIGURE,
                    text="Figure 1. Low-light accuracy comparison.",
                    availability=ContentAvailability.VISUAL_ONLY,
                ),
            )
        ],
        candidate_relations=[
            ControllerCandidateRelation(
                relation_id="relation:candidate",
                relation_type=RelationType.CONTINUED_ON,
                source_id="element:table:1",
                target_id="element:table:2",
                current_endpoint_id="element:table:1",
                related_source_preview=_endpoint(
                    "element:table:2",
                    element_type=ElementType.TABLE,
                    text="Continuation containing the final totals.",
                    availability=ContentAvailability.STRUCTURED,
                ),
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


def test_stop_keeps_incomplete_state_without_requiring_a_handle() -> None:
    action = validate_controller_action(
        {
            "action": "STOP",
            "reason": "No visible route is likely to resolve the gap.",
        },
        _input(),
    )
    assert isinstance(action, ControllerStopAction)


def test_unfrozen_action_names_are_rejected() -> None:
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        validate_controller_action(
            {"action": "INSPECT_REGION", "region_id": "region:1"},
            _input(),
        )
