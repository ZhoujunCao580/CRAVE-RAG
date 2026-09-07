import json
from typing import Any

import pytest

from softdoc.controller import (
    ControllerCandidatePreview,
    ControllerConfirmedRelation,
    ControllerGap,
    ControllerInput,
    ControllerReadingLocation,
    ControllerRecentAction,
    ControllerRelationEndpointPreview,
    ControllerSearchTab,
    ControllerVisibleSearchView,
)
from softdoc.controller_ollama import (
    OllamaControllerBackend,
    OllamaControllerConfig,
    OllamaControllerError,
    VLLMControllerBackend,
)
from softdoc.controller_prompt import CONTROLLER_SYSTEM_PROMPT
from softdoc.models import ContentAvailability, ElementType, RelationType
from softdoc.openai_compatible import OpenAICompatibleConfig
from softdoc.reading_state import (
    ActionExecutionStatus,
    EvidenceStatus,
    ReadingSourceType,
    RootQuestion,
)


class FakeTransport:
    def __init__(self, response: dict[str, Any] | list[dict[str, Any]]) -> None:
        self.responses = response if isinstance(response, list) else [response]
        self.calls: list[tuple[str, dict[str, Any], float]] = []

    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        self.calls.append((url, payload, timeout_seconds))
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]


class FakeStructuredClient:
    def __init__(self, actions: list[dict[str, Any]]) -> None:
        self.actions = list(actions)
        self.calls: list[dict[str, Any]] = []
        self.last_raw_content: str | None = None

    def generate(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        action = kwargs["output_model"].validate_python(
            self.actions[min(len(self.calls) - 1, len(self.actions) - 1)]
        )
        self.last_raw_content = action.model_dump_json()
        return action


def controller_input() -> ControllerInput:
    return ControllerInput(
        reading_session_id="reading:1",
        root_question=RootQuestion(
            question_id="root:1",
            text="What revenue was reported?",
        ),
        root_status=EvidenceStatus.INCOMPLETE,
        current_gap=ControllerGap(
            question_id="root:1",
            description="The reported revenue is unknown.",
        ),
        search_tabs=[
            ControllerSearchTab(
                search_session_id="search:1",
                query="reported revenue",
                has_more=True,
            )
        ],
        visible_search_view=ControllerVisibleSearchView(
            search_session_id="search:1",
            candidate_previews=[
                ControllerCandidatePreview(
                    element_id="element:table:1",
                    element_type=ElementType.TABLE,
                    page_id="page:opaque",
                    section_path=["Results"],
                    matched_snippet="Revenue was 12 million.",
                    content_availability=ContentAvailability.STRUCTURED,
                )
            ],
        ),
        remaining_action_budget=4,
    )


def test_ollama_controller_sends_frozen_prompt_and_schema() -> None:
    raw_action = {
        "action": "READ_SOURCE",
        "source_ids": ["element:table:1"],
        "local_problem": "Read the reported revenue.",
    }
    transport = FakeTransport(
        {
            "model": "qwen3:8b",
            "message": {"content": json.dumps(raw_action)},
            "prompt_eval_count": 123,
        }
    )
    backend = OllamaControllerBackend(
        OllamaControllerConfig(base_url="http://localhost:11434/", seed=7),
        transport,
    )

    generation = backend.generate(controller_input())

    assert generation.action.action.value == "READ_SOURCE"
    assert generation.metadata == {
        "prompt_eval_count": 123,
        "validation_attempts": 1,
    }
    url, payload, timeout = transport.calls[0]
    assert url == "http://localhost:11434/api/chat"
    assert payload["messages"][0] == {
        "role": "system",
        "content": CONTROLLER_SYSTEM_PROMPT,
    }
    assert ControllerInput.model_validate_json(payload["messages"][1]["content"])
    assert payload["format"]["oneOf"]
    assert payload["think"] is False
    assert payload["options"]["temperature"] == 0.0
    assert payload["options"]["seed"] == 7
    assert timeout == 180.0


def test_ollama_controller_rejects_invalid_json_and_preserves_raw_output() -> None:
    transport = FakeTransport({"message": {"content": "not-json"}})
    backend = OllamaControllerBackend(transport=transport)

    with pytest.raises(OllamaControllerError) as captured:
        backend.decide(controller_input())

    assert captured.value.raw_content == "not-json"


def test_ollama_controller_rejects_invented_handle() -> None:
    raw_action = {
        "action": "READ_SOURCE",
        "source_ids": ["element:invented"],
        "local_problem": "Read the revenue.",
    }
    transport = FakeTransport(
        {"message": {"content": json.dumps(raw_action)}}
    )
    backend = OllamaControllerBackend(transport=transport)

    with pytest.raises(OllamaControllerError, match="not visible"):
        backend.decide(controller_input())

    assert len(transport.calls) == 2
    repair_prompt = transport.calls[1][1]["messages"][1]["content"]
    assert "element:invented" in repair_prompt
    assert '"read_source_ids"' in repair_prompt
    assert "element:table:1" in repair_prompt
    assert "Do not guess, repair, or fuzzy-match an ID" in repair_prompt


def test_ollama_controller_repairs_historical_id_without_spending_an_action() -> None:
    state = controller_input().model_copy(
        update={
            "recent_actions": [
                ControllerRecentAction(
                    action_id="action:search:old",
                    question_id="root:1",
                    action_name="SEARCH",
                    target_ids=["element:old-batch"],
                    execution_status=ActionExecutionStatus.SUCCEEDED,
                )
            ]
        }
    )
    transport = FakeTransport(
        [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "action": "READ_SOURCE",
                            "source_ids": ["element:old-batch"],
                            "local_problem": "Read the reported revenue.",
                        }
                    )
                }
            },
            {
                "message": {
                    "content": json.dumps(
                        {
                            "action": "READ_SOURCE",
                            "source_ids": ["element:table:1"],
                            "local_problem": "Read the reported revenue.",
                        }
                    )
                }
            },
        ]
    )
    generation = OllamaControllerBackend(transport=transport).generate(state)

    assert generation.action.source_ids == ["element:table:1"]
    assert generation.metadata["validation_attempts"] == 2
    assert len(generation.metadata["rejected_attempts"]) == 1
    repair_prompt = transport.calls[1][1]["messages"][1]["content"]
    assert '"historical_non_actionable_ids": [\n    "element:old-batch"' in repair_prompt


def test_ollama_controller_repairs_page_id_to_page_context_action() -> None:
    state = controller_input().model_copy(
        update={
            "reading_locations": [
                ControllerReadingLocation(
                    source_id="element:opened",
                    source_type=ReadingSourceType.ELEMENT,
                    page_id="page:opaque",
                    physical_page_number=4,
                )
            ]
        }
    )
    transport = FakeTransport(
        [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "action": "READ_SOURCE",
                            "source_ids": ["page:opaque"],
                            "local_problem": "Read the complete page.",
                        }
                    )
                }
            },
            {
                "message": {
                    "content": json.dumps(
                        {
                            "action": "READ_PAGE_CONTEXT",
                            "base_page_id": "page:opaque",
                            "offset": 0,
                            "local_problem": "Read the complete page.",
                        }
                    )
                }
            },
        ]
    )
    generation = OllamaControllerBackend(transport=transport).generate(state)

    assert generation.action.action.value == "READ_PAGE_CONTEXT"
    assert generation.action.base_page_id == "page:opaque"
    assert generation.metadata["validation_attempts"] == 2


def test_ollama_controller_repairs_malformed_long_id_without_fuzzy_mapping() -> None:
    valid_id = "doc:guide:hash:page:0016:element:0000:figure:main"
    malformed_id = "doc:guide:hash:page:0016:element:0000:main"
    state = controller_input().model_copy(
        update={
            "visible_search_view": ControllerVisibleSearchView(
                search_session_id="search:1",
                candidate_previews=[
                    ControllerCandidatePreview(
                        element_id=valid_id,
                        element_type=ElementType.FIGURE,
                        page_id="doc:guide:hash:page:0016",
                        matched_snippet="A map candidate.",
                        content_availability=ContentAvailability.VISUAL_ONLY,
                    )
                ],
            )
        }
    )
    transport = FakeTransport(
        [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "action": "READ_SOURCE",
                            "source_ids": [malformed_id],
                            "local_problem": "Read the map.",
                        }
                    )
                }
            },
            {
                "message": {
                    "content": json.dumps(
                        {
                            "action": "READ_SOURCE",
                            "source_ids": [valid_id],
                            "local_problem": "Read the map.",
                        }
                    )
                }
            },
        ]
    )

    generation = OllamaControllerBackend(transport=transport).generate(state)

    assert generation.action.source_ids == [valid_id]
    assert generation.metadata["validation_attempts"] == 2
    repair_prompt = transport.calls[1][1]["messages"][1]["content"]
    assert malformed_id in repair_prompt
    assert valid_id in repair_prompt
    assert "fuzzy-match" in repair_prompt


def test_vllm_controller_repairs_relation_endpoint_to_relation_action() -> None:
    relation = ControllerConfirmedRelation(
        relation_id="relation:caption-of",
        relation_type=RelationType.CAPTION_OF,
        source_id="element:opened",
        target_id="element:related-figure",
        current_endpoint_id="element:opened",
        related_source_preview=ControllerRelationEndpointPreview(
            source_id="element:related-figure",
            source_type=ReadingSourceType.ELEMENT,
            page_id="page:opaque",
            element_type=ElementType.FIGURE,
            label_or_snippet="The related figure.",
            content_availability=ContentAvailability.VISUAL_ONLY,
        ),
    )
    state = controller_input().model_copy(
        update={"confirmed_relations": [relation]}
    )
    client = FakeStructuredClient(
        [
            {
                "action": "READ_SOURCE",
                "source_ids": ["element:related-figure"],
                "local_problem": "Read the related figure.",
            },
            {
                "action": "FOLLOW_RELATION",
                "relation_id": "relation:caption-of",
                "local_problem": "Read the related figure.",
            },
        ]
    )
    backend = VLLMControllerBackend(
        OpenAICompatibleConfig(model="fake"),
        client=client,
    )

    generation = backend.generate(state)

    assert generation.action.action.value == "FOLLOW_RELATION"
    assert generation.metadata["validation_attempts"] == 2
    assert len(client.calls) == 2
    assert "relation:caption-of" in client.calls[1]["user_prompt"]
    assert "element:related-figure" not in json.loads(
        client.calls[1]["user_prompt"].split("Current actionable permissions:\n", 1)[1]
        .split("\n\nPermission rules:", 1)[0]
    )["read_source_ids"]


def test_ollama_controller_reports_missing_message_content() -> None:
    backend = OllamaControllerBackend(
        transport=FakeTransport({"error": "model not found"})
    )

    with pytest.raises(OllamaControllerError, match="model not found"):
        backend.decide(controller_input())
