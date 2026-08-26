import json
from typing import Any

import pytest

from softdoc.controller import (
    ControllerCandidatePreview,
    ControllerGap,
    ControllerInput,
    ControllerSearchTab,
    ControllerVisibleSearchView,
)
from softdoc.controller_ollama import (
    OllamaControllerBackend,
    OllamaControllerConfig,
    OllamaControllerError,
)
from softdoc.controller_prompt import CONTROLLER_SYSTEM_PROMPT
from softdoc.models import ContentAvailability, ElementType
from softdoc.reading_state import EvidenceStatus, RootQuestion


class FakeTransport:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, Any], float]] = []

    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        self.calls.append((url, payload, timeout_seconds))
        return self.response


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
    assert generation.metadata == {"prompt_eval_count": 123}
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


def test_ollama_controller_reports_missing_message_content() -> None:
    backend = OllamaControllerBackend(
        transport=FakeTransport({"error": "model not found"})
    )

    with pytest.raises(OllamaControllerError, match="model not found"):
        backend.decide(controller_input())
