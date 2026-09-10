from __future__ import annotations

import json

from softdoc.prompt_registry import PromptComponent, get_prompt
from softdoc.teacher_data import audit_controller_sft_jsonl


def _controller_input(*, session: str = "reading:1", budget: int = 3) -> dict[str, object]:
    return {
        "reading_session_id": session,
        "root_question": {"question_id": "root:1", "text": "What is the value?"},
        "root_status": "incomplete",
        "subquestions": [],
        "evidence": [],
        "current_gap": {
            "question_id": "root:1",
            "description": "The requested value is unknown.",
        },
        "reading_locations": [],
        "recent_actions": [],
        "confirmed_relations": [],
        "candidate_relations": [],
        "search_tabs": [],
        "visible_search_view": None,
        "remaining_action_budget": budget,
    }


def _record(
    example_id: str,
    *,
    controller_input: dict[str, object] | None = None,
    target: object | None = None,
) -> dict[str, object]:
    prompt = get_prompt(PromptComponent.CONTROLLER)
    return {
        "example_id": example_id,
        "component": "controller",
        "prompt_version": prompt.version,
        "input_text": json.dumps(controller_input or _controller_input()),
        "target": target
        or {"action": "SEARCH", "operation": "new", "query": "requested value"},
    }


def _write_jsonl(path, records: list[object]) -> None:
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )


def test_controller_sft_audit_reports_validity_distribution_and_warning(tmp_path) -> None:
    path = tmp_path / "controller.jsonl"
    _write_jsonl(
        path,
        [
            _record("example:1"),
            _record(
                "example:2",
                controller_input=_controller_input(budget=2),
            ),
        ],
    )

    report = audit_controller_sft_jsonl(path)

    assert report.passed is True
    assert report.record_count == 2
    assert report.target_json_valid_count == 2
    assert report.action_valid_count == 2
    assert report.visible_id_valid_count == 2
    assert report.action_distribution == {"SEARCH:new": 2}
    assert report.consecutive_duplicate_action_count == 1
    assert [item.severity for item in report.issues] == ["warning"]


def test_controller_sft_audit_rejects_invisible_source_id(tmp_path) -> None:
    path = tmp_path / "controller.jsonl"
    _write_jsonl(
        path,
        [
            _record(
                "example:1",
                target={
                    "action": "READ_SOURCE",
                    "source_ids": ["element:not-visible"],
                    "local_problem": "Read the requested value.",
                },
            )
        ],
    )

    report = audit_controller_sft_jsonl(path)

    assert report.passed is False
    assert report.action_valid_count == 1
    assert report.visible_id_valid_count == 0
    assert report.action_distribution == {"READ_SOURCE": 1}
    assert any(item.category == "visible_id" for item in report.issues)


def test_controller_sft_audit_reports_duplicate_state_pair_and_bad_json(tmp_path) -> None:
    path = tmp_path / "controller.jsonl"
    duplicate = _record("example:1")
    path.write_text(
        json.dumps(duplicate)
        + "\n"
        + json.dumps({**duplicate, "example_id": "example:2"})
        + "\n{not-json}\n",
        encoding="utf-8",
    )

    report = audit_controller_sft_jsonl(path)

    assert report.passed is False
    assert report.record_count == 3
    assert report.record_json_valid_count == 2
    assert report.duplicate_state_count == 1
    assert report.duplicate_state_action_count == 1
    categories = {item.category for item in report.issues}
    assert {"duplicate_state", "duplicate_state_action", "record_json"} <= categories
