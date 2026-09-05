import json

import pytest

from softdoc.training_data import (
    OpenAIMessagesSFTRecord,
    SFTExample,
    load_openai_messages_sft_jsonl,
    load_sft_jsonl,
)


def _record(**updates):
    payload = {
        "example_id": "example:1",
        "component": "controller",
        "prompt_version": "controller-policy-v0.10",
        "input_text": '{"current_gap":{"description":"Find revenue"}}',
        "target": {"action": "STOP", "reason": "No justified route remains."},
    }
    payload.update(updates)
    return payload


def test_sft_example_materializes_current_prompt() -> None:
    example = SFTExample.model_validate(_record())
    messages = example.messages()
    assert messages[0]["role"] == "system"
    assert "Reading Controller" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert json.loads(example.target_text())["action"] == "STOP"
    training_messages = example.training_messages()
    assert [message["role"] for message in training_messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert json.loads(training_messages[2]["content"])["action"] == "STOP"


def test_openai_messages_loader_requires_one_complete_controller_turn(tmp_path) -> None:
    path = tmp_path / "messages.jsonl"
    valid = OpenAIMessagesSFTRecord.model_validate(
        {
            "messages": [
                {"role": "system", "content": "Controller Prompt"},
                {"role": "user", "content": '{"current_gap":{}}'},
                {"role": "assistant", "content": '{"action":"STOP"}'},
            ]
        }
    )
    path.write_text(valid.model_dump_json() + "\n", encoding="utf-8")
    loaded = load_openai_messages_sft_jsonl(path)
    assert loaded == [valid]


def test_sft_example_rejects_stale_prompt_version() -> None:
    with pytest.raises(ValueError, match="registry contains"):
        SFTExample.model_validate(_record(prompt_version="controller-policy-old"))


def test_jsonl_loader_rejects_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "teacher.jsonl"
    line = json.dumps(_record())
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate example_id"):
        load_sft_jsonl(path)


def test_repository_training_example_is_valid() -> None:
    examples = load_sft_jsonl(
        __import__("pathlib").Path("configs/training/sft_example.jsonl")
    )
    assert len(examples) == 2
    planner = next(item for item in examples if item.component.value == "planner")
    assert planner.prompt_version == "planner-v0.21"
    assert json.loads(planner.target_text())["subquestions"] == []
