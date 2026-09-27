from __future__ import annotations

import pytest

from scripts.train_sft import _encode_example, _extract_input_ids
from softdoc.prompt_registry import PromptComponent, get_prompt
from softdoc.training_data import SFTExample


def test_extract_input_ids_accepts_legacy_flat_list() -> None:
    assert _extract_input_ids([1, 2, 3], field="prompt") == [1, 2, 3]


def test_extract_input_ids_accepts_transformers_5_mapping() -> None:
    assert _extract_input_ids(
        {"input_ids": [4, 5], "attention_mask": [1, 1]}, field="prompt"
    ) == [4, 5]


def test_extract_input_ids_unwraps_single_batch() -> None:
    assert _extract_input_ids({"input_ids": [[6, 7]]}, field="prompt") == [6, 7]


def test_extract_input_ids_rejects_mapping_keys_as_tokens() -> None:
    with pytest.raises(TypeError, match="flat integer sequence"):
        _extract_input_ids(["input_ids"], field="prompt")


class _LengthControlledTokenizer:
    eos_token = "<eos>"

    def __init__(self, prompt_length: int, target_length: int) -> None:
        self.prompt_length = prompt_length
        self.target_length = target_length

    def apply_chat_template(self, *_args, **_kwargs):
        return list(range(self.prompt_length))

    def __call__(self, _text: str, **_kwargs):
        return {"input_ids": list(range(self.target_length))}


def _controller_example() -> SFTExample:
    return SFTExample(
        example_id="controller:length-check",
        component=PromptComponent.CONTROLLER,
        prompt_version=get_prompt(PromptComponent.CONTROLLER).version,
        input_text='{"reading_session_id":"session:test"}',
        target={"action": "STOP", "reason": "No supported action remains."},
    )


def test_encode_example_refuses_to_silently_left_truncate_prompt() -> None:
    with pytest.raises(ValueError, match="refusing to left-truncate"):
        _encode_example(
            _controller_example(),
            _LengthControlledTokenizer(prompt_length=9, target_length=4),
            max_length=12,
        )


def test_encode_example_preserves_complete_prompt_when_it_fits() -> None:
    encoded = _encode_example(
        _controller_example(),
        _LengthControlledTokenizer(prompt_length=6, target_length=4),
        max_length=12,
    )

    assert encoded.input_ids == [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]
    assert encoded.labels == [-100] * 6 + [0, 1, 2, 3]
