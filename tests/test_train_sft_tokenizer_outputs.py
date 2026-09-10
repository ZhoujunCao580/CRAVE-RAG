from __future__ import annotations

import pytest

from scripts.train_sft import _extract_input_ids


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
