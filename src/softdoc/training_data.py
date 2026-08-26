"""Version-bound supervised-training records for model-facing components."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from softdoc.prompt_registry import PromptComponent, get_prompt


class SFTExample(BaseModel):
    """One prompt input and its Teacher output.

    Prompt text is materialized from the canonical registry at training time,
    so a dataset cannot silently train against a different prompt version.
    """

    model_config = ConfigDict(extra="forbid")

    example_id: str = Field(min_length=1)
    component: PromptComponent
    prompt_version: str = Field(min_length=1)
    input_text: str = Field(min_length=1)
    target: str | dict[str, Any]

    @model_validator(mode="after")
    def validate_prompt_binding(self) -> "SFTExample":
        spec = get_prompt(self.component)
        if self.prompt_version != spec.version:
            raise ValueError(
                f"{self.component.value} example expects prompt {self.prompt_version!r}, "
                f"but the registry contains {spec.version!r}"
            )
        if isinstance(self.target, str) and not self.target.strip():
            raise ValueError("target must not be blank")
        return self

    def messages(self) -> list[dict[str, str]]:
        spec = get_prompt(self.component)
        if self.component == PromptComponent.PLANNER:
            return [{"role": "user", "content": spec.render(self.input_text)}]
        return [
            {"role": "system", "content": spec.render()},
            {"role": "user", "content": self.input_text},
        ]

    def target_text(self) -> str:
        if isinstance(self.target, str):
            return self.target
        return json.dumps(self.target, ensure_ascii=False, separators=(",", ":"))


def load_sft_jsonl(path: Path) -> list[SFTExample]:
    """Load and strictly validate a UTF-8 JSONL Teacher dataset."""

    examples: list[SFTExample] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        try:
            example = SFTExample.model_validate(payload)
        except Exception as exc:
            raise ValueError(f"{path}:{line_number}: invalid SFT example: {exc}") from exc
        if example.example_id in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate example_id {example.example_id!r}")
        seen_ids.add(example.example_id)
        examples.append(example)
    if not examples:
        raise ValueError(f"{path}: contains no SFT examples")
    return examples
