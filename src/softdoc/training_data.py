"""Version-bound supervised-training records for model-facing components."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Self

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

    def training_messages(self) -> list[dict[str, str]]:
        """Materialize one complete supervised conversation.

        The internal :class:`SFTExample` keeps provenance fields separate from
        model-facing content.  This method creates the standard OpenAI-style
        ``messages`` sequence consumed by LLaMA-Factory and similar trainers.
        """

        return [
            *self.messages(),
            {"role": "assistant", "content": self.target_text()},
        ]


class SFTMessage(BaseModel):
    """One role/content pair in an exported model-facing conversation."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class OpenAIMessagesSFTRecord(BaseModel):
    """LLaMA-Factory-compatible Controller SFT row."""

    model_config = ConfigDict(extra="forbid")

    messages: list[SFTMessage] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_controller_turn(self) -> Self:
        roles = [message.role for message in self.messages]
        if roles != ["system", "user", "assistant"]:
            raise ValueError(
                "Controller messages must be exactly system, user, assistant"
            )
        return self


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


def load_openai_messages_sft_jsonl(path: Path) -> list[OpenAIMessagesSFTRecord]:
    """Load the model-facing JSONL exported for LLaMA-Factory."""

    records: list[OpenAIMessagesSFTRecord] = []
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw_line.strip():
            continue
        try:
            records.append(OpenAIMessagesSFTRecord.model_validate_json(raw_line))
        except Exception as exc:
            raise ValueError(
                f"{path}:{line_number}: invalid OpenAI messages SFT record: {exc}"
            ) from exc
    if not records:
        raise ValueError(f"{path}: contains no OpenAI messages SFT records")
    return records
