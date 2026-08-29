"""Single discovery and audit entry point for all model-facing prompts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Callable

from softdoc.answering import ANSWERER_PROMPT_VERSION, ANSWERER_SYSTEM_PROMPT
from softdoc.checking_prompt import CHECKER_PROMPT_VERSION, CHECKER_SYSTEM_PROMPT
from softdoc.controller_prompt import CONTROLLER_PROMPT_VERSION, CONTROLLER_SYSTEM_PROMPT
from softdoc.planning.prompt import (
    INITIAL_PLANNER_PROMPT_VERSION,
    build_initial_planner_prompt,
)
from softdoc.visual_reading import (
    VISUAL_READER_PROMPT_VERSION,
    VISUAL_READER_SYSTEM_PROMPT,
)


class PromptComponent(StrEnum):
    PLANNER = "planner"
    VISUAL_READER = "visual_reader"
    CHECKER = "checker"
    ANSWERER = "answerer"
    CONTROLLER = "controller"


@dataclass(frozen=True)
class PromptSpec:
    component: PromptComponent
    version: str
    prompt_kind: str
    source_module: str
    canonical_text: str
    renderer: Callable[[str], str] | None = None

    @property
    def sha256(self) -> str:
        return sha256(self.canonical_text.encode("utf-8")).hexdigest()

    def render(self, input_text: str | None = None) -> str:
        if self.renderer is None:
            if input_text is not None:
                raise ValueError(f"{self.component} is a system prompt and takes no input text")
            return self.canonical_text
        if input_text is None or not input_text.strip():
            raise ValueError(f"{self.component} requires non-blank input text")
        return self.renderer(input_text)

    def manifest(self) -> dict[str, str | int]:
        return {
            "component": self.component.value,
            "version": self.version,
            "prompt_kind": self.prompt_kind,
            "source_module": self.source_module,
            "sha256": self.sha256,
            "character_count": len(self.canonical_text),
        }


_PLANNER_SENTINEL = "<ROOT_QUESTION>"


PROMPT_REGISTRY: dict[PromptComponent, PromptSpec] = {
    PromptComponent.PLANNER: PromptSpec(
        component=PromptComponent.PLANNER,
        version=INITIAL_PLANNER_PROMPT_VERSION,
        prompt_kind="system_and_user_prompt",
        source_module="softdoc.planning.prompt",
        canonical_text=build_initial_planner_prompt(_PLANNER_SENTINEL),
        renderer=build_initial_planner_prompt,
    ),
    PromptComponent.VISUAL_READER: PromptSpec(
        component=PromptComponent.VISUAL_READER,
        version=VISUAL_READER_PROMPT_VERSION,
        prompt_kind="system_prompt",
        source_module="softdoc.visual_reading",
        canonical_text=VISUAL_READER_SYSTEM_PROMPT,
    ),
    PromptComponent.CHECKER: PromptSpec(
        component=PromptComponent.CHECKER,
        version=CHECKER_PROMPT_VERSION,
        prompt_kind="system_prompt",
        source_module="softdoc.checking_prompt",
        canonical_text=CHECKER_SYSTEM_PROMPT,
    ),
    PromptComponent.ANSWERER: PromptSpec(
        component=PromptComponent.ANSWERER,
        version=ANSWERER_PROMPT_VERSION,
        prompt_kind="system_prompt",
        source_module="softdoc.answering",
        canonical_text=ANSWERER_SYSTEM_PROMPT,
    ),
    PromptComponent.CONTROLLER: PromptSpec(
        component=PromptComponent.CONTROLLER,
        version=CONTROLLER_PROMPT_VERSION,
        prompt_kind="system_prompt",
        source_module="softdoc.controller_prompt",
        canonical_text=CONTROLLER_SYSTEM_PROMPT,
    ),
}


def get_prompt(component: PromptComponent | str) -> PromptSpec:
    """Return one frozen prompt specification by stable component name."""

    return PROMPT_REGISTRY[PromptComponent(component)]


def prompt_manifest() -> list[dict[str, str | int]]:
    """Return a deterministic, serializable manifest for experiment logging."""

    return [PROMPT_REGISTRY[item].manifest() for item in PromptComponent]
