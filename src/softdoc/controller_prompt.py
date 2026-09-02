"""Frozen model-facing prompt for the Reading Controller v0."""

from __future__ import annotations

from softdoc.controller import ControllerInput

from softdoc.prompts import load_prompt_text


# Frozen after the first Controller policy review. Semantic changes require a
# new version and evaluation; do not patch this prompt for individual cases.
CONTROLLER_PROMPT_VERSION = "controller-policy-v0.8"


CONTROLLER_SYSTEM_PROMPT = load_prompt_text("controller_policy_v0_8.txt")


def build_controller_user_prompt(controller_input: ControllerInput) -> str:
    """Serialize the validated state as the sole user message."""

    return controller_input.model_dump_json(indent=2)
