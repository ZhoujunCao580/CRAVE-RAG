"""Deterministic prompt construction for the initial planner."""

from __future__ import annotations

import json

from softdoc.prompts import load_prompt_text


# Frozen after the Conservative + Deferred design review. Any semantic change
# requires an explicit new version and a new evaluation; do not patch this
# prompt in response to individual dataset errors.
INITIAL_PLANNER_PROMPT_VERSION = "planner-v0.20"
_PLANNER_SYSTEM_TEMPLATE = load_prompt_text("planner_v0_20.txt")


def build_initial_planner_system_prompt(
    *,
    max_subquestions: int = 6,
    max_depth: int = 4,
) -> str:
    """Render the stable Planner instructions for the system message."""

    if max_subquestions < 0:
        raise ValueError("max_subquestions must be non-negative")
    if max_depth < 1:
        raise ValueError("max_depth must include the Root at depth 1")
    return _PLANNER_SYSTEM_TEMPLATE.replace(
        "<<MAX_SUBQUESTIONS>>", str(max_subquestions)
    ).replace("<<MAX_DEPTH>>", str(max_depth))


def build_initial_planner_user_prompt(question: str) -> str:
    """Serialize the original question once as the dynamic user message."""

    stripped = question.strip()
    if not stripped:
        raise ValueError("The Planner question must not be blank")
    return json.dumps(
        {"original_question": stripped},
        ensure_ascii=False,
        indent=2,
    )


def build_initial_planner_prompt(
    question: str,
    *,
    max_subquestions: int = 6,
    max_depth: int = 4,
) -> str:
    """Render an auditable view of the system and user Planner messages."""

    return (
        build_initial_planner_system_prompt(
            max_subquestions=max_subquestions,
            max_depth=max_depth,
        )
        + "\n\n# User message\n\n"
        + build_initial_planner_user_prompt(question)
    )
