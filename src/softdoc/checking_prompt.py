"""Frozen model-facing prompt for the Evidence Checker."""

from __future__ import annotations

from softdoc.prompts import load_prompt_text


CHECKER_PROMPT_VERSION = "checker-v1.9"


CHECKER_SYSTEM_PROMPT = load_prompt_text("checker_v1_9.txt").removesuffix("\n")
