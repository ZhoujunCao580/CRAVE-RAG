"""Frozen model-facing prompt for the Evidence Checker."""

from __future__ import annotations

from softdoc.prompts import load_prompt_text


CHECKER_PROMPT_VERSION = "checker-v2.1"


CHECKER_SYSTEM_PROMPT = load_prompt_text("checker_v2_1.txt").removesuffix("\n")
