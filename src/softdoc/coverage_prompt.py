"""Frozen prompt for per-item semantic coverage decisions."""

from __future__ import annotations

from softdoc.prompts import load_prompt_text


COVERAGE_CHECKER_PROMPT_VERSION = "coverage-checker-v0.1"
COVERAGE_CHECKER_SYSTEM_PROMPT = load_prompt_text(
    "coverage_checker_v0_1.txt"
).removesuffix("\n")
