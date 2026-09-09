"""Frozen prompt for question-directed page-batch visual scans."""

from __future__ import annotations

import json

from softdoc.prompts import load_prompt_text
from softdoc.visual_scan import VisualScanBatchInput


VISUAL_SCAN_PROMPT_VERSION = "visual-scan-v0.5"
VISUAL_SCAN_SYSTEM_PROMPT = load_prompt_text("visual_scan_v0_1.txt")


def build_visual_scan_user_prompt(scan_input: VisualScanBatchInput) -> str:
    """Serialize only text metadata; page images are attached out of band."""

    return json.dumps(scan_input.model_dump(mode="json"), ensure_ascii=False, indent=2)
