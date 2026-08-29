"""Export the exact model-output schemas used by Prompt evaluations."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from softdoc.answering import AnswerResult
from softdoc.controller import ControllerAction
from softdoc.planning.models import PlannerDraft
from softdoc.reading_state import EvidenceCheckResult
from softdoc.visual_reading import VisualReadResult


OUTPUT_DIR = Path("evals/prompts/schemas")


def export_schemas(output_dir: Path = OUTPUT_DIR) -> list[Path]:
    schemas = {
        "planner_output.schema.json": PlannerDraft.model_json_schema(),
        "visual_reader_output.schema.json": VisualReadResult.model_json_schema(),
        "checker_output.schema.json": EvidenceCheckResult.model_json_schema(),
        "controller_output.schema.json": TypeAdapter(ControllerAction).json_schema(),
        "answerer_output.schema.json": AnswerResult.model_json_schema(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, schema in schemas.items():
        path = output_dir / filename
        path.write_text(
            json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    return written


if __name__ == "__main__":
    for exported_path in export_schemas():
        print(exported_path)
