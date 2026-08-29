import json
from pathlib import Path

from pydantic import TypeAdapter

from scripts.export_prompt_schemas import export_schemas
from softdoc.answering import AnswerInput
from softdoc.controller import ControllerInput
from softdoc.planning.models import PlannerDraft
from softdoc.reading_state import EvidenceCheckInput
from softdoc.visual_reading import VisualReadRequest


ROOT = Path(__file__).parents[1]
EVAL_ROOT = ROOT / "evals" / "prompts"


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_prompt_suites_are_split_and_inputs_validate() -> None:
    suites = {
        "planner": ("planner_cases_v1.jsonl", "planner_gold_v1.jsonl", None),
        "controller": (
            "controller_cases_v1.jsonl",
            "controller_gold_v1.jsonl",
            TypeAdapter(ControllerInput),
        ),
        "visual_reader": (
            "visual_reader_cases_v1.jsonl",
            "visual_reader_gold_v1.jsonl",
            TypeAdapter(VisualReadRequest),
        ),
        "checker": (
            "checker_cases_v1.jsonl",
            "checker_gold_v1.jsonl",
            TypeAdapter(EvidenceCheckInput),
        ),
        "answerer": (
            "answerer_cases_v1.jsonl",
            "answerer_gold_v1.jsonl",
            TypeAdapter(AnswerInput),
        ),
    }
    payload_keys = {
        "controller": "controller_input",
        "visual_reader": "request",
        "checker": "checker_input",
        "answerer": "answer_input",
    }

    for component, (input_name, gold_name, adapter) in suites.items():
        model_rows = _jsonl(EVAL_ROOT / component / "model_inputs" / input_name)
        gold_rows = _jsonl(EVAL_ROOT / component / "review_only" / gold_name)
        assert model_rows
        assert {row["case_id"] for row in model_rows} == {
            row["case_id"] for row in gold_rows
        }
        assert len({row["case_id"] for row in model_rows}) == len(model_rows)
        for row in model_rows:
            assert "split" not in row
            assert not ({"expected", "gold", "label", "score"} & set(row))
            if component == "planner":
                PlannerDraft(original_question=str(row["question"]), subquestions=[])
            else:
                assert adapter is not None
                adapter.validate_python(row[payload_keys[component]])


def test_visual_reader_case_assets_exist() -> None:
    rows = _jsonl(
        EVAL_ROOT / "visual_reader" / "model_inputs" / "visual_reader_cases_v1.jsonl"
    )
    for row in rows:
        request = VisualReadRequest.model_validate(row["request"])
        for visual_input in request.visual_inputs:
            path = visual_input.page_image_path
            if not path.is_absolute():
                path = ROOT / path
            assert path.is_file(), f"Missing visual asset for {row['case_id']}: {path}"


def test_exported_output_schemas_match_runtime(tmp_path: Path) -> None:
    written = export_schemas(tmp_path)
    assert {path.name for path in written} == {
        "planner_output.schema.json",
        "visual_reader_output.schema.json",
        "checker_output.schema.json",
        "controller_output.schema.json",
        "answerer_output.schema.json",
    }
    assert all(json.loads(path.read_text(encoding="utf-8")) for path in written)
