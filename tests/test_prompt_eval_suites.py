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


def test_visual_reader_case_assets_are_tracked_or_explicitly_external() -> None:
    dependency_path = (
        EVAL_ROOT / "visual_reader" / "external_data_dependencies.json"
    )
    dependency_payload = json.loads(dependency_path.read_text(encoding="utf-8"))
    assert dependency_payload["schema_version"] == "prompt-eval-external-data-v0.1"
    prefixes = {
        item["path_prefix"]: item["dataset_id"]
        for item in dependency_payload["datasets"]
    }
    assert prefixes
    rows = _jsonl(
        EVAL_ROOT / "visual_reader" / "model_inputs" / "visual_reader_cases_v1.jsonl"
    )
    external_dependencies: set[str] = set()
    for row in rows:
        request = VisualReadRequest.model_validate(row["request"])
        for visual_input in request.visual_inputs:
            path = visual_input.page_image_path
            assert not path.is_absolute(), (
                f"Visual eval paths must be repository-relative: {row['case_id']}: {path}"
            )
            relative = path.as_posix()
            matched = [
                dataset_id
                for prefix, dataset_id in prefixes.items()
                if relative == prefix or relative.startswith(prefix + "/")
            ]
            if matched:
                assert len(matched) == 1
                external_dependencies.add(matched[0])
                continue
            resolved = (ROOT / path).resolve()
            assert resolved.is_relative_to(ROOT.resolve())
            assert resolved.is_file(), (
                f"Missing tracked visual asset for {row['case_id']}: {resolved}"
            )
    assert external_dependencies == set(prefixes.values())


def test_exported_output_schemas_match_runtime(tmp_path: Path) -> None:
    written = export_schemas(tmp_path)
    assert {path.name for path in written} == {
        "planner_output.schema.json",
        "visual_retrieval_output.schema.json",
        "visual_reader_output.schema.json",
        "checker_output.schema.json",
        "controller_output.schema.json",
        "answerer_output.schema.json",
    }
    assert all(json.loads(path.read_text(encoding="utf-8")) for path in written)
