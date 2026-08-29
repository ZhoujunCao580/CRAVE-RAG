import json
from pathlib import Path

import pytest

from scripts.validate_prompt_run import validate_run


def test_validate_prompt_run_records_valid_and_invalid_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "raw_outputs.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "case_id": "P1",
                        "raw_output": {
                            "original_question": "What is Figure 3's title?",
                            "subquestions": [],
                        },
                    }
                ),
                json.dumps({"case_id": "P2", "raw_output": {"unexpected": True}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary = validate_run("planner", input_path, tmp_path / "report")
    assert summary == {"case_count": 2, "schema_valid": 1}
    rows = [
        json.loads(line)
        for line in (tmp_path / "report" / "schema_validation.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["schema_valid"] for row in rows] == [True, False]


def test_validate_prompt_run_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    row = json.dumps(
        {
            "case_id": "A1",
            "raw_output": {"answer": "12", "used_evidence_ids": ["E1"]},
        }
    )
    input_path = tmp_path / "raw_outputs.jsonl"
    input_path.write_text(f"{row}\n{row}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate case_id"):
        validate_run("answerer", input_path, tmp_path / "report")
