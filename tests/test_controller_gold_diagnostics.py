from __future__ import annotations

import json
from pathlib import Path

from softdoc.controller import ControllerInput, validate_controller_action


FIXTURE = Path(__file__).parent / "fixtures" / "controller_gold_5_diagnostics_v0.json"


def test_fixed_controller_diagnostics_are_valid_and_complete() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert payload["diagnostic_version"] == "controller-gold-5-v0.1"
    assert [case["question_index"] for case in payload["cases"]] == [
        14,
        230,
        546,
        641,
        1007,
    ]
    assert sum(len(case["steps"]) for case in payload["cases"]) == 9

    for case in payload["cases"]:
        assert case["steps"]
        for step in case["steps"]:
            assert step["acceptable_actions"]
            assert step["expected_progress"]
            assert step["teacher_rationale"]
            if step["stage"] == "environment":
                assert step["controller_input"] is None
                assert step["acceptable_actions"][0]["action"] == "AUTO_READ_EXACT"
                continue
            state = ControllerInput.model_validate(step["controller_input"])
            for action in step["acceptable_actions"]:
                validate_controller_action(action, state)
