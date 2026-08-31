from __future__ import annotations

import pytest
from pydantic import ValidationError

from softdoc.teacher_data import CheckerReview, TeacherReview


def _review(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "teacher-review-v0",
        "reading_session_id": "reading:1",
        "episode_status": "pending",
        "controller_steps": [
            {
                "controller_call_index": 0,
                "action_id": "action:1",
                "training_label_status": "pending",
                "review_note": None,
            }
        ],
        "first_corrupted_action_id": None,
    }
    payload.update(updates)
    return payload


def test_finalized_review_cannot_leave_pending_controller_steps() -> None:
    with pytest.raises(ValidationError, match="cannot contain pending"):
        TeacherReview.model_validate(_review(episode_status="rejected"))


def test_rejected_controller_step_requires_a_note() -> None:
    payload = _review(episode_status="rejected")
    payload["controller_steps"][0]["training_label_status"] = "rejected"  # type: ignore[index]
    with pytest.raises(ValidationError, match="requires review_note"):
        TeacherReview.model_validate(payload)


def test_first_corrupted_action_must_be_a_rejected_reviewed_step() -> None:
    payload = _review(episode_status="accepted", first_corrupted_action_id="action:1")
    payload["controller_steps"][0]["training_label_status"] = "accepted"  # type: ignore[index]
    with pytest.raises(ValidationError, match="must identify a rejected step"):
        TeacherReview.model_validate(payload)


def test_checker_replacement_output_must_match_action() -> None:
    payload = {
        "schema_version": "checker-review-v0",
        "reading_session_id": "reading:1",
        "episode_status": "accepted",
        "checker_steps": [
            {
                "checker_call_index": 0,
                "action_id": "action:1",
                "training_label_status": "accepted",
                "review_note": "Corrected Teacher delta.",
                "replacement_output": {
                    "action_id": "action:other",
                    "observation_assessments": [],
                    "evidence_updates": {"add": [], "replace": [], "remove": []},
                    "current_target_status": "incomplete",
                    "root_status": "incomplete",
                    "remaining_gap_description": "The value remains unknown.",
                },
            }
        ],
    }
    with pytest.raises(ValidationError, match="action_id must match"):
        CheckerReview.model_validate(payload)
