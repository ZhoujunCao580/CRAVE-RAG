from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from softdoc.answering import AnswerInput, AnswerResult
from softdoc.controller import ControllerInput
from softdoc.model_runner import (
    ModelBackedRunner,
    load_model_pipeline_run,
    write_model_pipeline_run,
)
from softdoc.models import Document, Element, ElementType, Page, Provenance
from softdoc.planning.models import InitialPlan, PlannerTrace
from softdoc.reading_environment import (
    DeterministicContentReader,
    ReadingEnvironmentConfig,
    ReadingRunStatus,
)
from softdoc.reading_state import (
    EvidenceAddition,
    EvidenceCheckInput,
    EvidenceCheckResult,
    EvidenceStatus,
    EvidenceUpdates,
    ObservationAssessment,
    QuestionStatus,
)
from softdoc.teacher_data import (
    CheckerReview,
    ReviewStatus,
    TeacherReview,
    build_checker_review_template,
    build_teacher_review_template,
    load_checker_reviewed_run,
    load_reviewed_run,
    write_checker_review,
    write_checker_sft_dataset,
    write_controller_sft_dataset,
    write_teacher_review,
)
from softdoc.training_data import load_openai_messages_sft_jsonl, load_sft_jsonl


class FixedPlanner:
    def create_plan(self, question: str) -> InitialPlan:
        return InitialPlan(
            original_question=question,
            subquestions=[],
            planner_trace=PlannerTrace(
                backend_name="fake",
                model="fake",
                prompt_version="test",
            ),
        )


class SearchThenReadController:
    def decide(self, controller_input: ControllerInput) -> dict[str, Any]:
        if controller_input.visible_search_view is None:
            return {
                "action": "SEARCH",
                "operation": "new",
                "query": controller_input.current_gap.description,
                "search_session_id": None,
            }
        return {
            "action": "READ_SOURCE",
            "source_ids": [
                controller_input.visible_search_view.candidate_previews[0].element_id
            ],
            "local_problem": controller_input.current_gap.description,
        }


class AcceptingChecker:
    def check(self, checker_input: EvidenceCheckInput) -> EvidenceCheckResult:
        observation = checker_input.observations[0]
        target = checker_input.evidence_memory.current_target
        assert target is not None
        return EvidenceCheckResult(
            action_id=checker_input.action_id,
            observation_assessments=[
                ObservationAssessment(
                    observation_id=observation.observation_id,
                    used_for_evidence=True,
                    assessment="The observation directly resolves the target.",
                )
            ],
            evidence_updates=EvidenceUpdates(
                add=[
                    EvidenceAddition(
                        statement=observation.text,
                        observation_ids=[observation.observation_id],
                        supports_question_ids=[target.question_id],
                    )
                ]
            ),
            current_target_status=QuestionStatus.SATISFIED,
            root_status=EvidenceStatus.READY,
            remaining_gap_description=None,
        )


class EvidenceAnswerer:
    def answer(self, answer_input: AnswerInput) -> AnswerResult:
        return AnswerResult(
            answer=answer_input.evidence[0].statement,
            used_evidence_ids=[answer_input.evidence[0].evidence_id],
        )


def _provenance(owner: str) -> Provenance:
    return Provenance(
        provenance_id=f"prov:{owner}",
        adapter="test",
        source_path=Path("fixture.json"),
        source_locator=owner,
    )


def _document() -> Document:
    doc_id = "doc:runner"
    element = Element(
        element_id="paragraph:1",
        document_id=doc_id,
        page_id="page:1",
        page_number=1,
        element_type=ElementType.PARAGRAPH,
        reading_order=0,
        text="Revenue in 2023 was 12 million.",
        provenance=_provenance("paragraph:1"),
    )
    page = Page(
        page_id="page:1",
        document_id=doc_id,
        page_index=0,
        page_number=1,
        width=100,
        height=100,
        element_ids=[element.element_id],
        reading_order=[element.element_id],
        provenance=_provenance("page:1"),
    )
    return Document(
        document_id=doc_id,
        source_path=Path("report.pdf"),
        pages=[page],
        elements=[element],
        provenance=_provenance(doc_id),
    )


def test_model_runner_records_every_executed_stage_and_writes_artifacts(tmp_path: Path) -> None:
    runner = ModelBackedRunner(
        planner=FixedPlanner(),
        controller=SearchThenReadController(),
        reader=DeterministicContentReader(),
        checker=AcceptingChecker(),
        answerer=EvidenceAnswerer(),
        environment_config=ReadingEnvironmentConfig(action_budget=4),
    )
    run = runner.run(
        document=_document(),
        asset_root=tmp_path,
        question="What was revenue in 2023?",
        question_id="root:model-run",
    )

    assert run.reading_run.status == ReadingRunStatus.READY
    assert run.plan.subquestions == []
    assert run.reading_run.evidence_memory.questions == []
    assert run.reading_run.evidence_memory.evidence[0].supports_question_ids == [
        "root:model-run"
    ]
    assert run.reading_run.answer is not None
    assert run.reading_run.answer.answer == "Revenue in 2023 was 12 million."
    assert all(item.elapsed_seconds is not None for item in run.stage_calls)
    assert [item.component for item in run.stage_calls] == [
        "planner",
        "controller",
        "controller",
        "reader",
        "checker",
        "answerer",
    ]
    controller_calls = [
        item for item in run.stage_calls if item.component == "controller"
    ]
    assert [item.action_id for item in controller_calls] == [
        item.action_id for item in run.reading_run.action_trace.entries
    ]
    reader_call = next(item for item in run.stage_calls if item.component == "reader")
    checker_call = next(item for item in run.stage_calls if item.component == "checker")
    assert reader_call.action_id == controller_calls[1].action_id
    assert checker_call.action_id == controller_calls[1].action_id

    output = tmp_path / "run"
    write_model_pipeline_run(run, output)
    assert (output / "planner.json").is_file()
    assert (output / "planner_calls.jsonl").is_file()
    assert (output / "candidate_batches.jsonl").is_file()
    assert (output / "action_trace.json").is_file()
    assert (output / "evidence_deltas.jsonl").is_file()
    assert len((output / "controller_calls.jsonl").read_text().splitlines()) == 2
    assert len((output / "reader_calls.jsonl").read_text().splitlines()) == 1
    assert len((output / "checker_calls.jsonl").read_text().splitlines()) == 1
    assert len((output / "answerer_calls.jsonl").read_text().splitlines()) == 1
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ready"
    assert manifest["call_counts"]["controller"] == 2
    assert [item["component"] for item in manifest["stage_call_order"]] == [
        item.component for item in run.stage_calls
    ]
    assert any(item["component"] == "controller" for item in manifest["prompts"])

    reloaded = load_model_pipeline_run(output)
    assert reloaded.model_dump(mode="json") == run.model_dump(mode="json")

    interleaved = run.model_copy(
        update={
            "stage_calls": [
                run.stage_calls[0],
                run.stage_calls[1],
                run.stage_calls[3],
                run.stage_calls[2],
                run.stage_calls[4],
                run.stage_calls[5],
            ]
        }
    )
    interleaved_output = tmp_path / "interleaved-run"
    write_model_pipeline_run(interleaved, interleaved_output)
    assert load_model_pipeline_run(interleaved_output).stage_calls == (
        interleaved.stage_calls
    )

    template = build_teacher_review_template(reloaded)
    assert template.episode_status == ReviewStatus.PENDING
    assert len(template.controller_steps) == 2
    review_payload = template.model_dump(mode="json")
    review_payload["episode_status"] = "accepted"
    for step in review_payload["controller_steps"]:
        step["training_label_status"] = "accepted"
        step["review_note"] = "The action is valid and advances the current gap."
    review = TeacherReview.model_validate(review_payload)
    write_teacher_review(review, output / "teacher_review.json")

    reviewed_run = load_reviewed_run(output)
    dataset_dir = tmp_path / "controller_dataset"
    dataset_manifest = write_controller_sft_dataset([reviewed_run], dataset_dir)
    assert dataset_manifest.example_count == 2
    examples = load_sft_jsonl(dataset_dir / "controller_sft.jsonl")
    assert [item.target["action"] for item in examples] == ["SEARCH", "READ_SOURCE"]
    assert json.loads(examples[0].input_text)["current_gap"]["question_id"] == (
        "root:model-run"
    )
    assert json.loads(
        (dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    )["generation_protocol"] == "teacher-no-gold-v0"
    message_records = load_openai_messages_sft_jsonl(
        dataset_dir / "controller_sft_messages.jsonl"
    )
    assert len(message_records) == 2
    assert [item.role for item in message_records[0].messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert json.loads(message_records[0].messages[1].content)["current_gap"][
        "question_id"
    ] == "root:model-run"
    assert json.loads(message_records[0].messages[2].content)["action"] == "SEARCH"
    dataset_info = json.loads(
        (dataset_dir / "dataset_info.json").read_text(encoding="utf-8")
    )
    assert dataset_info["crave_controller_sft"]["columns"] == {
        "messages": "messages"
    }

    checker_template = build_checker_review_template(reloaded)
    assert len(checker_template.checker_steps) == 1
    checker_payload = checker_template.model_dump(mode="json")
    checker_payload["episode_status"] = "accepted"
    checker_payload["checker_steps"][0]["training_label_status"] = "accepted"
    checker_payload["checker_steps"][0]["review_note"] = (
        "The delta is valid and closes the Root target."
    )
    checker_review = CheckerReview.model_validate(checker_payload)
    write_checker_review(checker_review, output / "checker_review.json")

    checker_dataset_dir = tmp_path / "checker_dataset"
    checker_manifest = write_checker_sft_dataset(
        [load_checker_reviewed_run(output)], checker_dataset_dir
    )
    assert checker_manifest.example_count == 1
    checker_examples = load_sft_jsonl(checker_dataset_dir / "checker_sft.jsonl")
    assert checker_examples[0].component.value == "checker"
    assert checker_examples[0].target["root_status"] == "ready"
    checker_message_records = load_openai_messages_sft_jsonl(
        checker_dataset_dir / "checker_sft_messages.jsonl"
    )
    assert json.loads(checker_message_records[0].messages[2].content)[
        "current_target_status"
    ] == "satisfied"
