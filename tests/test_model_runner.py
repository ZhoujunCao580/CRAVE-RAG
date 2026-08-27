from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from softdoc.answering import AnswerInput, AnswerResult
from softdoc.controller import ControllerInput
from softdoc.model_runner import ModelBackedRunner, write_model_pipeline_run
from softdoc.models import Document, Element, ElementType, Page, Provenance
from softdoc.planning.models import InitialPlan, PlannedSubQuestion, PlannerTrace
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


class FixedPlanner:
    def create_plan(self, question: str) -> InitialPlan:
        return InitialPlan(
            original_question=question,
            subquestions=[
                PlannedSubQuestion(
                    subquestion_id="Q1",
                    text=question,
                    depends_on=[],
                )
            ],
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
    )

    assert run.reading_run.status == ReadingRunStatus.READY
    assert run.reading_run.answer is not None
    assert run.reading_run.answer.answer == "Revenue in 2023 was 12 million."
    assert [item.component for item in run.stage_calls] == [
        "planner",
        "controller",
        "controller",
        "reader",
        "checker",
        "answerer",
    ]

    output = tmp_path / "run"
    write_model_pipeline_run(run, output)
    assert (output / "planner.json").is_file()
    assert len((output / "controller_calls.jsonl").read_text().splitlines()) == 2
    assert len((output / "reader_calls.jsonl").read_text().splitlines()) == 1
    assert len((output / "checker_calls.jsonl").read_text().splitlines()) == 1
    assert len((output / "answerer_calls.jsonl").read_text().splitlines()) == 1
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ready"
    assert manifest["call_counts"]["controller"] == 2
