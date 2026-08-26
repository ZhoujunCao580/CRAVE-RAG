"""Run two compact, replayable ReadingEnvironment v0 probes on real SoftDocs.

This is an integration audit, not an accuracy benchmark.  Scripted teacher
decisions stand in for unfinished Controller/Checker/Answerer models so the
state transitions and source grounding can be tested independently of model
quality.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

from softdoc.answering import AnswerInput, AnswerResult
from softdoc.controller import ControllerAction, ControllerInput
from softdoc.models import Document, RelationStatus, RelationType
from softdoc.reading_environment import (
    DeterministicContentReader,
    ReadingEnvironment,
    ReadingRunResult,
)
from softdoc.reading_state import (
    EvidenceAddition,
    EvidenceCheckInput,
    EvidenceCheckResult,
    EvidenceStatus,
    EvidenceUpdates,
    ObservationAssessment,
    QuestionStatus,
    RootQuestion,
)
from softdoc.retrieval import ExactAnchorLookup, SubQuestionInput


class SearchTeacherController:
    def decide(self, value: ControllerInput) -> ControllerAction | dict[str, Any]:
        if value.visible_search_view is None:
            return {
                "action": "SEARCH",
                "operation": "new",
                "query": "commanding officer first figure second page",
            }
        for preview in value.visible_search_view.candidate_previews:
            snippet = preview.matched_snippet.lower()
            if "john" in snippet and "commanding officer" in snippet:
                return {
                    "action": "READ_SOURCE",
                    "source_ids": [preview.element_id],
                    "local_problem": "Identify the named commanding officer.",
                }
        tab = next(
            item
            for item in value.search_tabs
            if item.search_session_id == value.visible_search_view.search_session_id
        )
        if tab.has_more:
            return {
                "action": "SEARCH",
                "operation": "next",
                "search_session_id": tab.search_session_id,
            }
        raise RuntimeError("Teacher could not find the grounded name candidate")


class RelationTeacherController:
    def decide(self, value: ControllerInput) -> ControllerAction | dict[str, Any]:
        if not value.confirmed_relations:
            raise RuntimeError("Exact Figure read exposed no confirmed relation")
        relation = next(
            (
                item
                for item in value.confirmed_relations
                if item.relation_type == RelationType.CAPTION_OF
            ),
            value.confirmed_relations[0],
        )
        return {
            "action": "FOLLOW_RELATION",
            "relation_id": relation.relation_id,
            "local_problem": "Read the Figure caption as a grounded description.",
        }


class TeacherChecker:
    def __init__(self, statement: Callable[[str], str | None]) -> None:
        self.statement = statement

    def check(self, value: EvidenceCheckInput) -> EvidenceCheckResult:
        accepted: list[tuple[str, str]] = []
        for observation in value.observations:
            statement = self.statement(observation.text)
            if statement:
                accepted.append((observation.observation_id, statement))
        accepted_ids = {item[0] for item in accepted}
        target = value.evidence_memory.current_target
        assert target is not None
        return EvidenceCheckResult(
            action_id=value.action_id,
            observation_assessments=[
                ObservationAssessment(
                    observation_id=item.observation_id,
                    used_for_evidence=item.observation_id in accepted_ids,
                    assessment=(
                        "Grounded and useful for the current gap."
                        if item.observation_id in accepted_ids
                        else "Readable but not useful for the current gap."
                    ),
                )
                for item in value.observations
            ],
            evidence_updates=EvidenceUpdates(
                add=[
                    EvidenceAddition(
                        statement=statement,
                        observation_ids=[observation_id],
                        supports_question_ids=[target.question_id],
                    )
                    for observation_id, statement in accepted
                ]
            ),
            current_target_status=(
                QuestionStatus.SATISFIED
                if accepted
                else QuestionStatus.INCOMPLETE
            ),
            root_status=(
                EvidenceStatus.READY if accepted else EvidenceStatus.INCOMPLETE
            ),
            remaining_gap_description=(None if accepted else target.gap_description),
        )


class TeacherAnswerer:
    def __init__(self, answer: Callable[[AnswerInput], str]) -> None:
        self.answer_text = answer

    def answer(self, value: AnswerInput) -> AnswerResult:
        return AnswerResult(
            answer=self.answer_text(value),
            used_evidence_ids=[item.evidence_id for item in value.evidence],
        )


def _load(path: Path) -> Document:
    return Document.model_validate_json(path.read_text(encoding="utf-8"))


def _find_figure_case(root: Path) -> tuple[Path, Document, str]:
    exact = ExactAnchorLookup()
    for path in sorted(root.glob("*/document.json")):
        document = _load(path)
        elements = {item.element_id: item for item in document.elements}
        for relation in document.relations:
            if not (
                relation.status == RelationStatus.CONFIRMED
                and relation.relation_type == RelationType.CAPTION_OF
            ):
                continue
            source = elements.get(relation.source_id)
            target = elements.get(relation.target_id)
            if source is None or target is None or not (source.text or "").strip():
                continue
            if target.visual_asset_path is None:
                continue
            visual_path = Path(target.visual_asset_path)
            if not visual_path.is_absolute():
                visual_path = path.parent / visual_path
            if not visual_path.is_file():
                continue
            labels: list[str] = []
            if target.reference_label:
                labels.append(target.reference_label.strip())
            match = re.search(
                r"(?<!\w)(?:Figure|Fig\.?)\s*([0-9]+)",
                source.text or "",
                flags=re.IGNORECASE,
            )
            if match:
                labels.append(match.group(1))
            for label in dict.fromkeys(labels):
                question = (
                    label if label.lower().startswith("fig") else f"Figure {label}"
                )
                lookup = exact.lookup(
                    SubQuestionInput(subquestion_id="probe:figure", text=question),
                    document,
                )
                if (
                    len(lookup.exact_anchor_matches) == 1
                    and lookup.exact_anchor_matches[0].target_id == target.element_id
                ):
                    return path, document, question
    raise RuntimeError("No real Figure + confirmed caption_of + unique label case found")


def _compact(result: ReadingRunResult, *, scenario: str, document_name: str) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "document": document_name,
        "status": result.status.value,
        "actions": [
            {
                "step": item.step_index,
                "action": item.action_name,
                "status": item.execution_status.value,
                "targets": item.target_ids,
                "feedback": item.metadata.get("controller_feedback"),
            }
            for item in result.action_trace.entries
        ],
        "evidence": [
            {
                "evidence_id": item.evidence_id,
                "statement": item.statement,
                "observation_ids": item.observation_ids,
            }
            for item in result.evidence_memory.evidence
        ],
        "answer": result.answer.answer if result.answer else None,
        "diagnostics": [item.model_dump(mode="json") for item in result.diagnostics],
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--softdoc-root",
        type=Path,
        default=Path("data/processed/representative_28/softdoc"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".runlogs/reading_environment_v0_real.json"),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Write the JSON report without printing the full payload.",
    )
    args = parser.parse_args()

    search_path = args.softdoc_root / "0b85477387a9d0cc33fca0f4becaa0e5" / "document.json"
    search_document = _load(search_path)
    search_result = ReadingEnvironment(
        search_document,
        asset_root=search_path.parent,
        controller=SearchTeacherController(),
        reader=DeterministicContentReader(),
        checker=TeacherChecker(
            lambda text: (
                "The commanding officer is Capt. John W. Sanders."
                if "john" in text.lower() and "commanding officer" in text.lower()
                else None
            )
        ),
        answerer=TeacherAnswerer(lambda value: value.evidence[0].statement),
    ).run(
        root_question=RootQuestion(
            question_id="root:real-search",
            text="Who is the commanding officer in the first figure on the second page?",
        ),
        run_key="real-search",
    )

    figure_path, figure_document, figure_anchor = _find_figure_case(args.softdoc_root)
    figure_result = ReadingEnvironment(
        figure_document,
        asset_root=figure_path.parent,
        controller=RelationTeacherController(),
        reader=DeterministicContentReader(),
        checker=TeacherChecker(lambda text: text.strip() or None),
        answerer=TeacherAnswerer(lambda value: value.evidence[0].statement),
    ).run(
        root_question=RootQuestion(
            question_id="root:real-figure",
            text=f"What does {figure_anchor} show?",
        ),
        run_key="real-figure",
    )

    report = {
        "purpose": (
            "Interface/state-transition audit with scripted teacher decisions; "
            "not an Agent accuracy score."
        ),
        "scenarios": [
            _compact(
                search_result,
                scenario="search_preview_read_checker_answer",
                document_name=search_path.parent.name,
            ),
            _compact(
                figure_result,
                scenario="exact_visual_limitation_follow_relation",
                document_name=figure_path.parent.name,
            ),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not args.quiet:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
