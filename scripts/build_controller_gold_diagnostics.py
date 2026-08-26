"""Build the fixed five-question Controller diagnostic fixture.

The fixture stores the exact model-visible ControllerInput for every audited
decision point, one or more acceptable next actions, and the observed local
8B action.  It is a diagnostic set, not an answer-accuracy benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from softdoc.controller import (
    ControllerActionFeedback,
    ControllerCandidatePreview,
    ControllerConfirmedRelation,
    ControllerEvidence,
    ControllerGap,
    ControllerInput,
    ControllerObservationAssessment,
    ControllerReadingLocation,
    ControllerRecentAction,
    ControllerSearchTab,
    ControllerVisibleSearchView,
)
from softdoc.models import RelationType
from softdoc.reading_state import (
    ActionExecutionStatus,
    EvidenceStatus,
    ReadingSourceType,
    RootQuestion,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / ".runlogs" / "controller_gold_5"
DESTINATION = ROOT / "tests" / "fixtures" / "controller_gold_5_diagnostics_v0.json"

Q641_TABLE_ID = "doc:2023.acl-long.386:9064dd38:page:0007:element:0006:table:main"
Q641_PARAGRAPH_ID = "doc:2023.acl-long.386:9064dd38:page:0006:element:0007:paragraph:main"
Q641_CAPTION_ID = "doc:2023.acl-long.386:9064dd38:page:0006:element:0003:caption:chart"
Q641_CHART_ID = "doc:2023.acl-long.386:9064dd38:page:0006:element:0003:chart:main"
Q641_RELATION_ID = "rel:refers_to:77e9f2f6df3d"


def main() -> int:
    cases = {
        item["question_index"]: item
        for item in json.loads((SOURCE_DIR / "candidates.json").read_text(encoding="utf-8"))
    }
    initial_results = {
        item["question_index"]: item
        for item in json.loads(
            (SOURCE_DIR / "initial_controller_results.json").read_text(encoding="utf-8")
        )
    }
    followups = {
        item["label"]: item
        for item in json.loads(
            (SOURCE_DIR / "followup_controller_results.json").read_text(encoding="utf-8")
        )
    }
    q641_retry = json.loads(
        (SOURCE_DIR / "followup_q641_retry.json").read_text(encoding="utf-8")
    )[0]
    q641_relation = json.loads(
        (SOURCE_DIR / "q641_relation_step.json").read_text(encoding="utf-8")
    )

    output = {
        "diagnostic_version": "controller-gold-5-v0.1",
        "controller_input_version": "controller-input-v0.1",
        "controller_action_version": "controller-action-v0.1",
        "controller_prompt_version": "controller-policy-v0.2",
        "scope": (
            "Five fixed real-document trajectory diagnostics. Acceptable actions "
            "are next-step teacher labels; this file is not an accuracy estimate."
        ),
        "cases": [
            _q14(cases[14], initial_results[14]),
            _q230(cases[230], initial_results[230]),
            _q546(cases[546], initial_results[546], followups["Q546-after-next-batch"]),
            _q641(cases[641], initial_results[641], q641_retry, q641_relation),
            _q1007(cases[1007], initial_results[1007], followups["Q1007-after-next-batch"]),
        ],
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(DESTINATION)
    print(f"cases={len(output['cases'])}")
    print(f"steps={sum(len(case['steps']) for case in output['cases'])}")
    return 0


def _preview(item: dict[str, Any]) -> ControllerCandidatePreview:
    return ControllerCandidatePreview(
        element_id=item["element_id"],
        element_type=item["element_type"],
        page_id=item["page_id"],
        section_path=item["section_path"],
        matched_snippet=item["matched_snippet"],
        content_availability=item["content_availability"],
    )


def _initial_input(case: dict[str, Any]) -> ControllerInput:
    batch = case["batches"][0]
    return ControllerInput(
        reading_session_id=f"reading:gold-{case['question_index']}",
        root_question=RootQuestion(
            question_id=f"root:{case['question_index']}", text=case["question"]
        ),
        root_status=EvidenceStatus.INCOMPLETE,
        current_gap=ControllerGap(
            question_id=f"root:{case['question_index']}",
            description=case["question"],
        ),
        search_tabs=[
            ControllerSearchTab(
                search_session_id=batch["search_session_id"],
                query=case["question"],
                has_more=not batch["exhausted"],
            )
        ],
        visible_search_view=ControllerVisibleSearchView(
            search_session_id=batch["search_session_id"],
            candidate_previews=[_preview(item) for item in batch["candidate_previews"]],
        ),
        remaining_action_budget=6,
    )


def _after_next_input(case: dict[str, Any]) -> ControllerInput:
    batch = case["batches"][1]
    question_id = f"root:{case['question_index']}"
    previews = [_preview(item) for item in batch["candidate_previews"]]
    return ControllerInput(
        reading_session_id=f"reading:gold-{case['question_index']}",
        root_question=RootQuestion(question_id=question_id, text=case["question"]),
        root_status=EvidenceStatus.INCOMPLETE,
        current_gap=ControllerGap(question_id=question_id, description=case["question"]),
        recent_actions=[
            ControllerRecentAction(
                action_id=f"action:{case['question_index']}:search-next",
                question_id=question_id,
                action_name="SEARCH",
                target_ids=[item.element_id for item in previews],
                execution_status=ActionExecutionStatus.SUCCEEDED,
            )
        ],
        search_tabs=[
            ControllerSearchTab(
                search_session_id=batch["search_session_id"],
                query=case["question"],
                has_more=not batch["exhausted"],
            )
        ],
        visible_search_view=ControllerVisibleSearchView(
            search_session_id=batch["search_session_id"],
            candidate_previews=previews,
        ),
        remaining_action_budget=5,
    )


def _controller_step(
    *,
    step_id: str,
    controller_input: ControllerInput,
    acceptable_actions: list[dict[str, Any]],
    expected_progress: str,
    teacher_rationale: str,
    observed_action: dict[str, Any] | None,
    observed_result: str,
) -> dict[str, Any]:
    return {
        "step_id": step_id,
        "stage": "controller",
        "controller_input": controller_input.model_dump(mode="json"),
        "acceptable_actions": acceptable_actions,
        "expected_progress": expected_progress,
        "teacher_rationale": teacher_rationale,
        "observed_local_qwen3_8b_action": observed_action,
        "observed_result": observed_result,
    }


def _case_header(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_index": case["question_index"],
        "question": case["question"],
        "answer": case["answer"],
        "gold_pages": case["gold_pages"],
    }


def _q14(case: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    value = _case_header(case)
    value["diagnostic_focus"] = "Exact Anchor automatic page read"
    value["steps"] = [
        {
            "step_id": "Q14-exact-auto-read",
            "stage": "environment",
            "controller_input": None,
            "acceptable_actions": [
                {
                    "action": "AUTO_READ_EXACT",
                    "target_ids": observed["target_ids"],
                }
            ],
            "expected_progress": "evidence",
            "teacher_rationale": (
                "A unique explicit Page 14 anchor bypasses Controller selection and "
                "the Environment reads the page directly."
            ),
            "observed_local_qwen3_8b_action": None,
            "observed_result": "correct automatic route",
        }
    ]
    return value


def _q230(case: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    state = _initial_input(case)
    value = _case_header(case)
    value["diagnostic_focus"] = "Invalid page metadata handle and same-page capability gap"
    value["steps"] = [
        _controller_step(
            step_id="Q230-initial-search-view",
            controller_input=state,
            acceptable_actions=[
                {
                    "action": "SEARCH",
                    "operation": "new",
                    "query": "NMRC commanding officer sends John Sanders",
                }
            ],
            expected_progress="navigation",
            teacher_rationale=(
                "The visible headings locate the right page but do not contain the name, "
                "and page_id is not a readable CandidatePreview handle."
            ),
            observed_action=json.loads(observed["raw_content"]),
            observed_result="invalid action rejected; no progress",
        )
    ]
    return value


def _q546(
    case: dict[str, Any],
    observed_initial: dict[str, Any],
    observed_followup: dict[str, Any],
) -> dict[str, Any]:
    initial = _initial_input(case)
    caption_id = case["batches"][0]["candidate_previews"][0]["element_id"]
    after_next = _after_next_input(case)
    value = _case_header(case)
    value["diagnostic_focus"] = "Reading a non-answering Caption as a relation gateway"
    value["steps"] = [
        _controller_step(
            step_id="Q546-initial-search-view",
            controller_input=initial,
            acceptable_actions=[
                {
                    "action": "READ_SOURCE",
                    "source_ids": [caption_id],
                    "local_problem": (
                        "Locate the chart that shows missed-instance proportions at 60 visible instances."
                    ),
                }
            ],
            expected_progress="navigation",
            teacher_rationale=(
                "The top Caption is the cheapest structural entrance; reading it can expose "
                "the confirmed caption_of relation to the chart."
            ),
            observed_action=observed_initial["action"],
            observed_result="skipped gateway and moved to next batch",
        ),
        _controller_step(
            step_id="Q546-after-next-batch",
            controller_input=after_next,
            acceptable_actions=[
                {
                    "action": "SEARCH",
                    "operation": "new",
                    "query": "Figure 12 visible instances 60 missed category DETR",
                }
            ],
            expected_progress="navigation",
            teacher_rationale=(
                "The second batch has no answerable source; a targeted recovery search is "
                "preferable to reading broad lexical matches."
            ),
            observed_action=observed_followup["action"],
            observed_result="read an unrelated paragraph; no progress",
        ),
    ]
    return value


def _q641(
    case: dict[str, Any],
    observed_initial: dict[str, Any],
    observed_after_table: dict[str, Any],
    observed_relation: dict[str, Any],
) -> dict[str, Any]:
    initial = _initial_input(case)
    batch = case["batches"][0]
    session_id = batch["search_session_id"]
    visible = ControllerVisibleSearchView(
        search_session_id=session_id,
        candidate_previews=[_preview(item) for item in batch["candidate_previews"]],
    )
    after_table = ControllerInput(
        reading_session_id="reading:gold-641",
        root_question=RootQuestion(question_id="root:641", text=case["question"]),
        root_status=EvidenceStatus.INCOMPLETE,
        evidence=[
            ControllerEvidence(
                evidence_id="evidence:q641:table3",
                statement=(
                    "Table 3 gives InstructGPT Self-Ask macro-F1 values: HOVER 2-hop "
                    "51.54, HOVER 3-hop 51.47, HOVER 4-hop 52.45, and FEVEROUS 56.82."
                ),
                supports_question_ids=["root:641"],
            )
        ],
        current_gap=ControllerGap(
            question_id="root:641",
            description=(
                "Identify which dataset has the highest ProgramFC retrieval recall at "
                "10; the Self-Ask scores are already known."
            ),
        ),
        reading_locations=[
            ControllerReadingLocation(
                source_id=Q641_TABLE_ID,
                source_type=ReadingSourceType.ELEMENT,
                page_id="doc:2023.acl-long.386:9064dd38:page:0007",
            )
        ],
        recent_actions=[
            ControllerRecentAction(
                action_id="action:641:read-table",
                question_id="root:641",
                action_name="READ_SOURCE",
                target_ids=[Q641_TABLE_ID],
                execution_status=ActionExecutionStatus.SUCCEEDED,
                observation_ids=["observation:641:table3"],
                feedback=ControllerActionFeedback(
                    observation_assessments=[
                        ControllerObservationAssessment(
                            observation_id="observation:641:table3",
                            used_for_evidence=True,
                            assessment=(
                                "Useful Self-Ask values were obtained, but the dataset "
                                "with the highest retrieval recall is still unknown."
                            ),
                        )
                    ]
                ),
            )
        ],
        search_tabs=[
            ControllerSearchTab(
                search_session_id=session_id,
                query=case["question"],
                has_more=True,
            )
        ],
        visible_search_view=visible,
        remaining_action_budget=5,
    )
    after_paragraph = ControllerInput(
        reading_session_id="reading:gold-641",
        root_question=RootQuestion(question_id="root:641", text=case["question"]),
        root_status=EvidenceStatus.INCOMPLETE,
        evidence=after_table.evidence,
        current_gap=ControllerGap(
            question_id="root:641",
            description="Identify which dataset has the highest ProgramFC retrieval recall at 10.",
        ),
        reading_locations=[
            ControllerReadingLocation(
                source_id=Q641_PARAGRAPH_ID,
                source_type=ReadingSourceType.ELEMENT,
                page_id="doc:2023.acl-long.386:9064dd38:page:0006",
            )
        ],
        confirmed_relations=[
            ControllerConfirmedRelation(
                relation_id=Q641_RELATION_ID,
                relation_type=RelationType.REFERS_TO,
                source_id=Q641_PARAGRAPH_ID,
                target_id=Q641_CHART_ID,
            )
        ],
        remaining_action_budget=4,
    )
    value = _case_header(case)
    value["diagnostic_focus"] = "Evidence-gap narrowing followed by confirmed typed navigation"
    value["steps"] = [
        _controller_step(
            step_id="Q641-initial-search-view",
            controller_input=initial,
            acceptable_actions=[
                {
                    "action": "READ_SOURCE",
                    "source_ids": [Q641_TABLE_ID],
                    "local_problem": "Read the InstructGPT Self-Ask values for every dataset.",
                },
                {
                    "action": "READ_SOURCE",
                    "source_ids": [Q641_PARAGRAPH_ID],
                    "local_problem": "Find the source that identifies the highest ProgramFC recall@10 dataset.",
                },
                {
                    "action": "READ_SOURCE",
                    "source_ids": [Q641_CAPTION_ID],
                    "local_problem": "Locate the Figure 5 recall@10 chart.",
                },
            ],
            expected_progress="evidence_or_navigation",
            teacher_rationale=(
                "The question requires both the Table 3 value and the Figure 5 winner; "
                "any of these three visible entry points can begin a valid route."
            ),
            observed_action=observed_initial["action"],
            observed_result="partial Evidence acquired from Table 3",
        ),
        _controller_step(
            step_id="Q641-after-table-read",
            controller_input=after_table,
            acceptable_actions=[
                {
                    "action": "READ_SOURCE",
                    "source_ids": [Q641_PARAGRAPH_ID],
                    "local_problem": "Identify the dataset with the highest ProgramFC retrieval recall at 10.",
                },
                {
                    "action": "READ_SOURCE",
                    "source_ids": [Q641_CAPTION_ID],
                    "local_problem": "Locate the Figure 5 retrieval recall chart.",
                },
            ],
            expected_progress="navigation",
            teacher_rationale="Only the highest-recall dataset remains unknown.",
            observed_action=observed_after_table["action"],
            observed_result="paragraph read exposed a confirmed refers_to relation",
        ),
        _controller_step(
            step_id="Q641-after-paragraph-read",
            controller_input=after_paragraph,
            acceptable_actions=[
                {
                    "action": "FOLLOW_RELATION",
                    "relation_id": Q641_RELATION_ID,
                    "local_problem": "Identify the dataset with the highest ProgramFC retrieval recall at 10.",
                }
            ],
            expected_progress="evidence",
            teacher_rationale="The confirmed refers_to relation is the direct route to Figure 5.",
            observed_action=observed_relation["action"],
            observed_result="correct confirmed Relation navigation",
        ),
    ]
    return value


def _q1007(
    case: dict[str, Any],
    observed_initial: dict[str, Any],
    observed_followup: dict[str, Any],
) -> dict[str, Any]:
    initial = _initial_input(case)
    after_next = _after_next_input(case)
    value = _case_header(case)
    value["diagnostic_focus"] = "Gold-page hit versus answerable Element and exclusivity evidence"
    value["steps"] = [
        _controller_step(
            step_id="Q1007-initial-search-view",
            controller_input=initial,
            acceptable_actions=[
                {
                    "action": "SEARCH",
                    "operation": "next",
                    "search_session_id": case["batches"][0]["search_session_id"],
                },
                {
                    "action": "SEARCH",
                    "operation": "new",
                    "query": "One40 compatibility Apple Watch iPhone iPad",
                },
            ],
            expected_progress="navigation",
            teacher_rationale="The first batch has no answerable One40 compatibility statement.",
            observed_action=observed_initial["action"],
            observed_result="reasonable rejection; next batch exposed",
        ),
        _controller_step(
            step_id="Q1007-after-next-batch",
            controller_input=after_next,
            acceptable_actions=[
                {
                    "action": "SEARCH",
                    "operation": "new",
                    "query": "One40 compatibility Apple Watch iPhone iPad",
                }
            ],
            expected_progress="navigation",
            teacher_rationale=(
                "The visible Apple Watch section labels are on a Gold page but do not "
                "state One40 compatibility or exclusivity."
            ),
            observed_action=observed_followup["action"],
            observed_result="read weak headings; no Evidence progress",
        ),
    ]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
