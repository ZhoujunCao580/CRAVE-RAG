"""Evaluate the frozen Controller v0 policy with synthetic reading states.

The cases exercise action selection only. They do not evaluate final-answer
accuracy, Reader quality, retrieval recall, or Evidence Checker correctness.
Each model output is first contract-validated and then compared with a small
Teacher reference action.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from softdoc.controller import (
    AdjacentPageDirection,
    ControllerAction,
    ControllerActionFeedback,
    ControllerCandidatePreview,
    ControllerCandidateRelation,
    ControllerConfirmedRelation,
    ControllerEvidence,
    ControllerGap,
    ControllerInput,
    ControllerLimitation,
    ControllerObservationAssessment,
    ControllerReadingLocation,
    ControllerRecentAction,
    ControllerSearchTab,
    ControllerSubQuestion,
    ControllerVisibleSearchView,
)
from softdoc.controller_ollama import (
    OllamaControllerBackend,
    OllamaControllerConfig,
    OllamaControllerError,
)
from softdoc.controller_prompt import (
    CONTROLLER_PROMPT_VERSION,
    CONTROLLER_SYSTEM_PROMPT,
)
from softdoc.models import (
    ContentAvailability,
    ElementType,
    RelationType,
)
from softdoc.reading_state import (
    ActionExecutionStatus,
    EvidenceStatus,
    QuestionStatus,
    ReadingSourceType,
    RootQuestion,
)


@dataclass(frozen=True)
class ControllerCase:
    case_id: str
    category: str
    description: str
    controller_input: ControllerInput
    expected: dict[str, Any]


def preview(
    element_id: str,
    text: str,
    *,
    element_type: ElementType = ElementType.PARAGRAPH,
    page_id: str = "page:opaque-a",
    availability: ContentAvailability = ContentAvailability.TEXT_ONLY,
) -> ControllerCandidatePreview:
    return ControllerCandidatePreview(
        element_id=element_id,
        element_type=element_type,
        page_id=page_id,
        section_path=["Relevant section"],
        matched_snippet=text,
        content_availability=availability,
    )


def make_input(
    case_id: str,
    root_text: str,
    gap: str,
    *,
    subquestions: list[ControllerSubQuestion] | None = None,
    evidence: list[ControllerEvidence] | None = None,
    locations: list[ControllerReadingLocation] | None = None,
    recent_actions: list[ControllerRecentAction] | None = None,
    confirmed: list[ControllerConfirmedRelation] | None = None,
    candidates: list[ControllerCandidateRelation] | None = None,
    tabs: list[ControllerSearchTab] | None = None,
    visible_session_id: str | None = None,
    previews: list[ControllerCandidatePreview] | None = None,
    budget: int = 4,
) -> ControllerInput:
    root_id = f"root:{case_id}"
    questions = subquestions or []
    current_question_id = next(
        (
            item.question_id
            for item in questions
            if item.status == QuestionStatus.INCOMPLETE
            and all(
                dependency.status == QuestionStatus.SATISFIED
                for dependency in questions
                if dependency.question_id in item.depends_on
            )
        ),
        root_id,
    )
    return ControllerInput(
        reading_session_id=f"reading:{case_id}",
        root_question=RootQuestion(question_id=root_id, text=root_text),
        root_status=EvidenceStatus.INCOMPLETE,
        subquestions=questions,
        evidence=evidence or [],
        current_gap=ControllerGap(
            question_id=current_question_id,
            description=gap,
        ),
        reading_locations=locations or [],
        recent_actions=recent_actions or [],
        confirmed_relations=confirmed or [],
        candidate_relations=candidates or [],
        search_tabs=tabs or [],
        visible_search_view=(
            ControllerVisibleSearchView(
                search_session_id=visible_session_id,
                candidate_previews=previews or [],
            )
            if visible_session_id is not None
            else None
        ),
        remaining_action_budget=budget,
    )


def build_cases() -> list[ControllerCase]:
    cases: list[ControllerCase] = []

    cases.append(
        ControllerCase(
            "K01",
            "new_search",
            "No source, relation, or SearchSession exists, so a new search is required.",
            make_input(
                "K01",
                "What was Northwind's operating margin in 2023?",
                "Northwind's 2023 operating margin is unknown.",
            ),
            {"action": "SEARCH", "operation": "new"},
        )
    )

    cases.append(
        ControllerCase(
            "K02",
            "read_preview",
            "A visible table preview directly contains the missing metric.",
            make_input(
                "K02",
                "What was Northwind's operating margin in 2023?",
                "Northwind's 2023 operating margin is unknown.",
                tabs=[ControllerSearchTab(search_session_id="search:a", query="Northwind 2023 operating margin", has_more=True)],
                visible_session_id="search:a",
                previews=[preview("element:metric-table", "The 2023 operating margin was 18.4%.", element_type=ElementType.TABLE, availability=ContentAvailability.STRUCTURED)],
            ),
            {"action": "READ_SOURCE", "source_ids": ["element:metric-table"]},
        )
    )

    cases.append(
        ControllerCase(
            "K03",
            "next_batch",
            "The current batch is irrelevant and the same tab has more candidates.",
            make_input(
                "K03",
                "Why did Northwind's operating margin decline?",
                "The document-grounded reason for the margin decline is missing.",
                tabs=[ControllerSearchTab(search_session_id="search:margin", query="Northwind margin decline reason", has_more=True)],
                visible_session_id="search:margin",
                previews=[preview("element:headcount", "Employee headcount by region in 2023.")],
            ),
            {"action": "SEARCH", "operation": "next", "search_session_id": "search:margin"},
        )
    )

    cases.append(
        ControllerCase(
            "K04",
            "switch_tab",
            "Another existing search tab is explicitly about the current gap.",
            make_input(
                "K04",
                "Why did Northwind's operating margin decline?",
                "The document-grounded reason for the margin decline is missing.",
                tabs=[
                    ControllerSearchTab(search_session_id="search:old", query="regional headcount", has_more=False),
                    ControllerSearchTab(search_session_id="search:cause", query="margin decline cause cost inflation", has_more=True),
                ],
                visible_session_id="search:old",
                previews=[preview("element:people", "Regional employee distribution.")],
            ),
            {"action": "SEARCH", "operation": "switch", "search_session_id": "search:cause"},
        )
    )

    location_caption = ControllerReadingLocation(
        source_id="element:caption-a",
        source_type=ReadingSourceType.ELEMENT,
        page_id="page:opaque-b",
    )
    cases.append(
        ControllerCase(
            "K05",
            "confirmed_relation",
            "The opened caption has a confirmed caption_of link to the needed chart.",
            make_input(
                "K05",
                "Which method has the highest low-light accuracy in Figure 3?",
                "The method with the highest low-light accuracy is unknown.",
                locations=[location_caption],
                confirmed=[ControllerConfirmedRelation(relation_id="relation:caption-chart", relation_type=RelationType.CAPTION_OF, source_id="element:caption-a", target_id="element:chart-a")],
            ),
            {"action": "FOLLOW_RELATION", "relation_id": "relation:caption-chart"},
        )
    )

    location_table = ControllerReadingLocation(
        source_id="element:table-front",
        source_type=ReadingSourceType.ELEMENT,
        page_id="page:opaque-c",
    )
    cases.append(
        ControllerCase(
            "K06",
            "candidate_relation",
            "A plausible continued_on candidate is the only concrete route to missing rows.",
            make_input(
                "K06",
                "What is the total shown in the final row of the table?",
                "The final rows and total are not present in the opened table fragment.",
                locations=[location_table],
                candidates=[ControllerCandidateRelation(relation_id="relation:possible-continuation", relation_type=RelationType.CONTINUED_ON, source_id="element:table-front", target_id="element:table-back", confidence=0.84)],
            ),
            {"action": "EXPLORE_CANDIDATE_RELATION", "relation_id": "relation:possible-continuation"},
        )
    )

    cases.append(
        ControllerCase(
            "K07",
            "adjacent_next",
            "The last opened page ends mid-sentence and no continuation relation exists.",
            make_input(
                "K07",
                "What limitation of the method is described?",
                "The method limitation sentence is incomplete after 'however'.",
                locations=[ControllerReadingLocation(source_id="element:tail-paragraph", source_type=ReadingSourceType.ELEMENT, page_id="page:opaque-d")],
                recent_actions=[
                    ControllerRecentAction(
                        action_id="action:tail",
                        question_id="root:K07",
                        action_name="READ_SOURCE",
                        target_ids=["element:tail-paragraph"],
                        execution_status=ActionExecutionStatus.DEGRADED,
                        observation_ids=["obs:partial"],
                        feedback=ControllerActionFeedback(
                            limitations=[
                                ControllerLimitation(
                                    description="The sentence ends at the physical page boundary after 'however'.",
                                    source_ids=["element:tail-paragraph"],
                                )
                            ]
                        ),
                    )
                ],
            ),
            {"action": "READ_ADJACENT_PAGE", "from_page_id": "page:opaque-d", "direction": AdjacentPageDirection.NEXT.value},
        )
    )

    cases.append(
        ControllerCase(
            "K08",
            "adjacent_previous",
            "The opened page contains values but the preceding page is needed for column headers.",
            make_input(
                "K08",
                "Which column contains the 42.7 value?",
                "The value is visible but its column header is missing from the opened page.",
                locations=[ControllerReadingLocation(source_id="page:opened", source_type=ReadingSourceType.PAGE, page_id="page:opaque-e")],
                recent_actions=[
                    ControllerRecentAction(
                        action_id="action:values",
                        question_id="root:K08",
                        action_name="READ_SOURCE",
                        target_ids=["page:opened"],
                        execution_status=ActionExecutionStatus.DEGRADED,
                        observation_ids=["obs:value"],
                        feedback=ControllerActionFeedback(
                            limitations=[
                                ControllerLimitation(
                                    description="Column headers begin on the preceding physical page.",
                                    source_ids=["page:opened"],
                                )
                            ]
                        ),
                    )
                ],
            ),
            {"action": "READ_ADJACENT_PAGE", "from_page_id": "page:opaque-e", "direction": AdjacentPageDirection.PREVIOUS.value},
        )
    )

    cases.append(
        ControllerCase(
            "K09",
            "avoid_failed_repeat",
            "A visual read failed, but a confirmed caption relation offers a different source.",
            make_input(
                "K09",
                "What does Figure 8 report about Method C?",
                "The finding about Method C is still unknown because chart labels were unreadable.",
                locations=[ControllerReadingLocation(source_id="element:chart-small", source_type=ReadingSourceType.ELEMENT, page_id="page:opaque-f")],
                recent_actions=[ControllerRecentAction(action_id="action:small-chart", question_id="root:K09", action_name="READ_SOURCE", target_ids=["element:chart-small"], execution_status=ActionExecutionStatus.DEGRADED, feedback=ControllerActionFeedback(limitations=[ControllerLimitation(description="Chart labels are too small to read reliably.", source_ids=["element:chart-small"])]))],
                confirmed=[ControllerConfirmedRelation(relation_id="relation:chart-caption", relation_type=RelationType.CAPTION_OF, source_id="element:caption-detail", target_id="element:chart-small")],
            ),
            {"action": "FOLLOW_RELATION", "relation_id": "relation:chart-caption"},
        )
    )

    cases.append(
        ControllerCase(
            "K10",
            "verification",
            "Conflicting reported values justify reading a visible audited table.",
            make_input(
                "K10",
                "What was the audited revenue in 2023?",
                "Resolve whether the audited revenue is 12 or 14 million.",
                evidence=[ControllerEvidence(evidence_id="evidence:old", statement="A narrative paragraph reports revenue of 12 million.", supports_question_ids=["root:K10"])],
                tabs=[ControllerSearchTab(search_session_id="search:audit", query="audited revenue 2023", has_more=False)],
                visible_session_id="search:audit",
                previews=[preview("element:audited-table", "Audited consolidated revenue table for 2023.", element_type=ElementType.TABLE, availability=ContentAvailability.STRUCTURED)],
            ),
            {"action": "READ_SOURCE", "source_ids": ["element:audited-table"]},
        )
    )

    cases.append(
        ControllerCase(
            "K11",
            "candidate_selection",
            "Only one of three visible candidates directly addresses the requested year and metric.",
            make_input(
                "K11",
                "What was the 2021 research and development expense?",
                "The 2021 research and development expense is unknown.",
                tabs=[ControllerSearchTab(search_session_id="search:rd", query="2021 research development expense", has_more=True)],
                visible_session_id="search:rd",
                previews=[
                    preview("element:sales-2021", "Net sales in 2021."),
                    preview("element:rd-2021", "Research and development expenses for 2021, 2022 and 2023.", element_type=ElementType.TABLE, availability=ContentAvailability.STRUCTURED),
                    preview("element:rd-policy", "Accounting policy for development costs."),
                ],
            ),
            {"action": "READ_SOURCE", "source_ids": ["element:rd-2021"]},
        )
    )

    cases.append(
        ControllerCase(
            "K12",
            "relation_over_weak_preview",
            "A confirmed refers_to relation is direct while the search preview is only lexical noise.",
            make_input(
                "K12",
                "What value is shown for the referenced Figure 5 condition?",
                "The value in the referenced Figure 5 is unknown.",
                locations=[ControllerReadingLocation(source_id="element:reference-sentence", source_type=ReadingSourceType.ELEMENT, page_id="page:opaque-g")],
                confirmed=[ControllerConfirmedRelation(relation_id="relation:explicit-figure", relation_type=RelationType.REFERS_TO, source_id="element:reference-sentence", target_id="element:figure-five")],
                tabs=[ControllerSearchTab(search_session_id="search:noise", query="Figure 5 condition value", has_more=True)],
                visible_session_id="search:noise",
                previews=[preview("element:bibliography-five", "Reference number 5 discusses experimental conditions.")],
            ),
            {"action": "FOLLOW_RELATION", "relation_id": "relation:explicit-figure"},
        )
    )

    cases.append(
        ControllerCase(
            "K13",
            "preview_over_weak_candidate",
            "A direct preview should beat a low-confidence continuation hypothesis.",
            make_input(
                "K13",
                "What is the warranty period?",
                "The warranty period is unknown.",
                locations=[ControllerReadingLocation(source_id="element:old-section", source_type=ReadingSourceType.ELEMENT, page_id="page:opaque-h")],
                candidates=[ControllerCandidateRelation(relation_id="relation:weak", relation_type=RelationType.CONTINUED_ON, source_id="element:old-section", target_id="element:possible-next", confidence=0.31)],
                tabs=[ControllerSearchTab(search_session_id="search:warranty", query="warranty period", has_more=False)],
                visible_session_id="search:warranty",
                previews=[preview("element:warranty", "The limited warranty period is two years from purchase.")],
            ),
            {"action": "READ_SOURCE", "source_ids": ["element:warranty"]},
        )
    )

    cases.append(
        ControllerCase(
            "K14",
            "joint_visual_read",
            "The local question genuinely requires comparing two visible charts together.",
            make_input(
                "K14",
                "Which chart shows the steeper visual increase?",
                "A joint visual comparison of the two trends is missing.",
                tabs=[ControllerSearchTab(search_session_id="search:charts", query="two trend charts", has_more=False)],
                visible_session_id="search:charts",
                previews=[
                    preview("element:chart-left", "Chart A: yearly trend.", element_type=ElementType.CHART, availability=ContentAvailability.VISUAL_ONLY),
                    preview("element:chart-right", "Chart B: yearly trend.", element_type=ElementType.CHART, availability=ContentAvailability.VISUAL_ONLY),
                ],
            ),
            {"action": "READ_SOURCE", "source_ids_set": ["element:chart-left", "element:chart-right"]},
        )
    )

    cases.append(
        ControllerCase(
            "K15",
            "reformulated_search",
            "The exhausted current tab is irrelevant, so a new query is required.",
            make_input(
                "K15",
                "Why was the product launch delayed?",
                "The stated cause of the launch delay is missing.",
                tabs=[ControllerSearchTab(search_session_id="search:date", query="product launch date", has_more=False)],
                visible_session_id="search:date",
                previews=[preview("element:schedule", "The product launched on 18 June.")],
            ),
            {"action": "SEARCH", "operation": "new"},
        )
    )

    return cases


def action_dict(action: ControllerAction) -> dict[str, Any]:
    return action.model_dump(mode="json", exclude_none=True)


def matches_expected(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        if key == "source_ids_set":
            if set(actual.get("source_ids", [])) != set(expected_value):
                return False
        elif actual.get(key) != expected_value:
            return False
    return True


def semantic_signature(record: dict[str, Any]) -> dict[str, Any] | None:
    action = record.get("action")
    if not isinstance(action, dict):
        return None
    return {
        key: value
        for key, value in action.items()
        if key not in {"local_problem", "query"}
    }


def run_pass(
    cases: list[ControllerCase],
    *,
    pass_name: str,
    backend: OllamaControllerBackend,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{pass_name}] {index:02d}/{len(cases)} {case.case_id} {case.category}", flush=True)
        started = time.perf_counter()
        try:
            generation = backend.generate(case.controller_input)
            elapsed = time.perf_counter() - started
            actual = action_dict(generation.action)
            records.append(
                {
                    "pass": pass_name,
                    "case_id": case.case_id,
                    "category": case.category,
                    "description": case.description,
                    "input": case.controller_input.model_dump(mode="json"),
                    "expected": case.expected,
                    "valid_action": True,
                    "teacher_match": matches_expected(actual, case.expected),
                    "action": actual,
                    "raw_output": generation.raw_content,
                    "elapsed_seconds": round(elapsed, 3),
                    "ollama_metrics": generation.metadata,
                }
            )
        except OllamaControllerError as exc:
            records.append(
                {
                    "pass": pass_name,
                    "case_id": case.case_id,
                    "category": case.category,
                    "description": case.description,
                    "input": case.controller_input.model_dump(mode="json"),
                    "expected": case.expected,
                    "valid_action": False,
                    "teacher_match": False,
                    "raw_output": exc.raw_content,
                    "error": str(exc),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )
    return records


def write_report(
    output_dir: Path,
    *,
    model: str,
    cases: list[ControllerCase],
    records: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "controller_system_prompt.txt").write_text(
        CONTROLLER_SYSTEM_PROMPT,
        encoding="utf-8",
    )
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    by_key = {(record["pass"], record["case_id"]): record for record in records}
    stability = []
    for case in cases:
        first = by_key.get(("A", case.case_id), {})
        second = by_key.get(("B", case.case_id), {})
        stability.append(
            {
                "case_id": case.case_id,
                "exact": first.get("action") == second.get("action") and bool(first.get("action")),
                "semantic": semantic_signature(first) == semantic_signature(second) and semantic_signature(first) is not None,
            }
        )

    valid = sum(bool(record.get("valid_action")) for record in records)
    correct = sum(bool(record.get("teacher_match")) for record in records)
    total = len(records)
    semantic_stable = sum(item["semantic"] for item in stability)
    summary = {
        "model": model,
        "prompt_version": CONTROLLER_PROMPT_VERSION,
        "case_count": len(cases),
        "pass_count": len({record["pass"] for record in records}),
        "generation_count": total,
        "valid_action_count": valid,
        "valid_action_rate": valid / total if total else 0.0,
        "teacher_match_count": correct,
        "teacher_match_rate": correct / total if total else 0.0,
        "semantic_stability_count": semantic_stable,
        "semantic_stability_rate": semantic_stable / len(cases) if cases else 0.0,
        "stability": stability,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    failures = [record for record in records if not record.get("teacher_match")]
    lines = [
        "# Controller v0 Qwen3 8B mock evaluation",
        "",
        f"- Model: `{model}`",
        f"- Prompt: `{CONTROLLER_PROMPT_VERSION}`",
        f"- Synthetic cases: {len(cases)}",
        f"- Generations: {total}",
        f"- Contract-valid actions: {valid}/{total} ({valid / total:.1%})",
        f"- Teacher-reference matches: {correct}/{total} ({correct / total:.1%})",
        f"- Two-pass semantic stability: {semantic_stable}/{len(cases)} ({semantic_stable / len(cases):.1%})",
        "",
        "This is an action-selection probe, not an end-to-end QA score.",
        "",
        "## Non-matching generations",
        "",
    ]
    if not failures:
        lines.append("None.")
    for record in failures:
        lines.extend(
            [
                f"### {record['pass']} / {record['case_id']} / {record['category']}",
                "",
                record["description"],
                "",
                f"- Expected: `{json.dumps(record['expected'], ensure_ascii=False)}`",
                f"- Actual: `{json.dumps(record.get('action'), ensure_ascii=False)}`",
                f"- Error: `{record.get('error')}`" if record.get("error") else "",
                "",
            ]
        )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--passes", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".runlogs/controller_v0_qwen3_8b_eval"),
    )
    args = parser.parse_args()

    cases = build_cases()
    records: list[dict[str, Any]] = []
    pass_names = ["A", "B"][: args.passes]
    for offset, pass_name in enumerate(pass_names):
        backend = OllamaControllerBackend(
            OllamaControllerConfig(
                base_url=args.base_url,
                model=args.model,
                timeout_seconds=args.timeout,
                seed=args.seed + offset,
            )
        )
        records.extend(run_pass(cases, pass_name=pass_name, backend=backend))

    write_report(
        args.output,
        model=args.model,
        cases=cases,
        records=records,
    )
    correct = sum(bool(record.get("teacher_match")) for record in records)
    valid = sum(bool(record.get("valid_action")) for record in records)
    print(
        f"Completed {len(records)} generations: valid={valid}, "
        f"teacher_match={correct}. Reports: {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
