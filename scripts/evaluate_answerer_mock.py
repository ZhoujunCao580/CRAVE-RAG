"""Evaluate the frozen Answerer prompt on compact synthetic cases."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import request

from pydantic import ValidationError

from softdoc.answering import (
    ANSWERER_PROMPT_VERSION,
    ANSWERER_SYSTEM_PROMPT,
    AnswerEvidence,
    AnswerInput,
    AnswerQuestionNode,
    AnswerResult,
    answerer_user_prompt,
    validate_answer_result,
)
from softdoc.reading_state import RootQuestion


@dataclass(frozen=True)
class Case:
    case_id: str
    category: str
    root_text: str
    evidence: tuple[tuple[str, str, tuple[str, ...]], ...]
    graph: tuple[tuple[str, str, tuple[str, ...]], ...] = ()
    required_ids: frozenset[str] = field(default_factory=frozenset)
    forbidden_ids: frozenset[str] = field(default_factory=frozenset)
    required_answer_terms: tuple[str, ...] = ()
    forbidden_answer_terms: tuple[str, ...] = ()
    expect_limitation: bool = False

    def answer_input(self) -> AnswerInput:
        return AnswerInput(
            reading_session_id=f"reading:{self.case_id}",
            root_question=RootQuestion(
                question_id=f"root:{self.case_id}",
                text=self.root_text,
            ),
            question_graph=[
                AnswerQuestionNode(
                    question_id=question_id,
                    text=text,
                    depends_on=list(depends_on),
                )
                for question_id, text, depends_on in self.graph
            ],
            evidence=[
                AnswerEvidence(
                    evidence_id=evidence_id,
                    statement=statement,
                    supports_question_ids=list(supports),
                )
                for evidence_id, statement, supports in self.evidence
            ],
        )


def cases() -> list[Case]:
    return [
        Case(
            case_id="01_single",
            category="single_evidence",
            root_text="What was the company's 2023 revenue? Answer in one sentence.",
            evidence=(("E1", "The company's 2023 revenue was $12 million.", ("root:01_single",)),),
            required_ids=frozenset({"E1"}),
            required_answer_terms=("12",),
        ),
        Case(
            case_id="02_calculation",
            category="multiple_evidence_calculation",
            root_text="How much and by what percentage did revenue increase from 2022 to 2023?",
            evidence=(
                ("E1", "Revenue in 2022 was $10 million.", ("Q1",)),
                ("E2", "Revenue in 2023 was $12 million.", ("Q2",)),
                ("E3", "The headquarters is in Seattle.", ("root:02_calculation",)),
            ),
            graph=(("Q1", "What was 2022 revenue?", ()), ("Q2", "What was 2023 revenue?", ())),
            required_ids=frozenset({"E1", "E2"}),
            forbidden_ids=frozenset({"E3"}),
            required_answer_terms=("2", "20"),
        ),
        Case(
            case_id="03_insufficient",
            category="insufficient_evidence",
            root_text="What were the revenues in both 2022 and 2023?",
            evidence=(("E1", "Revenue in 2022 was $10 million.", ("Q1",)),),
            graph=(("Q1", "What was 2022 revenue?", ()), ("Q2", "What was 2023 revenue?", ())),
            required_ids=frozenset({"E1"}),
            required_answer_terms=("2023",),
            expect_limitation=True,
        ),
        Case(
            case_id="04_conflict",
            category="conflicting_evidence",
            root_text="What was the company's 2023 revenue?",
            evidence=(
                ("E1", "The company's 2023 revenue was $12 million.", ("root:04_conflict",)),
                ("E2", "The company's 2023 revenue was $14 million.", ("root:04_conflict",)),
            ),
            required_ids=frozenset({"E1", "E2"}),
            expect_limitation=True,
        ),
        Case(
            case_id="05_irrelevant",
            category="irrelevant_evidence_filtering",
            root_text="What was net income in 2023?",
            evidence=(
                ("E1", "Net income in 2023 was $5 million.", ("root:05_irrelevant",)),
                ("E2", "The company employed 10,000 people in 2023.", ("root:05_irrelevant",)),
            ),
            required_ids=frozenset({"E1"}),
            forbidden_ids=frozenset({"E2"}),
            required_answer_terms=("5",),
        ),
        Case(
            case_id="06_graph_distractor",
            category="question_graph_interference",
            root_text="What was revenue in 2023?",
            evidence=(
                ("E1", "The company employed 10,000 people in 2023.", ("Q1",)),
                ("E2", "Revenue in 2023 was $12 million.", ("Q2",)),
            ),
            graph=(
                ("Q1", "How many employees did the company have?", ()),
                ("Q2", "What was revenue in 2023?", ()),
            ),
            required_ids=frozenset({"E2"}),
            forbidden_ids=frozenset({"E1"}),
            required_answer_terms=("12",),
        ),
        Case(
            case_id="07_condition_mismatch",
            category="condition_mismatch",
            root_text="What was Method A's accuracy under low-light conditions?",
            evidence=(
                ("E1", "Method A achieved 90% accuracy under normal-light conditions.", ("root:07_condition_mismatch",)),
                ("E2", "Method B achieved 79% accuracy under low-light conditions.", ("root:07_condition_mismatch",)),
            ),
            forbidden_answer_terms=("method a achieved 90% under low-light", "method a's low-light accuracy was 79"),
            expect_limitation=True,
        ),
        Case(
            case_id="08_language",
            category="language_and_format",
            root_text="What was the company's 2023 revenue? Answer in one English sentence.",
            evidence=(("E1", "The company's 2023 revenue was 12 million yuan.", ("root:08_language",)),),
            required_ids=frozenset({"E1"}),
            required_answer_terms=("12 million yuan",),
        ),
        Case(
            case_id="09_empty_graph",
            category="no_planner_graph",
            root_text="How many employees did the company have in 2023?",
            evidence=(("E1", "The company had 10,000 employees in 2023.", ("root:09_empty_graph",)),),
            required_ids=frozenset({"E1"}),
            required_answer_terms=("10,000",),
        ),
        Case(
            case_id="10_dependency",
            category="dag_dependency_synthesis",
            root_text="Which year had the largest revenue increase, and what caused it?",
            evidence=(
                ("E1", "The largest revenue increase occurred in 2023.", ("Q1",)),
                ("E2", "The 2023 increase was primarily caused by higher product demand.", ("Q2",)),
            ),
            graph=(
                ("Q1", "Which year had the largest increase?", ()),
                ("Q2", "What caused the increase in that year?", ("Q1",)),
            ),
            required_ids=frozenset({"E1", "E2"}),
            required_answer_terms=("2023", "demand"),
        ),
        Case(
            case_id="11_no_reason",
            category="unsupported_assumption",
            root_text="Why did profit rise from 2022 to 2023?",
            evidence=(
                ("E1", "Profit was $10 million in 2022.", ("root:11_no_reason",)),
                ("E2", "Profit was $12 million in 2023.", ("root:11_no_reason",)),
            ),
            required_ids=frozenset({"E1", "E2"}),
            forbidden_answer_terms=("lower costs", "higher demand", "price increase"),
            expect_limitation=True,
        ),
        Case(
            case_id="12_average",
            category="all_calculation_inputs",
            root_text="What was the average annual revenue across 2021, 2022, and 2023?",
            evidence=(
                ("E1", "Revenue in 2021 was $10 million.", ("root:12_average",)),
                ("E2", "Revenue in 2022 was $20 million.", ("root:12_average",)),
                ("E3", "Revenue in 2023 was $30 million.", ("root:12_average",)),
                ("E4", "The company was founded in 1998.", ("root:12_average",)),
            ),
            required_ids=frozenset({"E1", "E2", "E3"}),
            forbidden_ids=frozenset({"E4"}),
            required_answer_terms=("20",),
        ),
        Case(
            case_id="13_ranking",
            category="comparison_and_ordering",
            root_text="Which method had the highest accuracy?",
            evidence=(
                ("E1", "Method A accuracy was 72%.", ("root:13_ranking",)),
                ("E2", "Method B accuracy was 79%.", ("root:13_ranking",)),
                ("E3", "Method C accuracy was 76%.", ("root:13_ranking",)),
            ),
            required_ids=frozenset({"E1", "E2", "E3"}),
            required_answer_terms=("method b",),
        ),
        Case(
            case_id="14_graph_false_fact",
            category="graph_is_not_evidence",
            root_text="Where is the headquarters?",
            evidence=(("E1", "The headquarters is in Seattle.", ("Q2",)),),
            graph=(
                ("Q1", "Is the headquarters in Boston?", ()),
                ("Q2", "Where is the headquarters?", ()),
            ),
            required_ids=frozenset({"E1"}),
            required_answer_terms=("seattle",),
            forbidden_answer_terms=("boston",),
        ),
        Case(
            case_id="15_percentage_points",
            category="percentage_point_change",
            root_text="By how many percentage points did market share increase from 2022 to 2023?",
            evidence=(
                ("E1", "Market share was 38% in 2022.", ("root:15_percentage_points",)),
                ("E2", "Market share was 44% in 2023.", ("root:15_percentage_points",)),
            ),
            required_ids=frozenset({"E1", "E2"}),
            required_answer_terms=("6 percentage points",),
            forbidden_answer_terms=("15.79%", "15.8%"),
        ),
        Case(
            case_id="16_relative_percent",
            category="relative_percentage_change",
            root_text="By what percentage did market share increase from 38% to 44%? Round to two decimal places.",
            evidence=(
                ("E1", "The initial market share was 38%.", ("root:16_relative_percent",)),
                ("E2", "The final market share was 44%.", ("root:16_relative_percent",)),
            ),
            required_ids=frozenset({"E1", "E2"}),
            required_answer_terms=("15.79%",),
            forbidden_answer_terms=("6%", "6 percentage points"),
        ),
        Case(
            case_id="17_unit_conversion",
            category="unit_conversion",
            root_text="Express the reported revenue of 1.2 billion yuan in millions of yuan.",
            evidence=(("E1", "Reported revenue was 1.2 billion yuan.", ("root:17_unit_conversion",)),),
            required_ids=frozenset({"E1"}),
            required_answer_terms=("1,200 million yuan",),
        ),
        Case(
            case_id="18_rounding",
            category="rounding_instruction",
            root_text="What is 17 divided by 6? Round the result to two decimal places.",
            evidence=(
                ("E1", "The total number of units is 17.", ("root:18_rounding",)),
                ("E2", "The units are divided equally among 6 groups.", ("root:18_rounding",)),
            ),
            required_ids=frozenset({"E1", "E2"}),
            required_answer_terms=("2.83",),
        ),
        Case(
            case_id="19_exclusion",
            category="set_filtering_with_exclusion",
            root_text="Which models other than Model A exceeded 90% accuracy?",
            evidence=((
                "E1",
                "Model A scored 92%, Model B scored 94%, Model C scored 91%, and Model D scored 88% accuracy.",
                ("root:19_exclusion",),
            ),),
            required_ids=frozenset({"E1"}),
            required_answer_terms=("model b", "model c"),
            forbidden_answer_terms=("model a exceeded", "model d"),
        ),
        Case(
            case_id="20_yes_no",
            category="yes_no_comparison",
            root_text="Did Model B outperform Model C? Answer only yes or no.",
            evidence=(
                ("E1", "Model B accuracy was 79%.", ("root:20_yes_no",)),
                ("E2", "Model C accuracy was 76%.", ("root:20_yes_no",)),
            ),
            required_ids=frozenset({"E1", "E2"}),
            required_answer_terms=("yes",),
        ),
        Case(
            case_id="21_temporal_scope",
            category="temporal_scope_with_distractors",
            root_text="What was the company's revenue in FY2023?",
            evidence=(
                ("E1", "Revenue in FY2022 was $10 million.", ("root:21_temporal_scope",)),
                ("E2", "Revenue in FY2023 was $12 million.", ("root:21_temporal_scope",)),
                ("E3", "Revenue in the fourth quarter of 2023 was $4 million.", ("root:21_temporal_scope",)),
            ),
            required_ids=frozenset({"E2"}),
            forbidden_ids=frozenset({"E1", "E3"}),
            required_answer_terms=("12",),
        ),
        Case(
            case_id="22_explicit_cause",
            category="causal_statement_with_correlation_distractor",
            root_text="What caused the revenue increase in 2023?",
            evidence=(
                ("E1", "The 2023 revenue increase was primarily caused by higher product demand.", ("root:22_explicit_cause",)),
                ("E2", "A new marketing campaign was launched in 2023.", ("root:22_explicit_cause",)),
                ("E3", "Revenue increased in 2023.", ("root:22_explicit_cause",)),
            ),
            required_ids=frozenset({"E1"}),
            forbidden_ids=frozenset({"E2"}),
            required_answer_terms=("higher product demand",),
            forbidden_answer_terms=("marketing campaign caused",),
        ),
        Case(
            case_id="23_partial_multi_part",
            category="partial_multi_part_defensive",
            root_text="What was the company's 2023 revenue, and why did it increase?",
            evidence=(("E1", "The company's 2023 revenue was $12 million.", ("root:23_partial_multi_part",)),),
            required_ids=frozenset({"E1"}),
            required_answer_terms=("12",),
            forbidden_answer_terms=("higher demand", "lower costs", "price increase"),
            expect_limitation=True,
        ),
        Case(
            case_id="24_long_noisy_pack",
            category="long_evidence_pack_filtering",
            root_text="What was operating income in 2023?",
            evidence=(
                ("E1", "The company was founded in 1998.", ("root:24_long_noisy_pack",)),
                ("E2", "Revenue in 2023 was $120 million.", ("root:24_long_noisy_pack",)),
                ("E3", "The headquarters is in Seattle.", ("root:24_long_noisy_pack",)),
                ("E4", "The company employed 10,000 people in 2023.", ("root:24_long_noisy_pack",)),
                ("E5", "Operating income in 2022 was $11.2 million.", ("root:24_long_noisy_pack",)),
                ("E6", "Net income in 2023 was $9.1 million.", ("root:24_long_noisy_pack",)),
                ("E7", "Operating income in 2023 was $14.6 million.", ("root:24_long_noisy_pack",)),
                ("E8", "Research spending in 2023 was $7.4 million.", ("root:24_long_noisy_pack",)),
            ),
            required_ids=frozenset({"E7"}),
            forbidden_ids=frozenset({"E1", "E2", "E3", "E4", "E5", "E6", "E8"}),
            required_answer_terms=("14.6",),
        ),
    ]


def call_ollama(*, base_url: str, model: str, answer_input: AnswerInput) -> tuple[str, float]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": ANSWERER_SYSTEM_PROMPT},
            {"role": "user", "content": answerer_user_prompt(answer_input)},
        ],
        "stream": False,
        "think": False,
        "format": AnswerResult.model_json_schema(),
        "options": {"temperature": 0.0, "seed": 42, "num_ctx": 4096},
    }
    started = time.perf_counter()
    req = request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=180) as response:
        body = json.loads(response.read().decode("utf-8"))
    elapsed = time.perf_counter() - started
    message = body.get("message", {})
    return (message.get("content") or message.get("thinking") or ""), elapsed


def judge(case: Case, result: AnswerResult) -> tuple[bool, list[str]]:
    problems: list[str] = []
    used = set(result.used_evidence_ids)
    missing = case.required_ids.difference(used)
    forbidden = case.forbidden_ids.intersection(used)
    if missing:
        problems.append("missing evidence IDs: " + ", ".join(sorted(missing)))
    if forbidden:
        problems.append("used irrelevant evidence IDs: " + ", ".join(sorted(forbidden)))

    normalized = " ".join(result.answer.casefold().split())
    for term in case.required_answer_terms:
        if term.casefold() not in normalized:
            problems.append(f"answer missing expected term: {term}")
    for term in case.forbidden_answer_terms:
        if term.casefold() in normalized:
            problems.append(f"answer contains forbidden claim: {term}")
    if case.expect_limitation:
        limitation_markers = (
            "insufficient",
            "cannot",
            "not provided",
            "does not provide",
            "no evidence",
            "conflict",
            "inconsistent",
        )
        if not any(marker in normalized for marker in limitation_markers):
            problems.append("expected an explicit limitation")
    return not problems, problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/answerer_mock_v0_2/results.json"),
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for case in cases():
        answer_input = case.answer_input()
        raw = ""
        try:
            raw, elapsed = call_ollama(
                base_url=args.base_url,
                model=args.model,
                answer_input=answer_input,
            )
            result = AnswerResult.model_validate_json(raw)
            validate_answer_result(answer_input, result)
            passed, problems = judge(case, result)
            row = {
                "case_id": case.case_id,
                "category": case.category,
                "passed": passed,
                "problems": problems,
                "elapsed_seconds": round(elapsed, 3),
                "input": answer_input.model_dump(mode="json"),
                "result": result.model_dump(mode="json"),
                "raw_output": raw,
            }
        except (ValidationError, ValueError, json.JSONDecodeError, OSError) as exc:
            row = {
                "case_id": case.case_id,
                "category": case.category,
                "passed": False,
                "problems": [f"{type(exc).__name__}: {exc}"],
                "input": answer_input.model_dump(mode="json"),
                "raw_output": raw,
            }
        rows.append(row)
        print(f"{case.case_id}: {'PASS' if row['passed'] else 'FAIL'}")

    report = {
        "prompt_version": ANSWERER_PROMPT_VERSION,
        "system_prompt": ANSWERER_SYSTEM_PROMPT,
        "model": args.model,
        "passed": sum(bool(row["passed"]) for row in rows),
        "total": len(rows),
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"passed={report['passed']}/{report['total']}")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
