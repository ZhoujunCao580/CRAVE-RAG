"""Evaluate the frozen initial Planner prompt on compact synthetic cases."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from softdoc.planning import (
    INITIAL_PLANNER_PROMPT_VERSION,
    InitialPlanner,
    OllamaPlannerBackend,
    OllamaPlannerConfig,
    PlannerOutputError,
)


@dataclass(frozen=True)
class PlannerCase:
    case_id: str
    category: str
    question: str
    min_nodes: int
    max_nodes: int
    required_terms: tuple[str, ...] = ()
    require_dependency: bool = False
    empty_plan: bool = False


def cases() -> list[PlannerCase]:
    return [
        PlannerCase(
            "P01",
            "simple_fact",
            "What is the title of Figure 3?",
            0,
            0,
            ("figure 3", "title"),
            empty_plan=True,
        ),
        PlannerCase(
            "P02",
            "parallel_facts",
            "What were the revenues in 2022 and 2023?",
            2,
            2,
            ("2022", "2023", "revenue"),
        ),
        PlannerCase(
            "P03",
            "root_calculation",
            "How much did revenue change from 2022 to 2023?",
            2,
            2,
            ("2022", "2023", "revenue"),
        ),
        PlannerCase(
            "P04",
            "true_dependency",
            "Which system has the highest accuracy, and what method does that system use?",
            2,
            3,
            ("highest", "method"),
            True,
        ),
        PlannerCase(
            "P05",
            "conservative_ambiguous",
            "Which approach is best in these situations, and why?",
            0,
            0,
            ("best",),
            empty_plan=True,
        ),
        PlannerCase(
            "P06",
            "single_local_comparison",
            "According to Table 2, which method has the highest F1 score?",
            0,
            0,
            ("table 2", "highest", "f1"),
            empty_plan=True,
        ),
    ]


def judge(case: PlannerCase, plan: Any) -> tuple[bool, list[str]]:
    problems: list[str] = []
    count = len(plan.subquestions)
    if not case.min_nodes <= count <= case.max_nodes:
        problems.append(
            f"expected {case.min_nodes}..{case.max_nodes} nodes, got {count}"
        )
    # An empty plan intentionally has no SubQuestion text. In that case,
    # scope preservation is checked against the verbatim original question.
    combined = (
        plan.original_question.casefold()
        if case.empty_plan
        else " ".join(item.text.casefold() for item in plan.subquestions)
    )
    for term in case.required_terms:
        if term.casefold() not in combined:
            problems.append(f"missing required scope term: {term}")
    has_dependency = any(item.depends_on for item in plan.subquestions)
    if case.require_dependency and not has_dependency:
        problems.append("expected at least one true dependency")
    return not problems, problems


def run_pass(
    evaluation_cases: list[PlannerCase],
    *,
    pass_name: str,
    model: str,
    base_url: str,
    timeout: float,
) -> list[dict[str, Any]]:
    backend = OllamaPlannerBackend(
        OllamaPlannerConfig(
            model=model,
            base_url=base_url,
            timeout_seconds=timeout,
        )
    )
    planner = InitialPlanner(backend)
    records: list[dict[str, Any]] = []
    for index, case in enumerate(evaluation_cases, start=1):
        print(f"[{pass_name}] {index:02d}/{len(evaluation_cases)} {case.case_id}", flush=True)
        started = time.perf_counter()
        try:
            plan = planner.create_plan(case.question)
            passed, problems = judge(case, plan)
            records.append(
                {
                    "pass": pass_name,
                    "case_id": case.case_id,
                    "category": case.category,
                    "question": case.question,
                    "valid_output": True,
                    "passed": passed,
                    "problems": problems,
                    "plan": plan.model_dump(mode="json"),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )
        except (PlannerOutputError, ValueError, OSError) as exc:
            records.append(
                {
                    "pass": pass_name,
                    "case_id": case.case_id,
                    "category": case.category,
                    "question": case.question,
                    "valid_output": False,
                    "passed": False,
                    "problems": [f"{type(exc).__name__}: {exc}"],
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--passes", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".runlogs/planner_mock_eval/results.json"),
    )
    args = parser.parse_args()

    evaluation_cases = cases()
    records: list[dict[str, Any]] = []
    for pass_name in ("A", "B")[: args.passes]:
        records.extend(
            run_pass(
                evaluation_cases,
                pass_name=pass_name,
                model=args.model,
                base_url=args.base_url,
                timeout=args.timeout,
            )
        )
    report = {
        "component": "planner",
        "model": args.model,
        "prompt_version": INITIAL_PLANNER_PROMPT_VERSION,
        "case_count": len(evaluation_cases),
        "generation_count": len(records),
        "valid_count": sum(bool(item["valid_output"]) for item in records),
        "passed_count": sum(bool(item["passed"]) for item in records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
