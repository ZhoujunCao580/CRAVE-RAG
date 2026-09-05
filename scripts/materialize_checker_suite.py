"""Materialize the frozen Checker component suite without calling a model."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.evaluate_checker_mock import Case, build_cases
from softdoc.checking_prompt import CHECKER_PROMPT_VERSION


SUITE_ROOT = Path("evals/prompts/checker")
MODEL_INPUT_PATH = SUITE_ROOT / "model_inputs/checker_cases_v1.jsonl"
REVIEW_GOLD_PATH = SUITE_ROOT / "review_only/checker_gold_v1.jsonl"
MANIFEST_PATH = SUITE_ROOT / "suite_manifest.json"


def case_split(case: Case) -> str:
    return "controlled_boundary" if int(case.case_id[1:]) >= 21 else "contract_regression"


def model_row(case: Case) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "checker_input": case.checker_input.model_dump(mode="json"),
    }


def gold_row(case: Case) -> dict[str, Any]:
    expected = case.expected
    return {
        "case_id": case.case_id,
        "split": case_split(case),
        "category": case.category,
        "source": "scripts/evaluate_checker_mock.py",
        "description": case.description,
        "expected": {
            "used_for_evidence": expected.used,
            "add_count": expected.add,
            "replace_count": expected.replace,
            "remove_count": expected.remove,
            "current_target_status": expected.target_status,
            "root_status": expected.root_status,
            "gap_required": expected.gap_required,
            "gap_required_terms": list(expected.gap_terms),
            "gap_forbidden_terms": list(expected.gap_forbidden_terms),
            "evidence_required_terms": list(expected.evidence_required_terms),
            "evidence_forbidden_terms": list(expected.evidence_forbidden_terms),
            "next_target": expected.next_target,
        },
        "chained_from": case.chained_from,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def materialize() -> None:
    cases = build_cases()
    if len(cases) != 27:
        raise ValueError(f"Expected 27 Checker cases, found {len(cases)}")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Checker case IDs must be unique")

    model_rows = [model_row(case) for case in cases]
    gold_rows = [gold_row(case) for case in cases]
    write_jsonl(MODEL_INPUT_PATH, model_rows)
    write_jsonl(REVIEW_GOLD_PATH, gold_rows)

    split_counts = Counter(case_split(case) for case in cases)
    category_counts = Counter(case.category for case in cases)
    manifest = {
        "suite_id": "checker-dev-suite-v1",
        "case_set_version": "checker-cases-v1",
        "prompt_version_under_test": CHECKER_PROMPT_VERSION,
        "case_count": len(cases),
        "model_input_file": MODEL_INPUT_PATH.relative_to(SUITE_ROOT).as_posix(),
        "review_gold_file": REVIEW_GOLD_PATH.relative_to(SUITE_ROOT).as_posix(),
        "split_counts": dict(sorted(split_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "design_notes": {
            "synthetic_only": True,
            "gold_is_not_model_visible": True,
            "c11_c12_form_a_two_turn_regression": True,
            "c21_preserves_single_current_target_updates": True,
            "c27_covers_state_only_root_finalization": True,
            "real_reader_to_checker_packets_still_required": True,
            "unseen_real_holdout_still_required": True,
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    materialize()
    print(f"Wrote 27 Checker cases under {SUITE_ROOT.resolve()}")
