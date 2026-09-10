"""Score persisted CRAVE-RAG answers with deterministic checks and an optional LLM judge.

This is a diagnostic scorer rather than the official MMLongBench answer
extractor.  Later output roots override earlier roots, which makes it possible
to combine a frozen batch with narrowly scoped recovery runs without modifying
the original artifacts.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


NOT_ANSWERABLE = {
    "not answerable",
    "not_answerable",
    "unanswerable",
    "cannot be answered",
}


def _clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip().casefold()
    text = re.sub(r"[\s\u00a0]+", " ", text)
    return text.rstrip(". ")


def _is_not_answerable(value: Any) -> bool:
    return _clean_text(value) in NOT_ANSWERABLE


def _parse_collection(value: Any) -> list[str] | None:
    if isinstance(value, (list, tuple)):
        parsed = list(value)
    else:
        text = str(value).strip()
        if not (text.startswith("[") and text.endswith("]")):
            return None
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return None
        if not isinstance(parsed, (list, tuple)):
            return None
    return sorted(_clean_text(item) for item in parsed)


def _deterministic_score(gold: Any, prediction: Any) -> tuple[bool | None, str]:
    gold_na = _is_not_answerable(gold)
    pred_na = _is_not_answerable(prediction)
    if gold_na or pred_na:
        return gold_na and pred_na, "not_answerable_exact"
    if _clean_text(gold) == _clean_text(prediction):
        return True, "normalized_exact"
    gold_list = _parse_collection(gold)
    pred_list = _parse_collection(prediction)
    if gold_list is not None and pred_list is not None:
        return gold_list == pred_list, "collection_equivalence"
    return None, "requires_semantic_judge"


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _judge_one(
    row: dict[str, Any], *, base_url: str, model: str, timeout: float
) -> tuple[bool, str, dict[str, Any]]:
    system = (
        "Judge whether a predicted answer is semantically equivalent to the gold answer. "
        "Ignore harmless formatting, punctuation, capitalization, explanation, and list syntax. "
        "Do not ignore wrong values, missing requested items, wrong units, wrong direction, or "
        "unsupported extra claims. Return only the requested JSON object."
    )
    user = json.dumps(
        {
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "predicted_answer": row["prediction"],
        },
        ensure_ascii=False,
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "semantic_answer_judgment",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "correct": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["correct", "reason"],
                    "additionalProperties": False,
                },
            },
        },
    }
    response = _post_json(
        f"{base_url.rstrip('/')}/chat/completions", payload, timeout
    )
    content = response["choices"][0]["message"]["content"]
    judgment = json.loads(content)
    metadata = {
        "finish_reason": response["choices"][0].get("finish_reason"),
        "usage": response.get("usage"),
    }
    return bool(judgment["correct"]), str(judgment["reason"]), metadata


def _read_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_predictions(roots: list[Path]) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for root in roots:
        for run_path in sorted(root.glob("Q*/reading_run.json")):
            run = json.loads(run_path.read_text(encoding="utf-8"))
            answer = run.get("answer") or {}
            predictions[run_path.parent.name] = {
                "prediction": answer.get("answer") if isinstance(answer, dict) else answer,
                "reading_status": run.get("status"),
                "run_path": str(run_path),
            }
    return predictions


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the project content-accuracy and generalized-F1 aggregates.

    The positive class is an answerable question.  A persisted ``ready`` run is
    treated as a positive prediction; terminal fallback runs persist
    ``Not answerable`` and are treated as negative predictions.  This matches
    the frozen Test baseline accounting in
    ``docs/MMLONGBENCH_SPLIT_AND_SCORECARD_CN.md``.
    """

    status_counts = Counter(row["reading_status"] for row in records)
    method_counts = Counter(row["scoring_method"] for row in records)
    correct = sum(bool(row["correct"]) for row in records)
    ready = [row for row in records if row["reading_status"] == "ready"]
    answerable = [
        row for row in records if not _is_not_answerable(row["gold_answer"])
    ]
    answerable_correct = [row for row in answerable if bool(row["correct"])]
    precision = len(answerable_correct) / len(ready) if ready else 0.0
    recall = len(answerable_correct) / len(answerable) if answerable else 0.0
    generalized_f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "case_count": len(records),
        "correct": correct,
        "accuracy": correct / len(records) if records else None,
        "gold_answerable_count": len(answerable),
        "gold_not_answerable_count": len(records) - len(answerable),
        "predicted_answerable_count": len(ready),
        "answerable_correct": len(answerable_correct),
        "precision": precision,
        "recall": recall,
        "generalized_f1": generalized_f1,
        "ready_count": len(ready),
        "ready_correct": sum(bool(row["correct"]) for row in ready),
        "ready_accuracy": (
            sum(bool(row["correct"]) for row in ready) / len(ready)
            if ready
            else None
        ),
        "status_counts": dict(status_counts),
        "scoring_method_counts": dict(method_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--gold-parquet", type=Path, required=True)
    parser.add_argument("--output-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="/workspace/models/Qwen3.5-27B")
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument(
        "--baseline-score",
        type=Path,
        help=(
            "Optional prior score JSON containing accuracy and generalized_f1. "
            "Only aggregate deltas are read; per-question baseline records are ignored."
        ),
    )
    args = parser.parse_args()

    import pyarrow.parquet as pq

    gold_rows = pq.read_table(args.gold_parquet).to_pylist()
    predictions = _read_predictions(args.output_roots)
    records: list[dict[str, Any]] = []
    for case in _read_cases(args.cases):
        case_id = case["case_id"]
        index = int(case_id.removeprefix("Q"))
        gold = gold_rows[index]
        pred = predictions.get(case_id, {})
        prediction = pred.get("prediction")
        result, method = _deterministic_score(gold["answer"], prediction)
        records.append(
            {
                "case_id": case_id,
                "question": gold["question"],
                "gold_answer": gold["answer"],
                "answer_format": gold.get("answer_format"),
                "prediction": prediction,
                "reading_status": pred.get("reading_status", "program_failure"),
                "run_path": pred.get("run_path"),
                "correct": result,
                "scoring_method": method,
                "reason": None,
                "judge_metadata": None,
            }
        )

    pending = [row for row in records if row["correct"] is None]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _judge_one,
                row,
                base_url=args.base_url,
                model=args.model,
                timeout=args.timeout,
            ): row
            for row in pending
        }
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            try:
                correct, reason, metadata = future.result()
            except Exception as exc:  # preserve scorer failures for audit
                row["correct"] = False
                row["scoring_method"] = "judge_failure"
                row["reason"] = f"{type(exc).__name__}: {exc}"
            else:
                row["correct"] = correct
                row["scoring_method"] = "llm_semantic_judge"
                row["reason"] = reason
                row["judge_metadata"] = metadata

    summary = _summarize_records(records)
    payload = {
        "schema_version": "semantic-answer-diagnostic-v0.1",
        **summary,
        "records": sorted(records, key=lambda row: int(row["case_id"][1:])),
    }
    if args.baseline_score is not None:
        baseline = json.loads(args.baseline_score.read_text(encoding="utf-8"))
        baseline_accuracy = baseline.get("accuracy")
        baseline_f1 = baseline.get("generalized_f1")
        if not isinstance(baseline_accuracy, (int, float)) or not isinstance(
            baseline_f1, (int, float)
        ):
            raise ValueError(
                "--baseline-score must contain numeric accuracy and generalized_f1"
            )
        payload["baseline_comparison"] = {
            "baseline_score": str(args.baseline_score),
            "baseline_accuracy": baseline_accuracy,
            "baseline_generalized_f1": baseline_f1,
            "accuracy_delta": summary["accuracy"] - baseline_accuracy,
            "generalized_f1_delta": summary["generalized_f1"] - baseline_f1,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
