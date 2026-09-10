"""Compare base and SFT Controller policies on reviewed training states.

This evaluator never opens documents, executes actions, or reads benchmark
Gold answers.  It replays only exported ControllerInput states and reports
contract-level policy metrics against the reviewed Teacher action.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any

from pydantic import TypeAdapter

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from softdoc.controller import ControllerAction, ControllerInput, validate_controller_action
from softdoc.controller_ollama import OllamaControllerError, VLLMControllerBackend
from softdoc.openai_compatible import OpenAICompatibleConfig
from softdoc.training_data import load_sft_jsonl


ACTION_ADAPTER = TypeAdapter(ControllerAction)


def _canonical(action: ControllerAction) -> str:
    return json.dumps(
        action.model_dump(mode="json", exclude_none=True),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _route_signature(action: ControllerAction) -> str:
    """Compare the selected route while allowing free-text wording variants."""

    payload = action.model_dump(mode="json", exclude_none=True)
    payload.pop("local_problem", None)
    if payload.get("action") == "SEARCH" and payload.get("operation") == "new":
        payload.pop("query", None)
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _first_attempt_quality(
    backend: VLLMControllerBackend,
    controller_input: ControllerInput,
    final_action: ControllerAction | None,
) -> tuple[bool, bool, bool]:
    rejected = backend.last_rejected_attempts
    if not rejected:
        valid = final_action is not None
        return valid, valid, valid
    raw = rejected[0].get("raw_content", "")
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return False, False, False
    if not isinstance(payload, dict):
        return True, False, False
    try:
        action = ACTION_ADAPTER.validate_python(payload)
    except Exception:
        return True, False, False
    try:
        validate_controller_action(action, controller_input)
    except ValueError:
        return True, True, False
    return True, True, True


def _parse_model_spec(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be LABEL=SERVER_MODEL_NAME")
    label, model = value.split("=", 1)
    if not label.strip() or not model.strip():
        raise argparse.ArgumentTypeError("model label and name must be nonblank")
    return label.strip(), model.strip()


def evaluate(
    *,
    data: Path,
    model_specs: list[tuple[str, str]],
    base_url: str,
    output: Path,
    max_tokens: int,
    timeout: float,
    limit: int | None,
) -> dict[str, Any]:
    examples = [item for item in load_sft_jsonl(data) if item.component.value == "controller"]
    if limit is not None:
        examples = examples[:limit]
    if not examples:
        raise ValueError("No Controller examples were found")

    output.mkdir(parents=True, exist_ok=False)
    all_results: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for label, model in model_specs:
        backend = VLLMControllerBackend(
            OpenAICompatibleConfig(
                model=model,
                base_url=base_url,
                timeout_seconds=timeout,
                max_tokens=max_tokens,
                temperature=0.0,
                seed=20260910,
            )
        )
        counters: Counter[str] = Counter()
        action_distribution: Counter[str] = Counter()
        previous_by_session: dict[str, str] = {}
        elapsed_total = 0.0
        for index, example in enumerate(examples, 1):
            controller_input = ControllerInput.model_validate_json(example.input_text)
            teacher_action = ACTION_ADAPTER.validate_python(example.target)
            started = time.perf_counter()
            predicted: ControllerAction | None = None
            error: str | None = None
            try:
                predicted = backend.generate(controller_input).action
            except OllamaControllerError as exc:
                error = str(exc)
            elapsed = time.perf_counter() - started
            elapsed_total += elapsed
            first_json, first_schema, first_visible = _first_attempt_quality(
                backend, controller_input, predicted
            )
            counters["first_json_valid"] += first_json
            counters["first_action_valid"] += first_schema
            counters["first_visible_id_valid"] += first_visible

            exact_match = False
            route_match = False
            repeat = False
            predicted_payload: dict[str, Any] | None = None
            if predicted is not None:
                counters["final_valid"] += 1
                predicted_payload = predicted.model_dump(mode="json", exclude_none=True)
                exact_match = _canonical(predicted) == _canonical(teacher_action)
                route_match = _route_signature(predicted) == _route_signature(teacher_action)
                counters["teacher_exact_match"] += exact_match
                counters["teacher_route_match"] += route_match
                action_name = str(predicted_payload["action"])
                if action_name == "SEARCH":
                    action_name += f":{predicted_payload['operation']}"
                action_distribution[action_name] += 1
                signature = _route_signature(predicted)
                session_id = controller_input.reading_session_id
                repeat = previous_by_session.get(session_id) == signature
                counters["consecutive_prediction_repeat"] += repeat
                previous_by_session[session_id] = signature

            all_results.append(
                {
                    "model_label": label,
                    "model": model,
                    "index": index,
                    "example_id": example.example_id,
                    "reading_session_id": controller_input.reading_session_id,
                    "first_attempt_json_valid": first_json,
                    "first_attempt_action_valid": first_schema,
                    "first_attempt_visible_id_valid": first_visible,
                    "final_valid": predicted is not None,
                    "teacher_exact_match": exact_match,
                    "teacher_route_match": route_match,
                    "consecutive_prediction_repeat": repeat,
                    "teacher_action": teacher_action.model_dump(mode="json", exclude_none=True),
                    "predicted_action": predicted_payload,
                    "elapsed_seconds": round(elapsed, 3),
                    "error": error,
                }
            )

        total = len(examples)
        summaries[label] = {
            "model": model,
            "examples": total,
            "first_attempt_json_valid_rate": counters["first_json_valid"] / total,
            "first_attempt_action_valid_rate": counters["first_action_valid"] / total,
            "first_attempt_visible_id_valid_rate": counters["first_visible_id_valid"] / total,
            "final_valid_rate_after_repair": counters["final_valid"] / total,
            "teacher_exact_agreement_rate": counters["teacher_exact_match"] / total,
            "teacher_route_agreement_rate": counters["teacher_route_match"] / total,
            "consecutive_prediction_repeat_rate": (
                counters["consecutive_prediction_repeat"] / total
            ),
            "action_distribution": dict(sorted(action_distribution.items())),
            "mean_seconds": elapsed_total / total,
        }

    with (output / "results.jsonl").open("w", encoding="utf-8") as handle:
        for record in all_results:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    report = {
        "schema_version": "controller-offline-comparison-v0.1",
        "data": str(data),
        "summaries": summaries,
        "notes": {
            "teacher_route_agreement": (
                "Ignores local_problem wording and SEARCH-new query wording; "
                "all typed handles and operation choices must still match."
            ),
            "repeat_rate": (
                "Consecutive structural prediction repeats within exported "
                "session order; interpret only when accepted steps are contiguous."
            ),
            "scope": "Controller states only; not an end-to-end QA evaluation.",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        type=_parse_model_spec,
        required=True,
        help="Repeat LABEL=SERVER_MODEL_NAME for base and adapter aliases.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = evaluate(
        data=args.data,
        model_specs=args.model,
        base_url=args.base_url,
        output=args.output,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
