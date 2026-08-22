"""Run the frozen observation-only Visual Reader prompt on 25 real cases."""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
import time
from typing import Any
from urllib import request as urlrequest

from pydantic import ValidationError

from softdoc.ids import stable_digest
from softdoc.serialization import load_document
from softdoc.visual_reading import (
    VISUAL_READER_SYSTEM_PROMPT,
    VisualInput,
    VisualReadRequest,
    VisualReadResult,
    validate_visual_read_result,
    visual_reader_user_prompt,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "visual_reader_v0_probe_25.json"
DEFAULT_CORPUS = ROOT / "data" / "processed" / "representative_28" / "softdoc"
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "representative_28"
    / "reports"
    / "visual_reader_v0_probe_25"
)
VISUAL_TYPES = {"figure", "chart", "equation"}


def _document_dir(corpus_root: Path, doc_id: str) -> Path:
    name = Path(doc_id).stem
    direct = corpus_root / name
    if direct.is_dir():
        return direct
    matches = [path for path in corpus_root.iterdir() if path.name.casefold() == name.casefold()]
    if len(matches) != 1:
        raise FileNotFoundError(f"cannot resolve SoftDoc directory for {doc_id!r}")
    return matches[0]


def _resolve_path(document_dir: Path, value: Path | None) -> Path | None:
    if value is None:
        return None
    return value if value.is_absolute() else document_dir / value


def _largest_visual_element(document: Any, page_id: str) -> Any | None:
    candidates = [
        element
        for element in document.elements
        if element.page_id == page_id
        and element.element_type.value in VISUAL_TYPES
        and element.image_path is not None
    ]
    if not candidates:
        return None

    def area(element: Any) -> float:
        x1, y1, x2, y2 = element.bbox.normalized
        return (x2 - x1) * (y2 - y1)

    return max(candidates, key=lambda item: (area(item), item.element_id))


def _request_for_case(
    case: dict[str, Any], corpus_root: Path
) -> tuple[VisualReadRequest, list[Path], list[dict[str, Any]]]:
    visual_inputs: list[VisualInput] = []
    attached_images: list[Path] = []
    resolution: list[dict[str, Any]] = []
    document_ids: set[str] = set()
    source_names: set[str] = set()
    for index, input_spec in enumerate(case["inputs"], start=1):
        document_dir = _document_dir(corpus_root, input_spec["doc_id"])
        document = load_document(document_dir)
        document_ids.add(document.document_id)
        source_names.add(input_spec["doc_id"])
        page = next(
            (item for item in document.pages if item.page_number == input_spec["page_number"]),
            None,
        )
        if page is None:
            raise ValueError(
                f"{case['probe_id']}: page {input_spec['page_number']} "
                f"not found in {input_spec['doc_id']}"
            )
        page_image = _resolve_path(document_dir, page.image_path)
        if page_image is None or not page_image.is_file():
            raise FileNotFoundError(
                f"{case['probe_id']}: missing page image for page {page.page_number}"
            )

        selected_element = None
        attached_image = page_image
        if input_spec["selection"] == "largest_visual_element":
            selected_element = _largest_visual_element(document, page.page_id)
            if selected_element is not None:
                candidate = _resolve_path(document_dir, selected_element.image_path)
                if candidate is not None and candidate.is_file():
                    attached_image = candidate

        bbox = None
        element_id = None
        element_type = None
        if selected_element is not None and attached_image != page_image:
            bbox = tuple(float(value) for value in selected_element.bbox.normalized)
            element_id = selected_element.element_id
            element_type = selected_element.element_type.value

        input_id = f"I{index}"
        visual_asset_id = "visual:" + stable_digest(
            next(iter(document_ids)), page.page_id, element_id, attached_image.name
        )
        visual_inputs.append(
            VisualInput(
                input_id=input_id,
                visual_asset_id=visual_asset_id,
                page_id=page.page_id,
                page_number=page.page_number,
                display_page_label=page.display_page_label,
                page_image_path=page.image_path,
                element_id=element_id,
                element_type=element_type,
                bbox=bbox,
            )
        )
        attached_images.append(attached_image.resolve())
        resolution.append(
            {
                "input_id": input_id,
                "visual_asset_id": visual_asset_id,
                "doc_id": input_spec["doc_id"],
                "selection_requested": input_spec["selection"],
                "selection_resolved": "element" if element_id else "page",
                "resolved_image_path": str(attached_image.resolve()),
                "element_id": element_id,
                "element_type": element_type,
            }
        )
    if len(document_ids) != 1 or len(source_names) != 1:
        raise ValueError("one read request must stay within one document")
    request = VisualReadRequest(
        action_id=f"action:probe:{case['probe_id']}",
        subquestion_id=f"{case['probe_id']}:SQ",
        document_id=next(iter(document_ids)),
        source_name=next(iter(source_names)),
        problem=case["problem"],
        visual_inputs=visual_inputs,
    )
    return request, attached_images, resolution


def _encoded_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _call_ollama(
    model: str,
    read_request: VisualReadRequest,
    images: list[Path],
    timeout_seconds: int,
) -> tuple[str, dict[str, Any], float]:
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        # Ollama accepts a JSON Schema object here.  This constrains the actual
        # decoding process; a prose request to "return JSON" cannot prevent a
        # small model from emitting malformed or schema-incompatible output.
        "format": VisualReadResult.model_json_schema(),
        "messages": [
            {"role": "system", "content": VISUAL_READER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": visual_reader_user_prompt(read_request),
                "images": [_encoded_image(path) for path in images],
            },
        ],
        "options": {"temperature": 0, "seed": 42, "num_ctx": 8192},
    }
    body = json.dumps(payload).encode("utf-8")
    started = time.perf_counter()
    http_request = urlrequest.Request(
        "http://127.0.0.1:11434/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(http_request, timeout=timeout_seconds) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    elapsed = time.perf_counter() - started
    message = response_payload["message"]
    raw_output = message.get("content") or message.get("thinking") or ""
    response_payload["visual_reader_output_channel"] = (
        "content" if message.get("content") else "thinking"
    )
    return raw_output, response_payload, elapsed


def _parse_json_output(raw_output: str) -> Any:
    stripped = raw_output.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    return json.loads(stripped)


def _validate_output(
    raw_output: str, read_request: VisualReadRequest
) -> tuple[VisualReadResult | None, list[str]]:
    errors: list[str] = []
    try:
        parsed = _parse_json_output(raw_output)
    except (json.JSONDecodeError, TypeError) as exc:
        return None, [f"invalid_json: {exc}"]
    try:
        result = VisualReadResult.model_validate(parsed)
    except ValidationError as exc:
        return None, [f"schema_validation: {error['msg']} at {error['loc']}" for error in exc.errors()]
    try:
        validate_visual_read_result(read_request, result)
    except ValueError as exc:
        errors.append(str(exc))
    return result, errors


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_review_html(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    cards: list[str] = []
    for row in rows:
        images = "".join(
            f'<figure><figcaption>{html.escape(item["input_id"])} · '
            f'{html.escape(item["selection_resolved"])} · '
            f'{html.escape(Path(item["resolved_image_path"]).name)}</figcaption>'
            f'<a href="{Path(item["resolved_image_path"]).as_uri()}">'
            f'<img src="{Path(item["resolved_image_path"]).as_uri()}" loading="lazy"></a></figure>'
            for item in row["image_resolution"]
        )
        validation = "PASS" if not row.get("validation_errors") else "FAIL"
        cards.append(
            f'<article id="{html.escape(row["probe_id"])}">'
            f'<h2>{html.escape(row["probe_id"])} · '
            f'{html.escape(row["category"])} · schema {validation}</h2>'
            f'<p><strong>Problem:</strong> {html.escape(row["problem"])}</p>'
            f'<p><strong>Expected probe fact:</strong> '
            f'{html.escape(str(row.get("expected")))}</p>'
            f'<div class="images">{images}</div>'
            f'<h3>Validated output</h3><pre>{html.escape(json.dumps(row.get("validated_output"), ensure_ascii=False, indent=2))}</pre>'
            f'<h3>Validation errors</h3><pre>{html.escape(json.dumps(row.get("validation_errors", []), ensure_ascii=False, indent=2))}</pre>'
            f'<details><summary>Full request and raw output</summary>'
            f'<pre>{html.escape(json.dumps(row["request"], ensure_ascii=False, indent=2))}</pre>'
            f'<pre>{html.escape(row.get("raw_output", ""))}</pre></details></article>'
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Visual Reader v0 probe</title>
<style>
body{{font-family:system-ui,sans-serif;background:#111827;color:#e5e7eb;margin:0;padding:24px}}
article{{background:#1f2937;margin:0 0 28px;padding:20px;border-radius:12px}}
.images{{display:flex;gap:16px;align-items:flex-start;overflow:auto}}
figure{{margin:0;min-width:320px;max-width:48%}} img{{width:100%;max-height:760px;object-fit:contain;background:white}}
figcaption{{margin:0 0 8px;color:#fbbf24}} pre{{white-space:pre-wrap;background:#0f172a;padding:14px;overflow:auto}}
h2{{color:#93c5fd}} strong{{color:#f9fafb}}
</style></head><body><h1>Visual Reader v0 · 25 real-image probe</h1>{''.join(cards)}</body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def _write_summary(output_dir: Path, rows: list[dict[str, Any]], model: str) -> None:
    categories = Counter(row["category"] for row in rows)
    valid = sum(not row.get("validation_errors") for row in rows)
    observation_counts = [
        len((row.get("validated_output") or {}).get("observations", []))
        for row in rows
    ]
    limitation_cases = sum(
        bool((row.get("validated_output") or {}).get("limitations")) for row in rows
    )
    lines = [
        "# Visual Reader v0 probe",
        "",
        f"- Model: `{model}`",
        f"- Requests: {len(rows)}",
        f"- Categories: {dict(categories)}",
        f"- Schema-valid: {valid}/{len(rows)}",
        f"- Cases with limitations: {limitation_cases}",
        f"- Total observations: {sum(observation_counts)}",
        "",
        "This is a development probe, not an accuracy benchmark. Expected facts are",
        "kept outside the model request and are shown only for manual review.",
        "",
        "Open `index.html` to inspect every supplied image and raw output.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    model = args.model or config["model"]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved: list[dict[str, Any]] = []
    for case in config["cases"]:
        read_request, images, image_resolution = _request_for_case(
            case, args.corpus_root.resolve()
        )
        row = {
            "probe_id": case["probe_id"],
            "category": case["category"],
            "dataset_index": case.get("dataset_index"),
            "comparison_group": case.get("comparison_group"),
            "parent_problem": case.get("parent_problem"),
            "problem": case["problem"],
            "expected": case.get("expected"),
            "request": read_request.model_dump(mode="json"),
            "image_resolution": image_resolution,
        }
        if not args.prepare_only:
            try:
                raw_output, response_payload, elapsed = _call_ollama(
                    model, read_request, images, args.timeout_seconds
                )
                validated, validation_errors = _validate_output(raw_output, read_request)
                row.update(
                    {
                        "raw_output": raw_output,
                        "validated_output": validated.model_dump(mode="json")
                        if validated is not None
                        else None,
                        "validation_errors": validation_errors,
                        "elapsed_seconds": round(elapsed, 3),
                        "output_channel": response_payload.get(
                            "visual_reader_output_channel"
                        ),
                        "ollama_metrics": {
                            key: response_payload.get(key)
                            for key in (
                                "total_duration",
                                "load_duration",
                                "prompt_eval_count",
                                "prompt_eval_duration",
                                "eval_count",
                                "eval_duration",
                            )
                        },
                    }
                )
            except Exception as exc:  # keep the remaining real cases runnable
                row.update(
                    {
                        "raw_output": "",
                        "validated_output": None,
                        "validation_errors": [f"runtime_error: {type(exc).__name__}: {exc}"],
                    }
                )
        resolved.append(row)
        print(
            f"{case['probe_id']} {case['category']} "
            f"images={len(images)} errors={len(row.get('validation_errors', []))}",
            flush=True,
        )

    metadata = {
        "probe_name": config["name"],
        "model": model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prepare_only": args.prepare_only,
        "system_prompt": VISUAL_READER_SYSTEM_PROMPT,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_jsonl(output_dir / "results.jsonl", resolved)
    _write_review_html(output_dir, resolved)
    _write_summary(output_dir, resolved, model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
