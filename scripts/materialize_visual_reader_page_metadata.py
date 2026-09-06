"""Materialize canonical page-boundary metadata in the frozen Visual Reader suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from softdoc.serialization import load_document
from softdoc.visual_reading import VisualReadRequest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "evals"
    / "prompts"
    / "visual_reader"
    / "model_inputs"
    / "visual_reader_cases_v1.jsonl"
)
DEFAULT_CORPUS = ROOT / "data" / "processed" / "representative_28" / "softdoc"


def _document_dir(corpus_root: Path, source_name: str) -> Path:
    expected = Path(source_name).stem.casefold()
    matches = [path for path in corpus_root.iterdir() if path.name.casefold() == expected]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one SoftDoc directory for {source_name!r}, found {len(matches)}"
        )
    return matches[0]


def _page_count(request: dict[str, Any], corpus_root: Path) -> int:
    if request["source_name"] == "controlled_visual_reader_cases":
        return max(
            int(item.get("physical_page_number", item.get("page_number")))
            for item in request["visual_inputs"]
        )
    return len(load_document(_document_dir(corpus_root, request["source_name"])).pages)


def materialize(path: Path, corpus_root: Path) -> int:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        request = row["request"]
        page_count = _page_count(request, corpus_root)
        for visual_input in request["visual_inputs"]:
            old_page_number = visual_input.pop("page_number", None)
            physical_page_number = int(
                visual_input.get("physical_page_number", old_page_number)
            )
            visual_input["physical_page_number"] = physical_page_number
            visual_input["document_page_count"] = page_count
            visual_input["is_last_page"] = physical_page_number == page_count
        VisualReadRequest.model_validate(request)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args()
    count = materialize(args.input.resolve(), args.corpus_root.resolve())
    print(f"materialized page metadata for {count} Visual Reader cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
