"""Select a deterministic, document-diverse pilot from non-diagnostic Train cases."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("case_id"), str):
                raise ValueError(f"{path}:{line_number}: invalid case row")
            rows.append(row)
    return rows


def select_nonhard_pilot(
    train_rows: list[dict[str, Any]],
    hard_ids: set[str],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    eligible = [row for row in train_rows if row["case_id"] not in hard_ids]
    if count < 1 or count > len(eligible):
        raise ValueError(f"count must be within 1..{len(eligible)}")

    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        document_id = row.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError(f"case {row['case_id']} has no document_id")
        by_document[document_id].append(row)

    rng = random.Random(seed)
    document_ids = sorted(by_document)
    rng.shuffle(document_ids)
    queues: dict[str, deque[dict[str, Any]]] = {}
    for document_id in document_ids:
        rows = sorted(by_document[document_id], key=lambda row: row["case_id"])
        rng.shuffle(rows)
        queues[document_id] = deque(rows)

    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        made_progress = False
        for document_id in document_ids:
            if queues[document_id]:
                selected.append(queues[document_id].popleft())
                made_progress = True
                if len(selected) == count:
                    break
        if not made_progress:
            raise AssertionError("eligible cases were exhausted before reaching count")

    return sorted(selected, key=lambda row: (row["document_id"], row["case_id"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--hard-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args(argv)

    train_rows = _load_jsonl(args.train)
    hard_rows = _load_jsonl(args.hard_cases)
    hard_ids = {row["case_id"] for row in hard_rows}
    train_hard_ids = {row["case_id"] for row in train_rows}.intersection(hard_ids)
    selected = select_nonhard_pilot(
        train_rows, hard_ids, count=args.count, seed=args.seed
    )
    selected_ids = {row["case_id"] for row in selected}
    if selected_ids.intersection(hard_ids):
        raise AssertionError("pilot overlaps the diagnostic hard-case set")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    runtime_rows = [
        {
            key: row[key]
            for key in ("case_id", "question_id", "document_dir", "question", "run_key")
            if key in row
        }
        for row in selected
    ]
    serialized = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in runtime_rows
    )
    args.output.write_text(serialized, encoding="utf-8", newline="\n")

    per_document = Counter(row["document_id"] for row in selected)
    metadata = {
        "schema_version": "train-nonhard-pilot-selection-v0.1",
        "seed": args.seed,
        "selection_policy": (
            "Exclude all original unresolved-177 diagnostic cases, then use "
            "seeded document-round-robin sampling over the remaining Train cases."
        ),
        "train_case_count": len(train_rows),
        "excluded_train_hard_case_count": len(train_hard_ids),
        "eligible_nonhard_case_count": len(train_rows) - len(train_hard_ids),
        "selected_case_count": len(selected),
        "selected_document_count": len(per_document),
        "hard_overlap_count": 0,
        "selected_cases_sha256": sha256(serialized.encode("utf-8")).hexdigest(),
        "selected_cases_by_document": dict(sorted(per_document.items())),
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
