"""Compare deterministic BM25/Dense candidate merge policies from saved runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


K_VALUES = (1, 3, 5, 10, 20, 50)
Entry = dict[str, Any]
Policy = Callable[[list[Entry], list[Entry]], list[Entry]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_jsonl", type=Path)
    parser.add_argument("--corpus-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.results_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    config = json.loads(args.corpus_config.read_text(encoding="utf-8"))
    existing = set(config["existing_documents"])
    extension = {item["doc_id"] for item in config["extension_documents"]}
    splits = {
        "all": rows,
        "existing_14": [row for row in rows if row["doc_id"] in existing],
        "extension_14": [row for row in rows if row["doc_id"] in extension],
    }
    policies = _policies()
    report = {
        "scope": (
            "Post-hoc comparison over saved top-50 BM25 and Dense Element lists. "
            "All policies prepend Exact handles and deduplicate by candidate_id."
        ),
        "limitation": (
            "This diagnostic uses truncated top-50 source lists. It must not be used "
            "alone to choose production RRF parameters because truncation removes "
            "single-source tail contributions. The full-ranking evaluator is authoritative."
        ),
        "selection_rule": (
            "Prefer first-batch hit@5, then MRR and hit@20; require similar "
            "behaviour on existing and extension document splits."
        ),
        "splits": {
            split: {
                name: _metrics(split_rows, policy)
                for name, policy in policies.items()
            }
            for split, split_rows in splits.items()
        },
    }
    report["ranking"] = sorted(
        policies,
        key=lambda name: (
            -report["splits"]["all"][name]["hit_rate"]["5"],
            -report["splits"]["all"][name]["mrr"],
            -report["splits"]["all"][name]["hit_rate"]["20"],
            name,
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "candidate_policy_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "candidate_policy_comparison.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _policies() -> dict[str, Policy]:
    return {
        "bm25_only": lambda bm25, dense: _deduplicate(bm25),
        "dense_only": lambda bm25, dense: _deduplicate(dense),
        "round_robin_bm25_first": lambda bm25, dense: _interleave(
            bm25, dense, ("bm25", "dense")
        ),
        "round_robin_dense_first": lambda bm25, dense: _interleave(
            bm25, dense, ("dense", "bm25")
        ),
        "interleave_bm25_3_dense_2": lambda bm25, dense: _interleave(
            bm25, dense, ("bm25", "bm25", "bm25", "dense", "dense")
        ),
        "interleave_bm25_2_dense_3": lambda bm25, dense: _interleave(
            bm25, dense, ("bm25", "bm25", "dense", "dense", "dense")
        ),
        "interleave_bm25_1_dense_2": lambda bm25, dense: _interleave(
            bm25, dense, ("bm25", "dense", "dense")
        ),
        "interleave_bm25_1_dense_3": lambda bm25, dense: _interleave(
            bm25, dense, ("bm25", "dense", "dense", "dense")
        ),
        "rrf_equal_k60": lambda bm25, dense: _rrf(bm25, dense, 60, 1.0, 1.0),
        "rrf_dense_1_25_k60": lambda bm25, dense: _rrf(
            bm25, dense, 60, 1.0, 1.25
        ),
        "rrf_dense_1_5_k60": lambda bm25, dense: _rrf(
            bm25, dense, 60, 1.0, 1.5
        ),
        "rrf_equal_k20": lambda bm25, dense: _rrf(bm25, dense, 20, 1.0, 1.0),
        "rrf_dense_1_25_k20": lambda bm25, dense: _rrf(
            bm25, dense, 20, 1.0, 1.25
        ),
    }


def _interleave(
    bm25: list[Entry],
    dense: list[Entry],
    pattern: tuple[str, ...],
) -> list[Entry]:
    sources = {"bm25": bm25, "dense": dense}
    cursors = {"bm25": 0, "dense": 0}
    result: list[Entry] = []
    seen: set[str] = set()
    while any(cursors[name] < len(sources[name]) for name in sources):
        progress = False
        for name in pattern:
            collection = sources[name]
            while cursors[name] < len(collection):
                entry = collection[cursors[name]]
                cursors[name] += 1
                candidate_id = str(entry["candidate_id"])
                if candidate_id in seen:
                    continue
                seen.add(candidate_id)
                result.append(entry)
                progress = True
                break
        if not progress:
            break
    return result


def _rrf(
    bm25: list[Entry],
    dense: list[Entry],
    k: int,
    bm25_weight: float,
    dense_weight: float,
) -> list[Entry]:
    scores: dict[str, float] = {}
    best_entry: dict[str, Entry] = {}
    best_rank: dict[str, int] = {}
    for collection, weight in ((bm25, bm25_weight), (dense, dense_weight)):
        for fallback_rank, entry in enumerate(collection, start=1):
            candidate_id = str(entry["candidate_id"])
            rank = int(entry.get("source_rank") or fallback_rank)
            scores[candidate_id] = scores.get(candidate_id, 0.0) + weight / (k + rank)
            if rank < best_rank.get(candidate_id, 1_000_000):
                best_rank[candidate_id] = rank
                best_entry[candidate_id] = entry
    return [
        best_entry[candidate_id]
        for candidate_id in sorted(
            scores,
            key=lambda item: (
                -scores[item],
                best_rank[item],
                int(best_entry[item]["page_number"]),
                item,
            ),
        )
    ]


def _deduplicate(entries: list[Entry]) -> list[Entry]:
    result: list[Entry] = []
    seen: set[str] = set()
    for entry in entries:
        candidate_id = str(entry["candidate_id"])
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        result.append(entry)
    return result


def _ranking(row: dict[str, Any], policy: Policy) -> list[Entry]:
    exact = row["exact"].get("entries", [])
    merged = policy(
        row["bm25"].get("top_candidates", []),
        row["dense"].get("top_candidates", []),
    )
    return _deduplicate([*exact, *merged])


def _metrics(rows: list[dict[str, Any]], policy: Policy) -> dict[str, Any]:
    evaluable = [row for row in rows if row.get("has_gold_pages")]
    ranks: list[int | None] = []
    for row in evaluable:
        gold = {int(page) for page in row["gold_pages"]}
        rank = next(
            (
                index
                for index, entry in enumerate(_ranking(row, policy), start=1)
                if int(entry["page_number"]) in gold
            ),
            None,
        )
        ranks.append(rank)
    return {
        "questions": len(rows),
        "evaluable_questions": len(evaluable),
        "hit_rate": {
            str(k): sum(rank is not None and rank <= k for rank in ranks) / len(ranks)
            if ranks
            else None
            for k in K_VALUES
        },
        "mrr": sum(1.0 / rank if rank else 0.0 for rank in ranks) / len(ranks)
        if ranks
        else None,
        "mean_first_gold_rank_when_found": (
            sum(rank for rank in ranks if rank is not None)
            / sum(rank is not None for rank in ranks)
            if any(rank is not None for rank in ranks)
            else None
        ),
        "unretrieved_within_saved_union": sum(rank is None for rank in ranks),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Candidate policy comparison",
        "",
        str(report["scope"]),
        "",
        f"Limitation: {report['limitation']}",
        "",
        f"Selection rule: {report['selection_rule']}",
        "",
        "## Overall",
        "",
        "| Policy | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | Hit@50 | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in report["ranking"]:
        metrics = report["splits"]["all"][name]
        hit = metrics["hit_rate"]
        lines.append(
            f"| {name} | {hit['1']:.4f} | {hit['3']:.4f} | {hit['5']:.4f} | "
            f"{hit['10']:.4f} | {hit['20']:.4f} | {hit['50']:.4f} | "
            f"{metrics['mrr']:.4f} |"
        )
    for split in ("existing_14", "extension_14"):
        lines.extend(
            [
                "",
                f"## {split}",
                "",
                "| Policy | Hit@5 | Hit@20 | MRR |",
                "|---|---:|---:|---:|",
            ]
        )
        for name in report["ranking"]:
            metrics = report["splits"][split][name]
            lines.append(
                f"| {name} | {metrics['hit_rate']['5']:.4f} | "
                f"{metrics['hit_rate']['20']:.4f} | {metrics['mrr']:.4f} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
