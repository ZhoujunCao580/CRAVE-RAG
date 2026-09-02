"""Tune only the stage-3B winning weighted-RRF policy offline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_stage3_retrieval import (
    K_VALUES,
    _distribution,
    _ranking_metrics,
)
from softdoc.ids import stable_digest


DEFAULT_COMPARE_ROOT = ROOT / ".runlogs" / "stage3_retrieval" / "runs"
DEFAULT_OUTPUT_ROOT = ROOT / ".runlogs" / "stage3_retrieval" / "tuning"
TUNING_VERSION = "stage3-rrf-limited-tuning-v0.1"
K_CANDIDATES = (10, 20, 40, 60)
DENSE_WEIGHT_CANDIDATES = (1.0, 1.25, 1.5)
BATCH_SIZE_CANDIDATES = (3, 5, 10)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-run", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)

    comparison_run = (
        args.comparison_run.resolve()
        if args.comparison_run
        else _latest_completed_comparison(DEFAULT_COMPARE_ROOT)
    )
    winner = json.loads((comparison_run / "winner.json").read_text(encoding="utf-8"))
    if winner.get("policy") != "weighted_rrf":
        raise RuntimeError("RRF tuning is forbidden because weighted RRF did not win")
    results_path = comparison_run / "retrieval_results.jsonl"
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows:
        raise RuntimeError("Comparison run has no result rows")

    run_id = args.run_id or (
        "rrf-tuning-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("Invalid run-id")
    run_dir = args.output_root.resolve() / run_id
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True)

    tuning_config = {
        "tuning_version": TUNING_VERSION,
        "comparison_run": str(comparison_run),
        "comparison_results_sha256": _sha256(results_path),
        "questions": len(rows),
        "winner_confirmed": "weighted_rrf",
        "fixed_policy_tuned": False,
        "bm25_weight": 1.0,
        "k_candidates": list(K_CANDIDATES),
        "dense_weight_candidates": list(DENSE_WEIGHT_CANDIDATES),
        "fusion_selection_rule": [
            "higher mean_gold_page_recall@5",
            "higher complete_gold_page_hit_rate@5",
            "higher hit_rate@5",
            "higher mrr",
            "lower mean previews until first Gold",
        ],
        "batch_size_candidates": list(BATCH_SIZE_CANDIDATES),
        "batch_selection_rule": [
            "lower mean shown previews until first Gold",
            "lower mean shown previews until all Gold",
            "lower mean batches until first Gold",
            "smaller batch size",
        ],
    }
    _write_json(run_dir / "tuning_config.json", tuning_config)
    _write_json(
        run_dir / "run_state.json",
        {
            "tuning_version": TUNING_VERSION,
            "run_id": run_id,
            "state": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "config_fingerprint": stable_digest(tuning_config, length=32),
        },
    )

    started = time.perf_counter()
    _validate_default_replay(rows)
    fusion_variants: dict[str, dict[str, Any]] = {}
    per_variant_metrics: dict[str, list[dict[str, Any]]] = {}
    for k in K_CANDIDATES:
        for dense_weight in DENSE_WEIGHT_CANDIDATES:
            name = _variant_name(k, dense_weight)
            metrics = [
                _metrics_for_row(
                    row,
                    k=k,
                    dense_weight=dense_weight,
                    batch_size=5,
                )
                for row in rows
            ]
            per_variant_metrics[name] = metrics
            fusion_variants[name] = _aggregate(metrics)
    selected_fusion = max(
        fusion_variants,
        key=lambda name: (_fusion_key(fusion_variants[name]), name),
    )
    selected_k, selected_dense_weight = _parse_variant(selected_fusion)

    batch_variants: dict[str, dict[str, Any]] = {}
    frozen_metrics_by_question: dict[int, dict[str, Any]] = {}
    for batch_size in BATCH_SIZE_CANDIDATES:
        metrics = [
            _metrics_for_row(
                row,
                k=selected_k,
                dense_weight=selected_dense_weight,
                batch_size=batch_size,
            )
            for row in rows
        ]
        batch_variants[str(batch_size)] = _aggregate(metrics, batch_size=batch_size)
        if batch_size == 5:
            frozen_metrics_by_question = {
                int(row["question_index"]): metric
                for row, metric in zip(rows, metrics, strict=True)
            }
    selected_batch = min(
        BATCH_SIZE_CANDIDATES,
        key=lambda batch_size: (
            _batch_key(batch_variants[str(batch_size)]),
            batch_size,
        ),
    )
    if selected_batch != 5:
        frozen_metrics_by_question = {
            int(row["question_index"]): _metrics_for_row(
                row,
                k=selected_k,
                dense_weight=selected_dense_weight,
                batch_size=selected_batch,
            )
            for row in rows
        }

    frozen_rows: list[dict[str, Any]] = []
    for row in rows:
        ranking = _weighted_rrf_ranking(
            row["source_rankings"]["bm25"],
            row["source_rankings"]["dense"],
            exact_ids={item[0] for item in row["exact_entries"]},
            k=selected_k,
            bm25_weight=1.0,
            dense_weight=selected_dense_weight,
        )
        frozen_rows.append(
            {
                "question_id": row["question_id"],
                "question_index": row["question_index"],
                "document_id": row["document_id"],
                "gold_pages": row["gold_pages"],
                "exact_entries": row["exact_entries"],
                "top_50_candidates": ranking[:50],
                "metrics": frozen_metrics_by_question[int(row["question_index"])],
            }
        )

    frozen_config = {
        "merge_policy": "weighted_rrf",
        "rrf_k": selected_k,
        "bm25_weight": 1.0,
        "dense_weight": selected_dense_weight,
        "candidate_batch_size": selected_batch,
        "exact_anchor": "shared upstream unchanged",
        "visual_descriptions": "SearchUnit search-only metadata",
        "fallback_enabled": False,
        "source_comparison_run": str(comparison_run),
        "selection_metrics": batch_variants[str(selected_batch)],
    }
    summary = {
        "tuning_version": TUNING_VERSION,
        "questions": len(rows),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "default_replay_validated": True,
        "fusion_variants": fusion_variants,
        "selected_fusion_variant": selected_fusion,
        "batch_variants": batch_variants,
        "selected_batch_size": selected_batch,
        "frozen_config": frozen_config,
    }
    _write_json(run_dir / "tuning_summary.json", summary)
    _write_json(run_dir / "frozen_config.json", frozen_config)
    _write_jsonl(run_dir / "frozen_question_metrics.jsonl", frozen_rows)
    (run_dir / "tuning_summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    _write_json(
        run_dir / "run_state.json",
        {
            "tuning_version": TUNING_VERSION,
            "run_id": run_id,
            "state": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "config_fingerprint": stable_digest(tuning_config, length=32),
            "selected_fusion_variant": selected_fusion,
            "selected_batch_size": selected_batch,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(run_dir)
    return 0


def _metrics_for_row(
    row: dict[str, Any], *, k: int, dense_weight: float, batch_size: int
) -> dict[str, Any]:
    ranking = _weighted_rrf_ranking(
        row["source_rankings"]["bm25"],
        row["source_rankings"]["dense"],
        exact_ids={item[0] for item in row["exact_entries"]},
        k=k,
        bm25_weight=1.0,
        dense_weight=dense_weight,
    )
    metrics = _ranking_metrics(
        candidate_entries=ranking,
        exact_entries=row["exact_entries"],
        gold_pages=row["gold_pages"],
        batch_size=batch_size,
    )
    metrics["candidate_count"] = len(ranking)
    return metrics


def _weighted_rrf_ranking(
    bm25: list[list[Any]],
    dense: list[list[Any]],
    *,
    exact_ids: set[str],
    k: int,
    bm25_weight: float,
    dense_weight: float,
) -> list[list[Any]]:
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    pages: dict[str, int] = {}
    for entries, weight in ((bm25, bm25_weight), (dense, dense_weight)):
        for rank, entry in enumerate(entries, start=1):
            element_id = str(entry[0])
            page_number = int(entry[1])
            scores[element_id] = scores.get(element_id, 0.0) + weight / (k + rank)
            best_rank[element_id] = min(best_rank.get(element_id, rank), rank)
            pages[element_id] = page_number
    ordered = sorted(
        scores,
        key=lambda element_id: (
            -scores[element_id],
            best_rank[element_id],
            pages[element_id],
            element_id,
        ),
    )
    return [
        [element_id, pages[element_id]]
        for element_id in ordered
        if element_id not in exact_ids
    ]


def _aggregate(
    metrics: list[dict[str, Any]], batch_size: int | None = None
) -> dict[str, Any]:
    first_ranks = [item["first_gold_candidate_rank"] for item in metrics]
    all_ranks = [item["all_gold_candidate_rank"] for item in metrics]
    result = {
        "hit_rate": {
            str(k): sum(bool(item["hit_rate"][str(k)]) for item in metrics)
            / len(metrics)
            for k in K_VALUES
        },
        "mean_gold_page_recall": {
            str(k): sum(float(item["gold_page_recall"][str(k)]) for item in metrics)
            / len(metrics)
            for k in K_VALUES
        },
        "complete_gold_page_hit_rate": {
            str(k): sum(
                bool(item["complete_gold_page_hit"][str(k)]) for item in metrics
            )
            / len(metrics)
            for k in K_VALUES
        },
        "mrr": sum(
            1.0 if rank == 0 else 1.0 / rank if rank is not None else 0.0
            for rank in first_ranks
        )
        / len(metrics),
        "previews_until_first_gold": _distribution(
            [rank for rank in first_ranks if rank is not None]
        ),
        "previews_until_all_gold": _distribution(
            [rank for rank in all_ranks if rank is not None]
        ),
        "batches_until_first_gold": _distribution(
            [
                item["batches_until_first_gold"]
                for item in metrics
                if item["batches_until_first_gold"] is not None
            ]
        ),
        "batches_until_all_gold": _distribution(
            [
                item["batches_until_all_gold"]
                for item in metrics
                if item["batches_until_all_gold"] is not None
            ]
        ),
    }
    if batch_size is not None:
        shown_first = [
            _shown_previews(rank, batch_size)
            for rank in first_ranks
            if rank is not None
        ]
        shown_all = [
            _shown_previews(rank, batch_size)
            for rank in all_ranks
            if rank is not None
        ]
        result.update(
            {
                "batch_size": batch_size,
                "shown_previews_until_first_gold": _distribution(shown_first),
                "shown_previews_until_all_gold": _distribution(shown_all),
            }
        )
    return result


def _fusion_key(item: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(item["mean_gold_page_recall"]["5"]),
        float(item["complete_gold_page_hit_rate"]["5"]),
        float(item["hit_rate"]["5"]),
        float(item["mrr"]),
        -float(item["previews_until_first_gold"]["mean"]),
    )


def _batch_key(item: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(item["shown_previews_until_first_gold"]["mean"]),
        float(item["shown_previews_until_all_gold"]["mean"]),
        float(item["batches_until_first_gold"]["mean"]),
    )


def _shown_previews(rank: int, batch_size: int) -> int:
    return 0 if rank == 0 else ((rank + batch_size - 1) // batch_size) * batch_size


def _validate_default_replay(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        replay = _weighted_rrf_ranking(
            row["source_rankings"]["bm25"],
            row["source_rankings"]["dense"],
            exact_ids={item[0] for item in row["exact_entries"]},
            k=20,
            bm25_weight=1.0,
            dense_weight=1.25,
        )
        if replay != row["policies"]["weighted_rrf"]["candidate_ranking"]:
            raise RuntimeError(
                f"Default RRF replay mismatch for {row['question_id']}"
            )


def _variant_name(k: int, dense_weight: float) -> str:
    return f"k{k}-bm25_1-dense_{str(dense_weight).replace('.', '_')}"


def _parse_variant(name: str) -> tuple[int, float]:
    match = re.fullmatch(r"k(\d+)-bm25_1-dense_(\d+)_(\d+)", name)
    if not match:
        raise ValueError(name)
    return int(match.group(1)), float(f"{match.group(2)}.{match.group(3)}")


def _latest_completed_comparison(root: Path) -> Path:
    candidates = []
    for path in root.glob("default-compare-*"):
        state_path = path / "run_state.json"
        winner_path = path / "winner.json"
        if not state_path.is_file() or not winner_path.is_file():
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("state") == "completed":
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No completed default comparison found")
    return max(candidates, key=lambda path: path.name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _summary_markdown(summary: dict[str, Any]) -> str:
    frozen = summary["frozen_config"]
    lines = [
        "# Stage 3B weighted-RRF limited tuning",
        "",
        f"- Questions: {summary['questions']}",
        f"- Selected fusion: `{summary['selected_fusion_variant']}`",
        f"- Selected batch size: `{summary['selected_batch_size']}`",
        f"- Frozen config: `{frozen}`",
        "",
        "Only weighted RRF was tuned after it won the default comparison.",
        "",
        "| Variant | Recall@5 | Complete@5 | Hit@5 | MRR |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in summary["fusion_variants"].items():
        lines.append(
            f"| {name} | {item['mean_gold_page_recall']['5']:.6f} | "
            f"{item['complete_gold_page_hit_rate']['5']:.6f} | "
            f"{item['hit_rate']['5']:.6f} | {item['mrr']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
