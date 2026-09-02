"""Compare three-route RRF with fixed 3-text + 2-visual candidate batches.

The runner reuses a frozen Stage-3 text retrieval result so both policies see
identical Exact, BM25, and text-Dense rankings.  It only computes the ColSmol
query-to-image ranking and then applies the two deterministic fusion policies.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np

from softdoc.retrieval.visual_dense import (
    fixed_text_visual_quota_ranking,
    weighted_rrf_three_route_ranking,
)


K_VALUES = (1, 3, 5, 10, 20, 50)
POLICIES = ("weighted_rrf_three_route", "fixed_quota_3_text_2_visual")
EVALUATION_VERSION = "stage3-visual-fusion-comparison-v0.1"


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-results", type=Path, required=True)
    parser.add_argument("--visual-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="vidore/colSmol-500M")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--similarity-chunk-elements", type=int, default=16_000_000)
    args = parser.parse_args(argv)

    state = json.loads((args.visual_index / "state.json").read_text(encoding="utf-8"))
    if state.get("state") != "completed" or state.get("pending_image_count") != 0:
        raise RuntimeError("Visual embedding index is not complete")

    rows = _read_jsonl(args.text_results)
    assets = _read_jsonl(args.visual_index / "assets.jsonl")
    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    required_hashes: set[str] = set()
    for asset in assets:
        by_document[str(asset["document_id"])].append(asset)
        required_hashes.add(str(asset["image_sha256"]))
    vectors = _load_vectors(args.visual_index / "shards", required_hashes)

    from sentence_transformers import MultiVectorEncoder

    started = time.perf_counter()
    model = MultiVectorEncoder(args.model, device=args.device)
    query_embeddings = model.encode_query(
        [str(row["question"]) for row in rows],
        batch_size=args.query_batch_size,
        show_progress_bar=True,
    )
    if hasattr(query_embeddings, "detach"):
        query_embeddings = list(query_embeddings)

    row_indexes_by_document: dict[str, list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        row_indexes_by_document[str(row["softdoc_document_id"])].append(row_index)
    visual_rankings: dict[int, list[list[Any]]] = {}
    for document_index, (document_id, row_indexes) in enumerate(
        row_indexes_by_document.items(), start=1
    ):
        rankings = _visual_rankings_for_document(
            model,
            [query_embeddings[index] for index in row_indexes],
            by_document.get(document_id, []),
            vectors,
            device=args.device,
            chunk_elements=args.similarity_chunk_elements,
        )
        visual_rankings.update(zip(row_indexes, rankings, strict=True))
        if document_index % 10 == 0 or document_index == len(row_indexes_by_document):
            print(
                f"visual_documents={document_index}/{len(row_indexes_by_document)}",
                flush=True,
            )

    output_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        visual_ranking = visual_rankings[row_index]
        exact_entries = [list(item) for item in row.get("exact_entries", [])]
        excluded = {str(item[0]) for item in exact_entries}
        bm25 = row["source_rankings"]["bm25"]
        dense = row["source_rankings"]["dense"]
        text_fused = row["policies"]["weighted_rrf"]["candidate_ranking"]
        rrf = weighted_rrf_three_route_ranking(
            bm25,
            dense,
            visual_ranking,
            rrf_k=20,
            bm25_weight=1.0,
            dense_weight=1.25,
            visual_weight=1.0,
            excluded_ids=excluded,
        )
        quota = fixed_text_visual_quota_ranking(
            text_fused,
            visual_ranking,
            batch_size=5,
            text_quota=3,
            visual_quota=2,
            excluded_ids=excluded,
        )
        gold_pages = [int(value) for value in row["gold_pages"]]
        output_rows.append(
            {
                "question_id": row["question_id"],
                "case_id": row.get("case_id"),
                "document_id": row["document_id"],
                "softdoc_document_id": row["softdoc_document_id"],
                "question": row["question"],
                "gold_pages": gold_pages,
                "exact_entries": exact_entries,
                "visual_candidate_count": len(visual_ranking),
                "visual_ranking": visual_ranking,
                "policies": {
                    "weighted_rrf_three_route": {
                        "candidate_ranking": rrf,
                        "metrics": _ranking_metrics(rrf, exact_entries, gold_pages),
                    },
                    "fixed_quota_3_text_2_visual": {
                        "candidate_ranking": quota,
                        "metrics": _ranking_metrics(quota, exact_entries, gold_pages),
                    },
                },
            }
        )
    aggregates = {policy: _aggregate(output_rows, policy) for policy in POLICIES}
    winner = max(POLICIES, key=lambda policy: _winner_key(aggregates[policy]))
    runner_up = next(policy for policy in POLICIES if policy != winner)
    summary = {
        "evaluation_version": EVALUATION_VERSION,
        "state": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "scope": "same 808 all-Gold-pages-reachable questions as frozen text run",
        "questions": len(output_rows),
        "documents": len({row["document_id"] for row in output_rows}),
        "visual_index": {
            "model": args.model,
            "inventory_fingerprint": state.get("inventory_fingerprint"),
            "unique_image_count": state.get("unique_image_count"),
            "skipped_count": state.get("skipped_count"),
        },
        "fairness": {
            "exact_shared": True,
            "bm25_shared": True,
            "text_dense_shared": True,
            "visual_dense_shared": True,
            "candidate_batch_size": 5,
            "fallback_enabled": False,
        },
        "policies": aggregates,
        "winner": {
            "policy": winner,
            "runner_up": runner_up,
            "selection_key": list(_winner_key(aggregates[winner])),
            "rule": (
                "lexicographic: mean gold-page recall@5, complete gold-page "
                "hit@5, hit@5, MRR"
            ),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "retrieval_results.jsonl", output_rows)
    _write_json(args.output_dir / "retrieval_comparison_summary.json", summary)
    _write_json(args.output_dir / "winner.json", summary["winner"])
    _write_json(
        args.output_dir / "run_state.json",
        {
            "state": "completed",
            "evaluation_version": EVALUATION_VERSION,
            "questions": len(output_rows),
            "winner": winner,
            "completed_at": summary["completed_at"],
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def _visual_rankings_for_document(
    model: Any,
    query_embeddings: Sequence[Any],
    assets: Sequence[dict[str, Any]],
    vectors: dict[str, np.ndarray],
    *,
    device: str,
    chunk_elements: int,
) -> list[list[list[Any]]]:
    if not assets:
        return [[] for _ in query_embeddings]
    # The index is intentionally stored as float16; ColSmol may emit bfloat16
    # queries on Ampere GPUs, so align dtypes before late-interaction scoring.
    aligned_queries = [
        embedding.half() if hasattr(embedding, "half") else embedding
        for embedding in query_embeddings
    ]
    unique_hashes = sorted({str(asset["image_sha256"]) for asset in assets})
    score_matrix = model.similarity(
        aligned_queries,
        [vectors[digest] for digest in unique_hashes],
        device=device,
        chunk_elements=chunk_elements,
    )
    if hasattr(score_matrix, "detach"):
        score_matrix = score_matrix.detach().float().cpu().numpy()
    results: list[list[list[Any]]] = []
    for scores in np.asarray(score_matrix):
        score_by_hash = dict(zip(unique_hashes, scores.tolist(), strict=True))
        ranked = sorted(
            assets,
            key=lambda asset: (
                -float(score_by_hash[str(asset["image_sha256"])]),
                int(asset["page_number"]),
                str(asset["element_id"]),
            ),
        )
        results.append(
            [
                [str(asset["element_id"]), int(asset["page_number"])]
                for asset in ranked
            ]
        )
    return results


def _ranking_metrics(
    candidates: Sequence[Sequence[Any]],
    exact_entries: Sequence[Sequence[Any]],
    gold_pages: Sequence[int],
) -> dict[str, Any]:
    gold = set(gold_pages)
    exact_gold = {int(item[1]) for item in exact_entries} & gold
    candidate_pages = [int(item[1]) for item in candidates]
    ranks: dict[int, int] = {}
    for rank, page in enumerate(candidate_pages, start=1):
        if page in gold and page not in ranks:
            ranks[page] = rank
    first_rank = 0 if exact_gold else min(ranks.values(), default=None)
    remaining = gold - exact_gold
    all_rank = (
        0
        if not remaining
        else max(ranks[page] for page in remaining)
        if remaining.issubset(ranks)
        else None
    )
    return {
        "first_gold_candidate_rank": first_rank,
        "all_gold_candidate_rank": all_rank,
        "hit_rate": {
            str(k): bool(exact_gold | (set(candidate_pages[:k]) & gold))
            for k in K_VALUES
        },
        "gold_page_recall": {
            str(k): len(exact_gold | (set(candidate_pages[:k]) & gold)) / len(gold)
            for k in K_VALUES
        },
        "complete_gold_page_hit": {
            str(k): gold.issubset(exact_gold | set(candidate_pages[:k]))
            for k in K_VALUES
        },
    }


def _aggregate(rows: Sequence[dict[str, Any]], policy: str) -> dict[str, Any]:
    metrics = [row["policies"][policy]["metrics"] for row in rows]
    return {
        "hit_rate": {
            str(k): statistics.fmean(float(item["hit_rate"][str(k)]) for item in metrics)
            for k in K_VALUES
        },
        "mean_gold_page_recall": {
            str(k): statistics.fmean(item["gold_page_recall"][str(k)] for item in metrics)
            for k in K_VALUES
        },
        "complete_gold_page_hit_rate": {
            str(k): statistics.fmean(
                float(item["complete_gold_page_hit"][str(k)]) for item in metrics
            )
            for k in K_VALUES
        },
        "mrr": statistics.fmean(
            1.0 / rank if rank not in (None, 0) else 1.0 if rank == 0 else 0.0
            for rank in (item["first_gold_candidate_rank"] for item in metrics)
        ),
    }


def _winner_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(item["mean_gold_page_recall"]["5"]),
        float(item["complete_gold_page_hit_rate"]["5"]),
        float(item["hit_rate"]["5"]),
        float(item["mrr"]),
    )


def _load_vectors(shards_dir: Path, required: set[str]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for path in sorted(shards_dir.glob("shard-*.npz")):
        with np.load(path) as shard:
            for digest in shard.files:
                if digest in required:
                    result[digest] = shard[digest]
    missing = required - result.keys()
    if missing:
        raise RuntimeError(f"Missing {len(missing)} visual embeddings")
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
