from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_stage3_retrieval import (
    _choose_winner,
    _ranking_metrics,
    _resolved_gold_pages_by_question,
)
from scripts.tune_stage3_rrf import _parse_variant, _weighted_rrf_ranking


def test_ranking_metrics_treat_exact_as_shared_zero_preview_upstream() -> None:
    metrics = _ranking_metrics(
        candidate_entries=[["e1", 1], ["e2", 3], ["e3", 4]],
        exact_entries=[["page:2", 2]],
        gold_pages=[2, 4],
        batch_size=5,
    )

    assert metrics["first_gold_candidate_rank"] == 0
    assert metrics["all_gold_candidate_rank"] == 3
    assert metrics["batches_until_first_gold"] == 0
    assert metrics["batches_until_all_gold"] == 1
    assert metrics["gold_page_recall"]["1"] == 0.5
    assert metrics["gold_page_recall"]["3"] == 1.0
    assert metrics["complete_gold_page_hit"]["3"] is True


def test_predeclared_winner_rule_prefers_recall_before_latency() -> None:
    policies = {
        "weighted_rrf": {
            "mean_gold_page_recall": {"5": 0.81},
            "complete_gold_page_hit_rate": {"5": 0.70},
            "hit_rate": {"5": 0.90},
            "mrr": 0.75,
            "fusion_latency_ms": {"mean": 2.0},
        },
        "fixed_quota_3_2": {
            "mean_gold_page_recall": {"5": 0.80},
            "complete_gold_page_hit_rate": {"5": 0.75},
            "hit_rate": {"5": 0.92},
            "mrr": 0.80,
            "fusion_latency_ms": {"mean": 1.0},
        },
    }

    assert _choose_winner(policies)["policy"] == "weighted_rrf"


def test_offline_rrf_replay_uses_core_ties_and_exact_exclusion() -> None:
    ranking = _weighted_rrf_ranking(
        [["a", 1], ["b", 2], ["c", 3]],
        [["b", 2], ["d", 4], ["a", 1]],
        exact_ids={"b"},
        k=20,
        bm25_weight=1.0,
        dense_weight=1.25,
    )

    assert ranking == [["a", 1], ["d", 4], ["c", 3]]
    assert _parse_variant("k40-bm25_1-dense_1_25") == (40, 1.25)


def test_stage3b_reads_resolved_physical_gold_pages(tmp_path: Path) -> None:
    path = tmp_path / "reachability.jsonl"
    rows = [
        {
            "question_id": "q1",
            "gold_page_id": 8,
            "all_gold_pages_reachable": True,
        },
        {
            "question_id": "q1",
            "gold_page_id": 9,
            "all_gold_pages_reachable": True,
        },
        {
            "question_id": "q2",
            "gold_page_id": 115,
            "all_gold_pages_reachable": False,
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    assert _resolved_gold_pages_by_question(path) == {"q1": [8, 9]}
