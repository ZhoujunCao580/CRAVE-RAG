from scripts.score_semantic_answers import _summarize_records


def test_summarize_records_matches_frozen_test_accounting() -> None:
    records = []
    records.extend(
        {
            "gold_answer": "answer",
            "reading_status": "ready",
            "correct": True,
            "scoring_method": "normalized_exact",
        }
        for _ in range(44)
    )
    records.extend(
        {
            "gold_answer": "answer",
            "reading_status": "ready",
            "correct": False,
            "scoring_method": "llm_semantic_judge",
        }
        for _ in range(35)
    )
    records.extend(
        {
            "gold_answer": "answer",
            "reading_status": "budget_exhausted",
            "correct": False,
            "scoring_method": "not_answerable_exact",
        }
        for _ in range(20)
    )
    records.extend(
        {
            "gold_answer": "Not answerable",
            "reading_status": "stopped_incomplete",
            "correct": True,
            "scoring_method": "not_answerable_exact",
        }
        for _ in range(18)
    )
    records.extend(
        {
            "gold_answer": "Not answerable",
            "reading_status": "budget_exhausted",
            "correct": False,
            "scoring_method": "llm_semantic_judge",
        }
        for _ in range(7)
    )

    summary = _summarize_records(records)

    assert summary["case_count"] == 124
    assert summary["correct"] == 62
    assert summary["accuracy"] == 0.5
    assert summary["gold_answerable_count"] == 99
    assert summary["gold_not_answerable_count"] == 25
    assert summary["predicted_answerable_count"] == 79
    assert summary["answerable_correct"] == 44
    assert summary["precision"] == 44 / 79
    assert summary["recall"] == 44 / 99
    assert summary["generalized_f1"] == 2 * (44 / 79) * (44 / 99) / (
        44 / 79 + 44 / 99
    )
