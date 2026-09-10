from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPLIT_ROOT = ROOT / "configs" / "training" / "mmlongbench_sft_split_v0_1"


def test_frozen_sft_split_has_no_document_leakage_or_omission() -> None:
    manifest = json.loads((SPLIT_ROOT / "split_manifest.json").read_text(encoding="utf-8"))
    documents = {
        name: set(payload["documents"])
        for name, payload in manifest["splits"].items()
    }
    assert sum(len(items) for items in documents.values()) == 135
    assert len(set().union(*documents.values())) == 135
    names = list(documents)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            assert documents[left].isdisjoint(documents[right])


def test_runtime_case_manifests_do_not_expose_gold() -> None:
    for path in SPLIT_ROOT.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            assert "answer" not in row
            assert "evidence_pages" not in row
            assert "evidence_sources" not in row


def test_frozen_split_counts_match_design() -> None:
    manifest = json.loads((SPLIT_ROOT / "split_manifest.json").read_text(encoding="utf-8"))
    expected = {
        "train": (42, 357, 140),
        "diagnostic_dev": (12, 97, 37),
        "clean_dev": (8, 68, 0),
        "test_core": (10, 81, 0),
        "test_challenge": (6, 43, 0),
        "reserve": (57, 445, 0),
    }
    observed = {
        name: (
            payload["summary"]["document_count"],
            payload["summary"]["question_count"],
            payload["summary"]["diagnostic_hard_case_count"],
        )
        for name, payload in manifest["splits"].items()
    }
    assert observed == expected
