from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from softdoc.evaluation_protocol import (
    build_experiment_snapshot,
    load_evaluation_protocol,
    write_experiment_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "evaluation" / "mmlongbench_doc_reference_v0_1.json"


def required_bindings(protocol) -> dict[str, object]:
    return {key: {"frozen": key} for key in protocol.required_experiment_bindings}


def test_canonical_protocol_is_full_reference_corpus_without_holdout() -> None:
    protocol = load_evaluation_protocol(PROTOCOL)

    assert protocol.frozen is True
    assert protocol.benchmark.dataset_role == "development_reference"
    assert protocol.benchmark.split_policy == "full_corpus_no_split"
    assert protocol.benchmark.clean_holdout is False
    assert protocol.benchmark.document_count == 135
    assert protocol.benchmark.question_count == 1091
    assert protocol.answer_scoring.primary_metric_id == "official_v1_f1"


def test_snapshot_fingerprint_is_deterministic_and_id_records_time() -> None:
    protocol = load_evaluation_protocol(PROTOCOL)
    bindings = required_bindings(protocol)
    first = build_experiment_snapshot(
        protocol,
        bindings,
        created_at=datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc),
    )
    second = build_experiment_snapshot(
        protocol,
        bindings,
        created_at=datetime(2026, 9, 1, 0, 1, tzinfo=timezone.utc),
    )

    assert first.configuration_fingerprint == second.configuration_fingerprint
    assert first.protocol_sha256 == second.protocol_sha256
    assert first.experiment_id != second.experiment_id


def test_snapshot_requires_every_binding() -> None:
    protocol = load_evaluation_protocol(PROTOCOL)

    with pytest.raises(ValueError, match="missing required experiment bindings"):
        build_experiment_snapshot(protocol, {})


def test_snapshot_writer_refuses_to_overwrite(tmp_path: Path) -> None:
    protocol = load_evaluation_protocol(PROTOCOL)
    snapshot = build_experiment_snapshot(
        protocol,
        required_bindings(protocol),
        created_at=datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc),
    )

    output = write_experiment_snapshot(snapshot, tmp_path)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == snapshot.experiment_id
    with pytest.raises(FileExistsError):
        write_experiment_snapshot(snapshot, tmp_path)
