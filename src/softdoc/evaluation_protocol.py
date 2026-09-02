"""Frozen evaluation protocols and immutable experiment snapshots.

An evaluation protocol says *what* is measured and under which default runtime
conditions.  An experiment snapshot binds that protocol to the exact code,
prompts, models, corpus audit, retrieval settings, and cost table used by one
run.  Results are never written into either object.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


EVALUATION_PROTOCOL_VERSION = "crave-evaluation-protocol-v0.1"
EXPERIMENT_SNAPSHOT_VERSION = "crave-experiment-snapshot-v0.1"
_SLUG = re.compile(r"^[a-z0-9]+(?:[a-z0-9-]*[a-z0-9])?$")


class EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LockedArtifact(EvaluationModel):
    artifact_id: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    revision: str | None = None
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BenchmarkSpec(EvaluationModel):
    dataset_id: str = Field(min_length=1)
    dataset_role: Literal["development_reference"]
    split_policy: Literal["full_corpus_no_split"]
    clean_holdout: Literal[False]
    document_count: int = Field(gt=0)
    question_count: int = Field(gt=0)
    annotation_artifact: LockedArtifact
    source_revision: str = Field(min_length=1)
    coverage_report_required: Literal[True]
    reporting_disclaimer: str = Field(min_length=1)


class AnswerScoringSpec(EvaluationModel):
    protocol: Literal["mmlongbench_doc_v1_official"]
    primary_metric_id: str = Field(min_length=1)
    reported_metric_ids: list[str] = Field(min_length=1)
    evaluator_artifacts: list[LockedArtifact] = Field(min_length=1)
    extractor_model: str = Field(min_length=1)
    extractor_temperature: float = Field(ge=0)
    normalization_authority: Literal["pinned_upstream_eval_score"]
    comparability_note: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_primary_is_reported(self) -> "AnswerScoringSpec":
        if self.primary_metric_id not in self.reported_metric_ids:
            raise ValueError("primary answer metric must also be reported")
        return self


class MetricSpec(EvaluationModel):
    metric_id: str = Field(min_length=1)
    direction: Literal["higher", "lower", "descriptive"]
    definition: str = Field(min_length=1)
    aggregation: str = Field(min_length=1)


class RuntimeDefaults(EvaluationModel):
    random_seed: int
    temperature: float = Field(ge=0)
    context_length: int = Field(gt=0)
    action_budget: int = Field(gt=0)
    candidate_batch_size: int = Field(gt=0)
    request_timeout_seconds: float = Field(gt=0)
    case_timeout_seconds: float = Field(gt=0)
    deterministic_algorithms: bool


class EvaluationProtocol(EvaluationModel):
    schema_version: Literal[EVALUATION_PROTOCOL_VERSION] = EVALUATION_PROTOCOL_VERSION
    protocol_id: str = Field(min_length=1)
    frozen: Literal[True]
    benchmark: BenchmarkSpec
    answer_scoring: AnswerScoringSpec
    evidence_metrics: list[MetricSpec] = Field(min_length=1)
    retrieval_metrics: list[MetricSpec] = Field(min_length=1)
    reading_metrics: list[MetricSpec] = Field(min_length=1)
    efficiency_metrics: list[MetricSpec] = Field(min_length=1)
    runtime_defaults: RuntimeDefaults
    required_experiment_bindings: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_identifiers(self) -> "EvaluationProtocol":
        if not _SLUG.fullmatch(self.protocol_id):
            raise ValueError("protocol_id must be a lowercase kebab-case slug")
        metric_ids = [
            item.metric_id
            for group in (
                self.evidence_metrics,
                self.retrieval_metrics,
                self.reading_metrics,
                self.efficiency_metrics,
            )
            for item in group
        ]
        duplicates = sorted({item for item in metric_ids if metric_ids.count(item) > 1})
        if duplicates:
            raise ValueError(f"duplicate metric_id values: {duplicates}")
        if len(set(self.required_experiment_bindings)) != len(
            self.required_experiment_bindings
        ):
            raise ValueError("required_experiment_bindings contains duplicates")
        return self


class ExperimentSnapshot(EvaluationModel):
    schema_version: Literal[EXPERIMENT_SNAPSHOT_VERSION] = EXPERIMENT_SNAPSHOT_VERSION
    experiment_id: str
    created_at: datetime
    protocol_id: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    bindings: dict[str, Any]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def load_evaluation_protocol(path: Path) -> EvaluationProtocol:
    return EvaluationProtocol.model_validate_json(Path(path).read_text(encoding="utf-8"))


def protocol_sha256(protocol: EvaluationProtocol) -> str:
    return sha256(canonical_json_bytes(protocol.model_dump(mode="json"))).hexdigest()


def build_experiment_snapshot(
    protocol: EvaluationProtocol,
    bindings: dict[str, Any],
    *,
    created_at: datetime | None = None,
) -> ExperimentSnapshot:
    missing = [
        key for key in protocol.required_experiment_bindings if key not in bindings
    ]
    if missing:
        raise ValueError(f"missing required experiment bindings: {missing}")
    timestamp = created_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("created_at must be timezone-aware")
    normalized_bindings = json.loads(canonical_json_bytes(bindings))
    fingerprint = sha256(
        canonical_json_bytes(
            {
                "protocol": protocol.model_dump(mode="json"),
                "bindings": normalized_bindings,
            }
        )
    ).hexdigest()
    stamp = timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ExperimentSnapshot(
        experiment_id=f"{protocol.protocol_id}-{stamp}-{fingerprint[:12]}",
        created_at=timestamp,
        protocol_id=protocol.protocol_id,
        protocol_sha256=protocol_sha256(protocol),
        configuration_fingerprint=fingerprint,
        bindings=normalized_bindings,
    )


def write_experiment_snapshot(
    snapshot: ExperimentSnapshot,
    output_root: Path,
) -> Path:
    """Create one immutable run directory; never overwrite an existing ID."""

    run_dir = Path(output_root) / snapshot.experiment_id
    run_dir.mkdir(parents=True, exist_ok=False)
    output = run_dir / "experiment_snapshot.json"
    output.write_text(
        json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return output
