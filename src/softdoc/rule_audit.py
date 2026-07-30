"""Read-only coverage audit for deterministic SoftDoc rules."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from pydantic import Field

from softdoc.models import Document, RelationStatus, SoftDocModel


_RULE_KEY_NAMES = {
    "anchor_rule",
    "recovery_rule",
    "resolution_rule",
    "target_resolution_rule",
}
_EVIDENCE_SAMPLE_LIMIT = 20


class RuleEvidenceSample(SoftDocModel):
    document_id: str
    element_id: str | None = None
    relation_id: str | None = None
    section_id: str | None = None
    status: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class RuleCoverageEntry(SoftDocModel):
    rule_id: str
    namespace: str
    rule_name: str
    firing_count: int = Field(ge=0)
    affected_documents: list[str] = Field(default_factory=list)
    affected_elements: list[str] = Field(default_factory=list)
    affected_relations: list[str] = Field(default_factory=list)
    affected_sections: list[str] = Field(default_factory=list)
    confirmed_count: int = Field(default=0, ge=0)
    candidate_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)
    evidence_sample_limit: int = _EVIDENCE_SAMPLE_LIMIT
    evidence: list[RuleEvidenceSample] = Field(default_factory=list)


class RuleCoverageReport(SoftDocModel):
    schema_version: str = "1.0"
    catalog_version: str = "softdoc-rule-catalog-v1"
    scope: str = "observed_semantic_state_rule_firings"
    document_count: int = Field(ge=0)
    rule_count: int = Field(ge=0)
    total_firings: int = Field(ge=0)
    rules: list[RuleCoverageEntry] = Field(default_factory=list)


@dataclass
class _Accumulator:
    namespace: str
    rule_name: str
    firing_count: int = 0
    affected_documents: set[str] = field(default_factory=set)
    affected_elements: set[str] = field(default_factory=set)
    affected_relations: set[str] = field(default_factory=set)
    affected_sections: set[str] = field(default_factory=set)
    confirmed_count: int = 0
    candidate_count: int = 0
    rejected_count: int = 0
    evidence: list[RuleEvidenceSample] = field(default_factory=list)


class RuleAuditCollector:
    """Collect rule firings from existing auditable document side effects.

    The collector never writes a ``rule_id`` back into the Document. Stable IDs
    are a reporting concern, so the canonical SoftDoc JSON remains byte-for-byte
    comparable with the frozen milestone.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _Accumulator] = {}
        self._seen_firings: set[
            tuple[
                str,
                str,
                str | None,
                str | None,
                str | None,
                str | None,
            ]
        ] = set()

    def collect(self, documents: Iterable[Document]) -> RuleCoverageReport:
        documents = list(documents)
        for document in documents:
            self._collect_document(document)
        rules = [
            self._entry(rule_id, accumulator)
            for rule_id, accumulator in sorted(self._entries.items())
        ]
        return RuleCoverageReport(
            document_count=len(documents),
            rule_count=len(rules),
            total_firings=sum(item.firing_count for item in rules),
            rules=rules,
        )

    def _collect_document(self, document: Document) -> None:
        metadata = document.metadata
        profile = metadata.get("document_profile")
        if isinstance(profile, dict):
            evidence = profile.get("evidence")
            if isinstance(evidence, dict):
                self._add_from_mapping(
                    document,
                    namespace="profile",
                    mapping=evidence,
                    primary_key="rule",
                )

        for decision in _mapping_list(
            metadata.get("element_normalization_decisions")
        ):
            self._add(
                document,
                namespace="normalization",
                rule_name=decision.get("rule"),
                element_id=_string(decision.get("element_id")),
                evidence=_mapping(decision.get("evidence")),
            )
            self._collect_nested_named_rules(
                document,
                namespace="normalization_detail",
                payload=decision.get("evidence"),
                element_id=_string(decision.get("element_id")),
            )

        for decision in _mapping_list(
            metadata.get("repeated_header_footer_decisions")
        ):
            evidence = _mapping(decision.get("evidence"))
            self._add(
                document,
                namespace="repeated_region",
                rule_name=evidence.get("rule"),
                element_id=_string(decision.get("element_id")),
                evidence=evidence,
            )

        for decision in _mapping_list(
            metadata.get("heading_eligibility_decisions")
        ):
            self._add(
                document,
                namespace="heading_eligibility",
                rule_name=decision.get("reason"),
                element_id=_string(decision.get("element_id")),
                evidence=_mapping(decision.get("evidence")),
            )

        for decision in _mapping_list(metadata.get("heading_decisions")):
            evidence = _mapping(decision.get("evidence"))
            self._add(
                document,
                namespace="heading_hierarchy",
                rule_name=evidence.get("rule"),
                element_id=_string(decision.get("element_id")),
                evidence=evidence,
            )

        for section in document.sections:
            self._collect_nested_named_rules(
                document,
                namespace="section_builder",
                payload=section.metadata,
                section_id=section.section_id,
            )

        for relation in document.relations:
            for evidence in relation.evidence:
                self._add(
                    document,
                    namespace="relation",
                    rule_name=evidence.rule,
                    relation_id=relation.relation_id,
                    status=relation.status,
                    evidence=evidence.model_dump(mode="json"),
                )
                self._collect_nested_named_rules(
                    document,
                    namespace="relation_resolution",
                    payload=evidence.data,
                    relation_id=relation.relation_id,
                    status=relation.status,
                )
            self._collect_nested_named_rules(
                document,
                namespace="relation_resolution",
                payload=relation.metadata,
                relation_id=relation.relation_id,
                status=relation.status,
            )

        for decision in _mapping_list(
            metadata.get("section_resolution_decisions")
        ):
            self._add(
                document,
                namespace="floating_section",
                rule_name=decision.get("rule"),
                element_id=_string(decision.get("element_id")),
                status=_string(decision.get("status")),
                evidence={
                    "evidence_relation_ids": decision.get(
                        "evidence_relation_ids", []
                    ),
                    "metadata": decision.get("metadata", {}),
                    "original_section_id": decision.get(
                        "original_section_id"
                    ),
                    "resolved_section_id": decision.get(
                        "resolved_section_id"
                    ),
                    "candidate_section_id": decision.get(
                        "candidate_section_id"
                    ),
                },
            )

        coverage = metadata.get("coverage_recovery")
        if isinstance(coverage, dict):
            recovered_ids = [
                item
                for item in coverage.get("recovered_element_ids", [])
                if isinstance(item, str)
            ]
            for element_id in recovered_ids:
                self._add(
                    document,
                    namespace="coverage",
                    rule_name="native_pdf_text_layer_uncovered",
                    element_id=element_id,
                    evidence={
                        "source": coverage.get("source"),
                        "scanned_line_count": coverage.get(
                            "scanned_line_count"
                        ),
                    },
                )

    def _add_from_mapping(
        self,
        document: Document,
        *,
        namespace: str,
        mapping: dict[str, Any],
        primary_key: str,
    ) -> None:
        self._add(
            document,
            namespace=namespace,
            rule_name=mapping.get(primary_key),
            evidence=mapping,
        )

    def _collect_nested_named_rules(
        self,
        document: Document,
        *,
        namespace: str,
        payload: Any,
        element_id: str | None = None,
        relation_id: str | None = None,
        section_id: str | None = None,
        status: RelationStatus | str | None = None,
        path: tuple[str, ...] = (),
    ) -> None:
        if isinstance(payload, dict):
            for key, value in payload.items():
                current_path = (*path, str(key))
                if key in _RULE_KEY_NAMES and isinstance(value, str):
                    self._add(
                        document,
                        namespace=namespace,
                        rule_name=value,
                        element_id=element_id,
                        relation_id=relation_id,
                        section_id=section_id,
                        status=status,
                        evidence={"path": list(current_path), key: value},
                    )
                else:
                    self._collect_nested_named_rules(
                        document,
                        namespace=namespace,
                        payload=value,
                        element_id=element_id,
                        relation_id=relation_id,
                        section_id=section_id,
                        status=status,
                        path=current_path,
                    )
        elif isinstance(payload, list):
            for index, item in enumerate(payload):
                self._collect_nested_named_rules(
                    document,
                    namespace=namespace,
                    payload=item,
                    element_id=element_id,
                    relation_id=relation_id,
                    section_id=section_id,
                    status=status,
                    path=(*path, str(index)),
                )

    def _add(
        self,
        document: Document,
        *,
        namespace: str,
        rule_name: Any,
        element_id: str | None = None,
        relation_id: str | None = None,
        section_id: str | None = None,
        status: RelationStatus | str | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(rule_name, str) or not rule_name.strip():
            return
        normalized_name = rule_name.strip()
        identifier = stable_rule_id(namespace, normalized_name)
        status_value = (
            status.value if isinstance(status, RelationStatus) else status
        )
        firing_key = (
            document.document_id,
            identifier,
            element_id,
            relation_id,
            section_id,
            status_value,
        )
        if firing_key in self._seen_firings:
            return
        self._seen_firings.add(firing_key)
        accumulator = self._entries.setdefault(
            identifier,
            _Accumulator(namespace=namespace, rule_name=normalized_name),
        )
        accumulator.firing_count += 1
        accumulator.affected_documents.add(document.document_id)
        if element_id:
            accumulator.affected_elements.add(element_id)
        if relation_id:
            accumulator.affected_relations.add(relation_id)
        if section_id:
            accumulator.affected_sections.add(section_id)
        if status_value == "confirmed":
            accumulator.confirmed_count += 1
        elif status_value == "candidate":
            accumulator.candidate_count += 1
        elif status_value == "rejected":
            accumulator.rejected_count += 1
        if len(accumulator.evidence) < _EVIDENCE_SAMPLE_LIMIT:
            accumulator.evidence.append(
                RuleEvidenceSample(
                    document_id=document.document_id,
                    element_id=element_id,
                    relation_id=relation_id,
                    section_id=section_id,
                    status=status_value,
                    evidence=evidence or {},
                )
            )

    @staticmethod
    def _entry(
        rule_id: str,
        accumulator: _Accumulator,
    ) -> RuleCoverageEntry:
        return RuleCoverageEntry(
            rule_id=rule_id,
            namespace=accumulator.namespace,
            rule_name=accumulator.rule_name,
            firing_count=accumulator.firing_count,
            affected_documents=sorted(accumulator.affected_documents),
            affected_elements=sorted(accumulator.affected_elements),
            affected_relations=sorted(accumulator.affected_relations),
            affected_sections=sorted(accumulator.affected_sections),
            confirmed_count=accumulator.confirmed_count,
            candidate_count=accumulator.candidate_count,
            rejected_count=accumulator.rejected_count,
            evidence=list(accumulator.evidence),
        )


def stable_rule_id(namespace: str, rule_name: str) -> str:
    """Return a readable, versioned, deterministic rule identifier."""

    normalized_namespace = _slug(namespace)
    normalized_rule = _slug(rule_name)
    return f"softdoc.{normalized_namespace}.{normalized_rule}.v1"


def collect_rule_coverage(
    documents: Iterable[Document],
) -> RuleCoverageReport:
    return RuleAuditCollector().collect(documents)


def write_rule_coverage_reports(
    documents: Iterable[Document],
    output_dir: Path,
) -> RuleCoverageReport:
    report = collect_rule_coverage(documents)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "rule_coverage_report.json"
    markdown_path = output_dir / "rule_coverage_report.md"
    json_path.write_text(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        rule_coverage_markdown(report),
        encoding="utf-8",
    )
    return report


def rule_coverage_markdown(report: RuleCoverageReport) -> str:
    lines = [
        "# SoftDoc deterministic rule coverage",
        "",
        f"- Documents: {report.document_count}",
        f"- Rules fired: {report.rule_count}",
        f"- Total firings: {report.total_firings}",
        f"- Scope: `{report.scope}`",
        (
            "- Evidence: complete affected ID sets plus up to "
            f"{_EVIDENCE_SAMPLE_LIMIT} representative evidence records per rule"
        ),
        "",
        "| Rule ID | Firings | Docs | Elements | Relations | Confirmed | Candidate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in report.rules:
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{entry.rule_id}`",
                    str(entry.firing_count),
                    str(len(entry.affected_documents)),
                    str(len(entry.affected_elements)),
                    str(len(entry.affected_relations)),
                    str(entry.confirmed_count),
                    str(entry.candidate_count),
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return result or "unnamed"


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _mapping_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
