from __future__ import annotations

import json
from pathlib import Path

from softdoc.models import RelationStatus, RelationType
from softdoc.rule_audit import (
    collect_rule_coverage,
    stable_rule_id,
    write_rule_coverage_reports,
)


def test_stable_rule_id_is_namespaced_readable_and_versioned() -> None:
    assert (
        stable_rule_id("relation", "explicit_numbered_reference")
        == "softdoc.relation.explicit_numbered_reference.v1"
    )
    assert (
        stable_rule_id("Heading Eligibility", "Page Header / Footer")
        == "softdoc.heading_eligibility.page_header_footer.v1"
    )


def test_rule_audit_is_read_only_repeatable_and_counts_statuses(
    parsed_document,
) -> None:
    before = parsed_document.model_dump(mode="json")

    first = collect_rule_coverage([parsed_document])
    second = collect_rule_coverage([parsed_document])

    assert parsed_document.model_dump(mode="json") == before
    assert first == second
    assert first.rule_count > 0
    assert first.total_firings == sum(
        entry.firing_count for entry in first.rules
    )
    assert all(
        entry.rule_id.startswith(f"softdoc.{entry.namespace}.")
        and entry.rule_id.endswith(".v1")
        for entry in first.rules
    )
    assert len({entry.rule_id for entry in first.rules}) == first.rule_count

    continuation_relations = [
        relation
        for relation in parsed_document.relations
        if relation.relation_type == RelationType.CONTINUED_ON
    ]
    if continuation_relations:
        entry = next(
            item
            for item in first.rules
            if item.rule_name.startswith("bounded_cross_page_")
        )
        expected_candidates = sum(
            relation.status == RelationStatus.CANDIDATE
            for relation in continuation_relations
            for evidence in relation.evidence
            if evidence.rule == entry.rule_name
        )
        assert entry.candidate_count == expected_candidates


def test_rule_reports_are_stable_and_include_affected_ids(
    parsed_document,
    tmp_path: Path,
) -> None:
    first = write_rule_coverage_reports([parsed_document], tmp_path)
    first_json = (tmp_path / "rule_coverage_report.json").read_bytes()
    first_markdown = (tmp_path / "rule_coverage_report.md").read_bytes()
    second = write_rule_coverage_reports([parsed_document], tmp_path)

    assert first == second
    assert (tmp_path / "rule_coverage_report.json").read_bytes() == first_json
    assert (tmp_path / "rule_coverage_report.md").read_bytes() == first_markdown
    payload = json.loads(first_json)
    assert payload["document_count"] == 1
    assert any(
        row["affected_relations"]
        for row in payload["rules"]
        if row["namespace"] == "relation"
    )
