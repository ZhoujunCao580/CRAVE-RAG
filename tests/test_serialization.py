from __future__ import annotations

import json

from softdoc.models import RelationType
from softdoc.serialization import load_document, write_document
from softdoc.store import DocumentStore


def test_output_files_and_json_round_trip(parsed_document, tmp_path) -> None:
    output = tmp_path / "output"
    write_document(parsed_document, output)
    expected = {
        "document.json",
        "pages.jsonl",
        "sections.jsonl",
        "elements.jsonl",
        "relations.jsonl",
        "debug/cross_page_relations.json",
        "debug/adapter_warnings.json",
        "debug/document_outline.json",
        "debug/document_outline.md",
        "debug/heading_decisions.json",
        "debug/section_resolution_decisions.json",
        "debug/page_overlays/page_0001.png",
        "debug/page_overlays/page_0002.png",
        "debug/page_overlays/page_0003.png",
        "debug/cross_page_overlays/pages_0002_0003.png",
    }
    assert all((output / relative).exists() for relative in expected)
    restored = load_document(output)
    assert restored == parsed_document
    assert DocumentStore(restored).validate_references() == []
    with (output / "debug" / "cross_page_relations.json").open(encoding="utf-8") as handle:
        cross_page = json.load(handle)
    assert any(item["relation_type"] == RelationType.REFERS_TO.value for item in cross_page)
