"""Portable JSON/JSONL artifact serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from softdoc.models import Document, RelationType
from softdoc.outline import build_document_outline, outline_markdown
from softdoc.store import DocumentStore
from softdoc.visualization import (
    render_cross_page_relation_overlays,
    render_page_overlays,
)


def write_document(document: Document, output_dir: Path, *, render_overlays: bool = True) -> None:
    output_dir = Path(output_dir)
    DocumentStore(document).validate_references(raise_on_error=True)
    for directory in (
        output_dir,
        output_dir / "assets" / "pages",
        output_dir / "assets" / "elements",
        output_dir / "debug" / "page_overlays",
        output_dir / "debug" / "cross_page_overlays",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    _write_json(output_dir / "document.json", document.model_dump(mode="json"))
    _write_jsonl(output_dir / "pages.jsonl", [page.model_dump(mode="json") for page in document.pages])
    _write_jsonl(output_dir / "sections.jsonl", [section.model_dump(mode="json") for section in document.sections])
    _write_jsonl(output_dir / "elements.jsonl", [element.model_dump(mode="json") for element in document.elements])
    _write_jsonl(output_dir / "relations.jsonl", [relation.model_dump(mode="json") for relation in document.relations])
    _write_json(output_dir / "debug" / "cross_page_relations.json", _cross_page_relations(document))
    _write_json(
        output_dir / "debug" / "adapter_warnings.json",
        document.metadata.get("adapter_warnings", []),
    )
    outline = build_document_outline(document)
    _write_json(
        output_dir / "debug" / "document_outline.json",
        outline.model_dump(mode="json"),
    )
    _write_text(
        output_dir / "debug" / "document_outline.md",
        outline_markdown(outline),
    )
    _write_json(
        output_dir / "debug" / "heading_decisions.json",
        document.metadata.get("heading_decisions", []),
    )
    _write_json(
        output_dir / "debug" / "heading_eligibility_decisions.json",
        document.metadata.get("heading_eligibility_decisions", []),
    )
    _write_json(
        output_dir / "debug" / "element_normalization_decisions.json",
        document.metadata.get("element_normalization_decisions", []),
    )
    _write_json(
        output_dir / "debug" / "document_profile.json",
        document.metadata.get("document_profile", {}),
    )
    _write_json(
        output_dir / "debug" / "page_label_decisions.json",
        document.metadata.get("page_label_decisions", []),
    )
    _write_json(
        output_dir / "debug" / "section_resolution_decisions.json",
        document.metadata.get("section_resolution_decisions", []),
    )
    if render_overlays:
        render_page_overlays(document, output_dir)
        render_cross_page_relation_overlays(document, output_dir)


def load_document(output_dir: Path) -> Document:
    document_path = Path(output_dir) / "document.json"
    with document_path.open("r", encoding="utf-8") as handle:
        return Document.model_validate(json.load(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def _cross_page_relations(document: Document) -> list[dict[str, Any]]:
    element_pages = {element.element_id: element.page_id for element in document.elements}
    cross_page_types = {
        RelationType.CAPTION_OF,
        RelationType.FOOTNOTE_OF,
        RelationType.REFERS_TO,
        RelationType.CONTINUED_ON,
    }
    result: list[dict[str, Any]] = []
    for relation in document.relations:
        if relation.relation_type not in cross_page_types:
            continue
        source_page = element_pages.get(relation.source_id)
        target_page = element_pages.get(relation.target_id)
        if not source_page or not target_page or source_page == target_page:
            continue
        result.append(
            {
                "relation_id": relation.relation_id,
                "source_page_id": source_page,
                "source_element_id": relation.source_id,
                "target_page_id": target_page,
                "target_element_id": relation.target_id,
                "relation_type": relation.relation_type.value,
                "confidence": relation.confidence,
                "status": relation.status.value,
                "created_by": relation.created_by.value,
                "evidence": [item.model_dump(mode="json") for item in relation.evidence],
            }
        )
    return result
