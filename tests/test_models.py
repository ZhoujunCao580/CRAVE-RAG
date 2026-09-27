from __future__ import annotations

import pytest
from pydantic import ValidationError

from softdoc.models import BoundingBox, Document


def test_bbox_validates_and_normalizes() -> None:
    bbox = BoundingBox.from_raw(
        bbox_id="bbox:test",
        raw=(100, 200, 500, 800),
        page_width=1000,
        page_height=1000,
    )
    assert bbox.normalized == (0.1, 0.2, 0.5, 0.8)


@pytest.mark.parametrize(
    "raw",
    [
        (5, 0, 5, 10),
        (0, 8, 5, 8),
        (-1, 0, 5, 8),
        (0, 0, 1001, 8),
    ],
)
def test_bbox_rejects_invalid_geometry_or_normalized_range(raw: tuple[int, int, int, int]) -> None:
    with pytest.raises((ValidationError, ValueError)):
        BoundingBox.from_raw(
            bbox_id="bbox:bad",
            raw=raw,
            page_width=1000,
            page_height=1000,
        )


def test_document_json_round_trip(parsed_document: Document) -> None:
    restored = Document.model_validate_json(parsed_document.model_dump_json())
    assert restored == parsed_document
    assert restored.relations == parsed_document.relations
    assert all(element.summary is None and element.keywords == [] for element in restored.elements)


def test_document_rejects_duplicate_or_gapped_physical_page_sequence(
    parsed_document: Document,
) -> None:
    payload = parsed_document.model_dump(mode="json")
    payload["pages"][0]["page_index"] = payload["pages"][1]["page_index"]

    with pytest.raises(ValidationError, match="page_index values"):
        Document.model_validate(payload)

    payload = parsed_document.model_dump(mode="json")
    payload["pages"][0]["page_number"] = payload["pages"][1]["page_number"]
    with pytest.raises(ValidationError, match="page_number values"):
        Document.model_validate(payload)


@pytest.mark.parametrize("collection", ["pages", "sections", "elements"])
def test_document_rejects_nested_objects_owned_by_another_document(
    parsed_document: Document,
    collection: str,
) -> None:
    payload = parsed_document.model_dump(mode="json")
    assert payload[collection]
    payload[collection][0]["document_id"] = "doc:wrong-owner"

    with pytest.raises(ValidationError, match="owned by another document"):
        Document.model_validate(payload)
