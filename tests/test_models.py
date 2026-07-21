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

