"""Parser-neutral Pydantic models for the soft document structure."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ElementType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    CHART = "chart"
    CODE = "code"
    ALGORITHM = "algorithm"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    LIST = "list"
    EQUATION = "equation"


class ElementParseStatus(str, Enum):
    PARSED = "parsed"
    DEGRADED = "degraded"


class ContentAvailability(str, Enum):
    STRUCTURED = "structured"
    TEXT_ONLY = "text_only"
    VISUAL_ONLY = "visual_only"
    MIXED = "mixed"
    UNAVAILABLE = "unavailable"


class RelationType(str, Enum):
    CONTAINS = "contains"
    NEXT_PAGE = "next_page"
    NEXT_IN_READING_ORDER = "next_in_reading_order"
    BELONGS_TO_SECTION = "belongs_to_section"
    CAPTION_OF = "caption_of"
    FOOTNOTE_OF = "footnote_of"
    REFERS_TO = "refers_to"
    CONTINUED_ON = "continued_on"


class RelationStatus(str, Enum):
    CONFIRMED = "confirmed"
    CANDIDATE = "candidate"
    REJECTED = "rejected"


class RelationSource(str, Enum):
    PARSER = "parser"
    DETERMINISTIC_RULE = "deterministic_rule"
    EXPLICIT_REFERENCE = "explicit_reference"
    LAYOUT_HEURISTIC = "layout_heuristic"
    LLM = "llm"
    HUMAN = "human"


class SoftDocModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class BoundingBox(SoftDocModel):
    bbox_id: str
    raw: tuple[float, float, float, float]
    normalized: tuple[float, float, float, float]
    coordinate_system: str = "page"

    @field_validator("raw", "normalized")
    @classmethod
    def validate_geometry(cls, value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = value
        if not all(float("-inf") < coordinate < float("inf") for coordinate in value):
            raise ValueError("Bounding-box coordinates must be finite")
        if x1 >= x2:
            raise ValueError("Bounding box requires x1 < x2")
        if y1 >= y2:
            raise ValueError("Bounding box requires y1 < y2")
        return tuple(float(coordinate) for coordinate in value)

    @field_validator("normalized")
    @classmethod
    def validate_normalized_range(
        cls, value: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        if any(coordinate < 0.0 or coordinate > 1.0 for coordinate in value):
            raise ValueError("Normalized bounding-box coordinates must be in [0, 1]")
        return value

    @classmethod
    def from_raw(
        cls,
        *,
        bbox_id: str,
        raw: list[float] | tuple[float, float, float, float],
        page_width: float,
        page_height: float,
        coordinate_system: str = "page",
    ) -> Self:
        if page_width <= 0 or page_height <= 0:
            raise ValueError("Page width and height must be positive")
        values = tuple(float(value) for value in raw)
        if len(values) != 4:
            raise ValueError("Bounding box must contain four coordinates")
        x1, y1, x2, y2 = values
        if coordinate_system == "normalized_1000":
            normalized = (x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0)
        else:
            normalized = (x1 / page_width, y1 / page_height, x2 / page_width, y2 / page_height)
        return cls(
            bbox_id=bbox_id,
            raw=values,
            normalized=normalized,
            coordinate_system=coordinate_system,
        )

    @property
    def width(self) -> float:
        return self.normalized[2] - self.normalized[0]

    @property
    def height(self) -> float:
        return self.normalized[3] - self.normalized[1]


class Provenance(SoftDocModel):
    provenance_id: str
    adapter: str
    source_path: Path
    source_locator: str
    parser_version: str | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Element(SoftDocModel):
    element_id: str
    document_id: str
    page_id: str
    page_number: int = Field(ge=1)
    element_type: ElementType
    reading_order: int = Field(ge=0)
    bbox: BoundingBox | None = None
    column_index: int | None = Field(default=None, ge=0)
    section_id: str | None = None
    section_path: list[str] | None = None
    heading_level: int | None = Field(default=None, ge=1)
    text: str | None = None
    html: str | None = None
    image_path: Path | None = None
    crop_image_path: Path | None = None
    parse_status: ElementParseStatus = ElementParseStatus.PARSED
    content_availability: ContentAvailability | None = None
    reference_label: str | None = None
    summary: str | None = None
    keywords: list[str] = Field(default_factory=list)
    provenance: Provenance
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        if self.content_availability is None:
            has_text = bool((self.text or "").strip())
            has_structured = bool((self.html or "").strip())
            has_visual = self.image_path is not None or self.crop_image_path is not None
            if has_visual and (has_text or has_structured):
                availability = ContentAvailability.MIXED
            elif has_structured:
                availability = ContentAvailability.STRUCTURED
            elif has_text:
                availability = ContentAvailability.TEXT_ONLY
            elif has_visual:
                availability = ContentAvailability.VISUAL_ONLY
            else:
                availability = ContentAvailability.UNAVAILABLE
            object.__setattr__(self, "content_availability", availability)
        if self.element_type == ElementType.HEADING and not (self.text or "").strip():
            raise ValueError("Heading elements require text")
        if (
            self.element_type == ElementType.TABLE
            and not any((self.text, self.html, self.image_path, self.crop_image_path))
            and self.parse_status != ElementParseStatus.DEGRADED
        ):
            raise ValueError("Table elements require text, HTML, or an image path")
        if self.element_type in {ElementType.FIGURE, ElementType.CHART} and not any(
            (self.image_path, self.crop_image_path, self.text)
        ):
            raise ValueError("Visual elements require an image reference, crop, or text")
        return self


class Page(SoftDocModel):
    page_id: str
    document_id: str
    page_index: int = Field(ge=0)
    page_number: int = Field(ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    element_ids: list[str] = Field(default_factory=list)
    reading_order: list[str] = Field(default_factory=list)
    image_path: Path | None = None
    provenance: Provenance
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_page_lists(self) -> Self:
        if len(self.element_ids) != len(set(self.element_ids)):
            raise ValueError("Page element_ids must be unique")
        if len(self.reading_order) != len(set(self.reading_order)):
            raise ValueError("Page reading_order IDs must be unique")
        if set(self.reading_order) != set(self.element_ids):
            raise ValueError("Page reading_order must contain exactly the page element IDs")
        return self


class Section(SoftDocModel):
    section_id: str
    document_id: str
    title: str
    level: int = Field(ge=1)
    heading_element_id: str
    parent_section_id: str | None = None
    section_path: list[str]
    page_ids: list[str] = Field(default_factory=list)
    element_ids: list[str] = Field(default_factory=list)
    provenance: Provenance
    metadata: dict[str, Any] = Field(default_factory=dict)


class RelationEvidence(SoftDocModel):
    rule: str
    description: str
    source_ids: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)


class Relation(SoftDocModel):
    relation_id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    confidence: float = Field(ge=0.0, le=1.0)
    status: RelationStatus
    created_by: RelationSource
    evidence: list[RelationEvidence] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def disallow_self_relation(self) -> Self:
        if self.source_id == self.target_id:
            raise ValueError("Relation source and target must differ")
        return self


class Document(SoftDocModel):
    document_id: str
    title: str | None = None
    source_path: Path
    pages: list[Page] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    elements: list[Element] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    provenance: Provenance
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("pages")
    @classmethod
    def pages_are_ordered(cls, pages: list[Page]) -> list[Page]:
        indexes = [page.page_index for page in pages]
        if indexes != sorted(indexes):
            raise ValueError("Document pages must be ordered by page_index")
        return pages
