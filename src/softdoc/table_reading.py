"""Frozen contract for question-directed multimodal table reading."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
import re
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from softdoc.prompts import load_prompt_text


InputId = Annotated[str, Field(min_length=1, pattern=r"^I[1-9][0-9]*$")]

MULTIMODAL_TABLE_READER_PROMPT_VERSION = "multimodal-table-reader-v0.3"


class TableHeaderStatus(StrEnum):
    LOCAL = "local"
    INFERRED_LOCAL = "inferred_local"
    CONFIRMED_INHERITED = "confirmed_inherited"
    INFERRED_INHERITED = "inferred_inherited"
    UNRESOLVED = "unresolved"
    NOT_AVAILABLE_IN_STRUCTURE = "not_available_in_structure"


class TableLimitationCode(StrEnum):
    MISSING_HEADER_CONTEXT = "missing_header_context"
    REPRESENTATION_CONFLICT = "representation_conflict"
    UNREADABLE = "unreadable"
    OTHER = "other"


class TableReadCell(BaseModel):
    """One structured cell supplied alongside an optional table rendering."""

    model_config = ConfigDict(extra="forbid")

    cell_id: str = Field(min_length=1)
    row: int = Field(ge=0)
    column: int = Field(ge=0)
    rowspan: int = Field(default=1, ge=1)
    colspan: int = Field(default=1, ge=1)
    text: str | None = None


class TableReadInput(BaseModel):
    """One selected table, with either structured cells, pixels, or both."""

    model_config = ConfigDict(extra="forbid")

    input_id: InputId
    element_id: str = Field(min_length=1)
    page_id: str = Field(min_length=1)
    physical_page_number: int = Field(ge=1)
    document_page_count: int = Field(ge=1)
    is_last_page: bool
    display_page_label: str | None = None
    row_count: int | None = Field(default=None, ge=0)
    column_count: int | None = Field(default=None, ge=0)
    structured_cells: list[TableReadCell] = Field(default_factory=list)
    extracted_text: str | None = Field(default=None, min_length=1)
    visual_asset_id: str | None = Field(default=None, min_length=1)
    visual_asset_path: Path | None = None
    headers: list[str] = Field(default_factory=list)
    header_status: TableHeaderStatus = TableHeaderStatus.UNRESOLVED
    header_source_element_id: str | None = Field(default=None, min_length=1)
    table_group_id: str | None = Field(default=None, min_length=1)
    fragment_index: int | None = Field(default=None, ge=0)
    fragment_count: int | None = Field(default=None, ge=1)
    relevant_row_hints: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_representations(self) -> "TableReadInput":
        if self.physical_page_number > self.document_page_count:
            raise ValueError(
                "physical_page_number cannot exceed document_page_count"
            )
        if self.is_last_page != (
            self.physical_page_number == self.document_page_count
        ):
            raise ValueError(
                "is_last_page must agree with physical_page_number and "
                "document_page_count"
            )
        if (self.visual_asset_id is None) != (self.visual_asset_path is None):
            raise ValueError(
                "visual_asset_id and visual_asset_path must be provided together"
            )
        if (
            not self.structured_cells
            and self.extracted_text is None
            and self.visual_asset_path is None
        ):
            raise ValueError(
                "A table input requires structured cells, extracted text, "
                "or a visual asset"
            )
        if self.structured_cells and (
            self.row_count is None or self.column_count is None
        ):
            raise ValueError(
                "Structured table cells require row_count and column_count"
            )
        if self.header_status in {
            TableHeaderStatus.LOCAL,
            TableHeaderStatus.INFERRED_LOCAL,
            TableHeaderStatus.CONFIRMED_INHERITED,
            TableHeaderStatus.INFERRED_INHERITED,
        } and not self.headers:
            raise ValueError("Resolved header status requires non-empty headers")
        if (
            self.header_status
            in {
                TableHeaderStatus.CONFIRMED_INHERITED,
                TableHeaderStatus.INFERRED_INHERITED,
            }
            and self.header_source_element_id is None
        ):
            raise ValueError(
                "Confirmed inherited headers require header_source_element_id"
            )
        if (self.fragment_index is None) != (self.fragment_count is None):
            raise ValueError(
                "fragment_index and fragment_count must be provided together"
            )
        if (
            self.fragment_index is not None
            and self.fragment_count is not None
            and self.fragment_index >= self.fragment_count
        ):
            raise ValueError("fragment_index must be smaller than fragment_count")
        return self


def select_relevant_row_hints(
    cells: list[TableReadCell], problem: str, *, limit: int = 3
) -> list[str]:
    """Surface lexically relevant rows without assigning missing headers."""

    problem_tokens = {
        token.casefold()
        for token in re.findall(r"[\w.-]+", problem, flags=re.UNICODE)
        if token.strip(".-_")
    }
    rows: dict[int, list[TableReadCell]] = {}
    for cell in cells:
        if cell.text and cell.text.strip():
            rows.setdefault(cell.row, []).append(cell)
    scored: list[tuple[int, int, str]] = []
    for row_index, row_cells in rows.items():
        text = " | ".join(
            cell.text.strip()
            for cell in sorted(row_cells, key=lambda item: item.column)
            if cell.text and cell.text.strip()
        )
        row_tokens = {
            token.casefold()
            for token in re.findall(r"[\w.-]+", text, flags=re.UNICODE)
            if token.strip(".-_")
        }
        overlap = len(problem_tokens.intersection(row_tokens))
        if overlap:
            scored.append((-overlap, row_index, text))
    scored.sort()
    if not scored:
        return []
    best_overlap = scored[0][0]
    return [
        text for score, _, text in scored if score == best_overlap
    ][:limit]


class TableReadRequest(BaseModel):
    """The local problem and selected table inputs for one READ_SOURCE action."""

    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(min_length=1)
    subquestion_id: str | None = None
    document_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    problem: str = Field(min_length=1)
    table_inputs: list[TableReadInput] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_input_ids(self) -> "TableReadRequest":
        input_ids = [item.input_id for item in self.table_inputs]
        if len(input_ids) != len(set(input_ids)):
            raise ValueError("input IDs must be unique within one action")
        return self


class TableObservationSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_id: InputId
    cell_id: str | None = Field(default=None, min_length=1)


class TableObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    sources: list[TableObservationSource] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_inputs(self) -> Self:
        source_refs = [
            (source.input_id, source.cell_id) for source in self.sources
        ]
        if len(source_refs) != len(set(source_refs)):
            raise ValueError(
                "Observation source references must not contain exact duplicates"
            )
        return self


class TableLimitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: TableLimitationCode = TableLimitationCode.OTHER
    description: str = Field(min_length=1)
    input_ids: list[InputId] = Field(default_factory=list)
    relevant_visible_content: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_missing_header_context(self) -> Self:
        if (
            self.code == TableLimitationCode.MISSING_HEADER_CONTEXT
            and not self.relevant_visible_content
        ):
            raise ValueError(
                "missing_header_context requires relevant_visible_content"
            )
        return self


class TableReadResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observations: list[TableObservation] = Field(max_length=16)
    limitations: list[TableLimitation]

    @model_validator(mode="after")
    def validate_nonempty(self) -> Self:
        if not self.observations and not self.limitations:
            raise ValueError("Table Reader result requires an observation or limitation")
        return self


def validate_table_read_result(
    request: TableReadRequest,
    result: TableReadResult,
) -> TableReadResult:
    """Reject references to table inputs or cells not supplied by this action."""

    inputs_by_id = {item.input_id: item for item in request.table_inputs}
    referenced_ids = {
        source.input_id
        for observation in result.observations
        for source in observation.sources
    }
    referenced_ids.update(
        input_id
        for limitation in result.limitations
        for input_id in limitation.input_ids
    )
    unknown_ids = referenced_ids.difference(inputs_by_id)
    if unknown_ids:
        raise ValueError(
            "Table Reader output references inputs not supplied by the action: "
            + ", ".join(sorted(unknown_ids))
        )
    for observation in result.observations:
        for source in observation.sources:
            known_cells = {
                cell.cell_id for cell in inputs_by_id[source.input_id].structured_cells
            }
            if source.cell_id is not None and source.cell_id not in known_cells:
                raise ValueError(
                    "Table Reader output references a cell not supplied by the action: "
                    + source.cell_id
                )
    return result


MULTIMODAL_TABLE_READER_SYSTEM_PROMPT = load_prompt_text(
    "multimodal_table_reader_v0_2_system.txt"
)
MULTIMODAL_TABLE_READER_USER_PROMPT_TEMPLATE = load_prompt_text(
    "multimodal_table_reader_v0_3_user.txt"
)


def table_reader_user_prompt(request: TableReadRequest) -> str:
    """Render the reviewable request without leaking host filesystem paths."""

    request_json = request.model_dump_json(
        indent=2,
        exclude={"table_inputs": {"__all__": {"visual_asset_path"}}},
    )
    output_shape = json.dumps(
        {
            "observations": [
                {
                    "text": "<one concrete table fact relevant to the local problem>",
                    "sources": [
                        {
                            "input_id": request.table_inputs[0].input_id,
                            "cell_id": None,
                        }
                    ],
                }
            ],
            "limitations": [],
        },
        indent=2,
    )
    rendered = MULTIMODAL_TABLE_READER_USER_PROMPT_TEMPLATE.replace(
        "<<TABLE_READ_REQUEST_JSON>>", request_json
    ).replace("<<OUTPUT_SHAPE_JSON>>", output_shape)
    if "<<" in rendered or ">>" in rendered:
        raise ValueError("Unresolved Multimodal Table Reader prompt placeholder")
    return rendered


__all__ = [
    "MULTIMODAL_TABLE_READER_PROMPT_VERSION",
    "MULTIMODAL_TABLE_READER_SYSTEM_PROMPT",
    "MULTIMODAL_TABLE_READER_USER_PROMPT_TEMPLATE",
    "TableLimitation",
    "TableLimitationCode",
    "TableHeaderStatus",
    "TableObservation",
    "TableObservationSource",
    "TableReadCell",
    "TableReadInput",
    "TableReadRequest",
    "TableReadResult",
    "select_relevant_row_hints",
    "table_reader_user_prompt",
    "validate_table_read_result",
]
