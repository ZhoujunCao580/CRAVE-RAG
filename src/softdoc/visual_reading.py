"""Typed boundary for question-directed visual observations.

This module deliberately contains no model client and no evidence logic.  It
only defines what a Visual Reader may receive and return.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from softdoc.prompts import load_prompt_text


NormalizedCoordinate = Annotated[float, Field(ge=0.0, le=1.0)]
InputId = Annotated[str, Field(min_length=1, pattern=r"^I[1-9][0-9]*$")]
NormalizedRegion = tuple[
    NormalizedCoordinate,
    NormalizedCoordinate,
    NormalizedCoordinate,
    NormalizedCoordinate,
]


VISUAL_READER_PROMPT_VERSION = "visual-reader-v0.4"


class VisualInput(BaseModel):
    """One image supplied to a Visual Reader request."""

    model_config = ConfigDict(extra="forbid")

    input_id: InputId
    visual_asset_id: str = Field(min_length=1)
    page_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    display_page_label: str | None = None
    page_image_path: Path
    element_id: str | None = None
    element_type: str | None = None
    bbox: NormalizedRegion | None = None

    @model_validator(mode="after")
    def validate_bbox(self) -> "VisualInput":
        if self.bbox is not None:
            x1, y1, x2, y2 = self.bbox
            if not x1 < x2 or not y1 < y2:
                raise ValueError("bbox must satisfy x1 < x2 and y1 < y2")
        return self


class VisualReadRequest(BaseModel):
    """A local problem plus the visual inputs chosen by the Controller."""

    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(min_length=1)
    subquestion_id: str | None = None
    document_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    problem: str = Field(min_length=1)
    visual_inputs: list[VisualInput] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_input_ids(self) -> "VisualReadRequest":
        input_ids = [item.input_id for item in self.visual_inputs]
        if len(input_ids) != len(set(input_ids)):
            raise ValueError("input IDs must be unique within one action")
        asset_ids = [item.visual_asset_id for item in self.visual_inputs]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("visual asset IDs must be unique within one action")
        return self


class ObservationRegion(BaseModel):
    """Optional localization of an observation within one supplied image."""

    model_config = ConfigDict(extra="forbid")

    input_id: InputId
    bbox: NormalizedRegion | None = None

    @model_validator(mode="after")
    def validate_bbox(self) -> "ObservationRegion":
        if self.bbox is not None:
            x1, y1, x2, y2 = self.bbox
            if not x1 < x2 or not y1 < y2:
                raise ValueError("bbox must satisfy x1 < x2 and y1 < y2")
        return self


class VisualObservation(BaseModel):
    """One concrete fact visibly supported by one or more supplied images."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    sources: list[ObservationRegion] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sources(self) -> "VisualObservation":
        input_ids = [source.input_id for source in self.sources]
        if len(input_ids) != len(set(input_ids)):
            raise ValueError("Observation input IDs must not contain duplicates")
        return self


class VisualLimitation(BaseModel):
    """A specific reason why the requested observation is incomplete."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    input_ids: list[InputId] = Field(default_factory=list)


class VisualReadResult(BaseModel):
    """Pure observation output; answers and navigation actions are forbidden."""

    model_config = ConfigDict(extra="forbid")

    observations: list[VisualObservation] = Field(max_length=16)
    limitations: list[VisualLimitation]


def validate_visual_read_result(
    request: VisualReadRequest,
    result: VisualReadResult,
) -> VisualReadResult:
    """Reject model output grounded to images not supplied by this action."""

    allowed_ids = {item.input_id for item in request.visual_inputs}
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
    unknown_ids = referenced_ids.difference(allowed_ids)
    if unknown_ids:
        raise ValueError(
            "Visual Reader output references inputs not supplied by the action: "
            + ", ".join(sorted(unknown_ids))
        )
    return result


VISUAL_READER_SYSTEM_PROMPT = load_prompt_text("visual_reader_v0_4.txt")


def visual_reader_user_prompt(request: VisualReadRequest) -> str:
    """Render the deterministic user prompt for one read request."""

    request_json = request.model_dump_json(
        indent=2,
        exclude={"visual_inputs": {"__all__": {"bbox"}}},
    )
    example_ids = [item.input_id for item in request.visual_inputs]
    example_regions = [{"input_id": input_id} for input_id in example_ids]
    output_shape = json.dumps(
        {
            "observations": [
                {
                    "text": "<one concrete fact visibly supported by the listed images>",
                    "sources": example_regions,
                }
            ],
            "limitations": [],
        },
        indent=2,
    )
    return f"""Read request:

{request_json}

Inspect the attached image or images and return exactly this JSON shape:

{output_shape}

When the local problem cannot be resolved completely, keep any reliable partial
observations and describe only the remaining uncertainty in limitations. Do not
add an answer or conclusion field. The example lists all supplied images to show
the shape of a jointly supported Observation; an actual Observation must omit any
 image that does not support that particular fact. Do not add IDs to the output;
 the Environment links this result to the request action_id and assigns stable
 Observation IDs after validation.
"""


ProbeCategory = Literal[
    "single_visual_fact",
    "decomposed_comparison_fact",
    "joint_visual_relation",
]
