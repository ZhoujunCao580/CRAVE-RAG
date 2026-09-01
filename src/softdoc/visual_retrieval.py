"""Auditable, retrieval-only descriptions for visual SoftDoc Elements.

The descriptions produced here are candidate-discovery metadata.  They are
never Observations or Evidence, and a Reader must still inspect the original
visual asset before a fact can enter EvidenceMemory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field, field_validator, model_validator

from softdoc.ids import stable_digest
from softdoc.models import (
    Document,
    Element,
    ElementType,
    RelationStatus,
    RelationType,
    SoftDocModel,
)
from softdoc.prompts import load_prompt_text


VISUAL_RETRIEVAL_REQUEST_VERSION = "visual-retrieval-request-v0.1"
VISUAL_RETRIEVAL_DESCRIPTOR_VERSION = "visual-retrieval-descriptor-v0.1"
VISUAL_RETRIEVAL_METADATA_KEY = "visual_retrieval_descriptor"
VISUAL_RETRIEVAL_PROMPT_VERSION = "visual-retrieval-v0.1"
VISUAL_RETRIEVAL_SYSTEM_PROMPT = load_prompt_text(
    "visual_retrieval_v0_1.txt"
).removesuffix("\n")

_SUPPORTED_ELEMENT_TYPES = {
    ElementType.FIGURE,
    ElementType.CHART,
    ElementType.TABLE,
}


class VisualRetrievalInput(SoftDocModel):
    """One real image supplied to the retrieval-description generator."""

    input_id: str = Field(min_length=1)
    element_id: str = Field(min_length=1)
    element_type: ElementType
    visual_asset_id: str = Field(min_length=1)
    visual_asset_path: Path
    visual_asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_label: str | None = Field(default=None, min_length=1)
    section_path: list[str] = Field(default_factory=list)
    caption_texts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_visual_type(self) -> "VisualRetrievalInput":
        if self.element_type not in _SUPPORTED_ELEMENT_TYPES:
            raise ValueError(
                "Visual retrieval inputs must be Figure, Chart, or Table Elements"
            )
        return self

    @field_validator("section_path", "caption_texts")
    @classmethod
    def normalize_context(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            item = " ".join(value.split())
            if not item:
                raise ValueError("Visual retrieval context must not contain blanks")
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(item)
        return normalized


class VisualRetrievalRequest(SoftDocModel):
    """Batch of visual assets that may receive search-only descriptions."""

    request_version: Literal["visual-retrieval-request-v0.1"] = (
        VISUAL_RETRIEVAL_REQUEST_VERSION
    )
    document_id: str = Field(min_length=1)
    visual_inputs: list[VisualRetrievalInput] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_inputs(self) -> "VisualRetrievalRequest":
        input_ids = [item.input_id for item in self.visual_inputs]
        element_ids = [item.element_id for item in self.visual_inputs]
        if len(input_ids) != len(set(input_ids)):
            raise ValueError("Visual retrieval input IDs must be unique")
        if len(element_ids) != len(set(element_ids)):
            raise ValueError("Each Element may appear only once in a request")
        return self


class VisualSearchIdentity(SoftDocModel):
    """Exact schema returned by the retrieval-only visual model call."""

    search_summary: str = Field(min_length=12, max_length=300)
    keywords: list[str] = Field(min_length=3, max_length=8)

    @field_validator("search_summary")
    @classmethod
    def normalize_search_summary(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Visual retrieval search summary must not be blank")
        return normalized

    @field_validator("keywords")
    @classmethod
    def normalize_keywords(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            keyword = " ".join(value.split())
            if not keyword:
                raise ValueError("Visual retrieval keywords must not be blank")
            key = keyword.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(keyword)
        if len(normalized) < 3:
            raise ValueError("At least three unique visual keywords are required")
        return normalized


class VisualRetrievalDraft(VisualSearchIdentity):
    """Model output bound by the program to one requested image."""

    input_id: str = Field(min_length=1)


class VisualRetrievalResult(SoftDocModel):
    """Raw structured output from one descriptor-generation call."""

    descriptors: list[VisualRetrievalDraft] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_inputs(self) -> "VisualRetrievalResult":
        input_ids = [item.input_id for item in self.descriptors]
        if len(input_ids) != len(set(input_ids)):
            raise ValueError("Visual retrieval output input IDs must be unique")
        return self


class VisualRetrievalBackend(Protocol):
    """Injectable generator boundary for offline visual search metadata."""

    prompt_version: str

    def describe(self, request: VisualRetrievalRequest) -> VisualRetrievalResult: ...


class VisualRetrievalDescriptor(SoftDocModel):
    """Persisted provenance for one retrieval-only visual description."""

    descriptor_version: Literal["visual-retrieval-descriptor-v0.1"] = (
        VISUAL_RETRIEVAL_DESCRIPTOR_VERSION
    )
    descriptor_id: str = Field(min_length=1)
    purpose: Literal["search_only"] = "search_only"
    element_id: str = Field(min_length=1)
    visual_asset_id: str = Field(min_length=1)
    visual_asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generator_model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    search_summary: str = Field(min_length=12, max_length=300)
    keywords: list[str] = Field(min_length=3, max_length=8)


def visual_retrieval_user_prompt(visual_input: VisualRetrievalInput) -> str:
    """Render only small, trusted SoftDoc context beside the supplied pixels."""

    payload = {
        "input_id": visual_input.input_id,
        "element_type": visual_input.element_type.value,
        "reference_label": visual_input.reference_label,
        "section_path": visual_input.section_path,
        "confirmed_caption_texts": visual_input.caption_texts,
    }
    return (
        "Generate the search identity for the supplied visual element.\n\n"
        "SoftDoc context:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def build_visual_retrieval_request(
    document: Document,
    asset_root: Path,
    *,
    element_ids: set[str] | None = None,
) -> VisualRetrievalRequest:
    """Collect real Figure/Chart/Table images without inventing page assets."""

    asset_root = Path(asset_root)
    inputs: list[VisualRetrievalInput] = []
    elements_by_id = {element.element_id: element for element in document.elements}
    captions_by_target: dict[str, list[str]] = {}
    for relation in document.relations:
        if (
            relation.relation_type != RelationType.CAPTION_OF
            or relation.status != RelationStatus.CONFIRMED
        ):
            continue
        caption = elements_by_id.get(relation.source_id)
        target = elements_by_id.get(relation.target_id)
        if (
            caption is None
            or target is None
            or caption.element_type != ElementType.CAPTION
            or not (caption.text or "").strip()
        ):
            continue
        captions_by_target.setdefault(target.element_id, []).append(caption.text or "")
    for element in document.elements:
        if element_ids is not None and element.element_id not in element_ids:
            continue
        if element.element_type not in _SUPPORTED_ELEMENT_TYPES:
            continue
        relative_path = element.visual_asset_path
        if relative_path is None:
            continue
        resolved = _resolve_asset_path(relative_path, asset_root)
        if not resolved.is_file():
            continue
        asset_sha256 = _sha256_file(resolved)
        visual_asset_id = "visual:" + stable_digest(
            document.document_id,
            element.element_id,
            asset_sha256,
        )
        inputs.append(
            VisualRetrievalInput(
                input_id=f"I{len(inputs) + 1}",
                element_id=element.element_id,
                element_type=element.element_type,
                visual_asset_id=visual_asset_id,
                visual_asset_path=resolved,
                visual_asset_sha256=asset_sha256,
                reference_label=(element.reference_label or None),
                section_path=list(element.section_path or []),
                caption_texts=captions_by_target.get(element.element_id, []),
            )
        )
    return VisualRetrievalRequest(
        document_id=document.document_id,
        visual_inputs=inputs,
    )


def apply_visual_retrieval_result(
    document: Document,
    request: VisualRetrievalRequest,
    result: VisualRetrievalResult,
    *,
    generator_model: str,
    prompt_version: str,
) -> list[VisualRetrievalDescriptor]:
    """Atomically attach validated search metadata to its source Elements."""

    if request.document_id != document.document_id:
        raise ValueError("Visual retrieval request belongs to another Document")
    inputs_by_id = {item.input_id: item for item in request.visual_inputs}
    elements_by_id = {element.element_id: element for element in document.elements}
    unknown_inputs = {
        item.input_id for item in result.descriptors
    } - inputs_by_id.keys()
    if unknown_inputs:
        raise ValueError(
            "Visual retrieval result references unknown inputs: "
            + ", ".join(sorted(unknown_inputs))
        )

    descriptors: list[VisualRetrievalDescriptor] = []
    target_elements: list[Element] = []
    for draft in result.descriptors:
        visual_input = inputs_by_id[draft.input_id]
        element = elements_by_id.get(visual_input.element_id)
        if element is None:
            raise ValueError(
                f"Visual retrieval Element is missing: {visual_input.element_id}"
            )
        if element.element_type != visual_input.element_type:
            raise ValueError("Visual retrieval Element type changed after request")
        if element.visual_asset_path is None:
            raise ValueError("Visual retrieval Element lost its visual asset")
        current_path = Path(visual_input.visual_asset_path)
        if not current_path.is_file():
            raise ValueError(f"Visual retrieval asset is missing: {current_path}")
        if _sha256_file(current_path) != visual_input.visual_asset_sha256:
            raise ValueError("Visual retrieval asset changed after generation")
        descriptor_id = "visual-descriptor:" + stable_digest(
            document.document_id,
            element.element_id,
            visual_input.visual_asset_sha256,
            generator_model,
            prompt_version,
            draft.search_summary,
            draft.keywords,
        )
        descriptors.append(
            VisualRetrievalDescriptor(
                descriptor_id=descriptor_id,
                element_id=element.element_id,
                visual_asset_id=visual_input.visual_asset_id,
                visual_asset_sha256=visual_input.visual_asset_sha256,
                generator_model=generator_model,
                prompt_version=prompt_version,
                search_summary=draft.search_summary,
                keywords=draft.keywords,
            )
        )
        target_elements.append(element)

    for element, descriptor in zip(target_elements, descriptors, strict=True):
        element.summary = descriptor.search_summary
        element.keywords = list(descriptor.keywords)
        element.metadata[VISUAL_RETRIEVAL_METADATA_KEY] = descriptor.model_dump(
            mode="json"
        )
    return descriptors


def enrich_visual_retrieval(
    document: Document,
    asset_root: Path,
    backend: VisualRetrievalBackend,
    *,
    generator_model: str,
    element_ids: set[str] | None = None,
) -> list[VisualRetrievalDescriptor]:
    """Generate and atomically attach a complete visual retrieval batch."""

    request = build_visual_retrieval_request(
        document,
        asset_root,
        element_ids=element_ids,
    )
    result = backend.describe(request)
    expected = {item.input_id for item in request.visual_inputs}
    returned = {item.input_id for item in result.descriptors}
    if returned != expected:
        missing = sorted(expected - returned)
        extra = sorted(returned - expected)
        raise ValueError(
            "Visual retrieval result must cover the request exactly; "
            f"missing={missing}, extra={extra}"
        )
    return apply_visual_retrieval_result(
        document,
        request,
        result,
        generator_model=generator_model,
        prompt_version=backend.prompt_version,
    )


def visual_retrieval_descriptor(
    element: Element,
) -> VisualRetrievalDescriptor | None:
    """Return only internally consistent, explicitly search-only metadata."""

    payload = element.metadata.get(VISUAL_RETRIEVAL_METADATA_KEY)
    if not isinstance(payload, dict):
        return None
    try:
        descriptor = VisualRetrievalDescriptor.model_validate(payload)
    except ValueError:
        return None
    if descriptor.element_id != element.element_id:
        return None
    if descriptor.search_summary != element.summary:
        return None
    if descriptor.keywords != element.keywords:
        return None
    return descriptor


def _resolve_asset_path(path: Path, asset_root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = asset_root / candidate
    return candidate.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
