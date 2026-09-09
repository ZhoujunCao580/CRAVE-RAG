from __future__ import annotations

import pytest
from pydantic import ValidationError

from softdoc.models import ElementType
from softdoc.visual_scan import (
    PagesVisualScanScope,
    SectionAnchorStatus,
    SectionScopeResolver,
    SectionVisualScanScope,
    VisualScanAggregateResult,
    VisualScanBatchInput,
    VisualScanBatchResult,
    VisualScanItem,
    VisualScanLimitation,
    VisualScanScopeKind,
    validate_visual_scan_batch_result,
)
from softdoc.visual_scan_prompt import build_visual_scan_user_prompt


def _document_with_headings(parsed_document, headings):
    templates = parsed_document.elements[: len(headings)]
    elements = []
    for template, (text, page_number) in zip(templates, headings, strict=True):
        page = parsed_document.pages[page_number - 1]
        elements.append(
            template.model_copy(
                update={
                    "element_id": f"heading:{page_number}:{len(elements)}",
                    "page_id": page.page_id,
                    "page_number": page_number,
                    "element_type": ElementType.HEADING,
                    "text": text,
                }
            )
        )
    return parsed_document.model_copy(update={"elements": elements})


def test_section_scope_resolves_unique_heading_and_preserves_page_bbox(
    parsed_document,
) -> None:
    document = _document_with_headings(
        parsed_document,
        [("Academics and RelatedResources", 2), ("Student Services", 3)],
    )

    result = SectionScopeResolver().resolve(
        document,
        SectionVisualScanScope(
            kind=VisualScanScopeKind.SECTION,
            anchor_text="Academics and Related Resources",
        ),
    )

    assert result.status == SectionAnchorStatus.READY
    assert result.resolved_anchor is not None
    assert result.resolved_anchor.page_number == 2
    assert result.resolved_anchor.heading_text == "Academics and RelatedResources"
    assert result.resolved_anchor.heading_bbox is not None
    # The resolver intentionally does not trust the next parsed Heading as the
    # section end; that boundary is verified during the normal page scan.
    assert len(result.candidates) == 1


def test_missing_section_heading_becomes_controller_gap(parsed_document) -> None:
    document = _document_with_headings(parsed_document, [("Student Services", 2)])

    result = SectionScopeResolver().resolve(
        document,
        SectionVisualScanScope(
            kind=VisualScanScopeKind.SECTION,
            anchor_text="Academics and Related Resources",
        ),
    )

    assert result.status == SectionAnchorStatus.NEEDS_CONTROLLER_ANCHOR
    assert result.resolved_anchor is None
    assert 'Locate the section heading "Academics and Related Resources"' in (
        result.controller_gap or ""
    )


def test_duplicate_heading_on_different_pages_is_ambiguous(parsed_document) -> None:
    document = _document_with_headings(
        parsed_document,
        [("Academics and Related Resources", 2), ("Academics and Related Resources", 3)],
    )

    result = SectionScopeResolver().resolve(
        document,
        SectionVisualScanScope(
            kind=VisualScanScopeKind.SECTION,
            anchor_text="Academics and Related Resources",
        ),
    )

    assert result.status == SectionAnchorStatus.AMBIGUOUS
    assert result.resolved_anchor is None
    assert len(result.candidates) == 2


def test_visual_scan_user_json_has_no_fake_image_or_page_namespace_fields() -> None:
    scan_input = VisualScanBatchInput(
        target_id="Root",
        question="How many tables are included in Pages 5-10?",
        scope=PagesVisualScanScope(
            kind=VisualScanScopeKind.PAGES,
            text="Pages 5-10",
        ),
        batch_index=1,
        batch_count=4,
        input_ids=[f"I{index:03d}" for index in range(1, 6)],
    )

    prompt = build_visual_scan_user_prompt(scan_input)

    assert '"input_ids": [' in prompt
    assert '"I001"' in prompt
    assert "image" not in prompt
    assert "physical_page_number" not in prompt
    assert "visible_page_label" not in prompt


def test_twenty_page_scan_uses_global_ids_across_four_batches() -> None:
    all_ids = [f"I{index:03d}" for index in range(1, 21)]
    batches = [all_ids[index : index + 5] for index in range(0, 20, 5)]

    assert batches[0] == ["I001", "I002", "I003", "I004", "I005"]
    assert batches[-1] == ["I016", "I017", "I018", "I019", "I020"]
    assert len({item for batch in batches for item in batch}) == 20


def test_resolved_batch_requires_exact_partial_count() -> None:
    result = VisualScanBatchResult(
        batch_index=1,
        items=[
            VisualScanItem(input_id="I002", description="One table.", count=1),
            VisualScanItem(input_id="I004", description="Two tables.", count=2),
        ],
        partial_count=3,
    )
    assert result.partial_count == 3

    with pytest.raises(ValidationError, match="sum of item counts"):
        VisualScanBatchResult(
            batch_index=1,
            items=[VisualScanItem(input_id="I002", description="One table.", count=1)],
            partial_count=2,
        )


def test_unreadable_page_cannot_be_reported_as_zero() -> None:
    with pytest.raises(ValidationError, match="must be null"):
        VisualScanBatchResult(
            batch_index=1,
            items=[],
            partial_count=0,
            limitations=[
                VisualScanLimitation(
                    input_id="I007",
                    description="The page is too blurred to assess.",
                )
            ],
        )


def test_batch_result_cannot_reference_invisible_input_id() -> None:
    scan_input = VisualScanBatchInput(
        target_id="Root",
        question="How many figures are shown?",
        scope={"kind": "whole_document"},
        batch_index=1,
        batch_count=1,
        input_ids=["I001"],
    )
    result = VisualScanBatchResult(
        batch_index=1,
        items=[VisualScanItem(input_id="I999", description="One figure.", count=1)],
        partial_count=1,
    )

    with pytest.raises(ValueError, match="not visible"):
        validate_visual_scan_batch_result(scan_input, result)


def test_complete_aggregate_requires_answer_and_no_limitation() -> None:
    result = VisualScanAggregateResult(
        answer_candidate="7",
        supporting_input_ids=["I002", "I004"],
        scope_complete=True,
    )
    assert result.answer_candidate == "7"

    with pytest.raises(ValidationError, match="requires an answer"):
        VisualScanAggregateResult(scope_complete=True)

    incomplete = VisualScanAggregateResult(
        scope_complete=False,
        limitation="I007 remains unreadable.",
    )
    assert incomplete.answer_candidate is None
