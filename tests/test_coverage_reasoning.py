from __future__ import annotations

from softdoc.coverage_reasoning import (
    CoverageBatchCheckInput,
    CoverageBatchCheckResult,
    CoverageExecutionStatus,
    CoverageInventoryStatus,
    CoverageItemAssessment,
    CoverageItemVerdict,
    CoverageObservation,
    CoverageOperator,
    CoverageRequirement,
    CoverageScopeResolver,
    CoverageScopeStatus,
    CoverageSourceType,
    PageNumberNamespace,
    QuestionCoveragePlan,
    apply_coverage_batch_result,
    build_coverage_inventory,
    validate_coverage_batch_result,
)
from softdoc.models import ElementType


def _requirement(
    scope_text: str,
    *,
    item_type: str = "figure",
    source_type: CoverageSourceType = CoverageSourceType.FIGURE,
    predicate: str | None = None,
) -> CoverageRequirement:
    return CoverageRequirement(
        operator=CoverageOperator.COUNT,
        scope_text=scope_text,
        item_type=item_type,
        source_type=source_type,
        predicate=predicate,
    )


def test_page_scope_records_both_interpretations_and_supports_override(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    # Printed pages 1-2 are physical pages 2-3.
    for page in document.pages:
        object.__setattr__(page, "display_page_label", None)
        object.__setattr__(page, "display_page_label_confidence", None)
        object.__setattr__(page, "page_label_aliases", [])
    for page, label in zip(document.pages[1:3], ["1", "2"], strict=True):
        object.__setattr__(page, "display_page_label", label)
        object.__setattr__(page, "display_page_label_confidence", 0.99)
        object.__setattr__(page, "page_label_aliases", [label])

    resolver = CoverageScopeResolver()
    automatic = resolver.resolve(_requirement("Pages 1-2"), document)

    assert automatic.chosen_namespace == PageNumberNamespace.PRINTED_PAGE_LABEL
    assert automatic.status == CoverageScopeStatus.RESOLVED
    assert automatic.requires_review is True
    assert automatic.resolved_page_ids == [
        document.pages[1].page_id,
        document.pages[2].page_id,
    ]
    assert {candidate.namespace for candidate in automatic.candidates} == {
        PageNumberNamespace.PRINTED_PAGE_LABEL,
        PageNumberNamespace.PHYSICAL_PAGE_ORDER,
    }

    corrected = resolver.resolve(
        _requirement("Pages 1-2"),
        document,
        namespace_override=PageNumberNamespace.PHYSICAL_PAGE_ORDER,
    )
    assert corrected.decision_reason == "explicit_namespace_override"
    assert corrected.resolved_page_ids == [
        document.pages[0].page_id,
        document.pages[1].page_id,
    ]


def test_page_range_never_mixes_printed_and_physical_namespaces(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    for page in document.pages:
        object.__setattr__(page, "display_page_label", None)
        object.__setattr__(page, "display_page_label_confidence", None)
        object.__setattr__(page, "page_label_aliases", [])
    page = document.pages[2]
    object.__setattr__(page, "display_page_label", "1")
    object.__setattr__(page, "display_page_label_confidence", 0.99)
    object.__setattr__(page, "page_label_aliases", ["1"])

    resolution = CoverageScopeResolver().resolve(
        _requirement("Pages 1-2"), document
    )

    assert resolution.chosen_namespace == PageNumberNamespace.PRINTED_PAGE_LABEL
    assert resolution.status == CoverageScopeStatus.PARTIAL
    assert resolution.resolved_page_ids == [page.page_id]
    assert resolution.missing_labels == ["2"]
    assert resolution.requires_review is True

    inventory = build_coverage_inventory(
        _requirement("Pages 1-2"), resolution, document
    )
    assert inventory.status == CoverageInventoryStatus.BLOCKED
    assert inventory.structural_count is None


def test_slide_scope_uses_physical_order_even_when_printed_labels_exist(
    parsed_document,
) -> None:
    document = parsed_document.model_copy(deep=True)
    page = document.pages[2]
    object.__setattr__(page, "display_page_label", "1")
    object.__setattr__(page, "display_page_label_confidence", 0.99)
    object.__setattr__(page, "page_label_aliases", ["1"])

    resolution = CoverageScopeResolver().resolve(
        _requirement("Slides 1-2"), document
    )

    assert resolution.chosen_namespace == PageNumberNamespace.PHYSICAL_PAGE_ORDER
    assert resolution.resolved_page_ids == [
        document.pages[0].page_id,
        document.pages[1].page_id,
    ]


def test_out_of_range_scope_is_unresolved_and_cannot_generate_count(
    parsed_document,
) -> None:
    requirement = _requirement("Pages 400-640")
    resolution = CoverageScopeResolver().resolve(requirement, parsed_document)
    inventory = build_coverage_inventory(requirement, resolution, parsed_document)

    assert resolution.status == CoverageScopeStatus.UNRESOLVED
    assert resolution.missing_labels[0] == "400"
    assert resolution.missing_labels[-1] == "640"
    assert inventory.status == CoverageInventoryStatus.BLOCKED
    assert inventory.structural_count is None


def test_structural_count_is_only_emitted_for_literal_matching_types(
    parsed_document,
) -> None:
    resolver = CoverageScopeResolver()
    requirement = _requirement("entire document")
    resolution = resolver.resolve(requirement, parsed_document)
    inventory = build_coverage_inventory(requirement, resolution, parsed_document)
    expected = sum(
        element.element_type == ElementType.FIGURE
        for element in parsed_document.elements
    )

    assert inventory.status == CoverageInventoryStatus.COMPLETE
    assert inventory.structural_count == expected

    semantic = _requirement(
        "entire document",
        item_type="person",
        source_type=CoverageSourceType.FIGURE,
    )
    semantic_inventory = build_coverage_inventory(
        semantic,
        resolver.resolve(semantic, parsed_document),
        parsed_document,
    )
    assert semantic_inventory.structural_count is None


def test_confirmed_cross_page_table_fragments_count_once(parsed_document) -> None:
    document = parsed_document.model_copy(deep=True)
    tables = [
        element
        for element in document.elements
        if element.element_type == ElementType.TABLE
    ]
    assert tables
    second = tables[0].model_copy(deep=True)
    second.element_id += ":continuation"
    second.reading_order = max(
        element.reading_order
        for element in document.elements
        if element.page_id == second.page_id
    ) + 1
    document.elements.append(second)
    page = next(item for item in document.pages if item.page_id == second.page_id)
    page.element_ids.append(second.element_id)
    page.reading_order.append(second.element_id)
    tables.append(second)
    for index, table in enumerate(tables[:2]):
        table.metadata["cross_page_table_fragment"] = {
            "status": "confirmed",
            "group_id": "table-group-1",
            "fragment_index": index,
        }
    requirement = _requirement(
        "entire document",
        item_type="table",
        source_type=CoverageSourceType.TABLE,
    )

    inventory = build_coverage_inventory(
        requirement,
        CoverageScopeResolver().resolve(requirement, document),
        document,
    )

    assert inventory.structural_count == len(tables) - 1
    grouped = next(
        item
        for item in inventory.items
        if item.inventory_id == "table-group:table-group-1"
    )
    assert grouped.source_ids == [table.element_id for table in tables[:2]]
    assert grouped.deduplication_reason == "confirmed_cross_page_table_group"


def test_semantic_batch_must_cover_exact_inventory_and_use_grounded_observations(
    parsed_document,
) -> None:
    requirement = _requirement(
        "entire document",
        item_type="person",
        source_type=CoverageSourceType.FIGURE,
    )
    resolution = CoverageScopeResolver().resolve(requirement, parsed_document)
    inventory = build_coverage_inventory(requirement, resolution, parsed_document)
    assert len(inventory.items) >= 2
    first, second = inventory.items[:2]
    checker_input = CoverageBatchCheckInput(
        action_id="action:coverage",
        question_id="Q1",
        question_text="How many people are shown in all figures?",
        requirement=requirement,
        inventory_items=[first, second],
        observations=[
            CoverageObservation(
                observation_id="obs:first",
                text="Two people are visible.",
                source_ids=first.source_ids,
                page_ids=first.page_ids,
            ),
            CoverageObservation(
                observation_id="obs:second",
                text="No people are visible.",
                source_ids=second.source_ids,
                page_ids=second.page_ids,
            ),
        ],
    )

    incomplete = CoverageBatchCheckResult(
        action_id="action:coverage",
        assessments=[
            CoverageItemAssessment(
                inventory_id=first.inventory_id,
                verdict=CoverageItemVerdict.MATCHED,
                matched_count=2,
                observation_ids=["obs:first"],
                rationale="Two distinct people are visible.",
            )
        ],
    )
    try:
        validate_coverage_batch_result(checker_input, incomplete)
    except ValueError as exc:
        assert "every supplied inventory item exactly once" in str(exc)
    else:
        raise AssertionError("Missing inventory assessment must be rejected")

    wrong_source = CoverageBatchCheckResult(
        action_id="action:coverage",
        assessments=[
            CoverageItemAssessment(
                inventory_id=first.inventory_id,
                verdict=CoverageItemVerdict.MATCHED,
                matched_count=2,
                observation_ids=["obs:second"],
                rationale="This cites the wrong source.",
            ),
            CoverageItemAssessment(
                inventory_id=second.inventory_id,
                verdict=CoverageItemVerdict.NOT_MATCHED,
                matched_count=0,
                observation_ids=["obs:second"],
                rationale="No people are visible.",
            ),
        ],
    )
    try:
        validate_coverage_batch_result(checker_input, wrong_source)
    except ValueError as exc:
        assert "not grounded" in str(exc)
    else:
        raise AssertionError("Cross-item Observation leakage must be rejected")


def test_semantic_coverage_completes_only_after_every_item_is_resolved(
    parsed_document,
) -> None:
    requirement = _requirement(
        "entire document",
        item_type="person",
        source_type=CoverageSourceType.FIGURE,
    )
    resolution = CoverageScopeResolver().resolve(requirement, parsed_document)
    inventory = build_coverage_inventory(requirement, resolution, parsed_document)
    assert len(inventory.items) >= 2
    first, second = inventory.items[:2]
    plan = QuestionCoveragePlan(
        question_id="Q1",
        requirement=requirement,
        scope_resolution=resolution,
        inventory=inventory.model_copy(update={"items": [first, second]}),
    )

    partial = apply_coverage_batch_result(
        plan,
        CoverageBatchCheckResult(
            action_id="action:1",
            assessments=[
                CoverageItemAssessment(
                    inventory_id=first.inventory_id,
                    verdict=CoverageItemVerdict.MATCHED,
                    matched_count=2,
                    observation_ids=["obs:1"],
                    rationale="Two people are visible.",
                )
            ],
        ),
    )
    assert partial.execution.status == CoverageExecutionStatus.IN_PROGRESS
    assert partial.execution.matched_count is None

    complete = apply_coverage_batch_result(
        partial,
        CoverageBatchCheckResult(
            action_id="action:2",
            assessments=[
                CoverageItemAssessment(
                    inventory_id=second.inventory_id,
                    verdict=CoverageItemVerdict.NOT_MATCHED,
                    matched_count=0,
                    observation_ids=["obs:2"],
                    rationale="No people are visible.",
                )
            ],
        ),
    )
    assert complete.execution.status == CoverageExecutionStatus.COMPLETE
    assert complete.execution.matched_count == 2
    assert complete.execution.completed_batch_count == 2
