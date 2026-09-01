from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from softdoc.models import (
    Element,
    ElementType,
    Relation,
    RelationSource,
    RelationStatus,
    RelationType,
)
from softdoc.retrieval import BM25Index, SearchUnitBuilder, SubQuestionInput
from softdoc.visual_retrieval import (
    VISUAL_RETRIEVAL_METADATA_KEY,
    VisualRetrievalDraft,
    VisualRetrievalResult,
    VisualSearchIdentity,
    apply_visual_retrieval_result,
    build_visual_retrieval_request,
    enrich_visual_retrieval,
    visual_retrieval_descriptor,
)


def test_visual_search_identity_requires_three_distinct_keywords() -> None:
    with pytest.raises(ValueError, match="three unique visual keywords"):
        VisualSearchIdentity(
            search_summary="A chart about quarterly revenue.",
            keywords=["revenue", "Revenue", " revenue "],
        )


def _visual_document(parsed_document, tmp_path: Path):
    document = parsed_document.model_copy(deep=True)
    figure = next(
        item for item in document.elements if item.element_type == ElementType.FIGURE
    )
    image_path = tmp_path / "figure.png"
    Image.new("RGB", (80, 60), color=(220, 230, 240)).save(image_path)
    figure.image_path = image_path
    figure.crop_image_path = None
    figure.reference_label = None
    figure.text = None
    figure.summary = None
    figure.keywords = []
    figure.metadata.pop(VISUAL_RETRIEVAL_METADATA_KEY, None)
    return document, figure


def test_visual_description_becomes_auditable_bm25_search_metadata(
    parsed_document,
    tmp_path: Path,
) -> None:
    document, figure = _visual_document(parsed_document, tmp_path)
    request = build_visual_retrieval_request(document, tmp_path)
    visual_input = next(
        item for item in request.visual_inputs if item.element_id == figure.element_id
    )
    result = VisualRetrievalResult(
        descriptors=[
            VisualRetrievalDraft(
                input_id=visual_input.input_id,
                search_summary=(
                    "A bar chart compares quarterly revenue for the Orion product."
                ),
                keywords=["Orion", "quarterly revenue", "bar chart"],
            )
        ]
    )

    descriptors = apply_visual_retrieval_result(
        document,
        request,
        result,
        generator_model="mock-vlm",
        prompt_version="visual-retrieval-prompt-draft",
    )
    units = SearchUnitBuilder().build(document)
    figure_units = [
        unit for unit in units.units if unit.element_id == figure.element_id
    ]
    bm25 = BM25Index(units).search(
        SubQuestionInput(subquestion_id="Q1", text="Orion quarterly revenue")
    )

    assert figure.text is None
    assert figure.summary == result.descriptors[0].search_summary
    assert figure.keywords == result.descriptors[0].keywords
    assert descriptors[0].purpose == "search_only"
    assert visual_retrieval_descriptor(figure) == descriptors[0]
    assert len(figure_units) == 1
    assert figure_units[0].visual_descriptor_id == descriptors[0].descriptor_id
    assert "Visual search summary" in figure_units[0].search_text
    assert bm25.candidates[0].element_id == figure.element_id


def test_ordinary_summary_is_not_silently_indexed(
    parsed_document,
    tmp_path: Path,
) -> None:
    document, figure = _visual_document(parsed_document, tmp_path)
    figure.summary = "Unverified nebula revenue claim"
    figure.keywords = ["nebula"]

    units = SearchUnitBuilder().build(document)

    assert not [unit for unit in units.units if unit.element_id == figure.element_id]


def test_tampered_descriptor_metadata_is_not_indexed(
    parsed_document,
    tmp_path: Path,
) -> None:
    document, figure = _visual_document(parsed_document, tmp_path)
    request = build_visual_retrieval_request(document, tmp_path)
    visual_input = next(
        item for item in request.visual_inputs if item.element_id == figure.element_id
    )
    apply_visual_retrieval_result(
        document,
        request,
        VisualRetrievalResult(
            descriptors=[
                VisualRetrievalDraft(
                    input_id=visual_input.input_id,
                    search_summary="A process diagram for the Aurora workflow.",
                    keywords=["Aurora", "workflow", "process diagram"],
                )
            ]
        ),
        generator_model="mock-vlm",
        prompt_version="visual-retrieval-prompt-draft",
    )
    figure.summary = "A conflicting summary added after generation."

    units = SearchUnitBuilder().build(document)

    assert visual_retrieval_descriptor(figure) is None
    assert not [unit for unit in units.units if unit.element_id == figure.element_id]


def test_unknown_visual_output_is_rejected_before_document_mutation(
    parsed_document,
    tmp_path: Path,
) -> None:
    document, figure = _visual_document(parsed_document, tmp_path)
    request = build_visual_retrieval_request(document, tmp_path)

    with pytest.raises(ValueError, match="unknown inputs"):
        apply_visual_retrieval_result(
            document,
            request,
            VisualRetrievalResult(
                descriptors=[
                    VisualRetrievalDraft(
                        input_id="I-does-not-exist",
                        search_summary="An unsupported visual retrieval output.",
                        keywords=["unsupported", "visual", "retrieval"],
                    )
                ]
            ),
            generator_model="mock-vlm",
            prompt_version="visual-retrieval-prompt-draft",
        )

    assert figure.summary is None
    assert figure.keywords == []
    assert VISUAL_RETRIEVAL_METADATA_KEY not in figure.metadata


def test_request_exposes_only_confirmed_caption_context(
    parsed_document,
    tmp_path: Path,
) -> None:
    document, figure = _visual_document(parsed_document, tmp_path)
    confirmed = Element(
        element_id="caption:confirmed",
        document_id=document.document_id,
        page_id=figure.page_id,
        page_number=figure.page_number,
        element_type=ElementType.CAPTION,
        reading_order=figure.reading_order + 1,
        text="Figure 7. Orion quarterly revenue.",
        provenance=figure.provenance.model_copy(deep=True),
    )
    candidate = confirmed.model_copy(
        update={
            "element_id": "caption:candidate",
            "text": "Unverified candidate caption.",
            "reading_order": figure.reading_order + 2,
        }
    )
    document.elements.extend([confirmed, candidate])
    document.relations.extend(
        [
            Relation(
                relation_id="relation:confirmed-caption",
                source_id=confirmed.element_id,
                target_id=figure.element_id,
                relation_type=RelationType.CAPTION_OF,
                confidence=1.0,
                status=RelationStatus.CONFIRMED,
                created_by=RelationSource.DETERMINISTIC_RULE,
            ),
            Relation(
                relation_id="relation:candidate-caption",
                source_id=candidate.element_id,
                target_id=figure.element_id,
                relation_type=RelationType.CAPTION_OF,
                confidence=0.6,
                status=RelationStatus.CANDIDATE,
                created_by=RelationSource.LAYOUT_HEURISTIC,
            ),
        ]
    )

    request = build_visual_retrieval_request(document, tmp_path)
    visual_input = next(
        item for item in request.visual_inputs if item.element_id == figure.element_id
    )

    assert "Figure 7. Orion quarterly revenue." in visual_input.caption_texts
    assert "Unverified candidate caption." not in visual_input.caption_texts


def test_enrichment_requires_exact_batch_coverage(
    parsed_document,
    tmp_path: Path,
) -> None:
    document, figure = _visual_document(parsed_document, tmp_path)

    class EmptyBackend:
        prompt_version = "visual-retrieval-test"

        def describe(self, request):
            return VisualRetrievalResult(descriptors=[])

    with pytest.raises(ValueError, match="cover the request exactly"):
        enrich_visual_retrieval(
            document,
            tmp_path,
            EmptyBackend(),
            generator_model="mock-vlm",
        )

    assert figure.summary is None
    assert figure.keywords == []
