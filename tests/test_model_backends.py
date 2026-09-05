from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from softdoc.answering import AnswerEvidence, AnswerInput, AnswerResult
from softdoc.model_backends import (
    OllamaAnswererBackend,
    OllamaEvidenceCheckerBackend,
    OllamaModelConfig,
    OllamaStructuredClient,
    OllamaVisualRetrievalBackend,
    OllamaVisualReaderBackend,
)
from softdoc.models import Document, Element, ElementType, Page, Provenance
from softdoc.reading_environment import ReaderContext
from softdoc.reading_state import (
    EvidenceCheckInput,
    ObservationSourceRef,
    ReadInput,
    ReaderKind,
    ReadingSourceType,
    ReadRepresentation,
    RootQuestion,
    StoredObservation,
    initialize_evidence_memory,
)
from softdoc.visual_retrieval import build_visual_retrieval_request


class FakeTransport:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, Any], float]] = []

    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        self.calls.append((url, payload, timeout_seconds))
        return self.response


def _provenance(owner: str) -> Provenance:
    return Provenance(
        provenance_id=f"prov:{owner}",
        adapter="test",
        source_path=Path("fixture.json"),
        source_locator=owner,
    )


def test_visual_reader_backend_sends_pixels_and_maps_local_sources(tmp_path: Path) -> None:
    image_path = tmp_path / "figure.png"
    Image.new("RGB", (20, 20), "white").save(image_path)
    doc_id = "doc:test"
    page = Page(
        page_id="page:1",
        document_id=doc_id,
        page_index=0,
        page_number=1,
        width=100,
        height=100,
        element_ids=["figure:1"],
        reading_order=["figure:1"],
        provenance=_provenance("page:1"),
    )
    element = Element(
        element_id="figure:1",
        document_id=doc_id,
        page_id=page.page_id,
        page_number=1,
        element_type=ElementType.FIGURE,
        reading_order=0,
        image_path=image_path,
        provenance=_provenance("figure:1"),
    )
    document = Document(
        document_id=doc_id,
        source_path=Path("paper.pdf"),
        pages=[page],
        elements=[element],
        provenance=_provenance(doc_id),
    )
    transport = FakeTransport(
        {
            "model": "vision-test",
            "message": {
                "content": (
                    '{"observations":[{"text":"The bar is 42.",'
                    '"sources":[{"input_id":"I1","bbox":null}]}],'
                    '"limitations":[]}'
                )
            },
        }
    )
    client = OllamaStructuredClient(
        OllamaModelConfig(model="vision-test"), transport
    )
    backend = OllamaVisualReaderBackend(client)
    output = backend.read(
        ReaderContext(
            action_id="action:1",
            question_id="Q1",
            local_problem="What is the bar value?",
            document=document,
            inputs=(
                ReadInput(
                    input_id="I1",
                    source_id=element.element_id,
                    source_type=ReadingSourceType.ELEMENT,
                    representation=ReadRepresentation.ELEMENT_VISUAL,
                    document_id=doc_id,
                    page_id=page.page_id,
                    element_id=element.element_id,
                    visual_asset_id="visual:1",
                    visual_asset_path=image_path,
                ),
            ),
            elements_by_id={element.element_id: element},
            pages_by_id={page.page_id: page},
            table_views_by_id={},
        )
    )

    assert output.reader_kind == ReaderKind.VISUAL
    assert output.observations[0].text == "The bar is 42."
    assert output.observations[0].sources[0].input_id == "I1"
    payload = transport.calls[0][1]
    assert payload["format"]["title"] == "VisualReadResult"
    assert len(payload["messages"][1]["images"]) == 1
    assert client.call_records[0].component == "visual_reader"


def test_visual_retrieval_backend_returns_program_bound_identity(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "chart.png"
    Image.new("RGB", (20, 20), "white").save(image_path)
    doc_id = "doc:visual-index"
    page = Page(
        page_id="page:visual-index",
        document_id=doc_id,
        page_index=0,
        page_number=1,
        width=100,
        height=100,
        element_ids=["chart:1"],
        reading_order=["chart:1"],
        provenance=_provenance("page:visual-index"),
    )
    chart = Element(
        element_id="chart:1",
        document_id=doc_id,
        page_id=page.page_id,
        page_number=1,
        element_type=ElementType.CHART,
        reading_order=0,
        image_path=image_path,
        section_path=["Results"],
        provenance=_provenance("chart:1"),
    )
    document = Document(
        document_id=doc_id,
        source_path=Path("paper.pdf"),
        pages=[page],
        elements=[chart],
        provenance=_provenance(doc_id),
    )
    transport = FakeTransport(
        {
            "model": "vision-test",
            "message": {
                "content": (
                    '{"search_summary":"A line chart compares AP and AP50 '
                    'across decoder layers.","keywords":["AP","AP50",'
                    '"decoder layers"]}'
                )
            },
        }
    )
    client = OllamaStructuredClient(
        OllamaModelConfig(model="vision-test"), transport
    )
    request = build_visual_retrieval_request(document, tmp_path)

    result = OllamaVisualRetrievalBackend(client).describe(request)

    assert result.descriptors[0].input_id == "I1"
    assert result.descriptors[0].keywords == ["AP", "AP50", "decoder layers"]
    payload = transport.calls[0][1]
    assert payload["format"]["title"] == "VisualSearchIdentity"
    assert payload["messages"][1]["images"]
    assert '"element_type": "chart"' in payload["messages"][1]["content"]
    assert client.call_records[0].component == "visual_retrieval"


def test_checker_and_answerer_backends_use_their_frozen_schemas() -> None:
    root = RootQuestion(question_id="root:1", text="What was revenue?")
    memory = initialize_evidence_memory(
        reading_session_id="reading:1",
        root_question_id=root.question_id,
        root_question_text=root.text,
        questions=[],
    )
    observation = StoredObservation(
        observation_id="obs:1",
        action_id="action:1",
        text="Revenue was 12 million.",
        sources=[ObservationSourceRef(input_id="I1")],
    )
    checker_input = EvidenceCheckInput(
        action_id="action:1",
        root_question=root,
        evidence_memory=memory,
        observations=[observation],
    )
    checker_transport = FakeTransport(
        {
            "message": {
                "content": (
                    '{"action_id":"action:1","observation_assessments":['
                    '{"observation_id":"obs:1",'
                    '"assessment":"Direct support."}],"evidence_updates":{'
                    '"add":[{"statement":"Revenue was 12 million.",'
                    '"observation_ids":["obs:1"],'
                    '"supports_question_ids":["root:1"]}],"replace":[],"remove":[]},'
                    '"current_target_status":"satisfied","root_status":"ready",'
                    '"remaining_gap_description":null}'
                )
            }
        }
    )
    checker = OllamaEvidenceCheckerBackend(
        OllamaStructuredClient(
            OllamaModelConfig(model="text-test"), checker_transport
        )
    )
    result = checker.check(checker_input)
    assert result.root_status.value == "ready"
    assert result.observation_assessments[0].used_for_evidence is True
    assert checker_transport.calls[0][1]["format"]["title"] == "EvidenceCheckDecision"

    answer_input = AnswerInput(
        reading_session_id="reading:1",
        root_question=root,
        evidence=[
            AnswerEvidence(
                evidence_id="E1",
                statement="Revenue was 12 million.",
                supports_question_ids=["root:1"],
            )
        ],
    )
    answer_transport = FakeTransport(
        {
            "message": {
                "content": (
                    '{"answer":"Revenue was 12 million.",'
                    '"used_evidence_ids":["E1"]}'
                )
            }
        }
    )
    answerer = OllamaAnswererBackend(
        OllamaStructuredClient(
            OllamaModelConfig(model="text-test"), answer_transport
        )
    )
    answer = answerer.answer(answer_input)
    assert answer.used_evidence_ids == ["E1"]
    assert answer_transport.calls[0][1]["format"]["title"] == "AnswerResult"


def test_structured_client_accepts_validated_output_from_thinking_channel() -> None:
    transport = FakeTransport(
        {
            "model": "vision-test",
            "message": {
                "content": "",
                "thinking": (
                    '{"answer":"The value is 42.",'
                    '"used_evidence_ids":["E1"]}'
                ),
            },
        }
    )
    client = OllamaStructuredClient(
        OllamaModelConfig(model="vision-test"), transport
    )

    answer = client.generate(
        component="answerer",
        system_prompt="Return JSON.",
        user_prompt="Answer from E1.",
        output_model=AnswerResult,
    )

    assert answer.answer == "The value is 42."
    assert client.call_records[0].metadata["response_channel"] == "thinking"
