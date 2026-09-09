from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from softdoc.answering import AnswerEvidence, AnswerInput, AnswerResult
from softdoc.coverage_reasoning import (
    CoverageBatchCheckInput,
    CoverageInventoryItem,
    CoverageItemVerdict,
    CoverageObservation,
    CoverageOperator,
    CoverageRequirement,
    CoverageSourceType,
)
from softdoc.model_backends import (
    OllamaAnswererBackend,
    OllamaEvidenceCheckerBackend,
    OllamaModelConfig,
    OllamaStructuredClient,
    OllamaVisualRetrievalBackend,
    OllamaVisualReaderBackend,
)
from softdoc.models import Document, Element, ElementType, Page, Provenance
from softdoc.openai_compatible import (
    OpenAICompatibleConfig,
    OpenAICompatibleStructuredClient,
)
from softdoc.reading_environment import ReaderContext
from softdoc.reading_state import (
    CurrentTarget,
    EvidenceItem,
    EvidenceCheckInput,
    ObservationSourceRef,
    ReadInput,
    ReaderKind,
    ReadingSourceType,
    ReadRepresentation,
    RootQuestion,
    QuestionState,
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


class SequenceFakeTransport:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any], float]] = []

    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        self.calls.append((url, payload, timeout_seconds))
        if not self.responses:
            raise AssertionError("SequenceFakeTransport ran out of responses")
        return self.responses.pop(0)


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
    assert '"physical_page_number": 1' in payload["messages"][1]["content"]
    assert '"document_page_count": 1' in payload["messages"][1]["content"]
    assert '"is_last_page": true' in payload["messages"][1]["content"]
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
                    '"current_target_status":"satisfied",'
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


def test_openai_answerer_repairs_missing_evidence_ids_once() -> None:
    root = RootQuestion(question_id="root:answer-repair", text="What was revenue?")
    answer_input = AnswerInput(
        reading_session_id="reading:answer-repair",
        root_question=root,
        evidence=[
            AnswerEvidence(
                evidence_id="E1",
                statement="Revenue was 12 million.",
                supports_question_ids=[root.question_id],
            )
        ],
    )
    transport = SequenceFakeTransport(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"answer":"12 million","used_evidence_ids":[]}'
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"answer":"12 million",'
                                '"used_evidence_ids":["E1"]}'
                            )
                        }
                    }
                ]
            },
        ]
    )
    answerer = OllamaAnswererBackend(
        OpenAICompatibleStructuredClient(
            OpenAICompatibleConfig(model="text-test"), transport
        )
    )

    answer = answerer.answer(answer_input)

    assert answer.used_evidence_ids == ["E1"]
    assert len(transport.calls) == 2
    repair_prompt = transport.calls[1][1]["messages"][1]["content"]
    assert "same Answerer invocation" in repair_prompt
    assert '"E1"' in repair_prompt
    assert len(answerer.last_rejected_attempts) == 1


def test_checker_repairs_duplicate_evidence_replacement_without_new_read() -> None:
    root = RootQuestion(question_id="root:Q771", text="Which figures use line plots?")
    memory = initialize_evidence_memory(
        reading_session_id="reading:Q771",
        root_question_id=root.question_id,
        root_question_text=root.text,
        questions=[],
    ).model_copy(
        update={
            "evidence": [
                EvidenceItem(
                    evidence_id="evidence:old:0000",
                    statement="Figure 1 is a line plot.",
                    observation_ids=["obs:old:00"],
                    supports_question_ids=[root.question_id],
                )
            ]
        }
    )
    checker_input = EvidenceCheckInput(
        action_id="action:Q771:2",
        root_question=root,
        evidence_memory=memory,
        observations=[
            StoredObservation(
                observation_id="obs:new:00",
                action_id="action:Q771:2",
                text="The newly read figure is a bar chart, not a line plot.",
                sources=[ObservationSourceRef(input_id="I1")],
            )
        ],
    )
    duplicate_replace = (
        '{"action_id":"action:Q771:2","observation_assessments":['
        '{"observation_id":"obs:new:00","assessment":"Contradicts the old claim."}],'
        '"evidence_updates":{"add":[],"replace":['
        '{"evidence_id":"evidence:old:0000","statement":"The new figure is a bar chart.",'
        '"observation_ids":["obs:new:00"],"supports_question_ids":["root:Q771"]},'
        '{"evidence_id":"evidence:old:0000","statement":"The old line-plot claim is invalid.",'
        '"observation_ids":["obs:new:00"],"supports_question_ids":["root:Q771"]}],'
        '"remove":[]},"current_target_status":"incomplete",'
        '"remaining_gap_description":"Other figures remain unchecked."}'
    )
    repaired = (
        '{"action_id":"action:Q771:2","observation_assessments":['
        '{"observation_id":"obs:new:00","assessment":"The new source is a bar chart."}],'
        '"evidence_updates":{"add":[],"replace":['
        '{"evidence_id":"evidence:old:0000","statement":"The newly read figure is a bar chart.",'
        '"observation_ids":["obs:new:00"],"supports_question_ids":["root:Q771"]}],'
        '"remove":[]},"current_target_status":"incomplete",'
        '"remaining_gap_description":"Other figures remain unchecked."}'
    )
    transport = SequenceFakeTransport(
        [
            {"model": "text-test", "message": {"content": duplicate_replace}},
            {"model": "text-test", "message": {"content": repaired}},
        ]
    )
    backend = OllamaEvidenceCheckerBackend(
        OllamaStructuredClient(OllamaModelConfig(model="text-test"), transport)
    )

    result = backend.check(checker_input)

    assert len(transport.calls) == 2
    assert len(result.evidence_updates.replace) == 1
    assert len(backend.last_rejected_attempts) == 1
    repair_prompt = transport.calls[1][1]["messages"][1]["content"]
    assert "evidence:old:0000" in repair_prompt
    assert "at most once across replace and remove" in repair_prompt
    assert "do not guess, edit, or fuzzy-match" in repair_prompt


def test_checker_repairs_mistyped_observation_id_without_fuzzy_mapping() -> None:
    root = RootQuestion(question_id="root:Q165", text="What color is Beijing?")
    memory = initialize_evidence_memory(
        reading_session_id="reading:Q165",
        root_question_id=root.question_id,
        root_question_text=root.text,
        questions=[],
    )
    checker_input = EvidenceCheckInput(
        action_id="action:Q165:1",
        root_question=root,
        evidence_memory=memory,
        observations=[
            StoredObservation(
                observation_id="obs:6883d66e1cce:00",
                action_id="action:Q165:1",
                text="The supplied map does not contain Beijing.",
                sources=[ObservationSourceRef(input_id="I1")],
            )
        ],
    )
    mistyped = (
        '{"action_id":"action:Q165:1","observation_assessments":['
        '{"observation_id":"obs:6883d666e1cce:00","assessment":"No Beijing is shown."}],'
        '"evidence_updates":{"add":[{"statement":"The map does not contain Beijing.",'
        '"observation_ids":["obs:6883d666e1cce:00"],'
        '"supports_question_ids":["root:Q165"]}],"replace":[],"remove":[]},'
        '"current_target_status":"satisfied","remaining_gap_description":null}'
    )
    repaired = mistyped.replace("obs:6883d666e1cce:00", "obs:6883d66e1cce:00")
    transport = SequenceFakeTransport(
        [
            {"model": "text-test", "message": {"content": mistyped}},
            {"model": "text-test", "message": {"content": repaired}},
        ]
    )
    backend = OllamaEvidenceCheckerBackend(
        OllamaStructuredClient(OllamaModelConfig(model="text-test"), transport)
    )

    result = backend.check(checker_input)

    assert result.root_status.value == "ready"
    assert result.observation_assessments[0].observation_id == "obs:6883d66e1cce:00"
    assert len(transport.calls) == 2
    assert len(backend.last_rejected_attempts) == 1
    repair_prompt = transport.calls[1][1]["messages"][1]["content"]
    assert '"obs:6883d66e1cce:00"' in repair_prompt
    assert "obs:6883d666e1cce:00" in repair_prompt
    assert "fuzzy-match" in repair_prompt


def test_checker_contract_repair_works_with_openai_compatible_client() -> None:
    root = RootQuestion(question_id="root:vllm", text="What is the value?")
    memory = initialize_evidence_memory(
        reading_session_id="reading:vllm",
        root_question_id=root.question_id,
        root_question_text=root.text,
        questions=[],
    )
    checker_input = EvidenceCheckInput(
        action_id="action:vllm:1",
        root_question=root,
        evidence_memory=memory,
        observations=[
            StoredObservation(
                observation_id="obs:vllm:00",
                action_id="action:vllm:1",
                text="The value is 42.",
                sources=[ObservationSourceRef(input_id="I1")],
            )
        ],
    )
    bad = (
        '{"action_id":"action:vllm:1","observation_assessments":['
        '{"observation_id":"obs:vllm:000","assessment":"Direct support."}],'
        '"evidence_updates":{"add":[],"replace":[],"remove":[]},'
        '"current_target_status":"incomplete",'
        '"remaining_gap_description":"The value is still needed."}'
    )
    repaired = bad.replace("obs:vllm:000", "obs:vllm:00")
    transport = SequenceFakeTransport(
        [
            {"choices": [{"message": {"content": bad}}]},
            {"choices": [{"message": {"content": repaired}}]},
        ]
    )
    backend = OllamaEvidenceCheckerBackend(
        OpenAICompatibleStructuredClient(
            OpenAICompatibleConfig(model="text-test"), transport
        )
    )

    result = backend.check(checker_input)

    assert result.current_target_status.value == "incomplete"
    assert len(transport.calls) == 2
    assert transport.calls[1][1]["response_format"]["type"] == "json_schema"
    assert len(backend.last_rejected_attempts) == 1


def test_checker_repairs_satisfied_target_recheck_missing_reused_evidence() -> None:
    root = RootQuestion(
        question_id="root:Q366",
        text="What was the absolute shortfall from the forecast?",
    )
    memory = initialize_evidence_memory(
        reading_session_id="reading:Q366",
        root_question_id=root.question_id,
        root_question_text=root.text,
        questions=[
            QuestionState(
                question_id="Q1",
                text="What was the actual growth rate?",
            ),
            QuestionState(
                question_id="Q2",
                text="What was the forecast growth rate?",
            ),
        ],
    ).model_copy(
        update={
            "questions": [
                QuestionState(
                    question_id="Q1",
                    text="What was the actual growth rate?",
                    status="satisfied",
                ),
                QuestionState(
                    question_id="Q2",
                    text="What was the forecast growth rate?",
                ),
            ],
            "evidence": [
                EvidenceItem(
                    evidence_id="evidence:actual",
                    statement="Actual GDP growth was 4.3%.",
                    observation_ids=["obs:gdp"],
                    supports_question_ids=["Q1"],
                ),
                EvidenceItem(
                    evidence_id="evidence:forecast",
                    statement="Forecast GDP growth was 6.7%.",
                    observation_ids=["obs:gdp"],
                    supports_question_ids=["Q1"],
                ),
            ],
            "current_target": CurrentTarget(
                question_id="Q2",
                gap_description="What was the forecast growth rate?",
            ),
        }
    )
    checker_input = EvidenceCheckInput(
        action_id="action:Q366:target-recheck:Q2",
        root_question=root,
        evidence_memory=memory,
        observations=[],
        limitations=[],
    )
    missing_reuse = (
        '{"action_id":"action:Q366:target-recheck:Q2",'
        '"observation_assessments":[],"evidence_updates":'
        '{"add":[],"replace":[],"remove":[]},"reused_evidence_ids":[],'
        '"current_target_status":"satisfied",'
        '"remaining_gap_description":null}'
    )
    repaired = missing_reuse.replace(
        '"reused_evidence_ids":[]',
        '"reused_evidence_ids":["evidence:forecast"]',
    )
    transport = SequenceFakeTransport(
        [
            {"choices": [{"message": {"content": missing_reuse}}]},
            {"choices": [{"message": {"content": repaired}}]},
        ]
    )
    backend = OllamaEvidenceCheckerBackend(
        OpenAICompatibleStructuredClient(
            OpenAICompatibleConfig(model="text-test"), transport
        )
    )

    result = backend.check(checker_input)

    assert result.reused_evidence_ids == ["evidence:forecast"]
    assert len(transport.calls) == 2
    assert len(backend.last_rejected_attempts) == 1
    repair_prompt = transport.calls[1][1]["messages"][1]["content"]
    assert "evidence:forecast" in repair_prompt
    assert "Forecast GDP growth was 6.7%." in repair_prompt
    assert "If existing Evidence fully satisfies current_target" in repair_prompt


def test_coverage_checker_backend_uses_frozen_schema_without_claiming_completion(
) -> None:
    checker_input = CoverageBatchCheckInput(
        action_id="action:coverage:1",
        question_id="Q913",
        question_text="How many people appear in the figures on Pages 18-19?",
        requirement=CoverageRequirement(
            operator=CoverageOperator.COUNT,
            scope_text="Pages 18-19",
            item_type="person",
            source_type=CoverageSourceType.FIGURE,
        ),
        inventory_items=[
            CoverageInventoryItem(
                inventory_id="figure:1",
                source_ids=["figure:1"],
                page_ids=["page:18"],
                physical_page_numbers=[18],
                source_type=CoverageSourceType.FIGURE,
                element_type=ElementType.FIGURE,
                deduplication_reason="canonical_element",
            )
        ],
        observations=[
            CoverageObservation(
                observation_id="obs:coverage:1",
                text="Two distinct people are visible in the figure.",
                source_ids=["figure:1"],
                page_ids=["page:18"],
            )
        ],
    )
    transport = FakeTransport(
        {
            "model": "vision-test",
            "message": {
                "content": (
                    '{"action_id":"action:coverage:1","assessments":['
                    '{"inventory_id":"figure:1","verdict":"matched",'
                    '"matched_count":2,"matched_values":[],'
                    '"observation_ids":["obs:coverage:1"],'
                    '"rationale":"Two distinct people are visible."}]}'
                )
            },
        }
    )
    backend = OllamaEvidenceCheckerBackend(
        OllamaStructuredClient(OllamaModelConfig(model="vision-test"), transport)
    )

    result = backend.check_coverage(checker_input)

    assert result.assessments[0].verdict == CoverageItemVerdict.MATCHED
    payload = transport.calls[0][1]
    assert payload["format"]["title"] == "CoverageBatchCheckResult"
    assert "Environment, not you" in payload["messages"][0]["content"]
    assert backend.client.call_records[0].component == "coverage_checker"


def test_coverage_checker_repairs_missing_grounded_observation_ids() -> None:
    checker_input = CoverageBatchCheckInput(
        action_id="action:coverage:repair",
        question_id="Q4",
        question_text="How many charts match the criterion?",
        requirement=CoverageRequirement(
            operator=CoverageOperator.COUNT,
            scope_text="this report",
            item_type="chart",
            source_type=CoverageSourceType.CHART,
            predicate="compares two groups",
        ),
        inventory_items=[
            CoverageInventoryItem(
                inventory_id="chart:1",
                source_ids=["chart:1"],
                page_ids=["page:1"],
                physical_page_numbers=[1],
                source_type=CoverageSourceType.CHART,
                element_type=ElementType.CHART,
                deduplication_reason="canonical_element",
            )
        ],
        observations=[
            CoverageObservation(
                observation_id="obs:chart:1",
                text="The chart compares Group A and Group B.",
                source_ids=["chart:1"],
                page_ids=["page:1"],
            )
        ],
    )
    rejected = (
        '{"action_id":"action:coverage:repair","assessments":['
        '{"inventory_id":"chart:1","verdict":"matched",'
        '"matched_count":1,"matched_values":[],"observation_ids":[],'
        '"rationale":"The chart compares the groups."}]}'
    )
    repaired = rejected.replace(
        '"observation_ids":[]',
        '"observation_ids":["obs:chart:1"]',
    )
    transport = SequenceFakeTransport(
        [
            {"model": "vision-test", "message": {"content": rejected}},
            {"model": "vision-test", "message": {"content": repaired}},
        ]
    )
    backend = OllamaEvidenceCheckerBackend(
        OllamaStructuredClient(OllamaModelConfig(model="vision-test"), transport)
    )

    result = backend.check_coverage(checker_input)

    assert result.assessments[0].observation_ids == ["obs:chart:1"]
    assert len(transport.calls) == 2
    assert "Allowed grounded Observation IDs" in transport.calls[1][1]["messages"][1]["content"]
    assert len(backend.last_rejected_attempts) == 1


def test_coverage_checker_repairs_truncated_json_within_same_invocation() -> None:
    checker_input = CoverageBatchCheckInput(
        action_id="action:coverage:truncated",
        question_id="Q12",
        question_text="How many figures match the criterion?",
        requirement=CoverageRequirement(
            operator=CoverageOperator.COUNT,
            scope_text="this report",
            item_type="figure",
            source_type=CoverageSourceType.FIGURE,
            predicate="contains a warning",
        ),
        inventory_items=[
            CoverageInventoryItem(
                inventory_id="figure:1",
                source_ids=["figure:1"],
                page_ids=["page:1"],
                physical_page_numbers=[1],
                source_type=CoverageSourceType.FIGURE,
                element_type=ElementType.FIGURE,
                deduplication_reason="canonical_element",
            )
        ],
        observations=[
            CoverageObservation(
                observation_id="obs:figure:1",
                text="The figure contains a warning.",
                source_ids=["figure:1"],
                page_ids=["page:1"],
            )
        ],
    )
    truncated = '{"action_id":"action:coverage:truncated","assessments":['
    repaired = (
        '{"action_id":"action:coverage:truncated","assessments":['
        '{"inventory_id":"figure:1","verdict":"matched",'
        '"matched_count":1,"matched_values":["figure:1"],'
        '"observation_ids":["obs:figure:1"],'
        '"rationale":"The warning is visible."}]}'
    )
    transport = SequenceFakeTransport(
        [
            {"model": "vision-test", "message": {"content": truncated}},
            {"model": "vision-test", "message": {"content": repaired}},
        ]
    )
    backend = OllamaEvidenceCheckerBackend(
        OllamaStructuredClient(OllamaModelConfig(model="vision-test"), transport)
    )

    result = backend.check_coverage(checker_input)

    assert result.assessments[0].inventory_id == "figure:1"
    assert len(transport.calls) == 2
    assert len(backend.last_rejected_attempts) == 1
    assert backend.last_rejected_attempts[0]["raw_content"] == truncated
    assert "return one complete strict JSON object again" in (
        transport.calls[1][1]["messages"][1]["content"]
    )


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
