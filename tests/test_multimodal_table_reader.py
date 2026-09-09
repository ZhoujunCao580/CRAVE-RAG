from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
import pytest
from pydantic import ValidationError

from softdoc.controller import ControllerCandidatePreview
from softdoc.model_backends import (
    ModelBackedReader,
    MultimodalTableReaderBackend,
    OllamaVisualReaderBackend,
)
from softdoc.models import (
    ContentAvailability,
    Document,
    Element,
    ElementParseStatus,
    ElementType,
    Page,
    Provenance,
)
from softdoc.reading_environment import ReaderContext, ReadingEnvironment
from softdoc.reading_environment import DocumentSearchService
from softdoc.reading_state import (
    ObservationLimitation,
    ReadInput,
    ReadRecord,
    ReaderKind,
    ReadingSourceType,
    ReadRepresentation,
)
from softdoc.table_reading import TableReadInput
from softdoc.table_reading import (
    TableReadRequest,
    TableReadResult,
    TableReadCell,
    select_relevant_row_hints,
    validate_table_read_result,
)
from softdoc.table_fragments import FRAGMENT_METADATA_KEY
from softdoc.table_view import TableMaterializer
from softdoc.retrieval import (
    CandidateMergePolicy,
    RetrievalSource,
    SearchSessionConfig,
    SnippetSource,
    SubQuestionInput,
    VisualElementCandidate,
    VisualSearchResult,
)


class FakeStructuredClient:
    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response = response

    def generate(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return kwargs["output_model"].model_validate(
            self.response
            or {
                "observations": [
                    {
                        "text": "Revenue was 42 thousand rupees.",
                        "sources": [{"input_id": "I1", "cell_id": None}],
                    }
                ],
                "limitations": [],
            }
        )


class FakeVisualSearchBackend:
    def __init__(self, table: Element) -> None:
        self.table = table

    def search(self, subquestion: SubQuestionInput) -> VisualSearchResult:
        return VisualSearchResult(
            subquestion_id=subquestion.subquestion_id,
            document_id=self.table.document_id,
            index_fingerprint="fixture-index",
            model_name="fixture-visual-dense",
            total_visual_assets=1,
            total_candidates=1,
            candidates=[
                VisualElementCandidate(
                    element_id=self.table.element_id,
                    visual_asset_id="visual:table-1",
                    visual_score=0.9,
                    visual_rank=1,
                    page_id=self.table.page_id,
                    page_number=self.table.page_number,
                    preview_text="A financial table containing revenue values.",
                    element_type=ElementType.TABLE,
                    content_availability=self.table.content_availability,
                )
            ],
        )


def _provenance(owner: str) -> Provenance:
    return Provenance(
        provenance_id=f"prov:{owner}",
        adapter="test",
        source_path=Path("fixture.json"),
        source_locator=owner,
    )


def _table_document(
    tmp_path: Path,
    *,
    html: str | None,
    visual: bool,
    text: str | None = None,
    degraded: bool = False,
) -> tuple[Document, Element, Path | None]:
    doc_id = "doc:table-reader"
    image_path = Path("table.png") if visual else None
    if image_path is not None:
        Image.new("RGB", (160, 80), "white").save(tmp_path / image_path)
    table = Element(
        element_id="table:1",
        document_id=doc_id,
        page_id="page:1",
        page_number=1,
        element_type=ElementType.TABLE,
        reading_order=0,
        html=html,
        text=text,
        image_path=image_path,
        parse_status=(
            ElementParseStatus.DEGRADED if degraded else ElementParseStatus.PARSED
        ),
        provenance=_provenance("table:1"),
    )
    page = Page(
        page_id="page:1",
        document_id=doc_id,
        page_index=0,
        page_number=1,
        width=200,
        height=300,
        element_ids=[table.element_id],
        reading_order=[table.element_id],
        provenance=_provenance("page:1"),
    )
    return (
        Document(
            document_id=doc_id,
            source_path=Path("fixture.pdf"),
            pages=[page],
            elements=[table],
            provenance=_provenance(doc_id),
        ),
        table,
        (tmp_path / image_path) if image_path is not None else None,
    )


def _context(
    tmp_path: Path,
    *,
    html: str | None,
    visual: bool,
    text: str | None = None,
) -> tuple[ReaderContext, FakeStructuredClient]:
    document, table, image_path = _table_document(
        tmp_path, html=html, visual=visual, text=text
    )
    table_views = {}
    if html is not None:
        view = TableMaterializer().materialize(
            table, document_root=tmp_path
        ).view
        table_views[view.table_view_id] = view
        read_input = ReadInput(
            input_id="I1",
            source_id=view.table_view_id,
            source_type=ReadingSourceType.TABLE_VIEW,
            representation=ReadRepresentation.TABLE_VIEW,
            document_id=document.document_id,
            page_id=table.page_id,
            element_id=table.element_id,
            table_view_id=view.table_view_id,
            visual_asset_id="visual:table-1" if image_path is not None else None,
            visual_asset_path=image_path,
        )
    else:
        assert image_path is not None
        read_input = ReadInput(
            input_id="I1",
            source_id=table.element_id,
            source_type=ReadingSourceType.ELEMENT,
            representation=ReadRepresentation.ELEMENT_VISUAL,
            document_id=document.document_id,
            page_id=table.page_id,
            element_id=table.element_id,
            visual_asset_id="visual:table-1",
            visual_asset_path=image_path,
        )
    context = ReaderContext(
        action_id="action:1",
        question_id="Q1",
        local_problem="What was revenue and what unit applies?",
        document=document,
        inputs=(read_input,),
        elements_by_id={table.element_id: table},
        pages_by_id={document.pages[0].page_id: document.pages[0]},
        table_views_by_id=table_views,
    )
    return context, FakeStructuredClient()


def _reader(client: FakeStructuredClient) -> ModelBackedReader:
    return ModelBackedReader(
        OllamaVisualReaderBackend(client),
        table_reader=MultimodalTableReaderBackend(client),
    )


def test_table_reader_schema_bounds_compact_observations() -> None:
    schema = TableReadResult.model_json_schema()

    assert schema["properties"]["observations"]["maxItems"] == 4
    with pytest.raises(ValidationError, match="at most 4 items"):
        TableReadResult.model_validate(
            {
                "observations": [
                    {
                        "text": f"fact {index}",
                        "sources": [{"input_id": "I1", "cell_id": None}],
                    }
                    for index in range(5)
                ],
                "limitations": [],
            }
        )


def test_html_and_image_table_uses_multimodal_reader(tmp_path: Path) -> None:
    context, client = _context(
        tmp_path,
        html="<table><tr><th>Revenue</th><th>2024</th></tr>"
        "<tr><td>Net sales</td><td>42</td></tr></table>",
        visual=True,
    )

    output = _reader(client).read(context)

    assert output.reader_kind == ReaderKind.TABLE
    assert output.observations[0].text == "Revenue was 42 thousand rupees."
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["component"] == "multimodal_table_reader"
    assert len(call["image_paths"]) == 1
    assert '"structured_cells"' in call["user_prompt"]
    assert '"text": "Revenue"' in call["user_prompt"]
    assert (
        '"problem": "What was revenue and what unit applies?"'
        in call["user_prompt"]
    )
    assert "table.png" not in call["user_prompt"]


def test_table_reader_uses_controller_local_problem_without_rewriting(
    tmp_path: Path,
) -> None:
    context, client = _context(
        tmp_path,
        html="<table><tr><td>Revenue</td><td>42</td></tr></table>",
        visual=False,
    )
    local_problem = (
        "Read only the 2024 net-sales value and its explicitly reported unit."
    )
    context = ReaderContext(
        action_id=context.action_id,
        question_id=context.question_id,
        local_problem=local_problem,
        document=context.document,
        inputs=context.inputs,
        elements_by_id=context.elements_by_id,
        pages_by_id=context.pages_by_id,
        table_views_by_id=context.table_views_by_id,
    )

    _reader(client).read(context)

    assert f'"problem": "{local_problem}"' in client.calls[0]["user_prompt"]


def test_html_only_table_uses_question_directed_table_reader(
    tmp_path: Path,
) -> None:
    context, client = _context(
        tmp_path,
        html="<table><tr><td>Revenue</td><td>42</td></tr></table>",
        visual=False,
    )

    output = _reader(client).read(context)

    assert output.reader_kind == ReaderKind.TABLE
    assert output.observations[0].text == "Revenue was 42 thousand rupees."
    assert len(client.calls) == 1
    assert client.calls[0]["image_paths"] == []
    assert '"structured_cells"' in client.calls[0]["user_prompt"]


def test_image_only_table_uses_table_reader_without_fake_html(
    tmp_path: Path,
) -> None:
    context, client = _context(tmp_path, html=None, visual=True)

    output = _reader(client).read(context)

    assert output.reader_kind == ReaderKind.TABLE
    assert len(client.calls) == 1
    prompt = client.calls[0]["user_prompt"]
    assert '"structured_cells": []' in prompt
    assert '"extracted_text": null' in prompt
    assert '"visual_asset_id": "visual:table-1"' in prompt


def test_extracted_text_and_image_table_keeps_both_inputs(tmp_path: Path) -> None:
    context, client = _context(
        tmp_path,
        html=None,
        text="Revenue 2024 42",
        visual=True,
    )

    output = _reader(client).read(context)

    assert output.reader_kind == ReaderKind.TABLE
    assert len(client.calls) == 1
    prompt = client.calls[0]["user_prompt"]
    assert '"extracted_text": "Revenue 2024 42"' in prompt
    assert len(client.calls[0]["image_paths"]) == 1


def test_text_only_table_uses_question_directed_table_reader(tmp_path: Path) -> None:
    document, table, _ = _table_document(
        tmp_path,
        html=None,
        text="Revenue 2024 42",
        visual=False,
    )
    read_input = ReadInput(
        input_id="I1",
        source_id=table.element_id,
        source_type=ReadingSourceType.ELEMENT,
        representation=ReadRepresentation.ELEMENT_TEXT,
        document_id=document.document_id,
        page_id=table.page_id,
        element_id=table.element_id,
    )
    context = ReaderContext(
        action_id="action:1",
        question_id="Q1",
        local_problem="What was revenue?",
        document=document,
        inputs=(read_input,),
        elements_by_id={table.element_id: table},
        pages_by_id={document.pages[0].page_id: document.pages[0]},
        table_views_by_id={},
    )
    client = FakeStructuredClient()

    output = _reader(client).read(context)

    assert output.reader_kind == ReaderKind.TABLE
    assert output.observations[0].text == "Revenue was 42 thousand rupees."
    assert len(client.calls) == 1
    assert '"extracted_text": "Revenue 2024 42"' in client.calls[0]["user_prompt"]


def test_unreadable_table_is_rejected_before_reader_dispatch(tmp_path: Path) -> None:
    document, table, _ = _table_document(
        tmp_path,
        html=None,
        visual=False,
        degraded=True,
    )
    environment = ReadingEnvironment(
        document,
        asset_root=tmp_path,
        controller=object(),  # type: ignore[arg-type]
        reader=object(),  # type: ignore[arg-type]
        checker=object(),  # type: ignore[arg-type]
        answerer=object(),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="no readable representation"):
        environment._read_input(table.element_id, 0)

    with pytest.raises(ValueError, match="structured cells, extracted text"):
        TableReadInput(
            input_id="I1",
            element_id=table.element_id,
            page_id=table.page_id,
            physical_page_number=1,
            document_page_count=1,
            is_last_page=True,
        )


def test_environment_preserves_both_table_representations(tmp_path: Path) -> None:
    document, table, image_path = _table_document(
        tmp_path,
        html="<table><tr><td>Revenue</td><td>42</td></tr></table>",
        visual=True,
    )
    environment = ReadingEnvironment(
        document,
        asset_root=tmp_path,
        controller=object(),  # type: ignore[arg-type]
        reader=object(),  # type: ignore[arg-type]
        checker=object(),  # type: ignore[arg-type]
        answerer=object(),  # type: ignore[arg-type]
    )

    read_input = environment._read_input(table.element_id, 0)

    assert read_input.representation == ReadRepresentation.TABLE_VIEW
    assert read_input.table_view_id is not None
    assert read_input.visual_asset_id is not None
    assert read_input.visual_asset_path == image_path


def test_controller_candidate_contract_does_not_expose_retrieval_routes() -> None:
    preview = ControllerCandidatePreview(
        element_id="table:1",
        element_type=ElementType.TABLE,
        page_id="page:1",
        matched_snippet="Columns: Metric | 2024\nNet sales | 42",
        content_availability=ContentAvailability.MIXED,
    )

    payload = preview.model_dump(mode="json")

    assert "matched_by" not in payload
    assert payload["matched_snippet"].startswith("Columns:")


def test_text_and_visual_table_retrieval_dedupes_then_reads_both_representations(
    tmp_path: Path,
) -> None:
    document, table, image_path = _table_document(
        tmp_path,
        html="<table><tr><th>Metric</th><th>2024</th></tr>"
        "<tr><td>Revenue</td><td>42</td></tr></table>",
        visual=True,
    )
    service = DocumentSearchService(
        document,
        visual_backend=FakeVisualSearchBackend(table),
        config=SearchSessionConfig(
            merge_policy=CandidateMergePolicy.FIXED_TEXT_VISUAL_QUOTA,
            batch_size=5,
            text_quota=3,
            visual_quota=2,
        ),
    )

    session, batch = service.start(question_id="Q1", query="Revenue 2024")

    assert len(session.candidate_catalog) == 1
    assert session.candidate_catalog[0].matched_by == [
        RetrievalSource.BM25,
        RetrievalSource.VISUAL_DENSE,
    ]
    assert len(batch.candidate_previews) == 1
    candidate = batch.candidate_previews[0]
    assert candidate.snippet_source == SnippetSource.TABLE_PREVIEW
    assert "Columns: Metric | 2024" in candidate.matched_snippet
    assert "Matched row: Metric=Revenue | 2024=42" in candidate.matched_snippet
    assert "financial table" not in candidate.matched_snippet

    environment = ReadingEnvironment(
        document,
        asset_root=tmp_path,
        controller=object(),  # type: ignore[arg-type]
        reader=object(),  # type: ignore[arg-type]
        checker=object(),  # type: ignore[arg-type]
        answerer=object(),  # type: ignore[arg-type]
        search_service=service,
    )
    read_input = environment._read_input(candidate.element_id, 0)
    client = FakeStructuredClient()
    output = _reader(client).read(
        ReaderContext(
            action_id="action:1",
            question_id="Q1",
            local_problem="What was revenue in 2024 and what unit applies?",
            document=document,
            inputs=(read_input,),
            elements_by_id={table.element_id: table},
            pages_by_id={document.pages[0].page_id: document.pages[0]},
            table_views_by_id=environment._table_views,
        )
    )

    assert output.reader_kind == ReaderKind.TABLE
    assert output.observations[0].text == "Revenue was 42 thousand rupees."
    assert client.calls[0]["image_paths"] == [image_path]


def _cross_page_context(tmp_path: Path) -> ReaderContext:
    doc_id = "doc:cross-page-table"
    group_id = "table-group:1"
    tables = [
        Element(
            element_id="table:header",
            document_id=doc_id,
            page_id="page:1",
            page_number=1,
            element_type=ElementType.TABLE,
            reading_order=0,
            html=(
                "<table><tr><th>Company</th><th>2022</th><th>2023</th></tr>"
                "<tr><td>A Company</td><td>10</td><td>12</td></tr></table>"
            ),
            metadata={
                FRAGMENT_METADATA_KEY: {
                    "status": "confirmed",
                    "group_id": group_id,
                    "fragment_index": 0,
                    "fragment_count": 2,
                }
            },
            provenance=_provenance("table:header"),
        ),
        Element(
            element_id="table:data",
            document_id=doc_id,
            page_id="page:2",
            page_number=2,
            element_type=ElementType.TABLE,
            reading_order=0,
            html=(
                "<table><tr><td>B Company</td><td>15</td><td>18</td>"
                "</tr></table>"
            ),
            metadata={
                FRAGMENT_METADATA_KEY: {
                    "status": "confirmed",
                    "group_id": group_id,
                    "fragment_index": 1,
                    "fragment_count": 2,
                }
            },
            provenance=_provenance("table:data"),
        ),
    ]
    pages = [
        Page(
            page_id=f"page:{index}",
            document_id=doc_id,
            page_index=index - 1,
            page_number=index,
            width=200,
            height=300,
            element_ids=[tables[index - 1].element_id],
            reading_order=[tables[index - 1].element_id],
            provenance=_provenance(f"page:{index}"),
        )
        for index in (1, 2)
    ]
    document = Document(
        document_id=doc_id,
        source_path=Path("fixture.pdf"),
        pages=pages,
        elements=tables,
        provenance=_provenance(doc_id),
    )
    views = {
        table.element_id: TableMaterializer().materialize(
            table, document_root=tmp_path
        ).view
        for table in tables
    }
    inputs = tuple(
        ReadInput(
            input_id=f"I{index}",
            source_id=views[table.element_id].table_view_id,
            source_type=ReadingSourceType.TABLE_VIEW,
            representation=ReadRepresentation.TABLE_VIEW,
            document_id=doc_id,
            page_id=table.page_id,
            element_id=table.element_id,
            table_view_id=views[table.element_id].table_view_id,
        )
        for index, table in enumerate(reversed(tables), start=1)
    )
    return ReaderContext(
        action_id="action:cross-page",
        question_id="Q1",
        local_problem="What were B Company's values in 2022 and 2023?",
        document=document,
        inputs=inputs,
        elements_by_id={table.element_id: table for table in tables},
        pages_by_id={page.page_id: page for page in pages},
        table_views_by_id={view.table_view_id: view for view in views.values()},
    )


def test_confirmed_continuation_gets_inherited_headers_and_relevant_row_hint(
    tmp_path: Path,
) -> None:
    context = _cross_page_context(tmp_path)
    context = ReaderContext(
        action_id=context.action_id,
        question_id=context.question_id,
        local_problem=context.local_problem,
        document=context.document,
        inputs=(context.inputs[0],),
        elements_by_id=context.elements_by_id,
        pages_by_id=context.pages_by_id,
        table_views_by_id=context.table_views_by_id,
    )
    client = FakeStructuredClient()

    _reader(client).read(context)

    prompt = client.calls[0]["user_prompt"]
    assert '"header_status": "confirmed_inherited"' in prompt
    assert '"headers": [' in prompt
    assert '"Company"' in prompt
    assert '"2022"' in prompt
    assert '"2023"' in prompt
    assert '"B Company | 15 | 18"' in prompt


def test_missing_header_limitation_requires_question_relevant_visible_content() -> None:
    with pytest.raises(ValueError, match="relevant_visible_content"):
        TableReadResult.model_validate(
            {
                "observations": [],
                "limitations": [
                    {
                        "code": "missing_header_context",
                        "description": "B Company's values have no headers.",
                        "input_ids": ["I1"],
                        "relevant_visible_content": [],
                    }
                ],
            }
        )


def test_missing_header_limitation_reaches_controller_feedback_directly(
    tmp_path: Path,
) -> None:
    context, _ = _context(
        tmp_path,
        html="<table><tr><td>B Company</td><td>15</td><td>18</td></tr></table>",
        visual=False,
    )
    record = ReadRecord(
        action_id="action:1",
        reader_kind=ReaderKind.TABLE,
        document_id=context.document.document_id,
        subquestion_id="Q1",
        local_problem=context.local_problem,
        inputs=list(context.inputs),
        limitations=[
            ObservationLimitation(
                code="missing_header_context",
                description=(
                    "B Company's values 15 and 18 cannot be mapped without headers."
                ),
                input_ids=["I1"],
                relevant_visible_content=["B Company | 15 | 18"],
            )
        ],
    )

    feedback = ReadingEnvironment._controller_limitations(
        record.limitations, record
    )

    assert feedback[0].code == "missing_header_context"
    assert feedback[0].source_ids == [context.inputs[0].element_id]
    assert feedback[0].relevant_visible_content == ["B Company | 15 | 18"]


def test_relevant_row_hint_selects_matching_row_not_arbitrary_first_row() -> None:
    cells = [
        # The first row is deliberately unrelated and must not be preserved.
        *[
            {"cell_id": f"a{column}", "row": 0, "column": column, "text": text}
            for column, text in enumerate(["A Company", "10", "12"])
        ],
        *[
            {"cell_id": f"b{column}", "row": 1, "column": column, "text": text}
            for column, text in enumerate(["B Company", "15", "18"])
        ],
    ]
    table_cells = [TableReadCell.model_validate(cell) for cell in cells]

    hints = select_relevant_row_hints(
        table_cells, "What are the values for B Company?"
    )

    assert hints == ["B Company | 15 | 18"]


def test_confirmed_joint_read_can_create_one_observation_grounded_in_both_fragments(
    tmp_path: Path,
) -> None:
    context = _cross_page_context(tmp_path)
    client = FakeStructuredClient(
        {
            "observations": [
                {
                    "text": "B Company's 2022 and 2023 values were 15 and 18.",
                    "sources": [
                        {"input_id": "I1", "cell_id": None},
                        {"input_id": "I2", "cell_id": None},
                    ],
                }
            ],
            "limitations": [],
        }
    )

    output = _reader(client).read(context)

    assert [source.input_id for source in output.observations[0].sources] == [
        "I1",
        "I2",
    ]


def test_one_table_observation_can_reference_distinct_cells_from_same_input() -> None:
    result = TableReadResult.model_validate(
        {
            "observations": [
                {
                    "text": "Revenue was 10 in 2022 and 12 in 2023.",
                    "sources": [
                        {"input_id": "I1", "cell_id": "table:1#r1c1"},
                        {"input_id": "I1", "cell_id": "table:1#r1c2"},
                    ],
                }
            ],
            "limitations": [],
        }
    )

    assert len(result.observations[0].sources) == 2


def test_one_table_observation_rejects_exact_duplicate_source_reference() -> None:
    with pytest.raises(ValueError, match="exact duplicates"):
        TableReadResult.model_validate(
            {
                "observations": [
                    {
                        "text": "Revenue was 10 in 2022.",
                        "sources": [
                            {"input_id": "I1", "cell_id": "table:1#r1c1"},
                            {"input_id": "I1", "cell_id": "table:1#r1c1"},
                        ],
                    }
                ],
                "limitations": [],
            }
        )


def test_unrelated_table_inputs_can_return_limitation_without_false_merge() -> None:
    request = TableReadRequest(
        action_id="action:1",
        document_id="doc:1",
        source_name="fixture.pdf",
        problem="Compare the values.",
        table_inputs=[
            TableReadInput(
                input_id="I1",
                element_id="table:1",
                page_id="page:1",
                physical_page_number=1,
                document_page_count=2,
                is_last_page=False,
                extracted_text="A | 1",
            ),
            TableReadInput(
                input_id="I2",
                element_id="table:2",
                page_id="page:2",
                physical_page_number=2,
                document_page_count=2,
                is_last_page=True,
                extracted_text="B | 2",
            ),
        ],
    )
    result = TableReadResult.model_validate(
        {
            "observations": [],
            "limitations": [
                {
                    "code": "other",
                    "description": "The inputs do not establish one table.",
                    "input_ids": ["I1", "I2"],
                    "relevant_visible_content": ["A | 1", "B | 2"],
                }
            ],
        }
    )

    validated = validate_table_read_result(request, result)

    assert validated.observations == []
    assert validated.limitations[0].input_ids == ["I1", "I2"]


def test_merged_multilevel_header_is_not_flattened_into_false_reliable_headers(
    tmp_path: Path,
) -> None:
    context, client = _context(
        tmp_path,
        html=(
            "<table><tr><th rowspan='2'>Company</th><th colspan='2'>Revenue</th></tr>"
            "<tr><th>2022</th><th>2023</th></tr>"
            "<tr><td>B Company</td><td>15</td><td>18</td></tr></table>"
        ),
        visual=True,
    )

    _reader(client).read(context)

    prompt = client.calls[0]["user_prompt"]
    assert '"header_status": "unresolved"' in prompt
    assert '"rowspan": 2' in prompt
    assert '"colspan": 2' in prompt


def test_readable_representation_conflict_keeps_partial_fact_and_limitation() -> None:
    result = TableReadResult.model_validate(
        {
            "observations": [
                {
                    "text": "The table period is 2023.",
                    "sources": [{"input_id": "I1", "cell_id": None}],
                }
            ],
            "limitations": [
                {
                    "code": "representation_conflict",
                    "description": (
                        "Structured cells show 18 while the readable image shows 16."
                    ),
                    "input_ids": ["I1"],
                    "relevant_visible_content": ["B Company | 18 (cells) | 16 (image)"],
                }
            ],
        }
    )

    assert len(result.observations) == 1
    assert result.limitations[0].code.value == "representation_conflict"
