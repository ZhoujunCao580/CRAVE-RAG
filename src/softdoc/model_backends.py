"""Reusable model-backed Reader, Checker, and Answerer adapters.

The reading loop depends on small protocols rather than a particular serving
stack.  This module supplies Ollama implementations for local development while
keeping the transport injectable for offline unit tests and future server
adapters.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Protocol, TypeVar
from urllib import error, request

from pydantic import BaseModel, Field, ValidationError, field_validator

from softdoc.answering import (
    ANSWERER_SYSTEM_PROMPT,
    AnswerInput,
    AnswerResult,
    answerer_user_prompt,
)
from softdoc.checking_prompt import CHECKER_SYSTEM_PROMPT
from softdoc.coverage_prompt import COVERAGE_CHECKER_SYSTEM_PROMPT
from softdoc.coverage_reasoning import (
    CoverageBatchCheckInput,
    CoverageBatchCheckResult,
    validate_coverage_batch_result,
)
from softdoc.models import ElementType, SoftDocModel
from softdoc.reading_environment import (
    DeterministicContentReader,
    ReaderContext,
    ReaderObservationDraft,
    ReaderOutput,
)
from softdoc.reading_state import (
    EvidenceCheckDecision,
    EvidenceCheckInput,
    EvidenceCheckResult,
    ObservationLimitation,
    ObservationSourceRef,
    ReaderKind,
    ReadRepresentation,
    materialize_evidence_check_decision,
)
from softdoc.table_reading import (
    MULTIMODAL_TABLE_READER_SYSTEM_PROMPT,
    TableHeaderStatus,
    TableReadCell,
    TableReadInput,
    TableReadRequest,
    TableReadResult,
    select_relevant_row_hints,
    table_reader_user_prompt,
    validate_table_read_result,
)
from softdoc.visual_reading import (
    VISUAL_READER_SYSTEM_PROMPT,
    VisualInput,
    VisualReadRequest,
    VisualReadResult,
    validate_visual_read_result,
    visual_reader_user_prompt,
)
from softdoc.visual_retrieval import (
    VISUAL_RETRIEVAL_PROMPT_VERSION,
    VISUAL_RETRIEVAL_SYSTEM_PROMPT,
    VisualRetrievalDraft,
    VisualRetrievalRequest,
    VisualRetrievalResult,
    VisualSearchIdentity,
    visual_retrieval_user_prompt,
)


class OllamaModelError(RuntimeError):
    """Raised when an Ollama call or its structured output is invalid."""

    def __init__(self, message: str, *, raw_content: str | None = None) -> None:
        super().__init__(message)
        self.raw_content = raw_content


class OllamaModelConfig(SoftDocModel):
    """Shared deterministic generation settings for one model role."""

    model: str
    base_url: str = "http://localhost:11434"
    timeout_seconds: float = Field(default=180.0, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int = 42
    context_length: int = Field(default=8192, ge=1024)
    think: bool = False
    keep_alive: str = "30m"

    @field_validator("model", "keep_alive")
    @classmethod
    def clean_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Ollama configuration values must not be blank")
        return stripped

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        stripped = value.strip().rstrip("/")
        if not stripped:
            raise ValueError("Ollama base_url must not be blank")
        return stripped


class OllamaModelTransport(Protocol):
    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]: ...


class UrllibOllamaModelTransport:
    """Standard-library HTTP transport with no Ollama SDK dependency."""

    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = request.Request(
            url,
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except (error.URLError, TimeoutError, OSError) as exc:
            raise OllamaModelError(
                f"Could not call the Ollama service at {url}: {exc}"
            ) from exc
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise OllamaModelError("Ollama returned invalid response JSON") from exc
        if not isinstance(decoded, dict):
            raise OllamaModelError("Ollama response must be a JSON object")
        return decoded


class ModelCallRecord(SoftDocModel):
    """Small audit record for one structured model generation."""

    component: str
    model: str
    raw_content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


TModel = TypeVar("TModel", bound=BaseModel)


class OllamaStructuredClient:
    """Call Ollama with schema-constrained JSON and retain an audit trail."""

    def __init__(
        self,
        config: OllamaModelConfig,
        transport: OllamaModelTransport | None = None,
    ) -> None:
        self.config = config
        self.transport = transport or UrllibOllamaModelTransport()
        self.call_records: list[ModelCallRecord] = []

    def generate(
        self,
        *,
        component: str,
        system_prompt: str,
        user_prompt: str,
        output_model: type[TModel],
        image_paths: list[Path] | None = None,
    ) -> TModel:
        user_message: dict[str, Any] = {"role": "user", "content": user_prompt}
        if image_paths:
            user_message["images"] = [
                base64.b64encode(Path(path).read_bytes()).decode("ascii")
                for path in image_paths
            ]
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                user_message,
            ],
            "stream": False,
            "think": self.config.think,
            "format": output_model.model_json_schema(),
            "keep_alive": self.config.keep_alive,
            "options": {
                "temperature": self.config.temperature,
                "seed": self.config.seed,
                "num_ctx": self.config.context_length,
            },
        }
        response = self.transport.post_json(
            f"{self.config.base_url}/api/chat",
            payload,
            self.config.timeout_seconds,
        )
        message = response.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        response_channel = "content"
        if (not isinstance(content, str) or not content.strip()) and isinstance(
            message, dict
        ):
            # Some Ollama vision-model builds place schema-constrained output in
            # ``thinking`` even when thinking is disabled. Accept it only as the
            # candidate structured payload; Pydantic validation below remains
            # the authority on whether it is usable output.
            thinking = message.get("thinking")
            if isinstance(thinking, str) and thinking.strip():
                content = thinking
                response_channel = "thinking"
        if not isinstance(content, str) or not content.strip():
            detail = response.get("error")
            suffix = f": {detail}" if detail else ""
            raise OllamaModelError(
                f"Ollama response has no message content{suffix}"
            )
        try:
            result = output_model.model_validate_json(content)
        except ValidationError as exc:
            raise OllamaModelError(
                f"Ollama returned invalid {component} JSON: {exc}",
                raw_content=content,
            ) from exc

        metadata_keys = (
            "done_reason",
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "eval_count",
        )
        self.call_records.append(
            ModelCallRecord(
                component=component,
                model=str(response.get("model") or self.config.model),
                raw_content=content,
                metadata={
                    **{key: response[key] for key in metadata_keys if key in response},
                    "response_channel": response_channel,
                },
            )
        )
        return result


def _is_p10_checker_contract_error(exc: Exception) -> bool:
    """Limit automatic repair to the two audited P10 contract failures.

    P09 truncation is deliberately excluded until its targeted re-test has run.
    Infrastructure failures and unrelated semantic validation errors also pass
    through unchanged.
    """

    message = str(exc)
    return (
        "Evidence replacement IDs must be unique" in message
        or (
            "Checker must assess every" in message
            and "Observation exactly once and no others" in message
        )
        or "Checker delta references unavailable Observations" in message
    )


def _last_checker_raw_content(client: Any, exc: Exception) -> str:
    raw_content = getattr(exc, "raw_content", None)
    if isinstance(raw_content, str) and raw_content.strip():
        return raw_content

    raw_content = getattr(client, "last_raw_content", None)
    if isinstance(raw_content, str) and raw_content.strip():
        return raw_content

    call_records = getattr(client, "call_records", None)
    if isinstance(call_records, list):
        for record in reversed(call_records):
            if getattr(record, "component", None) == "checker":
                value = getattr(record, "raw_content", None)
                if isinstance(value, str) and value.strip():
                    return value
    return "<raw Checker output was unavailable>"


def _build_checker_contract_repair_prompt(
    *,
    checker_input: EvidenceCheckInput,
    original_user_prompt: str,
    rejected_content: str,
    validation_error: str,
) -> str:
    presented_observation_ids = [
        item.observation_id
        for item in [
            *checker_input.observations,
            *checker_input.recalled_observations,
        ]
    ]
    existing_evidence_ids = [
        item.evidence_id for item in checker_input.evidence_memory.evidence
    ]
    return (
        original_user_prompt
        + "\n\nThe previous Checker response was rejected by the deterministic "
        "contract validator. Repair only the structural contract error and "
        "return one complete strict JSON object again. This repair is part of "
        "the same Checker invocation; do not invent a new read or action.\n"
        + f"Validation error: {validation_error}\n"
        + "Allowed Observation IDs (copy exactly; do not guess, edit, or "
        "fuzzy-match):\n"
        + json.dumps(presented_observation_ids, ensure_ascii=False, indent=2)
        + "\nExisting Evidence IDs that may be replaced or removed (copy "
        "exactly):\n"
        + json.dumps(existing_evidence_ids, ensure_ascii=False, indent=2)
        + "\nRepair rules:\n"
        + "- Assess every allowed Observation ID exactly once and no other ID.\n"
        + "- Reference only allowed Observation IDs in Evidence updates.\n"
        + "- Each existing Evidence ID may occur at most once across replace "
        "and remove. If duplicate replacements were attempted, consolidate "
        "them into one unambiguous final replacement.\n"
        + "- Do not silently select an arbitrary duplicate and do not alter "
        "factual content merely to satisfy the schema.\n"
        + "Rejected response:\n"
        + rejected_content
    )


class OllamaEvidenceCheckerBackend:
    def __init__(self, client: OllamaStructuredClient) -> None:
        self.client = client
        self.last_rejected_attempts: list[dict[str, str]] = []

    def check(self, checker_input: EvidenceCheckInput) -> EvidenceCheckResult:
        original_user_prompt = checker_input.model_dump_json(indent=2)
        user_prompt = original_user_prompt
        self.last_rejected_attempts = []
        for attempt in range(2):
            try:
                decision = self.client.generate(
                    component="checker",
                    system_prompt=CHECKER_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    output_model=EvidenceCheckDecision,
                )
                return materialize_evidence_check_decision(checker_input, decision)
            except Exception as exc:
                if not _is_p10_checker_contract_error(exc):
                    raise
                raw_content = _last_checker_raw_content(self.client, exc)
                self.last_rejected_attempts.append(
                    {
                        "raw_content": raw_content,
                        "validation_error": str(exc),
                    }
                )
                if attempt == 1:
                    raise
                user_prompt = _build_checker_contract_repair_prompt(
                    checker_input=checker_input,
                    original_user_prompt=original_user_prompt,
                    rejected_content=raw_content,
                    validation_error=str(exc),
                )

        raise AssertionError("Checker repair loop exited without a result")

    def check_coverage(
        self, checker_input: CoverageBatchCheckInput
    ) -> CoverageBatchCheckResult:
        """Judge one bounded inventory batch without claiming completeness."""

        result = self.client.generate(
            component="coverage_checker",
            system_prompt=COVERAGE_CHECKER_SYSTEM_PROMPT,
            user_prompt=checker_input.model_dump_json(indent=2),
            output_model=CoverageBatchCheckResult,
        )
        return validate_coverage_batch_result(checker_input, result)


class OllamaAnswererBackend:
    def __init__(self, client: OllamaStructuredClient) -> None:
        self.client = client

    def answer(self, answer_input: AnswerInput) -> AnswerResult:
        return self.client.generate(
            component="answerer",
            system_prompt=ANSWERER_SYSTEM_PROMPT,
            user_prompt=answerer_user_prompt(answer_input),
            output_model=AnswerResult,
        )


class OllamaVisualRetrievalBackend:
    """Generate one schema-bound, retrieval-only identity per visual asset."""

    prompt_version = VISUAL_RETRIEVAL_PROMPT_VERSION

    def __init__(self, client: OllamaStructuredClient) -> None:
        self.client = client

    def describe(self, request: VisualRetrievalRequest) -> VisualRetrievalResult:
        descriptors: list[VisualRetrievalDraft] = []
        for visual_input in request.visual_inputs:
            identity = self.client.generate(
                component="visual_retrieval",
                system_prompt=VISUAL_RETRIEVAL_SYSTEM_PROMPT,
                user_prompt=visual_retrieval_user_prompt(visual_input),
                output_model=VisualSearchIdentity,
                image_paths=[visual_input.visual_asset_path],
            )
            descriptors.append(
                VisualRetrievalDraft(
                    input_id=visual_input.input_id,
                    search_summary=identity.search_summary,
                    keywords=identity.keywords,
                )
            )
        return VisualRetrievalResult(descriptors=descriptors)


class OllamaVisualReaderBackend:
    """Convert Environment visual inputs to the frozen Visual Reader contract."""

    _VISUAL_REPRESENTATIONS = {
        ReadRepresentation.ELEMENT_VISUAL,
        ReadRepresentation.PAGE_VISUAL,
        ReadRepresentation.REGION_CROP,
    }

    def __init__(self, client: OllamaStructuredClient) -> None:
        self.client = client

    def read(self, context: ReaderContext) -> ReaderOutput:
        if not context.inputs or any(
            item.representation not in self._VISUAL_REPRESENTATIONS
            for item in context.inputs
        ):
            raise ValueError("OllamaVisualReaderBackend accepts visual inputs only")

        visual_inputs: list[VisualInput] = []
        image_paths: list[Path] = []
        for item in context.inputs:
            page = context.pages_by_id[item.page_id]
            assert item.visual_asset_id is not None
            assert item.visual_asset_path is not None
            image_paths.append(item.visual_asset_path)
            element = (
                context.elements_by_id.get(item.element_id or "")
                if item.element_id is not None
                else None
            )
            visual_inputs.append(
                VisualInput(
                    input_id=item.input_id,
                    visual_asset_id=item.visual_asset_id,
                    page_id=item.page_id,
                    physical_page_number=page.page_number,
                    document_page_count=len(context.document.pages),
                    is_last_page=(page.page_id == context.document.pages[-1].page_id),
                    display_page_label=page.display_page_label,
                    page_image_path=item.visual_asset_path,
                    element_id=item.element_id,
                    element_type=(
                        element.element_type.value if element is not None else None
                    ),
                    bbox=item.bbox,
                )
            )

        visual_request = VisualReadRequest(
            action_id=context.action_id,
            subquestion_id=context.question_id,
            document_id=context.document.document_id,
            source_name=(
                context.document.title or context.document.source_path.name
            ),
            problem=context.local_problem,
            visual_inputs=visual_inputs,
        )
        result = self.client.generate(
            component="visual_reader",
            system_prompt=VISUAL_READER_SYSTEM_PROMPT,
            user_prompt=visual_reader_user_prompt(visual_request),
            output_model=VisualReadResult,
            image_paths=image_paths,
        )
        validate_visual_read_result(visual_request, result)
        return ReaderOutput(
            reader_kind=(
                ReaderKind.PAGE
                if all(
                    item.representation == ReadRepresentation.PAGE_VISUAL
                    for item in context.inputs
                )
                else ReaderKind.VISUAL
            ),
            observations=[
                ReaderObservationDraft(
                    text=item.text,
                    sources=[
                        ObservationSourceRef(input_id=source.input_id, bbox=source.bbox)
                        for source in item.sources
                    ],
                )
                for item in result.observations
            ],
            limitations=[
                ObservationLimitation(
                    description=item.description,
                    input_ids=item.input_ids,
                )
                for item in result.limitations
            ],
        )


class MultimodalTableReaderBackend:
    """Read selected Tables from structured cells plus optional table pixels."""

    _TABLE_REPRESENTATIONS = {
        ReadRepresentation.ELEMENT_TEXT,
        ReadRepresentation.TABLE_VIEW,
        ReadRepresentation.ELEMENT_VISUAL,
    }

    def __init__(self, client: OllamaStructuredClient) -> None:
        self.client = client

    def read(self, context: ReaderContext) -> ReaderOutput:
        if not context.inputs:
            raise ValueError("MultimodalTableReaderBackend requires table inputs")

        # Imported lazily to avoid a module cycle through ReadingEnvironment's
        # retrieval package imports.
        from softdoc.retrieval.units import table_header_context

        resolved_headers = table_header_context(context.document)
        table_inputs: list[TableReadInput] = []
        image_paths: list[Path] = []
        for item in context.inputs:
            element = context.elements_by_id.get(item.element_id or "")
            if (
                element is None
                or element.element_type != ElementType.TABLE
                or item.representation not in self._TABLE_REPRESENTATIONS
            ):
                raise ValueError(
                    "MultimodalTableReaderBackend accepts Table inputs only"
                )

            page = context.pages_by_id[item.page_id]
            header_context = resolved_headers.get(element.element_id)
            cells: list[TableReadCell] = []
            row_count: int | None = None
            column_count: int | None = None
            if item.representation == ReadRepresentation.TABLE_VIEW:
                view = context.table_views_by_id[item.table_view_id or ""]
                row_count = view.row_count
                column_count = view.column_count
                cells = [
                    TableReadCell(
                        cell_id=cell.cell_id,
                        row=cell.row,
                        column=cell.column,
                        rowspan=cell.rowspan,
                        colspan=cell.colspan,
                        text=cell.text,
                    )
                    for cell in view.cells
                    if cell.text is not None
                ]

            if item.visual_asset_path is not None:
                assert item.visual_asset_id is not None
                image_paths.append(item.visual_asset_path)

            table_inputs.append(
                TableReadInput(
                    input_id=item.input_id,
                    element_id=element.element_id,
                    page_id=item.page_id,
                    physical_page_number=page.page_number,
                    document_page_count=len(context.document.pages),
                    is_last_page=(page.page_id == context.document.pages[-1].page_id),
                    display_page_label=page.display_page_label,
                    row_count=row_count,
                    column_count=column_count,
                    structured_cells=cells,
                    extracted_text=(
                        (element.text or "").strip() or None
                        if not cells
                        else None
                    ),
                    visual_asset_id=item.visual_asset_id,
                    visual_asset_path=item.visual_asset_path,
                    headers=(
                        list(header_context.headers) if header_context else []
                    ),
                    header_status=(
                        TableHeaderStatus.CONFIRMED_INHERITED
                        if header_context
                        and header_context.headers
                        and header_context.is_continuation
                        and not header_context.header_is_inferred
                        and header_context.header_source_element_id
                        != element.element_id
                        else TableHeaderStatus.INFERRED_INHERITED
                        if header_context
                        and header_context.headers
                        and header_context.is_continuation
                        and header_context.header_source_element_id
                        != element.element_id
                        else TableHeaderStatus.LOCAL
                        if header_context
                        and header_context.headers
                        and not header_context.header_is_inferred
                        else TableHeaderStatus.INFERRED_LOCAL
                        if header_context and header_context.headers
                        else TableHeaderStatus.NOT_AVAILABLE_IN_STRUCTURE
                        if not cells and not (element.html or "").strip()
                        else TableHeaderStatus.UNRESOLVED
                    ),
                    header_source_element_id=(
                        header_context.header_source_element_id
                        if header_context
                        else None
                    ),
                    table_group_id=(
                        header_context.group_id if header_context else None
                    ),
                    fragment_index=(
                        header_context.fragment_index if header_context else None
                    ),
                    fragment_count=(
                        header_context.fragment_count if header_context else None
                    ),
                    relevant_row_hints=select_relevant_row_hints(
                        cells, context.local_problem
                    ),
                )
            )

        table_request = TableReadRequest(
            action_id=context.action_id,
            subquestion_id=context.question_id,
            document_id=context.document.document_id,
            source_name=(
                context.document.title or context.document.source_path.name
            ),
            problem=context.local_problem,
            table_inputs=table_inputs,
        )
        result = self.client.generate(
            component="multimodal_table_reader",
            system_prompt=MULTIMODAL_TABLE_READER_SYSTEM_PROMPT,
            user_prompt=table_reader_user_prompt(table_request),
            output_model=TableReadResult,
            image_paths=image_paths,
        )
        validate_table_read_result(table_request, result)
        return ReaderOutput(
            reader_kind=ReaderKind.TABLE,
            observations=[
                ReaderObservationDraft(
                    text=observation.text,
                    sources=[
                        ObservationSourceRef(
                            input_id=source.input_id,
                            cell_id=source.cell_id,
                        )
                        for source in observation.sources
                    ],
                )
                for observation in result.observations
            ],
            limitations=[
                ObservationLimitation(
                    code=limitation.code.value,
                    description=limitation.description,
                    input_ids=limitation.input_ids,
                    relevant_visible_content=(
                        limitation.relevant_visible_content
                    ),
                )
                for limitation in result.limitations
            ],
        )


class ModelBackedReader:
    """Use deterministic structured reading and a VLM only for pixel inputs."""

    _VISUAL_REPRESENTATIONS = OllamaVisualReaderBackend._VISUAL_REPRESENTATIONS

    def __init__(
        self,
        visual_reader: OllamaVisualReaderBackend,
        deterministic_reader: DeterministicContentReader | None = None,
        table_reader: MultimodalTableReaderBackend | None = None,
    ) -> None:
        self.visual_reader = visual_reader
        self.deterministic_reader = deterministic_reader or DeterministicContentReader()
        self.table_reader = table_reader

    def read(self, context: ReaderContext) -> ReaderOutput:
        table_model_inputs = tuple(
            item
            for item in context.inputs
            if self.table_reader is not None
            and context.elements_by_id.get(item.element_id or "") is not None
            and context.elements_by_id[item.element_id or ""].element_type
            == ElementType.TABLE
            and item.representation
            in MultimodalTableReaderBackend._TABLE_REPRESENTATIONS
        )
        table_input_ids = {item.input_id for item in table_model_inputs}
        visual_inputs = tuple(
            item
            for item in context.inputs
            if item.representation in self._VISUAL_REPRESENTATIONS
            and item.input_id not in table_input_ids
        )
        structured_inputs = tuple(
            item
            for item in context.inputs
            if item.representation not in self._VISUAL_REPRESENTATIONS
            and item.input_id not in table_input_ids
        )
        outputs: list[ReaderOutput] = []
        if structured_inputs:
            outputs.append(
                self.deterministic_reader.read(
                    ReaderContext(
                        action_id=context.action_id,
                        question_id=context.question_id,
                        local_problem=context.local_problem,
                        document=context.document,
                        inputs=structured_inputs,
                        elements_by_id=context.elements_by_id,
                        pages_by_id=context.pages_by_id,
                        table_views_by_id=context.table_views_by_id,
                    )
                )
            )
        if table_model_inputs:
            assert self.table_reader is not None
            outputs.append(
                self.table_reader.read(
                    ReaderContext(
                        action_id=context.action_id,
                        question_id=context.question_id,
                        local_problem=context.local_problem,
                        document=context.document,
                        inputs=table_model_inputs,
                        elements_by_id=context.elements_by_id,
                        pages_by_id=context.pages_by_id,
                        table_views_by_id=context.table_views_by_id,
                    )
                )
            )
        if visual_inputs:
            outputs.append(
                self.visual_reader.read(
                    ReaderContext(
                        action_id=context.action_id,
                        question_id=context.question_id,
                        local_problem=context.local_problem,
                        document=context.document,
                        inputs=visual_inputs,
                        elements_by_id=context.elements_by_id,
                        pages_by_id=context.pages_by_id,
                        table_views_by_id=context.table_views_by_id,
                    )
                )
            )
        if not outputs:
            raise ValueError("ReaderContext contains no inputs")
        if len(outputs) == 1:
            return outputs[0]
        reader_kinds = {output.reader_kind for output in outputs}
        return ReaderOutput(
            reader_kind=(
                next(iter(reader_kinds))
                if len(reader_kinds) == 1
                else ReaderKind.VISUAL
            ),
            observations=[item for output in outputs for item in output.observations],
            limitations=[item for output in outputs for item in output.limitations],
        )
