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
from softdoc.models import SoftDocModel
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


class OllamaEvidenceCheckerBackend:
    def __init__(self, client: OllamaStructuredClient) -> None:
        self.client = client

    def check(self, checker_input: EvidenceCheckInput) -> EvidenceCheckResult:
        decision = self.client.generate(
            component="checker",
            system_prompt=CHECKER_SYSTEM_PROMPT,
            user_prompt=checker_input.model_dump_json(indent=2),
            output_model=EvidenceCheckDecision,
        )
        return materialize_evidence_check_decision(checker_input, decision)


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
                    page_number=page.page_number,
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


class ModelBackedReader:
    """Use deterministic structured reading and a VLM only for pixel inputs."""

    _VISUAL_REPRESENTATIONS = OllamaVisualReaderBackend._VISUAL_REPRESENTATIONS

    def __init__(
        self,
        visual_reader: OllamaVisualReaderBackend,
        deterministic_reader: DeterministicContentReader | None = None,
    ) -> None:
        self.visual_reader = visual_reader
        self.deterministic_reader = deterministic_reader or DeterministicContentReader()

    def read(self, context: ReaderContext) -> ReaderOutput:
        visual_inputs = tuple(
            item
            for item in context.inputs
            if item.representation in self._VISUAL_REPRESENTATIONS
        )
        structured_inputs = tuple(
            item
            for item in context.inputs
            if item.representation not in self._VISUAL_REPRESENTATIONS
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
        return ReaderOutput(
            reader_kind=ReaderKind.VISUAL,
            observations=[item for output in outputs for item in output.observations],
            limitations=[item for output in outputs for item in output.limitations],
        )
