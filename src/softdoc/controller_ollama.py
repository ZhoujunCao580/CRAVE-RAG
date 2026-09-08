"""Ollama backend for the frozen Reading Controller v0 contract."""

from __future__ import annotations

import json
from typing import Any, Protocol
from urllib import error, request

from pydantic import Field, TypeAdapter, ValidationError, field_validator

from softdoc.controller import (
    ControllerAction,
    ControllerInput,
    validate_controller_action,
)
from softdoc.controller_prompt import (
    CONTROLLER_SYSTEM_PROMPT,
    build_controller_user_prompt,
)
from softdoc.models import SoftDocModel
from softdoc.openai_compatible import (
    OpenAICompatibleConfig,
    OpenAICompatibleError,
    OpenAICompatibleStructuredClient,
)


class OllamaControllerError(RuntimeError):
    """Raised when Ollama cannot produce a valid Controller action."""

    def __init__(self, message: str, *, raw_content: str | None = None) -> None:
        super().__init__(message)
        self.raw_content = raw_content


class OllamaControllerConfig(SoftDocModel):
    base_url: str = "http://localhost:11434"
    model: str = "qwen3:8b"
    timeout_seconds: float = Field(default=180.0, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int = 42
    context_length: int = Field(default=8192, ge=1024)
    think: bool = False
    keep_alive: str = "30m"

    @field_validator("base_url", "model", "keep_alive")
    @classmethod
    def clean_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Ollama configuration values must not be blank")
        return stripped

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        return value.rstrip("/")


class OllamaControllerTransport(Protocol):
    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]: ...


class UrllibOllamaControllerTransport:
    """Standard-library transport so the backend adds no runtime dependency."""

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
            raise OllamaControllerError(
                f"Could not call the local Ollama service at {url}: {exc}"
            ) from exc
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise OllamaControllerError("Ollama returned invalid response JSON") from exc
        if not isinstance(decoded, dict):
            raise OllamaControllerError("Ollama response must be a JSON object")
        return decoded


class ControllerGeneration(SoftDocModel):
    """One auditable model generation and its validated action."""

    raw_content: str
    action: ControllerAction
    model: str
    metadata: dict[str, Any] = Field(default_factory=dict)


_ACTION_ADAPTER = TypeAdapter(ControllerAction)
_MAX_CONTROLLER_VALIDATION_ATTEMPTS = 2


def _visible_read_source_ids(controller_input: ControllerInput) -> list[str]:
    source_ids = {item.source_id for item in controller_input.reading_locations}
    if controller_input.visible_search_view is not None:
        source_ids.update(
            item.target_id
            for item in controller_input.visible_search_view.exact_anchor_matches
        )
        source_ids.update(
            item.element_id
            for item in controller_input.visible_search_view.candidate_previews
        )
    return sorted(source_ids)


def _build_controller_repair_user_prompt(
    controller_input: ControllerInput,
    *,
    rejected_content: str,
    validation_error: str,
) -> str:
    """Build one bounded repair request without relaxing source visibility."""

    current_read_ids = _visible_read_source_ids(controller_input)
    current_read_set = set(current_read_ids)
    historical_only_ids = sorted(
        {
            target_id
            for action in controller_input.recent_actions
            for target_id in action.target_ids
            if target_id not in current_read_set
        }
    )
    permission_view = {
        "read_source_ids": current_read_ids,
        "confirmed_relation_ids": sorted(
            item.relation_id for item in controller_input.confirmed_relations
        ),
        "candidate_relation_ids": sorted(
            item.relation_id for item in controller_input.candidate_relations
        ),
        "page_context_base_page_ids": sorted(
            {item.page_id for item in controller_input.reading_locations}
        ),
        "historical_non_actionable_ids": historical_only_ids,
    }
    return (
        "The previous Controller action was rejected by the deterministic "
        "validator. Return one complete corrected action JSON. The retry is part "
        "of the same decision and does not grant access to any new source.\n\n"
        f"Validation error:\n{validation_error}\n\n"
        "Current actionable permissions:\n"
        + json.dumps(permission_view, ensure_ascii=False, indent=2)
        + "\n\nPermission rules:\n"
        "- READ_SOURCE may use only read_source_ids.\n"
        "- IDs listed only under historical_non_actionable_ids are history, not "
        "current READ_SOURCE handles.\n"
        "- Use FOLLOW_RELATION with a confirmed_relation_id.\n"
        "- Use EXPLORE_CANDIDATE_RELATION with a candidate_relation_id.\n"
        "- Use READ_PAGE_CONTEXT with a page_context_base_page_id; never pass a "
        "Page ID to READ_SOURCE.\n"
        "- Do not guess, repair, or fuzzy-match an ID. Choose a complete ID exactly "
        "as listed, or choose another legal action.\n\n"
        f"Rejected action:\n{rejected_content}\n\n"
        "ControllerInput remains unchanged:\n"
        + build_controller_user_prompt(controller_input)
    )


class OllamaControllerBackend:
    """Select exactly one action using an Ollama-served language model."""

    def __init__(
        self,
        config: OllamaControllerConfig | None = None,
        transport: OllamaControllerTransport | None = None,
    ) -> None:
        self._config = config or OllamaControllerConfig()
        self._transport = transport or UrllibOllamaControllerTransport()
        self.last_generation: ControllerGeneration | None = None
        self.last_rejected_attempts: list[dict[str, str]] = []

    @property
    def backend_name(self) -> str:
        return "ollama"

    def generate(self, controller_input: ControllerInput) -> ControllerGeneration:
        user_prompt = build_controller_user_prompt(controller_input)
        rejected_attempts: list[dict[str, str]] = []
        self.last_generation = None
        self.last_rejected_attempts = []
        for attempt in range(1, _MAX_CONTROLLER_VALIDATION_ATTEMPTS + 1):
            payload: dict[str, Any] = {
                "model": self._config.model,
                "messages": [
                    {"role": "system", "content": CONTROLLER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "think": self._config.think,
                "format": _ACTION_ADAPTER.json_schema(),
                "keep_alive": self._config.keep_alive,
                "options": {
                    "temperature": self._config.temperature,
                    "seed": self._config.seed,
                    "num_ctx": self._config.context_length,
                },
            }
            response = self._transport.post_json(
                f"{self._config.base_url}/api/chat",
                payload,
                self._config.timeout_seconds,
            )
            message = response.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str) or not content.strip():
                detail = response.get("error")
                suffix = f": {detail}" if detail else ""
                raise OllamaControllerError(
                    f"Ollama response has no message content{suffix}"
                )

            try:
                decoded = json.loads(content)
                action = validate_controller_action(decoded, controller_input)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                rejected_attempts.append(
                    {"raw_content": content, "validation_error": str(exc)}
                )
                self.last_rejected_attempts = list(rejected_attempts)
                if attempt == _MAX_CONTROLLER_VALIDATION_ATTEMPTS:
                    raise OllamaControllerError(
                        f"Ollama returned an invalid Controller action: {exc}",
                        raw_content=content,
                    ) from exc
                user_prompt = _build_controller_repair_user_prompt(
                    controller_input,
                    rejected_content=content,
                    validation_error=str(exc),
                )
                continue

            metadata_keys = (
                "done_reason",
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "eval_count",
            )
            metadata = {
                key: response[key] for key in metadata_keys if key in response
            }
            metadata["validation_attempts"] = attempt
            if rejected_attempts:
                metadata["rejected_attempts"] = rejected_attempts
            generation = ControllerGeneration(
                raw_content=content,
                action=action,
                model=str(response.get("model") or self._config.model),
                metadata=metadata,
            )
            self.last_generation = generation
            self.last_rejected_attempts = list(rejected_attempts)
            return generation

        raise AssertionError("Controller validation loop ended without a result")

    def decide(self, controller_input: ControllerInput) -> ControllerAction:
        """Implement the ReadingEnvironment ControllerBackend protocol."""

        return self.generate(controller_input).action


class VLLMControllerBackend:
    """Controller backend for vLLM (or any OpenAI-compatible chat server)."""

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        client: OpenAICompatibleStructuredClient | None = None,
    ) -> None:
        self._config = config
        self._client = client or OpenAICompatibleStructuredClient(config)
        self.last_generation: ControllerGeneration | None = None
        self.last_rejected_attempts: list[dict[str, str]] = []

    @property
    def backend_name(self) -> str:
        return "vllm-openai-compatible"

    def generate(self, controller_input: ControllerInput) -> ControllerGeneration:
        user_prompt = build_controller_user_prompt(controller_input)
        rejected_attempts: list[dict[str, str]] = []
        self.last_generation = None
        self.last_rejected_attempts = []
        for attempt in range(1, _MAX_CONTROLLER_VALIDATION_ATTEMPTS + 1):
            try:
                action = self._client.generate(
                    component="controller_action",
                    system_prompt=CONTROLLER_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    output_model=_ACTION_ADAPTER,
                )
                raw_content = self._client.last_raw_content or action.model_dump_json()
                validated = validate_controller_action(action, controller_input)
            except OpenAICompatibleError as exc:
                if not exc.raw_content:
                    raise OllamaControllerError(str(exc)) from exc
                raw_content = exc.raw_content
                validation_error = str(exc)
            except (ValidationError, ValueError) as exc:
                validation_error = str(exc)
            else:
                metadata: dict[str, Any] = {
                    "transport": "openai-compatible",
                    "base_url": self._config.base_url,
                    "validation_attempts": attempt,
                }
                if rejected_attempts:
                    metadata["rejected_attempts"] = rejected_attempts
                generation = ControllerGeneration(
                    raw_content=raw_content,
                    action=validated,
                    model=self._config.model,
                    metadata=metadata,
                )
                self.last_generation = generation
                self.last_rejected_attempts = list(rejected_attempts)
                return generation

            rejected_attempts.append(
                {
                    "raw_content": raw_content,
                    "validation_error": validation_error,
                }
            )
            self.last_rejected_attempts = list(rejected_attempts)
            if attempt == _MAX_CONTROLLER_VALIDATION_ATTEMPTS:
                raise OllamaControllerError(
                    "OpenAI-compatible backend returned an invalid Controller "
                    f"action after repair: {validation_error}",
                    raw_content=raw_content,
                )
            user_prompt = _build_controller_repair_user_prompt(
                controller_input,
                rejected_content=raw_content,
                validation_error=validation_error,
            )

        raise AssertionError("Controller validation loop ended without a result")

    def decide(self, controller_input: ControllerInput) -> ControllerAction:
        return self.generate(controller_input).action
