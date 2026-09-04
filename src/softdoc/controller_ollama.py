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


class OllamaControllerBackend:
    """Select exactly one action using an Ollama-served language model."""

    def __init__(
        self,
        config: OllamaControllerConfig | None = None,
        transport: OllamaControllerTransport | None = None,
    ) -> None:
        self._config = config or OllamaControllerConfig()
        self._transport = transport or UrllibOllamaControllerTransport()

    @property
    def backend_name(self) -> str:
        return "ollama"

    def generate(self, controller_input: ControllerInput) -> ControllerGeneration:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": CONTROLLER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_controller_user_prompt(controller_input),
                },
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
            raise OllamaControllerError(
                f"Ollama returned an invalid Controller action: {exc}",
                raw_content=content,
            ) from exc

        metadata_keys = (
            "done_reason",
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "eval_count",
        )
        return ControllerGeneration(
            raw_content=content,
            action=action,
            model=str(response.get("model") or self._config.model),
            metadata={key: response[key] for key in metadata_keys if key in response},
        )

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

    @property
    def backend_name(self) -> str:
        return "vllm-openai-compatible"

    def generate(self, controller_input: ControllerInput) -> ControllerGeneration:
        try:
            action = self._client.generate(
                component="controller_action",
                system_prompt=CONTROLLER_SYSTEM_PROMPT,
                user_prompt=build_controller_user_prompt(controller_input),
                output_model=_ACTION_ADAPTER,
            )
        except OpenAICompatibleError as exc:
            raise OllamaControllerError(str(exc), raw_content=exc.raw_content) from exc
        validated = validate_controller_action(action, controller_input)
        return ControllerGeneration(
            raw_content=validated.model_dump_json(),
            action=validated,
            model=self._config.model,
            metadata={"transport": "openai-compatible", "base_url": self._config.base_url},
        )

    def decide(self, controller_input: ControllerInput) -> ControllerAction:
        return self.generate(controller_input).action
