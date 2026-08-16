"""Ollama implementation of the injectable Planner backend."""

from __future__ import annotations

import json
from typing import Any, Protocol
from urllib import error, request

from pydantic import Field, field_validator

from softdoc.models import SoftDocModel
from softdoc.planning.models import PlannerBackendResponse, PlannerDraft


class OllamaPlannerError(RuntimeError):
    """Raised when the local Ollama service cannot produce a response."""


class OllamaPlannerConfig(SoftDocModel):
    base_url: str = "http://localhost:11434"
    model: str = "qwen3:4b-instruct-2507-q4_K_M"
    timeout_seconds: float = Field(default=180.0, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    think: bool = False

    @field_validator("base_url", "model")
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


class OllamaTransport(Protocol):
    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]: ...


class UrllibOllamaTransport:
    """Small standard-library HTTP transport; no Ollama SDK is required."""

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
            raise OllamaPlannerError(
                f"Could not call the local Ollama service at {url}: {exc}"
            ) from exc
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise OllamaPlannerError("Ollama returned invalid response JSON") from exc
        if not isinstance(decoded, dict):
            raise OllamaPlannerError("Ollama response must be a JSON object")
        return decoded


class OllamaPlannerBackend:
    """Generate strict PlannerDraft JSON with a model served by Ollama."""

    def __init__(
        self,
        config: OllamaPlannerConfig | None = None,
        transport: OllamaTransport | None = None,
    ) -> None:
        self._config = config or OllamaPlannerConfig()
        self._transport = transport or UrllibOllamaTransport()

    @property
    def backend_name(self) -> str:
        return "ollama"

    def generate(self, prompt: str) -> PlannerBackendResponse:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": self._config.think,
            "format": PlannerDraft.model_json_schema(),
            "options": {"temperature": self._config.temperature},
        }
        response = self._transport.post_json(
            f"{self._config.base_url}/api/chat",
            payload,
            self._config.timeout_seconds,
        )
        message = response.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            error_message = response.get("error")
            detail = f": {error_message}" if error_message else ""
            raise OllamaPlannerError(f"Ollama response has no message content{detail}")

        metadata_keys = (
            "done_reason",
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "eval_count",
        )
        return PlannerBackendResponse(
            content=content,
            model=str(response.get("model") or self._config.model),
            metadata={key: response[key] for key in metadata_keys if key in response},
        )
