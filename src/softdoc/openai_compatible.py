"""Minimal OpenAI-compatible client for vLLM and compatible servers.

The client deliberately uses urllib instead of the OpenAI SDK so the project
does not gain a hard runtime dependency.  It supports JSON-schema constrained
chat completions and OpenAI vision messages (data-URI images).
"""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, Protocol, TypeVar
from urllib import error, request

from pydantic import Field, ValidationError, field_validator

from softdoc.models import SoftDocModel


class OpenAICompatibleError(RuntimeError):
    """Raised when an OpenAI-compatible server returns unusable output."""

    def __init__(self, message: str, *, raw_content: str | None = None) -> None:
        super().__init__(message)
        self.raw_content = raw_content


class OpenAICompatibleConfig(SoftDocModel):
    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    timeout_seconds: float = Field(default=300.0, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int = 42
    max_tokens: int = Field(default=2048, ge=1)

    @field_validator("model", "api_key")
    @classmethod
    def non_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("OpenAI-compatible configuration values must not be blank")
        return value

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value:
            raise ValueError("OpenAI-compatible base_url must not be blank")
        return value


class OpenAICompatibleTransport(Protocol):
    def post_json(self, url: str, payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]: ...


class UrllibOpenAICompatibleTransport:
    """Standard-library HTTP transport for vLLM's OpenAI server."""

    def __init__(self, api_key: str = "EMPTY") -> None:
        self.api_key = api_key

    def post_json(self, url: str, payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = request.Request(
            url,
            data=encoded,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except (error.URLError, TimeoutError, OSError) as exc:
            raise OpenAICompatibleError(f"Could not call OpenAI-compatible service at {url}: {exc}") from exc
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise OpenAICompatibleError("OpenAI-compatible server returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise OpenAICompatibleError("OpenAI-compatible response must be a JSON object")
        return decoded


TModel = TypeVar("TModel")


class OpenAICompatibleStructuredClient:
    """Schema-bound client usable by Controller, Reader, Checker and Answerer."""

    def __init__(self, config: OpenAICompatibleConfig, transport: OpenAICompatibleTransport | None = None) -> None:
        self.config = config
        self.transport = transport or UrllibOpenAICompatibleTransport(config.api_key)
        self.last_raw_content: str | None = None
        self.last_response: dict[str, Any] | None = None

    @staticmethod
    def _image_data_uri(path: Path) -> str:
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def generate(
        self,
        *,
        component: str,
        system_prompt: str,
        user_prompt: str,
        output_model: Any,
        image_paths: list[Path] | None = None,
    ) -> TModel:
        if image_paths:
            content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
            content.extend({"type": "image_url", "image_url": {"url": self._image_data_uri(path)}} for path in image_paths)
            user_message: dict[str, Any] = {"role": "user", "content": content}
        else:
            user_message = {"role": "user", "content": user_prompt}
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "system", "content": system_prompt}, user_message],
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "max_tokens": self.config.max_tokens,
            "stream": False,
            "response_format": {"type": "json_schema", "json_schema": {"name": component, "schema": output_model.model_json_schema() if hasattr(output_model, "model_json_schema") else output_model.json_schema(), "strict": True}},
        }
        response = self.transport.post_json(f"{self.config.base_url}/chat/completions", payload, self.config.timeout_seconds)
        self.last_response = response
        choices = response.get("choices")
        message = choices[0].get("message") if isinstance(choices, list) and choices else None
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
        if not isinstance(content, str) or not content.strip():
            raise OpenAICompatibleError(f"OpenAI-compatible response has no message content: {response.get('error', '')}")
        self.last_raw_content = content
        try:
            return output_model.model_validate_json(content) if hasattr(output_model, "model_validate_json") else output_model.validate_json(content)
        except ValidationError as exc:
            raise OpenAICompatibleError(f"OpenAI-compatible server returned invalid {component} JSON: {exc}", raw_content=content) from exc
