from pathlib import Path

from pydantic import BaseModel

from softdoc.openai_compatible import OpenAICompatibleConfig, OpenAICompatibleStructuredClient


class _FakeTransport:
    def __init__(self):
        self.url = None
        self.payload = None

    def post_json(self, url, payload, timeout_seconds):
        self.url = url
        self.payload = payload
        return {"choices": [{"message": {"content": '{"ok": true}'}}]}


class _Result(BaseModel):
    ok: bool


def test_openai_compatible_structured_json_payload():
    transport = _FakeTransport()
    client = OpenAICompatibleStructuredClient(
        OpenAICompatibleConfig(model="Qwen/Qwen3.5-27B"), transport=transport
    )
    result = client.generate(
        component="smoke",
        system_prompt="Return JSON.",
        user_prompt="test",
        output_model=_Result,
    )
    assert result.ok is True
    assert transport.url == "http://localhost:8000/v1/chat/completions"
    assert transport.payload["response_format"]["type"] == "json_schema"
    assert transport.payload["messages"][1]["content"] == "test"


def test_openai_compatible_image_message(tmp_path: Path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"png")
    transport = _FakeTransport()
    client = OpenAICompatibleStructuredClient(
        OpenAICompatibleConfig(model="Qwen/Qwen3.5-27B"), transport=transport
    )
    client.generate(
        component="smoke",
        system_prompt="Return JSON.",
        user_prompt="inspect",
        output_model=_Result,
        image_paths=[image],
    )
    content = transport.payload["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "inspect"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
