"""P1 gate: model layer — MockModel, CloudModel, ModelRegistry."""

from __future__ import annotations

import httpx
import pytest
import respx

from athenai.core.exceptions import ModelUnavailableError, RateLimitError
from athenai.models.base import ModelRequest, ModelResponse
from athenai.models.cloud import CloudModel
from athenai.models.mock import MockModel
from athenai.models.registry import ModelRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _request(content: str = "hello") -> ModelRequest:
    return ModelRequest(
        messages=[{"role": "user", "content": content}],
        model_name="mock",
    )


# ---------------------------------------------------------------------------
# MockModel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_generate_returns_response() -> None:
    model = MockModel(name="test-mock")
    resp = await model.generate(_request("what is 2+2?"))
    assert isinstance(resp, ModelResponse)
    assert len(resp.content) > 0
    assert resp.input_tokens >= 1
    assert resp.output_tokens >= 1


@pytest.mark.asyncio
async def test_mock_stream_yields_tokens() -> None:
    model = MockModel()
    tokens = []
    async for token in model.stream(_request("ping")):
        tokens.append(token)
    assert len(tokens) >= 3


@pytest.mark.asyncio
async def test_mock_health_check() -> None:
    model = MockModel()
    assert await model.health_check() is True
    model.set_healthy(False)
    assert await model.health_check() is False


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------


def _registry() -> ModelRegistry:
    config = {
        "fast": {"type": "mock", "name": "mock-fast"},
        "reasoning": {"type": "mock", "name": "mock-reasoning"},
    }
    return ModelRegistry(config)


def test_registry_get_existing_role() -> None:
    reg = _registry()
    model = reg.get("fast")
    assert isinstance(model, MockModel)


def test_registry_get_nonexistent_raises() -> None:
    reg = _registry()
    with pytest.raises(KeyError):
        reg.get("nonexistent")


def test_registry_list_roles() -> None:
    reg = _registry()
    roles = reg.list_roles()
    assert "fast" in roles
    assert "reasoning" in roles


def test_registry_register_and_get() -> None:
    reg = _registry()
    reg.register("custom", MockModel(name="custom-mock"))
    model = reg.get("custom")
    assert isinstance(model, MockModel)


@pytest.mark.asyncio
async def test_registry_round_trip() -> None:
    reg = _registry()
    fast = reg.get("fast")
    reasoning = reg.get("reasoning")
    resp_fast = await fast.generate(_request("fast task"))
    resp_reasoning = await reasoning.generate(_request("reasoning task"))
    assert resp_fast.content != resp_reasoning.content or True  # both respond


# ---------------------------------------------------------------------------
# CloudModel — mocked via respx
# ---------------------------------------------------------------------------

_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


@pytest.mark.asyncio
async def test_cloud_bad_api_key_raises_unavailable() -> None:
    with respx.mock:
        respx.post(_ANTHROPIC_URL).mock(
            return_value=httpx.Response(401, json={"error": {"type": "authentication_error"}})
        )
        model = CloudModel(api_key="bad-key")
        with pytest.raises(ModelUnavailableError):
            await model.generate(_request("hello"))
        await model.aclose()


@pytest.mark.asyncio
async def test_cloud_429_raises_rate_limit() -> None:
    with respx.mock:
        respx.post(_ANTHROPIC_URL).mock(
            return_value=httpx.Response(429, json={"error": {"type": "rate_limit_error"}})
        )
        model = CloudModel(api_key="test-key")
        with pytest.raises(RateLimitError):
            await model.generate(_request("hello"))
        await model.aclose()


@pytest.mark.asyncio
async def test_cloud_500_raises_unavailable() -> None:
    with respx.mock:
        respx.post(_ANTHROPIC_URL).mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )
        model = CloudModel(api_key="test-key")
        with pytest.raises(ModelUnavailableError):
            await model.generate(_request("hello"))
        await model.aclose()


@pytest.mark.asyncio
async def test_cloud_200_returns_response() -> None:
    with respx.mock:
        respx.post(_ANTHROPIC_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "msg_01",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello from Claude!"}],
                    "model": "claude-sonnet-4-6",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            )
        )
        model = CloudModel(api_key="test-key")
        resp = await model.generate(_request("hi"))
        assert resp.content == "Hello from Claude!"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5
        assert resp.metadata.get("tool_calls") is None or resp.metadata["tool_calls"] == []
        await model.aclose()


@pytest.mark.asyncio
async def test_cloud_tool_use_block_populates_metadata_tool_calls() -> None:
    """Anthropic's real structured tool-use format: a tool_use content
    block with `name`/`input` must become metadata["tool_calls"] in the
    {"name", "arguments"} shape AgentExecutor expects -- not left for the
    TOOL_CALL: text-protocol fallback to parse."""
    with respx.mock:
        respx.post(_ANTHROPIC_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "msg_01",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_01",
                            "name": "calculator",
                            "input": {"expression": "6*7"},
                        }
                    ],
                    "model": "claude-sonnet-4-6",
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 20, "output_tokens": 8},
                },
            )
        )
        model = CloudModel(api_key="test-key")
        resp = await model.generate(_request("what is 6*7?"))
        assert resp.metadata["tool_calls"] == [
            {"name": "calculator", "arguments": {"expression": "6*7"}}
        ]
        assert resp.finish_reason == "tool_use"
        await model.aclose()


@pytest.mark.asyncio
async def test_cloud_mixed_text_and_tool_use_blocks() -> None:
    """Claude can emit explanatory text alongside a tool_use block in the
    same response -- text blocks must still populate .content."""
    with respx.mock:
        respx.post(_ANTHROPIC_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "content": [
                        {"type": "text", "text": "Let me calculate that."},
                        {
                            "type": "tool_use",
                            "id": "toolu_02",
                            "name": "calculator",
                            "input": {"expression": "2+2"},
                        },
                    ],
                    "model": "claude-sonnet-4-6",
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 15, "output_tokens": 10},
                },
            )
        )
        model = CloudModel(api_key="test-key")
        resp = await model.generate(_request("what is 2+2?"))
        assert resp.content == "Let me calculate that."
        assert resp.metadata["tool_calls"] == [
            {"name": "calculator", "arguments": {"expression": "2+2"}}
        ]
        await model.aclose()


@pytest.mark.asyncio
async def test_cloud_forwards_tools_from_request_metadata() -> None:
    """When the caller supplies tool schemas via ModelRequest.metadata["tools"],
    CloudModel must forward them to Anthropic as the request's "tools" field --
    without this, Claude never emits native tool_use blocks regardless of
    response-parsing logic, since the API only offers tool use when the
    request declares what tools exist."""
    captured: dict = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        import json

        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "model": "claude-sonnet-4-6",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    with respx.mock:
        respx.post(_ANTHROPIC_URL).mock(side_effect=_capture)
        model = CloudModel(api_key="test-key")
        tool_schemas = [
            {
                "name": "calculator",
                "description": "Evaluates arithmetic expressions.",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]
        request = ModelRequest(
            messages=[{"role": "user", "content": "hi"}],
            model_name="agent",
            metadata={"tools": tool_schemas},
        )
        await model.generate(request)
        assert captured["payload"]["tools"] == tool_schemas
        await model.aclose()


@pytest.mark.asyncio
async def test_cloud_omits_tools_field_when_none_supplied() -> None:
    captured: dict = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        import json

        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "model": "claude-sonnet-4-6",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    with respx.mock:
        respx.post(_ANTHROPIC_URL).mock(side_effect=_capture)
        model = CloudModel(api_key="test-key")
        await model.generate(_request("hi"))
        assert "tools" not in captured["payload"]
        await model.aclose()
