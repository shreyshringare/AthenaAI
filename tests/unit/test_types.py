"""P0 gate: core types, protocols, and config."""

from __future__ import annotations

import dataclasses

import pytest

from athenai.core.config import AthenaConfig
from athenai.core.exceptions import (
    CircuitOpenError,
    ContextOverflowError,
    EmbeddingError,
    ModelUnavailableError,
    PolicyViolationError,
    RateLimitError,
    ToolDeniedError,
    ToolTimeoutError,
)
from athenai.core.protocols import (
    CacheBackend,
    EmbedderProtocol,
    MemoryStore,
    Model,
    Retriever,
    StreamingModel,
    Tool,
)
from athenai.core.types import (
    AIRequest,
    AIResponse,
    Message,
    MessageRole,
    RoutingContext,
    TokenUsage,
    TraceSpan,
)

# ---------------------------------------------------------------------------
# Frozen dataclass instantiation
# ---------------------------------------------------------------------------


def test_message_instantiation() -> None:
    msg = Message(role=MessageRole.USER, content="hello")
    assert msg.role == MessageRole.USER
    assert msg.content == "hello"


def test_message_frozen() -> None:
    msg = Message(role=MessageRole.USER, content="hello")
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        msg.content = "mutated"  # type: ignore[misc]


def test_token_usage_instantiation() -> None:
    usage = TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30)
    assert usage.total_tokens == 30


def test_token_usage_frozen() -> None:
    usage = TokenUsage(input_tokens=1, output_tokens=2, total_tokens=3)
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        usage.total_tokens = 999  # type: ignore[misc]


def test_ai_request_instantiation() -> None:
    msg = Message(role=MessageRole.USER, content="hi")
    req = AIRequest(
        messages=(msg,),
        user_id="u1",
        session_id="s1",
    )
    assert len(req.messages) == 1
    assert req.user_id == "u1"


def test_ai_request_frozen() -> None:
    msg = Message(role=MessageRole.USER, content="hi")
    req = AIRequest(messages=(msg,), user_id="u1", session_id="s1")
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        req.user_id = "u2"  # type: ignore[misc]


def test_ai_request_empty_messages_raises() -> None:
    with pytest.raises(ValueError):
        AIRequest(messages=(), user_id="u1", session_id="s1")


def test_ai_response_instantiation() -> None:
    usage = TokenUsage(input_tokens=5, output_tokens=10, total_tokens=15)
    resp = AIResponse(
        content="answer",
        model="mock-v1",
        trace_id="t1",
        usage=usage,
    )
    assert resp.content == "answer"
    assert resp.model == "mock-v1"


def test_ai_response_frozen() -> None:
    usage = TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2)
    resp = AIResponse(content="x", model="m", trace_id="t", usage=usage)
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        resp.content = "mutated"  # type: ignore[misc]


def test_trace_span_instantiation() -> None:
    span = TraceSpan(
        name="model_call",
        start_ms=0.0,
        end_ms=123.4,
        trace_id="t1",
        span_id="s1",
    )
    assert span.name == "model_call"
    assert span.parent_span_id is None


def test_trace_span_frozen() -> None:
    span = TraceSpan(name="x", start_ms=0.0, end_ms=1.0, trace_id="t", span_id="s")
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        span.end_ms = 999.0  # type: ignore[misc]


def test_routing_context_instantiation() -> None:
    ctx = RoutingContext(
        request_id="r1",
        user_id="u1",
        estimated_input_tokens=500,
        task_description="summarise",
    )
    assert ctx.complexity == "MEDIUM"


def test_routing_context_frozen() -> None:
    ctx = RoutingContext(
        request_id="r1",
        user_id="u1",
        estimated_input_tokens=500,
        task_description="summarise",
    )
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        ctx.complexity = "LOW"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Protocol imports (structural — no instantiation needed)
# ---------------------------------------------------------------------------


def test_protocol_imports() -> None:
    for proto in (Model, StreamingModel, MemoryStore, Retriever, Tool, CacheBackend,
                  EmbedderProtocol):
        assert proto is not None


# ---------------------------------------------------------------------------
# Exception imports
# ---------------------------------------------------------------------------


def test_exception_imports() -> None:
    for exc_cls in (
        ContextOverflowError,
        ModelUnavailableError,
        ToolDeniedError,
        ToolTimeoutError,
        PolicyViolationError,
        EmbeddingError,
        CircuitOpenError,
        RateLimitError,
    ):
        instance = exc_cls("test")
        assert isinstance(instance, Exception)


# ---------------------------------------------------------------------------
# AthenaConfig env loading
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    config = AthenaConfig()
    assert config.log_level == "INFO"
    assert "default" in config.model_registry
    assert "total" in config.token_budget


def test_config_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATHENA_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("ATHENA_ENVIRONMENT", "test")
    config = AthenaConfig()
    assert config.log_level == "DEBUG"
    assert config.environment == "test"


def test_config_token_budget_keys() -> None:
    config = AthenaConfig()
    required_keys = {"system", "conversation", "memory", "rag", "tools", "reserved", "total"}
    assert required_keys.issubset(config.token_budget.keys())
