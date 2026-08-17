"""P7 gate (unit): ToolValidator, CalculatorTool, HTTPTool domain check."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from athenai.core.exceptions import ToolDeniedError
from athenai.tools.calculator import CalculatorTool
from athenai.tools.http import HTTPTool
from athenai.tools.registry import ToolRegistry
from athenai.tools.validator import ToolValidator

# ---------------------------------------------------------------------------
# Minimal test tool
# ---------------------------------------------------------------------------


class _EchoTool:
    name = "echo"
    description = "Echoes its input."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "message": {"type": "string"},
            "count": {"type": "integer"},
        },
        "required": ["message"],
    }

    async def execute(self, arguments: dict[str, Any]) -> Any:
        return arguments["message"]


# ---------------------------------------------------------------------------
# ToolValidator — schema checks
# ---------------------------------------------------------------------------


def test_schema_valid_passes() -> None:
    v = ToolValidator()
    v.validate_schema(_EchoTool(), {"message": "hello"})


def test_schema_missing_required_raises() -> None:
    v = ToolValidator()
    with pytest.raises(ToolDeniedError, match="required argument 'message' missing"):
        v.validate_schema(_EchoTool(), {})


def test_schema_wrong_type_raises() -> None:
    v = ToolValidator()
    with pytest.raises(ToolDeniedError, match="must be string"):
        v.validate_schema(_EchoTool(), {"message": 42})


def test_schema_integer_type_check() -> None:
    v = ToolValidator()
    with pytest.raises(ToolDeniedError, match="must be integer"):
        v.validate_schema(_EchoTool(), {"message": "hi", "count": "five"})


# ---------------------------------------------------------------------------
# ToolValidator — permission checks
# ---------------------------------------------------------------------------


def test_permission_granted() -> None:
    v = ToolValidator()
    v.validate_permission(_EchoTool(), "user-1", {"echo", "calculator"})


def test_permission_denied_raises() -> None:
    v = ToolValidator()
    with pytest.raises(ToolDeniedError, match="does not have permission"):
        v.validate_permission(_EchoTool(), "user-1", {"calculator"})


def test_validate_schema_checked_before_permission() -> None:
    """Schema error must surface even when user has no permission."""
    v = ToolValidator()
    with pytest.raises(ToolDeniedError, match="required argument"):
        v.validate(_EchoTool(), {}, "user-1", set())


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


def test_registry_register_and_get() -> None:
    reg = ToolRegistry()
    reg.register(_EchoTool())
    tool = reg.get("echo")
    assert tool.name == "echo"


def test_registry_unknown_tool_raises() -> None:
    reg = ToolRegistry()
    with pytest.raises(KeyError):
        reg.get("nonexistent")


def test_registry_list_tools_sorted() -> None:
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    reg.register(_EchoTool())
    assert reg.list_tools() == ["calculator", "echo"]


# ---------------------------------------------------------------------------
# CalculatorTool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calculator_addition() -> None:
    result = await CalculatorTool().execute({"expression": "2 + 3"})
    assert result == 5


@pytest.mark.asyncio
async def test_calculator_complex_expression() -> None:
    result = await CalculatorTool().execute({"expression": "2 + 3 * 4 - 1"})
    assert result == 13


@pytest.mark.asyncio
async def test_calculator_power() -> None:
    result = await CalculatorTool().execute({"expression": "2 ** 10"})
    assert result == 1024


@pytest.mark.asyncio
async def test_calculator_float_division() -> None:
    result = await CalculatorTool().execute({"expression": "7 / 2"})
    assert result == 3.5


@pytest.mark.asyncio
async def test_calculator_negative_number() -> None:
    result = await CalculatorTool().execute({"expression": "-5 + 10"})
    assert result == 5


@pytest.mark.asyncio
async def test_calculator_rejects_code_injection() -> None:
    with pytest.raises(ToolDeniedError):
        await CalculatorTool().execute({"expression": "__import__('os')"})


@pytest.mark.asyncio
async def test_calculator_rejects_function_call() -> None:
    with pytest.raises(ToolDeniedError):
        await CalculatorTool().execute({"expression": "abs(-5)"})


@pytest.mark.asyncio
async def test_calculator_rejects_division_by_zero() -> None:
    with pytest.raises(ToolDeniedError, match="division by zero"):
        await CalculatorTool().execute({"expression": "10 / 0"})


@pytest.mark.asyncio
async def test_calculator_rejects_syntax_error() -> None:
    with pytest.raises(ToolDeniedError):
        await CalculatorTool().execute({"expression": "2 +"})


# ---------------------------------------------------------------------------
# HTTPTool domain checks (no network — just domain validation)
# ---------------------------------------------------------------------------


def test_http_tool_allowed_domain_passes() -> None:
    tool = HTTPTool(allowed_domains=["example.com"])
    tool._check_domain("https://example.com/path")


def test_http_tool_subdomain_passes() -> None:
    tool = HTTPTool(allowed_domains=["example.com"])
    tool._check_domain("https://api.example.com/v1/data")


def test_http_tool_disallowed_domain_raises() -> None:
    tool = HTTPTool(allowed_domains=["example.com"])
    with pytest.raises(ToolDeniedError, match="not in the allowed domains list"):
        tool._check_domain("https://evil.com/steal")


def test_http_tool_ssrf_localhost_blocked() -> None:
    tool = HTTPTool(allowed_domains=["example.com"])
    with pytest.raises(ToolDeniedError):
        tool._check_domain("http://localhost:8080/admin")


def test_http_tool_ssrf_metadata_blocked() -> None:
    tool = HTTPTool(allowed_domains=["example.com"])
    with pytest.raises(ToolDeniedError):
        tool._check_domain("http://169.254.169.254/latest/meta-data/")
