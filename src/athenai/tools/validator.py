"""ToolValidator — JSON Schema validation then permission check.

HOW IT FITS IN THE SYSTEM:
Not currently called by AgentExecutor (athenai.agents.executor), which
dispatches straight to tool.execute() — this is a standalone pre-flight
check intended for a caller sitting in front of the executor (e.g. an API
handler) that wants to reject a bad or unauthorized tool call before it
ever reaches the model loop.

WHY SCHEMA BEFORE PERMISSION:
Schema validation is a pure in-process dict traversal — zero I/O. Permission
checks may involve async lookups (policy engine, DB). Failing fast on schema
errors avoids unnecessary I/O and gives callers clearer error messages:
"argument 'expression' is required" is more actionable than "permission denied
for a call that would have failed anyway".
"""

from __future__ import annotations

from typing import Any

from athenai.core.exceptions import ToolDeniedError
from athenai.core.protocols import Tool

_TYPE_MAP: dict[str, type] = {
    "string": str,
    "number": (int, float),  # type: ignore[dict-item]
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _validate_schema(schema: dict[str, Any], arguments: dict[str, Any], tool_name: str) -> None:
    """Check required fields and, for recognized JSON Schema type names,
    argument types. Unknown properties and unrecognized type names are
    silently allowed through — this is a best-effort guard against obviously
    malformed calls, not a full JSON Schema validator.
    """
    props: dict[str, Any] = schema.get("properties", {})
    required: list[str] = schema.get("required", [])

    for field in required:
        if field not in arguments:
            raise ToolDeniedError(
                f"tool {tool_name!r}: required argument {field!r} missing"
            )

    for key, value in arguments.items():
        if key not in props:
            continue
        expected_type_name = props[key].get("type")
        if expected_type_name and expected_type_name in _TYPE_MAP:
            expected = _TYPE_MAP[expected_type_name]
            if not isinstance(value, expected):
                raise ToolDeniedError(
                    f"tool {tool_name!r}: argument {key!r} must be "
                    f"{expected_type_name}, got {type(value).__name__}"
                )


class ToolValidator:
    """Stateless pre-flight checks for a tool call: schema shape, then permission.

    The three methods are exposed separately (not just via validate()) so a
    caller can run only the check it needs — e.g. re-check permissions on a
    cached, already-schema-valid call without re-validating arguments.
    """

    def validate_schema(self, tool: Tool, arguments: dict[str, Any]) -> None:
        _validate_schema(tool.input_schema, arguments, tool.name)

    def validate_permission(
        self, tool: Tool, user_id: str, allowed_tools: set[str]
    ) -> None:
        if tool.name not in allowed_tools:
            raise ToolDeniedError(
                f"user {user_id!r} does not have permission to use tool {tool.name!r}"
            )

    def validate(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        user_id: str,
        allowed_tools: set[str],
    ) -> None:
        self.validate_schema(tool, arguments)
        self.validate_permission(tool, user_id, allowed_tools)
