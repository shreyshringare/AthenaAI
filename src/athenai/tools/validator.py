"""ToolValidator — JSON Schema validation then permission check.

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
