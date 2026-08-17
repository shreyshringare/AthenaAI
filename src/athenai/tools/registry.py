"""ToolRegistry — maps tool names to Tool protocol implementations."""

from __future__ import annotations

from typing import Any

from athenai.core.protocols import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"tool {name!r} not registered") from None

    def list_tools(self) -> list[str]:
        return sorted(self._tools)

    def get_schemas(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in self._tools.values()
        ]
