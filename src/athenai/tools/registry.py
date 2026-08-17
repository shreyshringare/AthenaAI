"""
ToolRegistry — maps tool names to Tool protocol implementations.

HOW IT FITS IN THE SYSTEM:
The ToolRegistry is the single source of truth for available tools at runtime.
The agent executor calls registry.get(name) to look up a tool by the name the
model returned. get_schemas() serialises all tool contracts into the format
sent to the model in the system prompt, so the model knows what tools exist.

WHY DICT NOT LIST:
O(1) name lookup matters when the agent loop calls tools many times per turn.
list() would require O(n) scan on every tool call.
"""

from __future__ import annotations

from typing import Any

from athenai.core.protocols import Tool


class ToolRegistry:
    """Runtime registry of available Tool implementations.

    Usage:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(HTTPTool(allowed_domains=["api.example.com"]))

        tool = registry.get("calculator")   # O(1) lookup
        schemas = registry.get_schemas()    # for model system prompt
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add a tool. Silently replaces if same name already registered."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return tool by name. Raises KeyError (not None) on miss."""
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"tool {name!r} not registered") from None

    def list_tools(self) -> list[str]:
        """Sorted list of registered tool names."""
        return sorted(self._tools)

    def get_schemas(self) -> list[dict[str, Any]]:
        """Serialised tool contracts — send this to the model as available tools."""
        return [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in self._tools.values()
        ]
