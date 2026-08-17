"""Shared types for the tool system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    success: bool
    output: Any
    error: str | None = None
