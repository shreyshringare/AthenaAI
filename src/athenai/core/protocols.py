"""
WHY PROTOCOL OVER ABC:
Protocols enable structural subtyping — a class satisfies a Protocol simply
by having the right methods, without inheriting from it. This decouples the
component graph: MockModel satisfies Model without knowing about it, which
makes test doubles trivial and prevents circular imports.

WHY NOT ABC:
ABCs require explicit inheritance, which creates a coupling between interface
and implementation. On the hot-path (model calls, cache lookups), structural
dispatch is also faster than MRO traversal.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Model(Protocol):
    """Synchronous-style async model adapter."""

    async def generate(self, request: Any) -> Any:
        ...

    async def health_check(self) -> bool:
        ...


@runtime_checkable
class StreamingModel(Protocol):
    """Model adapter that supports token streaming."""

    async def generate(self, request: Any) -> Any:
        ...

    async def stream(self, request: Any) -> AsyncIterator[str]:
        ...

    async def health_check(self) -> bool:
        ...


@runtime_checkable
class MemoryStore(Protocol):
    """Persistent or in-process memory backend."""

    async def add(self, key: str, value: Any, session_id: str) -> None:
        ...

    async def get(self, key: str, session_id: str) -> Any:
        ...

    async def clear(self, session_id: str) -> None:
        ...


@runtime_checkable
class Retriever(Protocol):
    """Document chunk retrieval interface."""

    async def search(
        self,
        query_embedding: list[float],
        k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Any]:
        ...


@runtime_checkable
class Tool(Protocol):
    """Executable tool with a JSON Schema input contract."""

    @property
    def name(self) -> str:
        ...

    @property
    def description(self) -> str:
        ...

    @property
    def input_schema(self) -> dict[str, Any]:
        ...

    async def execute(self, arguments: dict[str, Any]) -> Any:
        ...


@runtime_checkable
class CacheBackend(Protocol):
    """Generic async cache backend."""

    async def get(self, key: str) -> Any | None:
        ...

    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        ...

    async def delete(self, key: str) -> None:
        ...

    async def exists(self, key: str) -> bool:
        ...


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Text embedding interface."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...
