"""
AthenaAI — modular AI agent runtime: model routing, RAG, tools, agents,
observability.

HOW IT FITS IN THE SYSTEM:
This is the top-level package. It intentionally re-exports nothing —
AthenaAI is organised as independent subpackages (core, models, routing,
context, memory, rag, tools, agents, runtime, gateway, observability, resilience)
that each expose their own public API. Consumers import directly from the
subpackage they need, e.g. `from athenai.tools.registry import ToolRegistry`
or `from athenai.core.protocols import Tool`, rather than through this file.
`athenai.runtime` is the closest thing to a single entry point: it wires the
subpackages together behind one `execute(request)` call.

The only symbol defined here is `__version__`, kept in sync with the
`version` field in pyproject.toml.
"""

__version__ = "0.1.0"
