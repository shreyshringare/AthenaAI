"""
athenai.runtime — top-level orchestration layer.

HOW IT FITS IN THE SYSTEM:
This package sits between the HTTP gateway (`athenai.gateway`) and the
model adapters (`athenai.models`). It exposes `AthenaRuntime`
(`athenai.runtime.pipeline`), which the gateway constructs once at startup
and calls per-request for single-turn (non-agentic) chat. Multi-step
tool-use flows go through `athenai.agents.agent.Agent` instead — that
class wraps a model directly and does not use this package.
"""

from __future__ import annotations
