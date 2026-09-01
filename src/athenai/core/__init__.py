"""
Foundational types, config, protocols, and exceptions shared across AthenaAI.

HOW IT FITS IN THE SYSTEM:
Every other package (agents, tools, models, rag, routing, resilience, gateway,
runtime) imports from here — it sits at the bottom of the dependency graph and
must never import from a sibling package, or the codebase gains an import
cycle. Concretely:
  - types.py: frozen dataclasses (AIRequest, AIResponse, Message, ...) that
    flow through the whole request pipeline.
  - protocols.py: structural interfaces (Model, Tool, MemoryStore, ...) that
    let components depend on behavior instead of concrete classes.
  - exceptions.py: the AthenaError hierarchy raised by tools, models, and
    resilience primitives and handled uniformly at the gateway boundary.
  - config.py: AthenaConfig, the env-driven settings object.
  - lifecycle.py: the startup/shutdown context manager for wiring components.

This module intentionally re-exports nothing — import from the specific
submodule (e.g. `from athenai.core.types import AIRequest`) rather than from
`athenai.core` directly.
"""
