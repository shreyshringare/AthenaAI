"""
Model backends — pluggable LLM adapters behind a common interface.

HOW IT FITS IN THE SYSTEM:
Everything above this package (AgentExecutor, AthenaRuntime, memory
summarization) talks to a model only through the `Model`/`StreamingModel`
Protocol defined in athenai.core.protocols: `generate()`, optionally
`stream()`, and `health_check()`. This package provides three
implementations of that shape:
  - mock.MockModel   — deterministic echo, no API key, used as the default
                        backend so the gateway boots and demos work offline.
  - cloud.CloudModel — real Anthropic Claude calls via httpx, used when
                        ANTHROPIC_API_KEY is set (see gateway/app.py).
  - local.LocalModel — Ollama-backed local inference.
registry.ModelRegistry wires named "roles" (e.g. "fast", "smart") to
configured instances of the above for callers that need more than one
model at a time.
base.py defines the shared ModelRequest/ModelResponse payload types that
all three adapters consume and produce, so callers can swap adapters
without changing call sites.
"""
