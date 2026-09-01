"""
Context-window budgeting, ranking, and packing engine.

HOW IT FITS IN THE SYSTEM:
Every agent turn needs a single prompt assembled from several competing
sources — system instructions, conversation history, retrieved long-term
memory, RAG chunks, and prior tool results — that together may exceed the
model's context window. This package is the layer that decides what goes in
and what gets left out:

  - budget.py   — TokenBudgetManager: per-bucket token accounting with a hard
                  ceiling (raises rather than silently truncating).
  - ranking.py  — RelevanceRanker: scores candidate chunks by cosine
                  similarity to a query embedding so only the most relevant
                  ones are considered for inclusion.
  - packing.py  — ContextPacker: lays out items in fixed priority order
                  (system > conversation > memory > rag > tools), dropping
                  lowest-priority items first when the token ceiling is hit.
  - engine.py   — ContextEngine: the orchestrator. Fetches memory and RAG
                  content concurrently, converts everything into ContextItems,
                  and delegates to ContextPacker/TokenBudgetManager to produce
                  the final BuiltContext handed to the model.

Callers (e.g. the agent executor / runtime pipeline) depend on this package
so they never have to reason about token math or source-priority tradeoffs
themselves — they just supply content and get back something that is
guaranteed to fit.
"""
