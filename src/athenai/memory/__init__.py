"""
Memory subsystem: durable and in-process state for agent conversations.

HOW IT FITS IN THE SYSTEM:
An agent turn needs access to several kinds of "memory," each with different
durability and retrieval characteristics:

  - base.py         — MemoryType/MemoryEntry: the shared vocabulary every
                       memory layer returns, so callers can handle results
                       uniformly regardless of which layer produced them.
  - conversation.py — ConversationMemory: durable, ordered recent-message
                       log per session (PostgreSQL).
  - semantic.py      — SemanticMemory: durable, similarity-searchable facts
                       (PostgreSQL + pgvector).
  - summary.py       — SummaryMemory: wraps ConversationMemory and collapses
                       old messages into a model-generated summary once a
                       session grows past a threshold, so callers reading
                       "recent" history get bounded size regardless of how
                       long the conversation has run.
  - working.py       — AgentState: mutable, in-process-only scratch state for
                       a single in-flight agent run (steps taken, tool
                       results so far) — not persisted, not shared.

None of these classes are imported directly by athenai.context.engine; that
package takes memory access as an injected async callable (`memory_fn`)
instead, so callers wire whichever memory layer(s) they need (e.g. a
ConversationMemory.get_recent or SummaryMemory.get_recent) without this
package needing to depend on the context engine or vice versa.
"""
