"""Shared role\x00content encoding for passing role-tagged chat turns through
athenai.context.engine.ContextEngine's plain list[str] buckets.

HOW IT FITS IN THE SYSTEM:
Used by both athenai.runtime.pipeline.AthenaRuntime and
athenai.agents.executor.AgentExecutor wherever they build ContextEngine's
memory_fn/conversation inputs and need to reconstruct role-tagged message
dicts from ContextPacker's output afterward.

WHY NOT ConversationMemory's OWN "{role}: {content}" TEXT FORMAT:
That format is fine while you still have the MemoryEntry object (role is
also in .metadata, the real ground truth), but becomes ambiguous once
ContextPacker has flattened everything down to bare ContextItem.content
strings: there's no way to tell "a message that happens to start with the
text 'user: '" apart from "a message whose role is user" using text-prefix
matching alone. NUL can't appear in real text, so partitioning on it is
unambiguous.
"""

from __future__ import annotations


def encode_role_content(role: str, content: str, name: str | None = None) -> str:
    """`name` is optional (only tool-role messages carry one) -- encoded as
    a second \\x00-delimited field so AgentExecutor's tool-result messages
    (which need `name` to identify which tool produced a result) round-trip
    through ContextEngine same as AthenaRuntime's plain role+content ones."""
    return f"{role}\x00{name or ''}\x00{content}"


def decode_role_content(encoded: str) -> dict[str, str]:
    role, _, rest = encoded.partition("\x00")
    name, _, content = rest.partition("\x00")
    result = {"role": role, "content": content}
    if name:
        result["name"] = name
    return result
