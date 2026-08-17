"""Custom exceptions for the AthenaAI runtime."""


class AthenaError(Exception):
    """Base exception for all AthenaAI errors."""


class ContextOverflowError(AthenaError):
    """
    WHY HARD CEILING:
    Silent truncation is a worse failure mode than a loud error. If context
    silently drops memory or RAG chunks, the model hallucinates without the
    caller knowing why. A hard ceiling forces the caller to make an explicit
    decision about what to sacrifice.

    WHY NOT SILENT TRUNCATION:
    Truncation masks bugs in token estimation and leads to non-deterministic
    behavior that is nearly impossible to reproduce in testing.
    """


class ModelUnavailableError(AthenaError):
    """Raised when no healthy model can serve the request."""


class ToolDeniedError(AthenaError):
    """Raised when a tool call is rejected due to schema or permission failure."""


class ToolTimeoutError(AthenaError):
    """Raised when a tool call exceeds its configured time budget."""


class PolicyViolationError(AthenaError):
    """Raised when a request violates content or usage policy."""


class EmbeddingError(AthenaError):
    """Raised when embedding generation fails."""


class CircuitOpenError(AthenaError):
    """Raised when a circuit breaker is in the OPEN state."""


class RateLimitError(AthenaError):
    """Raised when a rate limiter bucket is exhausted."""
