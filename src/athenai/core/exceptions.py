"""
Custom exceptions for the AthenaAI runtime.

HOW IT FITS IN THE SYSTEM:
Every subsystem raises its own subset of these instead of bare Exception, so
the gateway's top-level error handler and the agent loop can catch
`AthenaError` once and still branch on the concrete type when needed
(e.g. mapping ModelUnavailableError -> HTTP 503, ToolDeniedError -> HTTP 400).
Raised by: tools/validator.py, tools/calculator.py, tools/http.py, tools/sql.py
(ToolDeniedError), context/budget.py (ContextOverflowError), models/local.py,
models/cloud.py (ModelUnavailableError, RateLimitError), routing/router.py
(ModelUnavailableError), resilience/timeout.py (ToolTimeoutError),
resilience/rate_limiter.py (RateLimitError), resilience/circuit_breaker.py
(CircuitOpenError), rag/embedder.py (EmbeddingError).
"""


class AthenaError(Exception):
    """Base exception for all AthenaAI errors — catch this at subsystem
    boundaries (e.g. the gateway) to convert any internal failure into a
    well-formed error response without enumerating every subclass."""


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
