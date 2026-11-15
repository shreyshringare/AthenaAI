"""
RoutedModel — the live-path model that ties routing and resilience together.

HOW IT FITS IN THE SYSTEM:
`gateway/app.py` builds one RoutedModel from whichever backends are
configured (cloud Anthropic, local Ollama, or the mock fallback) and hands
it to AthenaRuntime and Agent in place of a single concrete model. It
implements the same `generate()/stream()/health_check()` shape as every
adapter in this package, so callers don't know routing exists.

Per request it does, in order:
  1. Estimate input tokens and ask ModelRouter for the best model. The router
     buckets the request into a LOW/MEDIUM/HIGH complexity tier by token
     count and ranks candidates by weighted quality/cost/latency score.
     Models whose circuit breaker is OPEN are skipped at this step.
  2. Spend one token from that model's outbound rate limiter.
  3. Call the model through its CircuitBreaker, wrapped in with_retry
     (exponential backoff + jitter). Retry sits *outside* the breaker so each
     failed attempt counts toward tripping it, and an OPEN circuit surfaces
     immediately as a non-retryable CircuitOpenError.
  4. If that model still fails (retries exhausted, circuit open, rate
     limited), exclude it and re-route to the next-best model. Only when no
     candidate is left does the last error propagate.

WHY REWRITE request.model_name:
Callers put a routing *role* ("default") or nothing in ModelRequest.model_name.
Forwarding that to a provider would send "default" as a model id, so the
chosen backend's own concrete name is substituted before the call.

STREAMING:
A stream can only fail over before its first token is emitted — after that,
the caller has already seen partial output. The first token is therefore
fetched inside the breaker/retry path; later failures are recorded against
the breaker and re-raised.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, cast

import httpx

from athenai.core.exceptions import (
    CircuitOpenError,
    ModelUnavailableError,
    RateLimitError,
)
from athenai.core.types import RoutingContext
from athenai.models.base import ModelRequest, ModelResponse
from athenai.observability.logger import get_logger
from athenai.observability.metrics import (
    model_fallbacks_total,
    model_latency_seconds,
    routed_requests_total,
)
from athenai.resilience.circuit_breaker import CircuitBreaker
from athenai.resilience.rate_limiter import TokenBucketRateLimiter
from athenai.resilience.retry import RetryPolicy, with_retry
from athenai.routing.policies import RoutingPolicy
from athenai.routing.router import ModelRouter
from athenai.routing.scorer import ModelMetadata

_logger = get_logger(__name__)

_CHARS_PER_TOKEN = 4

# Transient failures worth retrying against the same model. Circuit-open and
# local rate-limit errors are deliberately absent: those mean "try another
# model", not "try this one again".
_RETRYABLE: tuple[type[Exception], ...] = (
    ModelUnavailableError,
    httpx.TransportError,
    TimeoutError,
)


# Failures that mean "this backend can't serve the request right now" — the
# caller drops it and re-routes. Superset of _RETRYABLE (retries already ran).
_FALLBACK_ON: tuple[type[Exception], ...] = (CircuitOpenError, RateLimitError, *_RETRYABLE)


@dataclass
class Backend:
    """One routable model: the adapter, its scoring metadata, and the
    resilience state that guards it."""

    metadata: ModelMetadata
    model: Any
    breaker: CircuitBreaker
    limiter: TokenBucketRateLimiter | None = None


def estimate_tokens(request: ModelRequest) -> int:
    """Rough input-token estimate (chars / 4) — same heuristic the local and
    mock adapters use, good enough to pick a complexity tier."""
    chars = len(request.system) + sum(len(m.get("content", "")) for m in request.messages)
    return max(1, chars // _CHARS_PER_TOKEN)


class RoutedModel:
    """Model facade that routes each call across several backends."""

    def __init__(
        self,
        backends: list[Backend],
        policy: RoutingPolicy | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        if not backends:
            raise ValueError("RoutedModel needs at least one backend")
        self._backends = {b.metadata.role: b for b in backends}
        self._policy = policy or RoutingPolicy.default()
        self._retry = retry_policy or RetryPolicy()
        self._router = ModelRouter(
            model_catalog=[b.metadata for b in backends],
            circuit_breakers={b.metadata.role: b.breaker for b in backends},
        )

    @property
    def roles(self) -> list[str]:
        return list(self._backends)

    def circuit_states(self) -> dict[str, str]:
        """Role -> breaker state name (CLOSED/OPEN/HALF_OPEN), for status views."""
        return {role: b.breaker.state.value for role, b in self._backends.items()}

    def reset_circuits(self) -> None:
        for backend in self._backends.values():
            backend.breaker.reset()

    def _context(self, request: ModelRequest) -> RoutingContext:
        task = request.messages[-1].get("content", "") if request.messages else ""
        return RoutingContext(
            request_id="",
            user_id="",
            estimated_input_tokens=estimate_tokens(request),
            task_description=task[:200],
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        context = self._context(request)
        tried: set[str] = set()
        last_exc: Exception | None = None
        while True:
            try:
                decision = self._router.select(context, self._policy, frozenset(tried))
            except ModelUnavailableError:
                if last_exc is not None:
                    raise last_exc from None
                raise
            role = decision.selected_role
            backend = self._backends[role]
            routed_requests_total.labels(role=role, complexity=decision.complexity).inc()
            call_req = replace(request, model_name=_model_id(backend, request))
            start = time.perf_counter()
            try:
                if backend.limiter is not None:
                    await backend.limiter.acquire()
                response = await with_retry(
                    partial(_guarded_generate, backend, call_req),
                    self._retry,
                    retryable=_RETRYABLE,
                )
            except _FALLBACK_ON as exc:
                last_exc = exc
                tried.add(role)
                model_fallbacks_total.labels(role=role, reason=type(exc).__name__).inc()
                _logger.warning(
                    "routing.fallback", role=role, error=type(exc).__name__, detail=str(exc)
                )
                continue
            model_latency_seconds.labels(model=role).observe(time.perf_counter() - start)
            return response

    async def stream(self, request: ModelRequest) -> AsyncIterator[str]:
        context = self._context(request)
        tried: set[str] = set()
        last_exc: Exception | None = None
        while True:
            try:
                decision = self._router.select(context, self._policy, frozenset(tried))
            except ModelUnavailableError:
                if last_exc is not None:
                    raise last_exc from None
                raise
            role = decision.selected_role
            backend = self._backends[role]
            routed_requests_total.labels(role=role, complexity=decision.complexity).inc()
            call_req = replace(request, model_name=_model_id(backend, request))
            try:
                if backend.limiter is not None:
                    await backend.limiter.acquire()
                iterator, first = await with_retry(
                    partial(_guarded_open_stream, backend, call_req),
                    self._retry,
                    retryable=_RETRYABLE,
                )
            except _FALLBACK_ON as exc:
                last_exc = exc
                tried.add(role)
                model_fallbacks_total.labels(role=role, reason=type(exc).__name__).inc()
                _logger.warning(
                    "routing.fallback", role=role, error=type(exc).__name__, detail=str(exc)
                )
                continue
            break

        if first is not None:
            yield first
        try:
            async for token in iterator:
                yield token
        except Exception:
            await backend.breaker.record_failure()
            raise

    async def health_check(self) -> bool:
        """Healthy if at least one backend is — the router can still serve."""
        for backend in self._backends.values():
            if await backend.model.health_check():
                return True
        return False

    async def aclose(self) -> None:
        for backend in self._backends.values():
            close = getattr(backend.model, "aclose", None)
            if close is not None:
                await close()


async def _guarded_generate(backend: Backend, request: ModelRequest) -> ModelResponse:
    """One attempt: call the backend through its circuit breaker."""
    return cast(
        ModelResponse, await backend.breaker.call(lambda: backend.model.generate(request))
    )


async def _guarded_open_stream(
    backend: Backend, request: ModelRequest
) -> tuple[AsyncIterator[str], str | None]:
    return cast(
        tuple[AsyncIterator[str], str | None],
        await backend.breaker.call(lambda: _open_stream(backend.model, request)),
    )


def _model_id(backend: Backend, request: ModelRequest) -> str:
    return getattr(backend.model, "model_name", None) or request.model_name


async def _open_stream(model: Any, request: ModelRequest) -> tuple[AsyncIterator[str], str | None]:
    """Start a stream and pull its first token so a dead backend fails here,
    inside the breaker/retry path, not after the caller has seen output.
    Backends without stream() (CloudModel) degrade to one full-response chunk.
    """
    if not hasattr(model, "stream"):
        response = await model.generate(request)
        return _empty(), response.content
    iterator = model.stream(request).__aiter__()
    try:
        return iterator, await iterator.__anext__()
    except StopAsyncIteration:
        return iterator, None


async def _empty() -> AsyncIterator[str]:
    return
    yield  # pragma: no cover - makes this an async generator
