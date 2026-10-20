"""
CloudModel — Anthropic Claude adapter used when a real API key is configured.

HOW IT FITS IN THE SYSTEM:
gateway/app.py builds this instead of MockModel when ANTHROPIC_API_KEY is
set in the environment; if construction fails it logs and falls back to
MockModel so the gateway still boots. AgentExecutor and AthenaRuntime call
generate() the same way regardless of which adapter is active.

TOOL USE: generate() forwards request.metadata["tools"] (a list of
{name, description, input_schema} dicts — exactly ToolRegistry.get_schemas()'s
shape, which AgentExecutor now passes through) as the Anthropic API
request's "tools" field, and translates any tool_use content blocks in the
response into ModelResponse.metadata["tool_calls"] (the {"name",
"arguments"} shape AgentExecutor's structured path expects). Without the
request side, Claude never emits native tool_use blocks regardless of
response-parsing — the API only offers tool use when the request declares
what tools exist. AgentExecutor's TOOL_CALL: text-protocol system-prompt
instructions stay in place as a fallback for MockModel/LocalModel and any
response that comes back as plain text instead.

ERROR SEMANTICS:
generate() never returns a partial/failed ModelResponse — network errors,
timeouts, 429s, and 4xx/5xx status codes all raise (ModelUnavailableError
or RateLimitError) so callers can apply a single retry/circuit-breaker
policy instead of checking response fields for failure.

WHY httpx.AsyncClient OVER ANTHROPIC SDK:
httpx gives us direct control over timeouts, connection pooling, retry hooks,
and status codes. The SDK abstracts these away, making it harder to integrate
with our resilience layer (circuit breaker, retry policy). respx can mock httpx
in tests without patching internals.

WHY NOT REQUESTS:
requests is synchronous — it blocks the event loop. All I/O in AthenaAI is async.
"""

from __future__ import annotations

from typing import Any

import httpx

from athenai.core.exceptions import ModelUnavailableError, RateLimitError
from athenai.models.base import ModelRequest, ModelResponse

_ANTHROPIC_API = "https://api.anthropic.com/v1/messages"
_DEFAULT_TIMEOUT = 60.0


class CloudModel:
    """Anthropic Claude adapter via httpx async client.

    Holds one long-lived httpx.AsyncClient for connection pooling across
    calls. Callers own its lifetime and must call aclose() on shutdown to
    release it — core/lifecycle.py's lifespan() does this automatically for
    any registered component exposing aclose(), but gateway/app.py's own
    lifespan (the one actually wired up today) does not yet call it.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "claude-sonnet-4-6",
        base_url: str = _ANTHROPIC_API,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Call the Anthropic Messages API and return the reply.

        Raises RateLimitError on HTTP 429 and ModelUnavailableError for
        timeouts, network errors, auth failures, and any other non-2xx
        status — never returns a response object for a failed call.
        Note: request.temperature is not forwarded to the API payload.
        """
        payload: dict[str, Any] = {
            "model": request.model_name or self.model_name,
            "max_tokens": request.max_tokens,
            "messages": request.messages,
        }
        if request.system:
            payload["system"] = request.system
        tools = request.metadata.get("tools")
        if tools:
            payload["tools"] = tools

        try:
            resp = await self._client.post(self.base_url, json=payload)
        except httpx.TimeoutException as exc:
            raise ModelUnavailableError(f"Anthropic API timeout: {exc}") from exc
        except httpx.RequestError as exc:
            raise ModelUnavailableError(f"Anthropic API network error: {exc}") from exc

        if resp.status_code == 429:
            raise RateLimitError("Anthropic rate limit exceeded (429)")

        if resp.status_code in (401, 403):
            raise ModelUnavailableError(f"Anthropic auth error ({resp.status_code})")

        if resp.status_code >= 500:
            raise ModelUnavailableError(
                f"Anthropic server error ({resp.status_code}): {resp.text[:200]}"
            )

        if resp.status_code >= 400:
            raise ModelUnavailableError(
                f"Anthropic API error ({resp.status_code}): {resp.text[:200]}"
            )

        data = resp.json()
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in data.get("content", []):
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "tool_use":
                tool_calls.append(
                    {"name": block.get("name", ""), "arguments": block.get("input", {})}
                )
        usage = data.get("usage", {})

        return ModelResponse(
            content="".join(text_parts),
            model_name=data.get("model", self.model_name),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            finish_reason=data.get("stop_reason", "stop"),
            metadata={"tool_calls": tool_calls} if tool_calls else {},
        )

    async def health_check(self) -> bool:
        """True unless the API is unreachable or returning 5xx.

        Deliberately lenient: a 401/403/404 still counts as "healthy"
        because it proves the endpoint is reachable — this check is for
        service availability, not credential validity.
        """
        try:
            resp = await self._client.get(
                "https://api.anthropic.com/v1/models",
                timeout=5.0,
            )
            return resp.status_code < 500
        except Exception:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()
