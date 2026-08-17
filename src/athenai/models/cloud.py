"""
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
    """Anthropic Claude adapter via httpx async client."""

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
        payload: dict[str, Any] = {
            "model": request.model_name or self.model_name,
            "max_tokens": request.max_tokens,
            "messages": request.messages,
        }
        if request.system:
            payload["system"] = request.system

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
        content = data["content"][0]["text"] if data.get("content") else ""
        usage = data.get("usage", {})

        return ModelResponse(
            content=content,
            model_name=data.get("model", self.model_name),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            finish_reason=data.get("stop_reason", "stop"),
        )

    async def health_check(self) -> bool:
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
