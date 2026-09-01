"""HTTPTool — domain-allowlisted HTTP GET with 10s timeout.

HOW IT FITS IN THE SYSTEM:
Registered into a ToolRegistry (athenai.tools.registry) at gateway startup
only if ATHENA_ALLOWED_DOMAINS is configured (athenai.gateway.app), so an
"http_get" tool call from the model reaches AgentExecutor's dispatch
(athenai.agents.executor) only when an operator has opted in to outbound
network access.

WHY ALLOWLIST (NOT DENYLIST):
Denylist approaches require predicting every malicious target — impossible.
An allowlist requires explicit operator approval for each domain. This prevents
SSRF (server-side request forgery) where an agent is tricked into fetching
internal services (169.254.x, 10.x, metadata endpoints).

WHY 10s TIMEOUT:
Agent loops have a hard iteration budget. A slow external service blocking
indefinitely stalls the entire loop. 10s is generous for API calls (typical
p99 < 2s) while still protecting against hangs.
"""

from __future__ import annotations

from typing import Any, ClassVar
from urllib.parse import urlparse

import httpx

from athenai.core.exceptions import ToolDeniedError

_MAX_BODY_CHARS = 8192
_TIMEOUT_S = 10.0


class HTTPTool:
    """Tool protocol implementation for allowlisted outbound HTTP GET requests."""

    name = "http_get"
    description = "Perform an HTTP GET request to an allowlisted domain."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Full URL to fetch (must be on the allowed domain list)",
            },
            "headers": {
                "type": "object",
                "description": "Optional HTTP headers to include",
            },
        },
        "required": ["url"],
    }

    def __init__(self, allowed_domains: list[str]) -> None:
        self._allowed_domains: frozenset[str] = frozenset(allowed_domains)

    def _check_domain(self, url: str) -> None:
        """Raise ToolDeniedError unless the URL host matches an allowed domain
        exactly or is a subdomain of one (host.endswith("." + domain)) — so
        allowing "example.com" also covers "api.example.com" without needing
        every subdomain listed explicitly.
        """
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if not any(host == d or host.endswith("." + d) for d in self._allowed_domains):
            raise ToolDeniedError(
                f"domain {host!r} is not in the allowed domains list"
            )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        url = arguments["url"]
        self._check_domain(url)

        headers: dict[str, str] = arguments.get("headers", {})
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            response = await client.get(url, headers=headers, follow_redirects=True)

        return {
            "status": response.status_code,
            "url": str(response.url),
            "body": response.text[:_MAX_BODY_CHARS],
            "headers": dict(response.headers),
        }
