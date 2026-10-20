"""API-key authentication dependency for the gateway's business endpoints.

HOW IT FITS IN THE SYSTEM:
`require_api_key` is a FastAPI dependency attached per-route (not a blanket
middleware) to every business endpoint in athenai.gateway.routes
(/v1/chat*, /v1/agents/run, /v1/documents/*) — never to /health, /ready, or
/metrics, which liveness probes and Prometheus scrapers hit without
credentials. It reads the configured key off `request.app.state.gateway_api_key`
(set once at startup in athenai.gateway.app.lifespan from AthenaConfig)
rather than from env/config directly, so tests can set a key per-app
without touching process environment variables.

WHY "NO KEY CONFIGURED" MEANS "NO AUTH ENFORCED" (NOT "REJECT EVERYTHING"):
Matches this gateway's existing degrade-to-permissive pattern elsewhere
(_build_model falls back to MockModel, _maybe_add_document_loader disables
RAG) rather than introducing a new "crash/reject if unconfigured" rule —
the zero-config local demo must keep working unauthenticated. app.py logs
a loud warning at startup when this happens so it's a deliberate, visible
choice, not a silent gap.
"""

from __future__ import annotations

from fastapi import Header, HTTPException, Request


async def require_api_key(
    request: Request, x_api_key: str | None = Header(default=None)
) -> None:
    configured_key = getattr(request.app.state, "gateway_api_key", "")
    if not configured_key:
        return
    if x_api_key != configured_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
