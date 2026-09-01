"""
HTTP gateway package.

HOW IT FITS IN THE SYSTEM:
This package is the outermost layer of AthenaAI — the FastAPI process that
external clients (curl, SDKs, the demo UI) talk to over HTTP. It has three
modules: app.py (application factory + startup wiring), routes.py (the actual
endpoint handlers), and schemas.py (the Pydantic request/response contracts).
Everything below this layer (runtime, agents, tools, RAG) is transport-agnostic
and knows nothing about HTTP; the gateway's job is purely to translate HTTP
in and out of those internal types.

Intentionally empty otherwise — submodules are imported directly
(e.g. `from athenai.gateway.routes import router`) rather than re-exported
here, so importing this package has no side effects.
"""
