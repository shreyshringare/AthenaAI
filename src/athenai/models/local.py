"""LocalModel: Ollama adapter via httpx async."""

from __future__ import annotations

import httpx

from athenai.core.exceptions import ModelUnavailableError
from athenai.models.base import ModelRequest, ModelResponse

_DEFAULT_OLLAMA_URL = "http://localhost:11434"


class LocalModel:
    """
    WHY OLLAMA:
    Ollama exposes a simple HTTP API for local LLMs (Llama, Mistral, etc.).
    Same httpx-based pattern as CloudModel — consistent resilience wrapping,
    no special SDK dependency, same test mock strategy.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = _DEFAULT_OLLAMA_URL,
        timeout: float = 120.0,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self._client = httpx.AsyncClient(timeout=timeout)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        prompt_parts = []
        if request.system:
            prompt_parts.append(f"System: {request.system}")
        for msg in request.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role.capitalize()}: {content}")
        prompt = "\n".join(prompt_parts)

        payload = {
            "model": request.model_name or self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": request.temperature},
        }

        try:
            resp = await self._client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
        except httpx.RequestError as exc:
            raise ModelUnavailableError(f"Ollama unavailable: {exc}") from exc

        if resp.status_code >= 400:
            raise ModelUnavailableError(
                f"Ollama error ({resp.status_code}): {resp.text[:200]}"
            )

        data = resp.json()
        content = data.get("response", "")
        input_tokens = len(prompt) // 4
        output_tokens = len(content) // 4

        return ModelResponse(
            content=content,
            model_name=self.model_name,
            input_tokens=max(1, input_tokens),
            output_tokens=max(1, output_tokens),
            finish_reason="stop" if data.get("done") else "length",
        )

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get(f"{self.base_url}/api/tags", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()
