"""
ModelRegistry: config-driven model lookup.

WHY CONFIG-DRIVEN:
Model selection at startup (not at request time) means routing decisions
are fast lookups, not dynamic instantiation. Rolling to a new model is a
config change — no code change required.
"""

from __future__ import annotations

from typing import Any

from athenai.models.cloud import CloudModel
from athenai.models.local import LocalModel
from athenai.models.mock import MockModel

# Supported model types
_MODEL_TYPES: dict[str, type[Any]] = {
    "mock": MockModel,
    "cloud": CloudModel,
    "local": LocalModel,
}


def _build_model(config: dict[str, str]) -> MockModel | CloudModel | LocalModel:
    model_type = config.get("type", "mock")
    name = config.get("name", "unknown")

    if model_type == "mock":
        return MockModel(name=name)

    if model_type == "cloud":
        api_key = config.get("api_key", "")
        base_url = config.get("base_url", "https://api.anthropic.com/v1/messages")
        return CloudModel(api_key=api_key, model_name=name, base_url=base_url)

    if model_type == "local":
        base_url = config.get("base_url", "http://localhost:11434")
        return LocalModel(model_name=name, base_url=base_url)

    raise ValueError(f"Unknown model type: {model_type!r}. Supported: {list(_MODEL_TYPES)}")


class ModelRegistry:
    """Config-driven registry mapping role names to model instances."""

    def __init__(self, registry_config: dict[str, dict[str, str]]) -> None:
        self._models: dict[str, MockModel | CloudModel | LocalModel] = {}
        for role, config in registry_config.items():
            self._models[role] = _build_model(config)

    def register(self, role: str, model: MockModel | CloudModel | LocalModel) -> None:
        self._models[role] = model

    def get(self, role: str) -> MockModel | CloudModel | LocalModel:
        try:
            return self._models[role]
        except KeyError:
            raise KeyError(
                f"No model registered for role {role!r}. Available: {self.list_roles()}"
            ) from None

    def list_roles(self) -> list[str]:
        return list(self._models.keys())

    async def health_check(self, role: str) -> bool:
        model = self.get(role)
        return await model.health_check()

    async def health_check_all(self) -> dict[str, bool]:
        results: dict[str, bool] = {}
        for role, model in self._models.items():
            results[role] = await model.health_check()
        return results
