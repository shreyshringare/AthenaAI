"""
AthenaConfig — the single source of truth for runtime configuration.

HOW IT FITS IN THE SYSTEM:
Designed to be constructed once at process startup and threaded through
`athenai.core.lifecycle.lifespan()` into the component registry, so every
component (models, tools, resilience primitives) reads its settings from one
place instead of calling `os.environ` directly.

NOTE FOR NEWCOMERS: as of this writing, `athenai.gateway.app.create_app()`
does not construct or pass around an AthenaConfig — it reads `os.environ`
ad hoc and builds its own components directly. AthenaConfig and
`lifecycle.lifespan()` are the intended integration point for that wiring;
don't be surprised if you don't see AthenaConfig imported outside of
lifecycle.py and its tests yet.

WHY PYDANTIC SETTINGS:
AthenaConfig reads all values from environment variables with type coercion
and validation at startup — fail fast rather than discovering a missing API
key mid-request. BaseSettings also supports `.env` files for local dev with
no code changes.

WHY NOT PLAIN DATACLASS:
Dataclasses don't support env-variable binding or nested model validation.
Pydantic Settings gives us both with zero boilerplate.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AthenaConfig(BaseSettings):
    """Env-driven runtime settings, e.g. `ATHENA_LOG_LEVEL=DEBUG` sets `log_level`.

    Fields with dict/list defaults (model_registry, token_budget,
    allowed_http_domains) use `default_factory` so each AthenaConfig instance
    gets its own mutable object rather than sharing one across instances —
    Pydantic will still let you override them wholesale via a JSON-encoded
    env var (e.g. `ATHENA_TOKEN_BUDGET='{"total": 20000}'`).
    """

    model_config = SettingsConfigDict(
        env_prefix="ATHENA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    log_level: str = Field(default="INFO")
    environment: str = Field(default="development")

    # Database
    db_url: str = Field(default="postgresql://localhost:5432/athenai")
    db_pool_size: int = Field(default=10)

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0")

    # Anthropic
    anthropic_api_key: str = Field(default="")
    anthropic_base_url: str = Field(default="https://api.anthropic.com")

    # Model registry: maps role name → model config dict
    model_registry: dict[str, dict[str, str]] = Field(
        default_factory=lambda: {
            "default": {"type": "mock", "name": "mock-default"},
            "fast": {"type": "mock", "name": "mock-fast"},
            "reasoning": {"type": "mock", "name": "mock-reasoning"},
        }
    )

    # Token budget: maps bucket name → max tokens
    token_budget: dict[str, int] = Field(
        default_factory=lambda: {
            "system": 1000,
            "conversation": 4000,
            "memory": 1000,
            "rag": 3000,
            "tools": 500,
            "reserved": 500,
            "total": 10000,
        }
    )

    # Routing
    max_concurrent_model_calls: int = Field(default=10)

    # HTTP tool allowlist
    allowed_http_domains: list[str] = Field(default_factory=list)

    # Resilience
    circuit_breaker_failure_threshold: int = Field(default=5)
    circuit_breaker_cooldown_s: float = Field(default=30.0)
    retry_max_attempts: int = Field(default=3)
    retry_base_delay_s: float = Field(default=0.5)
    request_timeout_s: float = Field(default=30.0)

    # Cache
    response_cache_ttl_s: int = Field(default=300)

    # Agent
    agent_max_iterations: int = Field(default=10)
