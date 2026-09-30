"""Environment-driven settings, shared by the library, the service and the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from dspy_jev import models

Harness = Literal["hermes-agent", "pi", "claude-code"]

#: How the provider request gets authenticated.
#:
#: ``key``    -- this process holds the key and sends it (``OLLAMA_API_KEY``).
#: ``proxy``  -- the platform injects the ``Authorization`` header on the way out
#:               and this process never sees the secret. Used by Claude Code's
#:               cloud "API credentials", and by any egress proxy that does the
#:               same. A placeholder is still sent, because the OpenAI client
#:               refuses to build a request without one.
AuthMode = Literal["key", "proxy"]

#: Sent as the API key in ``proxy`` mode. Inert on purpose: if it ever reaches a
#: provider unchanged, the request fails loudly rather than succeeding oddly.
PROXY_AUTH_PLACEHOLDER = "header-injected-by-platform"

#: The Claude Code harness is the one documented exception to "open weights only".
HARNESS_PROVIDER: dict[str, models.Provider] = {
    "hermes-agent": "ollama_cloud",
    "pi": "ollama_cloud",
    "claude-code": "anthropic",
}

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All configuration for a dspy-jev process.

    Every field is settable from the environment with the ``DSPY_JEV_`` prefix,
    except the two provider credentials, which keep their conventional names so
    an existing shell already works.
    """

    model_config = SettingsConfigDict(
        env_prefix="DSPY_JEV_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- harness ---------------------------------------------------------------
    harness: Harness = Field(
        default="hermes-agent",
        description="Which harness this process serves. Decides the provider family.",
    )

    # --- providers -------------------------------------------------------------
    ollama_api_key: SecretStr | None = Field(default=None, alias="OLLAMA_API_KEY")
    ollama_base_url: str = Field(default=models.OLLAMA_CLOUD_BASE_URL, alias="OLLAMA_BASE_URL")
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    auth_mode: AuthMode = Field(
        default="key",
        description="'key' reads the provider key from the environment; 'proxy' expects the "
        "platform to inject the Authorization header and sends a placeholder.",
    )

    # --- model selection (empty string means "use the registry default") -------
    decision_model: str = ""
    fast_model: str = ""
    judge_model: str = ""

    # --- inference behaviour ---------------------------------------------------
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)
    num_retries: int = Field(default=3, ge=0, le=10)
    request_timeout_s: float = Field(default=90.0, gt=0)
    cache: bool = Field(
        default=True,
        description="DSPy response cache. ReAnchor needs it on to calibrate without re-billing.",
    )

    # --- decision defaults -----------------------------------------------------
    autonomy_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="P(safe_to_proceed) required before an agent may act without review.",
    )

    # --- artifacts -------------------------------------------------------------
    artifact_dir: Path = Field(default=REPO_ROOT / "artifacts")
    data_dir: Path = Field(default=REPO_ROOT / "data")

    # --- observability ---------------------------------------------------------
    mlflow_enabled: bool = True
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    mlflow_experiment: str = "dspy-jev"
    otel_enabled: bool = False
    otel_endpoint: str = "http://127.0.0.1:4318/v1/traces"
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"
    audit_payloads: bool = Field(
        default=False,
        description="Log full prompt/response bodies. Off by default: decisions carry user data.",
    )

    # --- service ---------------------------------------------------------------
    service_host: str = "127.0.0.1"
    service_port: int = Field(default=8080, gt=0, lt=65536)
    service_api_key: SecretStr | None = Field(
        default=None,
        description="When set, /v1/* requires this value in the X-API-Key header.",
    )

    @model_validator(mode="after")
    def _normalise(self) -> Settings:
        object.__setattr__(self, "ollama_base_url", self.ollama_base_url.rstrip("/"))
        return self

    # --- derived ---------------------------------------------------------------
    @property
    def provider(self) -> models.Provider:
        """The provider family implied by the configured harness."""
        return HARNESS_PROVIDER[self.harness]

    def model_for(self, role: str = "decision") -> str:
        """Registry name of the model serving ``role``."""
        override = {"decision": self.decision_model, "fast": self.fast_model, "judge": self.judge_model}.get(role, "")
        return override or models.default_for(self.provider, role)

    def api_key_for(self, provider: models.Provider | None = None) -> str | None:
        """The value to send as the API key, or ``None`` when nothing is available.

        In ``proxy`` mode this is a placeholder: the real credential is added to
        the request by the platform, and never enters this process.
        """
        provider = provider or self.provider
        secret = self.ollama_api_key if provider == "ollama_cloud" else self.anthropic_api_key
        if secret:
            return secret.get_secret_value()
        return PROXY_AUTH_PLACEHOLDER if self.auth_mode == "proxy" else None

    @property
    def auth_description(self) -> str:
        """One line for `doctor` and `/healthz`, naming where auth comes from."""
        if self.auth_mode == "proxy":
            return "platform-injected Authorization header (no key in this process)"
        return "key read from the environment"

    @property
    def calibrated_artifact(self) -> Path:
        """Where the calibrated program for this harness is saved."""
        return self.artifact_dir / f"action_gate.{self.harness}.json"


_cached: Settings | None = None


def get_settings(refresh: bool = False, **overrides: object) -> Settings:
    """Process-wide settings.

    ``overrides`` bypasses the cache, which is what tests and the CLI use to
    construct a one-off configuration without touching the environment.
    """
    global _cached
    if overrides:
        return Settings(**overrides)  # type: ignore[arg-type]
    if _cached is None or refresh:
        _cached = Settings()
    return _cached


def reset_settings_cache() -> None:
    """Drop the cached settings. Used by tests and by ``dspy-jev serve --reload``."""
    global _cached
    _cached = None
