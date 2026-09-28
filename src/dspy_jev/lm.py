"""Build the ``dspy.LM`` for a role, for whichever harness is configured."""

from __future__ import annotations

import logging
from typing import Any

import dspy

from dspy_jev import models
from dspy_jev.config import Settings, get_settings

logger = logging.getLogger(__name__)


class MissingCredentialError(RuntimeError):
    """Raised when the provider for the active harness has no API key configured."""


def _credential_hint(provider: models.Provider) -> str:
    if provider == "ollama_cloud":
        return (
            "Set OLLAMA_API_KEY. Create a key at https://ollama.com/settings/keys, then "
            "`export OLLAMA_API_KEY=...` (or add it to .env)."
        )
    return "Set ANTHROPIC_API_KEY, or run inside a harness that injects it."


def build_lm(
    role: str = "decision",
    *,
    settings: Settings | None = None,
    model: str | None = None,
    **overrides: Any,
) -> dspy.LM:
    """Return a configured ``dspy.LM``.

    Ollama Cloud is addressed through its OpenAI-compatible endpoint, so the
    LiteLLM provider prefix is ``openai`` plus an ``api_base``. Anthropic models
    use the native prefix.

    Raises:
        MissingCredentialError: the provider family has no key configured.
        models.UnknownModelError: ``model`` is not in the registry.
    """
    settings = settings or get_settings()
    spec = models.resolve(model or settings.model_for(role))

    if spec.provider != settings.provider:
        logger.warning(
            "model %s belongs to provider %s but harness %s uses %s; honouring the explicit model",
            spec.name,
            spec.provider,
            settings.harness,
            settings.provider,
        )

    api_key = settings.api_key_for(spec.provider)
    if not api_key:
        raise MissingCredentialError(
            f"No API key for provider {spec.provider!r} (model {spec.name!r}). {_credential_hint(spec.provider)}"
        )

    kwargs: dict[str, Any] = {
        "model": spec.dspy_model,
        "api_key": api_key,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "cache": settings.cache,
        "num_retries": settings.num_retries,
        "timeout": settings.request_timeout_s,
    }
    if spec.provider == "ollama_cloud":
        kwargs["api_base"] = f"{settings.ollama_base_url}/"
    kwargs.update(overrides)

    logger.info(
        "building LM role=%s model=%s provider=%s cache=%s",
        role,
        spec.dspy_model,
        spec.provider,
        kwargs["cache"],
    )
    return dspy.LM(**kwargs)


def configure_dspy(
    role: str = "decision",
    *,
    settings: Settings | None = None,
    lm: dspy.LM | None = None,
    **overrides: Any,
) -> dspy.LM:
    """Build (or accept) an LM and install it as the DSPy default. Returns the LM."""
    settings = settings or get_settings()
    lm = lm or build_lm(role, settings=settings, **overrides)
    dspy.configure(lm=lm)
    return lm


def describe_lm(lm: dspy.LM) -> dict[str, Any]:
    """A credential-free description of an LM, safe to log or return over HTTP."""
    kwargs = getattr(lm, "kwargs", {}) or {}
    return {
        "model": getattr(lm, "model", None),
        "temperature": kwargs.get("temperature"),
        "max_tokens": kwargs.get("max_tokens"),
        "cache": getattr(lm, "cache", None),
        "api_base": kwargs.get("api_base"),
    }
