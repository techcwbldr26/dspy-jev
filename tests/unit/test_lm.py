"""Building the LM is where a misconfiguration becomes a bill or a leak."""

from __future__ import annotations

import logging

import pytest

from dspy_jev import models
from dspy_jev.config import Settings
from dspy_jev.lm import MissingCredentialError, build_lm, configure_dspy, describe_lm

pytestmark = pytest.mark.unit


def test_ollama_cloud_lm_targets_the_cloud_endpoint(settings: Settings):
    lm = build_lm("decision", settings=settings)
    assert lm.model == "openai/glm-5.3"
    assert lm.kwargs["api_base"] == "https://ollama.com/v1/"
    assert lm.kwargs["api_key"] == "test-key"


def test_anthropic_lm_needs_no_api_base(claude_settings: Settings):
    lm = build_lm("decision", settings=claude_settings)
    assert lm.model == "anthropic/claude-sonnet-5-5"
    assert "api_base" not in lm.kwargs


def test_missing_credentials_raise_with_an_actionable_hint():
    bare = Settings(harness="pi")
    with pytest.raises(MissingCredentialError, match="OLLAMA_API_KEY"):
        build_lm("decision", settings=bare)


def test_missing_anthropic_credentials_name_the_right_variable():
    with pytest.raises(MissingCredentialError, match="ANTHROPIC_API_KEY"):
        build_lm("decision", settings=Settings(harness="claude-code"))


def test_cache_is_on_by_default_because_reanchor_requires_it(settings: Settings):
    assert build_lm("decision", settings=settings).cache is True


def test_temperature_defaults_to_zero_for_reproducible_evidence(settings: Settings):
    assert build_lm("decision", settings=settings).kwargs["temperature"] == 0.0


@pytest.mark.parametrize("role", ["decision", "fast", "judge"])
def test_each_role_builds(settings: Settings, role: str):
    assert build_lm(role, settings=settings).model.startswith("openai/")


def test_explicit_model_wins_over_role(settings: Settings):
    assert build_lm("fast", settings=settings, model="kimi-k3").model == "openai/kimi-k3"


def test_cross_provider_model_warns_but_is_honoured(settings: Settings, caplog):
    """An operator asking for a specific model gets it, loudly."""
    settings = settings.model_copy(update={"anthropic_api_key": None})
    with caplog.at_level(logging.WARNING, logger="dspy_jev.lm"), pytest.raises(MissingCredentialError):
        build_lm("decision", settings=settings, model="claude-opus-5-5")
    assert any("belongs to provider" in record.message for record in caplog.records)


def test_unknown_model_is_rejected_before_any_request(settings: Settings):
    with pytest.raises(models.UnknownModelError):
        build_lm("decision", settings=settings, model="gpt-9-turbo")


def test_overrides_reach_the_client(settings: Settings):
    assert build_lm("decision", settings=settings, max_tokens=17).kwargs["max_tokens"] == 17


def test_describe_lm_never_leaks_the_key(settings: Settings):
    described = describe_lm(build_lm("decision", settings=settings))
    assert "api_key" not in described
    assert "test-key" not in str(described)
    assert described["model"] == "openai/glm-5.3"


def test_configure_dspy_installs_the_lm(settings: Settings, configured_dspy):
    import dspy

    lm = configure_dspy("decision", settings=settings)
    assert dspy.settings.lm is lm
