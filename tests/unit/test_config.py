"""Settings decide which provider a harness talks to. Get this wrong and a
'Claude Code only' exception silently becomes the rule."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from dspy_jev.config import Settings, get_settings, reset_settings_cache

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("harness", "provider"),
    [("hermes-agent", "ollama_cloud"), ("pi", "ollama_cloud"), ("claude-code", "anthropic")],
)
def test_harness_determines_provider(harness, provider):
    assert Settings(harness=harness).provider == provider


def test_only_claude_code_uses_a_closed_model():
    from dspy_jev import models

    for harness in ("hermes-agent", "pi"):
        spec = models.resolve(Settings(harness=harness).model_for("decision"))
        assert spec.open_weight, f"{harness} must use open weights"
    assert not models.resolve(Settings(harness="claude-code").model_for("decision")).open_weight


def test_explicit_model_overrides_the_registry_default():
    assert Settings(harness="pi", decision_model="kimi-k3").model_for("decision") == "kimi-k3"


def test_environment_prefix_is_honoured(monkeypatch):
    monkeypatch.setenv("DSPY_JEV_HARNESS", "claude-code")
    monkeypatch.setenv("DSPY_JEV_AUTONOMY_THRESHOLD", "0.99")
    reset_settings_cache()
    loaded = get_settings(refresh=True)
    assert loaded.harness == "claude-code"
    assert loaded.autonomy_threshold == 0.99


def test_provider_keys_keep_their_conventional_names(monkeypatch):
    """An existing shell with OLLAMA_API_KEY set should just work."""
    monkeypatch.setenv("OLLAMA_API_KEY", "sk-abc")
    reset_settings_cache()
    assert get_settings(refresh=True).api_key_for("ollama_cloud") == "sk-abc"


def test_api_key_is_not_exposed_by_repr():
    rendered = repr(Settings(harness="pi", OLLAMA_API_KEY="super-secret"))
    assert "super-secret" not in rendered


def test_base_url_trailing_slash_is_normalised():
    assert Settings(harness="pi", OLLAMA_BASE_URL="https://ollama.com/v1/").ollama_base_url.endswith("/v1")


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_autonomy_threshold_is_bounded(value):
    with pytest.raises(ValidationError):
        Settings(harness="pi", autonomy_threshold=value)


def test_artifact_path_is_per_harness(tmp_path: Path):
    paths = {
        harness: Settings(harness=harness, artifact_dir=tmp_path).calibrated_artifact
        for harness in ("hermes-agent", "pi", "claude-code")
    }
    assert len(set(paths.values())) == 3, "each harness needs its own calibration artifact"


def test_overrides_bypass_the_process_cache():
    reset_settings_cache()
    cached = get_settings()
    assert get_settings(harness="claude-code") is not cached
    assert get_settings() is cached


def test_audit_payloads_defaults_to_off():
    """Decision inputs carry user data; logging bodies must be an explicit choice."""
    assert Settings().audit_payloads is False


# --- auth modes -------------------------------------------------------------------


def test_key_mode_sends_the_real_key():
    assert Settings(harness="pi", OLLAMA_API_KEY="sk-real").api_key_for() == "sk-real"


def test_key_mode_reports_nothing_when_there_is_no_key():
    """A missing key must not silently become a placeholder."""
    assert Settings(harness="pi").api_key_for() is None


def test_proxy_mode_sends_an_inert_placeholder():
    """The platform replaces the header; we only need the client to build a request."""
    from dspy_jev.config import PROXY_AUTH_PLACEHOLDER

    value = Settings(harness="pi", auth_mode="proxy").api_key_for()
    assert value == PROXY_AUTH_PLACEHOLDER
    assert "sk-" not in value, "the placeholder must not look like a real key"


def test_a_real_key_still_wins_in_proxy_mode():
    assert Settings(harness="pi", auth_mode="proxy", OLLAMA_API_KEY="sk-real").api_key_for() == "sk-real"


@pytest.mark.parametrize("mode", ["key", "proxy"])
def test_the_auth_mode_is_described_for_doctor(mode):
    described = Settings(harness="pi", auth_mode=mode).auth_description
    assert described
    assert ("platform" in described) == (mode == "proxy")


def test_proxy_mode_covers_anthropic_too():
    from dspy_jev.config import PROXY_AUTH_PLACEHOLDER

    assert Settings(harness="claude-code", auth_mode="proxy").api_key_for() == PROXY_AUTH_PLACEHOLDER


def test_auth_mode_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("DSPY_JEV_AUTH_MODE", "proxy")
    reset_settings_cache()
    assert get_settings(refresh=True).auth_mode == "proxy"


def test_an_unknown_auth_mode_is_rejected():
    with pytest.raises(ValidationError):
        Settings(harness="pi", auth_mode="trust-me")
