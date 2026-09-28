"""The model registry is the project's promise about which models are used."""

from __future__ import annotations

import pytest

from dspy_jev import models

pytestmark = pytest.mark.unit


def test_every_ollama_cloud_model_is_open_weight():
    """The whole point of the Ollama Cloud path: state-of-the-art *open* weights."""
    closed = [s.name for s in models.OLLAMA_CLOUD_MODELS if not s.open_weight]
    assert closed == [], f"closed-weight models on the open-weight path: {closed}"


def test_anthropic_models_are_the_documented_exception():
    """Claude models exist in the registry only for the Claude Code harness."""
    assert all(spec.provider == "anthropic" for spec in models.ANTHROPIC_MODELS)
    assert all(not spec.open_weight for spec in models.ANTHROPIC_MODELS)


def test_ollama_cloud_uses_the_openai_compatible_prefix():
    """LiteLLM's ``ollama/`` prefix targets a local daemon; the cloud is OpenAI-compatible."""
    for spec in models.OLLAMA_CLOUD_MODELS:
        assert spec.dspy_model == f"openai/{spec.name}"
        assert not spec.dspy_model.startswith("ollama")


def test_anthropic_uses_the_native_prefix():
    for spec in models.ANTHROPIC_MODELS:
        assert spec.dspy_model == f"anthropic/{spec.name}"


@pytest.mark.parametrize("provider", ["ollama_cloud", "anthropic"])
@pytest.mark.parametrize("role", ["decision", "fast", "judge"])
def test_every_role_default_resolves_and_matches_its_provider(provider, role):
    spec = models.resolve(models.default_for(provider, role))
    assert spec.provider == provider


def test_resolve_accepts_aliases():
    assert models.resolve("gpt-oss").name == "gpt-oss:120b"


def test_resolve_rejects_unknown_models():
    with pytest.raises(models.UnknownModelError, match="Unknown model"):
        models.resolve("definitely-not-a-model")


def test_default_for_rejects_unknown_roles():
    with pytest.raises(ValueError, match="Unknown role"):
        models.default_for("ollama_cloud", "oracle")


def test_registry_has_no_duplicate_canonical_names():
    names = [s.name for s in (*models.OLLAMA_CLOUD_MODELS, *models.ANTHROPIC_MODELS)]
    assert len(names) == len(set(names))


def test_open_weight_names_preserve_declaration_order():
    names = models.open_weight_names()
    assert names[0] == "glm-5.3"
    assert len(names) == len(set(names))


def test_cloud_base_url_is_https_and_openai_shaped():
    assert models.OLLAMA_CLOUD_BASE_URL == "https://ollama.com/v1"
    assert models.OLLAMA_TAGS_URL.startswith("https://")


def test_registry_names_are_the_exact_cloud_tags():
    """The direct cloud API takes the name the listing returns, not the family.

    ``deepseek-v4-pro`` 404s; ``deepseek-v4-pro:0813`` is what is served. Any
    entry that needs a suffix must carry it here, with the bare spelling kept as
    an alias so obvious configuration still resolves.
    """
    needs_suffix = {"deepseek-v4-pro", "mistral-large-3", "gpt-oss", "gemma4", "nemotron-3-nano"}
    for spec in models.OLLAMA_CLOUD_MODELS:
        assert spec.name not in needs_suffix, f"{spec.name} needs its version suffix"


@pytest.mark.parametrize("bare", ["deepseek-v4-pro", "mistral-large-3", "gpt-oss"])
def test_the_bare_spelling_still_resolves_through_an_alias(bare):
    assert models.resolve(bare).name.startswith(bare)
    assert ":" in models.resolve(bare).name


def test_role_defaults_are_exact_tags():
    for role in ("decision", "fast", "judge"):
        name = models.default_for("ollama_cloud", role)
        assert models.resolve(name).name == name, f"{role} default is an alias, not the exact tag"
