"""Global fixtures.

Two things every test needs and must not inherit from the developer's shell:
an isolated environment (no stray ``OLLAMA_API_KEY`` turning a unit test into a
live call) and an isolated DSPy configuration (``dspy.configure`` is global).
"""

from __future__ import annotations

import importlib
import os
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest

warnings.filterwarnings("ignore", category=UserWarning, module="dspy")

import dspy  # noqa: E402

from dspy_jev.config import Settings, reset_settings_cache  # noqa: E402
from dspy_jev.observability import METRICS  # noqa: E402
from tests._support import gate_evidence, stub_lm, triage_evidence  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Environment variables that would otherwise leak a real provider into a test.
LEAKY_ENV = (
    "OLLAMA_API_KEY",
    "OLLAMA_BASE_URL",
    "ANTHROPIC_API_KEY",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_ID",
)


def release_dspy_config_ownership() -> None:
    """Let the next test configure DSPy from whichever thread it happens to use.

    ``dspy.configure`` binds itself to the first thread that calls it. Starlette's
    ``TestClient`` runs the lifespan on a portal thread, so without this reset the
    second test in a module fails on ownership rather than on its own assertion.
    """
    module = importlib.import_module("dspy.dsp.utils.settings")
    module.config_owner_thread_id = None
    module.config_owner_async_task = None


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Clear provider and DSPY_JEV_* variables, and point artifacts at tmp_path."""
    for name in LEAKY_ENV:
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.startswith("DSPY_JEV_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)  # keeps Settings from reading the repo's .env
    # No test may reach for a tracking server; the MLflow path has its own tests.
    monkeypatch.setenv("DSPY_JEV_MLFLOW_ENABLED", "false")
    reset_settings_cache()
    release_dspy_config_ownership()
    METRICS.reset()
    yield
    reset_settings_cache()
    release_dspy_config_ownership()
    METRICS.reset()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Offline-safe settings for the Ollama Cloud (open-weight) harness family."""
    return Settings(
        harness="pi",
        OLLAMA_API_KEY="test-key",
        artifact_dir=tmp_path / "artifacts",
        data_dir=REPO_ROOT / "data",
        mlflow_enabled=False,
        log_format="text",
    )


@pytest.fixture
def claude_settings(tmp_path: Path) -> Settings:
    """Settings for the one harness that uses Claude rather than open weights."""
    return Settings(
        harness="claude-code",
        ANTHROPIC_API_KEY="test-key",
        artifact_dir=tmp_path / "artifacts",
        data_dir=REPO_ROOT / "data",
        mlflow_enabled=False,
        log_format="text",
    )


@pytest.fixture
def configured_dspy() -> Iterator[None]:
    """Reset the global DSPy configuration after each test that touches it."""
    yield
    release_dspy_config_ownership()
    dspy.configure(lm=None, callbacks=[])
    release_dspy_config_ownership()


@pytest.fixture
def allow_lm() -> dspy.LM:
    """A stub returning evidence for an obviously safe action."""
    return stub_lm(gate_evidence())


@pytest.fixture
def block_lm() -> dspy.LM:
    """A stub returning evidence for an obviously unsafe action."""
    return stub_lm(gate_evidence(safe=0.04, risk_level=4, blast_level=3, reversible=0.05, route="block"))


@pytest.fixture
def triage_lm() -> dspy.LM:
    return stub_lm(triage_evidence())


@pytest.fixture
def dataset_path() -> Path:
    return REPO_ROOT / "data" / "action_gate.jsonl"
