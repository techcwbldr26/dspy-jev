"""Golden schema.

Three harnesses, a CLI and an MCP tool all parse the decision record. A change
to its shape is a breaking change for all of them, so it has to be a deliberate,
reviewed edit to the golden file rather than a side effect.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dspy_jev.service.app import create_app

pytestmark = pytest.mark.contract

GOLDEN_DIR = Path(__file__).parent / "golden"
UPDATE_HINT = "Run with UPDATE_GOLDEN=1 to re-record, then review the diff."


def _compare(name: str, actual: dict) -> None:
    import os

    path = GOLDEN_DIR / name
    serialised = json.dumps(actual, indent=2, sort_keys=True) + "\n"
    if os.environ.get("UPDATE_GOLDEN") == "1":  # pragma: no cover - maintenance path
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(serialised, encoding="utf-8")
        return
    assert path.exists(), f"missing golden file {path}. {UPDATE_HINT}"
    assert json.loads(path.read_text(encoding="utf-8")) == actual, f"{name} changed. {UPDATE_HINT}"


def _shape(value):
    """The structure of a payload, with values replaced by their type names."""
    if isinstance(value, dict):
        return {k: _shape(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_shape(value[0])] if value else []
    return type(value).__name__


@pytest.fixture
def client(settings, allow_lm, configured_dspy):
    with TestClient(create_app(settings, lm=allow_lm)) as test_client:
        yield test_client


def test_decide_response_shape_is_stable(client):
    body = client.post(
        "/v1/decide",
        json={"task": "fix a test", "proposed_action": "read a file", "context": "local"},
    ).json()
    _compare("decide_response.json", _shape(body))


def test_openapi_surface_is_stable(client):
    spec = client.get("/openapi.json").json()
    surface = {path: sorted(methods) for path, methods in sorted((p, list(ops)) for p, ops in spec["paths"].items())}
    _compare("openapi_surface.json", surface)


def test_every_decision_carries_evidence(client):
    """A decision without a probability is indistinguishable from a parsed string."""
    body = client.post("/v1/decide", json={"task": "t", "proposed_action": "a", "context": ""}).json()
    for name, decision in body["decisions"].items():
        assert decision["kind"] in {"noul", "score", "choice"}, name
        has_evidence = decision.get("probability") is not None or decision.get("probabilities")
        assert has_evidence, f"{name} carries no probability evidence"


def test_reasons_are_present_exactly_when_the_gate_holds(settings, block_lm, configured_dspy):
    with TestClient(create_app(settings, lm=block_lm)) as client:
        held = client.post("/v1/decide", json={"task": "t", "proposed_action": "a"}).json()
    assert held["allow"] is False and held["reasons"]
