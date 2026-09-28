"""The HTTP contract. Three harnesses parse these responses."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from dspy_jev.service.app import REQUEST_ID_HEADER, create_app
from tests._support import gate_evidence, stub_lm, triage_evidence

pytestmark = pytest.mark.contract


@pytest.fixture
def client(settings, allow_lm, triage_lm, configured_dspy):
    with TestClient(create_app(settings, lm=allow_lm)) as test_client:
        yield test_client


@pytest.fixture
def blocking_client(settings, block_lm, configured_dspy):
    with TestClient(create_app(settings, lm=block_lm)) as test_client:
        yield test_client


PAYLOAD = {"task": "fix a test", "proposed_action": "read tests/test_x.py", "context": "local checkout"}


# --- ops ------------------------------------------------------------------------


def test_healthz_reports_model_and_calibration(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["harness"] == "pi"
    assert body["model"] == "dummy"
    assert body["checks"] == {"lm": True, "calibration": False}


def test_readyz_is_503_without_a_model(settings, configured_dspy):
    bare = settings.model_copy(update={"ollama_api_key": None})
    with TestClient(create_app(bare)) as client:
        assert client.get("/readyz").status_code == 503
        assert client.get("/healthz").status_code == 200  # liveness still answers


def test_metrics_are_prometheus_shaped(client):
    client.post("/v1/decide", json=PAYLOAD)
    body = client.get("/metrics").text
    assert "# TYPE dspy_jev_http_requests_total counter" in body
    assert "dspy_jev_decisions_total" in body


# --- decide ---------------------------------------------------------------------


def test_decide_returns_the_full_decision_record(client):
    body = client.post("/v1/decide", json=PAYLOAD).json()
    assert body["allow"] is True
    assert body["route"] == "auto_execute"
    assert body["reasons"] == []
    assert set(body["decisions"]) == {"safe_to_proceed", "risk", "blast_radius", "reversible", "route"}
    assert body["decisions"]["safe_to_proceed"]["probability"] == pytest.approx(0.95)
    assert body["outputs"]["rationale"]


def test_decide_holds_an_unsafe_action_with_reasons(blocking_client):
    body = blocking_client.post("/v1/decide", json=PAYLOAD).json()
    assert body["allow"] is False
    assert body["route"] == "block"
    assert body["reasons"]


def test_a_held_decision_is_still_http_200(blocking_client):
    """ "Hold" is a successful decision, not a server error."""
    assert blocking_client.post("/v1/decide", json=PAYLOAD).status_code == 200


def test_per_call_threshold_can_only_tighten(settings, configured_dspy):
    with TestClient(create_app(settings, lm=stub_lm(gate_evidence(safe=0.90)))) as client:
        assert client.post("/v1/decide", json={**PAYLOAD, "autonomy_threshold": 0.99}).json()["allow"] is False
        # 0.10 is below the service floor of 0.85, so it must be ignored, not honoured.
        loosened = client.post("/v1/decide", json={**PAYLOAD, "autonomy_threshold": 0.10}).json()
        assert loosened["metadata"]["autonomy_threshold"] == pytest.approx(0.85)


def test_missing_required_field_is_422(client):
    assert client.post("/v1/decide", json={"task": "t"}).status_code == 422


def test_unknown_field_is_rejected(client):
    """extra='forbid': a client typo must not be silently dropped."""
    assert client.post("/v1/decide", json={**PAYLOAD, "urgency": "high"}).status_code == 422


def test_empty_action_is_rejected(client):
    assert client.post("/v1/decide", json={**PAYLOAD, "proposed_action": ""}).status_code == 422


def test_out_of_range_threshold_is_rejected(client):
    assert client.post("/v1/decide", json={**PAYLOAD, "autonomy_threshold": 1.5}).status_code == 422


def test_decide_is_503_when_no_model_is_configured(settings, configured_dspy):
    bare = settings.model_copy(update={"ollama_api_key": None})
    with TestClient(create_app(bare)) as client:
        response = client.post("/v1/decide", json=PAYLOAD)
    assert response.status_code == 503
    assert "OLLAMA_API_KEY" in response.json()["detail"]


def test_a_provider_failure_becomes_502_with_a_request_id(settings, configured_dspy):
    class ExplodingProgram:
        autonomy_threshold = 0.85
        max_risk_level = 1
        max_blast_level = 1
        gate = type("G", (), {"fields": {}, "demos": []})()

        def load_calibration(self, *a, **k):
            return False

        def decide(self, **kwargs):
            raise RuntimeError("upstream 500")

    with TestClient(
        create_app(settings, lm=stub_lm(gate_evidence()), gate=ExplodingProgram()),
        raise_server_exceptions=False,
    ) as client:
        response = client.post("/v1/decide", json=PAYLOAD)
    assert response.status_code == 502
    assert response.json()["request_id"]


# --- triage ---------------------------------------------------------------------


def test_triage_returns_evidence(settings, configured_dspy):
    with TestClient(create_app(settings, lm=stub_lm(triage_evidence(category="billing")))) as client:
        body = client.post("/v1/triage", json={"ticket": "charged twice"}).json()
    assert body["decisions"]["category"]["value"] == "billing"
    assert body["decisions"]["urgent"]["kind"] == "noul"


# --- cross-cutting --------------------------------------------------------------


def test_request_id_is_echoed_when_supplied(client):
    response = client.post("/v1/decide", json=PAYLOAD, headers={REQUEST_ID_HEADER: "trace-42"})
    assert response.headers[REQUEST_ID_HEADER] == "trace-42"
    assert response.json()["request_id"] == "trace-42"


def test_request_id_is_generated_when_absent(client):
    assert client.post("/v1/decide", json=PAYLOAD).json()["request_id"]


def test_api_key_is_required_when_configured(settings, allow_lm, configured_dspy):
    secured = settings.model_copy(update={"service_api_key": SecretStr("s3cret")})
    with TestClient(create_app(secured, lm=allow_lm)) as client:
        assert client.post("/v1/decide", json=PAYLOAD).status_code == 401
        assert client.post("/v1/decide", json=PAYLOAD, headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.post("/v1/decide", json=PAYLOAD, headers={"X-API-Key": "s3cret"}).status_code == 200


def test_ops_endpoints_stay_open_for_probes(settings, allow_lm, configured_dspy):
    secured = settings.model_copy(update={"service_api_key": SecretStr("s3cret")})
    with TestClient(create_app(secured, lm=allow_lm)) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/metrics").status_code == 200


def test_no_response_body_contains_the_api_key(settings, allow_lm, configured_dspy):
    secured = settings.model_copy(update={"service_api_key": SecretStr("s3cret")})
    with TestClient(create_app(secured, lm=allow_lm)) as client:
        body = client.post("/v1/decide", json=PAYLOAD, headers={"X-API-Key": "s3cret"}).text
        health = client.get("/healthz").text
    assert "s3cret" not in body
    assert "s3cret" not in health
