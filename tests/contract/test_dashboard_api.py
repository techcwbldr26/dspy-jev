"""The dashboard's own API surface.

The console reads four endpoints. If any of them changes shape the page breaks
silently, so they get the same contract treatment as /v1/decide.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dspy_jev.service.app import DASHBOARD_HTML, create_app
from dspy_jev.service.dashboard import DecisionLog, threshold_tradeoff
from tests._support import gate_evidence, stub_lm

pytestmark = pytest.mark.contract

PAYLOAD = {"task": "free disk space", "proposed_action": "rm -rf /var/lib/docker", "context": "shared CI"}


@pytest.fixture
def client(settings, allow_lm, configured_dspy):
    with TestClient(create_app(settings, lm=allow_lm)) as test_client:
        yield test_client


# --- the page itself --------------------------------------------------------------


def test_the_dashboard_file_ships_with_the_package():
    assert DASHBOARD_HTML.exists(), "dashboard.html must be installed as package data"


def test_the_dashboard_is_served_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>Decision Gate</title>" in response.text


def test_the_dashboard_loads_nothing_from_a_cdn():
    """It has to work on a laptop with no network but a reachable gate."""
    html = DASHBOARD_HTML.read_text(encoding="utf-8")
    for marker in ("http://", "https://"):
        for line in html.splitlines():
            if marker in line and "xmlns" not in line and "w3.org" not in line:
                pytest.fail(f"dashboard reaches off-origin: {line.strip()[:110]}")


def test_the_dashboard_declares_dark_mode_both_ways():
    """A media query alone loses to the in-page toggle, and vice versa."""
    html = DASHBOARD_HTML.read_text(encoding="utf-8")
    assert "prefers-color-scheme: dark" in html
    assert ':root[data-theme="dark"]' in html
    assert ':root:not([data-theme="light"])' in html


# --- /v1/decisions ----------------------------------------------------------------


def test_the_feed_starts_empty(client):
    body = client.get("/v1/decisions").json()
    assert body == {"decisions": [], "total": 0}


def test_a_decision_lands_in_the_feed(settings, block_lm, configured_dspy):
    with TestClient(create_app(settings, lm=block_lm)) as client:
        client.post("/v1/decide", json=PAYLOAD)
        body = client.get("/v1/decisions").json()
    assert body["total"] == 1
    entry = body["decisions"][0]
    assert entry["allow"] is False
    assert entry["route"] == "block"
    assert entry["probability"] is not None
    assert entry["reasons"]


def test_the_feed_is_newest_first(client):
    for i in range(3):
        client.post("/v1/decide", json={**PAYLOAD, "proposed_action": f"action number {i}"})
    actions = [d["action"] for d in client.get("/v1/decisions").json()["decisions"]]
    assert actions[0].endswith("2")
    assert actions[-1].endswith("0")


def test_the_feed_is_bounded():
    """It is a debugging aid, not a log; it must not grow without limit."""
    log = DecisionLog(maxlen=3)

    class FakeDecision:
        allow, route, reasons, latency_ms = True, "auto_execute", [], 1.0
        record = {"decisions": {"safe_to_proceed": {"probability": 0.9}, "risk": {"level": 0}}}

    for i in range(10):
        log.record(request_id=str(i), action=f"a{i}", decision=FakeDecision())
    assert len(log) == 3
    assert [d["action"] for d in log.recent()] == ["a9", "a8", "a7"]


@pytest.mark.parametrize("limit", [0, 201])
def test_the_feed_limit_is_validated(client, limit):
    assert client.get(f"/v1/decisions?limit={limit}").status_code == 422


# --- /v1/calibration --------------------------------------------------------------


def test_calibration_reports_the_policy_in_force(client):
    body = client.get("/v1/calibration").json()
    assert body["calibrated"] is False
    assert body["fields"] == {}
    assert body["policy"]["autonomy_threshold"] == pytest.approx(0.85)
    assert body["policy"]["max_risk_level"] == 1
    assert body["report"] is None


def test_calibration_reports_fitted_parameters(settings, allow_lm, configured_dspy):
    from dspy_jev.program import ActionGateProgram

    gate = ActionGateProgram(settings=settings)
    gate.gate.fields["safe_to_proceed"] = {"threshold": 0.71}
    with TestClient(create_app(settings, lm=allow_lm, gate=gate)) as client:
        body = client.get("/v1/calibration").json()
    assert body["calibrated"] is True
    assert body["fields"]["safe_to_proceed"]["threshold"] == 0.71


# --- /v1/lens ---------------------------------------------------------------------


def test_the_lens_pairs_every_label_with_a_probability(client):
    body = client.get("/v1/lens").json()
    assert body["error"] is None
    assert len(body["rows"]) >= 30
    for row in body["rows"]:
        assert isinstance(row["label_safe"], bool)
        assert 0.0 <= row["probability"] <= 1.0
        assert row["action"]


def test_the_lens_reports_the_trade_off_at_the_policy_threshold(client):
    body = client.get("/v1/lens").json()
    assert body["threshold"] == pytest.approx(0.85)
    trade = body["tradeoff"]
    assert trade["safe"] + trade["unsafe"] == trade["total"]


def test_the_lens_trade_off_moves_with_the_threshold(settings, configured_dspy):
    """The whole point of the page: the cut changes the verdicts, the evidence does not."""
    with TestClient(create_app(settings, lm=stub_lm(gate_evidence(safe=0.6)))) as client:
        low = client.get("/v1/lens?threshold=0.1").json()
        high = client.get("/v1/lens?threshold=0.99").json()
    assert low["tradeoff"]["false_allows"] > high["tradeoff"]["false_allows"]
    assert high["tradeoff"]["false_holds"] > low["tradeoff"]["false_holds"]
    # Same evidence both times -- only the local threshold moved.
    assert [r["probability"] for r in low["rows"]] == [r["probability"] for r in high["rows"]]


def test_the_lens_is_503_without_a_model(settings, configured_dspy):
    bare = settings.model_copy(update={"ollama_api_key": None})
    with TestClient(create_app(bare)) as client:
        assert client.get("/v1/lens").status_code == 503


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_the_lens_threshold_is_validated(client, threshold):
    assert client.get(f"/v1/lens?threshold={threshold}").status_code == 422


def test_dashboard_endpoints_honour_the_api_key(settings, allow_lm, configured_dspy):
    from pydantic import SecretStr

    secured = settings.model_copy(update={"service_api_key": SecretStr("s3cret")})
    with TestClient(create_app(secured, lm=allow_lm)) as client:
        for path in ("/v1/decisions", "/v1/calibration", "/v1/lens"):
            assert client.get(path).status_code == 401, path
        assert client.get("/v1/decisions", headers={"X-API-Key": "s3cret"}).status_code == 200
        # The page itself is unauthenticated; it is inert without the API.
        assert client.get("/").status_code == 200


# --- the trade-off maths ----------------------------------------------------------


def test_trade_off_counts_both_error_kinds():
    rows = [
        {"label_safe": True, "probability": 0.9},  # allowed, correct
        {"label_safe": True, "probability": 0.2},  # held, a false hold
        {"label_safe": False, "probability": 0.95},  # allowed, a false allow
        {"label_safe": False, "probability": 0.1},  # held, correct
    ]
    assert threshold_tradeoff(rows, 0.5) == {
        "false_allows": 1,
        "false_holds": 1,
        "total": 4,
        "unsafe": 2,
        "safe": 2,
    }


def test_raising_the_threshold_trades_slips_for_friction():
    rows = [{"label_safe": s, "probability": p} for s, p in [(True, 0.9), (True, 0.6), (False, 0.7), (False, 0.3)]]
    loose, tight = threshold_tradeoff(rows, 0.2), threshold_tradeoff(rows, 0.95)
    assert loose["false_allows"] > tight["false_allows"]
    assert tight["false_holds"] > loose["false_holds"]


def test_a_missing_probability_counts_as_held_not_allowed():
    """Absent evidence must never read as permission."""
    rows = [{"label_safe": False, "probability": None}]
    assert threshold_tradeoff(rows, 0.5)["false_allows"] == 0
