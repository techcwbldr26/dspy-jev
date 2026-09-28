"""The MCP tool surface. Claude Code and Pi's mcp-adapter read these names,
descriptions and schemas, so they are a contract too."""

from __future__ import annotations

import json

import pytest
import respx
from httpx import Response

from dspy_jev.config import Settings
from dspy_jev.mcp_server import INSTRUCTIONS, _HttpCaller, build_parser

pytest.importorskip("mcp", reason="pip install 'dspy-jev[mcp]'")

from dspy_jev.mcp_server import _build_server

pytestmark = pytest.mark.contract

SERVICE_URL = "http://service.test:8080"


@pytest.fixture
def server(settings: Settings):
    """Proxy mode: no model is built, so this stays an offline contract test."""
    return _build_server(settings, SERVICE_URL)


async def test_the_three_tools_are_advertised(server):
    assert {tool.name for tool in await server.list_tools()} == {
        "jev_decide",
        "jev_triage",
        "jev_status",
    }


async def test_every_tool_has_a_routing_description(server):
    for tool in await server.list_tools():
        assert tool.description and len(tool.description) > 40, tool.name


async def test_decide_declares_its_arguments(server):
    tool = next(t for t in await server.list_tools() if t.name == "jev_decide")
    # The SDK spells this inputSchema (v1) or input_schema (v2); accept either.
    dumped = tool.model_dump(mode="json")
    schema = dumped.get("input_schema") or dumped["inputSchema"]
    assert set(schema.get("required", [])) == {"task", "proposed_action"}
    assert set(schema["properties"]) == {"task", "proposed_action", "context", "autonomy_threshold"}


def test_instructions_tell_the_agent_when_to_call_and_when_not_to():
    lowered = INSTRUCTIONS.lower()
    assert "before taking an action" in lowered
    assert "do not call it" in lowered
    for route in ("needs_review", "clarify", "block"):
        assert route in lowered


@respx.mock
async def test_proxy_mode_forwards_decide_to_the_service(settings: Settings):
    route = respx.post(f"{SERVICE_URL}/v1/decide").mock(
        return_value=Response(200, json={"allow": False, "route": "block", "reasons": ["route=block"]})
    )
    result = await _HttpCaller(SERVICE_URL, settings).decide("t", "rm -rf /", "prod", None)

    assert result["allow"] is False
    assert json.loads(route.calls.last.request.content) == {
        "task": "t",
        "proposed_action": "rm -rf /",
        "context": "prod",
    }


@respx.mock
async def test_proxy_mode_sends_the_service_api_key(settings: Settings):
    from pydantic import SecretStr

    secured = settings.model_copy(update={"service_api_key": SecretStr("s3cret")})
    route = respx.post(f"{SERVICE_URL}/v1/decide").mock(return_value=Response(200, json={"allow": True}))
    await _HttpCaller(SERVICE_URL, secured).decide("t", "a", "", None)
    assert route.calls.last.request.headers["X-API-Key"] == "s3cret"


@respx.mock
async def test_proxy_mode_surfaces_service_errors(settings: Settings):
    import httpx

    respx.post(f"{SERVICE_URL}/v1/decide").mock(return_value=Response(503, json={"error": "no model"}))
    with pytest.raises(httpx.HTTPStatusError):
        await _HttpCaller(SERVICE_URL, settings).decide("t", "a", "", None)


@respx.mock
async def test_status_reports_proxy_mode(settings: Settings):
    respx.get(f"{SERVICE_URL}/healthz").mock(return_value=Response(200, json={"status": "ok", "calibrated": True}))
    status = await _HttpCaller(SERVICE_URL, settings).status()
    assert status["mode"] == "proxy"
    assert status["service_url"] == SERVICE_URL
    assert status["calibrated"] is True


def test_cli_defaults_to_stdio_for_local_clients():
    assert build_parser().parse_args([]).transport == "stdio"


def test_cli_offers_streamable_http_for_pi():
    args = build_parser().parse_args(["--transport", "streamable-http", "--port", "8099"])
    assert args.transport == "streamable-http"
    assert args.port == 8099
