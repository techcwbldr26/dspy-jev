"""MCP server exposing the gate as tools.

Two deployment shapes, one implementation:

* **in-process** (default) -- the server builds the program itself. Best for
  stdio clients that already live on the same machine as the credentials, such
  as Claude Code and hermes-agent's local profile.
* **proxy** -- with ``--service-url``, every tool call forwards to a running
  HTTP service. Best when one calibrated artifact should serve several
  harnesses, which is the recommended topology.

Transports: ``stdio`` for Claude Code, ``streamable-http`` for Pi's
``pi-mcp-adapter``, which takes an HTTP ``url`` for a server.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

from dspy_jev import __version__
from dspy_jev.config import Settings, get_settings

logger = logging.getLogger(__name__)

INSTRUCTIONS = """\
Call `jev_decide` before taking an action whose safety is not already obvious:
anything that writes outside the current task's scope, leaves the machine,
spends money, or cannot be undone. It returns `allow`, a `route`, and the
probability evidence behind both. Treat `allow: false` as binding and follow
`route`: needs_review means ask a person, clarify means ask the requester, block
means do not do it. Do not call it for routine reads of files you already have
permission to read.
"""


def _build_server(settings: Settings, service_url: str | None):
    """Construct the MCP server, importing the SDK lazily.

    Supports the mcp 2.x ``MCPServer`` and the 1.x ``FastMCP`` name.
    """
    try:
        from mcp.server.mcpserver import MCPServer as _Server
    except ImportError:  # pragma: no cover - mcp 1.x fallback
        from mcp.server.fastmcp import FastMCP as _Server

    server = _Server(
        name="dspy-jev",
        title="dspy-jev decision gate",
        version=__version__,
        instructions=INSTRUCTIONS,
    )

    caller = _HttpCaller(service_url.rstrip("/"), settings) if service_url else _LocalCaller(settings)

    @server.tool(
        name="jev_decide",
        title="Gate a proposed action",
        description=(
            "Judge whether an agent may take one proposed action. Returns a calibrated "
            "allow/hold decision, a route (auto_execute | needs_review | clarify | block), "
            "the reasons any policy condition failed, and the probability evidence behind "
            "each judgement."
        ),
    )
    async def jev_decide(
        task: str,
        proposed_action: str,
        context: str = "",
        autonomy_threshold: float | None = None,
    ) -> dict[str, Any]:
        return await caller.decide(task, proposed_action, context, autonomy_threshold)

    @server.tool(
        name="jev_triage",
        title="Triage a support ticket",
        description="Score a support ticket for urgency, severity and category with calibrated thresholds.",
    )
    async def jev_triage(ticket: str) -> dict[str, Any]:
        return await caller.triage(ticket)

    @server.tool(
        name="jev_status",
        title="Decision gate status",
        description="Report the model, harness, calibration state and policy parameters currently in force.",
    )
    async def jev_status() -> dict[str, Any]:
        return await caller.status()

    return server


class _LocalCaller:
    """Runs the program in this process."""

    def __init__(self, settings: Settings) -> None:
        import dspy

        from dspy_jev.lm import build_lm
        from dspy_jev.observability import configure_observability, install_audit_callback
        from dspy_jev.program import ActionGateProgram, TicketTriageProgram

        configure_observability(settings)
        self._settings = settings
        self._lm = build_lm("decision", settings=settings)
        dspy.configure(lm=self._lm)
        install_audit_callback(settings)
        self._gate = ActionGateProgram(settings=settings)
        self._calibrated = self._gate.load_calibration()
        self._triage = TicketTriageProgram()

    async def decide(
        self, task: str, proposed_action: str, context: str, autonomy_threshold: float | None
    ) -> dict[str, Any]:
        gate = self._gate
        if autonomy_threshold is not None:
            gate.autonomy_threshold = max(float(autonomy_threshold), self._settings.autonomy_threshold)
        decision = gate.decide(task=task, proposed_action=proposed_action, context=context)
        payload = decision.to_dict()
        payload["decisions"] = payload["record"]["decisions"]
        payload["outputs"] = payload["record"]["outputs"]
        payload.pop("record")
        return payload

    async def triage(self, ticket: str) -> dict[str, Any]:
        from dspy_jev.decisions import decision_record

        return decision_record(self._triage(ticket=ticket))

    async def status(self) -> dict[str, Any]:
        from dspy_jev.lm import describe_lm

        return {
            "version": __version__,
            "harness": self._settings.harness,
            "mode": "local",
            "calibrated": self._calibrated,
            "autonomy_threshold": self._gate.autonomy_threshold,
            "fields": dict(self._gate.gate.fields),
            "lm": describe_lm(self._lm),
        }


class _HttpCaller:
    """Forwards to a running dspy-jev HTTP service."""

    def __init__(self, base_url: str, settings: Settings) -> None:
        self._base_url = base_url
        self._settings = settings

    def _headers(self) -> dict[str, str]:
        key = self._settings.service_api_key
        return {"X-API-Key": key.get_secret_value()} if key else {}

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        import httpx

        async with httpx.AsyncClient(timeout=self._settings.request_timeout_s) as client:
            response = await client.post(f"{self._base_url}{path}", json=payload, headers=self._headers())
            response.raise_for_status()
            return response.json()

    async def decide(
        self, task: str, proposed_action: str, context: str, autonomy_threshold: float | None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"task": task, "proposed_action": proposed_action, "context": context}
        if autonomy_threshold is not None:
            payload["autonomy_threshold"] = autonomy_threshold
        return await self._post("/v1/decide", payload)

    async def triage(self, ticket: str) -> dict[str, Any]:
        return await self._post("/v1/triage", {"ticket": ticket})

    async def status(self) -> dict[str, Any]:
        import httpx

        async with httpx.AsyncClient(timeout=self._settings.request_timeout_s) as client:
            response = await client.get(f"{self._base_url}/healthz", headers=self._headers())
            response.raise_for_status()
            return {"mode": "proxy", "service_url": self._base_url, **response.json()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dspy-jev-mcp", description="MCP server for the dspy-jev decision gate")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="stdio for Claude Code and local clients; streamable-http for Pi's pi-mcp-adapter.",
    )
    parser.add_argument("--host", default=None, help="Bind host for HTTP transports.")
    parser.add_argument("--port", type=int, default=None, help="Bind port for HTTP transports.")
    parser.add_argument(
        "--service-url",
        default=None,
        help="Proxy tool calls to a running dspy-jev HTTP service instead of running the program in-process.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - process entry point
    args = build_parser().parse_args(argv)
    settings = get_settings()
    server = _build_server(settings, args.service_url)
    kwargs: dict[str, Any] = {}
    if args.transport != "stdio":
        kwargs["host"] = args.host or settings.service_host
        kwargs["port"] = args.port or (settings.service_port + 1)
    server.run(transport=args.transport, **kwargs)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
