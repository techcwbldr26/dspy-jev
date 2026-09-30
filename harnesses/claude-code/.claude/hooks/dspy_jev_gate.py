#!/usr/bin/env python3
"""Claude Code PreToolUse hook: enforce the dspy-jev decision gate.

Claude Code runs this before a matched tool call, passing the call as JSON on
stdin and reading a decision from stdout. That makes it a real control rather
than advice: a denied call does not run.

Contract (https://code.claude.com/docs/en/hooks):

    stdin  {"hook_event_name": "PreToolUse", "tool_name": ..., "tool_input": {...},
            "cwd": ..., "session_id": ...}
    stdout {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                   "permissionDecision": "allow"|"deny"|"ask",
                                   "permissionDecisionReason": "..."}}

Exit 0 with no output means "no decision" -- the call proceeds under Claude
Code's normal permission rules. That is what this hook emits for reads and
in-scope edits, so the gate is not consulted for work that never needed it.

**Failure denies.** An unreachable service, a timeout, a missing credential, a
crash in this script: all of them deny. A gate that cannot answer is not
permission, and an error is exactly when an attacker would most like one.

Install: see harnesses/claude-code/.claude/settings.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

EVENT = "PreToolUse"
EXIT_OK = 0


def _decide(permission: str, reason: str = "") -> dict:
    output = {"hookEventName": EVENT, "permissionDecision": permission}
    if reason:
        output["permissionDecisionReason"] = reason
    return {"hookSpecificOutput": output}


def _emit(payload: dict | None) -> int:
    if payload is not None:
        json.dump(payload, sys.stdout)
        sys.stdout.write("\n")
    return EXIT_OK


def _ensure_importable() -> None:
    """Allow running from a checkout without installing the package."""
    if os.environ.get("DSPY_JEV_SRC"):
        sys.path.insert(0, os.environ["DSPY_JEV_SRC"])
        return
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "src" / "dspy_jev" / "__init__.py"
        if candidate.exists():
            sys.path.insert(0, str(parent / "src"))
            return


def main() -> int:
    try:
        raw = sys.stdin.read()
        event = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as exc:
        return _emit(_decide("deny", f"dspy-jev gate: could not parse the hook input ({exc})."))

    tool_name = str(event.get("tool_name") or "")
    tool_input = event.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        tool_input = {"value": tool_input}

    _ensure_importable()
    try:
        from dspy_jev.config import get_settings
        from dspy_jev.enforce import Disposition, Enforcer, ToolCall
    except ImportError as exc:
        return _emit(
            _decide(
                "deny",
                "dspy-jev gate: the package is not importable, so the action cannot be checked "
                f"({exc}). Install it, or set DSPY_JEV_SRC to the repository's src directory.",
            )
        )

    settings = get_settings()
    context_parts = [f"Working directory: {event.get('cwd') or Path.cwd()}"]
    if event.get("mcp_server"):
        server = event["mcp_server"]
        # Trust the declared source, not the server's own name.
        context_parts.append(
            f"MCP server: name={server.get('name')!r} source={server.get('source')!r}"
        )
    if os.environ.get("DSPY_JEV_CONTEXT"):
        context_parts.append(os.environ["DSPY_JEV_CONTEXT"])

    enforcer = Enforcer(
        settings=settings,
        service_url=os.environ.get("DSPY_JEV_SERVICE_URL"),
        interactive=True,  # Claude Code can ask the user, so `ask` is meaningful here.
    )
    verdict = enforcer.evaluate(
        ToolCall(tool=tool_name, arguments=tool_input),
        task=os.environ.get("DSPY_JEV_TASK") or "The user's current request in this Claude Code session.",
        context="\n".join(context_parts),
    )

    if not verdict.gated:
        return _emit(None)  # never gated: stay out of Claude Code's way
    if verdict.disposition is Disposition.ALLOW:
        return _emit(None)  # gate agreed, but let normal permission rules still apply
    if verdict.disposition is Disposition.ASK:
        return _emit(_decide("ask", verdict.message))
    return _emit(_decide("deny", verdict.message))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # a crashing gate must still deny
        json.dump(
            _decide("deny", f"dspy-jev gate: the hook itself failed ({type(exc).__name__}: {exc})."),
            sys.stdout,
        )
        sys.stdout.write("\n")
        raise SystemExit(EXIT_OK) from exc
