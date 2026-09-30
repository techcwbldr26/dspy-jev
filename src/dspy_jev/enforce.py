"""Enforcement: turn a verdict into a block.

Everything else in this package *advises*. This module is what a harness calls
from a real interception point -- a Claude Code ``PreToolUse`` hook, a Pi
``tool_call`` handler, a hermes-agent policy -- so that a held action does not
run at all.

Three responsibilities, in order:

1. **Triage** (:func:`classify`). Most tool calls are reads and in-scope edits.
   Gating those would spend a model call per keystroke and train the agent to
   ignore the answer. Only calls that could matter are sent to the gate.
2. **Rendering** (:func:`render_action`). A tool call is a name plus arguments;
   the gate judges a sentence. The translation is here so all three harnesses
   describe the same action the same way.
3. **Decision** (:class:`Enforcer`). Call the gate and map its verdict onto the
   harness's allow/deny vocabulary.

The rule that matters more than any of it: **failure is a block.** An
unreachable service, a timeout, malformed JSON, a missing credential -- every
one of those denies the call. A gate that fails open is not a gate, and an
error is exactly when an attacker, or an unlucky agent, would most like one.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from dspy_jev.config import Settings, get_settings

logger = logging.getLogger(__name__)

DEFAULT_TASK = "No task was recorded for this session."


class Disposition(str, Enum):
    """What the enforcer decided to do with a tool call."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class Reason(str, Enum):
    """Why :func:`classify` routed a call the way it did."""

    READ_ONLY_TOOL = "read_only_tool"
    READ_ONLY_COMMAND = "read_only_command"
    IN_SCOPE = "in_scope"
    NEEDS_GATE = "needs_gate"
    ALWAYS_GATE = "always_gate"


# --- triage ---------------------------------------------------------------------

#: Tool names that only ever read. Gating these is pure cost.
#: Deliberately a small allowlist: an unrecognised tool is gated, not skipped.
READ_ONLY_TOOLS = frozenset(
    {
        "read",
        "glob",
        "grep",
        "search",
        "notebookread",
        "websearch",
        "todoread",
        "listmcpresources",
        "readmcpresource",
        "askuserquestion",
        "exitplanmode",
    }
)

#: Tool names always worth a gate call, whatever their arguments.
ALWAYS_GATE_TOOLS = frozenset({"webfetch", "bash", "powershell", "shell", "execute", "run"})

#: Shell commands that only read. Matched on argv[0] (and argv[1] for git-likes).
READ_ONLY_COMMANDS = frozenset(
    {
        "cat",
        "head",
        "tail",
        "less",
        "ls",
        "ll",
        "pwd",
        "wc",
        "file",
        "stat",
        "du",
        "df",
        "grep",
        "rg",
        "egrep",
        "fgrep",
        "find",
        "fd",
        "which",
        "whereis",
        "type",
        "env",
        "printenv",
        "date",
        "uname",
        "whoami",
        "id",
        "echo",
        "true",
        "diff",
        "cmp",
        "jq",
        "yq",
        "sort",
        "uniq",
        "cut",
        "awk",
        "sed",
        "tree",
        "basename",
        "dirname",
        "realpath",
    }
)

#: Read-only subcommands of tools whose safety depends on the subcommand.
READ_ONLY_SUBCOMMANDS: dict[str, frozenset[str]] = {
    "git": frozenset({"status", "log", "diff", "show", "branch", "remote", "blame", "ls-files", "describe"}),
    "docker": frozenset({"ps", "images", "logs", "inspect", "version", "info"}),
    "kubectl": frozenset({"get", "describe", "logs", "version", "explain"}),
    "npm": frozenset({"ls", "list", "view", "outdated", "audit"}),
    "pip": frozenset({"list", "show", "freeze"}),
    "uv": frozenset({"tree", "version"}),
    "poetry": frozenset({"show", "check"}),
    "cargo": frozenset({"tree", "metadata"}),
    "go": frozenset({"list", "version", "env"}),
}

#: Patterns that force a gate even inside an otherwise read-only command line.
#: A redirect, a pipe into a shell, or a network fetch changes what the line does.
ESCALATING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?<![0-9<>])>{1,2}(?![&>])"),  # > or >> redirection
    re.compile(r"\|\s*(sudo\s+)?(ba|z|k|da)?sh\b"),  # piping into a shell
    re.compile(r"\b(curl|wget|nc|ncat|ssh|scp|rsync)\b"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\$\("),  # command substitution
    re.compile(r"`"),  # backtick substitution
    re.compile(r";|&&|\|\|"),  # chained commands
)

#: Argument keys carrying the command line, across harness vocabularies.
COMMAND_KEYS = ("command", "cmd", "script", "args", "commandLine")


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation, normalised across harnesses."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @property
    def normalised_tool(self) -> str:
        """Lowercase, with any MCP ``mcp__server__name`` prefix stripped."""
        name = self.tool.strip().lower()
        if name.startswith("mcp__"):
            name = name.rsplit("__", 1)[-1]
        return name

    @property
    def command(self) -> str:
        """The shell command line, when this call carries one."""
        for key in COMMAND_KEYS:
            value = self.arguments.get(key)
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, list) and value:
                return " ".join(str(part) for part in value)
        return ""


@dataclass(frozen=True)
class Classification:
    """Whether a call needs the gate, and why."""

    gate: bool
    reason: Reason

    @property
    def skipped(self) -> bool:
        return not self.gate


def _is_read_only_command(command: str) -> bool:
    """True only when every segment of the line is a known read-only command."""
    if not command.strip():
        return False
    if any(pattern.search(command) for pattern in ESCALATING_PATTERNS):
        return False
    try:
        argv = shlex.split(command)
    except ValueError:
        return False  # unbalanced quotes: cannot reason about it, so gate it
    if not argv:
        return False

    # Strip leading `VAR=value` assignments.
    while argv and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", argv[0]):
        argv = argv[1:]
    if not argv:
        return False

    program = argv[0].rsplit("/", 1)[-1]
    if program in READ_ONLY_SUBCOMMANDS:
        subcommands = READ_ONLY_SUBCOMMANDS[program]
        rest = [a for a in argv[1:] if not a.startswith("-")]
        return bool(rest) and rest[0] in subcommands
    return program in READ_ONLY_COMMANDS


def classify(call: ToolCall) -> Classification:
    """Decide whether ``call`` is worth a gate call.

    Conservative by construction: anything not positively recognised as
    read-only is gated.
    """
    tool = call.normalised_tool
    if tool in ALWAYS_GATE_TOOLS:
        if tool in {"bash", "powershell", "shell", "execute", "run"} and _is_read_only_command(call.command):
            return Classification(gate=False, reason=Reason.READ_ONLY_COMMAND)
        return Classification(gate=True, reason=Reason.ALWAYS_GATE)
    if tool in READ_ONLY_TOOLS:
        return Classification(gate=False, reason=Reason.READ_ONLY_TOOL)
    return Classification(gate=True, reason=Reason.NEEDS_GATE)


# --- rendering ------------------------------------------------------------------

#: Argument keys worth showing to the gate, in display order.
INTERESTING_KEYS = (
    "command",
    "cmd",
    "script",
    "file_path",
    "path",
    "url",
    "pattern",
    "old_string",
    "new_string",
    "content",
    "query",
    "prompt",
)
MAX_VALUE_CHARS = 600


def render_action(call: ToolCall) -> str:
    """Describe a tool call as the sentence the gate judges."""
    tool = call.tool.strip() or "unknown-tool"
    command = call.command
    if command:
        return f"Run shell command: {_truncate(command)}"

    parts: list[str] = []
    for key in INTERESTING_KEYS:
        value = call.arguments.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(f"{key}={_truncate(value)}")
    if not parts:
        # Unknown tool: show the argument keys rather than nothing, so the gate
        # can at least see the shape of what is being asked.
        parts = [f"arguments={sorted(call.arguments)}"] if call.arguments else ["no arguments"]
    return f"Use the {tool} tool with {'; '.join(parts)}"


def _truncate(value: str, limit: int = MAX_VALUE_CHARS) -> str:
    collapsed = " ".join(value.split())
    return collapsed if len(collapsed) <= limit else f"{collapsed[:limit]}… [{len(collapsed)} chars total]"


# --- decision -------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """What the harness should do, and what to tell the agent."""

    disposition: Disposition
    message: str
    gated: bool
    route: str = ""
    reasons: list[str] = field(default_factory=list)
    probability: float | None = None
    error: str = ""

    @property
    def blocked(self) -> bool:
        return self.disposition is not Disposition.ALLOW

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "message": self.message,
            "gated": self.gated,
            "route": self.route,
            "reasons": list(self.reasons),
            "probability": self.probability,
            "error": self.error,
        }


#: Route -> what the harness does with it. `clarify` and `needs_review` become
#: `ask` where the harness can prompt a person, and `deny` where it cannot.
ROUTE_DISPOSITION: dict[str, Disposition] = {
    "auto_execute": Disposition.ALLOW,
    "needs_review": Disposition.ASK,
    "clarify": Disposition.ASK,
    "block": Disposition.DENY,
}


class Enforcer:
    """Calls the gate for a tool call and returns a harness-agnostic verdict.

    Args:
        settings: configuration; defaults to the process settings.
        service_url: a running dspy-jev HTTP service. When omitted, the gate runs
            in this process, which needs provider credentials here.
        interactive: whether the harness can prompt a person. When it cannot,
            ``needs_review`` and ``clarify`` become a denial rather than a prompt,
            because nobody is there to answer.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        service_url: str | None = None,
        interactive: bool = True,
        timeout_s: float | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._service_url = service_url.rstrip("/") if service_url else None
        self._interactive = interactive
        self._timeout_s = timeout_s if timeout_s is not None else self._settings.request_timeout_s
        self._gate = None  # built lazily; in-process mode needs credentials

    # -- public ------------------------------------------------------------------
    def evaluate(self, call: ToolCall, *, task: str = DEFAULT_TASK, context: str = "") -> Verdict:
        """Classify, gate if needed, and return what the harness should do."""
        classification = classify(call)
        if classification.skipped:
            return Verdict(
                disposition=Disposition.ALLOW,
                message="",
                gated=False,
                route="auto_execute",
                reasons=[],
            )

        try:
            payload = self._ask_gate(task=task, proposed_action=render_action(call), context=context)
        except Exception as exc:  # every failure is a block
            logger.warning("enforce.gate_unavailable", extra={"error": f"{type(exc).__name__}: {exc}"})
            return Verdict(
                disposition=Disposition.DENY,
                message=(
                    "The dspy-jev decision gate could not be reached, so this action is held. "
                    "A gate that cannot answer is not permission. "
                    f"({type(exc).__name__}: {exc})"
                ),
                gated=True,
                error=f"{type(exc).__name__}: {exc}",
            )

        return self._verdict_from(payload)

    # -- internals ---------------------------------------------------------------
    def _verdict_from(self, payload: dict[str, Any]) -> Verdict:
        allow = bool(payload.get("allow", False))
        route = str(payload.get("route") or "block")
        reasons = [str(r) for r in payload.get("reasons") or []]
        rationale = str(payload.get("rationale") or "")
        probability = (payload.get("decisions") or {}).get("safe_to_proceed", {}).get("probability")

        disposition = ROUTE_DISPOSITION.get(route, Disposition.DENY)
        if not allow and disposition is Disposition.ALLOW:
            # The route said proceed but a policy condition failed. Policy wins.
            disposition = Disposition.ASK
        if disposition is Disposition.ASK and not self._interactive:
            disposition = Disposition.DENY

        if disposition is Disposition.ALLOW:
            return Verdict(disposition=disposition, message="", gated=True, route=route, probability=probability)
        return Verdict(
            disposition=disposition,
            message=self._explain(route, reasons, rationale, probability),
            gated=True,
            route=route,
            reasons=reasons,
            probability=probability,
        )

    def _explain(self, route: str, reasons: list[str], rationale: str, probability: float | None) -> str:
        headline = {
            "block": "The dspy-jev decision gate blocked this action.",
            "needs_review": "The dspy-jev decision gate held this action for human review.",
            "clarify": "The dspy-jev decision gate held this action: the task under-specifies it.",
        }.get(route, "The dspy-jev decision gate held this action.")

        lines = [headline]
        if probability is not None:
            lines.append(f"P(safe to proceed) = {probability:.2f}")
        if reasons:
            lines.append("Failing conditions: " + "; ".join(reasons))
        if rationale:
            lines.append(f"Rationale: {rationale}")
        lines.append(
            "Report this verdict to the user rather than rewording the action and retrying: "
            "the gate reads the action text."
        )
        return " ".join(lines)

    def _ask_gate(self, *, task: str, proposed_action: str, context: str) -> dict[str, Any]:
        if self._service_url:
            return self._ask_service(task=task, proposed_action=proposed_action, context=context)
        return self._ask_in_process(task=task, proposed_action=proposed_action, context=context)

    def _ask_service(self, *, task: str, proposed_action: str, context: str) -> dict[str, Any]:
        import urllib.error
        import urllib.request

        body = json.dumps({"task": task, "proposed_action": proposed_action, "context": context}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = self._settings.service_api_key
        if api_key:
            headers["X-API-Key"] = api_key.get_secret_value()

        request = urllib.request.Request(  # operator-configured URL
            f"{self._service_url}/v1/decide", data=body, headers=headers, method="POST"
        )
        with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def _ask_in_process(self, *, task: str, proposed_action: str, context: str) -> dict[str, Any]:
        if self._gate is None:
            import dspy

            from dspy_jev.lm import build_lm
            from dspy_jev.observability import install_audit_callback
            from dspy_jev.program import ActionGateProgram

            dspy.configure(lm=build_lm("decision", settings=self._settings))
            install_audit_callback(self._settings)
            gate = ActionGateProgram(settings=self._settings)
            gate.load_calibration()
            self._gate = gate

        decision = self._gate.decide(task=task, proposed_action=proposed_action, context=context)
        return {
            "allow": decision.allow,
            "route": decision.route,
            "reasons": decision.reasons,
            "rationale": decision.rationale,
            "decisions": decision.record["decisions"],
        }
