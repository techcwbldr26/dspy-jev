"""The hooks, run the way the harness runs them.

The Claude Code hook is a script Claude Code executes with JSON on stdin, and
`dspy-jev guard` is a command a shell runs. Testing them by importing a function
would skip the part that actually enforces. These run the real process against a
stub HTTP gate and check both the decision and, for `guard`, whether the command
left a trace on disk.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK = REPO_ROOT / "harnesses" / "claude-code" / ".claude" / "hooks" / "dspy_jev_gate.py"

ALLOW = {"allow": True, "route": "auto_execute", "reasons": [], "decisions": {"safe_to_proceed": {"probability": 0.97}}}
BLOCK = {
    "allow": False,
    "route": "block",
    "reasons": ["risk level 4 > max 1"],
    "rationale": "Destroys a shared runner.",
    "decisions": {"safe_to_proceed": {"probability": 0.02}},
}
REVIEW = {
    "allow": False,
    "route": "needs_review",
    "reasons": ["not self-reversible"],
    "decisions": {"safe_to_proceed": {"probability": 0.41}},
}


class StubGate:
    """A minimal HTTP service that answers /v1/decide with a canned verdict."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # BaseHTTPRequestHandler dictates this name
                length = int(self.headers.get("Content-Length", 0))
                outer.requests.append(json.loads(self.rfile.read(length) or b"{}"))
                body = json.dumps(outer.payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):  # keep pytest output clean
                pass

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def __enter__(self) -> StubGate:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()


def run_hook(event: dict, *, service_url: str | None, env: dict | None = None):
    environment = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "DSPY_JEV_HARNESS": "claude-code",
        "DSPY_JEV_MLFLOW_ENABLED": "false",
        **({"DSPY_JEV_SERVICE_URL": service_url} if service_url else {}),
        **(env or {}),
    }
    completed = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(event),
        capture_output=True,
        text=True,
        env=environment,
        timeout=90,
        check=False,
    )
    stdout = completed.stdout.strip()
    return completed, (json.loads(stdout) if stdout else None)


def decision_of(payload) -> str | None:
    return None if payload is None else payload["hookSpecificOutput"]["permissionDecision"]


# --- Claude Code PreToolUse hook -------------------------------------------------


def test_the_hook_is_executable():
    assert HOOK.exists()
    assert HOOK.stat().st_mode & 0o111, "Claude Code runs this directly; it must be executable"


def test_a_read_produces_no_decision_and_never_calls_the_gate():
    with StubGate(BLOCK) as gate:
        completed, payload = run_hook(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Read",
                "tool_input": {"file_path": "/a/b.py"},
                "cwd": "/repo",
            },
            service_url=gate.url,
        )
    assert completed.returncode == 0
    assert payload is None, "a read must not spend a gate call"
    assert gate.requests == []


def test_a_blocked_action_is_denied_with_its_reasons():
    with StubGate(BLOCK) as gate:
        _, payload = run_hook(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Bash",
                "tool_input": {"command": "rm -rf /"},
                "cwd": "/repo",
            },
            service_url=gate.url,
        )
    assert decision_of(payload) == "deny"
    reason = payload["hookSpecificOutput"]["permissionDecisionReason"]
    assert "risk level 4 > max 1" in reason
    assert "Destroys a shared runner." in reason
    assert gate.requests[0]["proposed_action"] == "Run shell command: rm -rf /"


def test_an_allowed_action_produces_no_decision():
    """Allow means 'nothing to add', so Claude Code's own rules still apply."""
    with StubGate(ALLOW) as gate:
        _, payload = run_hook(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Bash",
                "tool_input": {"command": "rm build/x"},
                "cwd": "/repo",
            },
            service_url=gate.url,
        )
    assert payload is None
    assert len(gate.requests) == 1


def test_needs_review_escalates_to_the_user():
    with StubGate(REVIEW) as gate:
        _, payload = run_hook(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Bash",
                "tool_input": {"command": "kubectl apply -f k8s/"},
                "cwd": "/repo",
            },
            service_url=gate.url,
        )
    assert decision_of(payload) == "ask"


def test_an_unreachable_gate_denies():
    """The rule the hook exists for."""
    _, payload = run_hook(
        {"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": "rm -rf /"}, "cwd": "/repo"},
        service_url="http://127.0.0.1:9",
    )
    assert decision_of(payload) == "deny"
    assert "not permission" in payload["hookSpecificOutput"]["permissionDecisionReason"]


def test_malformed_hook_input_denies():
    completed = subprocess.run(
        [sys.executable, str(HOOK)],
        input="{not json",
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO_ROOT / "src")},
        timeout=60,
        check=False,
    )
    assert decision_of(json.loads(completed.stdout)) == "deny"


def test_an_unimportable_package_denies():
    """A broken install must not silently disable the gate."""
    completed = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps({"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": "rm -rf /"}}),
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "DSPY_JEV_SRC": "/nonexistent", "PYTHONPATH": "/nonexistent"},
        cwd="/tmp",
        timeout=60,
        check=False,
    )
    assert decision_of(json.loads(completed.stdout)) == "deny"


def test_the_hook_always_exits_zero():
    """A non-zero exit is a hook *error*; the decision travels in the JSON."""
    for service_url in (None, "http://127.0.0.1:9"):
        completed, _ = run_hook(
            {"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": "rm -rf /"}},
            service_url=service_url,
        )
        assert completed.returncode == 0, completed.stderr


def test_the_mcp_server_source_is_recorded_in_the_context():
    """Trust follows the declared source, not the server's chosen name."""
    with StubGate(BLOCK) as gate:
        run_hook(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "mcp__totally_safe__deploy",
                "tool_input": {"target": "prod"},
                "cwd": "/repo",
                "mcp_server": {"name": "totally_safe", "source": "project"},
            },
            service_url=gate.url,
        )
    context = gate.requests[0]["context"]
    assert "source='project'" in context


# --- dspy-jev guard --------------------------------------------------------------


def run_guard(args: list[str], *, service_url: str | None):
    environment = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "DSPY_JEV_HARNESS": "pi",
        "DSPY_JEV_MLFLOW_ENABLED": "false",
        "OLLAMA_API_KEY": "test-key",
        **({"DSPY_JEV_SERVICE_URL": service_url} if service_url else {}),
    }
    return subprocess.run(
        [sys.executable, "-m", "dspy_jev.cli", "guard", *args],
        capture_output=True,
        text=True,
        env=environment,
        timeout=90,
        check=False,
    )


def test_guard_executes_only_after_an_allow(tmp_path: Path):
    marker = tmp_path / "allowed"
    with StubGate(ALLOW) as gate:
        completed = run_guard(["--task", "t", "--", "touch", str(marker)], service_url=gate.url)
    assert completed.returncode == 0, completed.stderr
    assert marker.exists(), "an allowed command must actually run"
    assert len(gate.requests) == 1


def test_guard_does_not_run_a_blocked_command(tmp_path: Path):
    marker = tmp_path / "blocked"
    with StubGate(BLOCK) as gate:
        completed = run_guard(["--task", "t", "--", "touch", str(marker)], service_url=gate.url)
    assert completed.returncode == 10
    assert not marker.exists(), "a blocked command must leave no trace"
    payload = json.loads(completed.stdout)
    assert payload["ran"] is False
    assert payload["disposition"] == "deny"


def test_guard_does_not_run_when_the_gate_is_unreachable(tmp_path: Path):
    marker = tmp_path / "unreachable"
    completed = run_guard(["--task", "t", "--", "touch", str(marker)], service_url="http://127.0.0.1:9")
    assert completed.returncode == 1
    assert not marker.exists()
    assert json.loads(completed.stdout)["ran"] is False


def test_guard_denies_needs_review_because_nobody_can_answer(tmp_path: Path):
    marker = tmp_path / "review"
    with StubGate(REVIEW) as gate:
        completed = run_guard(["--task", "t", "--", "touch", str(marker)], service_url=gate.url)
    assert completed.returncode == 10
    assert not marker.exists()


def test_guard_passes_through_a_read_only_command_without_gating(tmp_path: Path):
    with StubGate(BLOCK) as gate:
        completed = run_guard(["--task", "t", "--", "echo", "hello"], service_url=gate.url)
    assert completed.returncode == 0
    assert "hello" in completed.stdout
    assert gate.requests == [], "a read-only command must not spend a gate call"


def test_guard_returns_the_commands_own_exit_code():
    """42 is neither guard's hold (10) nor its error (1), so there is no ambiguity."""
    with StubGate(ALLOW) as gate:
        completed = run_guard(["--task", "t", "--", "sh", "-c", "exit 42"], service_url=gate.url)
    assert completed.returncode == 42
    assert len(gate.requests) == 1


def test_guard_requires_a_command():
    completed = run_guard(["--task", "t"], service_url=None)
    assert completed.returncode == 1
    assert json.loads(completed.stdout)["error"] == "NothingToRun"
