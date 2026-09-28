"""The shipped harness assets are deliverables, not documentation.

A skill whose frontmatter does not parse never loads; an MCP config with a typo
fails silently at startup. These are cheap tests that catch both.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESSES = REPO_ROOT / "harnesses"

SKILLS = [
    HARNESSES / "pi" / "skills" / "dspy-jev" / "SKILL.md",
    HARNESSES / "claude-code" / ".claude" / "skills" / "dspy-jev" / "SKILL.md",
]
JSON_CONFIGS = sorted(HARNESSES.rglob("*.json"))


def parse_frontmatter(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{path} has no frontmatter"
    _, raw, body = text.split("---\n", 2)
    return yaml.safe_load(raw), body


def strip_comments(payload: dict) -> dict:
    """Drop the ``$comment`` keys these files use for operator notes."""
    return {k: v for k, v in payload.items() if not k.startswith("$")}


# --- skills ---------------------------------------------------------------------


@pytest.mark.parametrize("path", SKILLS, ids=lambda p: p.parts[-4])
def test_skill_frontmatter_matches_the_agent_skills_spec(path: Path):
    meta, body = parse_frontmatter(path)
    assert meta["name"] == "dspy-jev"
    assert len(meta["name"]) <= 64
    assert meta["name"] == meta["name"].lower()
    assert 0 < len(meta["description"]) <= 1024
    assert body.strip(), "a skill with no instructions loads nothing"


@pytest.mark.parametrize("path", SKILLS, ids=lambda p: p.parts[-4])
def test_skill_description_says_both_what_and_when(path: Path):
    """Pi routes on the description alone; 'helps with decisions' is useless."""
    description = parse_frontmatter(path)[0]["description"].lower()
    assert "use before" in description or "use when" in description
    assert "do not use" in description, "the skill must also say when *not* to load"


@pytest.mark.parametrize("path", SKILLS, ids=lambda p: p.parts[-4])
def test_skill_documents_the_four_routes(path: Path):
    body = parse_frontmatter(path)[1]
    for route in ("auto_execute", "needs_review", "clarify", "block"):
        assert route in body, f"{path} never explains {route}"


@pytest.mark.parametrize("path", SKILLS, ids=lambda p: p.parts[-4])
def test_skill_states_that_a_hold_is_binding(path: Path):
    assert "binding" in parse_frontmatter(path)[1].lower()


def test_the_claude_skill_is_the_only_one_naming_claude():
    """Open weights everywhere except the documented exception."""
    pi_body = parse_frontmatter(SKILLS[0])[1].lower()
    assert "anthropic_api_key" not in pi_body
    claude_body = parse_frontmatter(SKILLS[1])[1].lower()
    assert "claude" in claude_body


# --- json configs ---------------------------------------------------------------


@pytest.mark.parametrize("path", JSON_CONFIGS, ids=lambda p: str(p.relative_to(HARNESSES)))
def test_json_configs_parse(path: Path):
    json.loads(path.read_text(encoding="utf-8"))


def test_claude_code_mcp_config_launches_the_stdio_server():
    config = json.loads((HARNESSES / "claude-code" / ".mcp.json").read_text())
    server = config["mcpServers"]["dspy-jev"]
    assert server["command"] == "dspy-jev"
    assert server["args"][:2] == ["mcp", "--transport"]
    assert server["env"]["DSPY_JEV_HARNESS"] == "claude-code"


def test_hermes_mcp_config_proxies_to_the_shared_service():
    server = json.loads((HARNESSES / "hermes-agent" / ".mcp.json").read_text())["mcpServers"]["dspy-jev"]
    assert "--service-url" in server["args"]
    assert server["env"]["DSPY_JEV_HARNESS"] == "hermes-agent"


def test_pi_models_config_targets_ollama_cloud_with_open_weights():
    from dspy_jev import models

    config = strip_comments(json.loads((HARNESSES / "pi" / "config" / "models.json").read_text()))
    provider = config["providers"]["ollama-cloud"]
    assert provider["baseUrl"] == models.OLLAMA_CLOUD_BASE_URL
    assert provider["api"] == "openai-completions"
    assert provider["apiKey"] == "${OLLAMA_API_KEY}"

    listed = {entry["id"] for entry in provider["models"]}
    known = set(models.open_weight_names())
    assert listed <= known, f"models.json lists models absent from the registry: {listed - known}"


def test_pi_mcp_config_uses_an_http_url_not_a_command():
    """pi-mcp-adapter reaches the shared service over StreamableHTTP."""
    server = strip_comments(json.loads((HARNESSES / "pi" / "config" / "mcp.json").read_text()))["mcpServers"][
        "dspy-jev"
    ]
    assert server["url"].startswith("http")
    assert set(server["directTools"]) == {"jev_decide", "jev_triage", "jev_status"}


# --- hermes profile -------------------------------------------------------------


def test_hermes_profile_declares_the_decide_tool():
    profile = yaml.safe_load((HARNESSES / "hermes-agent" / "profiles" / "dspy-jev.yaml").read_text())
    decide = next(tool for tool in profile["tools"] if tool["name"] == "jev_decide")
    assert decide["endpoint"] == "POST /v1/decide"
    assert set(decide["parameters"]) == {"task", "proposed_action", "context", "autonomy_threshold"}
    assert decide["parameters"]["task"]["required"] is True
    assert decide["parameters"]["context"]["required"] is False


def test_hermes_profile_treats_a_gate_error_as_a_hold():
    """The one rule that must never be 'fail open'."""
    profile = yaml.safe_load((HARNESSES / "hermes-agent" / "profiles" / "dspy-jev.yaml").read_text())
    assert profile["policy"]["on_error"] == "halt_and_report"
    assert profile["policy"]["on_hold"] == "halt_and_report"


def test_hermes_profile_declares_no_model_credentials():
    """The service holds the provider key; the harness holds only a service key."""
    text = (HARNESSES / "hermes-agent" / "profiles" / "dspy-jev.yaml").read_text()
    assert "OLLAMA_API_KEY" not in text
    assert "ANTHROPIC_API_KEY" not in text


def test_hermes_profile_endpoints_exist_in_the_service(settings, allow_lm, configured_dspy):
    """The profile is only useful if the paths it names are real."""
    from fastapi.testclient import TestClient

    from dspy_jev.service.app import create_app

    profile = yaml.safe_load((HARNESSES / "hermes-agent" / "profiles" / "dspy-jev.yaml").read_text())
    declared = {tool["endpoint"].split()[1] for tool in profile["tools"]}
    declared |= {value.split()[1] for value in profile["health"].values()}

    with TestClient(create_app(settings, lm=allow_lm)) as client:
        available = set(client.get("/openapi.json").json()["paths"])
    assert declared <= available, f"profile names endpoints the service does not serve: {declared - available}"


# --- preambles ------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        HARNESSES / "hermes-agent" / "prompts" / "system-preamble.md",
        HARNESSES / "pi" / "prompts" / "gate.md",
        HARNESSES / "claude-code" / "CLAUDE.md",
        HARNESSES / "pi" / "AGENTS.md",
    ],
    ids=lambda p: str(p.relative_to(HARNESSES)),
)
def test_every_preamble_exists_and_is_substantial(path: Path):
    assert path.read_text(encoding="utf-8").strip().count("\n") >= 4


def test_preambles_forbid_rewording_to_pass_the_gate():
    """Prompt-level defence against the most likely failure mode."""
    for path in (
        HARNESSES / "hermes-agent" / "prompts" / "system-preamble.md",
        HARNESSES / "claude-code" / "CLAUDE.md",
        HARNESSES / "pi" / "skills" / "dspy-jev" / "SKILL.md",
    ):
        text = path.read_text(encoding="utf-8").lower()
        assert "reword" in text or "rephrasing" in text, path
