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
# The real shape, taken from techcwbldr26/hermes-agent-profiles: each profile is a
# self-contained distribution directory that `hermes profile install` consumes.

PROFILE = HARNESSES / "hermes-agent" / "decision-gate"


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_the_profile_is_a_self_contained_distribution():
    """`hermes profile install ./decision-gate` needs all of this at the root."""
    for relative in (
        "distribution.yaml",
        "config.yaml",
        "SOUL.md",
        "README.md",
        "docs/README.md",
        "skills/decision-gate/SKILL.md",
        ".gitignore",
    ):
        assert (PROFILE / relative).exists(), f"missing {relative}"


def test_distribution_declares_what_hermes_reads():
    distribution = load_yaml(PROFILE / "distribution.yaml")
    assert distribution["name"] == "decision-gate" == PROFILE.name
    assert distribution["hermes_requires"].startswith(">=")
    required = {entry["name"] for entry in distribution["env_requires"] if entry.get("required")}
    assert required == {"OLLAMA_API_KEY"}, "only the provider key is mandatory"


def test_the_profile_is_named_by_role_not_by_model():
    """Repo convention: models change, roles do not."""
    text = (PROFILE / "distribution.yaml").read_text() + PROFILE.name
    for model_family in ("glm", "kimi", "minimax", "deepseek", "nemotron", "claude"):
        assert model_family not in text.lower(), f"{model_family} appears in the profile's name or id"


def test_config_sets_a_model_and_provider_and_no_base_url():
    """A hand-set base_url under `model:` causes HTTP 404s with ollama-cloud."""
    config = load_yaml(PROFILE / "config.yaml")
    assert set(config) == {"model"}
    assert config["model"]["provider"] == "ollama-cloud"
    assert config["model"]["default"].endswith(":cloud")
    assert "base_url" not in config["model"]


def test_the_profile_model_is_open_weight():
    from dspy_jev import models

    family = load_yaml(PROFILE / "config.yaml")["model"]["default"].split(":")[0]
    known = {name.split(":")[0] for name in models.open_weight_names()}
    assert family in known, f"{family} is not an open-weight model in the registry"


def test_the_profile_gitignore_excludes_every_secret_path():
    """The repo's pre-commit hook blocks these; the profile must too."""
    ignored = set((PROFILE / ".gitignore").read_text().split())
    assert {".env", "auth.json", "memories/", "sessions/"} <= ignored


def test_the_profile_holds_no_model_credentials():
    for path in PROFILE.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert "ANTHROPIC_API_KEY" not in text, path
            assert "sk-" not in text, path


def test_the_skill_enforces_rather_than_advises():
    """The whole point of the rewrite: the skill wraps commands in `guard`."""
    meta, body = parse_frontmatter(PROFILE / "skills" / "decision-gate" / "SKILL.md")
    assert meta["name"] == "decision-gate"
    assert "dspy-jev guard" in body
    assert "did **not** run" in body or "did not run" in body


def test_the_skill_documents_all_four_routes_and_the_exit_codes():
    body = parse_frontmatter(PROFILE / "skills" / "decision-gate" / "SKILL.md")[1]
    for route in ("auto_execute", "needs_review", "clarify", "block"):
        assert route in body
    for code in ("`0`", "`10`", "`1`"):
        assert code in body


def test_the_soul_forbids_the_gate_from_acting_or_granting():
    soul = (PROFILE / "SOUL.md").read_text().lower()
    assert "never carries an action out" in soul or "decides; the caller acts" in soul
    assert "out of scope" in soul


# --- preambles ------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        HARNESSES / "hermes-agent" / "prompts" / "system-preamble.md",
        HARNESSES / "pi" / "prompts" / "gate.md",
        HARNESSES / "claude-code" / "CLAUDE.md",
        HARNESSES / "pi" / "AGENTS.md",
        HARNESSES / "pi" / "extensions" / "dspy-jev-gate" / "README.md",
        HARNESSES / "hermes-agent" / "decision-gate" / "README.md",
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


# --- enforcement assets ----------------------------------------------------------


def test_claude_code_registers_the_hook_on_acting_tools_only():
    """Read/Glob/Grep are absent on purpose: no process spawned per read."""
    settings = json.loads((HARNESSES / "claude-code" / ".claude" / "settings.json").read_text())
    entries = settings["hooks"]["PreToolUse"]
    assert len(entries) == 1
    matcher = entries[0]["matcher"]
    for acting in ("Bash", "Write", "Edit", "WebFetch", "mcp__.*"):
        assert acting in matcher
    for reading in ("Read", "Glob", "Grep"):
        assert f"{reading}|" not in matcher and not matcher.endswith(reading)
    hook = entries[0]["hooks"][0]
    assert hook["type"] == "command"
    assert hook["command"].endswith(".claude/hooks/dspy_jev_gate.py")
    assert hook["timeout"] >= 30


def test_claude_code_hook_script_is_executable_and_self_contained():
    hook = HARNESSES / "claude-code" / ".claude" / "hooks" / "dspy_jev_gate.py"
    assert hook.stat().st_mode & 0o111
    source = hook.read_text()
    assert "permissionDecision" in source
    assert '"deny"' in source


def test_the_pi_extension_blocks_on_the_tool_call_event():
    source = (HARNESSES / "pi" / "extensions" / "dspy-jev-gate" / "index.ts").read_text()
    assert 'pi.on("tool_call"' in source
    # Fail-closed: the catch inside the tool_call handler must block, not fall through.
    handler = source.split('pi.on("tool_call"', 1)[1].split("pi.registerCommand", 1)[0]
    assert "block: true" in handler
    catch_block = handler.split("} catch (error) {", 1)[1].split("}", 1)[0]
    assert "block: true" in catch_block


def test_every_harness_documents_that_a_gate_failure_is_a_hold():
    """The one rule that must never be 'fail open', stated in all three places."""
    sources = [
        HARNESSES / "hermes-agent" / "prompts" / "system-preamble.md",
        HARNESSES / "pi" / "extensions" / "dspy-jev-gate" / "README.md",
        HARNESSES / "claude-code" / ".claude" / "hooks" / "dspy_jev_gate.py",
    ]
    for path in sources:
        # Collapse wrapping: these are prose files, so the phrase spans line breaks.
        text = " ".join(path.read_text().lower().split())
        assert "is not permission" in text, path


# --- documentation ----------------------------------------------------------------

DOCS = REPO_ROOT / "docs"
IMAGES = DOCS / "images"


@pytest.mark.parametrize(
    "name",
    [
        "setup-1-open-environment.png",
        "setup-2-network-access.png",
        "setup-3-api-credentials.png",
        "setup-4-add-credential.png",
        "console-light.png",
        "console-dark.png",
        "lens-threshold-055.png",
        "policy-chain.png",
    ],
)
def test_every_documented_image_exists(name: str):
    path = IMAGES / name
    assert path.exists(), f"missing {path}"
    assert path.stat().st_size > 5_000, f"{name} looks truncated"


def test_no_markdown_link_points_at_a_missing_image():
    """A broken image in a setup guide is worse than no image."""
    import re

    missing = []
    for doc in [*DOCS.rglob("*.md"), REPO_ROOT / "README.md"]:
        for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
            if target.startswith("http"):
                continue
            if not (doc.parent / target).resolve().exists():
                missing.append(f"{doc.relative_to(REPO_ROOT)} -> {target}")
    assert not missing, "broken image links:\n" + "\n".join(missing)


def test_the_setup_guide_covers_both_cloud_settings():
    """Network access and API credentials are separate, and both are easy to miss."""
    text = (DOCS / "setup.md").read_text(encoding="utf-8")
    assert "Network access" in text
    assert "proxy refused CONNECT" in text, "the 403 symptom must be searchable"
    assert "API credentials" in text
    assert "Environment variables" in text, "must say which box NOT to use"


def test_the_setup_guide_never_tells_anyone_to_paste_a_key_into_chat():
    text = (DOCS / "setup.md").read_text(encoding="utf-8").lower()
    assert "do not paste the key into a chat" in text


def test_the_setup_guide_documents_both_auth_modes():
    text = (DOCS / "setup.md").read_text(encoding="utf-8")
    assert "DSPY_JEV_AUTH_MODE=proxy" in text
    assert "doctor --probe" in text


def test_the_scripts_understand_both_auth_modes():
    lib = (REPO_ROOT / "scripts" / "lib.sh").read_text(encoding="utf-8")
    assert "detect_auth_mode" in lib
    assert "network_note" in lib
    for name in ("setup_pi.sh", "setup_hermes_agent.sh", "setup_claude_code.sh"):
        script = (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "detect_auth_mode" in script, name
        assert "DSPY_JEV_AUTH_MODE=$AUTH_MODE" in script, f"{name} must record the mode in .env"
        assert "doctor --probe" in script, f"{name} must point at the probe"
