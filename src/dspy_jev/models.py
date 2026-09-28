"""Model registry.

Two provider families are supported, one per harness family:

* ``ollama_cloud`` -- state-of-the-art *open-weight* models served by Ollama Cloud
  through its OpenAI-compatible endpoint (``https://ollama.com/v1``). Used by the
  hermes-agent and Pi harnesses.
* ``anthropic`` -- Claude models, used only by the Claude Code harness.

Ollama retires cloud models over time (see https://ollama.com/settings for the
retirement schedule of models you have used). ``dspy-jev doctor`` resolves the
configured names against the live ``https://ollama.com/api/tags`` listing rather
than trusting this table, so a retirement surfaces as a failed check instead of a
runtime error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Provider = Literal["ollama_cloud", "anthropic"]

#: Roles a program can ask for. Keeping the call sites role-based rather than
#: model-based means a retirement is a one-line registry change.
Role = Literal["decision", "fast", "judge"]

OLLAMA_CLOUD_BASE_URL = "https://ollama.com/v1"
OLLAMA_TAGS_URL = "https://ollama.com/api/tags"


@dataclass(frozen=True)
class ModelSpec:
    """One entry in the registry."""

    name: str
    provider: Provider
    notes: str
    open_weight: bool = True
    supports_thinking: bool = False
    context_hint: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def dspy_model(self) -> str:
        """The ``provider/name`` string DSPy hands to LiteLLM.

        Ollama Cloud is reached through its OpenAI-compatible surface, so the
        LiteLLM provider prefix is ``openai`` with an ``api_base`` override --
        not ``ollama``/``ollama_chat``, which target a local daemon.
        """
        prefix = "openai" if self.provider == "ollama_cloud" else "anthropic"
        return f"{prefix}/{self.name}"


# --- Ollama Cloud, open weights -------------------------------------------------
# Ordered roughly by capability on agentic/reasoning work at the time of writing.
OLLAMA_CLOUD_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="glm-5.3",
        provider="ollama_cloud",
        notes="Z.ai flagship open-weights model; strongest general coding/agentic scores.",
        supports_thinking=True,
    ),
    ModelSpec(
        name="deepseek-v4-pro",
        provider="ollama_cloud",
        notes="Frontier MoE with three reasoning modes and a large context window.",
        supports_thinking=True,
    ),
    ModelSpec(
        name="kimi-k3",
        provider="ollama_cloud",
        notes="Moonshot's open-weight native-multimodal agentic model.",
        supports_thinking=True,
    ),
    ModelSpec(
        name="minimax-m3",
        provider="ollama_cloud",
        notes="1M context window, native multimodality, strong agentic coding.",
        supports_thinking=True,
    ),
    ModelSpec(
        name="nemotron-3-ultra",
        provider="ollama_cloud",
        notes="NVIDIA, built for high-throughput reasoning and long-running agent workflows.",
        supports_thinking=True,
    ),
    ModelSpec(
        name="glm-5.3-flash",
        provider="ollama_cloud",
        notes="18B active params, natively multimodal; good decision/latency trade-off.",
        supports_thinking=True,
    ),
    ModelSpec(
        name="deepseek-v4.1-flash",
        provider="ollama_cloud",
        notes="Fast DeepSeek variant; good default for high-volume gating.",
        supports_thinking=True,
    ),
    ModelSpec(
        name="gpt-oss:120b",
        provider="ollama_cloud",
        notes="OpenAI open-weight 120B; widely available, useful as an independent judge.",
        supports_thinking=True,
        aliases=("gpt-oss",),
    ),
    ModelSpec(
        name="gpt-oss:20b",
        provider="ollama_cloud",
        notes="OpenAI open-weight 20B; cheapest sane option for smoke tests.",
        supports_thinking=True,
    ),
    ModelSpec(
        name="mistral-large-3",
        provider="ollama_cloud",
        notes="General-purpose multimodal MoE for production workloads.",
    ),
    ModelSpec(
        name="nemotron-3-super",
        provider="ollama_cloud",
        notes="120B MoE activating 12B params; compute-efficient multi-agent workhorse.",
        supports_thinking=True,
    ),
)

# --- Anthropic, Claude Code harness only ---------------------------------------
ANTHROPIC_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="claude-sonnet-5-5",
        provider="anthropic",
        notes="Default for the Claude Code harness: balanced cost and judgement.",
        open_weight=False,
    ),
    ModelSpec(
        name="claude-opus-5-5",
        provider="anthropic",
        notes="Highest-capability Claude; use for the judge role or hard rubrics.",
        open_weight=False,
    ),
    ModelSpec(
        name="claude-haiku-4-5-20251001",
        provider="anthropic",
        notes="Fastest Claude; use for the fast role and high-volume gating.",
        open_weight=False,
    ),
)

REGISTRY: dict[str, ModelSpec] = {}
for _spec in (*OLLAMA_CLOUD_MODELS, *ANTHROPIC_MODELS):
    REGISTRY[_spec.name] = _spec
    for _alias in _spec.aliases:
        REGISTRY.setdefault(_alias, _spec)

#: Role -> model, per provider family. Overridable by environment variables.
DEFAULT_ROLES: dict[Provider, dict[str, str]] = {
    "ollama_cloud": {
        "decision": "glm-5.3",
        "fast": "glm-5.3-flash",
        "judge": "deepseek-v4-pro",
    },
    "anthropic": {
        "decision": "claude-sonnet-5-5",
        "fast": "claude-haiku-4-5-20251001",
        "judge": "claude-opus-5-5",
    },
}


class UnknownModelError(KeyError):
    """Raised when a model name is not in the registry."""


def resolve(name: str) -> ModelSpec:
    """Look a model up by registry name or alias."""
    try:
        return REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - message construction only
        known = ", ".join(sorted(REGISTRY))
        raise UnknownModelError(f"Unknown model {name!r}. Known models: {known}") from exc


def default_for(provider: Provider, role: str) -> str:
    """The default model name for ``role`` under ``provider``."""
    roles = DEFAULT_ROLES[provider]
    if role not in roles:
        raise ValueError(f"Unknown role {role!r}. Known roles: {', '.join(sorted(roles))}")
    return roles[role]


def open_weight_names() -> tuple[str, ...]:
    """Every open-weight model in the registry, in declaration order."""
    seen: list[str] = []
    for spec in OLLAMA_CLOUD_MODELS:
        if spec.open_weight and spec.name not in seen:
            seen.append(spec.name)
    return tuple(seen)
