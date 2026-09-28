"""Shared test helpers: evidence builders and stub language models.

The stubs return *probability evidence*, exactly as a real backend does, so the
tests exercise DSPy's real decoding path (thresholds, cuts, weights) rather than
a mock of it. That is what makes these tests worth having: a change in how DSPy
derives a value from evidence will fail them.
"""

from __future__ import annotations

import warnings
from typing import Any

from dspy.utils.dummies import DummyLM

#: Level counts for the two ordinal rubrics in ``ActionGate``.
RISK_LEVELS = 5
BLAST_LEVELS = 4
ROUTES = ("auto_execute", "needs_review", "clarify", "block")


def noul(probability: float) -> dict[str, float]:
    """Evidence for a ``Noul`` output."""
    return {"noul": float(probability)}


def score(probabilities: list[float], confidence: float = 0.8) -> dict[str, Any]:
    """Evidence for a ``Score`` output, keyed by level index."""
    return {
        "probabilities": {str(i): float(p) for i, p in enumerate(probabilities)},
        "confidence": float(confidence),
    }


def choice(probabilities: dict[str, float], confidence: float = 0.8) -> dict[str, Any]:
    """Evidence for a ``Choice`` output, keyed by option label."""
    return {"probabilities": {k: float(v) for k, v in probabilities.items()}, "confidence": float(confidence)}


def _spike(levels: int, level: int, mass: float = 0.8) -> list[float]:
    """A distribution concentrated on ``level``, the rest spread evenly."""
    rest = (1.0 - mass) / (levels - 1)
    return [mass if i == level else rest for i in range(levels)]


def gate_evidence(
    *,
    safe: float = 0.95,
    risk_level: int = 0,
    blast_level: int = 0,
    reversible: float = 0.95,
    route: str = "auto_execute",
    route_mass: float = 0.85,
    rationale: str = "Scoped, reversible, inside the declared task.",
) -> dict[str, Any]:
    """One full set of ``ActionGate`` evidence.

    Defaults describe an obviously safe action, so a test only states the axis
    it is actually about.
    """
    rest = (1.0 - route_mass) / (len(ROUTES) - 1)
    return {
        "safe_to_proceed": noul(safe),
        "risk": score(_spike(RISK_LEVELS, risk_level)),
        "blast_radius": score(_spike(BLAST_LEVELS, blast_level)),
        "reversible": noul(reversible),
        "route": choice({r: (route_mass if r == route else rest) for r in ROUTES}),
        "rationale": rationale,
    }


def triage_evidence(*, urgent: float = 0.9, severity_level: int = 2, category: str = "technical") -> dict[str, Any]:
    categories = ("billing", "technical", "account")
    rest = 0.2 / (len(categories) - 1)
    return {
        "urgent": noul(urgent),
        "severity": score(_spike(3, severity_level)),
        "category": choice({c: (0.8 if c == category else rest) for c in categories}),
    }


def stub_lm(evidence: dict[str, Any] | list[dict[str, Any]], repeats: int = 64) -> DummyLM:
    """A DummyLM that replays evidence.

    ``repeats`` guards against exhaustion: DSPy may call the LM more than once
    per logical decision (retries, adapter fallbacks), and a test should fail on
    its assertion, not on a drained queue.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        answers = evidence if isinstance(evidence, list) else [evidence]
        return DummyLM(list(answers) * repeats)


def keyed_stub_lm(table: dict[str, dict[str, Any]]) -> DummyLM:
    """A DummyLM that picks evidence by matching a key inside the final prompt.

    Used by the calibration tests, where each example must produce its own
    probability so ``ReAnchor`` has a distribution to fit against.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DummyLM(dict(table))
