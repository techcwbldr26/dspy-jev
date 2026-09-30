"""Programs: a DSPy predictor plus a deterministic policy layer.

Two layers, deliberately separated:

1. **Judgement** -- ``dspy.Predict`` over a Jev signature. The model supplies
   probability evidence; DSPy derives values from per-field parameters
   (``threshold``, ``cuts``, ``weights``) that :mod:`dspy_jev.calibrate` fits.
2. **Policy** -- ordinary Python over those derived values. No model involved,
   so it is testable, reviewable, and identical across harnesses.

Keeping them apart means a harness can tighten its own autonomy bar without
recalibrating, and an auditor can read the policy without reading a prompt.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import dspy

from dspy_jev.config import Settings, get_settings
from dspy_jev.decisions import ActionGate, TicketTriage, decision_record
from dspy_jev.observability import policy_span

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from dspy.experimental import Choice, Noul, Score  # noqa: F401  (re-exported for typing)

#: Routes the policy layer will allow to run without a person.
AUTONOMOUS_ROUTES = frozenset({"auto_execute"})

#: Highest ``risk.level`` (0-4) still eligible for autonomous execution.
MAX_AUTONOMOUS_RISK_LEVEL = 1

#: Highest ``blast_radius.level`` (0-3) still eligible for autonomous execution.
MAX_AUTONOMOUS_BLAST_LEVEL = 1


@dataclass(frozen=True)
class GateDecision:
    """The full, auditable result of one gate call."""

    allow: bool
    route: str
    reasons: list[str]
    record: dict[str, Any]
    rationale: str = ""
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActionGateProgram(dspy.Module):
    """Decide whether an agent may take a proposed action.

    Args:
        settings: configuration; defaults to the process settings.
        autonomy_threshold: minimum ``P(safe_to_proceed)`` for autonomous
            execution. This is a *policy* floor applied on top of whatever
            threshold the Noul field itself was calibrated to, so tightening it
            can only ever make the gate more conservative.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        autonomy_threshold: float | None = None,
        max_risk_level: int = MAX_AUTONOMOUS_RISK_LEVEL,
        max_blast_level: int = MAX_AUTONOMOUS_BLAST_LEVEL,
    ) -> None:
        super().__init__()
        self._settings = settings or get_settings()
        self.autonomy_threshold = (
            self._settings.autonomy_threshold if autonomy_threshold is None else float(autonomy_threshold)
        )
        if not 0.0 <= self.autonomy_threshold <= 1.0:
            raise ValueError("autonomy_threshold must be within [0, 1]")
        self.max_risk_level = int(max_risk_level)
        self.max_blast_level = int(max_blast_level)
        self.gate = dspy.Predict(ActionGate)

    # -- inference ---------------------------------------------------------------
    def forward(self, task: str, proposed_action: str, context: str = "") -> dspy.Prediction:
        return self.gate(task=task, proposed_action=proposed_action, context=context)

    def decide(self, task: str, proposed_action: str, context: str = "") -> GateDecision:
        """Run the gate and apply the policy layer.

        The span opens here rather than at each call site, so a trace has the
        same shape whoever asked -- the service, the CLI, ``guard`` or the MCP
        server. DSPy's own spans nest underneath it, which is what makes the
        policy layer legible as the step above the model rather than beside it.
        """
        started = time.perf_counter()
        with policy_span(
            "dspy_jev.decide",
            {"task": task, "proposed_action": proposed_action, "context": context},
        ):
            prediction = self(task=task, proposed_action=proposed_action, context=context)
            latency_ms = (time.perf_counter() - started) * 1000
            return self.apply_policy(prediction, latency_ms=latency_ms)

    # -- policy ------------------------------------------------------------------
    def apply_policy(self, prediction: dspy.Prediction, *, latency_ms: float = 0.0) -> GateDecision:
        """Derive ``allow`` from a prediction. Pure function of the prediction."""
        record = decision_record(prediction)
        decisions = record["decisions"]
        reasons: list[str] = []

        route = str(decisions.get("route", {}).get("value", "block"))
        safe = decisions.get("safe_to_proceed", {})
        safe_value = bool(safe.get("value", False))
        probability = safe.get("probability")
        risk_level = decisions.get("risk", {}).get("level")
        blast_level = decisions.get("blast_radius", {}).get("level")
        reversible = bool(decisions.get("reversible", {}).get("value", False))

        if route not in AUTONOMOUS_ROUTES:
            reasons.append(f"route={route}")
        if not safe_value:
            reasons.append("safe_to_proceed=False")
        if probability is not None and probability < self.autonomy_threshold:
            reasons.append(f"P(safe)={probability:.3f} < autonomy_threshold={self.autonomy_threshold:.3f}")
        if probability is None:
            reasons.append("no probability evidence for safe_to_proceed")
        if risk_level is not None and risk_level > self.max_risk_level:
            reasons.append(f"risk level {risk_level} > max {self.max_risk_level}")
        if blast_level is not None and blast_level > self.max_blast_level:
            reasons.append(f"blast radius level {blast_level} > max {self.max_blast_level}")
        if not reversible:
            reasons.append("action is not self-reversible")

        return GateDecision(
            allow=not reasons,
            route=route,
            reasons=reasons,
            record=record,
            rationale=str(record["outputs"].get("rationale", "")),
            latency_ms=round(latency_ms, 3),
            metadata={
                "autonomy_threshold": self.autonomy_threshold,
                "max_risk_level": self.max_risk_level,
                "max_blast_level": self.max_blast_level,
                "calibrated": bool(self.gate.fields),
                "fields": dict(self.gate.fields),
            },
        )

    # -- persistence -------------------------------------------------------------
    def load_calibration(self, path: Path | str | None = None, *, required: bool = False) -> bool:
        """Load fitted decision parameters. Returns ``True`` when something loaded.

        An uncalibrated gate is still a working gate -- it just uses the type
        defaults -- so a missing artifact is a warning, not an error, unless the
        caller asks for ``required``.
        """
        path = Path(path) if path is not None else self._settings.calibrated_artifact
        if not path.exists():
            if required:
                raise FileNotFoundError(f"No calibration artifact at {path}")
            return False
        self.gate.load(path)
        return True

    def save_calibration(self, path: Path | str | None = None) -> Path:
        """Persist the predictor, including fitted decision parameters."""
        path = Path(path) if path is not None else self._settings.calibrated_artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        self.gate.save(path)
        return path


class TicketTriageProgram(dspy.Module):
    """The tutorial rubric, kept as a second schema for regression coverage."""

    def __init__(self) -> None:
        super().__init__()
        self.triage = dspy.Predict(TicketTriage)

    def forward(self, ticket: str) -> dspy.Prediction:
        return self.triage(ticket=ticket)

    def record(self, ticket: str) -> dict[str, Any]:
        return decision_record(self(ticket=ticket))
