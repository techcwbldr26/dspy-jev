"""Data the dashboard needs, kept out of the route handlers.

Three things a person wants to see about a decision gate, and none of them is
visible from a log line:

* **The threshold trade-off.** Where the cut sits, and which labelled actions
  fall on the wrong side of it. :func:`lens_rows` produces one row per labelled
  example with the probability the model actually gave it.
* **What the last few decisions were.** :class:`DecisionLog` keeps a bounded
  ring buffer; nothing is written to disk, because decision inputs carry user
  data.
* **What calibration changed.** :func:`calibration_summary` reads the fitted
  parameters and the before/after report off disk.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dspy_jev.config import Settings
from dspy_jev.data import ActionGateExample, DatasetError, load_examples

#: How many recent decisions the dashboard can show. Bounded on purpose.
DECISION_LOG_SIZE = 200

#: Inputs are truncated before they are kept in memory for the feed.
FEED_TEXT_LIMIT = 240


@dataclass(frozen=True)
class LoggedDecision:
    """One decision, trimmed to what the feed displays."""

    at: float
    request_id: str
    action: str
    allow: bool
    route: str
    probability: float | None
    risk_level: int | None
    reasons: list[str]
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "request_id": self.request_id,
            "action": self.action,
            "allow": self.allow,
            "route": self.route,
            "probability": self.probability,
            "risk_level": self.risk_level,
            "reasons": list(self.reasons),
            "latency_ms": self.latency_ms,
        }


class DecisionLog:
    """A bounded, in-memory record of recent decisions.

    In memory rather than on disk because the action and context are the user's
    data; the audit log is the durable record, and it redacts.
    """

    def __init__(self, maxlen: int = DECISION_LOG_SIZE) -> None:
        self._entries: deque[LoggedDecision] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def record(self, *, request_id: str, action: str, decision: Any) -> None:
        record = decision.record["decisions"]
        entry = LoggedDecision(
            at=time.time(),
            request_id=request_id,
            action=_clip(action),
            allow=bool(decision.allow),
            route=str(decision.route),
            probability=record.get("safe_to_proceed", {}).get("probability"),
            risk_level=record.get("risk", {}).get("level"),
            reasons=list(decision.reasons),
            latency_ms=float(decision.latency_ms),
        )
        with self._lock:
            self._entries.append(entry)

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Newest first."""
        with self._lock:
            entries = list(self._entries)
        return [entry.to_dict() for entry in reversed(entries[-limit:])]

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def _clip(text: str, limit: int = FEED_TEXT_LIMIT) -> str:
    collapsed = " ".join(str(text).split())
    return collapsed if len(collapsed) <= limit else f"{collapsed[:limit]}…"


def calibration_summary(settings: Settings, *, fields: dict[str, Any]) -> dict[str, Any]:
    """The fitted parameters in force, plus the report that produced them."""
    artifact = settings.calibrated_artifact
    report_path = artifact.with_suffix(".report.json")
    summary: dict[str, Any] = {
        "artifact_path": str(artifact),
        "artifact_exists": artifact.exists(),
        "fields": fields,
        "report": None,
    }
    if report_path.exists():
        try:
            summary["report"] = json.loads(report_path.read_text(encoding="utf-8")).get("summary")
        except (OSError, json.JSONDecodeError):
            summary["report"] = None
    return summary


def lens_rows(program: Any, settings: Settings, dataset: Path | None = None) -> dict[str, Any]:
    """Run the labelled dataset through the gate and return one row per example.

    Each row pairs the label a human gave with the probability the model gave,
    which is what makes the threshold trade-off visible. DSPy caches responses,
    so the first call costs a pass over the dataset and later calls are free.
    """
    path = Path(dataset) if dataset else settings.data_dir / "action_gate.jsonl"
    try:
        examples = load_examples(path, ActionGateExample)
    except DatasetError as exc:
        return {"dataset": str(path), "error": str(exc), "rows": []}

    rows: list[dict[str, Any]] = []
    for example in examples:
        prediction = program(task=example.task, proposed_action=example.proposed_action, context=example.context)
        safe = getattr(prediction, "safe_to_proceed", None)
        risk = getattr(prediction, "risk", None)
        route = getattr(prediction, "route", None)
        rows.append(
            {
                "action": _clip(example.proposed_action, 110),
                "task": _clip(example.task, 80),
                "label_safe": bool(example.safe_to_proceed),
                "label_route": example.route,
                "probability": getattr(safe, "probability", None),
                "predicted_safe": bool(getattr(safe, "value", False)),
                "risk_level": getattr(risk, "level", None),
                "predicted_route": getattr(route, "value", None),
            }
        )
    return {"dataset": str(path), "error": None, "rows": rows}


def threshold_tradeoff(rows: list[dict[str, Any]], threshold: float) -> dict[str, int]:
    """Count the two kinds of error at one threshold.

    Recomputed in the browser as the reader drags, and here so the API can
    report the current setting without a round trip.
    """
    false_allows = sum(1 for r in rows if not r["label_safe"] and (r["probability"] or 0.0) >= threshold)
    false_holds = sum(1 for r in rows if r["label_safe"] and (r["probability"] or 0.0) < threshold)
    return {
        "false_allows": false_allows,
        "false_holds": false_holds,
        "total": len(rows),
        "unsafe": sum(1 for r in rows if not r["label_safe"]),
        "safe": sum(1 for r in rows if r["label_safe"]),
    }
