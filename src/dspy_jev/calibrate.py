"""Calibration: fit decision thresholds, cuts and weights with ``ReAnchor``.

``ReAnchor`` does not touch the prompt. It runs the program over a labelled
trainset, collects the probability evidence each call produced, and then
searches locally for the ``threshold`` / ``cuts`` / ``weights`` that maximise the
metric -- keeping a fitted value only when it beats the default *and* survives a
fold check. Because the search reuses cached evidence, calibrating a second time
costs no additional inference.

That is the whole reason this project exists: the same model, re-anchored, makes
materially different decisions, and the change is a reviewable number in a JSON
file rather than an edit to a prompt.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dspy

from dspy_jev.config import Settings, get_settings
from dspy_jev.data import ActionGateExample, load_examples, split
from dspy_jev.metrics import action_gate_metric, false_allow_rate, safety_recall
from dspy_jev.program import ActionGateProgram

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from dspy.experimental import ReAnchor

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Everything a reviewer needs to accept or reject a calibration."""

    artifact_path: Path
    report_path: Path
    report: dict[str, Any]
    fields: dict[str, Any]
    train_score_before: float
    train_score_after: float
    val_score_before: float | None
    val_score_after: float | None
    safety_recall_after: float | None = None
    false_allow_rate_after: float | None = None

    @property
    def improved(self) -> bool:
        """Did anything actually get better on held-out data (or on train, if none)?"""
        if self.val_score_before is not None and self.val_score_after is not None:
            return self.val_score_after > self.val_score_before
        return self.train_score_after > self.train_score_before

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_path": str(self.artifact_path),
            "report_path": str(self.report_path),
            "fields": self.fields,
            "train_score_before": self.train_score_before,
            "train_score_after": self.train_score_after,
            "val_score_before": self.val_score_before,
            "val_score_after": self.val_score_after,
            "safety_recall_after": self.safety_recall_after,
            "false_allow_rate_after": self.false_allow_rate_after,
            "improved": self.improved,
        }


def calibrate_program(
    program: dspy.Module,
    *,
    trainset: list[dspy.Example],
    valset: list[dspy.Example] | None = None,
    metric: Callable[..., float] = action_gate_metric,
    log_dir: Path | None = None,
    require_cache: bool = True,
    num_threads: int | None = None,
) -> tuple[dspy.Module, dict[str, Any]]:
    """Run ``ReAnchor`` and return ``(calibrated_program, report)``.

    The student is left untouched -- ``ReAnchor`` deep-copies it -- so a caller
    can compare before and after on the same object.
    """
    optimizer = ReAnchor(metric, num_threads=num_threads, log_dir=log_dir, require_cache=require_cache)
    calibrated = optimizer.compile(program, trainset=trainset, valset=valset)
    return calibrated, dict(optimizer.report)


def _score_gate(program: ActionGateProgram, examples: list[dspy.Example]) -> tuple[float, float]:
    predictions = [program(task=e.task, proposed_action=e.proposed_action, context=e.context) for e in examples]
    return safety_recall(examples, predictions), false_allow_rate(examples, predictions)


def calibrate_action_gate(
    *,
    dataset: Path | str | None = None,
    settings: Settings | None = None,
    program: ActionGateProgram | None = None,
    train_fraction: float = 0.7,
    require_cache: bool = True,
    artifact_path: Path | str | None = None,
    num_threads: int | None = None,
) -> CalibrationResult:
    """Calibrate the action gate end to end and write the artifact plus a report.

    An LM must already be configured (``dspy.configure(lm=...)``); this function
    does not choose one, so the same code path serves a live provider and a stub.
    """
    settings = settings or get_settings()
    dataset = Path(dataset) if dataset else settings.data_dir / "action_gate.jsonl"
    examples = load_examples(dataset, ActionGateExample)
    trainset, valset = split(examples, train_fraction=train_fraction)
    logger.info(
        "calibration.start",
        extra={"dataset": str(dataset), "train": len(trainset), "val": len(valset)},
    )

    program = program or ActionGateProgram(settings=settings)
    artifact_path = Path(artifact_path) if artifact_path else settings.calibrated_artifact
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    calibrated, report = calibrate_program(
        program,
        trainset=trainset,
        valset=valset or None,
        metric=action_gate_metric,
        log_dir=artifact_path.parent,
        require_cache=require_cache,
        num_threads=num_threads,
    )

    recall = rate = None
    if valset:
        recall, rate = _score_gate(calibrated, valset)

    calibrated.save_calibration(artifact_path)
    report_path = artifact_path.with_suffix(".report.json")
    result = CalibrationResult(
        artifact_path=artifact_path,
        report_path=report_path,
        report=report,
        fields=dict(calibrated.gate.fields),
        train_score_before=float(report.get("train_score_before", 0.0)),
        train_score_after=float(report.get("train_score", 0.0)),
        val_score_before=report.get("val_score_before"),
        val_score_after=report.get("val_score"),
        safety_recall_after=recall,
        false_allow_rate_after=rate,
    )
    report_path.write_text(
        json.dumps({"summary": result.to_dict(), "reanchor": report}, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("calibration.done", extra=result.to_dict())
    return result
