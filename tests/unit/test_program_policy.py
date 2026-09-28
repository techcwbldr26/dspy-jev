"""The policy layer.

Everything here is a pure function of a prediction, so it is tested without a
model at all. This is the layer an auditor reads: if it is wrong, a calibrated
model does not save you.
"""

from __future__ import annotations

import dspy
import pytest

from dspy_jev.program import (
    MAX_AUTONOMOUS_BLAST_LEVEL,
    MAX_AUTONOMOUS_RISK_LEVEL,
    ActionGateProgram,
    TicketTriageProgram,
)
from tests._support import gate_evidence, stub_lm, triage_evidence

pytestmark = pytest.mark.unit


def decide(settings, evidence, **program_kwargs):
    with dspy.context(lm=stub_lm(evidence)):
        program = ActionGateProgram(settings=settings, **program_kwargs)
        return program.decide(task="t", proposed_action="a", context="c")


def test_an_obviously_safe_action_is_allowed(settings, configured_dspy):
    decision = decide(settings, gate_evidence())
    assert decision.allow is True
    assert decision.reasons == []
    assert decision.route == "auto_execute"


@pytest.mark.parametrize(
    ("evidence", "expected_reason"),
    [
        (gate_evidence(route="needs_review"), "route=needs_review"),
        (gate_evidence(route="block"), "route=block"),
        (gate_evidence(route="clarify"), "route=clarify"),
        (gate_evidence(safe=0.10), "safe_to_proceed=False"),
        (gate_evidence(risk_level=3), "risk level 3"),
        (gate_evidence(blast_level=3), "blast radius level 3"),
        (gate_evidence(reversible=0.05), "not self-reversible"),
    ],
)
def test_each_hazard_blocks_on_its_own(settings, configured_dspy, evidence, expected_reason):
    """No single failing condition can be outvoted by the others."""
    decision = decide(settings, evidence)
    assert decision.allow is False
    assert any(expected_reason in reason for reason in decision.reasons)


def test_probability_below_the_policy_floor_holds_the_action(settings, configured_dspy):
    """0.60 clears the Noul threshold of 0.5 but not the 0.85 policy floor."""
    decision = decide(settings, gate_evidence(safe=0.60))
    assert decision.record["decisions"]["safe_to_proceed"]["value"] is True
    assert decision.allow is False
    assert any("autonomy_threshold" in reason for reason in decision.reasons)


def test_every_failing_condition_is_reported_not_just_the_first(settings, configured_dspy):
    decision = decide(settings, gate_evidence(safe=0.05, risk_level=4, blast_level=3, reversible=0.01, route="block"))
    assert len(decision.reasons) >= 5


def test_raising_the_threshold_only_ever_tightens(settings, configured_dspy):
    evidence = gate_evidence(safe=0.90)
    assert decide(settings, evidence, autonomy_threshold=0.85).allow is True
    assert decide(settings, evidence, autonomy_threshold=0.95).allow is False


def test_default_policy_ceilings_are_conservative():
    assert MAX_AUTONOMOUS_RISK_LEVEL <= 1
    assert MAX_AUTONOMOUS_BLAST_LEVEL <= 1


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_invalid_thresholds_are_rejected_at_construction(settings, threshold):
    with pytest.raises(ValueError, match="autonomy_threshold"):
        ActionGateProgram(settings=settings, autonomy_threshold=threshold)


def test_apply_policy_is_pure(settings, configured_dspy):
    """Same prediction, same verdict, however many times it is evaluated."""
    with dspy.context(lm=stub_lm(gate_evidence(safe=0.9))):
        program = ActionGateProgram(settings=settings)
        prediction = program(task="t", proposed_action="a")
    first, second = program.apply_policy(prediction), program.apply_policy(prediction)
    assert first.allow == second.allow
    assert first.reasons == second.reasons


def test_decision_metadata_reports_whether_calibration_was_in_force(settings, configured_dspy):
    decision = decide(settings, gate_evidence())
    assert decision.metadata["calibrated"] is False
    assert decision.metadata["autonomy_threshold"] == settings.autonomy_threshold


def test_latency_is_measured(settings, configured_dspy):
    assert decide(settings, gate_evidence()).latency_ms >= 0.0


def test_missing_calibration_artifact_is_not_fatal(settings):
    assert ActionGateProgram(settings=settings).load_calibration() is False


def test_required_calibration_raises_when_absent(settings):
    with pytest.raises(FileNotFoundError):
        ActionGateProgram(settings=settings).load_calibration(required=True)


def test_calibration_round_trips(settings, tmp_path, configured_dspy):
    with dspy.context(lm=stub_lm(gate_evidence())):
        original = ActionGateProgram(settings=settings)
        original.gate.fields["safe_to_proceed"] = {"threshold": 0.77}
        original.gate.fields["risk"] = {"cuts": [0.4, 1.4, 2.4, 3.4]}
        path = original.save_calibration(tmp_path / "gate.json")

        restored = ActionGateProgram(settings=settings)
        assert restored.load_calibration(path) is True
    assert restored.gate.fields["safe_to_proceed"]["threshold"] == 0.77
    assert restored.gate.fields["risk"]["cuts"] == [0.4, 1.4, 2.4, 3.4]


def test_restored_calibration_changes_the_verdict(settings, tmp_path, configured_dspy):
    """A saved artifact is only useful if loading it actually moves the decision."""
    evidence = gate_evidence(safe=0.90)
    with dspy.context(lm=stub_lm(evidence)):
        strict = ActionGateProgram(settings=settings)
        strict.gate.fields["safe_to_proceed"] = {"threshold": 0.95}
        path = strict.save_calibration(tmp_path / "strict.json")

        loaded = ActionGateProgram(settings=settings)
        loaded.load_calibration(path)
        decision = loaded.decide(task="t", proposed_action="a")
    assert decision.allow is False
    assert decision.metadata["calibrated"] is True


def test_triage_program_produces_a_record(configured_dspy):
    with dspy.context(lm=stub_lm(triage_evidence(category="billing"))):
        record = TicketTriageProgram().record(ticket="I was charged twice")
    assert record["decisions"]["category"]["value"] == "billing"
    assert set(record["decisions"]) == {"urgent", "severity", "category"}
