"""Signatures and the audit record. These shapes are what every harness parses."""

from __future__ import annotations

import json

import dspy
import pytest

from dspy_jev.decisions import ActionGate, TicketTriage, decision_record
from dspy_jev.program import ActionGateProgram
from tests._support import gate_evidence, stub_lm

pytestmark = pytest.mark.unit


def test_action_gate_declares_the_expected_surface():
    assert list(ActionGate.input_fields) == ["task", "proposed_action", "context"]
    assert list(ActionGate.output_fields) == [
        "safe_to_proceed",
        "risk",
        "blast_radius",
        "reversible",
        "route",
        "rationale",
    ]


@pytest.mark.parametrize("signature", [ActionGate, TicketTriage])
def test_every_output_field_has_a_description(signature):
    """DSPy raises before calling the LM when a decision output has no desc."""
    for name, field in signature.output_fields.items():
        assert field.json_schema_extra.get("desc"), f"{signature.__name__}.{name} has no description"


@pytest.mark.parametrize("signature", [ActionGate, TicketTriage])
def test_instructions_tell_the_model_to_treat_inputs_as_data(signature):
    """A gate that can be talked out of its own rubric is not a gate."""
    assert "data" in signature.instructions.lower()


def test_decision_record_separates_evidence_from_plain_outputs(settings, configured_dspy):
    with dspy.context(lm=stub_lm(gate_evidence())):
        prediction = ActionGateProgram(settings=settings)(task="t", proposed_action="a", context="c")
    record = decision_record(prediction)

    assert set(record["decisions"]) == {"safe_to_proceed", "risk", "blast_radius", "reversible", "route"}
    assert set(record["outputs"]) == {"rationale"}


def test_decision_record_is_json_serialisable(settings, configured_dspy):
    with dspy.context(lm=stub_lm(gate_evidence())):
        record = decision_record(ActionGateProgram(settings=settings)(task="t", proposed_action="a"))
    assert json.loads(json.dumps(record)) == json.loads(json.dumps(record))


def test_noul_record_carries_the_probability_and_the_derived_value(settings, configured_dspy):
    with dspy.context(lm=stub_lm(gate_evidence(safe=0.73))):
        record = decision_record(ActionGateProgram(settings=settings)(task="t", proposed_action="a"))
    safe = record["decisions"]["safe_to_proceed"]
    assert safe == {"kind": "noul", "value": True, "probability": 0.73, "confidence": pytest.approx(0.46)}


def test_score_record_carries_level_and_distribution(settings, configured_dspy):
    with dspy.context(lm=stub_lm(gate_evidence(risk_level=4))):
        record = decision_record(ActionGateProgram(settings=settings)(task="t", proposed_action="a"))
    risk = record["decisions"]["risk"]
    assert risk["kind"] == "score"
    assert risk["level"] == 4
    assert sum(risk["probabilities"].values()) == pytest.approx(1.0)


def test_choice_record_preserves_the_option_value(settings, configured_dspy):
    with dspy.context(lm=stub_lm(gate_evidence(route="clarify"))):
        record = decision_record(ActionGateProgram(settings=settings)(task="t", proposed_action="a"))
    assert record["decisions"]["route"]["value"] == "clarify"


def test_evidence_is_derived_locally_not_generated(settings, configured_dspy):
    """The LM supplies P(True); the value comes from the threshold, not the model.

    Same evidence, two thresholds, two different answers -- with no second call.
    """
    with dspy.context(lm=stub_lm(gate_evidence(safe=0.60))):
        program = ActionGateProgram(settings=settings)
        permissive = program(task="t", proposed_action="a").safe_to_proceed.value
        program.gate.fields["safe_to_proceed"] = {"threshold": 0.8}
        strict = program(task="t", proposed_action="a").safe_to_proceed.value
    assert permissive is True
    assert strict is False
