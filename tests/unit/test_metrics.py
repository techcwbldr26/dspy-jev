"""The metric is what ReAnchor maximises. A symmetric metric calibrates towards
a permissive gate, so the asymmetry is tested explicitly."""

from __future__ import annotations

import dspy
import pytest

from dspy_jev.metrics import (
    FALSE_ALLOW_PENALTY,
    GATE_WEIGHTS,
    action_gate_metric,
    false_allow_rate,
    safety_recall,
    ticket_triage_metric,
)

pytestmark = pytest.mark.unit


class FakeNoul:
    def __init__(self, value: bool):
        self.value = value


class FakeScore:
    def __init__(self, level: int):
        self.level = level
        self.value = float(level)


class FakeChoice:
    def __init__(self, value: str):
        self.value = value


def example(**overrides):
    base = {
        "task": "t",
        "proposed_action": "a",
        "context": "c",
        "safe_to_proceed": True,
        "risk": 0,
        "blast_radius": 0,
        "reversible": True,
        "route": "auto_execute",
    }
    return dspy.Example(**{**base, **overrides})


def prediction(**overrides):
    base = {
        "safe_to_proceed": FakeNoul(True),
        "risk": FakeScore(0),
        "blast_radius": FakeScore(0),
        "reversible": FakeNoul(True),
        "route": FakeChoice("auto_execute"),
    }
    return dspy.Prediction(**{**base, **overrides})


def test_weights_sum_to_one():
    assert sum(GATE_WEIGHTS.values()) == pytest.approx(1.0)


def test_a_perfect_prediction_scores_one():
    assert action_gate_metric(example(), prediction()) == pytest.approx(1.0)


def test_false_allow_costs_more_than_false_block():
    """Predicting 'safe' for an unsafe action must hurt more than the reverse."""
    false_allow = action_gate_metric(example(safe_to_proceed=False, route="block"), prediction())
    false_block = action_gate_metric(example(), prediction(safe_to_proceed=FakeNoul(False), route=FakeChoice("block")))
    assert false_allow < false_block


def test_the_penalty_reaches_beyond_the_safety_weight():
    """Otherwise a gate could buy back a false allow with well-judged rubrics."""
    assert FALSE_ALLOW_PENALTY > 0
    score = action_gate_metric(example(safe_to_proceed=False), prediction())
    assert score < 1.0 - GATE_WEIGHTS["safe_to_proceed"]


def test_auto_executing_a_blocked_action_is_penalised_separately():
    blocked = example(route="block", safe_to_proceed=False)
    assert action_gate_metric(blocked, prediction()) == pytest.approx(0.0, abs=0.06)


def test_ordinal_rubrics_give_partial_credit_for_near_misses():
    exact = action_gate_metric(example(risk=2), prediction(risk=FakeScore(2)))
    near = action_gate_metric(example(risk=2), prediction(risk=FakeScore(3)))
    far = action_gate_metric(example(risk=2), prediction(risk=FakeScore(0)))
    assert exact > near > far


def test_scores_stay_inside_the_unit_interval():
    worst = action_gate_metric(
        example(safe_to_proceed=False, risk=4, blast_radius=3, reversible=False, route="block"),
        prediction(),
    )
    assert 0.0 <= worst <= 1.0


def test_safety_recall_counts_only_unsafe_examples():
    examples = [example(safe_to_proceed=False), example(safe_to_proceed=False), example()]
    predictions = [
        prediction(safe_to_proceed=FakeNoul(False)),
        prediction(),
        prediction(),
    ]
    assert safety_recall(examples, predictions) == pytest.approx(0.5)


def test_safety_recall_is_one_when_nothing_is_unsafe():
    assert safety_recall([example()], [prediction()]) == 1.0


def test_false_allow_rate_is_over_the_whole_dataset():
    examples = [example(safe_to_proceed=False), example(), example(), example()]
    predictions = [prediction()] * 4
    assert false_allow_rate(examples, predictions) == pytest.approx(0.25)


def test_false_allow_rate_of_an_empty_dataset_is_zero():
    assert false_allow_rate([], []) == 0.0


def test_triage_metric_is_equal_weight():
    ex = dspy.Example(ticket="t", urgent=True, severity=2, category="billing")
    perfect = dspy.Prediction(urgent=FakeNoul(True), severity=FakeScore(2), category=FakeChoice("billing"))
    one_wrong = dspy.Prediction(urgent=FakeNoul(True), severity=FakeScore(2), category=FakeChoice("technical"))
    assert ticket_triage_metric(ex, perfect) == pytest.approx(1.0)
    assert ticket_triage_metric(ex, one_wrong) == pytest.approx(2 / 3)
