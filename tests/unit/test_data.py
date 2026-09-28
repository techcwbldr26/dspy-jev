"""Dataset validation. A mistyped label is a calibration bug that hides for weeks."""

from __future__ import annotations

from pathlib import Path

import pytest

from dspy_jev.data import (
    ActionGateExample,
    DatasetError,
    TicketTriageExample,
    load_examples,
    split,
)

pytestmark = pytest.mark.unit

VALID = {
    "task": "fix a test",
    "proposed_action": "read the file",
    "context": "local",
    "safe_to_proceed": True,
    "risk": 0,
    "blast_radius": 0,
    "reversible": True,
    "route": "auto_execute",
}


def write(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "dataset.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_the_shipped_action_gate_dataset_is_valid(dataset_path: Path):
    examples = load_examples(dataset_path, ActionGateExample)
    assert len(examples) >= 30, "calibration needs a meaningful number of examples"


def test_the_shipped_dataset_covers_both_outcomes(dataset_path: Path):
    """A dataset of only-safe examples calibrates a gate that allows everything."""
    examples = load_examples(dataset_path, ActionGateExample)
    safe = sum(1 for e in examples if e.safe_to_proceed)
    assert 0 < safe < len(examples)
    assert min(safe, len(examples) - safe) / len(examples) >= 0.25


def test_the_shipped_dataset_covers_every_route(dataset_path: Path):
    routes = {e.route for e in load_examples(dataset_path, ActionGateExample)}
    assert routes == {"auto_execute", "needs_review", "clarify", "block"}


def test_shipped_labels_are_internally_consistent(dataset_path: Path):
    """auto_execute must never be paired with an unsafe or high-risk label."""
    for e in load_examples(dataset_path, ActionGateExample):
        if e.route == "auto_execute":
            assert e.safe_to_proceed, f"{e.proposed_action!r}: auto_execute but not safe"
            assert e.risk <= 1 and e.blast_radius <= 1 and e.reversible
        if not e.safe_to_proceed:
            assert e.route != "auto_execute"


def test_the_shipped_triage_dataset_is_valid():
    path = Path(__file__).resolve().parents[2] / "data" / "ticket_triage.jsonl"
    assert len(load_examples(path, TicketTriageExample)) >= 10


def test_examples_declare_their_input_fields(dataset_path: Path):
    example = load_examples(dataset_path, ActionGateExample)[0]
    assert set(example.inputs().keys()) == {"task", "proposed_action", "context"}


def test_blank_lines_and_comments_are_skipped(tmp_path: Path):
    import json

    path = write(tmp_path, json.dumps(VALID), "", "// a note", json.dumps(VALID))
    assert len(load_examples(path, ActionGateExample)) == 2


def test_invalid_json_names_the_line(tmp_path: Path):
    path = write(tmp_path, "{not json")
    with pytest.raises(DatasetError, match=r":1: invalid JSON"):
        load_examples(path, ActionGateExample)


def test_an_out_of_range_level_is_rejected(tmp_path: Path):
    import json

    path = write(tmp_path, json.dumps({**VALID, "risk": 9}))
    with pytest.raises(DatasetError, match="risk"):
        load_examples(path, ActionGateExample)


def test_an_unknown_route_is_rejected(tmp_path: Path):
    import json

    path = write(tmp_path, json.dumps({**VALID, "route": "yolo"}))
    with pytest.raises(DatasetError, match="route"):
        load_examples(path, ActionGateExample)


def test_an_unexpected_column_is_rejected(tmp_path: Path):
    """extra='forbid' catches a renamed field before it silently stops being read."""
    import json

    path = write(tmp_path, json.dumps({**VALID, "sevrity": 2}))
    with pytest.raises(DatasetError):
        load_examples(path, ActionGateExample)


def test_a_missing_file_is_reported_clearly(tmp_path: Path):
    with pytest.raises(DatasetError, match="not found"):
        load_examples(tmp_path / "nope.jsonl", ActionGateExample)


def test_an_empty_dataset_is_rejected(tmp_path: Path):
    path = write(tmp_path, "")
    with pytest.raises(DatasetError, match="no examples"):
        load_examples(path, ActionGateExample)


def test_split_is_deterministic_and_order_preserving():
    items = list(range(10))
    assert split(items, train_fraction=0.7) == (items[:7], items[7:])
    assert split(items, train_fraction=0.7) == split(items, train_fraction=0.7)


def test_split_always_leaves_something_on_both_sides():
    train, val = split(list(range(3)), train_fraction=0.99)
    assert train and val


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.5, 2.0])
def test_split_rejects_degenerate_fractions(fraction):
    with pytest.raises(ValueError, match="train_fraction"):
        split(list(range(10)), train_fraction=fraction)
