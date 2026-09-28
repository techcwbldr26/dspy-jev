"""Dataset loading and validation.

Labelled examples live in JSONL so they diff well in review; each line is
validated against a Pydantic model before it becomes a ``dspy.Example``, because
a silently mistyped label is a calibration bug that is very hard to find later.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

import dspy
from pydantic import BaseModel, ConfigDict, Field, ValidationError

ROUTES = ("auto_execute", "needs_review", "clarify", "block")
CATEGORIES = ("billing", "technical", "account")


class DatasetError(ValueError):
    """Raised when a dataset file cannot be parsed or fails validation."""


class ActionGateExample(BaseModel):
    """One labelled action-gate example."""

    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1)
    proposed_action: str = Field(min_length=1)
    context: str = ""
    safe_to_proceed: bool
    risk: int = Field(ge=0, le=4, description="Index into the 5-level risk rubric.")
    blast_radius: int = Field(ge=0, le=3, description="Index into the 4-level blast rubric.")
    reversible: bool
    route: Literal["auto_execute", "needs_review", "clarify", "block"]
    id: str = ""

    def to_dspy(self) -> dspy.Example:
        return dspy.Example(**self.model_dump(exclude={"id"})).with_inputs("task", "proposed_action", "context")


class TicketTriageExample(BaseModel):
    """One labelled triage example (the tutorial rubric)."""

    model_config = ConfigDict(extra="forbid")

    ticket: str = Field(min_length=1)
    urgent: bool
    severity: int = Field(ge=0, le=2)
    category: Literal["billing", "technical", "account"]
    id: str = ""

    def to_dspy(self) -> dspy.Example:
        return dspy.Example(**self.model_dump(exclude={"id"})).with_inputs("ticket")


def iter_jsonl(path: Path | str) -> Iterator[tuple[int, dict]]:
    """Yield ``(line_number, object)`` for every non-blank, non-comment line."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Dataset not found: {path}")
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(parsed, dict):
                raise DatasetError(f"{path}:{number}: expected a JSON object, got {type(parsed).__name__}")
            yield number, parsed


def load_examples(
    path: Path | str,
    model: type[ActionGateExample] | type[TicketTriageExample] = ActionGateExample,
) -> list[dspy.Example]:
    """Parse, validate and convert a JSONL dataset."""
    examples: list[dspy.Example] = []
    for number, payload in iter_jsonl(path):
        try:
            examples.append(model.model_validate(payload).to_dspy())
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
            raise DatasetError(f"{Path(path)}:{number}: {problems}") from exc
    if not examples:
        raise DatasetError(f"{Path(path)}: contains no examples")
    return examples


def split(examples: Iterable[dspy.Example], *, train_fraction: float = 0.7) -> tuple[list, list]:
    """Deterministic train/validation split, preserving file order.

    Order-preserving rather than shuffled on purpose: a reviewer can see exactly
    which rows calibrated the program and which scored it.
    """
    items = list(examples)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    cut = max(1, round(len(items) * train_fraction))
    cut = min(cut, len(items) - 1) if len(items) > 1 else len(items)
    return items[:cut], items[cut:]
