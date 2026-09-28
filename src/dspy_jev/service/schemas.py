"""Request and response models for the HTTP service.

These are the *contract*. The golden-schema test in ``tests/contract`` fails on
any unreviewed change to them, because three harnesses and an MCP tool parse
this shape.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DecideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1, max_length=8000, description="What the agent was asked to accomplish.")
    proposed_action: str = Field(min_length=1, max_length=8000, description="The next concrete step.")
    context: str = Field(default="", max_length=32000, description="Environment facts relevant to the step.")
    autonomy_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Per-call override of the policy floor on P(safe_to_proceed). Only ever tightens.",
    )


class DecisionEvidence(BaseModel):
    """One decision output, with the probability evidence behind it."""

    model_config = ConfigDict(extra="allow")

    kind: Literal["noul", "score", "choice"]
    value: Any
    probability: float | None = None
    probabilities: dict[str, float] | None = None
    level: int | None = None
    confidence: float | None = None


class DecideResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allow: bool = Field(description="True only when every policy condition held.")
    route: str = Field(description="auto_execute | needs_review | clarify | block")
    reasons: list[str] = Field(description="Empty when allowed; otherwise every condition that failed.")
    rationale: str = ""
    decisions: dict[str, DecisionEvidence] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    request_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class TriageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket: str = Field(min_length=1, max_length=16000)


class TriageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decisions: dict[str, DecisionEvidence] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    request_id: str = ""


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded"]
    version: str
    harness: str
    model: str | None = None
    calibrated: bool = False
    checks: dict[str, bool] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: str
    detail: str = ""
    request_id: str = ""
