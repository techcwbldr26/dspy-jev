"""FastAPI application: one decision service, three harnesses.

The harnesses do not each embed DSPy. They call this. That keeps one calibrated
artifact, one audit trail, and one place where the policy layer lives -- which
is the point of the "standalone service + thin clients" shape.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import dspy
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from dspy_jev import __version__
from dspy_jev.config import Settings, get_settings
from dspy_jev.decisions import decision_record
from dspy_jev.lm import MissingCredentialError, build_lm, describe_lm
from dspy_jev.observability import (
    METRICS,
    DecisionAuditCallback,
    configure_observability,
    policy_span,
)
from dspy_jev.program import ActionGateProgram, TicketTriageProgram
from dspy_jev.service.dashboard import (
    DecisionLog,
    calibration_summary,
    lens_rows,
    threshold_tradeoff,
)
from dspy_jev.service.schemas import (
    DecideRequest,
    DecideResponse,
    ErrorResponse,
    HealthResponse,
    TriageRequest,
    TriageResponse,
)

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

#: The dashboard is a single self-contained file served from this package, so it
#: works offline, needs no build step, and shares an origin with the API.
DASHBOARD_HTML = Path(__file__).with_name("dashboard.html")


class AppState:
    """Everything the request handlers need, built once at startup."""

    def __init__(
        self,
        settings: Settings,
        *,
        lm: dspy.LM | None = None,
        gate: ActionGateProgram | None = None,
        triage: TicketTriageProgram | None = None,
    ) -> None:
        self.settings = settings
        self.lm = lm
        self.gate = gate or ActionGateProgram(settings=settings)
        self.triage = triage or TicketTriageProgram()
        self.calibrated = False
        self.lm_error: str | None = None
        self.decisions = DecisionLog()
        self._lens: dict[str, Any] | None = None

    def start(self) -> None:
        """Wire observability, resolve the LM, load the calibration artifact."""
        configure_observability(self.settings)
        dspy.configure(callbacks=[DecisionAuditCallback(self.settings)])
        if self.lm is None:
            try:
                self.lm = build_lm("decision", settings=self.settings)
            except MissingCredentialError as exc:
                # Start anyway so /healthz can report *why* the service is degraded.
                self.lm_error = str(exc)
                logger.error("startup.no_credentials", extra={"error": str(exc)})
        if self.lm is not None:
            dspy.configure(lm=self.lm)
        # Report whatever is actually in force: an artifact loaded here, or a gate
        # the caller injected with parameters already applied.
        loaded = self.gate.load_calibration()
        self.calibrated = bool(loaded or (getattr(self.gate, "gate", None) and self.gate.gate.fields))
        logger.info(
            "startup.ready",
            extra={
                "harness": self.settings.harness,
                "calibrated": self.calibrated,
                "lm": describe_lm(self.lm) if self.lm else None,
            },
        )

    @property
    def ready(self) -> bool:
        return self.lm is not None


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "")


def create_app(
    settings: Settings | None = None,
    *,
    lm: dspy.LM | None = None,
    gate: ActionGateProgram | None = None,
    triage: TicketTriageProgram | None = None,
) -> FastAPI:
    """Build the application.

    Every collaborator is injectable so the contract tests can run the real
    routing, the real policy layer and the real serialisation against a stub LM.
    """
    settings = settings or get_settings()
    state = AppState(settings, lm=lm, gate=gate, triage=triage)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state.start()
        yield

    app = FastAPI(
        title="dspy-jev decision service",
        version=__version__,
        summary="Calibrated, probability-backed decisions for agent harnesses.",
        lifespan=lifespan,
    )
    app.state.jev = state

    # --- middleware -------------------------------------------------------------
    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Response:
        request.state.request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            METRICS.increment("dspy_jev_http_requests_total", path=request.url.path, status="500")
            logger.exception("http.error", extra={"request_id": request.state.request_id})
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000
        METRICS.increment("dspy_jev_http_requests_total", path=request.url.path, status=str(response.status_code))
        METRICS.observe_latency("dspy_jev_http_latency", elapsed_ms)
        response.headers[REQUEST_ID_HEADER] = request.state.request_id
        logger.info(
            "http.request",
            extra={
                "request_id": request.state.request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round(elapsed_ms, 3),
            },
        )
        return response

    # --- auth -------------------------------------------------------------------
    async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        expected = settings.service_api_key
        if expected is None:
            return
        if x_api_key != expected.get_secret_value():
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    # --- errors -----------------------------------------------------------------
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        body = ErrorResponse(error=exc.__class__.__name__, detail=str(exc.detail), request_id=_request_id(request))
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    # --- routes -----------------------------------------------------------------
    @app.get("/healthz", response_model=HealthResponse, tags=["ops"])
    async def healthz() -> HealthResponse:
        """Liveness. Always 200 while the process can answer."""
        return HealthResponse(
            status="ok" if state.ready else "degraded",
            version=__version__,
            harness=settings.harness,
            model=getattr(state.lm, "model", None),
            calibrated=state.calibrated,
            checks={"lm": state.ready, "calibration": state.calibrated},
        )

    @app.get("/readyz", response_model=HealthResponse, tags=["ops"])
    async def readyz(response: Response) -> HealthResponse:
        """Readiness. 503 until an LM is resolved, so a load balancer holds traffic."""
        if not state.ready:
            response.status_code = 503
        return HealthResponse(
            status="ok" if state.ready else "degraded",
            version=__version__,
            harness=settings.harness,
            model=getattr(state.lm, "model", None),
            calibrated=state.calibrated,
            checks={"lm": state.ready, "calibration": state.calibrated},
        )

    @app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
    async def metrics() -> str:
        """Prometheus text exposition."""
        return METRICS.render_prometheus()

    @app.post(
        "/v1/decide",
        response_model=DecideResponse,
        tags=["decide"],
        dependencies=[Depends(require_api_key)],
    )
    async def decide(payload: DecideRequest, request: Request) -> DecideResponse:
        """Gate one proposed agent action."""
        if not state.ready:
            raise HTTPException(status_code=503, detail=state.lm_error or "no language model configured")

        gate = state.gate
        if payload.autonomy_threshold is not None:
            # Per-call overrides may only tighten the policy, never loosen it.
            gate = ActionGateProgram(
                settings=settings,
                autonomy_threshold=max(payload.autonomy_threshold, state.gate.autonomy_threshold),
                max_risk_level=state.gate.max_risk_level,
                max_blast_level=state.gate.max_blast_level,
            )
            gate.gate.fields = dict(state.gate.gate.fields)
            gate.gate.demos = list(state.gate.gate.demos)

        with policy_span(
            "dspy_jev.decide",
            {"task": payload.task, "proposed_action": payload.proposed_action, "context": payload.context},
        ):
            try:
                decision = gate.decide(
                    task=payload.task, proposed_action=payload.proposed_action, context=payload.context
                )
            except Exception as exc:
                METRICS.increment("dspy_jev_decisions_total", outcome="error")
                logger.exception("decide.failed", extra={"request_id": _request_id(request)})
                raise HTTPException(status_code=502, detail=f"decision failed: {exc}") from exc

        METRICS.increment(
            "dspy_jev_decisions_total", outcome="allow" if decision.allow else "hold", route=decision.route
        )
        state.decisions.record(request_id=_request_id(request), action=payload.proposed_action, decision=decision)
        return DecideResponse(
            allow=decision.allow,
            route=decision.route,
            reasons=decision.reasons,
            rationale=decision.rationale,
            decisions=decision.record["decisions"],
            outputs=decision.record["outputs"],
            latency_ms=decision.latency_ms,
            request_id=_request_id(request),
            metadata=decision.metadata,
        )

    # --- dashboard --------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard() -> HTMLResponse:
        """The console. One file, no build step, same origin as the API."""
        if not DASHBOARD_HTML.exists():  # pragma: no cover - packaging failure
            raise HTTPException(status_code=404, detail="dashboard.html is not installed")
        return HTMLResponse(DASHBOARD_HTML.read_text(encoding="utf-8"))

    @app.get("/v1/decisions", tags=["dashboard"], dependencies=[Depends(require_api_key)])
    async def decisions(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
        """Recent decisions, newest first. In memory only -- inputs are user data."""
        return {"decisions": state.decisions.recent(limit), "total": len(state.decisions)}

    @app.get("/v1/calibration", tags=["dashboard"], dependencies=[Depends(require_api_key)])
    async def calibration() -> dict[str, Any]:
        """The fitted parameters in force, and the report that produced them."""
        return {
            "calibrated": state.calibrated,
            "policy": {
                "autonomy_threshold": state.gate.autonomy_threshold,
                "max_risk_level": state.gate.max_risk_level,
                "max_blast_level": state.gate.max_blast_level,
            },
            **calibration_summary(settings, fields=dict(state.gate.gate.fields)),
        }

    @app.get("/v1/lens", tags=["dashboard"], dependencies=[Depends(require_api_key)])
    async def lens(
        threshold: float = Query(default=None, ge=0.0, le=1.0),
        refresh: bool = Query(default=False),
    ) -> dict[str, Any]:
        """One row per labelled example: the human label beside the model's probability.

        This is what makes the threshold trade-off visible. The first call runs
        the dataset through the gate; DSPy's cache makes later calls free.
        """
        if not state.ready:
            raise HTTPException(status_code=503, detail=state.lm_error or "no language model configured")
        if state._lens is None or refresh:
            try:
                state._lens = lens_rows(state.gate, settings)
            except Exception as exc:  # reported to the dashboard rather than crashing it
                logger.exception("lens.failed")
                raise HTTPException(status_code=502, detail=f"lens failed: {exc}") from exc
        cut = state.gate.autonomy_threshold if threshold is None else threshold
        return {
            **state._lens,
            "threshold": cut,
            "tradeoff": threshold_tradeoff(state._lens["rows"], cut),
        }

    @app.post(
        "/v1/triage",
        response_model=TriageResponse,
        tags=["decide"],
        dependencies=[Depends(require_api_key)],
    )
    async def triage(payload: TriageRequest, request: Request) -> TriageResponse:
        """Score a support ticket against the triage rubric."""
        if not state.ready:
            raise HTTPException(status_code=503, detail=state.lm_error or "no language model configured")
        try:
            record = decision_record(state.triage(ticket=payload.ticket))
        except Exception as exc:
            logger.exception("triage.failed", extra={"request_id": _request_id(request)})
            raise HTTPException(status_code=502, detail=f"triage failed: {exc}") from exc
        return TriageResponse(decisions=record["decisions"], outputs=record["outputs"], request_id=_request_id(request))

    return app


def app_factory() -> FastAPI:  # pragma: no cover - uvicorn entry point
    """``uvicorn dspy_jev.service.app:app_factory --factory``"""
    return create_app()
