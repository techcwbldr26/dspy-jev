"""Command line interface.

``dspy-jev decide`` is deliberately a first-class surface, not a debugging aid:
Pi's guidance is to prefer a CLI tool with a README over an MCP server, so the
Pi skill shells out to this command while Claude Code and hermes-agent use the
MCP server or the HTTP service against the same code path.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dspy_jev import __version__, models
from dspy_jev.config import Settings, get_settings

EXIT_OK = 0
EXIT_HOLD = 10  # decide: a decision was produced, and it was not "allow"
EXIT_ERROR = 1


def _emit(payload: Any, *, pretty: bool = True) -> None:
    print(json.dumps(payload, indent=2 if pretty else None, default=str))


def _settings_from_args(args: argparse.Namespace) -> Settings:
    overrides: dict[str, Any] = {}
    if getattr(args, "harness", None):
        overrides["harness"] = args.harness
    if getattr(args, "model", None):
        overrides["decision_model"] = args.model
    return get_settings(**overrides) if overrides else get_settings()


# --- doctor ---------------------------------------------------------------------


def _fetch_ollama_tags(timeout: float = 15.0) -> list[str]:
    """Live model names from Ollama Cloud, so a retirement shows up as a check."""
    request = urllib.request.Request(models.OLLAMA_TAGS_URL, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    entries = payload.get("models", payload) if isinstance(payload, dict) else payload
    names: list[str] = []
    for entry in entries or []:
        if isinstance(entry, dict):
            name = entry.get("name") or entry.get("model")
            if name:
                names.append(str(name))
    return names


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check everything a harness needs before its first decision."""
    settings = _settings_from_args(args)
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    check("harness", True, settings.harness)
    check("provider", True, settings.provider)

    key = settings.api_key_for()
    check(
        "credentials",
        bool(key),
        "present"
        if key
        else (
            "set OLLAMA_API_KEY (https://ollama.com/settings/keys)"
            if settings.provider == "ollama_cloud"
            else "set ANTHROPIC_API_KEY"
        ),
    )

    for role in ("decision", "fast", "judge"):
        try:
            spec = models.resolve(settings.model_for(role))
            check(f"model.{role}", True, spec.dspy_model)
        except models.UnknownModelError as exc:
            check(f"model.{role}", False, str(exc))

    if settings.provider == "ollama_cloud" and not args.offline:
        try:
            live = _fetch_ollama_tags()
            check("ollama.reachable", bool(live), f"{len(live)} cloud models listed")
            for role in ("decision", "fast", "judge"):
                wanted = settings.model_for(role)
                found = any(tag == wanted or tag.split(":")[0] == wanted.split(":")[0] for tag in live)
                check(
                    f"ollama.available.{role}",
                    found,
                    wanted if found else f"{wanted} is not in the live cloud listing (retired or renamed?)",
                )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            check("ollama.reachable", False, f"{type(exc).__name__}: {exc}")

    artifact = settings.calibrated_artifact
    check(
        "calibration",
        artifact.exists(),
        str(artifact) if artifact.exists() else f"no artifact at {artifact}; run `dspy-jev calibrate`",
    )

    for module, extra in (("fastapi", "service"), ("mlflow", "observability"), ("mcp", "mcp")):
        try:
            __import__(module)
            check(f"optional.{module}", True, "installed")
        except ImportError:
            check(f"optional.{module}", False, f"pip install 'dspy-jev[{extra}]'")

    failures = [c for c in checks if not c["ok"] and not c["check"].startswith("optional.")]
    _emit({"version": __version__, "checks": checks, "ok": not failures, "failures": len(failures)})
    return EXIT_OK if not failures else EXIT_ERROR


# --- models ---------------------------------------------------------------------


def cmd_models(args: argparse.Namespace) -> int:
    """List the registry, and optionally reconcile it with the live cloud listing."""
    settings = _settings_from_args(args)
    live: list[str] | None = None
    if args.live:
        try:
            live = _fetch_ollama_tags()
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            print(f"warning: could not reach {models.OLLAMA_TAGS_URL}: {exc}", file=sys.stderr)

    rows = []
    for spec in (*models.OLLAMA_CLOUD_MODELS, *models.ANTHROPIC_MODELS):
        row: dict[str, Any] = {
            "name": spec.name,
            "provider": spec.provider,
            "dspy_model": spec.dspy_model,
            "open_weight": spec.open_weight,
            "notes": spec.notes,
        }
        if live is not None and spec.provider == "ollama_cloud":
            row["live"] = any(t == spec.name or t.split(":")[0] == spec.name.split(":")[0] for t in live)
        rows.append(row)

    _emit(
        {
            "harness": settings.harness,
            "roles": {r: settings.model_for(r) for r in ("decision", "fast", "judge")},
            "models": rows,
        }
    )
    return EXIT_OK


# --- decide / triage ------------------------------------------------------------


def _prepare_runtime(settings: Settings, *, calibration: Path | None = None):
    import dspy

    from dspy_jev.lm import build_lm
    from dspy_jev.observability import DecisionAuditCallback, configure_observability
    from dspy_jev.program import ActionGateProgram

    configure_observability(settings)
    lm = build_lm("decision", settings=settings)
    dspy.configure(lm=lm, callbacks=[DecisionAuditCallback(settings)])
    gate = ActionGateProgram(settings=settings)
    gate.load_calibration(calibration)
    return gate


def cmd_decide(args: argparse.Namespace) -> int:
    """Gate one action. Exit 0 when allowed, 10 when held, 1 on failure."""
    settings = _settings_from_args(args)
    if args.autonomy_threshold is not None:
        settings = get_settings(
            harness=settings.harness,
            autonomy_threshold=args.autonomy_threshold,
            decision_model=settings.decision_model,
        )
    context = args.context
    if args.context_file:
        context = Path(args.context_file).read_text(encoding="utf-8")

    gate = _prepare_runtime(settings, calibration=Path(args.calibration) if args.calibration else None)
    decision = gate.decide(task=args.task, proposed_action=args.action, context=context)
    payload = decision.to_dict()
    payload["decisions"] = payload["record"]["decisions"]
    payload["outputs"] = payload["record"]["outputs"]
    payload.pop("record")
    _emit(payload, pretty=not args.compact)
    return EXIT_OK if decision.allow else EXIT_HOLD


def cmd_triage(args: argparse.Namespace) -> int:
    import dspy

    from dspy_jev.decisions import decision_record
    from dspy_jev.lm import build_lm
    from dspy_jev.observability import configure_observability
    from dspy_jev.program import TicketTriageProgram

    settings = _settings_from_args(args)
    configure_observability(settings)
    dspy.configure(lm=build_lm("decision", settings=settings))
    _emit(decision_record(TicketTriageProgram()(ticket=args.ticket)), pretty=not args.compact)
    return EXIT_OK


# --- calibrate / evaluate -------------------------------------------------------


def cmd_calibrate(args: argparse.Namespace) -> int:
    import dspy

    from dspy_jev.calibrate import calibrate_action_gate
    from dspy_jev.lm import build_lm
    from dspy_jev.observability import configure_observability

    settings = _settings_from_args(args)
    configure_observability(settings)
    dspy.configure(lm=build_lm("decision", settings=settings))
    result = calibrate_action_gate(
        dataset=args.dataset,
        settings=settings,
        train_fraction=args.train_fraction,
        require_cache=not args.allow_uncached,
        artifact_path=args.out,
        num_threads=args.num_threads,
    )
    _emit(result.to_dict())
    return EXIT_OK


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Score the current (calibrated) gate on a dataset, without fitting anything."""
    import dspy

    from dspy_jev.data import ActionGateExample, load_examples
    from dspy_jev.lm import build_lm
    from dspy_jev.metrics import action_gate_metric, false_allow_rate, safety_recall
    from dspy_jev.observability import configure_observability

    settings = _settings_from_args(args)
    configure_observability(settings)
    dspy.configure(lm=build_lm("decision", settings=settings))
    gate = _prepare_runtime(settings, calibration=Path(args.calibration) if args.calibration else None)

    examples = load_examples(args.dataset or settings.data_dir / "action_gate.jsonl", ActionGateExample)
    predictions = [gate(task=e.task, proposed_action=e.proposed_action, context=e.context) for e in examples]
    scores = [action_gate_metric(e, p) for e, p in zip(examples, predictions, strict=True)]
    summary = {
        "examples": len(examples),
        "mean_score": round(sum(scores) / len(scores), 4),
        "safety_recall": round(safety_recall(examples, predictions), 4),
        "false_allow_rate": round(false_allow_rate(examples, predictions), 4),
        "calibrated_fields": dict(gate.gate.fields),
    }
    _emit(summary)
    failed = args.min_safety_recall is not None and summary["safety_recall"] < args.min_safety_recall
    return EXIT_ERROR if failed else EXIT_OK


# --- serve / mcp ----------------------------------------------------------------


def cmd_serve(args: argparse.Namespace) -> int:  # pragma: no cover - runs a server
    import uvicorn

    settings = _settings_from_args(args)
    uvicorn.run(
        "dspy_jev.service.app:app_factory",
        factory=True,
        host=args.host or settings.service_host,
        port=args.port or settings.service_port,
        reload=args.reload,
        log_config=None,
    )
    return EXIT_OK


def cmd_mcp(args: argparse.Namespace) -> int:  # pragma: no cover - runs a server
    from dspy_jev.mcp_server import main as mcp_main

    argv = ["--transport", args.transport]
    if args.service_url:
        argv += ["--service-url", args.service_url]
    if args.host:
        argv += ["--host", args.host]
    if args.port:
        argv += ["--port", str(args.port)]
    return mcp_main(argv)


# --- parser ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dspy-jev", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"dspy-jev {__version__}")
    parser.add_argument(
        "--harness",
        choices=["hermes-agent", "pi", "claude-code"],
        help="Override DSPY_JEV_HARNESS for this invocation.",
    )
    parser.add_argument("--model", help="Override the decision model by registry name.")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Verify credentials, models, calibration and optional extras.")
    doctor.add_argument("--offline", action="store_true", help="Skip the live Ollama Cloud model listing.")
    doctor.set_defaults(func=cmd_doctor)

    models_cmd = sub.add_parser("models", help="List the model registry.")
    models_cmd.add_argument("--live", action="store_true", help="Reconcile with the live cloud listing.")
    models_cmd.set_defaults(func=cmd_models)

    decide = sub.add_parser("decide", help="Gate one proposed action. Exit 10 when the gate holds it.")
    decide.add_argument("--task", required=True)
    decide.add_argument("--action", required=True, help="The proposed next step.")
    decide.add_argument("--context", default="")
    decide.add_argument("--context-file", help="Read context from a file instead.")
    decide.add_argument("--autonomy-threshold", type=float, default=None)
    decide.add_argument("--calibration", help="Path to a calibration artifact.")
    decide.add_argument("--compact", action="store_true", help="Single-line JSON.")
    decide.set_defaults(func=cmd_decide)

    triage = sub.add_parser("triage", help="Score a support ticket.")
    triage.add_argument("--ticket", required=True)
    triage.add_argument("--compact", action="store_true")
    triage.set_defaults(func=cmd_triage)

    calibrate = sub.add_parser("calibrate", help="Fit decision parameters with ReAnchor.")
    calibrate.add_argument("--dataset")
    calibrate.add_argument("--out", help="Artifact path. Defaults to the per-harness artifact.")
    calibrate.add_argument("--train-fraction", type=float, default=0.7)
    calibrate.add_argument("--num-threads", type=int, default=None)
    calibrate.add_argument(
        "--allow-uncached",
        action="store_true",
        help="Calibrate even when the client does not cache. Costs one request per candidate.",
    )
    calibrate.set_defaults(func=cmd_calibrate)

    evaluate = sub.add_parser("evaluate", help="Score the gate on a dataset without fitting.")
    evaluate.add_argument("--dataset")
    evaluate.add_argument("--calibration")
    evaluate.add_argument(
        "--min-safety-recall",
        type=float,
        default=None,
        help="Exit non-zero below this. Use it as a CI gate.",
    )
    evaluate.set_defaults(func=cmd_evaluate)

    serve = sub.add_parser("serve", help="Run the HTTP service.")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    mcp = sub.add_parser("mcp", help="Run the MCP server.")
    mcp.add_argument("--transport", choices=["stdio", "streamable-http", "sse"], default="stdio")
    mcp.add_argument("--service-url", help="Proxy to a running HTTP service instead of running in-process.")
    mcp.add_argument("--host")
    mcp.add_argument("--port", type=int)
    mcp.set_defaults(func=cmd_mcp)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        return 130
    except Exception as exc:
        _emit({"error": type(exc).__name__, "detail": str(exc)})
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
