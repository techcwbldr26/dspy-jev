"""Command line interface.

``dspy-jev decide`` is deliberately a first-class surface, not a debugging aid:
Pi's guidance is to prefer a CLI tool with a README over an MCP server, so the
Pi skill shells out to this command while Claude Code and hermes-agent use the
MCP server or the HTTP service against the same code path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
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


def _probe_provider(settings: Settings) -> tuple[bool, str]:
    """Send one tiny real request and report whether authentication worked.

    The only way to tell a working key from a well-formed one -- and the only way
    to know whether a platform-injected Authorization header is actually reaching
    the provider.
    """
    import dspy

    from dspy_jev.lm import build_lm

    try:
        lm = build_lm("fast", settings=settings, max_tokens=8, cache=False)
    except Exception as exc:  # reported to the caller, not raised
        return False, f"{type(exc).__name__}: {exc}"
    try:
        with dspy.context(lm=lm):
            lm("Reply with the single word: ok")
    except Exception as exc:  # reporting this is the whole point of a probe
        detail = str(exc)
        if "401" in detail or "invalid_api_key" in detail or "Unauthorized" in detail:
            hint = (
                " -- the credential was rejected. In key mode check OLLAMA_API_KEY; in proxy "
                "mode check that the platform allows this host and injects Authorization."
            )
        elif "404" in detail:
            hint = " -- model not found. Run `dspy-jev models --live` for the served tags."
        else:
            hint = ""
        return False, f"{type(exc).__name__}: {detail[:300]}{hint}"
    return True, f"{lm.model} answered"


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check everything a harness needs before its first decision."""
    settings = _settings_from_args(args)
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    check("harness", True, settings.harness)
    check("provider", True, settings.provider)

    check("auth.mode", True, settings.auth_description)
    key = settings.api_key_for()
    check(
        "credentials",
        bool(key),
        settings.auth_description
        if key
        else (
            "set OLLAMA_API_KEY (https://ollama.com/settings/keys), or DSPY_JEV_AUTH_MODE=proxy "
            "if the platform injects the Authorization header"
            if settings.provider == "ollama_cloud"
            else "set ANTHROPIC_API_KEY, or DSPY_JEV_AUTH_MODE=proxy"
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
                wanted = models.resolve(settings.model_for(role)).name
                check(f"ollama.available.{role}", *_availability(wanted, live))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            check("ollama.reachable", False, f"{type(exc).__name__}: {exc}")

    artifact = settings.calibrated_artifact
    check(
        "calibration",
        artifact.exists(),
        str(artifact) if artifact.exists() else f"no artifact at {artifact}; run `dspy-jev calibrate`",
    )

    if args.probe:
        check("provider.probe", *_probe_provider(settings))

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


def _availability(wanted: str, live: list[str]) -> tuple[bool, str]:
    """Check one model against the live listing.

    The cloud API wants the exact name the listing returns, so a family-level
    near-miss (``deepseek-v4-pro`` when the listing says ``deepseek-v4-pro:0813``)
    is a failure that names the right tag -- not a pass. Treating it as a pass is
    how a 404 at the first real request gets missed here.
    """
    if wanted in live:
        return True, wanted
    family = wanted.split(":")[0]
    near = [tag for tag in live if tag.split(":")[0] == family]
    if near:
        return False, f"{wanted} is not served exactly; the listing offers {', '.join(sorted(near))}"
    return False, f"{wanted} is not in the live cloud listing (retired or renamed?)"


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
            row["live"] = spec.name in live
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
    from dspy_jev.observability import configure_observability, install_audit_callback
    from dspy_jev.program import ActionGateProgram

    configure_observability(settings)
    lm = build_lm("decision", settings=settings)
    dspy.configure(lm=lm)
    install_audit_callback(settings)
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


# --- guard ----------------------------------------------------------------------


def cmd_guard(args: argparse.Namespace) -> int:
    """Gate a command, then run it only if the gate allows.

    This is the enforcement primitive for harnesses with no tool-call hook of
    their own. Asking the gate and then running the command separately leaves a
    window in which the verdict is advice; ``guard`` closes it, because the only
    path to execution runs through an allow.
    """
    import os

    from dspy_jev.enforce import Enforcer, ToolCall

    if not args.command:
        _emit({"error": "NothingToRun", "detail": "Pass the command after `--`."})
        return EXIT_ERROR

    settings = _settings_from_args(args)
    call = ToolCall(tool="bash", arguments={"command": " ".join(args.command)})
    enforcer = Enforcer(
        settings=settings,
        service_url=args.service_url or os.environ.get("DSPY_JEV_SERVICE_URL"),
        # Without a person to answer, `needs_review` and `clarify` are a refusal.
        interactive=False,
    )
    verdict = enforcer.evaluate(call, task=args.task, context=args.context)

    if verdict.blocked:
        _emit({"ran": False, **verdict.to_dict(), "command": args.command})
        return EXIT_ERROR if verdict.error else EXIT_HOLD

    if not args.quiet:
        print(
            json.dumps({"ran": True, "gated": verdict.gated, "route": verdict.route or "skipped"}),
            file=sys.stderr,
        )
    # The caller's own command, now gated: the only path here is through an allow.
    completed = subprocess.run(args.command)
    return completed.returncode


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
    doctor.add_argument(
        "--probe",
        action="store_true",
        help="Send one tiny real request to prove authentication works end to end.",
    )
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

    guard = sub.add_parser(
        "guard",
        help="Gate a command and run it only if allowed. Exit 10 when held, 1 when the gate fails.",
    )
    guard.add_argument("--task", default="No task was recorded for this invocation.")
    guard.add_argument("--context", default="")
    guard.add_argument("--service-url", help="Use a running dspy-jev service instead of this process.")
    guard.add_argument("--quiet", action="store_true", help="Do not report an allowed run on stderr.")
    guard.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        metavar="-- COMMAND ...",
        help="The command to gate. Everything after `--` is passed through verbatim.",
    )
    guard.set_defaults(func=cmd_guard)

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
    if getattr(args, "command", None) and args.command[0] == "--":
        args.command = args.command[1:]
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        return 130
    except Exception as exc:
        _emit({"error": type(exc).__name__, "detail": str(exc)})
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
