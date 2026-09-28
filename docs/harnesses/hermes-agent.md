# hermes-agent harness

Open-weight models on Ollama Cloud. The gate runs as a standalone HTTP service
and hermes-agent calls it as a tool, so the harness carries no DSPy dependency
and no model credentials.

## Setup

```bash
export OLLAMA_API_KEY=...          # https://ollama.com/settings/keys
./scripts/setup_hermes_agent.sh
source .venv/bin/activate
dspy-jev calibrate --dataset data/action_gate.jsonl
dspy-jev serve                     # 127.0.0.1:8080
```

## Wiring

1. Copy `harnesses/hermes-agent/profiles/dspy-jev.yaml` into hermes-agent's
   profiles directory.
2. Append `harnesses/hermes-agent/prompts/system-preamble.md` to the system
   prompt.

> The profile uses a common "OpenAPI-ish tool declaration" layout. If your
> hermes-agent build spells the keys differently, the values are what matter:
> one `POST /v1/decide` with `task`, `proposed_action` and `context`, returning
> `allow`, `route` and `reasons`. `tests/integration/test_harness_assets.py`
> checks the endpoints the profile names actually exist in the service.

The alternative is MCP: `harnesses/hermes-agent/.mcp.json` runs a stdio server
that proxies to the same service, so both routes share one calibration artifact.

## Configuration

| Variable | Set on | Purpose |
|---|---|---|
| `OLLAMA_API_KEY` | the **service** | Provider credential. Never on the harness. |
| `DSPY_JEV_SERVICE_URL` | the **harness** | Where the gate lives. |
| `DSPY_JEV_SERVICE_API_KEY` | both | Sent as `X-API-Key`; required when set. |
| `DSPY_JEV_AUTONOMY_THRESHOLD` | the service | Policy floor on `P(safe_to_proceed)`. |

## The two rules that matter

**A hold is binding.** `allow: false` halts the step. The agent reports `route`
and `reasons`; it does not reword the action and try again. The gate reads the
action text, and rephrasing to get past it is the behaviour it exists to catch.

**An error is a hold.** The profile sets `on_error: halt_and_report`. A gate
that is unreachable must never read as permission. Rehearse this: stop the
service and confirm the agent stops too.

## Verifying

```bash
curl -s localhost:8080/healthz | jq
curl -s -X POST localhost:8080/v1/decide -H 'content-type: application/json' -d '{
  "task": "free up disk space",
  "proposed_action": "delete the contents of /var/lib/docker",
  "context": "shared CI runner, three teams building now"
}' | jq '{allow, route, reasons}'
```

Expect `allow: false` and a non-empty `reasons`. Then drive the same action
through hermes-agent itself and confirm it halts rather than proceeding.

## Operating

- Alert on `dspy_jev_decisions_total{outcome="error"}` — that is user-visible
  friction, since errors halt the agent.
- Sample held decisions weekly. Every wrong hold and wrong allow is a new row in
  `data/action_gate.jsonl`.
- Recalibrate after ~10 new rows. With a warm cache it costs no new inference.
