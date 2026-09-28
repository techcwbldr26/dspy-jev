# hermes-agent harness

Open-weight models on Ollama Cloud. The gate runs as a standalone HTTP service;
the profile calls it and carries no model credentials of its own.

Shipped as a **Hermes profile distribution** — the same shape as the profiles in
[`hermes-agent-profiles`](https://github.com/techcwbldr26/hermes-agent-profiles):
a self-contained directory that `hermes profile install` consumes.

```
harnesses/hermes-agent/decision-gate/
├── distribution.yaml       name, version, hermes_requires, env_requires
├── config.yaml             model + provider (no base_url — it 404s)
├── SOUL.md                 scope, routes, working style, out of scope
├── README.md               who it is for, install, troubleshooting
├── docs/README.md          where the rubric and thresholds live
├── skills/decision-gate/   the skill the agent loads
└── cron/                   (none yet)
```

## Setup

```bash
export OLLAMA_API_KEY=...          # https://ollama.com/settings/keys
./scripts/setup_hermes_agent.sh
source .venv/bin/activate
dspy-jev calibrate --dataset data/action_gate.jsonl
dspy-jev serve                     # 127.0.0.1:8080
```

Then install the profile into your Hermes setup:

```bash
cp -R harnesses/hermes-agent/decision-gate /path/to/hermes-agent-profiles/
cd /path/to/hermes-agent-profiles
hermes profile install ./decision-gate --alias
printf 'OLLAMA_API_KEY=<your key>\n' > decision-gate/.env
```

## Enforcement

Hermes has no tool-call hook this project can rely on, so enforcement happens at
the process boundary. The skill instructs the agent to run risky commands
**through** the gate rather than asking and then acting:

```bash
dspy-jev guard --task "$TASK" -- git push --force origin main
```

| Exit | Meaning |
|---|---|
| `0` | Allowed; the command ran, and its own exit code is returned |
| `10` | Held; the command did **not** run |
| `1` | The gate failed; the command did **not** run |

Asking first and running separately leaves a window where the verdict is advice.
`guard` closes it: the only path to execution runs through an allow. An
integration test asserts that a blocked command leaves no trace on disk.

`guard` runs non-interactively, so `needs_review` and `clarify` are a refusal
rather than a prompt — there is nobody to answer. The agent surfaces the reasons
and stops.

The alternative is MCP: `harnesses/hermes-agent/.mcp.json` runs a stdio server
that proxies to the same service, so both routes share one calibration artifact.
That route is advisory; `guard` is the enforcing one.

## Configuration

| Variable | Set on | Purpose |
|---|---|---|
| `OLLAMA_API_KEY` | the **service** | Provider credential. Never on the profile. |
| `DSPY_JEV_SERVICE_URL` | the **profile** | Where the gate lives. |
| `DSPY_JEV_SERVICE_API_KEY` | both | Sent as `X-API-Key`; required when set. |
| `DSPY_JEV_AUTONOMY_THRESHOLD` | the service | Policy floor on `P(safe_to_proceed)`. |

## Repo conventions this profile follows

These come from `hermes-agent-profiles/AGENTS.md`, and tests assert them:

- **Named by role, never by model.** `decision-gate`, not `glm-decision-gate`.
- **No `base_url` under `model:`.** The `ollama-cloud` provider supplies its own;
  a hand-set one causes HTTP 404s.
- **Bump `distribution.yaml` `version:`** on a model change, and name the switch
  in the commit message.
- **`.gitignore` excludes** `.env`, `auth.json`, `memories/`, `sessions/`.

Note the two different model spellings: `config.yaml` uses `glm-5.3:cloud`,
which is the signed-in-Ollama-server convention the `ollama-cloud` provider
expects. The gate service itself uses the exact API tag (`glm-5.3`). They are
separate settings and can differ — the profile's model writes the report, the
service's model makes the judgement.

## Verifying

```bash
curl -s localhost:8080/healthz | jq
dspy-jev guard --task "free up disk space" -- rm -rf /var/lib/docker
# exit 10, and nothing was deleted
```

Then drive the same action through hermes-agent and confirm it halts.

## Operating

- Alert on `dspy_jev_decisions_total{outcome="error"}` — errors halt the agent,
  so they are user-visible friction.
- Sample held decisions weekly. Every wrong hold and wrong allow is a new row in
  `data/action_gate.jsonl`.
- Recalibrate after ~10 new rows. With a warm cache it costs no new inference.
