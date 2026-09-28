# dspy-jev gate — Pi extension

Turns the decision gate from advice into a control. Pi's `tool_call` event can
block a tool before it runs, and Pi treats a handler that throws as a block too,
so there is no path where a failure becomes permission.

## Install

```bash
cp -R harnesses/pi/extensions/dspy-jev-gate ~/.pi/agent/extensions/
dspy-jev serve &
export DSPY_JEV_SERVICE_URL=http://127.0.0.1:8080
pi
```

Without `DSPY_JEV_SERVICE_URL` the extension shells out to `dspy-jev decide`
instead, which is slower but needs no running service. Either way the `dspy-jev`
command must be on `PATH` in the shell that starts Pi.

## What it gates

Reads, globs, greps and read-only shell commands (`ls`, `git status`, `cat`) go
straight through — no model call, no latency. Everything else is gated:
`bash` that writes or reaches the network, `write`, `edit`, `webfetch`, and any
tool the extension does not recognise.

A read-only command line stops being read-only the moment it redirects, pipes
into a shell, chains with `;` or `&&`, substitutes a command, or mentions
`curl`, `ssh` or `sudo`. Those all get gated.

## What happens on a hold

| Route | Behaviour |
|---|---|
| `auto_execute` | Runs |
| `needs_review`, `clarify` | Pi asks you; declining blocks the call |
| `block` | Blocked, with the reasons shown to the model |
| gate unreachable | Blocked — a gate that cannot answer is not permission |

## Commands

| Command | Effect |
|---|---|
| `/gate-status` | Where the gate is and whether it answers right now |
| `/gate-off` | Stop enforcing for this session (announced in the UI) |
| `/gate-on` | Resume enforcing |

## Keeping it honest

The triage rules are duplicated here in TypeScript and in
`src/dspy_jev/enforce.py`. `tests/integration/test_enforcement_parity.py` runs
both over the same cases and fails when they disagree, so the two cannot drift
apart silently.
