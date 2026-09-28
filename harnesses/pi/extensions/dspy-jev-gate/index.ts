/**
 * dspy-jev decision gate for Pi.
 *
 * Pi's `tool_call` event can block execution, which is what turns the gate from
 * advice into a control: `return { block: true, reason }` and the tool never
 * runs. Pi also blocks the tool when this handler throws, so a crash here is a
 * denial rather than a hole.
 *
 * Install:
 *   cp -R harnesses/pi/extensions/dspy-jev-gate ~/.pi/agent/extensions/
 *   dspy-jev serve &                       # or leave it unset to use the CLI
 *   export DSPY_JEV_SERVICE_URL=http://127.0.0.1:8080
 *
 * Commands:
 *   /gate-status   report where the gate is, and whether it answers
 *   /gate-off      disable enforcement for this session (announced, and logged)
 *   /gate-on       re-enable it
 */

import { execFile } from "node:child_process";
import { promisify } from "node:util";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const execFileAsync = promisify(execFile);

// --- triage -------------------------------------------------------------------
// Mirrors dspy_jev.enforce.classify. tests/integration/test_enforcement_parity.py
// drives both implementations over the same cases and fails when they disagree.

const READ_ONLY_TOOLS = new Set([
  "read", "glob", "grep", "search", "notebookread", "websearch", "todoread",
  "listmcpresources", "readmcpresource", "askuserquestion", "exitplanmode",
]);

const ALWAYS_GATE_TOOLS = new Set([
  "webfetch", "bash", "powershell", "shell", "execute", "run",
]);

const SHELL_TOOLS = new Set(["bash", "powershell", "shell", "execute", "run"]);

const READ_ONLY_COMMANDS = new Set([
  "cat", "head", "tail", "less", "ls", "ll", "pwd", "wc", "file", "stat", "du", "df",
  "grep", "rg", "egrep", "fgrep", "find", "fd", "which", "whereis", "type", "env",
  "printenv", "date", "uname", "whoami", "id", "echo", "true", "diff", "cmp", "jq",
  "yq", "sort", "uniq", "cut", "awk", "sed", "tree", "basename", "dirname", "realpath",
]);

const READ_ONLY_SUBCOMMANDS: Record<string, Set<string>> = {
  git: new Set(["status", "log", "diff", "show", "branch", "remote", "blame", "ls-files", "describe"]),
  docker: new Set(["ps", "images", "logs", "inspect", "version", "info"]),
  kubectl: new Set(["get", "describe", "logs", "version", "explain"]),
  npm: new Set(["ls", "list", "view", "outdated", "audit"]),
  pip: new Set(["list", "show", "freeze"]),
  uv: new Set(["tree", "version"]),
  poetry: new Set(["show", "check"]),
  cargo: new Set(["tree", "metadata"]),
  go: new Set(["list", "version", "env"]),
};

const ESCALATING_PATTERNS: RegExp[] = [
  /(?<![0-9<>])>{1,2}(?![&>])/,
  /\|\s*(sudo\s+)?(ba|z|k|da)?sh\b/,
  /\b(curl|wget|nc|ncat|ssh|scp|rsync)\b/,
  /\bsudo\b/,
  /\$\(/,
  /`/,
  /;|&&|\|\|/,
];

const COMMAND_KEYS = ["command", "cmd", "script", "args", "commandLine"];
const INTERESTING_KEYS = [
  "command", "cmd", "script", "file_path", "path", "url", "pattern",
  "old_string", "new_string", "content", "query", "prompt",
];
const MAX_VALUE_CHARS = 600;

/** Split a command line the way `shlex.split` does; null on unbalanced quotes. */
export function splitCommand(command: string): string[] | null {
  const argv: string[] = [];
  let current = "";
  let quote: string | null = null;
  let started = false;
  for (let i = 0; i < command.length; i++) {
    const ch = command[i];
    if (quote) {
      if (ch === quote) quote = null;
      else current += ch;
      continue;
    }
    if (ch === '"' || ch === "'") {
      quote = ch;
      started = true;
      continue;
    }
    if (ch === "\\" && i + 1 < command.length) {
      current += command[++i];
      started = true;
      continue;
    }
    if (/\s/.test(ch)) {
      if (current || started) argv.push(current);
      current = "";
      started = false;
      continue;
    }
    current += ch;
    started = true;
  }
  if (quote) return null;
  if (current || started) argv.push(current);
  return argv;
}

export function isReadOnlyCommand(command: string): boolean {
  if (!command.trim()) return false;
  if (ESCALATING_PATTERNS.some((p) => p.test(command))) return false;

  let argv = splitCommand(command);
  if (argv === null || argv.length === 0) return false;

  while (argv.length > 0 && /^[A-Za-z_][A-Za-z0-9_]*=.*$/s.test(argv[0])) argv = argv.slice(1);
  if (argv.length === 0) return false;

  const program = argv[0].split("/").pop() ?? argv[0];
  const subcommands = READ_ONLY_SUBCOMMANDS[program];
  if (subcommands) {
    const rest = argv.slice(1).filter((a) => !a.startsWith("-"));
    return rest.length > 0 && subcommands.has(rest[0]);
  }
  return READ_ONLY_COMMANDS.has(program);
}

export function normaliseTool(tool: string): string {
  const name = tool.trim().toLowerCase();
  return name.startsWith("mcp__") ? (name.split("__").pop() ?? name) : name;
}

export function commandOf(input: Record<string, unknown>): string {
  for (const key of COMMAND_KEYS) {
    const value = input[key];
    if (typeof value === "string" && value.trim()) return value;
    if (Array.isArray(value) && value.length > 0) return value.map(String).join(" ");
  }
  return "";
}

/** True when this call is worth sending to the gate. Unknown tools are gated. */
export function needsGate(tool: string, input: Record<string, unknown>): boolean {
  const name = normaliseTool(tool);
  if (ALWAYS_GATE_TOOLS.has(name)) {
    if (SHELL_TOOLS.has(name) && isReadOnlyCommand(commandOf(input))) return false;
    return true;
  }
  return !READ_ONLY_TOOLS.has(name);
}

function truncate(value: string, limit = MAX_VALUE_CHARS): string {
  const collapsed = value.split(/\s+/).filter(Boolean).join(" ");
  return collapsed.length <= limit
    ? collapsed
    : `${collapsed.slice(0, limit)}… [${collapsed.length} chars total]`;
}

export function renderAction(tool: string, input: Record<string, unknown>): string {
  const name = tool.trim() || "unknown-tool";
  const command = commandOf(input);
  if (command) return `Run shell command: ${truncate(command)}`;

  const parts: string[] = [];
  for (const key of INTERESTING_KEYS) {
    const value = input[key];
    if (typeof value === "string" && value.trim()) parts.push(`${key}=${truncate(value)}`);
  }
  if (parts.length === 0) {
    const keys = Object.keys(input).sort();
    parts.push(keys.length ? `arguments=[${keys.map((k) => `'${k}'`).join(", ")}]` : "no arguments");
  }
  return `Use the ${name} tool with ${parts.join("; ")}`;
}

// --- gate client ---------------------------------------------------------------

interface GateResponse {
  allow?: boolean;
  route?: string;
  reasons?: string[];
  rationale?: string;
  decisions?: Record<string, { probability?: number }>;
}

async function askService(
  serviceUrl: string,
  body: Record<string, string>,
  timeoutMs: number,
): Promise<GateResponse> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (process.env.DSPY_JEV_SERVICE_API_KEY) {
    headers["X-API-Key"] = process.env.DSPY_JEV_SERVICE_API_KEY;
  }
  const response = await fetch(`${serviceUrl.replace(/\/$/, "")}/v1/decide`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!response.ok) {
    throw new Error(`gate returned HTTP ${response.status}: ${await response.text()}`);
  }
  return (await response.json()) as GateResponse;
}

async function askCli(body: Record<string, string>, timeoutMs: number): Promise<GateResponse> {
  // `dspy-jev decide` exits 0 when allowed and 10 when held, so a non-zero exit
  // is expected rather than a failure. Anything else is a real error.
  const args = [
    "decide",
    "--task", body.task,
    "--action", body.proposed_action,
    "--context", body.context,
    "--compact",
  ];
  try {
    const { stdout } = await execFileAsync("dspy-jev", args, { timeout: timeoutMs });
    return JSON.parse(stdout) as GateResponse;
  } catch (error) {
    const stdout = (error as { stdout?: string }).stdout ?? "";
    const code = (error as { code?: number }).code;
    if (code === 10 && stdout.trim()) return JSON.parse(stdout) as GateResponse;
    throw error;
  }
}

function explain(payload: GateResponse): string {
  const route = payload.route ?? "block";
  const headline =
    route === "block"
      ? "The dspy-jev decision gate blocked this action."
      : route === "clarify"
        ? "The dspy-jev decision gate held this action: the task under-specifies it."
        : "The dspy-jev decision gate held this action for human review.";
  const lines = [headline];
  const probability = payload.decisions?.safe_to_proceed?.probability;
  if (typeof probability === "number") lines.push(`P(safe to proceed) = ${probability.toFixed(2)}`);
  if (payload.reasons?.length) lines.push(`Failing conditions: ${payload.reasons.join("; ")}`);
  if (payload.rationale) lines.push(`Rationale: ${payload.rationale}`);
  lines.push(
    "Report this verdict rather than rewording the action and retrying: the gate reads the action text.",
  );
  return lines.join(" ");
}

// --- extension -----------------------------------------------------------------

export default function (pi: ExtensionAPI) {
  const serviceUrl = process.env.DSPY_JEV_SERVICE_URL ?? "";
  const timeoutMs = Number(process.env.DSPY_JEV_TIMEOUT_MS ?? 60_000);
  let enabled = true;
  let task = "The user's current request in this Pi session.";

  pi.on("before_agent_start", async (event) => {
    // Keep the most recent user request as the task the gate judges against.
    const prompt = (event as { prompt?: unknown }).prompt;
    if (typeof prompt === "string" && prompt.trim()) task = prompt.trim().slice(0, 4000);
    return undefined;
  });

  pi.on("tool_call", async (event, ctx) => {
    if (!enabled) return undefined;

    const input = (event.input ?? {}) as Record<string, unknown>;
    if (!needsGate(event.toolName, input)) return undefined;

    const body = {
      task,
      proposed_action: renderAction(event.toolName, input),
      context: `Working directory: ${ctx.cwd ?? process.cwd()}`,
    };

    let payload: GateResponse;
    try {
      payload = serviceUrl
        ? await askService(serviceUrl, body, timeoutMs)
        : await askCli(body, timeoutMs);
    } catch (error) {
      // Fail closed. An unreachable gate is not permission.
      return {
        block: true,
        reason:
          "The dspy-jev decision gate could not be reached, so this action is held. " +
          `A gate that cannot answer is not permission. (${(error as Error).message})`,
      };
    }

    if (payload.allow === true && payload.route === "auto_execute") return undefined;

    // `needs_review` and `clarify` are a question for a person, when there is one.
    if (ctx.hasUI && payload.route !== "block") {
      const choice = await ctx.ui.select(`${explain(payload)}\n\nRun it anyway?`, ["No", "Yes"]);
      if (choice === "Yes") return undefined;
    }
    return { block: true, reason: explain(payload) };
  });

  pi.registerCommand("gate-status", {
    description: "Report where the dspy-jev decision gate is and whether it answers",
    handler: async (_args, ctx) => {
      const where = serviceUrl ? `service ${serviceUrl}` : "the dspy-jev CLI";
      try {
        const probe = await askService(serviceUrl || "http://127.0.0.1:8080", {
          task: "gate status probe",
          proposed_action: "Read a file inside the repository",
          context: "status probe",
        }, 10_000);
        ctx.ui.notify(
          `dspy-jev gate: ${enabled ? "enforcing" : "DISABLED"} via ${where}; ` +
            `probe route=${probe.route ?? "?"}`,
          "info",
        );
      } catch (error) {
        ctx.ui.notify(
          `dspy-jev gate: ${enabled ? "enforcing" : "DISABLED"} via ${where}; ` +
            `probe FAILED (${(error as Error).message}). Every gated tool call will be blocked.`,
          "warning",
        );
      }
    },
  });

  pi.registerCommand("gate-off", {
    description: "Disable dspy-jev enforcement for this session",
    handler: async (_args, ctx) => {
      enabled = false;
      ctx.ui.notify("dspy-jev gate DISABLED for this session. Risky actions will not be checked.", "warning");
    },
  });

  pi.registerCommand("gate-on", {
    description: "Re-enable dspy-jev enforcement",
    handler: async (_args, ctx) => {
      enabled = true;
      ctx.ui.notify("dspy-jev gate enforcing.", "info");
    },
  });
}
