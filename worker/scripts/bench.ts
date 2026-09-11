// Node 22.18 or newer is required to run TypeScript directly (type stripping).
//
// Drives the DEPLOYED worker through its streaming chat protocol (see
// worker/src/index.ts: POST /api/chat, headers content-type: application/json,
// accept: application/x-ndjson, optional x-tester-key; response lines
// {type:"progress",stage} then one {type:"result",data:{...}}) and prints
// docs/benchmark.md-shaped table rows. Native fetch only, no dependencies.
//
// Refuses to run against the network without TESTER_KEY: a run without it
// counts against the public daily budget and the per-IP conversation cap
// (see worker/src/rateLimit.ts), which is a cost this script must never
// impose silently.

import { appendFileSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

interface Case {
  id: string;
  description: string;
  answers: string[];
  expected: { status?: string; entry_codes?: string[]; outcome?: string };
}

// Mirrors the union the worker's /api/chat "result" line carries (see
// worker/src/index.ts runOnce() and worker/src/loop.ts TurnResult) — only the
// fields this script reads.
interface ResultData {
  type: "question" | "verdict" | "pathway" | "error";
  text?: string;
  messages?: unknown[];
  verdict?: { status: string; entry_codes?: string[] };
  pathway?: { outcome?: string };
  reason?: string;
}

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_DOCS = join(SCRIPT_DIR, "..", "..", "docs");

const USAGE = `Usage: npm run bench -- --cases <file.json> [--url <worker-url>] [--out <file.md>] [--dry-run]

  --cases <file>   JSON file of cases: [{id, description, answers: [string...],
                    expected: {status, entry_codes?, outcome?}}]
  --url <url>      Worker base URL (default: WORKER_URL in docs/config.js)
  --out <file>     Append the resulting markdown table rows to this file
                    (rows are always printed to stdout too)
  --dry-run        Print the plan (cases, url, canned-answer counts) and exit
                    without making any network call
  --help           Show this message

Requires the TESTER_KEY environment variable (sent as x-tester-key) for any
real run — a run without it counts against the worker's public daily budget
and per-IP conversation cap.`;

function defaultUrlFromConfig(): string {
  const configPath = join(REPO_DOCS, "config.js");
  let text: string;
  try {
    text = readFileSync(configPath, "utf8");
  } catch (err) {
    throw new Error(`could not read ${configPath}: ${(err as Error).message}`);
  }
  const m = text.match(/WORKER_URL:\s*"([^"]+)"/);
  if (!m) throw new Error(`WORKER_URL not found in ${configPath}`);
  return m[1];
}

function parseArgs(argv: string[]): {
  cases?: string;
  url?: string;
  out?: string;
  dryRun: boolean;
  help: boolean;
} {
  const args: { cases?: string; url?: string; out?: string; dryRun: boolean; help: boolean } = {
    dryRun: false,
    help: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--help" || a === "-h") args.help = true;
    else if (a === "--dry-run") args.dryRun = true;
    else if (a === "--cases") args.cases = argv[++i];
    else if (a === "--url") args.url = argv[++i];
    else if (a === "--out") args.out = argv[++i];
    else {
      console.error(`unknown argument: ${a}\n`);
      console.error(USAGE);
      process.exit(1);
    }
  }
  return args;
}

// Accepts a bare JSON array of cases, or an object {_comment?, cases: [...]}
// — the object form is what lets a cases file carry a top-level comment
// (see bench-cases.example.json) since plain JSON has no comment syntax.
function loadCases(path: string): Case[] {
  const raw = JSON.parse(readFileSync(path, "utf8"));
  const list = Array.isArray(raw) ? raw : (raw as { cases?: unknown }).cases;
  if (!Array.isArray(list)) {
    throw new Error(`${path} must be a JSON array of cases, or an object with a "cases" array`);
  }
  return list.filter((c) => c && typeof c === "object" && "id" in c) as Case[];
}

// Reads the NDJSON stream one line at a time, same framing the page's own
// docs/app.js readNdjson() relies on: progress lines update onProgress,
// exactly one result line carries the turn's data.
async function readNdjson(resp: Response, onProgress: (stage: string) => void): Promise<ResultData> {
  if (!resp.body) throw new Error("empty response body");
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  let data: ResultData | null = null;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl: number;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      let obj: { type?: string; stage?: string; data?: ResultData };
      try {
        obj = JSON.parse(line);
      } catch {
        continue;
      }
      if (obj.type === "progress" && obj.stage) onProgress(obj.stage);
      else if (obj.type === "result" && obj.data) data = obj.data;
    }
  }
  if (!data) throw new Error("stream ended without a result line");
  return data;
}

async function sendTurn(
  url: string,
  messages: unknown[],
  testerKey: string,
  caseId: string,
): Promise<ResultData> {
  const resp = await fetch(url.replace(/\/$/, "") + "/api/chat", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      accept: "application/x-ndjson",
      "x-tester-key": testerKey,
    },
    body: JSON.stringify({ messages }),
  });
  const contentType = resp.headers.get("content-type") ?? "";
  if (contentType.includes("application/x-ndjson")) {
    return readNdjson(resp, (stage) => {
      process.stderr.write(`  [${caseId}] ${stage}\n`);
    });
  }
  // Every non-streaming reply the worker sends (CORS/budget/bad-request
  // rejections before the streaming decision is even reached) is plain JSON.
  const body = (await resp.json()) as ResultData;
  if (!resp.ok) return { type: "error", reason: body.reason ?? `http_${resp.status}` };
  return body;
}

function mdEscape(s: string): string {
  return s.replace(/\|/g, "\\|").replace(/\r?\n/g, " ").trim();
}

function obtainedLabel(data: ResultData): string {
  if (data.type === "verdict") {
    const v = data.verdict;
    if (!v) return "verdict (no data)";
    return v.entry_codes && v.entry_codes.length ? `${v.status} (${v.entry_codes.join(",")})` : v.status;
  }
  if (data.type === "pathway") {
    const v = data.verdict;
    const outcome = data.pathway?.outcome ?? "?";
    return v ? `${v.status} / pathway:${outcome}` : `pathway:${outcome}`;
  }
  if (data.type === "error") return `error: ${data.reason ?? "unknown"}`;
  return "incomplete (ran out of canned answers)";
}

function isCorrect(data: ResultData, expected: Case["expected"]): boolean {
  if (data.type === "verdict") {
    if (!data.verdict) return false;
    if (expected.status && data.verdict.status !== expected.status) return false;
    if (expected.entry_codes) {
      const got = new Set(data.verdict.entry_codes ?? []);
      const want = new Set(expected.entry_codes);
      if (got.size !== want.size) return false;
      for (const c of want) if (!got.has(c)) return false;
    }
    return Boolean(expected.status || expected.entry_codes);
  }
  if (data.type === "pathway") {
    let ok = false;
    if (expected.status) {
      if (!data.verdict || data.verdict.status !== expected.status) return false;
      ok = true;
    }
    if (expected.outcome) {
      if (!data.pathway || data.pathway.outcome !== expected.outcome) return false;
      ok = true;
    }
    return ok;
  }
  return false;
}

async function runCase(
  url: string,
  testerKey: string,
  c: Case,
): Promise<{
  data: ResultData;
  turns: number;
  firstResultMs: number;
  totalMs: number;
  note: string;
}> {
  let messages: unknown[] = [{ role: "user", content: c.description }];
  let turns = 0;
  let firstResultMs = -1;
  let answerIndex = 0;
  let note = "";
  const start = Date.now();
  let data: ResultData = { type: "error", reason: "no_turns_run" };
  for (;;) {
    turns++;
    data = await sendTurn(url, messages, testerKey, c.id);
    if (firstResultMs < 0) firstResultMs = Date.now() - start;
    if (data.type === "verdict" || data.type === "pathway" || data.type === "error") break;
    // data.type === "question"
    if (answerIndex >= c.answers.length) {
      note = "stopped: ran out of canned answers";
      break;
    }
    if (!data.messages) {
      note = "stopped: question result carried no messages to continue from";
      break;
    }
    const next = c.answers[answerIndex++];
    messages = [...data.messages, { role: "user", content: next }];
  }
  const totalMs = Date.now() - start;
  return { data, turns, firstResultMs, totalMs, note };
}

function planLine(c: Case): string {
  const desc = c.description.length > 60 ? c.description.slice(0, 57) + "..." : c.description;
  const exp =
    c.expected.status ?? (c.expected.outcome ? `pathway:${c.expected.outcome}` : "(no status set)");
  return `  - ${c.id}: "${desc}" (${c.answers.length} canned answer(s), expects ${exp})`;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log(USAGE);
    return;
  }
  if (!args.cases) {
    console.error("missing --cases <file.json>\n");
    console.error(USAGE);
    process.exit(1);
  }
  const cases = loadCases(args.cases);
  if (cases.length === 0) {
    console.error(`${args.cases} contains no cases`);
    process.exit(1);
  }
  let url: string;
  try {
    url = args.url ?? defaultUrlFromConfig();
  } catch (err) {
    console.error((err as Error).message);
    process.exit(1);
    return;
  }

  if (args.dryRun) {
    console.log(`plan: ${cases.length} case(s) against ${url}`);
    for (const c of cases) console.log(planLine(c));
    console.log(args.out ? `rows will be appended to ${args.out}` : "rows will only be printed to stdout");
    console.log("dry run: no network call made");
    return;
  }

  const testerKey = process.env.TESTER_KEY;
  if (!testerKey) {
    console.error(
      "TESTER_KEY is not set. Refusing to run: an untested run counts against the " +
        "worker's public daily budget and per-IP conversation cap. Set TESTER_KEY and retry.",
    );
    process.exit(1);
    return;
  }

  const rows: string[] = [];
  for (const c of cases) {
    console.error(`running case ${c.id}...`);
    const { data, turns, firstResultMs, totalMs, note } = await runCase(url, testerKey, c);
    const obtained = obtainedLabel(data);
    const correct = isCorrect(data, c.expected);
    const expectedLabel =
      [c.expected.status, c.expected.entry_codes?.join(","), c.expected.outcome ? `pathway:${c.expected.outcome}` : ""]
        .filter(Boolean)
        .join(" ") || "(unset)";
    const row = [
      c.id,
      mdEscape(c.description),
      mdEscape(expectedLabel),
      mdEscape(obtained),
      correct ? "yes" : "no",
      String(turns),
      (firstResultMs / 1000).toFixed(2),
      (totalMs / 1000).toFixed(2),
      mdEscape(note),
    ];
    const line = `| ${row.join(" | ")} |`;
    rows.push(line);
    console.log(line);
  }

  if (args.out) {
    appendFileSync(args.out, rows.join("\n") + "\n");
    console.error(`appended ${rows.length} row(s) to ${args.out}`);
  }
}

main().catch((err) => {
  console.error(err instanceof Error ? err.stack ?? err.message : err);
  process.exit(1);
});
