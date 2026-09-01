// Cloudflare Worker entrypoint: CORS (strict origin allowlist — never *),
// budget gates BEFORE any model call, then one classification turn.
// handleRequest takes injectable deps so tests exercise routing, CORS and
// budget behaviour without any network.

import { loadAnnex, type AnnexDataset } from "./annexData";
import { AnthropicClient, type ClaudeClient } from "./claudeClient";
import { InvalidRequest, runTurn, type TurnResult } from "./loop";
import { checkBudget, recordSpend, type KVLike } from "./rateLimit";

// minimal ExecutionContext shape — keeps tests free of workers-types
export interface Ctx {
  waitUntil(promise: Promise<unknown>): void;
}

export interface Env {
  ANTHROPIC_API_KEY: string;
  BUDGET_KV: KVLike;
  ALLOWED_ORIGINS: string; // comma-separated, e.g. "https://rikiosso.github.io"
  ANNEX_URL: string;
  DAILY_BUDGET_USD: string;
  MONTHLY_BUDGET_USD: string;
  IP_DAILY_CONVERSATIONS: string;
  MAX_TURNS: string;
  IP_SALT: string;
  TESTER_KEY?: string;
  LOOP_MODEL: string;
  VERDICT_MODEL: string;
}

export interface Deps {
  annex: (env: Env) => Promise<AnnexDataset>;
  client: (env: Env) => ClaudeClient;
}

const REAL_DEPS: Deps = {
  annex: (env) => loadAnnex(env.ANNEX_URL),
  client: (env) => new AnthropicClient(env.ANTHROPIC_API_KEY),
};

const FIXED_DISCLAIMER =
  "Indicative automated triage only — not legal advice. Catch-all controls " +
  "(Art. 4 and 5, Regulation (EU) 2021/821) may apply regardless of listing; a " +
  "licensing authority or qualified counsel has the final word.";

const PATHWAY_DISCLAIMER =
  "Draft licensing determination — requires review by qualified export-control " +
  "counsel before any reliance. National general licences, sanctions regimes and " +
  "authority practice may change the outcome; only your national competent " +
  "authority can grant or confirm an authorisation.";

function corsHeaders(origin: string | null, allowed: string[]): Record<string, string> {
  const ok = origin !== null && allowed.includes(origin);
  return {
    "access-control-allow-origin": ok ? origin : (allowed[0] ?? ""),
    "access-control-allow-methods": "GET, POST, OPTIONS",
    "access-control-allow-headers": "content-type, x-tester-key",
    vary: "origin",
  };
}

function json(data: unknown, status: number, headers: Record<string, string>): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "content-type": "application/json",
      "x-content-type-options": "nosniff",
      ...headers,
    },
  });
}

function budgetOf(env: Env) {
  return {
    dailyUsd: parseFloat(env.DAILY_BUDGET_USD || "0.30"),
    monthlyUsd: parseFloat(env.MONTHLY_BUDGET_USD || "10"),
    ipDailyConversations: parseInt(env.IP_DAILY_CONVERSATIONS || "2", 10),
    // per-IP request meter: 2x headroom over a full legitimate day
    // (conversations x turns) so error retries never lock a real user out
    ipDailyRequests:
      parseInt(env.IP_DAILY_CONVERSATIONS || "2", 10) * parseInt(env.MAX_TURNS || "10", 10) * 2,
  };
}

// GET /api/health — lets the page show corpus status and an exhausted budget
// BEFORE a visitor types a whole description. Reads KV only; never the model.
// The tester key must count here exactly as it does on /api/chat: the page
// disables its input on an unavailable health result, and a tester who is
// exempt from daily pacing must not be locked out by the page chrome.
async function handleHealth(
  request: Request,
  env: Env,
  deps: Deps,
  cors: Record<string, string>,
): Promise<Response> {
  const day = new Date().toISOString().slice(0, 10);
  const month = day.slice(0, 7);
  const budget = budgetOf(env);
  const testerKey = request.headers.get("x-tester-key");
  const isTester = Boolean(env.TESTER_KEY && testerKey && testerKey === env.TESTER_KEY);
  const daySpend = parseFloat((await env.BUDGET_KV.get(`spend:${day}`)) ?? "0");
  const monthSpend = parseFloat((await env.BUDGET_KV.get(`spend:${month}`)) ?? "0");
  let corpus: { corpus_version: string; valid_from: string | null; entry_count: number } | null =
    null;
  try {
    const annex = await deps.annex(env);
    corpus = {
      corpus_version: annex.corpus_version,
      valid_from: annex.valid_from,
      entry_count: annex.entry_count,
    };
  } catch {
    // dataset temporarily unreachable — report health without it
  }
  return json(
    {
      ok: true,
      assistant_available: monthSpend < budget.monthlyUsd && (isTester || daySpend < budget.dailyUsd),
      corpus,
    },
    200,
    cors,
  );
}

// Failure taxonomy shared by the buffered and streaming paths. A model may
// have run before the throw — KEEP the reservation on upstream errors (refund
// only when no model ran) so mid-loop token burn still counts against the cap.
function failureReason(err: unknown): { reason: string; status: number; refund: boolean } {
  if (err instanceof InvalidRequest) {
    const reason = err.message === "conversation_too_long" ? "conversation_too_long" : "bad_request";
    return { reason, status: 400, refund: true };
  }
  console.error("turn failed:", err);
  // the workspace spend cap is the authoritative ceiling (see README) — when
  // Anthropic refuses for exhausted credit, that IS the budget stop, and the
  // page already shows a polite offline banner for this reason. Without the
  // mapping every visitor sees a raw "something went wrong" instead.
  if (err instanceof Error && /credit balance is too low/i.test(err.message)) {
    return { reason: "daily_budget_exhausted", status: 429, refund: false };
  }
  return { reason: "upstream_error", status: 502, refund: false };
}

export async function handleRequest(
  request: Request,
  env: Env,
  deps: Deps,
  ctx?: Ctx,
): Promise<Response> {
  const allowed = env.ALLOWED_ORIGINS.split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  const origin = request.headers.get("origin");
  const cors = corsHeaders(origin, allowed);

  if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: cors });
  if (origin !== null && !allowed.includes(origin)) {
    return json({ type: "error", reason: "forbidden_origin" }, 403, cors);
  }

  const url = new URL(request.url);
  if (request.method === "GET" && url.pathname === "/api/health") {
    return handleHealth(request, env, deps, cors);
  }
  if (request.method !== "POST" || url.pathname !== "/api/chat") {
    return json({ type: "error", reason: "not_found" }, 404, cors);
  }

  let body: { messages?: unknown };
  try {
    body = (await request.json()) as { messages?: unknown };
  } catch {
    return json({ type: "error", reason: "bad_json" }, 400, cors);
  }

  const messages = body.messages;
  const isStart =
    Array.isArray(messages) &&
    messages.filter((m) => (m as { role?: string }).role === "user").length <= 1;

  const testerKey = request.headers.get("x-tester-key");
  const isTester = Boolean(env.TESTER_KEY && testerKey && testerKey === env.TESTER_KEY);
  const t0 = Date.now();
  const gate = await checkBudget(
    env.BUDGET_KV,
    budgetOf(env),
    request.headers.get("cf-connecting-ip") ?? "unknown",
    env.IP_SALT || "dualuse",
    isStart,
    isTester,
  );
  const gateMs = Date.now() - t0;
  if (!gate.ok) return json({ type: "error", reason: gate.reason }, 429, cors);

  // Reserve a conservative worst-case cost BEFORE the model runs, so the spend
  // counter reflects this request while it is still in flight — a concurrent
  // burst sees the reservations rather than a stale zero. (KV is eventually
  // consistent and non-atomic, so this narrows but cannot fully close the race;
  // the AUTHORITATIVE hard ceiling is the dedicated API key's monthly spend
  // limit, enforced server-side by Anthropic. See README.)
  const RESERVE_USD = 0.15;
  const tReserve = Date.now();
  await recordSpend(env.BUDGET_KV, RESERVE_USD);
  const reserveMs = Date.now() - tReserve;

  // runs one turn end-to-end and builds the response envelope — shared by the
  // buffered and streaming paths so they can never drift apart.
  // modelStarted: an annex outage (or any pre-model throw) must REFUND the
  // reservation — keeping it burned the daily budget as phantom spend and an
  // outage could lock everyone out with zero actual model cost.
  let modelStarted = false;
  const runOnce = async (onStage?: (stage: string) => void, timeBudgetMs?: number) => {
    const tAnnex = Date.now();
    const annex = await deps.annex(env);
    const annexMs = Date.now() - tAnnex;
    const client = deps.client(env);
    const tTurn = Date.now();
    modelStarted = true;
    const result: TurnResult = await runTurn(
      client,
      annex,
      messages,
      {
        loop: env.LOOP_MODEL || "claude-haiku-4-5",
        verdict: env.VERDICT_MODEL || "claude-sonnet-5",
      },
      parseInt(env.MAX_TURNS || "10", 10),
      client, // question-gate judge: same key, cheap Haiku calls
      onStage,
      timeBudgetMs,
    );
    const turnMs = Date.now() - tTurn;
    await recordSpend(env.BUDGET_KV, result.usd - RESERVE_USD); // reconcile to actual
    // stage-by-stage latency: always in the logs (wrangler tail), returned in
    // the body ONLY for the tester key — public responses stay trim
    const perf = {
      total_ms: Date.now() - t0,
      gate_ms: gateMs,
      reserve_ms: reserveMs,
      annex_ms: annexMs,
      turn_ms: turnMs,
      type: result.type,
      stages: result.timings,
    };
    console.log("perf", JSON.stringify(perf));
    return {
      type: result.type,
      text: result.text,
      messages: result.transcript,
      ...(result.verdict ? { verdict: { ...result.verdict, disclaimer: FIXED_DISCLAIMER } } : {}),
      ...(result.pathway
        ? { pathway: { ...result.pathway, disclaimer: PATHWAY_DISCLAIMER } }
        : {}),
      ...(result.continueLicensing ? { continue_licensing: true } : {}),
      ...(isTester ? { timings: perf } : {}),
    };
  };

  const wantsStream = (request.headers.get("accept") ?? "").includes("application/x-ndjson");
  if (!wantsStream) {
    try {
      return json(await runOnce(undefined, 45_000), 200, cors);
    } catch (err) {
      const f = failureReason(err);
      if (f.refund || !modelStarted) await recordSpend(env.BUDGET_KV, -RESERVE_USD);
      return json({ type: "error", reason: f.reason }, f.status, cors);
    }
  }

  // STREAMING (NDJSON): bytes flow from the first model call — one progress
  // line per stage — so the edge's ~100s time-to-first-byte cutoff no longer
  // binds and the turn earns a doubled time budget. The final line carries the
  // exact envelope a buffered response would have been, errors included.
  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();
  const enc = new TextEncoder();
  const emit = (obj: unknown) =>
    writer.write(enc.encode(JSON.stringify(obj) + "\n")).catch(() => {});
  const pump = (async () => {
    try {
      const data = await runOnce((stage) => void emit({ type: "progress", stage }), 90_000);
      await emit({ type: "result", data });
    } catch (err) {
      const f = failureReason(err);
      if (f.refund || !modelStarted) await recordSpend(env.BUDGET_KV, -RESERVE_USD);
      await emit({ type: "result", data: { type: "error", reason: f.reason } });
    } finally {
      await writer.close().catch(() => {});
    }
  })();
  ctx?.waitUntil(pump);
  return new Response(readable, {
    status: 200,
    headers: {
      "content-type": "application/x-ndjson",
      "x-content-type-options": "nosniff",
      ...cors,
    },
  });
}

export default {
  async fetch(request: Request, env: Env, ctx: Ctx): Promise<Response> {
    return handleRequest(request, env, REAL_DEPS, ctx);
  },
};
