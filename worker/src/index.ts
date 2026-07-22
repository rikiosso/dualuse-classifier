// Cloudflare Worker entrypoint: CORS (strict origin allowlist — never *),
// budget gates BEFORE any model call, then one classification turn.
// handleRequest takes injectable deps so tests exercise routing, CORS and
// budget behaviour without any network.

import { loadAnnex, type AnnexDataset } from "./annexData";
import { AnthropicClient, type ClaudeClient } from "./claudeClient";
import { InvalidRequest, runTurn } from "./loop";
import { checkBudget, recordSpend, type KVLike } from "./rateLimit";

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

function corsHeaders(origin: string | null, allowed: string[]): Record<string, string> {
  const ok = origin !== null && allowed.includes(origin);
  return {
    "access-control-allow-origin": ok ? origin : (allowed[0] ?? ""),
    "access-control-allow-methods": "GET, POST, OPTIONS",
    "access-control-allow-headers": "content-type",
    vary: "origin",
  };
}

function json(data: unknown, status: number, headers: Record<string, string>): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}

function budgetOf(env: Env) {
  return {
    dailyUsd: parseFloat(env.DAILY_BUDGET_USD || "0.30"),
    monthlyUsd: parseFloat(env.MONTHLY_BUDGET_USD || "10"),
    ipDailyConversations: parseInt(env.IP_DAILY_CONVERSATIONS || "2", 10),
  };
}

// GET /api/health — lets the page show corpus status and an exhausted budget
// BEFORE a visitor types a whole description. Reads KV only; never the model.
async function handleHealth(env: Env, deps: Deps, cors: Record<string, string>): Promise<Response> {
  const day = new Date().toISOString().slice(0, 10);
  const month = day.slice(0, 7);
  const budget = budgetOf(env);
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
      assistant_available: daySpend < budget.dailyUsd && monthSpend < budget.monthlyUsd,
      corpus,
    },
    200,
    cors,
  );
}

export async function handleRequest(request: Request, env: Env, deps: Deps): Promise<Response> {
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
    return handleHealth(env, deps, cors);
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

  const gate = await checkBudget(
    env.BUDGET_KV,
    budgetOf(env),
    request.headers.get("cf-connecting-ip") ?? "unknown",
    env.IP_SALT || "dualuse",
    isStart,
  );
  if (!gate.ok) return json({ type: "error", reason: gate.reason }, 429, cors);

  // Reserve a conservative worst-case cost BEFORE the model runs, so the spend
  // counter reflects this request while it is still in flight — a concurrent
  // burst sees the reservations rather than a stale zero. (KV is eventually
  // consistent and non-atomic, so this narrows but cannot fully close the race;
  // the AUTHORITATIVE hard ceiling is the dedicated API key's monthly spend
  // limit, enforced server-side by Anthropic. See README.)
  const RESERVE_USD = 0.15;
  await recordSpend(env.BUDGET_KV, RESERVE_USD);

  try {
    const annex = await deps.annex(env);
    const client = deps.client(env);
    const result = await runTurn(
      client,
      annex,
      messages,
      {
        loop: env.LOOP_MODEL || "claude-haiku-4-5",
        verdict: env.VERDICT_MODEL || "claude-sonnet-5",
      },
      parseInt(env.MAX_TURNS || "10", 10),
    );
    await recordSpend(env.BUDGET_KV, result.usd - RESERVE_USD); // reconcile to actual
    return json(
      {
        type: result.type,
        text: result.text,
        messages: result.transcript,
        ...(result.verdict ? { verdict: { ...result.verdict, disclaimer: FIXED_DISCLAIMER } } : {}),
      },
      200,
      cors,
    );
  } catch (err) {
    if (err instanceof InvalidRequest) {
      // no model ran — refund the reservation
      await recordSpend(env.BUDGET_KV, -RESERVE_USD);
      const reason =
        err.message === "conversation_too_long" ? "conversation_too_long" : "bad_request";
      return json({ type: "error", reason }, 400, cors);
    }
    // a model may have run before the throw — KEEP the reservation (never refund
    // on an upstream error) so mid-loop token burn still counts against the cap
    console.error("turn failed:", err);
    return json({ type: "error", reason: "upstream_error" }, 502, cors);
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    return handleRequest(request, env, REAL_DEPS);
  },
};
