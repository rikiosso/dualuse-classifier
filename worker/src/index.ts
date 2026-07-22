// Cloudflare Worker entrypoint: CORS (strict origin allowlist — never *),
// budget gates BEFORE any model call, then one classification turn.

import { loadAnnex } from "./annexData";
import { AnthropicClient } from "./claudeClient";
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

const FIXED_DISCLAIMER =
  "Indicative automated triage only — not legal advice. Catch-all controls " +
  "(Art. 4 and 5, Regulation (EU) 2021/821) may apply regardless of listing; a " +
  "licensing authority or qualified counsel has the final word.";

function corsHeaders(origin: string | null, allowed: string[]): Record<string, string> {
  const ok = origin !== null && allowed.includes(origin);
  return {
    "access-control-allow-origin": ok ? origin : allowed[0] ?? "",
    "access-control-allow-methods": "POST, OPTIONS",
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

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const allowed = env.ALLOWED_ORIGINS.split(",").map((s) => s.trim()).filter(Boolean);
    const origin = request.headers.get("origin");
    const cors = corsHeaders(origin, allowed);

    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: cors });
    const url = new URL(request.url);
    if (request.method !== "POST" || url.pathname !== "/api/chat") {
      return json({ type: "error", reason: "not_found" }, 404, cors);
    }
    if (origin !== null && !allowed.includes(origin)) {
      return json({ type: "error", reason: "forbidden_origin" }, 403, cors);
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
      {
        dailyUsd: parseFloat(env.DAILY_BUDGET_USD || "0.30"),
        monthlyUsd: parseFloat(env.MONTHLY_BUDGET_USD || "10"),
        ipDailyConversations: parseInt(env.IP_DAILY_CONVERSATIONS || "2", 10),
      },
      request.headers.get("cf-connecting-ip") ?? "unknown",
      env.IP_SALT || "dualuse",
      isStart,
    );
    if (!gate.ok) return json({ type: "error", reason: gate.reason }, 429, cors);

    try {
      const annex = await loadAnnex(env.ANNEX_URL);
      const client = new AnthropicClient(env.ANTHROPIC_API_KEY);
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
      await recordSpend(env.BUDGET_KV, result.usd);
      return json(
        {
          type: result.type,
          text: result.text,
          messages: result.transcript,
          ...(result.verdict
            ? { verdict: { ...result.verdict, disclaimer: FIXED_DISCLAIMER } }
            : {}),
        },
        200,
        cors,
      );
    } catch (err) {
      if (err instanceof InvalidRequest) {
        const reason = err.message === "conversation_too_long" ? "conversation_too_long" : "bad_request";
        return json({ type: "error", reason }, 400, cors);
      }
      console.error("turn failed:", err);
      return json({ type: "error", reason: "upstream_error" }, 502, cors);
    }
  },
};
