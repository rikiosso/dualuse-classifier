// Routing, CORS and budget behaviour of the HTTP handler — injected deps, no network.
import { describe, expect, it } from "vitest";
import { handleRequest, type Deps, type Env } from "../src/index";
import { CannedClaudeClient, type ClaudeResponse } from "../src/claudeClient";
import type { AnnexDataset } from "../src/annexData";
import type { KVLike } from "../src/rateLimit";

class FakeKV implements KVLike {
  store = new Map<string, string>();
  async get(key: string) {
    return this.store.get(key) ?? null;
  }
  async put(key: string, value: string) {
    this.store.set(key, value);
  }
}

const ANNEX: AnnexDataset = {
  corpus_version: "02021R0821-20251115",
  celex: "02021R0821",
  valid_from: "2025-11-15",
  sha256: "deadbeef",
  attribution: "© EU",
  entry_count: 1,
  index: [{ code: "3A001", first_line: "3A001 Electronic items as follows:" }],
  entries: [
    {
      entry_code: "3A001",
      category: "3",
      verbatim_text: "3A001 Electronic items as follows:",
      parameters: [],
      applicable_notes: [],
    },
  ],
  docs: [],
};

const ORIGIN = "https://rikiosso.github.io";

function env(kv = new FakeKV()): Env {
  return {
    ANTHROPIC_API_KEY: "test",
    BUDGET_KV: kv,
    ALLOWED_ORIGINS: ORIGIN,
    ANNEX_URL: "https://example.test/annex.json",
    DAILY_BUDGET_USD: "0.30",
    MONTHLY_BUDGET_USD: "10",
    IP_DAILY_CONVERSATIONS: "2",
    MAX_TURNS: "10",
    IP_SALT: "s",
    LOOP_MODEL: "claude-haiku-4-5",
    VERDICT_MODEL: "claude-sonnet-5",
  };
}

function deps(responses: ClaudeResponse[]): Deps {
  return {
    annex: async () => ANNEX,
    client: () => new CannedClaudeClient(responses),
  };
}

const question: ClaudeResponse = {
  content: [{ type: "text", text: "What is the frequency range?" }],
  stop_reason: "end_turn",
  usage: { input_tokens: 100, output_tokens: 20 },
};

function chatRequest(origin: string | null = ORIGIN, ip = "1.2.3.4"): Request {
  return new Request("https://worker.test/api/chat", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "cf-connecting-ip": ip,
      ...(origin ? { origin } : {}),
    },
    body: JSON.stringify({ messages: [{ role: "user", content: "an RF amplifier" }] }),
  });
}

describe("handleRequest", () => {
  it("answers a chat turn and reflects the allowed origin", async () => {
    const resp = await handleRequest(chatRequest(), env(), deps([question]));
    expect(resp.status).toBe(200);
    expect(resp.headers.get("access-control-allow-origin")).toBe(ORIGIN);
    const data = (await resp.json()) as { type: string; text: string };
    expect(data.type).toBe("question");
    expect(data.text).toContain("frequency");
  });

  it("rejects foreign origins before any model call", async () => {
    const resp = await handleRequest(chatRequest("https://evil.example"), env(), deps([]));
    expect(resp.status).toBe(403);
    expect(resp.headers.get("access-control-allow-origin")).toBe(ORIGIN); // never echoes evil
  });

  it("handles OPTIONS preflight", async () => {
    const resp = await handleRequest(
      new Request("https://worker.test/api/chat", { method: "OPTIONS", headers: { origin: ORIGIN } }),
      env(),
      deps([]),
    );
    expect(resp.status).toBe(204);
  });

  it("gates on the global budget with 429 and never touches the model", async () => {
    const kv = new FakeKV();
    kv.store.set(`spend:${new Date().toISOString().slice(0, 10)}`, "0.31");
    const resp = await handleRequest(chatRequest(), env(kv), deps([]));
    expect(resp.status).toBe(429);
    const data = (await resp.json()) as { reason: string };
    expect(data.reason).toBe("daily_budget_exhausted");
  });

  it("health reports corpus and budget state", async () => {
    const kv = new FakeKV();
    kv.store.set(`spend:${new Date().toISOString().slice(0, 10)}`, "0.31");
    const resp = await handleRequest(
      new Request("https://worker.test/api/health", { headers: { origin: ORIGIN } }),
      env(kv),
      deps([]),
    );
    const data = (await resp.json()) as {
      assistant_available: boolean;
      corpus: { corpus_version: string };
    };
    expect(data.assistant_available).toBe(false);
    expect(data.corpus.corpus_version).toBe("02021R0821-20251115");
  });

  it("404s unknown paths and 400s bad JSON", async () => {
    const notFound = await handleRequest(
      new Request("https://worker.test/nope", { method: "POST", headers: { origin: ORIGIN } }),
      env(),
      deps([]),
    );
    expect(notFound.status).toBe(404);
    const bad = await handleRequest(
      new Request("https://worker.test/api/chat", {
        method: "POST",
        headers: { origin: ORIGIN },
        body: "not json",
      }),
      env(),
      deps([]),
    );
    expect(bad.status).toBe(400);
  });
});
