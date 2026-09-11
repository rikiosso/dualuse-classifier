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

  it("health honours the tester key: daily pacing ignored, monthly stop kept", async () => {
    const kv = new FakeKV();
    kv.store.set(`spend:${new Date().toISOString().slice(0, 10)}`, "0.31"); // day exhausted
    const testerEnv = { ...env(kv), TESTER_KEY: "tk" };
    const withTester = await handleRequest(
      new Request("https://worker.test/api/health", {
        headers: { origin: ORIGIN, "x-tester-key": "tk" },
      }),
      testerEnv,
      deps([]),
    );
    expect(((await withTester.json()) as { assistant_available: boolean }).assistant_available).toBe(true);
    kv.store.set(`spend:${new Date().toISOString().slice(0, 7)}`, "99"); // month exhausted
    const monthGated = await handleRequest(
      new Request("https://worker.test/api/health", {
        headers: { origin: ORIGIN, "x-tester-key": "tk" },
      }),
      testerEnv,
      deps([]),
    );
    expect(((await monthGated.json()) as { assistant_available: boolean }).assistant_available).toBe(false);
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

describe("exhausted API credit", () => {
  it("maps Anthropic's credit-balance refusal to the budget-exhausted banner, not a raw 502", async () => {
    const broke: Deps = {
      annex: async () => ANNEX,
      client: () => ({
        complete: async () => {
          throw new Error(
            'anthropic 400: {"type":"error","error":{"type":"invalid_request_error",' +
              '"message":"Your credit balance is too low to access the Anthropic API."}}',
          );
        },
      }),
    };
    const resp = await handleRequest(chatRequest(), env(), broke);
    expect(resp.status).toBe(429);
    expect(await resp.json()).toEqual({ type: "error", reason: "daily_budget_exhausted" });
  });
});

describe("streaming (NDJSON)", () => {
  it("streams progress lines and ends with the same envelope a buffered response carries", async () => {
    const req = new Request("https://worker.test/api/chat", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        accept: "application/x-ndjson",
        "cf-connecting-ip": "1.2.3.4",
        origin: ORIGIN,
      },
      body: JSON.stringify({ messages: [{ role: "user", content: "an RF amplifier" }] }),
    });
    const resp = await handleRequest(req, env(), deps([question]));
    expect(resp.status).toBe(200);
    expect(resp.headers.get("content-type")).toBe("application/x-ndjson");
    const lines = (await resp.text()).trim().split("\n").map((l) => JSON.parse(l));
    expect(lines[0]).toEqual({ type: "progress", stage: "interview" });
    const last = lines.at(-1);
    expect(last.type).toBe("result");
    expect(last.data.type).toBe("question");
    expect(last.data.text).toContain("frequency");
  });

  it("a mid-turn failure still ends the stream with an error envelope, never a broken body, and keeps the $0.15 reservation because modelStarted was true", async () => {
    const kv = new FakeKV();
    const broke: Deps = {
      annex: async () => ANNEX,
      client: () => ({
        complete: async () => {
          throw new Error("anthropic 500: upstream exploded");
        },
      }),
    };
    const req = new Request("https://worker.test/api/chat", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        accept: "application/x-ndjson",
        "cf-connecting-ip": "1.2.3.4",
        origin: ORIGIN,
      },
      body: JSON.stringify({ messages: [{ role: "user", content: "an RF amplifier" }] }),
    });
    const resp = await handleRequest(req, env(kv), broke);
    expect(resp.status).toBe(200); // status already committed — the error rides the body
    // (c) the body terminates: text() resolves rather than hanging on a
    // writer that never closes.
    const bodyText = await resp.text();
    const lines = bodyText.trim().split("\n").map((l) => JSON.parse(l));
    // (a) the canned client throws on its FIRST complete() call, which the
    // worker only reaches after onStage("interview"): loop.ts calls
    // onStage synchronously, before the `await client.complete(...)` that
    // throws, so the progress line is always queued on the stream ahead of
    // the failure. Proven here rather than assumed: if the worker ever
    // reordered this, this assertion (not lines.at(-1) alone) would catch it.
    expect(lines[0]).toEqual({ type: "progress", stage: "interview" });
    // (b) the stream still ends with the same error envelope a buffered
    // response would carry.
    expect(lines.at(-1)).toEqual({ type: "result", data: { type: "error", reason: "upstream_error" } });
    // (d) modelStarted flips true in index.ts's runOnce BEFORE runTurn (and
    // so before this throw), so failureReason's refund=false for
    // upstream_error must stand: the day spend key still carries the
    // reservation, not refunded as phantom-outage spend.
    const day = new Date().toISOString().slice(0, 10);
    expect(kv.store.get(`spend:${day}`)).toBe("0.150000");
  });

  it("refunds the $0.15 reservation when the annex loader fails before any model call, so the day key returns to its prior value (modelStarted stays false)", async () => {
    const kv = new FakeKV();
    const day = new Date().toISOString().slice(0, 10);
    kv.store.set(`spend:${day}`, "0.050000"); // prior spend from an earlier request today
    const annexDown: Deps = {
      annex: async () => {
        throw new Error("annex fetch failed: 503");
      },
      // never reached: the annex load throws before deps.client(env) is called
      client: () => new CannedClaudeClient([question]),
    };
    const req = new Request("https://worker.test/api/chat", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        accept: "application/x-ndjson",
        "cf-connecting-ip": "1.2.3.4",
        origin: ORIGIN,
      },
      body: JSON.stringify({ messages: [{ role: "user", content: "an RF amplifier" }] }),
    });
    const resp = await handleRequest(req, env(kv), annexDown);
    const lines = (await resp.text()).trim().split("\n").map((l) => JSON.parse(l));
    // no progress line at all: onStage is only reachable from inside runTurn,
    // which is never invoked when the annex load itself throws.
    expect(lines).toHaveLength(1);
    expect(lines[0]).toEqual({ type: "result", data: { type: "error", reason: "upstream_error" } });
    // reserved (+0.15) then refunded (-0.15) leaves the day key exactly where
    // it was before this request, not burned for zero model cost.
    expect(kv.store.get(`spend:${day}`)).toBe("0.050000");
  });
});
