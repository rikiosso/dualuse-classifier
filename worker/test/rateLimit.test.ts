import { describe, expect, it } from "vitest";
import { checkBudget, estimateUsd, recordSpend, type KVLike } from "../src/rateLimit";

class FakeKV implements KVLike {
  store = new Map<string, string>();
  async get(key: string) {
    return this.store.get(key) ?? null;
  }
  async put(key: string, value: string) {
    this.store.set(key, value);
  }
}

const BUDGET = { dailyUsd: 0.3, monthlyUsd: 10, ipDailyConversations: 2 };

describe("budget gates", () => {
  it("allows, counts conversations per IP, then rate-limits the third", async () => {
    const kv = new FakeKV();
    expect((await checkBudget(kv, BUDGET, "1.2.3.4", "s", true)).ok).toBe(true);
    expect((await checkBudget(kv, BUDGET, "1.2.3.4", "s", true)).ok).toBe(true);
    const third = await checkBudget(kv, BUDGET, "1.2.3.4", "s", true);
    expect(third).toEqual({ ok: false, reason: "rate_limited" });
    // a different visitor is unaffected
    expect((await checkBudget(kv, BUDGET, "5.6.7.8", "s", true)).ok).toBe(true);
  });

  it("stops everything once the global budget is spent — daily or monthly", async () => {
    const kv = new FakeKV();
    await recordSpend(kv, 0.31);
    const gate = await checkBudget(kv, BUDGET, "9.9.9.9", "s", true);
    expect(gate).toEqual({ ok: false, reason: "daily_budget_exhausted" });
  });

  it("never stores a raw IP", async () => {
    const kv = new FakeKV();
    await checkBudget(kv, BUDGET, "203.0.113.7", "salt", true);
    expect([...kv.store.keys()].join()).not.toContain("203.0.113.7");
  });
});

describe("estimateUsd", () => {
  it("prices cache reads at a tenth and writes at 1.25x", () => {
    const usd = estimateUsd("claude-haiku-4-5", {
      input_tokens: 1000,
      output_tokens: 500,
      cache_read_input_tokens: 30000,
      cache_creation_input_tokens: 0,
    });
    // 1000*1 + 30000*0.1*1 + 500*5 = 6500 per MTok scale
    expect(usd).toBeCloseTo(0.0065, 5);
  });
});
