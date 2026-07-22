// Budget and fairness limits — THE mechanism that makes a public LLM demo cost
// a bounded amount. Two independent layers:
//   - per-IP conversation cap (fairness: nobody drains the day for everyone)
//   - global daily + monthly spend caps (the bill guarantee; monthly is the
//     absolute stop even if daily accounting drifts)
// Counters live in Workers KV; we write once per conversation START and once
// per request for spend (well within the free tier's ~1k writes/day at the
// traffic these caps allow). Turn caps cost no KV at all — they are derived
// from the request body.

export interface KVLike {
  get(key: string): Promise<string | null>;
  put(key: string, value: string, opts?: { expirationTtl?: number }): Promise<void>;
}

export interface Budget {
  dailyUsd: number;
  monthlyUsd: number;
  ipDailyConversations: number;
}

const DAY_TTL = 26 * 60 * 60;
const MONTH_TTL = 32 * 24 * 60 * 60;

function today(): string {
  return new Date().toISOString().slice(0, 10);
}

function thisMonth(): string {
  return new Date().toISOString().slice(0, 7);
}

async function hashIp(ip: string, salt: string): Promise<string> {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(salt + ip));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export async function checkBudget(
  kv: KVLike,
  budget: Budget,
  ip: string,
  salt: string,
  isConversationStart: boolean,
): Promise<{ ok: true } | { ok: false; reason: "daily_budget_exhausted" | "rate_limited" }> {
  const daySpend = parseFloat((await kv.get(`spend:${today()}`)) ?? "0");
  const monthSpend = parseFloat((await kv.get(`spend:${thisMonth()}`)) ?? "0");
  if (daySpend >= budget.dailyUsd || monthSpend >= budget.monthlyUsd) {
    return { ok: false, reason: "daily_budget_exhausted" };
  }
  const ipKey = `ip:${await hashIp(ip, salt)}:${today()}`;
  const ipCount = parseInt((await kv.get(ipKey)) ?? "0", 10);
  if (isConversationStart) {
    if (ipCount >= budget.ipDailyConversations) return { ok: false, reason: "rate_limited" };
    await kv.put(ipKey, String(ipCount + 1), { expirationTtl: DAY_TTL });
  } else if (ipCount === 0) {
    // continuing a conversation that never started here (expired day or forged
    // history) still counts as a start — no free rides across midnight
    await kv.put(ipKey, "1", { expirationTtl: DAY_TTL });
  }
  return { ok: true };
}

export async function recordSpend(kv: KVLike, usd: number): Promise<void> {
  const dayKey = `spend:${today()}`;
  const monthKey = `spend:${thisMonth()}`;
  const day = parseFloat((await kv.get(dayKey)) ?? "0") + usd;
  const month = parseFloat((await kv.get(monthKey)) ?? "0") + usd;
  await kv.put(dayKey, day.toFixed(6), { expirationTtl: DAY_TTL });
  await kv.put(monthKey, month.toFixed(6), { expirationTtl: MONTH_TTL });
}

// Sticker prices per MTok (NOT intro prices — conservative), cache reads 0.1x,
// cache writes 1.25x. Estimating high is fine: the caps only trip earlier.
const PRICES: Record<string, { in: number; out: number }> = {
  "claude-haiku-4-5": { in: 1.0, out: 5.0 },
  "claude-sonnet-5": { in: 3.0, out: 15.0 },
};

export function estimateUsd(
  model: string,
  usage: {
    input_tokens: number;
    output_tokens: number;
    cache_creation_input_tokens?: number;
    cache_read_input_tokens?: number;
  },
): number {
  const p = PRICES[model] ?? { in: 3.0, out: 15.0 };
  return (
    (usage.input_tokens * p.in +
      (usage.cache_creation_input_tokens ?? 0) * p.in * 1.25 +
      (usage.cache_read_input_tokens ?? 0) * p.in * 0.1 +
      usage.output_tokens * p.out) /
    1_000_000
  );
}
