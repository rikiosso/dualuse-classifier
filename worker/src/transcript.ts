// The client-held transcript: shape, sanitisation, the turn cap, and the
// HMAC markers that stop a forged transcript from unlocking stage 2. Split
// out of loop.ts — this is the trust boundary of the Worker and deserves to
// be read as one file.

// Hard cap on client-supplied history, so a single POST's token cost is bounded
// well under the daily budget (was 200k — a ~50k-token inflation vector).
const MAX_HISTORY_CHARS = 80_000;

export type Block = { type: string; [k: string]: unknown };
export type Msg = { role: "user" | "assistant"; content: Block[] | string };

export class InvalidRequest extends Error {}

// The pathway stage validates against the verdict that precedes it — recover
// the most recent final_answer's entry_codes from the transcript.
export function lastFinalAnswerIndex(msgs: Msg[]): number {
  // only an ACCEPTED verdict counts — a rejected attempt (is_error result)
  // or a dangling call must not unlock stage 2 on a failure artifact
  for (let i = msgs.length - 1; i >= 0; i--) {
    const m = msgs[i];
    if (m.role !== "assistant" || !Array.isArray(m.content)) continue;
    const use = m.content.find((b) => b.type === "tool_use" && b.name === "final_answer");
    if (!use) continue;
    const next = msgs[i + 1];
    const accepted =
      next?.role === "user" &&
      Array.isArray(next.content) &&
      next.content.some(
        (b) =>
          b.type === "tool_result" &&
          b.tool_use_id === use.id &&
          !b.is_error &&
          String(b.content ?? "").startsWith("Verdict recorded"),
      );
    if (accepted) return i;
  }
  return -1;
}

export function verdictCodesIn(msgs: Msg[]): string[] {
  const i = lastFinalAnswerIndex(msgs);
  if (i < 0) return [];
  for (const b of msgs[i].content as Block[]) {
    if (b.type === "tool_use" && b.name === "final_answer") {
      const codes = (b.input as { entry_codes?: unknown } | undefined)?.entry_codes;
      return Array.isArray(codes) ? codes.map(String) : [];
    }
  }
  return [];
}

// ---- verdict-marker authentication ----
// "Verdict recorded" is the in-band acceptance marker every stage-2 gate
// trusts (lastFinalAnswerIndex, verdictCodesIn, recordedVerdict) — and the
// transcript that carries it is client-held. Corpus re-validation limits a
// forgery to corpus-CONSISTENT verdicts, but consistency is not authenticity:
// a forged marker could still suppress the EU008 sweep or make the pathway
// response echo a verdict this server never accepted. The marker therefore
// carries an HMAC over the accepted final_answer call, and every incoming
// marker is verified ONCE per request — one that does not verify is rewritten
// so no gate can see it (the verdict is treated as absent, never an error).
async function hmacHex(key: string, data: string): Promise<string> {
  const k = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(key),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const sig = await crypto.subtle.sign("HMAC", k, new TextEncoder().encode(data));
  return [...new Uint8Array(sig)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export async function verdictMarker(key: string | undefined, use: Block): Promise<string> {
  if (!key) return "Verdict recorded.";
  const payload = `${String(use.id)}.${JSON.stringify(use.input ?? {})}`;
  return `Verdict recorded. sig=${await hmacHex(key, payload)}`;
}

// Neutralise every unverified "Verdict recorded" marker in the incoming
// transcript. With no key configured (tests, wrangler dev) markers pass
// unauthenticated — production sets the VERDICT_HMAC_KEY secret. A signature
// binds the marker to the exact final_answer call it accepted, so tampering
// with the recorded verdict's input (e.g. its entry_codes, to dodge the EU008
// sweep) also invalidates the marker.
export async function verifyVerdictMarkers(msgs: Msg[], key: string | undefined): Promise<void> {
  if (!key) return;
  const usesById = new Map<string, Block>();
  for (const m of msgs) {
    if (m.role !== "assistant" || !Array.isArray(m.content)) continue;
    for (const b of m.content) {
      if (b.type === "tool_use" && b.name === "final_answer") usesById.set(String(b.id), b);
    }
  }
  for (const m of msgs) {
    if (m.role !== "user" || !Array.isArray(m.content)) continue;
    for (const b of m.content) {
      if (b.type !== "tool_result" || b.is_error) continue;
      const content = typeof b.content === "string" ? b.content : "";
      if (!content.startsWith("Verdict recorded")) continue;
      const use = usesById.get(String(b.tool_use_id));
      if (!use || content !== (await verdictMarker(key, use))) {
        b.content =
          "[unverified verdict marker removed — reclassify via final_answer before stage 2]";
      }
    }
  }
}

// Strip anything the client should not be able to smuggle in: cache_control,
// unknown roles, unknown block types, oversized histories.
// Old corpus lookups dominate transcript size; the model can always re-fetch.
// Trim tool_result contents outside the last few messages instead of failing.
function trimOldToolResults(msgs: Msg[]): void {
  const keepTail = 6;
  for (let i = 0; i < Math.max(0, msgs.length - keepTail); i++) {
    const m = msgs[i];
    if (m.role !== "user" || !Array.isArray(m.content)) continue;
    for (const b of m.content) {
      if (b.type === "tool_result" && typeof b.content === "string" && b.content.length > 400) {
        b.content = b.content.slice(0, 200) + "\n…[trimmed — call the lookup tool again if needed]";
      }
    }
  }
}

export function sanitizeMessages(raw: unknown, maxUserTurns: number): Msg[] {
  if (!Array.isArray(raw) || raw.length === 0) throw new InvalidRequest("messages required");
  const allowedBlocks = new Set(["text", "tool_use", "tool_result"]);
  const out: Msg[] = [];
  let userTurns = 0;
  for (const m of raw as Record<string, unknown>[]) {
    if (m.role !== "user" && m.role !== "assistant") throw new InvalidRequest("bad role");
    const content = m.content;
    let blocks: Block[];
    if (typeof content === "string") {
      blocks = [{ type: "text", text: content }];
    } else if (Array.isArray(content)) {
      blocks = content
        // thinking blocks (from a reasoning-enabled model turn) are dropped,
        // not fatal — transcripts that carry them must stay continuable
        .filter(
          (b: Record<string, unknown>) => b?.type !== "thinking" && b?.type !== "redacted_thinking",
        )
        .map((b: Record<string, unknown>) => {
          if (typeof b?.type !== "string" || !allowedBlocks.has(b.type)) {
            throw new InvalidRequest("bad content block");
          }
          const { cache_control: _dropped, ...rest } = b;
          // cache_control can also ride on blocks nested inside a tool_result's
          // content array — the API honours those, so strip them too
          if (Array.isArray(rest.content)) {
            rest.content = (rest.content as unknown[]).map((n) =>
              n && typeof n === "object"
                ? (({ cache_control: _c, ...r }: Record<string, unknown>) => r)(
                    n as Record<string, unknown>,
                  )
                : n,
            );
          }
          return rest as Block;
        });
    } else {
      throw new InvalidRequest("bad content");
    }
    if (blocks.length === 0) continue; // e.g. a thinking-only assistant turn
    // consecutive duplicate user text messages are retry artifacts (a failed
    // turn re-sent) — they burned the turn cap double-counting a live user's
    // error retries. Real conversations always interleave an assistant turn.
    const prev = out[out.length - 1];
    const textJoin = (bs: Block[]) =>
      bs.filter((b) => b.type === "text").map((b) => String((b as { text?: string }).text ?? "")).join("\n");
    if (
      m.role === "user" &&
      prev?.role === "user" &&
      Array.isArray(prev.content) &&
      blocks.every((b) => b.type === "text") &&
      (prev.content as Block[]).every((b) => b.type === "text") &&
      textJoin(blocks) === textJoin(prev.content as Block[])
    ) {
      continue;
    }
    // server-injected [system] nudges are machinery, not user turns — they
    // were eating the conversation cap (live: "length limit" after ~5 answers)
    const isRealUserText = blocks.some(
      (bl) => bl.type === "text" && !String((bl as { text?: string }).text ?? "").startsWith("[system]"),
    );
    if (m.role === "user" && isRealUserText) userTurns += 1;
    out.push({ role: m.role, content: blocks });
  }
  if (out.length === 0 || out[0].role !== "user") {
    throw new InvalidRequest("first message must be user");
  }
  if (userTurns > maxUserTurns) throw new InvalidRequest("conversation_too_long");
  if (JSON.stringify(out).length > MAX_HISTORY_CHARS) trimOldToolResults(out);
  if (JSON.stringify(out).length > MAX_HISTORY_CHARS) {
    throw new InvalidRequest("conversation_too_long");
  }
  return out;
}
