// One human turn = one call here. Internally the model may take several tool
// steps (lookups need no human input, so they loop server-side, bounded).
// Convergence is two-stage: the cheap loop model decides WHEN to conclude by
// calling final_answer; the verdict model then writes the authoritative verdict
// under a forced, strict schema — and the Worker validates it against the
// corpus before anyone sees it (a bare or uncited verdict is a bug, enforced
// by code, not prompt).

import type { AnnexDataset } from "./annexData";
import { definitionsFor, entryByCode, quoteAppearsIn } from "./annexData";
import type { ClaudeClient, ClaudeResponse } from "./claudeClient";
import { buildSystemBlocks, promptSha256 } from "./prompt";
import { FINAL_ANSWER_TOOL, LOOKUP_DEFINITIONS_TOOL, LOOKUP_ENTRIES_TOOL, Verdict } from "./tools";
import { estimateUsd } from "./rateLimit";

const MAX_TOOL_ITERATIONS = 4;
const LOOP_MAX_TOKENS = 2000;
const VERDICT_MAX_TOKENS = 8000;

export interface Models {
  loop: string;
  verdict: string;
}

type Block = { type: string; [k: string]: unknown };
type Msg = { role: "user" | "assistant"; content: Block[] | string };

export interface TurnResult {
  type: "question" | "verdict";
  text: string;
  transcript: Msg[];
  verdict?: Verdict & { corpus_version: string; corpus_sha256: string; prompt_sha256: string };
  usd: number;
}

export class InvalidRequest extends Error {}

// Strip anything the client should not be able to smuggle in: cache_control,
// unknown roles, unknown block types, oversized histories.
export function sanitizeMessages(raw: unknown, maxUserTurns: number): Msg[] {
  if (!Array.isArray(raw) || raw.length === 0) throw new InvalidRequest("messages required");
  if (JSON.stringify(raw).length > 200_000) throw new InvalidRequest("conversation too large");
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
      blocks = content.map((b: Record<string, unknown>) => {
        if (typeof b?.type !== "string" || !allowedBlocks.has(b.type)) {
          throw new InvalidRequest("bad content block");
        }
        const { cache_control: _dropped, ...rest } = b;
        return rest as Block;
      });
    } else {
      throw new InvalidRequest("bad content");
    }
    if (m.role === "user" && blocks.some((b) => b.type === "text")) userTurns += 1;
    out.push({ role: m.role, content: blocks });
  }
  if (out[0].role !== "user") throw new InvalidRequest("first message must be user");
  if (userTurns > maxUserTurns) throw new InvalidRequest("conversation_too_long");
  return out;
}

function withCache(blocks: Block[] | string): Block[] {
  const arr = typeof blocks === "string" ? [{ type: "text", text: blocks } as Block] : [...blocks];
  if (arr.length > 0) arr[arr.length - 1] = { ...arr[arr.length - 1], cache_control: { type: "ephemeral" } };
  return arr;
}

function textOf(resp: ClaudeResponse): string {
  return resp.content
    .filter((b) => b.type === "text")
    .map((b) => String((b as { text?: string }).text ?? ""))
    .join("\n")
    .trim();
}

function toolUses(resp: ClaudeResponse): Block[] {
  return resp.content.filter((b) => b.type === "tool_use");
}

function execLookup(annex: AnnexDataset, name: string, input: Record<string, unknown>): string {
  if (name === "lookup_entries") {
    const codes = (Array.isArray(input.codes) ? input.codes : []).slice(0, 6).map(String);
    if (codes.length === 0) return "No codes given.";
    return codes
      .map((c) => {
        const e = entryByCode(annex, c);
        return e
          ? `=== ${e.entry_code} (category ${e.category}) ===\n${e.verbatim_text}`
          : `No entry ${c} in this corpus version.`;
      })
      .join("\n\n");
  }
  if (name === "lookup_definitions") {
    return definitionsFor(annex, (Array.isArray(input.terms) ? input.terms : []).map(String));
  }
  return `Unknown tool ${name}.`;
}

// Server-side verdict validation — the NakedVerdict discipline. Returns a list
// of problems; empty list = acceptable.
export function validateVerdict(v: Verdict, annex: AnnexDataset): string[] {
  const problems: string[] = [];
  if (v.caveats.length === 0) problems.push("caveats must not be empty");
  if (v.status === "listed") {
    if (v.entry_codes.length === 0) problems.push("listed verdict needs entry_codes");
    if (v.reasoning.length === 0) problems.push("listed verdict needs reasoning");
  }
  for (const code of v.entry_codes) {
    if (!entryByCode(annex, code)) problems.push(`entry_code ${code} does not exist in the corpus`);
  }
  for (const r of v.reasoning) {
    const entry = entryByCode(annex, r.entry_code);
    if (!entry) {
      problems.push(`reasoning cites nonexistent entry ${r.entry_code}`);
    } else if (!quoteAppearsIn(r.verbatim_quote, entry.verbatim_text)) {
      problems.push(
        `verbatim_quote for ${r.entry_code} not found in the entry text — quotes must be copied exactly from lookup_entries output`,
      );
    }
  }
  return problems;
}

export async function runTurn(
  client: ClaudeClient,
  annex: AnnexDataset,
  incoming: unknown,
  models: Models,
  maxUserTurns: number,
): Promise<TurnResult> {
  const transcript = sanitizeMessages(incoming, maxUserTurns);
  const systemBlocks = buildSystemBlocks(annex);
  const system = withCache(systemBlocks as Block[]);
  const tools = [LOOKUP_ENTRIES_TOOL, LOOKUP_DEFINITIONS_TOOL, FINAL_ANSWER_TOOL];
  let usd = 0;

  const call = async (model: string, maxTokens: number, forced: boolean) => {
    const msgs = transcript.map((m, i) =>
      i === transcript.length - 1 ? { ...m, content: withCache(m.content) } : m,
    );
    const resp = await client.complete({
      model,
      max_tokens: maxTokens,
      system,
      messages: msgs,
      tools,
      ...(forced ? { tool_choice: { type: "tool", name: "final_answer" } } : {}),
    });
    usd += estimateUsd(model, resp.usage);
    return resp;
  };

  for (let i = 0; i < MAX_TOOL_ITERATIONS; i++) {
    const resp = await call(models.loop, LOOP_MAX_TOKENS, false);
    const uses = toolUses(resp);
    const finalCall = uses.find((u) => u.name === "final_answer");

    if (finalCall) {
      // The loop model decided to conclude. Hand the verdict itself to the
      // stronger model under a forced strict schema, with one retry on
      // validation failure.
      transcript.push({ role: "assistant", content: resp.content as Block[] });
      transcript.push({
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: finalCall.id,
            content:
              "Draft framework received. Now produce the authoritative final verdict by " +
              "calling final_answer with complete reasoning, exact verbatim quotes and " +
              "full caveats.",
          },
        ],
      });
      for (let attempt = 0; attempt < 2; attempt++) {
        const vResp = await call(models.verdict, VERDICT_MAX_TOKENS, true);
        const vUse = toolUses(vResp).find((u) => u.name === "final_answer");
        if (!vUse) break;
        const verdict = vUse.input as Verdict;
        const problems = validateVerdict(verdict, annex);
        transcript.push({ role: "assistant", content: vResp.content as Block[] });
        if (problems.length === 0) {
          return {
            type: "verdict",
            text: textOf(vResp),
            transcript,
            verdict: {
              ...verdict,
              corpus_version: annex.corpus_version,
              corpus_sha256: annex.sha256,
              prompt_sha256: await promptSha256(),
            },
            usd,
          };
        }
        transcript.push({
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: vUse.id,
              is_error: true,
              content: `Verdict rejected by corpus validation: ${problems.join("; ")}. Correct and call final_answer again.`,
            },
          ],
        });
      }
      // Fail-closed: no unverifiable verdict ever ships. Ask for more facts.
      transcript.push({
        role: "user",
        content: [
          {
            type: "text",
            text:
              "[system] The verdict could not be validated against the corpus. Ask the " +
              "user for the missing technical facts instead of concluding.",
          },
        ],
      });
      const retry = await call(models.loop, LOOP_MAX_TOKENS, false);
      transcript.push({ role: "assistant", content: retry.content as Block[] });
      return { type: "question", text: textOf(retry), transcript, usd };
    }

    if (uses.length > 0) {
      transcript.push({ role: "assistant", content: resp.content as Block[] });
      transcript.push({
        role: "user",
        content: uses.map((u) => ({
          type: "tool_result",
          tool_use_id: u.id,
          content: execLookup(annex, String(u.name), (u.input ?? {}) as Record<string, unknown>),
        })),
      });
      continue;
    }

    transcript.push({ role: "assistant", content: resp.content as Block[] });
    return { type: "question", text: textOf(resp), transcript, usd };
  }

  // Tool budget exhausted without an answer — surface the last state honestly.
  transcript.push({
    role: "assistant",
    content: [{ type: "text", text: "I need to narrow this down — let me ask you directly." }],
  });
  return {
    type: "question",
    text: "I need to narrow this down — could you give me the key technical parameters (e.g. performance figures, wavelengths, materials)?",
    transcript,
    usd,
  };
}
