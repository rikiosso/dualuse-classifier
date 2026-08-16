// One human turn = one call here. Internally the model may take several tool
// steps (lookups need no human input, so they loop server-side, bounded).
// Convergence is two-stage: the cheap loop model decides WHEN to conclude by
// calling final_answer; the verdict model then writes the authoritative verdict
// under a forced, strict schema — and the Worker validates it against the
// corpus before anyone sees it (a bare or uncited verdict is a bug, enforced
// by code, not prompt).

import type { AnnexDataset } from "./annexData";
import { definitionsFor, entryByCode, geaById, geaScopeText, provisionText, quoteAppearsIn } from "./annexData";
import type { ClaudeClient, ClaudeResponse } from "./claudeClient";
import { buildSystemBlocks, promptSha256 } from "./prompt";
import {
  FINAL_ANSWER_TOOL,
  LICENSE_PATHWAY_TOOL,
  LOOKUP_DEFINITIONS_TOOL,
  LOOKUP_ENTRIES_TOOL,
  LOOKUP_GEA_TOOL,
  Pathway,
  Verdict,
} from "./tools";
import { estimateUsd } from "./rateLimit";

const MAX_TOOL_ITERATIONS = 3;
const LOOP_MAX_TOKENS = 1200;
const VERDICT_MAX_TOKENS = 2500;
// Quotes shorter than this are too weak to anchor — a 3-char fragment appears
// everywhere. Real thresholds and provisions comfortably clear it.
const MIN_QUOTE_CHARS = 12;
// Hard cap on client-supplied history, so a single POST's token cost is bounded
// well under the daily budget (was 200k — a ~50k-token inflation vector).
const MAX_HISTORY_CHARS = 24_000;

export interface Models {
  loop: string;
  verdict: string;
}

type Block = { type: string; [k: string]: unknown };
type Msg = { role: "user" | "assistant"; content: Block[] | string };

export interface TurnResult {
  type: "question" | "verdict" | "pathway";
  text: string;
  transcript: Msg[];
  verdict?: Verdict & { corpus_version: string; corpus_sha256: string; prompt_sha256: string };
  pathway?: Pathway & { corpus_version: string; corpus_sha256: string; prompt_sha256: string };
  usd: number;
}

// Destinations whose sanctions regimes this tool must FLAG and never resolve —
// separate regulations with their own complexity; a wrong answer here is the
// most expensive mistake the tool could make. Enforced server-side.
const SANCTIONED_DESTINATIONS =
  /\b(russia|russian|belarus|iran|north korea|dprk|syria|crimea|donetsk|luhansk|myanmar|venezuela)\b/i;

export class InvalidRequest extends Error {}

// Strip anything the client should not be able to smuggle in: cache_control,
// unknown roles, unknown block types, oversized histories.
export function sanitizeMessages(raw: unknown, maxUserTurns: number): Msg[] {
  if (!Array.isArray(raw) || raw.length === 0) throw new InvalidRequest("messages required");
  if (JSON.stringify(raw).length > MAX_HISTORY_CHARS) {
    throw new InvalidRequest("conversation_too_long");
  }
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

function withCache(blocks: Block[] | string, ttl?: "1h"): Block[] {
  const arr = typeof blocks === "string" ? [{ type: "text", text: blocks } as Block] : [...blocks];
  if (arr.length > 0) {
    const cc = ttl ? { type: "ephemeral", ttl } : { type: "ephemeral" };
    arr[arr.length - 1] = { ...arr[arr.length - 1], cache_control: cc };
  }
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
  if (name === "lookup_gea") {
    const ids = (Array.isArray(input.ids) ? input.ids : []).slice(0, 4).map(String);
    if (ids.length === 0) return "No ids given.";
    return ids
      .map((id) => {
        const text = geaScopeText(annex, id);
        return text
          ? `=== ${id.toUpperCase().trim()} ===\n${text}`
          : `No GEA ${id} in this corpus version.`;
      })
      .join("\n\n");
  }
  return `Unknown tool ${name}.`;
}

// Pathway validation — same discipline as verdicts: quotes verbatim-in-scope,
// referenced GEAs must exist, sanctioned destinations MUST carry the sanctions
// outcome (never a green light), a GEA outcome needs quoted conditions.
export function validatePathway(pw: Pathway, annex: AnnexDataset): string[] {
  const problems: string[] = [];
  if (pw.caveats.length === 0) problems.push("caveats must not be empty");
  if (!pw.destination.trim()) problems.push("destination must be stated");
  if (SANCTIONED_DESTINATIONS.test(pw.destination) && pw.outcome !== "sanctions_review_required") {
    problems.push(
      `destination "${pw.destination}" is under an EU sanctions regime — outcome must be sanctions_review_required`,
    );
  }
  if (pw.outcome === "gea_available") {
    if (!pw.eligible_gea || !geaById(annex, pw.eligible_gea)) {
      problems.push(`eligible_gea ${pw.eligible_gea || "(empty)"} does not exist in the corpus`);
    }
    if (pw.conditions_quoted.length === 0) {
      problems.push("gea_available requires quoted conditions");
    }
  }
  for (const c of pw.conditions_quoted) {
    const scope = geaScopeText(annex, c.gea_id);
    if (!scope) {
      problems.push(`conditions cite nonexistent GEA ${c.gea_id}`);
    } else if (c.verbatim_quote.replace(/\s+/g, " ").trim().length < MIN_QUOTE_CHARS) {
      problems.push(`condition quote for ${c.gea_id} is too short to anchor`);
    } else if (!quoteAppearsIn(c.verbatim_quote, scope)) {
      problems.push(
        `condition quote for ${c.gea_id} not found in that authorisation's text — copy exactly from lookup_gea output`,
      );
    }
  }
  return problems;
}

// Server-side verdict validation — the NakedVerdict discipline. Returns a list
// of problems; empty list = acceptable.
export function validateVerdict(v: Verdict, annex: AnnexDataset): string[] {
  const problems: string[] = [];
  if (v.caveats.length === 0) problems.push("caveats must not be empty");

  const reasoningCodes = new Set(v.reasoning.map((r) => (r.entry_code || "").toUpperCase()));
  if (v.status === "listed") {
    if (v.entry_codes.length === 0) problems.push("listed verdict needs entry_codes");
    if (v.reasoning.length === 0) problems.push("listed verdict needs reasoning");
    // every headline code must be backed by a reasoning item, and vice versa —
    // the UI headlines entry_codes, so an unbacked code is an unsupported claim
    for (const code of v.entry_codes) {
      if (!reasoningCodes.has(code.toUpperCase())) {
        problems.push(`entry_code ${code} is headlined but has no reasoning with a verbatim quote`);
      }
    }
  }
  for (const code of v.entry_codes) {
    if (!entryByCode(annex, code)) problems.push(`entry_code ${code} does not exist in the corpus`);
  }
  for (const code of reasoningCodes) {
    if (v.status === "listed" && !v.entry_codes.map((c) => c.toUpperCase()).includes(code)) {
      problems.push(`reasoning cites ${code} which is not in entry_codes`);
    }
  }
  for (const r of v.reasoning) {
    const entry = entryByCode(annex, r.entry_code);
    if (!entry) {
      problems.push(`reasoning cites nonexistent entry ${r.entry_code}`);
      continue;
    }
    // the pinpoint path must belong to the cited entry
    const path = (r.dotted_path || "").trim();
    if (!path.toUpperCase().startsWith(r.entry_code.toUpperCase())) {
      problems.push(`dotted_path ${path} does not belong to entry ${r.entry_code}`);
      continue;
    }
    // the quote must appear in the SPECIFIC provision named by dotted_path — not
    // merely somewhere in the multi-page entry (blocks comparator/number flips
    // laundered from a sibling clause)
    const scope = provisionText(entry, path) ?? entry.verbatim_text;
    if (r.verbatim_quote.replace(/\s+/g, " ").trim().length < MIN_QUOTE_CHARS) {
      problems.push(`verbatim_quote for ${path} is too short to anchor a citation`);
    } else if (!quoteAppearsIn(r.verbatim_quote, scope)) {
      problems.push(
        `verbatim_quote for ${path} is not found in that provision's text — quotes must be copied exactly from lookup_entries output for the cited sub-item`,
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
  // count REAL user turns (excludes tool_results and our "[system]" nudges)
  const realUserTurns = transcript.filter(
    (m) =>
      m.role === "user" &&
      Array.isArray(m.content) &&
      m.content.some(
        (b) => b.type === "text" && !String((b as { text?: string }).text ?? "").startsWith("[system]"),
      ),
  ).length;
  const systemBlocks = buildSystemBlocks(annex);
  // 1h TTL: humans answer interview questions slower than the default 5-minute
  // cache — without this, every turn re-writes the ~30k-token prefix at 1.25x
  // and a single slow conversation costs ~3x more than it should
  const system = withCache(systemBlocks as Block[], "1h");
  const tools = [
    LOOKUP_ENTRIES_TOOL,
    LOOKUP_DEFINITIONS_TOOL,
    LOOKUP_GEA_TOOL,
    FINAL_ANSWER_TOOL,
    LICENSE_PATHWAY_TOOL,
  ];
  let usd = 0;

  const call = async (model: string, maxTokens: number, forced: false | string) => {
    const msgs = transcript.map((m, i) =>
      i === transcript.length - 1 ? { ...m, content: withCache(m.content) } : m,
    );
    const resp = await client.complete({
      model,
      max_tokens: maxTokens,
      system,
      messages: msgs,
      tools,
      ...(forced ? { tool_choice: { type: "tool", name: forced } } : {}),
    });
    usd += estimateUsd(model, resp.usage);
    return resp;
  };

  // One question, no tools: guarantees a real, contentful interview turn.
  const askOneQuestion = async (): Promise<TurnResult> => {
    const resp = await client.complete({
      model: models.loop,
      max_tokens: LOOP_MAX_TOKENS,
      system,
      messages: transcript.map((m, i) =>
        i === transcript.length - 1 ? { ...m, content: withCache(m.content) } : m,
      ),
      tools,
      tool_choice: { type: "none" },
    });
    usd += estimateUsd(models.loop, resp.usage);
    transcript.push({ role: "assistant", content: resp.content as Block[] });
    return { type: "question", text: textOf(resp), transcript, usd };
  };

  // The verdict stage: forced strict final_answer on the stronger model, with
  // one retry on validation failure; fail-closed to a question otherwise.
  const produceVerdict = async (): Promise<TurnResult> => {
    {
      for (let attempt = 0; attempt < 2; attempt++) {
        const vResp = await call(models.verdict, VERDICT_MAX_TOKENS, "final_answer");
        const vUse = toolUses(vResp).find((u) => u.name === "final_answer");
        if (!vUse) break;
        const verdict = vUse.input as Verdict;
        const problems = validateVerdict(verdict, annex);
        transcript.push({ role: "assistant", content: vResp.content as Block[] });
        if (problems.length === 0 && verdict.status === "needs_expert" && realUserTurns <= 1) {
          // Giving up on the opening message is premature — a fully specified
          // description may verdict on turn one, but "ask an expert" may not.
          transcript.push({
            role: "user",
            content: [
              {
                type: "tool_result",
                tool_use_id: vUse.id,
                is_error: true,
                content:
                  "[system] needs_expert is premature on the user's opening message. Ask " +
                  "the single most discriminating technical question instead (rule 2).",
              },
            ],
          });
          return askOneQuestion();
        }
        if (problems.length === 0) {
          // close the tool_use so the returned transcript is a valid Anthropic
          // array — a follow-up turn would otherwise 400 on an unpaired tool_use
          transcript.push({
            role: "user",
            content: [{ type: "tool_result", tool_use_id: vUse.id, content: "Verdict recorded." }],
          });
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
  };

  // Stage-2 twin of produceVerdict: forced strict license_pathway, validated,
  // one retry, fail-closed to a question.
  const producePathway = async (): Promise<TurnResult> => {
    for (let attempt = 0; attempt < 2; attempt++) {
      const pResp = await call(models.verdict, VERDICT_MAX_TOKENS, "license_pathway");
      const pUse = toolUses(pResp).find((u) => u.name === "license_pathway");
      if (!pUse) break;
      const pathway = pUse.input as Pathway;
      const problems = validatePathway(pathway, annex);
      transcript.push({ role: "assistant", content: pResp.content as Block[] });
      if (problems.length === 0) {
        transcript.push({
          role: "user",
          content: [{ type: "tool_result", tool_use_id: pUse.id, content: "Pathway recorded." }],
        });
        return {
          type: "pathway",
          text: textOf(pResp),
          transcript,
          pathway: {
            ...pathway,
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
            tool_use_id: pUse.id,
            is_error: true,
            content: `Pathway rejected by validation: ${problems.join("; ")}. Correct and call license_pathway again.`,
          },
        ],
      });
    }
    transcript.push({
      role: "user",
      content: [
        {
          type: "text",
          text:
            "[system] The licensing pathway could not be validated. Ask the user for the " +
            "missing facts (destination, end-use) instead of concluding.",
        },
      ],
    });
    return askOneQuestion();
  };

  for (let i = 0; i < MAX_TOOL_ITERATIONS; i++) {
    const resp = await call(models.loop, LOOP_MAX_TOKENS, false);
    const uses = toolUses(resp);
    const finalCall = uses.find((u) => u.name === "final_answer");
    const pathwayCall = uses.find((u) => u.name === "license_pathway");

    if (pathwayCall && !finalCall) {
      transcript.push({ role: "assistant", content: resp.content as Block[] });
      transcript.push({
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: pathwayCall.id,
            content:
              "Draft received. Now produce the authoritative licensing pathway by calling " +
              "license_pathway with exact verbatim quotes from lookup_gea and full caveats.",
          },
        ],
      });
      return producePathway();
    }

    if (finalCall) {
      // The loop model decided to conclude — the verdict itself is written by
      // the stronger model under the forced strict schema.
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
      return produceVerdict();
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

    const text = textOf(resp);
    transcript.push({ role: "assistant", content: resp.content as Block[] });

    // NAKED-VERDICT ESCALATION: a live test produced a full prose conclusion
    // ("Status: Listed ... 4A003.b") without calling final_answer — bypassing
    // corpus validation entirely. Conclusive-looking prose is never returned:
    // it is escalated into the validated verdict stage instead.
    const conclusive =
      /(^|\n)\s*\*{0,2}(status|result|classification)\*{0,2}\s*:\s*\*{0,2}(listed|not[_ ]?listed|needs[_ ]?expert)/i.test(text) ||
      /\b(is|are)\s+(therefore\s+|clearly\s+|thus\s+)?(listed|not listed)\s+in\s+annex\s+i\b/i.test(text) ||
      /classification result/i.test(text);
    if (conclusive) {
      transcript.push({
        role: "user",
        content: [
          {
            type: "text",
            text:
              "[system] Conclusions must be delivered ONLY through the final_answer tool, " +
              "never as prose. Call final_answer now with complete reasoning, exact " +
              "verbatim quotes and full caveats.",
          },
        ],
      });
      return produceVerdict();
    }

    return { type: "question", text, transcript, usd };
  }

  // Tool budget exhausted — force one real question instead of canned filler.
  transcript.push({
    role: "user",
    content: [
      {
        type: "text",
        text: "[system] Stop looking things up. Ask the user your single most important discriminating question now.",
      },
    ],
  });
  return askOneQuestion();
}
