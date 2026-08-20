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
const VERDICT_MAX_TOKENS = 4000;
// Quotes shorter than this are too weak to anchor — a 3-char fragment appears
// everywhere. Real thresholds and provisions comfortably clear it.
const MIN_QUOTE_CHARS = 12;
// Hard cap on client-supplied history, so a single POST's token cost is bounded
// well under the daily budget (was 200k — a ~50k-token inflation vector).
const MAX_HISTORY_CHARS = 80_000;

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

// The pathway stage validates against the verdict that precedes it — recover
// the most recent final_answer's entry_codes from the transcript.
function lastFinalAnswerIndex(msgs: Msg[]): number {
  for (let i = msgs.length - 1; i >= 0; i--) {
    const m = msgs[i];
    if (
      m.role === "assistant" &&
      Array.isArray(m.content) &&
      m.content.some((b) => b.type === "tool_use" && b.name === "final_answer")
    ) {
      return i;
    }
  }
  return -1;
}

function verdictCodesIn(msgs: Msg[]): string[] {
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
  if (JSON.stringify(out).length > MAX_HISTORY_CHARS) trimOldToolResults(out);
  if (JSON.stringify(out).length > MAX_HISTORY_CHARS) {
    throw new InvalidRequest("conversation_too_long");
  }
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

// The verdict model sometimes garbles an intended-empty eligible_gea into
// tool-syntax artifacts (seen live on a sanctions card — and rejection only
// re-triggers the same glitch on retry, fail-closing a correct outcome).
// Outside gea_available the field carries no meaning, so an id that does not
// resolve is normalised to empty instead of rejected; gea_available keeps
// strict validation because the card headlines the id.
export function normalizePathway(pw: Pathway, annex: AnnexDataset): Pathway {
  if (pw.outcome !== "gea_available" && pw.eligible_gea && !geaById(annex, pw.eligible_gea)) {
    return { ...pw, eligible_gea: "" };
  }
  return pw;
}

// Every Category 5 Part 2 item sits inside EU008's subject matter — a live run
// concluded individual_licence_required for a 5A002 item after testing only
// EU001/EU007, with EU008 never retrieved. Enforced in code, not prompt.
const CAT5P2 = /^5[ADE]002/i;

// trimOldToolResults shrinks old lookup outputs to keep long conversations
// under the history cap, telling the model to re-fetch — but the FORCED
// verdict/pathway stages pin tool_choice to the strict tool, so the model
// CANNOT re-fetch there. Seen live (stage-2 EU008 run): the model told the
// user its source text was truncated and would not classify fully. Before
// forcing, re-execute every trimmed lookup and restore its full output.
// (This can push one request past the history cap; correctness of quoted
// sources outranks the marginal token cost.)
function restoreTrimmedLookups(msgs: Msg[], annex: AnnexDataset): void {
  const usesById = new Map<string, Block>();
  for (const m of msgs) {
    if (m.role !== "assistant" || !Array.isArray(m.content)) continue;
    for (const b of m.content) if (b.type === "tool_use") usesById.set(String(b.id), b);
  }
  for (const m of msgs) {
    if (m.role !== "user" || !Array.isArray(m.content)) continue;
    for (const b of m.content) {
      if (b.type !== "tool_result" || typeof b.content !== "string" || !b.content.includes("…[trimmed")) {
        continue;
      }
      const use = usesById.get(String(b.tool_use_id));
      if (!use || !String(use.name).startsWith("lookup_")) continue;
      b.content = execLookup(annex, String(use.name), (use.input ?? {}) as Record<string, unknown>);
    }
  }
}

// Pathway validation — same discipline as verdicts: quotes verbatim-in-scope,
// referenced GEAs must exist, sanctioned destinations MUST carry the sanctions
// outcome (never a green light), a GEA outcome needs quoted conditions.
// verdictCodes (the stage-1 entry codes, when known) gates the EU008 sweep.
export function validatePathway(pw: Pathway, annex: AnnexDataset, verdictCodes: string[] = []): string[] {
  const problems: string[] = [];
  if (
    pw.outcome === "individual_licence_required" &&
    verdictCodes.some((c) => CAT5P2.test(c.trim())) &&
    geaById(annex, "EU008") &&
    !pw.conditions_quoted.some((c) => c.gea_id.trim().toUpperCase() === "EU008")
  ) {
    problems.push(
      "the classified item is Category 5 Part 2 (5A002/5D002/5E002) — retrieve EU008 via lookup_gea and either conclude gea_available under it or quote the EU008 scope/exclusion text that rules it out",
    );
  }
  if (pw.caveats.length === 0) problems.push("caveats must not be empty");
  if (!pw.destination.trim()) problems.push("destination must be stated");
  if (SANCTIONED_DESTINATIONS.test(pw.destination) && pw.outcome !== "sanctions_review_required") {
    problems.push(
      `destination "${pw.destination}" is under an EU sanctions regime — outcome must be sanctions_review_required`,
    );
  }
  // eligible_gea is either empty or a REAL GEA id — for every outcome (a live
  // run emitted garbage into this field under an individual_licence outcome)
  if (pw.eligible_gea && !geaById(annex, pw.eligible_gea)) {
    problems.push(`eligible_gea ${JSON.stringify(pw.eligible_gea).slice(0, 60)} does not exist in the corpus`);
  }
  if (pw.outcome === "gea_available") {
    if (!pw.eligible_gea) problems.push("gea_available requires eligible_gea");
    if (pw.conditions_quoted.length === 0) {
      problems.push("gea_available requires quoted conditions");
    }
  }
  if (pw.outcome === "individual_licence_required" && pw.conditions_quoted.length === 0) {
    problems.push(
      "individual_licence_required must quote the provision that rules the GEAs out (e.g. the coverage clause or exclusion tested)",
    );
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

  // Reasoning rows carry a met flag: met=false rows are rule-outs (an entry
  // or cross-reference tested and found NOT to apply). A headline needs at
  // least one SUPPORTING row, and rule-out rows are exempt from the headline
  // requirement — the old symmetric checks (every headline backed by any row,
  // every cited code headlined) forced a live verdict to headline "3B001,
  // 3B501" while its own reasoning ruled 3B001 out.
  const headlined = v.entry_codes.map((c) => c.toUpperCase());
  if (v.status === "listed") {
    if (v.entry_codes.length === 0) problems.push("listed verdict needs entry_codes");
    if (v.reasoning.length === 0) problems.push("listed verdict needs reasoning");
    for (const code of v.entry_codes) {
      const backed = v.reasoning.some(
        (r) => (r.entry_code || "").toUpperCase() === code.toUpperCase() && r.met !== false,
      );
      if (!backed) {
        problems.push(
          `entry_code ${code} is headlined but has no supporting reasoning — an entry whose rows all rule it out (met=false) must be removed from entry_codes`,
        );
      }
    }
    for (const r of v.reasoning) {
      const code = (r.entry_code || "").toUpperCase();
      if (r.met !== false && !headlined.includes(code)) {
        problems.push(`reasoning cites ${code} as met but it is not in entry_codes`);
      }
    }
  }
  for (const code of v.entry_codes) {
    if (!entryByCode(annex, code)) problems.push(`entry_code ${code} does not exist in the corpus`);
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
  // FORMULA-DEFINED TERMS (rule 18, enforced in code): where a Technical Note
  // on the cited provision or an ancestor defines a quoted term by formula,
  // any row quoting that term must SHOW the computation. A live first-turn
  // verdict adopted a user-claimed 38 nm MRF that the entry's own K=0.35
  // formula contradicts at any real numerical aperture.
  for (const r of v.reasoning) {
    const entry = entryByCode(annex, r.entry_code);
    if (!entry) continue;
    const cited = (r.dotted_path || "").trim().toUpperCase();
    if (!cited) continue;
    const definedTerms: string[] = [];
    for (const line of entry.verbatim_text.split("\n")) {
      if (!/technical note/i.test(line) || !/formula/i.test(line)) continue;
      const linePath = (line.split(/\s+/)[0] ?? "").toUpperCase();
      if (!linePath.includes(".")) continue;
      if (!(cited === linePath || cited.startsWith(linePath + "."))) continue;
      for (const m of line.matchAll(/['‘]([^'’]{2,60})['’]/g)) definedTerms.push(m[1]);
    }
    if (!definedTerms.some((t) => (r.verbatim_quote || "").includes(t))) continue;
    const expl = r.explanation || "";
    if (!(/formula|calculat/i.test(expl) && /=/.test(expl))) {
      problems.push(
        `${r.dotted_path} turns on a formula-defined term (see its Technical Note) — compute the value from the underlying parameters with the entry's own formula and constants, showing the calculation (e.g. 'MRF = (wavelength × K)/NA = …') in the explanation. Never adopt a user-claimed value for a defined term; if an input such as the numerical aperture is missing, do not conclude — ask the user for it`,
      );
    }
  }

  // N.B. / SEE ALSO cross-references carried by a cited provision (or an
  // ancestor of it) name sibling entries that catch similar equipment on
  // different criteria (3B001.f.1 ↔ 3B501.f: the same defined term with a
  // different K factor plus an overlay criterion). A live first-turn verdict
  // concluded on 3B001.f.1.b without ever testing 3B501 — prompt rules did
  // not stop it, so it is enforced here: every referenced entry must appear
  // somewhere in the verdict (entry_codes, reasoning or caveats), even if
  // only to say why it does not apply.
  if (v.status !== "needs_expert") {
    const mentioned = JSON.stringify(v).toUpperCase();
    const flagged = new Set<string>();
    for (const r of v.reasoning) {
      const entry = entryByCode(annex, r.entry_code);
      if (!entry) continue;
      const cited = (r.dotted_path || "").trim().toUpperCase();
      for (const line of entry.verbatim_text.split("\n")) {
        if (!/\bN\.B\.|SEE ALSO/i.test(line)) continue;
        const linePath = (line.split(/\s+/)[0] ?? "").toUpperCase();
        // root-level N.B.s ("3B001 N.B. SEE ALSO 2B226") span a whole entry —
        // generic context, not an obligation; requiring them taught the model
        // to interview users about isotope separators on a litho scanner
        if (!linePath.includes(".")) continue;
        if (!(cited === linePath || cited.startsWith(linePath + "."))) continue;
        for (const code of line.toUpperCase().match(/\b\d[A-E]\d{3}\b/g) ?? []) {
          if (code === r.entry_code.toUpperCase() || flagged.has(code)) continue;
          if (!entryByCode(annex, code)) continue;
          if (!mentioned.includes(code)) {
            flagged.add(code);
            problems.push(
              `the cited provision ${r.dotted_path} carries a cross-reference (N.B./SEE ALSO) to ${code} — either include ${code} in the verdict or state in caveats why it does not apply or cannot be assessed on the known facts`,
            );
          }
        }
      }
    }
  }
  return problems;
}

// Conclusive-prose detectors, shared by the main loop and the ask-fallback:
// conclusions must reach the user ONLY as validated cards, never as chat text.
function looksPathwayConclusive(text: string): boolean {
  return (
    (/\bEU00[1-8]\b/.test(text) && /(available|applies|eligible|covers|authoris)/i.test(text)) ||
    /individual (export )?(licence|license|authorisation) (is |will be )?(required|needed)/i.test(text) ||
    /\b(sanction|embargo)/i.test(text)
  );
}

function looksVerdictConclusive(text: string): boolean {
  return (
    /(^|\n)\s*\*{0,2}(status|result|classification)\*{0,2}\s*:\s*\*{0,2}(listed|not[_ ]?listed|needs[_ ]?expert)/i.test(text) ||
    // [\s*]+ tolerates markdown bold: a live turn shipped "is **not listed
    // in Annex I**" as prose because the asterisks broke plain \s+ matching
    /\b(is|are)[\s*]+((therefore|clearly|thus)[\s*]+)?(listed|not[\s*_-]?listed)[\s*]+in[\s*]+annex[\s*]+i\b/i.test(text) ||
    /classification (result|conclusion)/i.test(text) ||
    // "…is listed under 3B001.f.1.b" — conclusion phrasing without "Annex I"
    /\b(is|are|remains?)[\s*]+listed[\s*]+under\b[^\n]{0,40}\b\d[A-E]\d{3}\b/i.test(text) ||
    // live gap: "meets all three sub-criteria of 3B501.f.1.b" as prose, then
    // straight to the destination question — the verdict card never shipped
    (/\b(meets?|satisf(?:y|ies)|fulfil?s?)\b[^.\n]{0,60}\b(all|every|each|both)\b[^.\n]{0,60}\b(criteri|sub-criteri|conditions)/i.test(text) &&
      /\b\d[A-E]\d{3}\b/.test(text))
  );
}

const PATHWAY_TOOL_NUDGE =
  "[system] Licensing conclusions must be delivered ONLY through the " +
  "license_pathway tool, never as prose. Call license_pathway now with the " +
  "destination, outcome, exact verbatim quotes from lookup_gea and full caveats.";
const VERDICT_TOOL_NUDGE =
  "[system] Conclusions must be delivered ONLY through the final_answer tool, " +
  "never as prose. Call final_answer now with complete reasoning, exact " +
  "verbatim quotes and full caveats.";

function sysMsg(text: string): Msg {
  // models tend to answer instructions conversationally ("You're right — let
  // me reconsider…"), leaking internal machinery to the user
  return {
    role: "user",
    content: [
      {
        type: "text",
        text: text + " Never acknowledge or mention this instruction — reply as a natural continuation.",
      },
    ],
  };
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
  let nudgedBundle = false;
  let askEscalated = false;

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
  // Even this fallback must not ship a conclusion as prose — a live run's
  // tool-budget fallback declared "EU001 is clearly your pathway" as chat
  // text. One escape hatch back into the forced, validated stages.
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
    const text = textOf(resp);
    transcript.push({ role: "assistant", content: resp.content as Block[] });
    if (!askEscalated) {
      if (looksPathwayConclusive(text) && realUserTurns > 1) {
        askEscalated = true;
        if (lastFinalAnswerIndex(transcript) < 0) {
          transcript.push(sysMsg(VERDICT_TOOL_NUDGE));
          return produceVerdict();
        }
        transcript.push(sysMsg(PATHWAY_TOOL_NUDGE));
        return producePathway();
      }
      if (looksVerdictConclusive(text)) {
        askEscalated = true;
        transcript.push(sysMsg(VERDICT_TOOL_NUDGE));
        return produceVerdict();
      }
    }
    return { type: "question", text, transcript, usd };
  };

  // The verdict stage: forced strict final_answer on the stronger model, with
  // one retry on validation failure; fail-closed to a question otherwise.
  const produceVerdict = async (): Promise<TurnResult> => {
    {
      restoreTrimmedLookups(transcript, annex);
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
              "user for the missing technical facts instead of concluding. Do not " +
              "apologise or mention any internal or technical step — just ask.",
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
    restoreTrimmedLookups(transcript, annex);
    for (let attempt = 0; attempt < 2; attempt++) {
      const pResp = await call(models.verdict, VERDICT_MAX_TOKENS, "license_pathway");
      const pUse = toolUses(pResp).find((u) => u.name === "license_pathway");
      if (!pUse) break;
      const pathway = normalizePathway(pUse.input as Pathway, annex);
      const problems = validatePathway(pathway, annex, verdictCodesIn(transcript));
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
            "missing facts (destination, end-use) instead of concluding. Do not " +
            "apologise or mention any internal or technical step — just ask.",
        },
      ],
    });
    return askOneQuestion();
  };

  // One extra "decision" iteration past the lookup budget: the model is told
  // to conclude via the tools if the facts decide, or ask one question — a
  // live run burned every iteration on GEA lookups and the ask-only fallback
  // then had no way to conclude at all.
  for (let i = 0; i <= MAX_TOOL_ITERATIONS; i++) {
    if (i === MAX_TOOL_ITERATIONS) {
      transcript.push(
        sysMsg(
          "[system] Stop looking things up. If the known facts already decide the " +
            "outcome, call final_answer or license_pathway NOW; otherwise ask the " +
            "user your single most important discriminating question.",
        ),
      );
    }
    const resp = await call(models.loop, LOOP_MAX_TOKENS, false);
    const uses = toolUses(resp);
    const finalCall = uses.find((u) => u.name === "final_answer");
    const pathwayCall = uses.find((u) => u.name === "license_pathway");

    if (pathwayCall && !finalCall) {
      transcript.push({ role: "assistant", content: resp.content as Block[] });
      // STRUCTURAL INVARIANT: no licensing pathway without a validated verdict
      // card first. A live run classified in (inverted) prose, skipped
      // final_answer entirely and went straight to stage 2 — the pathway would
      // have been built on an unvalidated, wrong classification.
      if (lastFinalAnswerIndex(transcript) < 0) {
        transcript.push({
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: pathwayCall.id,
              is_error: true,
              content:
                "[system] No validated classification exists yet — deliver the verdict " +
                "through final_answer first (with the formula calculations shown); the " +
                "licensing pathway comes after.",
            },
          ],
        });
        return produceVerdict();
      }
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
      if (i === MAX_TOOL_ITERATIONS) break; // budget truly spent — fall to the ask
      continue;
    }

    const text = textOf(resp);
    transcript.push({ role: "assistant", content: resp.content as Block[] });

    // NAKED-VERDICT ESCALATION: a live test produced a full prose conclusion
    // ("Status: Listed ... 4A003.b") without calling final_answer — bypassing
    // corpus validation entirely. Conclusive-looking prose is never returned:
    // it is escalated into the validated verdict stage instead.
    if (looksPathwayConclusive(text) && realUserTurns > 1) {
      if (lastFinalAnswerIndex(transcript) < 0) {
        // pathway talk before any validated verdict: classify first
        transcript.push(sysMsg(VERDICT_TOOL_NUDGE));
        return produceVerdict();
      }
      transcript.push(sysMsg(PATHWAY_TOOL_NUDGE));
      return producePathway();
    }

    // ONE-QUESTION DISCIPLINE, enforced once per turn: a live run bundled
    // "scan or repeat?" (non-discriminating — the chapeau covers both) with
    // "system or component?" AFTER every controlling parameter was given.
    // A multi-question turn gets one chance to converge or ask one thing.
    const questionMarks = (text.match(/\?/g) ?? []).length;
    if (questionMarks >= 2 && !nudgedBundle) {
      nudgedBundle = true;
      transcript.push(
        sysMsg(
          "[system] Ask exactly ONE question, phrased once — and only if its answer " +
            "can change the classification. Never re-ask facts already given. If the " +
            "user has already supplied every controlling parameter, conclude now " +
            "(final_answer / license_pathway) instead of asking.",
        ),
      );
      continue;
    }

    // A trimmed source is the model's cue to re-fetch, never a fact to
    // report — a live stage-2 run told the user its text was truncated and
    // declined to classify fully. Nudge it to re-fetch and continue.
    if (/\btruncat|\[trimmed\b/i.test(text)) {
      transcript.push(
        sysMsg(
          "[system] Source text trimmed from this conversation must be re-fetched with " +
            "the lookup tools — do that now and continue. Never mention truncation or " +
            "trimming to the user.",
        ),
      );
      continue;
    }

    // A turn that narrates an intended lookup ("Let me look that up now")
    // without performing it — and asks the user nothing — is not a question.
    // Seen live after a parameter answer: the model announced a lookup and
    // ended its turn. Nudge it to act instead of surfacing the narration.
    const lookupNarration =
      !text.includes("?") &&
      /\b(let me|i need to|i will|i'll|i am going to)\b[^.?!]{0,80}\b(look|retriev|fetch|consult|check)/i.test(text);
    if (lookupNarration) {
      transcript.push(
        sysMsg(
          "[system] Do not narrate lookups — call the lookup tool now, then continue. " +
            "Never end a turn with a statement of intent.",
        ),
      );
      continue;
    }

    if (looksVerdictConclusive(text)) {
      transcript.push(sysMsg(VERDICT_TOOL_NUDGE));
      return produceVerdict();
    }

    // STAGE-2 CONVERGENCE: after a verdict, three answered turns carry the
    // destination, end-use and end-user several times over — a live run still
    // wanted a fourth optional question. Force the pathway tool instead; if
    // facts truly are missing, validation fails closed to a question anyway.
    const verdictAt = lastFinalAnswerIndex(transcript);
    if (verdictAt >= 0) {
      const answersSince = transcript.filter(
        (m, t) =>
          t > verdictAt &&
          m.role === "user" &&
          Array.isArray(m.content) &&
          m.content.some(
            (b) =>
              b.type === "text" && !String((b as { text?: string }).text ?? "").startsWith("[system]"),
          ),
      ).length;
      if (answersSince >= 3) {
        transcript.push(sysMsg(PATHWAY_TOOL_NUDGE));
        return producePathway();
      }
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
