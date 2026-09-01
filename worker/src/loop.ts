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
const LOOP_MAX_TOKENS = 900;
const VERDICT_MAX_TOKENS = 2800;
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

// One entry per model call, in order — the sequential chain IS the latency
// story, so every stage records its wall time and token/cache split.
export interface StageTiming {
  stage: string;
  model: string;
  ms: number;
  in: number;
  out: number;
  cache_read: number;
  cache_write: number;
}

export interface TurnResult {
  type: "question" | "verdict" | "pathway";
  text: string;
  transcript: Msg[];
  verdict?: Verdict & { corpus_version: string; corpus_sha256: string; prompt_sha256: string };
  pathway?: Pathway & { corpus_version: string; corpus_sha256: string; prompt_sha256: string };
  usd: number;
  timings: StageTiming[];
  // set on a LISTED verdict whose in-request licensing continuation could not
  // run (time budget spent) — the page quietly sends one follow-up turn
  continueLicensing?: boolean;
}

// Destinations whose sanctions regimes this tool must FLAG and never resolve —
// separate regulations with their own complexity; a wrong answer here is the
// most expensive mistake the tool could make. Enforced server-side.
const SANCTIONED_DESTINATIONS =
  /\b(russia|russian?|rusia|russie|russland|moscow|moscú|moskau|belarus|bielorrusia|biélorussie|belarusian|minsk|iran(?:ian)?|irán|tehe?ran|teherán|north[ -]?korea|corea del norte|corée du nord|nordkorea|pyongyang|dprk|democratic people'?s republic of korea|syria|siria|syrie|syrien|damascus|crimea|crimée|donetsk|luhansk|myanmar|burma|birmania|venezuela|caracas)\b/i;

export class InvalidRequest extends Error {}

// The pathway stage validates against the verdict that precedes it — recover
// the most recent final_answer's entry_codes from the transcript.
function lastFinalAnswerIndex(msgs: Msg[]): number {
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
const CAT5P2 = /^(5A00[2-4]|5B002|5D002|5E002)/i;

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
  if (!["gea_available", "individual_licence_required", "sanctions_review_required"].includes(String(pw.outcome))) {
    problems.push(`outcome ${JSON.stringify(pw.outcome).slice(0, 40)} is not a valid pathway outcome`);
  }
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
    } else if (
      pw.eligible_gea &&
      !pw.conditions_quoted.some((c) => c.gea_id.trim().toUpperCase() === pw.eligible_gea.trim().toUpperCase())
    ) {
      problems.push(
        `gea_available under ${pw.eligible_gea} must quote at least one condition from ${pw.eligible_gea} itself`,
      );
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
  if (!["listed", "not_listed", "needs_expert"].includes(String(v.status))) {
    problems.push(`status ${JSON.stringify(v.status).slice(0, 40)} is not a valid verdict status`);
  }
  // a not-listed card is the tool's green light — it must show its work:
  // which candidate entries were tested and why each was ruled out
  if (v.status === "not_listed" && v.reasoning.length === 0) {
    problems.push(
      "a not_listed verdict must include reasoning rows (met=false) showing the candidate entries tested and ruled out",
    );
  }
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
    const resolved = provisionText(entry, path);
    if (resolved === null && /^\d[A-E]\d{3}(\.[a-z0-9]+)+$/i.test(path)) {
      problems.push(
        `dotted_path ${path} does not resolve to a provision of ${r.entry_code} — cite the exact sub-item as printed in lookup_entries output`,
      );
      continue;
    }
    const scope = resolved ?? entry.verbatim_text;
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
      continue;
    }
    // arithmetic consistency: a shown computation must AGREE with the claim —
    // a live verdict computed 50,04 nm and declared it "at or below" a 45 nm
    // threshold. Narrow, safe direction only: a supporting row whose computed
    // value EXCEEDS an "…or less" threshold is a false conclusion. (European
    // decimal commas normalised; the last "= N nm" is the final result.)
    if (r.met !== false) {
      const calcs = [...expl.matchAll(/=\s*(\d+(?:[.,]\d+)?)\s*nm/gi)];
      const thr =
        /(\d+(?:[.,]\d+)?)\s*nm\s+or\s+less/i.exec(r.verbatim_quote || "") ??
        /less\s+than\s+or\s+equal\s+to\s+(\d+(?:[.,]\d+)?)\s*nm/i.exec(r.verbatim_quote || "");
      if (calcs.length > 0 && thr) {
        const value = parseFloat(calcs[calcs.length - 1][1].replace(",", "."));
        const threshold = parseFloat(thr[1].replace(",", "."));
        if (value > threshold) {
          problems.push(
            `${r.dotted_path}: the computed ${value} nm EXCEEDS the ${threshold} nm-or-less threshold — this criterion is NOT met. Mark it met=false, do not headline an entry on a failed computation, and if another entry's criteria need a missing parameter (e.g. 'dedicated chuck overlay'), ask the user for it instead of concluding`,
          );
        }
      }
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

// QUESTION-DEFECT DETECTORS (pure, deterministic — they run on every
// candidate question before any judge model is consulted). Fixtures: the
// live failures "confirm the exact numerical aperture again, e.g. 1.35 or
// 1.350?" (echo + equal alternatives) and the five-times re-asked
// destination question (near-duplicate).
const UNIT_FACTORS: Record<string, number> = {
  nm: 1, nanometre: 1, nanometres: 1, nanometer: 1, nanometers: 1,
  um: 1000, µm: 1000, micrometre: 1000, micron: 1000, microns: 1000,
  mm: 1e6, millimetre: 1e6, millimeter: 1e6,
};

function numberTokens(text: string): { value: number; unit: string }[] {
  const out: { value: number; unit: string }[] = [];
  const re = /(\d+(?:[.,]\d+)?)\s*(nm|nanometres?|nanometers?|um|µm|micrometres?|microns?|mm|millimetres?|millimeters?)?\b/gi;
  for (const m of text.matchAll(re)) {
    const value = parseFloat(m[1].replace(",", "."));
    if (!Number.isFinite(value)) continue;
    const unitRaw = (m[2] ?? "").toLowerCase();
    const factor = UNIT_FACTORS[unitRaw];
    out.push(factor ? { value: value * factor, unit: "nm" } : { value, unit: unitRaw || "" });
  }
  return out;
}

function isHedged(text: string, value: number): boolean {
  const re = new RegExp(
    "\\b(about|roughly|approx\\w*|around|circa|~)\\s*" + String(value).replace(".", "[.,]"),
    "i",
  );
  return re.test(text);
}

// a question that echoes a number the user already stated, in the same
// sentence as a confirm-verb, is asking for nothing
export function questionEchoesStatedValue(candidate: string, userTexts: string[]): boolean {
  const stated = userTexts.flatMap((t) => numberTokens(t));
  if (stated.length === 0) return false;
  for (const sentence of candidate.split(/(?<=[.?!])\s+/)) {
    if (!/\b(confirm|verify|double.?check|re.?state|again)\b/i.test(sentence)) continue;
    for (const tok of numberTokens(sentence)) {
      const echoed = stated.some((s) => s.unit === tok.unit && Math.abs(s.value - tok.value) < 1e-9);
      if (echoed && !userTexts.some((t) => isHedged(t, tok.value))) return true;
    }
  }
  return false;
}

// "e.g. 1.35 exactly, or a more precise decimal like 1.350" — alternatives
// that normalise to the same number ask for nothing
export function questionOffersEqualAlternatives(candidate: string): boolean {
  const re = /(\d+(?:[.,]\d+)?)\s*(nm|um|µm|mm)?[^.?\n\d]{0,24}\bor\b[^.?\n\d]{0,40}(\d+(?:[.,]\d+)?)\s*(nm|um|µm|mm)?/gi;
  for (const m of candidate.matchAll(re)) {
    const a = parseFloat(m[1].replace(",", ".")) * (UNIT_FACTORS[(m[2] ?? "").toLowerCase()] ?? 1);
    const b = parseFloat(m[3].replace(",", ".")) * (UNIT_FACTORS[(m[4] ?? "").toLowerCase()] ?? 1);
    if (Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) < 1e-9) return true;
  }
  return false;
}

// "Just classify it — I don't need the licence": the user may opt out of the
// licensing stage entirely; the classification card then ships alone and
// destination questions are blocked. PRECISION over recall: a false opt-out
// silently amputates half the product ("license key", "without licensing
// fees" and the like must never fire), while a missed opt-out costs nothing —
// the model still honours rule 21 on its own. A LATER message that explicitly
// asks about licensing opts back in; last signal wins, so a wrong call in
// either direction is always recoverable in one message.
const OPT_OUT_LICENSING =
  /\b(only|just)\s+(the\s+)?classif|\b(only|just)\b[^.!?\n]{0,15}\b(need|want)\b[^.!?\n]{0,25}\bclassif|\ball\s+i\s+need\b[^.!?\n]{0,25}\bclassif|\bclassif\w*[^.!?\n]{0,25}\bis\s+(all\s+i\s+need|enough)\b|\b(no|don'?t|do not|not)\b[^.!?\n]{0,25}\b(need|want|require|care about|interested in)\b[^.!?\n]{0,15}\b(the|a|any)?\s*(licen[cs]e|licensing|authori[sz]\w*|pathway)\b(?!\s*(key|keys|server|fee|fees|agreement|terms|token))|\b(don'?t|do not)\s+(worry|bother)\s+about\b[^.!?\n]{0,25}\b(licen[cs]\w*|licensing|pathway|authori[sz])|\bskip\b[^.!?\n]{0,20}\b(licen[cs]\w*|licensing|pathway|authori[sz])|\b(solo|s[oó]lo|solamente)\b[^.!?\n]{0,20}\bclasificaci|clasificaci[oó]n\s+(solo|s[oó]lo|solamente)\b/i;

// Re-opt-in must be an explicit licensing ASK, not a stray mention — live
// answers legitimately contain "license key", "authorized personnel", "EU001-
// compliant" and must not silently cancel a genuine opt-out.
const OPT_BACK_IN =
  /(licen[cs]\w*|licensing|authori[sz]\w*|pathway|GEA|EU00[1-8])\b[^.!?\n]{0,40}\?|\b(which|what)\b[^.!?\n]{0,30}\b(licen[cs]e|licensing|authorisation|authorization|pathway|GEA|EU00[1-8])\b|\b(do\s+)?(i|we)\s+(need|want|get|apply\s+for)\b[^.!?\n]{0,25}\b(a\s+|the\s+)?(licen[cs]e|authorisation|authorization|permit)\b/i;

export function wantsClassificationOnly(userTexts: string[]): boolean {
  let only = false;
  for (const t of userTexts) {
    if (OPT_OUT_LICENSING.test(t)) only = true;
    else if (only && OPT_BACK_IN.test(t)) only = false;
  }
  return only;
}

// Questions that only serve the licensing stage — blocked once the user has
// opted out of it. Deliberately narrow: "end-use"/"exported to" appear in
// legitimate ITEM questions (decontrol notes, cryptographic APIs), so only
// unambiguous destination asks are gated; rule 21 covers the rest.
export function questionAsksLicensingFacts(candidate: string): boolean {
  return /\b(destination|destin[oa]\b|country\s+of\s+destination|(which|what)\s+country|consignee|recipient\s+country)\b/i.test(
    candidate,
  );
}

function tokenSet(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter((w) => w.length > 2),
  );
}

// near-duplicate of a question the user has ALREADY answered — re-asking is
// forbidden whatever caused it (an unanswered question may be re-asked)
export function questionNearDuplicate(candidate: string, answeredQuestions: string[]): boolean {
  const c = tokenSet(candidate);
  if (c.size === 0) return false;
  for (const q of answeredQuestions) {
    const s = tokenSet(q);
    if (s.size === 0) continue;
    let inter = 0;
    for (const w of c) if (s.has(w)) inter++;
    const union = c.size + s.size - inter;
    if (union > 0 && inter / union >= 0.8) return true;
  }
  return false;
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

// raw tool-call syntax leaking as chat text: the model wrote its invocation
// inline (or was truncated mid-call) instead of calling the tool — a live
// turn shipped '<parameter name="status">listed' plus half a JSON array
export function looksToolSyntaxLeak(text: string): boolean {
  return /<parameter\s+name=|<\/?antml|<invoke\b|"dotted_path"\s*:|"entry_codes"\s*:|"conditions_quoted"\s*:|"verbatim_quote"\s*:/.test(
    text,
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
    // "This matches 5A002.a.1" / "falls under 3B501" / "is controlled under…"
    // — declarative entry-assignments are conclusions, whatever the phrasing
    /\b(matches|falls[\s*]+under|(controlled|classified|settled|resolved)[\s*]+under)\b[^\n]{0,40}\b\d[A-E]\d{3}\b/i.test(text) ||
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
const STAGE2_CONTINUE_NUDGE =
  "[system] Verdict recorded. Continue straight into the licensing stage " +
  "(rule 11): if the destination, end-use and end-user are already stated, " +
  "retrieve the relevant authorisations with lookup_gea and call " +
  "license_pathway; otherwise ask the single most important licensing " +
  "question (destination first).";

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
  judge?: ClaudeClient,
  // fires at the START of every model call — the streaming handler forwards
  // these as live progress lines so the page never shows a dead wait
  onStage?: (stage: string) => void,
  // elapsed-time ceiling for optional second attempts and the in-request
  // licensing continuation. Buffered responses keep 45s (the edge cancels
  // around 100s time-to-first-byte); a streaming response sends bytes from the
  // first stage, so its budget can safely be double that.
  timeBudgetMs?: number,
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
  const timings: StageTiming[] = [];
  const record = (stage: string, model: string, t0: number, resp: ClaudeResponse) => {
    timings.push({
      stage,
      model,
      ms: Date.now() - t0,
      in: resp.usage.input_tokens,
      out: resp.usage.output_tokens,
      cache_read: resp.usage.cache_read_input_tokens ?? 0,
      cache_write: resp.usage.cache_creation_input_tokens ?? 0,
    });
  };
  let nudgedBundle = false;
  let askEscalated = false;
  let conclusiveRegen = false;
  // Cloudflare's edge cancels requests around 100s — a forced 4k-token
  // retry on top of a long turn crosses it and the user sees a dead reply.
  // Past this elapsed budget, skip second forced attempts and fail closed
  // (the quick question turn keeps the response comfortably under the limit).
  const startedAt = Date.now();
  const budgetMs = timeBudgetMs ?? 45_000;
  const outOfTime = () => Date.now() - startedAt > budgetMs;
  // a 4k-token forced card alone takes ~60-80s to generate — affordable at
  // the start of a turn, fatal after slow interview pre-steps. Slow turns
  // get a tighter card budget; validation fail-closes if it truncates.
  const cardBudget = () => (Date.now() - startedAt > budgetMs * 0.45 ? 2400 : VERDICT_MAX_TOKENS);

  // real user answers given after the recorded verdict — the stage-2
  // convergence signal, needed by the main loop AND the ask-fallback
  const answersSinceVerdict = () => {
    const at = lastFinalAnswerIndex(transcript);
    if (at < 0) return -1;
    return transcript.filter(
      (m, t) =>
        t > at &&
        m.role === "user" &&
        Array.isArray(m.content) &&
        m.content.some(
          (b) => b.type === "text" && !String((b as { text?: string }).text ?? "").startsWith("[system]"),
        ),
    ).length;
  };

  // The forced pathway stage cannot fetch, so it must never be starved of
  // quotable text: inject the FULL Annex II corpus as a synthetic lookup
  // exchange once per turn. A live run looped five near-identical questions
  // because every forced card was rejected for unquotable GEA text.
  let geaInjected = false;
  const ensureGeaContext = () => {
    if (geaInjected) return;
    // the transcript is replayed every turn — a previous turn's injection
    // persists, and re-injecting would grow tokens linearly per turn
    if (
      transcript.some(
        (m) =>
          Array.isArray(m.content) &&
          m.content.some((b) => b.type === "tool_use" && String(b.id ?? "").startsWith("srv_gea_")),
      )
    ) {
      geaInjected = true;
      return;
    }
    geaInjected = true;
    const ids = ["EU001", "EU002", "EU003", "EU004", "EU005", "EU006", "EU007", "EU008", "COMMON_LIST"];
    const texts = ids
      .map((id) => {
        const t = geaScopeText(annex, id);
        return t ? `=== ${id} ===\n${t}` : `No GEA ${id} in this corpus version.`;
      })
      .join("\n\n");
    const useId = `srv_gea_${transcript.length}`;
    transcript.push({
      role: "assistant",
      content: [{ type: "tool_use", id: useId, name: "lookup_gea", input: { ids } }],
    });
    transcript.push({
      role: "user",
      content: [{ type: "tool_result", tool_use_id: useId, content: texts }],
    });
  };

  // QUESTION GATE: a question may only ship if it seeks a genuinely missing,
  // outcome-relevant fact. A cheap judge reads the user's stated facts and
  // the candidate question; REDUNDANT triggers one retry with a pointed
  // nudge. Live catalog this kills: "confirm the NA again with more
  // decimals", "measured or a marketing spec?", re-asked destination facts.
  // The judge is optional (tests) and can never block a turn on failure.
  let questionVetted = false;
  const vetQuestion = async (candidate: string): Promise<boolean> => {
    if (!judge || questionVetted) return true;
    questionVetted = true;
    const facts = transcript
      .filter((m) => m.role === "user" && Array.isArray(m.content))
      .map((m) =>
        (m.content as Block[])
          .filter((b) => b.type === "text")
          .map((b) => String((b as { text?: string }).text ?? ""))
          .join("\n"),
      )
      .filter((t) => t && !t.startsWith("[system]"))
      .join("\n---\n");
    try {
      const tJudge = Date.now();
      const resp = await judge.complete({
        model: "claude-haiku-4-5",
        max_tokens: 8,
        system:
          "You judge whether an interview question is worth asking in a technical-legal " +
          "classification interview. Reply with exactly one word. REDUNDANT if the " +
          "question is already answered by the user's stated facts, asks to confirm, " +
          "re-state or refine the precision of a stated value, or if every plausible " +
          "answer leads to the same outcome. Otherwise NEEDED. When unsure, NEEDED.",
        messages: [
          {
            role: "user",
            content: `FACTS THE USER HAS STATED:\n${facts}\n\nCANDIDATE QUESTION:\n${candidate}`,
          },
        ],
        thinking: { type: "disabled" },
      });
      record("question-judge", "claude-haiku-4-5", tJudge, resp);
      usd += estimateUsd("claude-haiku-4-5", resp.usage);
      return !/REDUNDANT/i.test(textOf(resp));
    } catch {
      return true;
    }
  };
  const QUESTION_GATE_NUDGE =
    "[system] That question is already answered by the user's stated facts, or no " +
    "answer to it would change the outcome. Re-read the user's messages, use the " +
    "facts exactly as stated, and either ask for a DIFFERENT genuinely missing " +
    "discriminating parameter or conclude now via final_answer / license_pathway.";

  // THE question chokepoint: every candidate question passes the
  // deterministic defect detectors on every attempt, then (once per turn)
  // the judge. First offence: one pointed retry. Second offence: conclude —
  // the forced stages fail closed to a question if facts genuinely are
  // missing, so a wrong forced conclude cannot ship.
  let gateNudged = false;
  let gateEscalated = false;
  const realUserTextList = (): string[] =>
    transcript
      .filter((m) => m.role === "user" && Array.isArray(m.content))
      .map((m) =>
        (m.content as Block[])
          .filter((b) => b.type === "text")
          .map((b) => String((b as { text?: string }).text ?? ""))
          .join("\n"),
      )
      .filter((t) => t && !t.startsWith("[system]"));
  const classifyOnly = () => wantsClassificationOnly(realUserTextList());
  const answeredAssistantQuestions = (): string[] => {
    const out: string[] = [];
    for (let i = 0; i < transcript.length - 1; i++) {
      const m = transcript[i];
      const next = transcript[i + 1];
      if (m.role !== "assistant" || !Array.isArray(m.content)) continue;
      const text = (m.content as Block[])
        .filter((b) => b.type === "text")
        .map((b) => String((b as { text?: string }).text ?? ""))
        .join("\n");
      if (!text.includes("?")) continue;
      const answered =
        next?.role === "user" &&
        Array.isArray(next.content) &&
        (next.content as Block[]).some(
          (b) => b.type === "text" && !String((b as { text?: string }).text ?? "").startsWith("[system]"),
        );
      if (answered) out.push(text);
    }
    return out;
  };
  const shipQuestion = async (
    text: string,
    retry: () => Promise<TurnResult>,
  ): Promise<TurnResult | null> => {
    const userTexts = realUserTextList();
    const blocked =
      questionEchoesStatedValue(text, userTexts) ||
      questionOffersEqualAlternatives(text) ||
      questionNearDuplicate(text, answeredAssistantQuestions()) ||
      (classifyOnly() && questionAsksLicensingFacts(text)) ||
      !(await vetQuestion(text));
    if (!blocked) return null;
    if (!gateNudged) {
      gateNudged = true;
      transcript.push(sysMsg(QUESTION_GATE_NUDGE));
      return retry();
    }
    if (!gateEscalated) {
      gateEscalated = true;
      if (lastFinalAnswerIndex(transcript) >= 0) {
        transcript.push(sysMsg(PATHWAY_TOOL_NUDGE));
        return producePathway();
      }
      transcript.push(sysMsg(VERDICT_TOOL_NUDGE));
      return produceVerdict();
    }
    return null; // bounded: after nudge + escalation, ship rather than loop
  };

  const call = async (model: string, maxTokens: number, forced: false | string) => {
    const msgs = transcript.map((m, i) =>
      i === transcript.length - 1 ? { ...m, content: withCache(m.content) } : m,
    );
    const t0 = Date.now();
    onStage?.(forced ? `card:${forced}` : "interview");
    const resp = await client.complete({
      model,
      max_tokens: maxTokens,
      system,
      messages: msgs,
      tools,
      // models with reasoning enabled by default (Sonnet 5) emit thinking
      // blocks that break textOf and poison the client-held transcript —
      // this pipeline's structured discipline needs plain responses
      thinking: { type: "disabled" },
      // disable_parallel_tool_use: a forced response carrying TWO parallel
      // final_answer blocks would leave an unpaired sibling tool_use and 400
      // the continuation (or the next turn), discarding a validated verdict
      ...(forced
        ? { tool_choice: { type: "tool", name: forced, disable_parallel_tool_use: true } }
        : {}),
    });
    record(forced ? `card:${forced}` : "interview", model, t0, resp);
    usd += estimateUsd(model, resp.usage);
    return resp;
  };

  // One question, no tools: guarantees a real, contentful interview turn.
  // Even this fallback must not ship a conclusion as prose — a live run's
  // tool-budget fallback declared "EU001 is clearly your pathway" as chat
  // text. One escape hatch back into the forced, validated stages.
  const askOneQuestion = async (): Promise<TurnResult> => {
    const tAsk = Date.now();
    onStage?.("ask-fallback");
    const resp = await client.complete({
      model: models.loop,
      max_tokens: LOOP_MAX_TOKENS,
      system,
      messages: transcript.map((m, i) =>
        i === transcript.length - 1 ? { ...m, content: withCache(m.content) } : m,
      ),
      tools,
      thinking: { type: "disabled" },
      tool_choice: { type: "none" },
    });
    record("ask-fallback", models.loop, tAsk, resp);
    usd += estimateUsd(models.loop, resp.usage);
    const text = textOf(resp);
    transcript.push({ role: "assistant", content: resp.content as Block[] });
    if (!askEscalated && looksToolSyntaxLeak(text)) {
      askEscalated = true;
      if (/license_pathway|"outcome"|"eligible_gea"/.test(text) && lastFinalAnswerIndex(transcript) >= 0) {
        transcript.push(sysMsg(PATHWAY_TOOL_NUDGE));
        return producePathway();
      }
      transcript.push(sysMsg(VERDICT_TOOL_NUDGE));
      return produceVerdict();
    }
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
      // stage-2 convergence applies to the fail-closed path too: the live
      // five-question loop lived entirely inside this fallback, where the
      // main loop's convergence check never runs
      if (answersSinceVerdict() >= 3 && !outOfTime()) {
        askEscalated = true;
        transcript.push(sysMsg(PATHWAY_TOOL_NUDGE));
        return producePathway();
      }
      // dead air applies here too: a fallback turn that asks nothing after a
      // verdict strands the user — one more forced attempt with feedback.
      // An entirely EMPTY reply is the extreme case of the same failure.
      if ((!text.includes("?") || !text) && answersSinceVerdict() >= 1 && !outOfTime()) {
        askEscalated = true;
        transcript.push(sysMsg(PATHWAY_TOOL_NUDGE));
        return producePathway();
      }
    }
    // after an escalation round-trip (askEscalated set) conclusive prose can
    // reach here again — regenerate once rather than ship a naked conclusion
    if (
      askEscalated &&
      !conclusiveRegen &&
      (looksVerdictConclusive(text) || (looksPathwayConclusive(text) && realUserTurns > 1))
    ) {
      conclusiveRegen = true;
      transcript.push(
        sysMsg("[system] State no conclusion in prose. Ask your single most important question, plainly."),
      );
      return askOneQuestion();
    }
    const escalated = await shipQuestion(text, () => askOneQuestion());
    if (escalated) return escalated;
    return { type: "question", text, transcript, usd, timings };
  };

  // The verdict stage: forced strict final_answer on the stronger model, with
  // one retry on validation failure; fail-closed to a question otherwise.
  const produceVerdict = async (): Promise<TurnResult> => {
    {
      restoreTrimmedLookups(transcript, annex);
      for (let attempt = 0; attempt < 2; attempt++) {
        if (attempt > 0 && outOfTime()) break;
        const vResp = await call(models.verdict, cardBudget(), "final_answer");
        const vUse = toolUses(vResp).find((u) => u.name === "final_answer");
        if (!vUse) break;
        // the API does not hard-enforce required fields on tool inputs — a
        // live call omitted an array and the validator crashed on .length.
        // Missing fields become validation problems, never TypeErrors.
        const verdict = {
          status: "needs_expert",
          entry_codes: [],
          reasoning: [],
          caveats: [],
          definitions_used: [],
          ...(vUse.input as Partial<Verdict>),
        } as Verdict;
        const problems = validateVerdict(verdict, annex);
        transcript.push({ role: "assistant", content: vResp.content as Block[] });
        // needs_expert is premature on the opening message, and equally when
        // the verdict's own text says a user-suppliable parameter is missing —
        // a live card declared "cannot be concluded because the overlay has
        // not been provided" instead of simply asking for the overlay.
        const missingParam =
          verdict.status === "needs_expert" &&
          /\b(parameter|value|figure|overlay|aperture|endurance|wavelength|specification)\b[^.]{0,80}\bnot (yet |been )*(provided|supplied|stated|given)|\bnot (yet |been )*(provided|supplied|stated|given)\b[^.]{0,40}\b(parameter|value|figure)\b/i.test(
            JSON.stringify(verdict),
          );
        if (problems.length === 0 && verdict.status === "needs_expert" && (realUserTurns <= 1 || missingParam)) {
          transcript.push({
            role: "user",
            content: [
              {
                type: "tool_result",
                tool_use_id: vUse.id,
                is_error: true,
                content:
                  "[system] needs_expert is premature when the user can still supply the " +
                  "missing fact. Ask the single most discriminating technical question " +
                  "instead (rule 2).",
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
          // ONE INTERVIEW, ONE CARD: a listed verdict flows straight into the
          // licensing stage in the SAME request (rule 11) — unless the user
          // opted out of licensing, or the time budget is already spent (the
          // page then quietly sends the one follow-up turn instead).
          if (verdict.status === "listed" && !classifyOnly() && !outOfTime()) {
            const cont = await continueToPathway();
            if (cont) return cont;
          }
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
            timings,
            ...(verdict.status === "listed" && !classifyOnly() ? { continueLicensing: true } : {}),
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
      // the guarded fallback carries every question/conclusion protection —
      // this exit used to run raw with tools enabled and no guards at all
      return askOneQuestion();
    }
  };

  // The single-card flow delivers classification and pathway together, so the
  // pathway result re-attaches the verdict recorded earlier in this
  // conversation. The transcript is client-held and untrusted: the recovered
  // verdict is re-validated against the corpus before it is echoed back, and
  // a forged one is simply dropped (the pathway card then stands alone).
  const recordedVerdict = ():
    | (Verdict & { corpus_version: string; corpus_sha256: string })
    | undefined => {
    const at = lastFinalAnswerIndex(transcript);
    if (at < 0) return undefined;
    const use = (transcript[at].content as Block[]).find(
      (b) => b.type === "tool_use" && b.name === "final_answer",
    );
    if (!use) return undefined;
    const v = {
      status: "needs_expert",
      entry_codes: [],
      reasoning: [],
      caveats: [],
      definitions_used: [],
      ...(use.input as Partial<Verdict>),
    } as Verdict;
    if (validateVerdict(v, annex).length > 0) return undefined;
    return { ...v, corpus_version: annex.corpus_version, corpus_sha256: annex.sha256 };
  };

  // Stage-2 twin of produceVerdict: forced strict license_pathway, validated,
  // one retry, fail-closed to a question.
  const producePathway = async (): Promise<TurnResult> => {
    // single chokepoint for the opt-out: every escalation route lands here,
    // so an opted-out user can never receive a pathway determination —
    // whatever prose or convergence rule tried to force one
    if (classifyOnly()) {
      transcript.push(
        sysMsg(
          "[system] The user asked for the classification only — do not determine or " +
            "discuss a licensing pathway. Answer their question or ask what else they " +
            "need about the classification.",
        ),
      );
      return askOneQuestion();
    }
    restoreTrimmedLookups(transcript, annex);
    ensureGeaContext();
    for (let attempt = 0; attempt < 2; attempt++) {
      if (attempt > 0 && outOfTime()) break;
      const pResp = await call(models.verdict, cardBudget(), "license_pathway");
      const pUse = toolUses(pResp).find((u) => u.name === "license_pathway");
      if (!pUse) break;
      // same field-defaulting discipline as the verdict stage — see above
      const pathway = normalizePathway(
        {
          destination: "",
          eligible_gea: "",
          outcome: "individual_licence_required",
          conditions_quoted: [],
          caveats: [],
          ...(pUse.input as Partial<Pathway>),
        } as Pathway,
        annex,
      );
      const problems = validatePathway(pathway, annex, verdictCodesIn(transcript));
      transcript.push({ role: "assistant", content: pResp.content as Block[] });
      if (problems.length === 0) {
        transcript.push({
          role: "user",
          content: [{ type: "tool_result", tool_use_id: pUse.id, content: "Pathway recorded." }],
        });
        const sha = await promptSha256();
        const rv = recordedVerdict();
        return {
          type: "pathway",
          text: textOf(pResp),
          transcript,
          ...(rv ? { verdict: { ...rv, prompt_sha256: sha } } : {}),
          pathway: {
            ...pathway,
            corpus_version: annex.corpus_version,
            corpus_sha256: annex.sha256,
            prompt_sha256: sha,
          },
          usd,
          timings,
        };
      }
      console.log("pathway rejected:", problems.join("; ").slice(0, 300));
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

  // The in-request licensing continuation: after a listed verdict records, let
  // the loop model take up to two more steps toward license_pathway — lookups
  // execute, a genuine licensing question ships through the same gates, and a
  // genuine license_pathway call proceeds to the forced validated stage.
  // Anything else — dead air, narration, conclusive prose, leaked tool syntax
  // — is ROLLED BACK, never escalated: with zero post-verdict user input a
  // forced pathway would have to fabricate the destination (the schema
  // requires one), and a fabricated destination can even mask a sanctioned
  // one. Returns null in that case (and on time/steps running out); the
  // verdict then ships alone with continueLicensing set, and the page's
  // follow-up turn re-enters the fully-gated stage-2 flow.
  const continueToPathway = async (): Promise<TurnResult | null> => {
    transcript.push(sysMsg(STAGE2_CONTINUE_NUDGE));
    for (let k = 0; k < 2; k++) {
      if (outOfTime()) return null;
      const resp = await call(models.loop, LOOP_MAX_TOKENS, false);
      const uses = toolUses(resp);
      const pathwayCall = uses.find((u) => u.name === "license_pathway");
      transcript.push({ role: "assistant", content: resp.content as Block[] });
      if (uses.length > 0) {
        transcript.push({
          role: "user",
          // every sibling tool_use must be answered or the next API call 400s
          content: uses.map((u) =>
            u === pathwayCall
              ? {
                  type: "tool_result",
                  tool_use_id: u.id,
                  content:
                    "Draft received. Now produce the authoritative licensing pathway by calling " +
                    "license_pathway with exact verbatim quotes from lookup_gea and full caveats.",
                }
              : {
                  type: "tool_result",
                  tool_use_id: u.id,
                  content: execLookup(annex, String(u.name), (u.input ?? {}) as Record<string, unknown>),
                },
          ),
        });
        if (pathwayCall) return producePathway();
        continue; // lookups only — one more step
      }
      const text = textOf(resp);
      if (
        !text ||
        !text.includes("?") ||
        looksToolSyntaxLeak(text) ||
        looksPathwayConclusive(text) ||
        looksVerdictConclusive(text)
      ) {
        transcript.pop(); // the reply never happened — the verdict ships clean
        return null;
      }
      const escalated = await shipQuestion(text, () => askOneQuestion());
      if (escalated) return escalated;
      return { type: "question", text, transcript, usd, timings };
    }
    return null;
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
        // every sibling tool_use must be answered or the next API call 400s
        content: uses.map((u) =>
          u === pathwayCall
            ? {
                type: "tool_result",
                tool_use_id: u.id,
                content:
                  "Draft received. Now produce the authoritative licensing pathway by calling " +
                  "license_pathway with exact verbatim quotes from lookup_gea and full caveats.",
              }
            : {
                type: "tool_result",
                tool_use_id: u.id,
                content: execLookup(annex, String(u.name), (u.input ?? {}) as Record<string, unknown>),
              },
        ),
      });
      return producePathway();
    }

    if (finalCall) {
      // The loop model decided to conclude — the verdict itself is written by
      // the stronger model under the forced strict schema.
      transcript.push({ role: "assistant", content: resp.content as Block[] });
      transcript.push({
        role: "user",
        // every sibling tool_use must be answered or the next API call 400s
        content: uses.map((u) =>
          u === finalCall
            ? {
                type: "tool_result",
                tool_use_id: u.id,
                content:
                  "Draft framework received. Now produce the authoritative final verdict by " +
                  "calling final_answer with complete reasoning, exact verbatim quotes and " +
                  "full caveats.",
              }
            : {
                type: "tool_result",
                tool_use_id: u.id,
                content: execLookup(annex, String(u.name), (u.input ?? {}) as Record<string, unknown>),
              },
        ),
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
    if (looksToolSyntaxLeak(text)) {
      if (/license_pathway|"outcome"|"eligible_gea"/.test(text) && lastFinalAnswerIndex(transcript) >= 0) {
        transcript.push(sysMsg(PATHWAY_TOOL_NUDGE));
        return producePathway();
      }
      transcript.push(sysMsg(VERDICT_TOOL_NUDGE));
      return produceVerdict();
    }

    if (looksPathwayConclusive(text) && realUserTurns > 1) {
      if (lastFinalAnswerIndex(transcript) < 0) {
        // pathway talk before any validated verdict: classify first
        transcript.push(sysMsg(VERDICT_TOOL_NUDGE));
        return produceVerdict();
      }
      transcript.push(sysMsg(PATHWAY_TOOL_NUDGE));
      return producePathway();
    }

    // EMPTY REPLY: a model turn with no text and no tool call must never
    // reach the user as a blank bubble — seen live after a destination
    // answer. Nudge once and continue; the loop bound still applies.
    if (!text) {
      transcript.push(
        sysMsg(
          "[system] Your reply was empty. Ask your single most important question, " +
            "or conclude now via final_answer / license_pathway.",
        ),
      );
      continue;
    }

    // POST-VERDICT DEAD AIR: a stage-2 turn that asks nothing and concludes
    // nothing ("No further facts are needed — let me finalize this.") ends
    // the turn with the user stranded. Asking nothing means it is time to
    // produce the card; validation still fails closed if facts are missing.
    if (!text.includes("?") && answersSinceVerdict() >= 1) {
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
      /\b(let me|i need to|i will|i'll|i am going to)\b[^.?!]{0,80}\b(look|retriev|fetch|consult|check|finali[sz]|conclud|proceed|deliver)/i.test(text);
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
    // PRE-VERDICT CONVERGENCE: six answered turns with no verdict is an
    // interview that will not land on its own — a live run declared "all the
    // technical facts are in hand" and asked another question anyway. Force
    // the verdict; fail-closed asks the one genuinely missing question.
    const verdictAt = lastFinalAnswerIndex(transcript);
    if (verdictAt < 0 && realUserTurns >= 6) {
      transcript.push(sysMsg(VERDICT_TOOL_NUDGE));
      return produceVerdict();
    }
    if (verdictAt >= 0) {
      const answersSince = answersSinceVerdict();
      if (answersSince >= 3) {
        transcript.push(sysMsg(PATHWAY_TOOL_NUDGE));
        return producePathway();
      }
    }

    const escalated = await shipQuestion(text, () => askOneQuestion());
    if (escalated) return escalated;

    return { type: "question", text, transcript, usd, timings };
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
