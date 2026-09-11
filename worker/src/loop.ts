// One human turn = one call here. Internally the model may take several tool
// steps (lookups need no human input, so they loop server-side, bounded).
// Convergence is two-stage: the cheap loop model decides WHEN to conclude by
// calling final_answer; the verdict model then writes the authoritative verdict
// under a forced, strict schema — and the Worker validates it against the
// corpus before anyone sees it (a bare or uncited verdict is a bug, enforced
// by code, not prompt).

import type { AnnexDataset } from "./annexData";
import { definitionsFor, entryByCode, geaScopeText } from "./annexData";
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

import {
  normalizePathway,
  validatePathway,
  validateVerdict,
} from "./validate";
import {
  questionAsksLicensingFacts,
  questionCitesProvision,
  questionEchoesStatedValue,
  questionNearDuplicate,
  questionOffersEqualAlternatives,
  wantsClassificationOnly,
} from "./questionGate";
import {
  InvalidRequest,
  lastFinalAnswerIndex,
  sanitizeMessages,
  verdictCodesIn,
  verdictMarker,
  verifyVerdictMarkers,
  type Block,
  type Msg,
} from "./transcript";

// Re-exports: the public surface of the loop module is unchanged for
// index.ts and the test suite.
export { InvalidRequest, sanitizeMessages, verifyVerdictMarkers } from "./transcript";
export { normalizePathway, validatePathway, validateVerdict } from "./validate";
export {
  questionAsksLicensingFacts,
  questionCitesProvision,
  questionEchoesStatedValue,
  questionNearDuplicate,
  questionOffersEqualAlternatives,
  wantsClassificationOnly,
} from "./questionGate";

const MAX_TOOL_ITERATIONS = 3;
const LOOP_MAX_TOKENS = 900;
const VERDICT_MAX_TOKENS = 2800;

export interface Models {
  loop: string;
  verdict: string;
}


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
// turn shipped '<parameter name="status">listed' plus half a JSON array,
// and another shipped 'antml:invoke name="final_answer">' with the leading
// '<' eaten, so the markers must match with or without their brackets
export function looksToolSyntaxLeak(text: string): boolean {
  return /<parameter\s+name=|antml|invoke\s+name=|"dotted_path"\s*:|"entry_codes"\s*:|"conditions_quoted"\s*:|"verbatim_quote"\s*:/.test(
    text,
  );
}

// Last-resort scrubber for the one bounded path that can still ship after an
// escalation round-trip: cut everything from the first leak marker on; if no
// real question survives, fall back to a safe generic one — raw tool syntax
// must never reach a user's screen, whatever the model did.
function stripLeakTail(text: string): string {
  const m = /<parameter\s+name=|antml|invoke\s+name=/.exec(text);
  if (!m) return text;
  const head = text.slice(0, m.index).trim();
  return head.includes("?") ? head : "";
}
const SAFE_FALLBACK_QUESTION =
  "Which additional technical parameter or export fact should I take into account?";

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
  // secret for signing/verifying the "Verdict recorded" acceptance marker —
  // absent in tests and dev, always set in production (VERDICT_HMAC_KEY)
  hmacKey?: string,
): Promise<TurnResult> {
  const transcript = sanitizeMessages(incoming, maxUserTurns);
  await verifyVerdictMarkers(transcript, hmacKey);
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
  // MEASURED, not enforced: a retry on every uncited question would add a
  // model call to most turns while latency is already the worst live defect.
  // The flag lands in the perf log so the rate can be read from `wrangler
  // tail` and the prompt tuned against real numbers instead of a hunch.

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
    if (lastFinalAnswerIndex(transcript) < 0) {
      console.log("question_cited", JSON.stringify({ cited: questionCitesProvision(text) }));
    }
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
    // after an escalation the checks above are one-shot — scrub raw tool
    // syntax AND dead-air (a "question" turn that asks nothing) here, in the
    // returned text and the transcript copy, before shipping
    if (askEscalated && (looksToolSyntaxLeak(text) || !text.includes("?"))) {
      const stripped = looksToolSyntaxLeak(text) ? stripLeakTail(text) : text;
      const safe = stripped.includes("?") ? stripped : SAFE_FALLBACK_QUESTION;
      transcript[transcript.length - 1] = {
        role: "assistant",
        content: [{ type: "text", text: safe }],
      };
      return { type: "question", text: safe, transcript, usd, timings };
    }
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
          missing_facts: [],
          ...(vUse.input as Partial<Verdict>),
        } as Verdict;
        const problems = validateVerdict(verdict, annex);
        transcript.push({ role: "assistant", content: vResp.content as Block[] });
        // needs_expert is premature on the opening message, and equally when
        // the verdict's own text says a user-suppliable parameter is missing —
        // a live card declared "cannot be concluded because the overlay has
        // not been provided" instead of simply asking for the overlay.
        // STRUCTURAL check first: the schema makes the model list the facts
        // the user could still supply. A non-empty list with needs_expert is
        // a contradiction by definition — the verdict names its own missing
        // question. The regex below stays only as a fallback for the prose
        // (a live card said "this fact has not yet been supplied" and slipped
        // past the regex because "fact" was not in its word list — pattern
        // matching on free text can never be the primary guard).
        const missingFacts = (verdict.missing_facts ?? []).map((f) => String(f).trim()).filter(Boolean);
        const missingParam =
          verdict.status === "needs_expert" &&
          /\b(parameter|value|figure|fact|capability|overlay|aperture|endurance|wavelength|specification)\b[^.]{0,80}\bnot (yet |been )*(provided|supplied|stated|given|established|confirmed)|\bnot (yet |been )*(provided|supplied|stated|given|established|confirmed)\b[^.]{0,40}\b(parameter|value|figure|fact)\b/i.test(
            JSON.stringify(verdict),
          );
        if (
          problems.length === 0 &&
          verdict.status === "needs_expert" &&
          (realUserTurns <= 1 || missingFacts.length > 0 || missingParam)
        ) {
          const first = missingFacts[0];
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
                  "instead (rule 2)." +
                  (first ? ` Your own missing_facts names it: ask about "${first}".` : ""),
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
            content: [
              { type: "tool_result", tool_use_id: vUse.id, content: await verdictMarker(hmacKey, vUse) },
            ],
          });
          // ONE INTERVIEW, ONE CARD: a listed verdict flows straight into the
          // licensing stage in the SAME request (rule 11) — unless the user
          // opted out of licensing, or the time budget is already spent (the
          // page then quietly sends the one follow-up turn instead).
          if (verdict.status === "listed" && !classifyOnly() && !outOfTime()) {
            const cont = await continueToPathway();
            // a continuation may fail-close through the forced pathway into a
            // reply that asks NOTHING ("Let me finalize the licensing
            // pathway.") — dead air must not ship as the turn's answer; the
            // verdict ships instead and the page's follow-up re-enters the
            // gated stage-2 flow
            if (cont && !(cont.type === "question" && !cont.text.includes("?"))) return cont;
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
        console.log("verdict rejected:", problems.join("; ").slice(0, 300));
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
      missing_facts: [],
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
      // LATENCY: the live first turn cost ~60 s because the loop model drafted
      // needs_expert on the opening message, the forced (slow, expensive)
      // verdict stage re-drafted the same needs_expert, and only THEN was it
      // bounced into a question. The draft already says it cannot conclude —
      // apply the same premature-needs_expert rule here and skip the forced
      // stage entirely. Genuine needs_expert drafts (after real interviewing,
      // with no user-suppliable fact missing) still go through the card stage.
      const draft = (finalCall.input ?? {}) as Partial<Verdict>;
      const draftMissing = (draft.missing_facts ?? []).map((f) => String(f).trim()).filter(Boolean);
      if (draft.status === "needs_expert" && (realUserTurns <= 1 || draftMissing.length > 0)) {
        transcript.push({ role: "assistant", content: resp.content as Block[] });
        transcript.push({
          role: "user",
          content: uses.map((u) =>
            u === finalCall
              ? {
                  type: "tool_result",
                  tool_use_id: u.id,
                  is_error: true,
                  content:
                    "[system] needs_expert is premature when the user can still supply the " +
                    "missing fact. Ask the single most discriminating technical question " +
                    "instead (rule 2)." +
                    (draftMissing[0] ? ` Your own missing_facts names it: ask about "${draftMissing[0]}".` : ""),
                }
              : {
                  type: "tool_result",
                  tool_use_id: u.id,
                  content: execLookup(annex, String(u.name), (u.input ?? {}) as Record<string, unknown>),
                },
          ),
        });
        return askOneQuestion();
      }
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
