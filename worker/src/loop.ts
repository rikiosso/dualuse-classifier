// One human turn = one call here. Internally the model may take several tool
// steps (lookups need no human input, so they loop server-side, bounded).
// Convergence is two-stage: the cheap loop model decides WHEN to conclude by
// calling final_answer; the verdict model then writes the authoritative verdict
// under a forced, strict schema — and the Worker validates it against the
// corpus before anyone sees it (a bare or uncited verdict is a bug, enforced
// by code, not prompt).

import type { AnnexDataset } from "./annexData";
import type { ClaudeClient } from "./claudeClient";
import { buildSystemBlocks } from "./prompt";
import {
  FINAL_ANSWER_TOOL,
  LICENSE_PATHWAY_TOOL,
  LOOKUP_DEFINITIONS_TOOL,
  LOOKUP_ENTRIES_TOOL,
  LOOKUP_GEA_TOOL,
  type Verdict,
} from "./tools";
import { estimateUsd } from "./rateLimit";
import {
  questionAsksLicensingFacts,
  questionCitesProvision,
  questionEchoesStatedValue,
  questionNearDuplicate,
  questionOffersEqualAlternatives,
} from "./questionGate";
import {
  InvalidRequest,
  lastFinalAnswerIndex,
  sanitizeMessages,
  verifyVerdictMarkers,
  type Block,
} from "./transcript";
import {
  call,
  classifyOnly,
  execLookup,
  looksPathwayConclusive,
  looksToolSyntaxLeak,
  looksVerdictConclusive,
  LOOP_MAX_TOKENS,
  outOfTime,
  PATHWAY_TOOL_NUDGE,
  realUserTextList,
  recordTiming,
  sysMsg,
  textOf,
  toolUses,
  VERDICT_TOOL_NUDGE,
  withCache,
  type Models,
  type StageTiming,
  type TurnContext,
  type TurnResult,
} from "./turnContext";
import { producePathway, produceVerdict } from "./stages";

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
export { looksToolSyntaxLeak } from "./turnContext";
export type { TurnResult };

const MAX_TOOL_ITERATIONS = 3;

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
  let nudgedBundle = false;
  let askEscalated = false;
  let conclusiveRegen = false;
  // Cloudflare's edge cancels requests around 100s — a forced 4k-token
  // retry on top of a long turn crosses it and the user sees a dead reply.
  // Past this elapsed budget, skip second forced attempts and fail closed
  // (the quick question turn keeps the response comfortably under the limit).
  const startedAt = Date.now();
  const budgetMs = timeBudgetMs ?? 45_000;

  // Everything the forced stages (worker/src/stages.ts — produceVerdict,
  // producePathway, continueToPathway) share with this interview loop.
  // `usd` is accessor-backed over the local variable above, so a stage's
  // spend is literally the same number this function returns; the
  // transcript and timings arrays need no accessor because array mutation
  // (push/pop) is already visible through any reference to the same array.
  // askOneQuestion and shipQuestion go the other way — a stage falling back
  // into the plain interview — so they are wrapped rather than assigned
  // directly: both consts are declared below this point, but neither
  // wrapper is CALLED until the turn is well underway, by which time both
  // exist (same deferred-closure pattern the original mutual recursion
  // between these functions always relied on).
  const ctx: TurnContext = {
    transcript,
    annex,
    client,
    models,
    onStage,
    hmacKey,
    system,
    tools,
    timings,
    realUserTurns,
    startedAt,
    budgetMs,
    get usd() {
      return usd;
    },
    set usd(v: number) {
      usd = v;
    },
    geaInjected: false,
    askOneQuestion: () => askOneQuestion(),
    shipQuestion: (text, retry) => shipQuestion(text, retry),
  };

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
      recordTiming(ctx, "question-judge", "claude-haiku-4-5", tJudge, resp);
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
    const userTexts = realUserTextList(transcript);
    if (lastFinalAnswerIndex(transcript) < 0) {
      console.log("question_cited", JSON.stringify({ cited: questionCitesProvision(text) }));
    }
    const blocked =
      questionEchoesStatedValue(text, userTexts) ||
      questionOffersEqualAlternatives(text) ||
      questionNearDuplicate(text, answeredAssistantQuestions()) ||
      (classifyOnly(transcript) && questionAsksLicensingFacts(text)) ||
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
        return producePathway(ctx);
      }
      transcript.push(sysMsg(VERDICT_TOOL_NUDGE));
      return produceVerdict(ctx);
    }
    return null; // bounded: after nudge + escalation, ship rather than loop
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
    recordTiming(ctx, "ask-fallback", models.loop, tAsk, resp);
    usd += estimateUsd(models.loop, resp.usage);
    const text = textOf(resp);
    transcript.push({ role: "assistant", content: resp.content as Block[] });
    if (!askEscalated && looksToolSyntaxLeak(text)) {
      askEscalated = true;
      if (/license_pathway|"outcome"|"eligible_gea"/.test(text) && lastFinalAnswerIndex(transcript) >= 0) {
        transcript.push(sysMsg(PATHWAY_TOOL_NUDGE));
        return producePathway(ctx);
      }
      transcript.push(sysMsg(VERDICT_TOOL_NUDGE));
      return produceVerdict(ctx);
    }
    if (!askEscalated) {
      if (looksPathwayConclusive(text) && realUserTurns > 1) {
        askEscalated = true;
        if (lastFinalAnswerIndex(transcript) < 0) {
          transcript.push(sysMsg(VERDICT_TOOL_NUDGE));
          return produceVerdict(ctx);
        }
        transcript.push(sysMsg(PATHWAY_TOOL_NUDGE));
        return producePathway(ctx);
      }
      if (looksVerdictConclusive(text)) {
        askEscalated = true;
        transcript.push(sysMsg(VERDICT_TOOL_NUDGE));
        return produceVerdict(ctx);
      }
      // stage-2 convergence applies to the fail-closed path too: the live
      // five-question loop lived entirely inside this fallback, where the
      // main loop's convergence check never runs
      if (answersSinceVerdict() >= 3 && !outOfTime(ctx)) {
        askEscalated = true;
        transcript.push(sysMsg(PATHWAY_TOOL_NUDGE));
        return producePathway(ctx);
      }
      // dead air applies here too: a fallback turn that asks nothing after a
      // verdict strands the user — one more forced attempt with feedback.
      // An entirely EMPTY reply is the extreme case of the same failure.
      if ((!text.includes("?") || !text) && answersSinceVerdict() >= 1 && !outOfTime(ctx)) {
        askEscalated = true;
        transcript.push(sysMsg(PATHWAY_TOOL_NUDGE));
        return producePathway(ctx);
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
    const resp = await call(ctx, models.loop, LOOP_MAX_TOKENS, false);
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
        return produceVerdict(ctx);
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
      return producePathway(ctx);
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
      return produceVerdict(ctx);
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
        return producePathway(ctx);
      }
      transcript.push(sysMsg(VERDICT_TOOL_NUDGE));
      return produceVerdict(ctx);
    }

    if (looksPathwayConclusive(text) && realUserTurns > 1) {
      if (lastFinalAnswerIndex(transcript) < 0) {
        // pathway talk before any validated verdict: classify first
        transcript.push(sysMsg(VERDICT_TOOL_NUDGE));
        return produceVerdict(ctx);
      }
      transcript.push(sysMsg(PATHWAY_TOOL_NUDGE));
      return producePathway(ctx);
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
      return producePathway(ctx);
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
      return produceVerdict(ctx);
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
      return produceVerdict(ctx);
    }
    if (verdictAt >= 0) {
      const answersSince = answersSinceVerdict();
      if (answersSince >= 3) {
        transcript.push(sysMsg(PATHWAY_TOOL_NUDGE));
        return producePathway(ctx);
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
