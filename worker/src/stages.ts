// The three forced, validated conversation stages that runTurn hands off to
// once the interview concludes: produceVerdict (final_answer), producePathway
// (license_pathway) and continueToPathway (the fused, same-request handoff
// from a listed verdict into licensing).
//
// Split out of loop.ts (second cut, Demetrio's P1) once runTurn crossed
// 1,200 lines. Every value a stage needs from runTurn's own interview loop
// (askOneQuestion, shipQuestion — both still in loop.ts) travels through the
// TurnContext defined in turnContext.ts, never through a direct import of
// loop.ts: the stages and the interview loop are mutually recursive (a
// stage that fails validation falls back to a question, and the interview
// loop escalates into a stage the moment the model tries to conclude), and
// two modules importing runtime values from each other would be a real
// circular dependency. See turnContext.ts's header for the full account of
// how TurnContext keeps state in sync across that boundary. This file
// imports only from turnContext.ts and the plain data/validation modules;
// loop.ts imports produceVerdict/producePathway/continueToPathway from here.
import type { AnnexDataset } from "./annexData";
import { geaScopeText } from "./annexData";
import { promptSha256 } from "./prompt";
import type { Pathway, Verdict } from "./tools";
import { normalizePathway, validatePathway, validateVerdict } from "./validate";
import {
  lastFinalAnswerIndex,
  verdictCodesIn,
  verdictMarker,
  type Block,
  type Msg,
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
  sysMsg,
  textOf,
  toolUses,
  type TurnContext,
  type TurnResult,
} from "./turnContext";

const VERDICT_MAX_TOKENS = 2800;
const STAGE2_CONTINUE_NUDGE =
  "[system] Verdict recorded. Continue straight into the licensing stage " +
  "(rule 11): if the destination, end-use and end-user are already stated, " +
  "retrieve the relevant authorisations with lookup_gea and call " +
  "license_pathway; otherwise ask the single most important licensing " +
  "question (destination first).";

// a 4k-token forced card alone takes ~60-80s to generate — affordable at the
// start of a turn, fatal after slow interview pre-steps. Slow turns get a
// tighter card budget; validation fail-closes if it truncates.
function cardBudget(ctx: TurnContext): number {
  return Date.now() - ctx.startedAt > ctx.budgetMs * 0.45 ? 2400 : VERDICT_MAX_TOKENS;
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

// The forced pathway stage cannot fetch, so it must never be starved of
// quotable text: inject the FULL Annex II corpus as a synthetic lookup
// exchange once per turn. A live run looped five near-identical questions
// because every forced card was rejected for unquotable GEA text.
function ensureGeaContext(ctx: TurnContext): void {
  if (ctx.geaInjected) return;
  // the transcript is replayed every turn — a previous turn's injection
  // persists, and re-injecting would grow tokens linearly per turn
  if (
    ctx.transcript.some(
      (m) =>
        Array.isArray(m.content) &&
        m.content.some((b) => b.type === "tool_use" && String(b.id ?? "").startsWith("srv_gea_")),
    )
  ) {
    ctx.geaInjected = true;
    return;
  }
  ctx.geaInjected = true;
  const ids = ["EU001", "EU002", "EU003", "EU004", "EU005", "EU006", "EU007", "EU008", "COMMON_LIST"];
  const texts = ids
    .map((id) => {
      const t = geaScopeText(ctx.annex, id);
      return t ? `=== ${id} ===\n${t}` : `No GEA ${id} in this corpus version.`;
    })
    .join("\n\n");
  const useId = `srv_gea_${ctx.transcript.length}`;
  ctx.transcript.push({
    role: "assistant",
    content: [{ type: "tool_use", id: useId, name: "lookup_gea", input: { ids } }],
  });
  ctx.transcript.push({
    role: "user",
    content: [{ type: "tool_result", tool_use_id: useId, content: texts }],
  });
}

// The single-card flow delivers classification and pathway together, so the
// pathway result re-attaches the verdict recorded earlier in this
// conversation. The transcript is client-held and untrusted: the recovered
// verdict is re-validated against the corpus before it is echoed back, and a
// forged one is simply dropped (the pathway card then stands alone).
function recordedVerdict(
  ctx: TurnContext,
): (Verdict & { corpus_version: string; corpus_sha256: string }) | undefined {
  const at = lastFinalAnswerIndex(ctx.transcript);
  if (at < 0) return undefined;
  const use = (ctx.transcript[at].content as Block[]).find(
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
  if (validateVerdict(v, ctx.annex).length > 0) return undefined;
  return { ...v, corpus_version: ctx.annex.corpus_version, corpus_sha256: ctx.annex.sha256 };
}

// The verdict stage: forced strict final_answer on the stronger model, with
// one retry on validation failure; fail-closed to a question otherwise.
export async function produceVerdict(ctx: TurnContext): Promise<TurnResult> {
  restoreTrimmedLookups(ctx.transcript, ctx.annex);
  for (let attempt = 0; attempt < 2; attempt++) {
    if (attempt > 0 && outOfTime(ctx)) break;
    const vResp = await call(ctx, ctx.models.verdict, cardBudget(ctx), "final_answer");
    const vUse = toolUses(vResp).find((u) => u.name === "final_answer");
    if (!vUse) break;
    // the API does not hard-enforce required fields on tool inputs — a live
    // call omitted an array and the validator crashed on .length. Missing
    // fields become validation problems, never TypeErrors.
    const verdict = {
      status: "needs_expert",
      entry_codes: [],
      reasoning: [],
      caveats: [],
      definitions_used: [],
      missing_facts: [],
      ...(vUse.input as Partial<Verdict>),
    } as Verdict;
    const problems = validateVerdict(verdict, ctx.annex);
    ctx.transcript.push({ role: "assistant", content: vResp.content as Block[] });
    // needs_expert is premature on the opening message, and equally when the
    // verdict's own text says a user-suppliable parameter is missing — a
    // live card declared "cannot be concluded because the overlay has not
    // been provided" instead of simply asking for the overlay. STRUCTURAL
    // check first: the schema makes the model list the facts the user could
    // still supply. A non-empty list with needs_expert is a contradiction by
    // definition — the verdict names its own missing question. The regex
    // below stays only as a fallback for the prose (a live card said "this
    // fact has not yet been supplied" and slipped past the regex because
    // "fact" was not in its word list — pattern matching on free text can
    // never be the primary guard).
    const missingFacts = (verdict.missing_facts ?? []).map((f) => String(f).trim()).filter(Boolean);
    const missingParam =
      verdict.status === "needs_expert" &&
      /\b(parameter|value|figure|fact|capability|overlay|aperture|endurance|wavelength|specification)\b[^.]{0,80}\bnot (yet |been )*(provided|supplied|stated|given|established|confirmed)|\bnot (yet |been )*(provided|supplied|stated|given|established|confirmed)\b[^.]{0,40}\b(parameter|value|figure|fact)\b/i.test(
        JSON.stringify(verdict),
      );
    if (
      problems.length === 0 &&
      verdict.status === "needs_expert" &&
      (ctx.realUserTurns <= 1 || missingFacts.length > 0 || missingParam)
    ) {
      const first = missingFacts[0];
      ctx.transcript.push({
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
      return ctx.askOneQuestion();
    }
    if (problems.length === 0) {
      // close the tool_use so the returned transcript is a valid Anthropic
      // array — a follow-up turn would otherwise 400 on an unpaired tool_use
      ctx.transcript.push({
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: vUse.id, content: await verdictMarker(ctx.hmacKey, vUse) },
        ],
      });
      // ONE INTERVIEW, ONE CARD: a listed verdict flows straight into the
      // licensing stage in the SAME request (rule 11) — unless the user
      // opted out of licensing, or the time budget is already spent (the
      // page then quietly sends the one follow-up turn instead).
      if (verdict.status === "listed" && !classifyOnly(ctx.transcript) && !outOfTime(ctx)) {
        const cont = await continueToPathway(ctx);
        // a continuation may fail-close through the forced pathway into a
        // reply that asks NOTHING ("Let me finalize the licensing pathway.")
        // — dead air must not ship as the turn's answer; the verdict ships
        // instead and the page's follow-up re-enters the gated stage-2 flow
        if (cont && !(cont.type === "question" && !cont.text.includes("?"))) return cont;
      }
      return {
        type: "verdict",
        text: textOf(vResp),
        transcript: ctx.transcript,
        verdict: {
          ...verdict,
          corpus_version: ctx.annex.corpus_version,
          corpus_sha256: ctx.annex.sha256,
          prompt_sha256: await promptSha256(),
        },
        usd: ctx.usd,
        timings: ctx.timings,
        ...(verdict.status === "listed" && !classifyOnly(ctx.transcript) ? { continueLicensing: true } : {}),
      };
    }
    ctx.transcript.push({
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
  ctx.transcript.push({
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
  // the guarded fallback carries every question/conclusion protection — this
  // exit used to run raw with tools enabled and no guards at all
  return ctx.askOneQuestion();
}

// Stage-2 twin of produceVerdict: forced strict license_pathway, validated,
// one retry, fail-closed to a question.
export async function producePathway(ctx: TurnContext): Promise<TurnResult> {
  // single chokepoint for the opt-out: every escalation route lands here, so
  // an opted-out user can never receive a pathway determination — whatever
  // prose or convergence rule tried to force one
  if (classifyOnly(ctx.transcript)) {
    ctx.transcript.push(
      sysMsg(
        "[system] The user asked for the classification only — do not determine or " +
          "discuss a licensing pathway. Answer their question or ask what else they " +
          "need about the classification.",
      ),
    );
    return ctx.askOneQuestion();
  }
  restoreTrimmedLookups(ctx.transcript, ctx.annex);
  ensureGeaContext(ctx);
  for (let attempt = 0; attempt < 2; attempt++) {
    if (attempt > 0 && outOfTime(ctx)) break;
    const pResp = await call(ctx, ctx.models.verdict, cardBudget(ctx), "license_pathway");
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
      ctx.annex,
    );
    const problems = validatePathway(pathway, ctx.annex, verdictCodesIn(ctx.transcript));
    ctx.transcript.push({ role: "assistant", content: pResp.content as Block[] });
    if (problems.length === 0) {
      ctx.transcript.push({
        role: "user",
        content: [{ type: "tool_result", tool_use_id: pUse.id, content: "Pathway recorded." }],
      });
      const sha = await promptSha256();
      const rv = recordedVerdict(ctx);
      return {
        type: "pathway",
        text: textOf(pResp),
        transcript: ctx.transcript,
        ...(rv ? { verdict: { ...rv, prompt_sha256: sha } } : {}),
        pathway: {
          ...pathway,
          corpus_version: ctx.annex.corpus_version,
          corpus_sha256: ctx.annex.sha256,
          prompt_sha256: sha,
        },
        usd: ctx.usd,
        timings: ctx.timings,
      };
    }
    console.log("pathway rejected:", problems.join("; ").slice(0, 300));
    ctx.transcript.push({
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
  ctx.transcript.push({
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
  return ctx.askOneQuestion();
}

// The in-request licensing continuation: after a listed verdict records, let
// the loop model take up to two more steps toward license_pathway — lookups
// execute, a genuine licensing question ships through the same gates, and a
// genuine license_pathway call proceeds to the forced validated stage.
// Anything else — dead air, narration, conclusive prose, leaked tool syntax —
// is ROLLED BACK, never escalated: with zero post-verdict user input a forced
// pathway would have to fabricate the destination (the schema requires one),
// and a fabricated destination can even mask a sanctioned one. Returns null
// in that case (and on time/steps running out); the verdict then ships alone
// with continueLicensing set, and the page's follow-up turn re-enters the
// fully-gated stage-2 flow.
export async function continueToPathway(ctx: TurnContext): Promise<TurnResult | null> {
  ctx.transcript.push(sysMsg(STAGE2_CONTINUE_NUDGE));
  for (let k = 0; k < 2; k++) {
    if (outOfTime(ctx)) return null;
    const resp = await call(ctx, ctx.models.loop, LOOP_MAX_TOKENS, false);
    const uses = toolUses(resp);
    const pathwayCall = uses.find((u) => u.name === "license_pathway");
    ctx.transcript.push({ role: "assistant", content: resp.content as Block[] });
    if (uses.length > 0) {
      ctx.transcript.push({
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
                content: execLookup(ctx.annex, String(u.name), (u.input ?? {}) as Record<string, unknown>),
              },
        ),
      });
      if (pathwayCall) return producePathway(ctx);
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
      ctx.transcript.pop(); // the reply never happened — the verdict ships clean
      return null;
    }
    const escalated = await ctx.shipQuestion(text, () => ctx.askOneQuestion());
    if (escalated) return escalated;
    return { type: "question", text, transcript: ctx.transcript, usd: ctx.usd, timings: ctx.timings };
  }
  return null;
}
