// The shared machinery between the plain interview loop (loop.ts) and the
// forced, validated stages (stages.ts): the per-turn TurnContext both sides
// read and mutate, the model-call wrapper that spends against it, and the
// small pure helpers (output classifiers, tool execution, the "conclude via
// a tool, never in prose" nudges) both sides need to interpret a model
// response the same way.
//
// This module exists so loop.ts and stages.ts don't import runtime values
// from each other: loop.ts's interview loop and stages.ts's three forced
// stages are mutually recursive (a stage falls back to a question, the loop
// escalates into a stage), but everything they have in common lives here
// instead, one level down. Both of the higher files import from here; this
// file imports from neither of them, so there is no cycle.
//
// TurnContext itself is what carries a turn's state across that boundary:
// fields runTurn never reassigns (annex, client, models, the system/tools
// blocks) are plain references; the transcript and timings arrays are
// naturally shared because array mutation (push/pop) is visible through any
// reference to the same array; `usd` is the one shared primitive that isn't
// reference-shared, so runTurn backs it with a get/set pair over its own
// local variable, and `ctx.usd += x` here changes the exact number runTurn
// later returns; askOneQuestion/shipQuestion are callback fields runTurn
// assigns once those closures exist, letting a stage in stages.ts fall back
// into the interview loop without ever importing loop.ts.

import type { AnnexDataset } from "./annexData";
import { definitionsFor, entryByCode, geaScopeText } from "./annexData";
import type { ClaudeClient, ClaudeResponse } from "./claudeClient";
import type { Pathway, Verdict } from "./tools";
import { estimateUsd } from "./rateLimit";
import { wantsClassificationOnly } from "./questionGate";
import type { Block, Msg } from "./transcript";

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

// Every field the interview loop and the forced stages share. runTurn builds
// exactly one of these per turn; see the header comment above for how each
// field stays in sync with runTurn's own state.
export interface TurnContext {
  readonly transcript: Msg[];
  readonly annex: AnnexDataset;
  readonly client: ClaudeClient;
  readonly models: Models;
  readonly onStage?: (stage: string) => void;
  readonly hmacKey?: string;
  readonly system: Block[];
  readonly tools: unknown[];
  readonly timings: StageTiming[];
  readonly realUserTurns: number;
  readonly startedAt: number;
  readonly budgetMs: number;
  // accessor-backed by runTurn over its own local `usd` — see header comment
  usd: number;
  // one-shot flag for stages.ts's ensureGeaContext, private to the pathway stage
  geaInjected: boolean;
  // the two ways a stage falls back into the plain interview loop, both of
  // which live in loop.ts; assigned once those closures exist
  askOneQuestion: () => Promise<TurnResult>;
  shipQuestion: (text: string, retry: () => Promise<TurnResult>) => Promise<TurnResult | null>;
}

export const LOOP_MAX_TOKENS = 900;

export function withCache(blocks: Block[] | string, ttl?: "1h"): Block[] {
  const arr = typeof blocks === "string" ? [{ type: "text", text: blocks } as Block] : [...blocks];
  if (arr.length > 0) {
    const cc = ttl ? { type: "ephemeral", ttl } : { type: "ephemeral" };
    arr[arr.length - 1] = { ...arr[arr.length - 1], cache_control: cc };
  }
  return arr;
}

export function textOf(resp: ClaudeResponse): string {
  return resp.content
    .filter((b) => b.type === "text")
    .map((b) => String((b as { text?: string }).text ?? ""))
    .join("\n")
    .trim();
}

export function toolUses(resp: ClaudeResponse): Block[] {
  return resp.content.filter((b) => b.type === "tool_use");
}

export function execLookup(annex: AnnexDataset, name: string, input: Record<string, unknown>): string {
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

// Conclusive-prose detectors, shared by the interview loop and the
// ask-fallback: conclusions must reach the user ONLY as validated cards,
// never as chat text.
export function looksPathwayConclusive(text: string): boolean {
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

export function looksVerdictConclusive(text: string): boolean {
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

export const PATHWAY_TOOL_NUDGE =
  "[system] Licensing conclusions must be delivered ONLY through the " +
  "license_pathway tool, never as prose. Call license_pathway now with the " +
  "destination, outcome, exact verbatim quotes from lookup_gea and full caveats.";
export const VERDICT_TOOL_NUDGE =
  "[system] Conclusions must be delivered ONLY through the final_answer tool, " +
  "never as prose. Call final_answer now with complete reasoning, exact " +
  "verbatim quotes and full caveats.";

export function sysMsg(text: string): Msg {
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

// Cloudflare's edge cancels requests around 100s — a forced 4k-token retry on
// top of a long turn crosses it and the user sees a dead reply. Past this
// elapsed budget, skip second forced attempts and fail closed (the quick
// question turn keeps the response comfortably under the limit).
export function outOfTime(ctx: TurnContext): boolean {
  return Date.now() - ctx.startedAt > ctx.budgetMs;
}

// pure — takes the transcript directly, so both loop.ts and stages.ts can
// call it without routing through the context object
export function realUserTextList(transcript: Msg[]): string[] {
  return transcript
    .filter((m) => m.role === "user" && Array.isArray(m.content))
    .map((m) =>
      (m.content as Block[])
        .filter((b) => b.type === "text")
        .map((b) => String((b as { text?: string }).text ?? ""))
        .join("\n"),
    )
    .filter((t) => t && !t.startsWith("[system]"));
}

export function classifyOnly(transcript: Msg[]): boolean {
  return wantsClassificationOnly(realUserTextList(transcript));
}

export function recordTiming(
  ctx: TurnContext,
  stage: string,
  model: string,
  t0: number,
  resp: ClaudeResponse,
): void {
  ctx.timings.push({
    stage,
    model,
    ms: Date.now() - t0,
    in: resp.usage.input_tokens,
    out: resp.usage.output_tokens,
    cache_read: resp.usage.cache_read_input_tokens ?? 0,
    cache_write: resp.usage.cache_creation_input_tokens ?? 0,
  });
}

export async function call(
  ctx: TurnContext,
  model: string,
  maxTokens: number,
  forced: false | string,
): Promise<ClaudeResponse> {
  const msgs = ctx.transcript.map((m, i) =>
    i === ctx.transcript.length - 1 ? { ...m, content: withCache(m.content) } : m,
  );
  const t0 = Date.now();
  ctx.onStage?.(forced ? `card:${forced}` : "interview");
  const resp = await ctx.client.complete({
    model,
    max_tokens: maxTokens,
    system: ctx.system,
    messages: msgs,
    tools: ctx.tools,
    // models with reasoning enabled by default (Sonnet 5) emit thinking
    // blocks that break textOf and poison the client-held transcript — this
    // pipeline's structured discipline needs plain responses
    thinking: { type: "disabled" },
    // disable_parallel_tool_use: a forced response carrying TWO parallel
    // final_answer blocks would leave an unpaired sibling tool_use and 400
    // the continuation (or the next turn), discarding a validated verdict
    ...(forced
      ? { tool_choice: { type: "tool", name: forced, disable_parallel_tool_use: true } }
      : {}),
  });
  recordTiming(ctx, forced ? `card:${forced}` : "interview", model, t0, resp);
  ctx.usd += estimateUsd(model, resp.usage);
  return resp;
}
