// Loop behaviour with a canned client — CI never touches the network.
import { describe, expect, it } from "vitest";
import type { AnnexDataset } from "../src/annexData";
import { CannedClaudeClient, type ClaudeResponse } from "../src/claudeClient";
import { runTurn, sanitizeMessages, validateVerdict, InvalidRequest } from "../src/loop";
import type { Verdict } from "../src/tools";

const ANNEX: AnnexDataset = {
  corpus_version: "02021R0821-20251115",
  celex: "02021R0821",
  valid_from: "2025-11-15",
  sha256: "abc",
  attribution: "© European Union",
  entry_count: 2,
  index: [
    { code: "3B501", first_line: "3B501 Test equipment as follows:" },
    { code: "4A003", first_line: "4A003 Digital computers as follows:" },
  ],
  entries: [
    {
      entry_code: "3B501",
      category: "3",
      verbatim_text:
        "3B501 Test equipment as follows:\n3B501.f.1.b.1 A light source wavelength equal to or longer than 193 nm;",
      parameters: ["3B501.f.1.b.1 A light source wavelength equal to or longer than 193 nm;"],
      applicable_notes: [],
    },
    {
      entry_code: "4A003",
      category: "4",
      verbatim_text:
        '4A003 Digital computers as follows:\n4A003.b "Digital computers" having an "Adjusted Peak Performance" ("APP") exceeding 70 Weighted TeraFLOPS (WT);',
      parameters: ['4A003.b ... exceeding 70 Weighted TeraFLOPS (WT);'],
      applicable_notes: [],
    },
  ],
  docs: [
    { doc_type: "general_notes", title: "GN", verbatim_text: "General notes text." },
    { doc_type: "definitions_annex", title: "Defs", verbatim_text: '"digital computer" means equipment.' },
  ],
};

const MODELS = { loop: "claude-haiku-4-5", verdict: "claude-sonnet-5" };
const usage = { input_tokens: 100, output_tokens: 50 };

const textResp = (text: string): ClaudeResponse => ({
  content: [{ type: "text", text }],
  stop_reason: "end_turn",
  usage,
});

const toolResp = (name: string, input: unknown, id = "tu_1"): ClaudeResponse => ({
  content: [{ type: "tool_use", id, name, input }],
  stop_reason: "tool_use",
  usage,
});

const GOOD_VERDICT: Verdict = {
  status: "listed",
  entry_codes: ["3B501"],
  reasoning: [
    {
      entry_code: "3B501",
      dotted_path: "3B501.f.1.b.1",
      verbatim_quote: "A light source wavelength equal to or longer than 193 nm;",
      explanation: "The described tool's 193nm source meets the threshold.",
    },
  ],
  caveats: ["Indicative only; Art. 4/5 catch-alls may apply."],
  definitions_used: [],
};

describe("sanitizeMessages", () => {
  it("strips cache_control and rejects bad roles", () => {
    const msgs = sanitizeMessages(
      [{ role: "user", content: [{ type: "text", text: "hi", cache_control: { type: "ephemeral" } }] }],
      10,
    );
    expect((msgs[0].content as { cache_control?: unknown }[])[0].cache_control).toBeUndefined();
    expect(() => sanitizeMessages([{ role: "system", content: "x" }], 10)).toThrow(InvalidRequest);
  });

  it("enforces the turn cap without KV", () => {
    const many = Array.from({ length: 12 }, () => ({ role: "user", content: "q" }));
    expect(() => sanitizeMessages(many, 10)).toThrow("conversation_too_long");
  });
});

describe("validateVerdict", () => {
  it("accepts a grounded verdict", () => {
    expect(validateVerdict(GOOD_VERDICT, ANNEX)).toEqual([]);
  });

  it("rejects invented quotes, unknown codes and empty caveats", () => {
    const bad: Verdict = {
      ...GOOD_VERDICT,
      entry_codes: ["9Z999"],
      reasoning: [{ ...GOOD_VERDICT.reasoning[0], verbatim_quote: "wavelength shorter than 5 nm" }],
      caveats: [],
    };
    const problems = validateVerdict(bad, ANNEX);
    expect(problems.join(" ")).toContain("9Z999");
    expect(problems.join(" ")).toContain("not found in the entry text");
    expect(problems.join(" ")).toContain("caveats");
  });
});

describe("runTurn", () => {
  it("returns a question turn and preserves the transcript", async () => {
    const client = new CannedClaudeClient([textResp("What is the light source wavelength?")]);
    const result = await runTurn(client, ANNEX, [{ role: "user", content: "I make litho tools" }], MODELS, 10);
    expect(result.type).toBe("question");
    expect(result.text).toContain("wavelength");
    expect(result.transcript.at(-1)?.role).toBe("assistant");
    expect(result.usd).toBeGreaterThan(0);
  });

  it("executes lookups server-side then answers", async () => {
    const client = new CannedClaudeClient([
      toolResp("lookup_entries", { codes: ["3B501"] }),
      textResp("Is the wavelength 193 nm or longer?"),
    ]);
    const result = await runTurn(client, ANNEX, [{ role: "user", content: "litho tool" }], MODELS, 10);
    expect(result.type).toBe("question");
    // the second request must carry the tool_result with REAL corpus text
    const second = client.requests[1];
    const lastMsg = JSON.stringify(second.messages.at(-1));
    expect(lastMsg).toContain("193 nm");
  });

  it("verdict flows through the verdict model and validation", async () => {
    const client = new CannedClaudeClient([
      toolResp("final_answer", GOOD_VERDICT),
      toolResp("final_answer", GOOD_VERDICT, "tu_2"),
    ]);
    const result = await runTurn(client, ANNEX, [{ role: "user", content: "193nm litho" }], MODELS, 10);
    expect(result.type).toBe("verdict");
    expect(result.verdict?.corpus_version).toBe("02021R0821-20251115");
    expect(result.verdict?.prompt_sha256).toHaveLength(64);
    // second call (the verdict call) is forced to final_answer on the verdict model
    expect(client.requests[1].model).toBe("claude-sonnet-5");
    expect(JSON.stringify(client.requests[1].tool_choice)).toContain("final_answer");
  });

  it("fails closed when the verdict cites invented text — asks instead", async () => {
    const badVerdict = {
      ...GOOD_VERDICT,
      reasoning: [{ ...GOOD_VERDICT.reasoning[0], verbatim_quote: "totally invented threshold" }],
    };
    const client = new CannedClaudeClient([
      toolResp("final_answer", badVerdict),
      toolResp("final_answer", badVerdict, "tu_2"),
      toolResp("final_answer", badVerdict, "tu_3"),
      textResp("Could you tell me the exact wavelength of the source?"),
    ]);
    const result = await runTurn(client, ANNEX, [{ role: "user", content: "litho" }], MODELS, 10);
    expect(result.type).toBe("question"); // never a verdict with invented quotes
  });
});
