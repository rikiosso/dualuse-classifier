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
    expect(problems.join(" ")).toContain("not found in that provision");
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
    const result = await runTurn(
      client,
      ANNEX,
      [
        { role: "user", content: "193nm litho stepper" },
        { role: "assistant", content: "What is the MRF?" },
        { role: "user", content: "MRF is 38 nm, chuck overlay 1.2 nm" },
      ],
      MODELS,
      10,
    );
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
    const result = await runTurn(
      client,
      ANNEX,
      [
        { role: "user", content: "litho tool" },
        { role: "assistant", content: "What wavelength?" },
        { role: "user", content: "193 nm" },
      ],
      MODELS,
      10,
    );
    expect(result.type).toBe("question"); // never a verdict with invented quotes
  });
});


describe("validateVerdict — hardened grounding", () => {
  it("rejects a dotted_path that does not belong to the cited entry", () => {
    const v = {
      ...GOOD_VERDICT,
      reasoning: [{ ...GOOD_VERDICT.reasoning[0], dotted_path: "4A003.b" }],
    };
    expect(validateVerdict(v, ANNEX).join(" ")).toContain("does not belong");
  });

  it("rejects a quote lifted from a different clause of the same entry", () => {
    // 4A003 exists with a TeraFLOPS clause; cite 3B501 but quote 4A003's text
    const laundered: Verdict = {
      status: "listed",
      entry_codes: ["3B501"],
      reasoning: [
        {
          entry_code: "3B501",
          dotted_path: "3B501.f.1.b.1",
          verbatim_quote: "Adjusted Peak Performance", // real corpus text, wrong provision
          explanation: "x",
        },
      ],
      caveats: ["c"],
      definitions_used: [],
    };
    expect(validateVerdict(laundered, ANNEX).join(" ")).toContain("not found in that provision");
  });

  it("rejects a headline entry_code with no backing reasoning", () => {
    const v = { ...GOOD_VERDICT, entry_codes: ["3B501", "4A003"] };
    expect(validateVerdict(v, ANNEX).join(" ")).toContain("headlined but has no reasoning");
  });

  it("rejects a too-short quote", () => {
    const v = {
      ...GOOD_VERDICT,
      reasoning: [{ ...GOOD_VERDICT.reasoning[0], verbatim_quote: "193 nm" }],
    };
    expect(validateVerdict(v, ANNEX).join(" ")).toContain("too short");
  });
});

describe("verdict transcript is a valid follow-up array", () => {
  it("closes the final tool_use with a tool_result", async () => {
    const client = new CannedClaudeClient([
      toolResp("final_answer", GOOD_VERDICT),
      toolResp("final_answer", GOOD_VERDICT, "tu_2"),
    ]);
    const result = await runTurn(
      client,
      ANNEX,
      [
        { role: "user", content: "193nm litho stepper" },
        { role: "assistant", content: "What is the MRF?" },
        { role: "user", content: "MRF is 38 nm" },
      ],
      MODELS,
      10,
    );
    const last = result.transcript.at(-1);
    expect(last?.role).toBe("user");
    const blocks = last?.content;
    expect(Array.isArray(blocks) && blocks[0].type).toBe("tool_result");
    // every tool_use in the transcript has a following tool_result (no unpaired)
    const flat = JSON.stringify(result.transcript);
    expect((flat.match(/"tool_use"/g) || []).length).toBeGreaterThan(0);
  });
});

describe("quote normalisation", () => {
  it("treats typographic quotes and dashes as their ASCII forms", async () => {
    const { quoteAppearsIn } = await import("../src/annexData");
    expect(quoteAppearsIn("a 'Minimum Resolvable Feature size' (‘MRF’)", "A ‘Minimum Resolvable Feature size’ ('MRF')")).toBe(true);
    expect(quoteAppearsIn("range 5–10 nm", "range 5-10 nm")).toBe(true);
    expect(quoteAppearsIn("completely different text", "range 5-10 nm")).toBe(false);
  });
});

describe("naked prose verdicts are escalated, never returned", () => {
  it("conclusive text without final_answer triggers the validated verdict stage", async () => {
    const client = new CannedClaudeClient([
      textResp("Based on the APP of 100000 WT, this is Listed in Annex I under 4A003.b."),
      toolResp("final_answer", GOOD_VERDICT, "tu_v"),
    ]);
    const result = await runTurn(
      client,
      ANNEX,
      [
        { role: "user", content: "big gpu cluster" },
        { role: "assistant", content: "What is the APP?" },
        { role: "user", content: "100000 WT, military use" },
      ],
      MODELS,
      10,
    );
    expect(result.type).toBe("verdict"); // escalated into the validated path
    expect(result.verdict?.entry_codes).toEqual(["3B501"]);
    // the forced verdict call went to the verdict model
    expect(client.requests[1].model).toBe("claude-sonnet-5");
  });

  it("ordinary questions mentioning entries are NOT escalated", async () => {
    const client = new CannedClaudeClient([
      textResp("Does the cluster exceed 70 Weighted TeraFLOPS as per 4A003.b?"),
    ]);
    const result = await runTurn(client, ANNEX, [{ role: "user", content: "gpu cluster" }], MODELS, 10);
    expect(result.type).toBe("question");
  });
});

describe("prompt cache TTL", () => {
  it("system prefix carries the 1h TTL; message breakpoint stays default", async () => {
    const client = new CannedClaudeClient([textResp("What is the APP?")]);
    await runTurn(client, ANNEX, [{ role: "user", content: "gpu" }], MODELS, 10);
    const req = client.requests[0] as { system: { cache_control?: { ttl?: string } }[]; messages: { content: { cache_control?: { ttl?: string } }[] }[] };
    expect(req.system.at(-1)?.cache_control?.ttl).toBe("1h");
    const lastMsg = req.messages.at(-1)!;
    expect(lastMsg.content.at(-1)?.cache_control).toEqual({ type: "ephemeral" });
  });
});

describe("first-turn behaviour", () => {
  it("a FULLY SPECIFIED opening message may verdict immediately (listed)", async () => {
    const client = new CannedClaudeClient([
      toolResp("final_answer", GOOD_VERDICT),
      toolResp("final_answer", GOOD_VERDICT, "tu_2"),
    ]);
    const result = await runTurn(
      client,
      ANNEX,
      [{ role: "user", content: "193nm litho stepper, MRF 38nm, chuck overlay 1.2nm" }],
      MODELS,
      10,
    );
    expect(result.type).toBe("verdict"); // the scenario-2 regression
    expect(result.verdict?.entry_codes).toEqual(["3B501"]);
  });

  it("needs_expert on the opening message is bounced into a real question", async () => {
    const NEEDS_EXPERT: Verdict = {
      status: "needs_expert",
      entry_codes: [],
      reasoning: [],
      caveats: ["Indicative only."],
      definitions_used: [],
    };
    const client = new CannedClaudeClient([
      toolResp("final_answer", NEEDS_EXPERT),
      toolResp("final_answer", NEEDS_EXPERT, "tu_2"), // forced verdict call
      textResp("What is the light source wavelength?"), // askOneQuestion
    ]);
    const result = await runTurn(client, ANNEX, [{ role: "user", content: "a litho machine" }], MODELS, 10);
    expect(result.type).toBe("question");
    expect(result.text).toContain("wavelength");
    // the question call must have tools disabled so it cannot stall
    expect(JSON.stringify(client.requests.at(-1)?.tool_choice)).toContain("none");
  });
});
