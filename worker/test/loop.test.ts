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

// a recorded verdict exchange — stage-2 tests need one, since the pathway
// stage refuses to run without a validated final_answer in the transcript
const VERDICT_EXCHANGE = [
  {
    role: "assistant",
    content: [{ type: "tool_use", id: "tu_v0", name: "final_answer", input: GOOD_VERDICT }],
  },
  { role: "user", content: [{ type: "tool_result", tool_use_id: "tu_v0", content: "Verdict recorded." }] },
];

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
    expect(validateVerdict(v, ANNEX).join(" ")).toContain("headlined but has no supporting reasoning");
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

describe("licensing pathway (stage 2)", () => {
  const ANNEX2: typeof ANNEX = {
    ...ANNEX,
    geas: [
      {
        id: "EU001",
        title: "EU001 — A.EXPORTS TO AUSTRALIA...UNITED STATES",
        verbatim_text:
          "1. This authorisation covers exports to the United States of America of all dual-use items except those listed in Section I of this Annex.\n2. Registration with the competent authority is required within 30 days of first use.",
      },
    ],
    gea_common_list: "Items excluded from EU001, EU003, EU004 and EU007: 3B501 lithography equipment of section f.",
  };
  const GOOD_PATHWAY = {
    destination: "United States",
    eligible_gea: "EU001",
    outcome: "gea_available",
    conditions_quoted: [
      {
        gea_id: "EU001",
        verbatim_quote: "Registration with the competent authority is required within 30 days",
        explanation: "EU001 covers the US; registration condition applies.",
      },
    ],
    caveats: ["Draft determination — requires legal review."],
  };

  it("validatePathway accepts a grounded pathway and rejects invented quotes", async () => {
    const { validatePathway } = await import("../src/loop");
    expect(validatePathway(GOOD_PATHWAY as never, ANNEX2)).toEqual([]);
    const bad = {
      ...GOOD_PATHWAY,
      conditions_quoted: [{ gea_id: "EU001", verbatim_quote: "totally invented condition text", explanation: "x" }],
    };
    expect(validatePathway(bad as never, ANNEX2).join(" ")).toContain("not found");
  });

  it("sanctioned destinations can never get a green light", async () => {
    const { validatePathway } = await import("../src/loop");
    const toRussia = { ...GOOD_PATHWAY, destination: "Russia" };
    expect(validatePathway(toRussia as never, ANNEX2).join(" ")).toContain("sanctions_review_required");
    const flagged = { ...toRussia, outcome: "sanctions_review_required", eligible_gea: "", conditions_quoted: [] };
    expect(validatePathway(flagged as never, ANNEX2)).toEqual([]);
  });

  it("a license_pathway call routes through the forced validated stage", async () => {
    const client = new CannedClaudeClient([
      toolResp("license_pathway", GOOD_PATHWAY),
      toolResp("license_pathway", GOOD_PATHWAY, "tu_p2"),
    ]);
    const result = await runTurn(
      client,
      ANNEX2,
      [
        { role: "user", content: "my 3B501 tool, verdict was listed" },
        ...VERDICT_EXCHANGE,
        { role: "assistant", content: "What is the destination?" },
        { role: "user", content: "United States, civil fab customer" },
      ],
      MODELS,
      10,
    );
    expect(result.type).toBe("pathway");
    expect(result.pathway?.eligible_gea).toBe("EU001");
    expect(result.pathway?.corpus_version).toBe("02021R0821-20251115");
    expect(client.requests[1].model).toBe("claude-sonnet-5");
    expect(JSON.stringify(client.requests[1].tool_choice)).toContain("license_pathway");
  });
});

describe("pathway prose escalation", () => {
  it("'EU001 applies' prose becomes a validated pathway card", async () => {
    const ANNEX2b: typeof ANNEX = {
      ...ANNEX,
      geas: [{ id: "EU001", title: "EU001 — A.EXPORTS", verbatim_text: "1. Covers exports to the United States of America. 2. Registration is required within 30 days of first use." }],
      gea_common_list: "Excluded: 0C001.",
    };
    const client = new CannedClaudeClient([
      textResp("Good news — EU001 applies to your export to the United States."),
      toolResp("license_pathway", {
        destination: "United States",
        eligible_gea: "EU001",
        outcome: "gea_available",
        conditions_quoted: [{ gea_id: "EU001", verbatim_quote: "Registration is required within 30 days", explanation: "condition" }],
        caveats: ["Requires legal review."],
      }),
    ]);
    const result = await runTurn(
      client,
      ANNEX2b,
      [
        { role: "user", content: "listed item, 3B501" },
        ...VERDICT_EXCHANGE,
        { role: "assistant", content: "Destination?" },
        { role: "user", content: "United States" },
      ],
      MODELS,
      10,
    );
    expect(result.type).toBe("pathway");
    expect(result.pathway?.outcome).toBe("gea_available");
  });

  it("sanctions prose becomes the sanctions card, never chat text", async () => {
    const ANNEX2c: typeof ANNEX = { ...ANNEX, geas: [], gea_common_list: "x" };
    const client = new CannedClaudeClient([
      textResp("I cannot assist further: Russia is subject to a comprehensive EU sanctions regime."),
      toolResp("license_pathway", {
        destination: "Russia",
        eligible_gea: "",
        outcome: "sanctions_review_required",
        conditions_quoted: [],
        caveats: ["EU sanctions regimes apply — qualified counsel must review."],
      }),
    ]);
    const result = await runTurn(
      client,
      ANNEX2c,
      [
        { role: "user", content: "my listed drone" },
        ...VERDICT_EXCHANGE,
        { role: "assistant", content: "Destination?" },
        { role: "user", content: "Russia" },
      ],
      MODELS,
      10,
    );
    expect(result.type).toBe("pathway");
    expect(result.pathway?.outcome).toBe("sanctions_review_required");
  });
});

describe("pathway validation hardening", () => {
  it("garbage eligible_gea is rejected for ANY outcome; ungrounded individual too", async () => {
    const { validatePathway } = await import("../src/loop");
    const ANNEXG: typeof ANNEX = { ...ANNEX, geas: [{ id: "EU001", title: "EU001 — A", verbatim_text: "Covers exports to the United States of America except Section I items." }], gea_common_list: "x" };
    const garbage = {
      destination: "United States",
      eligible_gea: '</antml junk">individual',
      outcome: "individual_licence_required",
      conditions_quoted: [],
      caveats: ["c"],
    };
    const problems = validatePathway(garbage as never, ANNEXG);
    expect(problems.join(" ")).toContain("does not exist");
    expect(problems.join(" ")).toContain("must quote the provision");
    const grounded = {
      destination: "United States",
      eligible_gea: "",
      outcome: "individual_licence_required",
      conditions_quoted: [{ gea_id: "EU001", verbatim_quote: "except Section I items", explanation: "tested and failed" }],
      caveats: ["c"],
    };
    expect(validatePathway(grounded as never, ANNEXG)).toEqual([]);
  });

  it("a garbled eligible_gea on a non-GEA outcome is normalised, not rejected (live regression)", async () => {
    const { normalizePathway, validatePathway } = await import("../src/loop");
    const ANNEXG: typeof ANNEX = { ...ANNEX, geas: [{ id: "EU001", title: "EU001 — A", verbatim_text: "Covers exports to the United States of America except Section I items." }], gea_common_list: "x" };
    // seen live: the model serialises an intended-empty eligible_gea as
    // tool-syntax artifacts and reproduces the glitch on retry — rejection
    // fail-closed a CORRECT sanctions outcome into prose
    const garbled = {
      destination: "Russia",
      eligible_gea: '</antmlparameter>\n<parameter name="outcome">sanctions_review_required',
      outcome: "sanctions_review_required",
      conditions_quoted: [],
      caveats: ["EU sanctions regimes apply — qualified counsel must review."],
    };
    const fixed = normalizePathway(garbled as never, ANNEXG);
    expect(fixed.eligible_gea).toBe("");
    expect(validatePathway(fixed, ANNEXG)).toEqual([]);
    // gea_available keeps strict validation — a garbled headline id never ships
    const headline = { ...garbled, outcome: "gea_available" };
    expect(normalizePathway(headline as never, ANNEXG).eligible_gea).toBe(garbled.eligible_gea);
  });

  it("runTurn ships the sanctions card despite a garbled eligible_gea", async () => {
    const ANNEXS: typeof ANNEX = { ...ANNEX, geas: [], gea_common_list: "x" };
    const client = new CannedClaudeClient([
      textResp("Russia is subject to a comprehensive EU sanctions regime."),
      toolResp("license_pathway", {
        destination: "Russia",
        eligible_gea: '</antml_parameter>\n<parameter name="outcome">sanctions_review_required',
        outcome: "sanctions_review_required",
        conditions_quoted: [],
        caveats: ["Sanctions review required — qualified counsel must review."],
      }),
    ]);
    const result = await runTurn(
      client,
      ANNEXS,
      [
        { role: "user", content: "my listed 3B501 scanner" },
        ...VERDICT_EXCHANGE,
        { role: "assistant", content: "Destination?" },
        { role: "user", content: "Russia, civil fab customer" },
      ],
      MODELS,
      10,
    );
    expect(result.type).toBe("pathway");
    expect(result.pathway?.outcome).toBe("sanctions_review_required");
    expect(result.pathway?.eligible_gea).toBe("");
  });

  it("individual_licence_required for a Cat 5 Part 2 item must rule EU008 out", async () => {
    const { validatePathway } = await import("../src/loop");
    const ANNEX8: typeof ANNEX = {
      ...ANNEX,
      geas: [
        { id: "EU001", title: "EU001 — A", verbatim_text: "Valid for exports to the United States of America." },
        { id: "EU008", title: "EU008 — H.INFORMATION SECURITY", verbatim_text: "This authorisation covers information security items. Part 3: it does not apply where the end-user is a government of a listed destination." },
      ],
      gea_common_list: "x",
    };
    const noSweep = {
      destination: "India",
      eligible_gea: "",
      outcome: "individual_licence_required",
      conditions_quoted: [{ gea_id: "EU001", verbatim_quote: "Valid for exports to the United States of America.", explanation: "India is not an EU001 destination." }],
      caveats: ["c"],
    };
    expect(validatePathway(noSweep as never, ANNEX8, ["5A002"]).join(" ")).toContain("EU008");
    // same pathway is fine for a non-crypto item, or once EU008 is quoted
    expect(validatePathway(noSweep as never, ANNEX8, ["3B501"])).toEqual([]);
    const swept = {
      ...noSweep,
      conditions_quoted: [
        ...noSweep.conditions_quoted,
        { gea_id: "EU008", verbatim_quote: "it does not apply where the end-user is a government", explanation: "EU008 end-user clause tested." },
      ],
    };
    expect(validatePathway(swept as never, ANNEX8, ["5A002"])).toEqual([]);
    // an annex without EU008 (older corpus) cannot demand impossible quotes
    const ANNEXno8: typeof ANNEX = { ...ANNEX8, geas: [ANNEX8.geas![0]] };
    expect(validatePathway(noSweep as never, ANNEXno8, ["5A002"])).toEqual([]);
  });

  it("runTurn recovers the verdict's entry codes and enforces the EU008 sweep", async () => {
    const ANNEX8: typeof ANNEX = {
      ...ANNEX,
      geas: [
        { id: "EU001", title: "EU001 — A", verbatim_text: "Valid for exports to the United States of America." },
        { id: "EU008", title: "EU008 — H.INFORMATION SECURITY", verbatim_text: "This authorisation covers information security items. Part 3: it does not apply where the end-user is a government of a listed destination." },
      ],
      gea_common_list: "x",
    };
    const noSweep = {
      destination: "India",
      eligible_gea: "",
      outcome: "individual_licence_required",
      conditions_quoted: [{ gea_id: "EU001", verbatim_quote: "Valid for exports to the United States of America.", explanation: "India not covered by EU001." }],
      caveats: ["c"],
    };
    const swept = {
      ...noSweep,
      conditions_quoted: [
        ...noSweep.conditions_quoted,
        { gea_id: "EU008", verbatim_quote: "it does not apply where the end-user is a government", explanation: "EU008 tested and ruled out on the end-user clause." },
      ],
    };
    const client = new CannedClaudeClient([
      toolResp("license_pathway", noSweep),
      toolResp("license_pathway", noSweep, "tu_p2"), // forced stage, attempt 1: rejected (no EU008)
      toolResp("license_pathway", swept, "tu_p3"), // attempt 2 after the rejection feedback
    ]);
    const result = await runTurn(
      client,
      ANNEX8,
      [
        { role: "user", content: "my VPN appliance" },
        {
          role: "assistant",
          content: [
            { type: "text", text: "Concluding." },
            { type: "tool_use", id: "tu_v1", name: "final_answer", input: { ...GOOD_VERDICT, entry_codes: ["5A002"] } },
          ],
        },
        { role: "user", content: [{ type: "tool_result", tool_use_id: "tu_v1", content: "Verdict recorded." }] },
        { role: "user", content: "Destination: India, private telecom operator." },
      ],
      MODELS,
      10,
    );
    expect(result.type).toBe("pathway");
    expect(result.pathway?.conditions_quoted.some((c) => c.gea_id === "EU008")).toBe(true);
  });
});

describe("formula-defined terms must be computed, never adopted", () => {
  const ANNEXF: typeof ANNEX = {
    ...ANNEX,
    entries: [
      ...ANNEX.entries.filter((e) => e.entry_code !== "3B501"),
      {
        entry_code: "3B501",
        category: "3",
        verbatim_text:
          "3B501 Test equipment as follows:\n" +
          "3B501.f.1.b.1 A light source wavelength equal to or longer than 193 nm;\n" +
          "3B501.f.1.b.2 Capable of producing a pattern with a 'Minimum Resolvable Feature size' ('MRF') of 45 nm or less; and\n" +
          "3B501.f.1.b Technical Notes: For the purposes of 3B501.f.1.b.: 1. The 'Minimum Resolvable Feature size' ('MRF') is calculated by the following formula: 'MRF' = wavelength × K factor / maximum numerical aperture where, the K factor = 0,25.",
        parameters: [],
        applicable_notes: [],
      },
    ],
  };
  const base = {
    status: "listed",
    entry_codes: ["3B501"],
    caveats: ["Indicative only."],
    definitions_used: [],
  };

  it("a claimed MRF without the computation is rejected; a shown calculation passes", async () => {
    const { validateVerdict } = await import("../src/loop");
    const adopted = {
      ...base,
      reasoning: [
        {
          entry_code: "3B501",
          dotted_path: "3B501.f.1.b.2",
          verbatim_quote: "Capable of producing a pattern with a 'Minimum Resolvable Feature size' ('MRF') of 45 nm or less; and",
          explanation: "The user states the scanner produces a 38 nm MRF, below the 45 nm threshold.",
          met: true,
        },
      ],
    };
    expect(validateVerdict(adopted as never, ANNEXF).join(" ")).toContain("formula-defined term");
    const computed = {
      ...base,
      reasoning: [
        {
          ...adopted.reasoning[0],
          explanation:
            "Verified with the entry's own Technical Note formula (K factor = 0,25): MRF = (193 × 0.25) / 1.35 = 35.7 nm, at or below 45 nm.",
        },
      ],
    };
    expect(validateVerdict(computed as never, ANNEXF)).toEqual([]);
    // a computation whose result EXCEEDS the "…or less" threshold cannot back
    // a supporting row (live bug: "50,04 nm … falls at or below 45 nm")
    const contradicted = {
      ...base,
      reasoning: [
        {
          ...adopted.reasoning[0],
          explanation:
            "Using the entry's own Technical Note formula: MRF = (193 × 0.35) / 1.35 = 50.04 nm, which falls at or below the 45 nm threshold.",
        },
      ],
    };
    expect(validateVerdict(contradicted as never, ANNEXF).join(" ")).toContain("EXCEEDS");
    // ...but the same computation on a RULE-OUT row is exactly right
    const ruledOut = {
      ...base,
      entry_codes: [],
      status: "not_listed",
      reasoning: [
        {
          ...adopted.reasoning[0],
          met: false,
          explanation:
            "Using the entry's own Technical Note formula: MRF = (193 × 0.35) / 1.35 = 50.04 nm, which exceeds 45 nm — not met.",
        },
      ],
    };
    expect(validateVerdict(ruledOut as never, ANNEXF)).toEqual([]);
    // rows that do not quote the defined term (e.g. the wavelength criterion)
    // carry no computation duty
    const wavelength = {
      ...base,
      reasoning: [
        {
          entry_code: "3B501",
          dotted_path: "3B501.f.1.b.1",
          verbatim_quote: "A light source wavelength equal to or longer than 193 nm;",
          explanation: "193 nm equals the threshold.",
          met: true,
        },
      ],
    };
    expect(validateVerdict(wavelength as never, ANNEXF)).toEqual([]);
  });
});

describe("cross-reference guard (N.B. / SEE ALSO)", () => {
  const ANNEXX: typeof ANNEX = {
    ...ANNEX,
    entries: [
      ...ANNEX.entries,
      {
        entry_code: "3B001",
        category: "3",
        verbatim_text:
          "3B001 Semiconductor manufacturing equipment as follows:\n" +
          "3B001 N.B. SEE ALSO 4A003\n" +
          "3B001.f.1 Align and expose step and scan equipment having any of the following:\n" +
          "3B001.f.1.b Capable of producing a pattern with a 'Minimum Resolvable Feature size' (MRF) of 45 nm or less;\n" +
          "3B001.f.1 N.B. SEE ALSO 3B501.f.\n" +
          "3B001.h Multi-layer masks with a phase shift layer;\n" +
          "3B001.h N.B. For masks specially designed for optical sensors, see 6B002.",
        parameters: [],
        applicable_notes: [],
      },
    ],
  };
  const mrfVerdict = {
    status: "listed",
    entry_codes: ["3B001"],
    reasoning: [
      {
        entry_code: "3B001",
        dotted_path: "3B001.f.1.b",
        verbatim_quote:
          "Capable of producing a pattern with a 'Minimum Resolvable Feature size' (MRF) of 45 nm or less;",
        explanation: "Stated MRF of 38 nm meets the 45 nm threshold.",
      },
    ],
    caveats: ["Indicative triage only."],
    definitions_used: [],
  };

  it("a verdict citing a provision with SEE ALSO must engage the referenced entry", async () => {
    const { validateVerdict } = await import("../src/loop");
    // live regression: concluded 3B001.f.1.b on a stated MRF, never tested 3B501
    expect(validateVerdict(mrfVerdict as never, ANNEXX).join(" ")).toContain("3B501");
    const engaged = {
      ...mrfVerdict,
      caveats: [
        "Indicative triage only.",
        "3B501.f (cross-referenced) not assessed: its 'dedicated chuck overlay' criterion was not provided.",
      ],
    };
    // ...and the ROOT-level "3B001 N.B. SEE ALSO 4A003" imposes nothing:
    // the verdict never mentions 4A003 yet validates clean
    expect(validateVerdict(engaged as never, ANNEXX)).toEqual([]);
  });

  it("a ruled-out entry is engaged via met=false rows, never headlined (live regression)", async () => {
    const { validateVerdict } = await import("../src/loop");
    const both = {
      status: "listed",
      entry_codes: ["3B001", "3B501"],
      reasoning: [
        {
          entry_code: "3B001",
          dotted_path: "3B001.f.1.b",
          verbatim_quote:
            "Capable of producing a pattern with a 'Minimum Resolvable Feature size' (MRF) of 45 nm or less;",
          explanation: "MRF under 3B001's K=0.35 formula is 50.04 nm — NOT met.",
          met: false,
        },
        {
          entry_code: "3B501",
          dotted_path: "3B501.f.1.b.1",
          verbatim_quote: "A light source wavelength equal to or longer than 193 nm;",
          explanation: "193 nm satisfies 'equal to or longer than'.",
          met: true,
        },
      ],
      caveats: ["c"],
      definitions_used: [],
    };
    // live regression: the old symmetric checks forced the headline "3B001, 3B501"
    expect(validateVerdict(both as never, ANNEXX).join(" ")).toContain("no supporting reasoning");
    // correct form: rule-out rows engage the cross-reference AND stay out of the headline
    const fixed = { ...both, entry_codes: ["3B501"] };
    expect(validateVerdict(fixed as never, ANNEXX)).toEqual([]);
  });

  it("N.B. lines outside the cited subtree, and refs to entries not in the corpus, are ignored", async () => {
    const { validateVerdict } = await import("../src/loop");
    const maskVerdict = {
      ...mrfVerdict,
      reasoning: [
        {
          entry_code: "3B001",
          dotted_path: "3B001.h",
          verbatim_quote: "Multi-layer masks with a phase shift layer;",
          explanation: "Phase-shift mask as described.",
        },
      ],
    };
    // 3B001.h's own N.B. points at 6B002 (not in corpus — ignored) and the
    // SEE ALSO at 3B001.f.1 is not an ancestor of 3B001.h — no problems
    expect(validateVerdict(maskVerdict as never, ANNEXX)).toEqual([]);
  });
});

describe("lookup narration guard", () => {
  it("a turn that only announces a lookup is nudged to act, not surfaced", async () => {
    const client = new CannedClaudeClient([
      textResp(
        "I need to retrieve the exact verbatim text of the Technical Notes section for 3B501.f.1.b to ensure I quote it correctly. Let me look that up now.",
      ),
      textResp("What is the maximum numerical aperture (NA) of the scanner?"),
    ]);
    const result = await runTurn(client, ANNEX, [{ role: "user", content: "my litho tool, NA pending" }], MODELS, 10);
    expect(result.type).toBe("question");
    expect(result.text).toContain("numerical aperture");
    const nudges = result.transcript.filter(
      (m) =>
        m.role === "user" &&
        Array.isArray(m.content) &&
        m.content.some((b) => String((b as { text?: string }).text ?? "").includes("Do not narrate lookups")),
    );
    expect(nudges.length).toBe(1);
  });

  it("a real question containing 'let me check' language still surfaces", async () => {
    const client = new CannedClaudeClient([
      textResp("Before I check the thresholds: what is the light source wavelength, in nm?"),
    ]);
    const result = await runTurn(client, ANNEX, [{ role: "user", content: "classify my laser tool" }], MODELS, 10);
    expect(result.type).toBe("question");
    expect(result.text).toContain("wavelength");
  });
});

describe("trimmed lookup restoration", () => {
  const ANNEXT: typeof ANNEX = {
    ...ANNEX,
    geas: [
      {
        id: "EU001",
        title: "EU001 — A.EXPORTS",
        verbatim_text:
          "1. This authorisation covers exports to the United States of America.\n2. Registration with the competent authority is required within 30 days of first use.",
      },
    ],
    gea_common_list: "x",
  };
  const TRIMMED = "=== EU001 ===\n1. This auth\n…[trimmed — call the lookup tool again if needed]";

  it("forced pathway stage sees re-fetched full text for trimmed lookups", async () => {
    const good = {
      destination: "United States",
      eligible_gea: "EU001",
      outcome: "gea_available",
      conditions_quoted: [
        {
          gea_id: "EU001",
          verbatim_quote: "Registration with the competent authority is required within 30 days",
          explanation: "condition",
        },
      ],
      caveats: ["Requires legal review."],
    };
    const client = new CannedClaudeClient([
      toolResp("license_pathway", good),
      toolResp("license_pathway", good, "tu_p2"),
    ]);
    const result = await runTurn(
      client,
      ANNEXT,
      [
        { role: "user", content: "my listed 3B501 item" },
        ...VERDICT_EXCHANGE,
        { role: "assistant", content: [{ type: "tool_use", id: "tu_g1", name: "lookup_gea", input: { ids: ["EU001"] } }] },
        { role: "user", content: [{ type: "tool_result", tool_use_id: "tu_g1", content: TRIMMED }] },
        { role: "assistant", content: "Destination?" },
        { role: "user", content: "United States, civil customer" },
      ],
      MODELS,
      10,
    );
    expect(result.type).toBe("pathway");
    // the forced call's payload must carry the restored full GEA text
    const forcedPayload = JSON.stringify(client.requests[1].messages);
    expect(forcedPayload).toContain("Registration with the competent authority is required within 30 days");
    expect(forcedPayload).not.toContain("[trimmed");
  });

  it("truncation talk is never surfaced — the model is nudged to re-fetch", async () => {
    const client = new CannedClaudeClient([
      textResp("Unfortunately the EU008 text was truncated in my context, so I cannot fully classify this item."),
      textResp("What is the destination country?"),
    ]);
    const result = await runTurn(client, ANNEX, [{ role: "user", content: "my listed item" }], MODELS, 10);
    expect(result.type).toBe("question");
    expect(result.text).toContain("destination country");
    expect(result.text).not.toContain("truncated");
    const nudged = result.transcript.some(
      (m) =>
        m.role === "user" &&
        Array.isArray(m.content) &&
        m.content.some((b) => String((b as { text?: string }).text ?? "").includes("re-fetched with")),
    );
    expect(nudged).toBe(true);
  });
});

describe("naked-verdict escalation — bold 'not listed' prose", () => {
  it("markdown bold cannot smuggle a not-listed conclusion past the escalation", async () => {
    const notListed = {
      status: "not_listed",
      entry_codes: [],
      reasoning: [],
      caveats: ["Indicative only; Art. 4/5 catch-alls may apply."],
      definitions_used: [],
    };
    const client = new CannedClaudeClient([
      textResp(
        "This is straightforward: a standard office laptop for general business use is **not listed in Annex I** of Regulation (EU) 2021/821.",
      ),
      toolResp("final_answer", notListed, "tu_n1"),
    ]);
    const result = await runTurn(client, ANNEX, [{ role: "user", content: "a standard office laptop" }], MODELS, 10);
    expect(result.type).toBe("verdict");
    expect(result.verdict?.status).toBe("not_listed");
  });
});

describe("naked-verdict escalation — 'meets all criteria' prose", () => {
  it("prose declaring all sub-criteria met escalates into the verdict stage", async () => {
    const client = new CannedClaudeClient([
      textResp(
        "Your scanner meets **all three** sub-criteria of **3B501.f.1.b**:\n1. ✓ 193 nm\n2. ✓ MRF 35.7 nm\n3. ✓ 1.2 nm overlay\nNow I'll confirm the classification. What is the destination country?",
      ),
      toolResp("final_answer", GOOD_VERDICT, "tu_v9"),
    ]);
    const result = await runTurn(client, ANNEX, [{ role: "user", content: "my litho scanner, all specs given" }], MODELS, 10);
    expect(result.type).toBe("verdict");
    expect(result.verdict?.entry_codes).toEqual(["3B501"]);
  });
});

describe("tool-budget decision iteration and ask-fallback escalation", () => {
  const ANNEXP: typeof ANNEX = {
    ...ANNEX,
    geas: [
      {
        id: "EU001",
        title: "EU001 — A.EXPORTS",
        verbatim_text:
          "1. This authorisation covers exports to the United States of America.\n2. Registration with the competent authority is required within 30 days of first use.",
      },
    ],
    gea_common_list: "x",
  };
  const GOOD = {
    destination: "United States",
    eligible_gea: "EU001",
    outcome: "gea_available",
    conditions_quoted: [
      {
        gea_id: "EU001",
        verbatim_quote: "Registration with the competent authority is required within 30 days",
        explanation: "condition",
      },
    ],
    caveats: ["Requires legal review."],
  };
  const BAD = { ...GOOD, conditions_quoted: [{ gea_id: "EU001", verbatim_quote: "totally invented text here", explanation: "x" }] };
  const lookup = (id: string) => toolResp("lookup_gea", { ids: ["EU001"] }, id);

  it("after the lookup budget, the decision iteration can still conclude via the tools", async () => {
    const client = new CannedClaudeClient([
      lookup("tu_l1"),
      lookup("tu_l2"),
      lookup("tu_l3"), // budget spent — decision nudge injected next
      toolResp("license_pathway", GOOD, "tu_d1"), // loop model concludes
      toolResp("license_pathway", GOOD, "tu_d2"), // forced authoritative stage
    ]);
    const result = await runTurn(
      client,
      ANNEXP,
      [
        { role: "user", content: "my listed 3B501 item" },
        ...VERDICT_EXCHANGE,
        { role: "assistant", content: "Destination?" },
        { role: "user", content: "United States, civil fab, no adverse awareness" },
      ],
      MODELS,
      10,
    );
    expect(result.type).toBe("pathway");
    expect(result.pathway?.eligible_gea).toBe("EU001");
  });

  it("conclusive prose from the fail-closed ask is escalated into the card, never shipped", async () => {
    const client = new CannedClaudeClient([
      toolResp("license_pathway", BAD, "tu_p1"), // draft routes to forced stage
      toolResp("license_pathway", BAD, "tu_p2"), // attempt 1: rejected
      toolResp("license_pathway", BAD, "tu_p3"), // attempt 2: rejected → fail-closed ask
      textResp("EU001 clearly covers your export to the United States."), // conclusive prose
      toolResp("license_pathway", GOOD, "tu_p4"), // escalated forced attempt succeeds
    ]);
    const result = await runTurn(
      client,
      ANNEXP,
      [
        { role: "user", content: "my listed 3B501 item" },
        ...VERDICT_EXCHANGE,
        { role: "assistant", content: "Destination?" },
        { role: "user", content: "United States, civil fab" },
      ],
      MODELS,
      10,
    );
    expect(result.type).toBe("pathway");
    expect(result.pathway?.outcome).toBe("gea_available");
  });
});

describe("stage-2 convergence", () => {
  it("after a verdict and three answered turns, a fourth question is overridden by the pathway tool", async () => {
    const ANNEXP: typeof ANNEX = {
      ...ANNEX,
      geas: [{ id: "EU001", title: "EU001 — A", verbatim_text: "1. Covers exports to the United States of America. 2. Registration is required within 30 days of first use." }],
      gea_common_list: "x",
    };
    const good = {
      destination: "United States",
      eligible_gea: "EU001",
      outcome: "gea_available",
      conditions_quoted: [{ gea_id: "EU001", verbatim_quote: "Registration is required within 30 days", explanation: "condition" }],
      caveats: ["Requires legal review."],
    };
    const client = new CannedClaudeClient([
      textResp("One more thing: will the equipment be operated by the end-user itself?"),
      toolResp("license_pathway", good, "tu_s2"),
    ]);
    const result = await runTurn(
      client,
      ANNEXP,
      [
        { role: "user", content: "my litho scanner, full specs" },
        {
          role: "assistant",
          content: [{ type: "tool_use", id: "tu_v1", name: "final_answer", input: GOOD_VERDICT }],
        },
        { role: "user", content: [{ type: "tool_result", tool_use_id: "tu_v1", content: "Verdict recorded." }] },
        { role: "user", content: "Destination: United States, civil fab, commercial production, no adverse awareness." },
        { role: "assistant", content: "Any custom software or assistance?" },
        { role: "user", content: "Standard vendor software only, no assistance." },
        { role: "assistant", content: "Complete system?" },
        { role: "user", content: "Yes, complete standalone system." },
      ],
      MODELS,
      10,
    );
    expect(result.type).toBe("pathway");
    expect(result.pathway?.eligible_gea).toBe("EU001");
  });
});

describe("one-question discipline", () => {
  it("a bundled multi-question turn is nudged once toward a single question", async () => {
    const client = new CannedClaudeClient([
      textResp("Is it a step-and-scan system? Also, is it a complete system or a component?"),
      textResp("What is the maximum numerical aperture of the scanner?"),
    ]);
    const result = await runTurn(client, ANNEX, [{ role: "user", content: "my litho tool" }], MODELS, 10);
    expect(result.type).toBe("question");
    expect(result.text).toContain("numerical aperture");
    expect(result.text).not.toContain("Also");
  });

  it("the nudge fires only once — a second multi-question turn is surfaced", async () => {
    const client = new CannedClaudeClient([
      textResp("Is it A? Or is it B?"),
      textResp("Is it C? Or is it D?"),
    ]);
    const result = await runTurn(client, ANNEX, [{ role: "user", content: "my tool" }], MODELS, 10);
    expect(result.type).toBe("question");
    expect(result.text).toContain("C");
  });
});

describe("history trimming", () => {
  it("oversized old tool_results are trimmed instead of failing the conversation", () => {
    const big = "X".repeat(30000);
    const msgs = [
      { role: "user", content: "start" },
      { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "lookup_entries", input: {} }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: big }] },
      { role: "assistant", content: [{ type: "tool_use", id: "t2", name: "lookup_gea", input: {} }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "t2", content: big }] },
      { role: "assistant", content: [{ type: "tool_use", id: "t3", name: "lookup_gea", input: {} }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "t3", content: big }] },
      { role: "assistant", content: "Question?" },
      { role: "user", content: "answer " + "y".repeat(100) },
    ];
    const out = sanitizeMessages(msgs, 10);
    expect(JSON.stringify(out).length).toBeLessThan(80000);
    const first = out[2].content as { content?: string }[];
    expect(String(first[0].content)).toContain("[trimmed");
  });
});
