// Tool definitions for the classification loop. lookup_* are read-only against
// the public dataset; final_answer is strict-schema so a verdict can only ever
// arrive fully formed (additionalProperties: false at every level).

export const LOOKUP_ENTRIES_TOOL = {
  name: "lookup_entries",
  description:
    "Retrieve the FULL verbatim text (all sub-items, parameters and notes) of " +
    "specific Annex I entries by their 4-character codes, e.g. ['3A001', '3B001']. " +
    "Call this before quoting or reasoning about any entry — never rely on memory.",
  input_schema: {
    type: "object",
    properties: {
      codes: {
        type: "array",
        items: { type: "string" },
        description: "Entry codes such as 3A001 (max 6 per call)",
      },
    },
    required: ["codes"],
    additionalProperties: false,
  },
} as const;

export const LOOKUP_DEFINITIONS_TOOL = {
  name: "lookup_definitions",
  description:
    "Retrieve verbatim definitions of quoted Annex I terms (terms in double " +
    "quotation marks in entry text are defined terms), e.g. ['digital computer', " +
    "'basic scientific research']. Use whenever a defined term is load-bearing.",
  input_schema: {
    type: "object",
    properties: {
      terms: { type: "array", items: { type: "string" }, description: "Terms to define" },
    },
    required: ["terms"],
    additionalProperties: false,
  },
} as const;

export const FINAL_ANSWER_TOOL = {
  name: "final_answer",
  description:
    "Deliver the final classification verdict. Call ONLY when the technical facts " +
    "gathered from the user are sufficient to conclude, or when concluding that " +
    "expert review is required. Every verbatim_quote must be copied exactly from " +
    "lookup_entries output.",
  strict: true,
  input_schema: {
    type: "object",
    properties: {
      status: { type: "string", enum: ["listed", "not_listed", "needs_expert"] },
      entry_codes: { type: "array", items: { type: "string" } },
      reasoning: {
        type: "array",
        items: {
          type: "object",
          properties: {
            entry_code: { type: "string" },
            dotted_path: { type: "string" },
            verbatim_quote: { type: "string" },
            explanation: { type: "string" },
          },
          required: ["entry_code", "dotted_path", "verbatim_quote", "explanation"],
          additionalProperties: false,
        },
      },
      caveats: { type: "array", items: { type: "string" } },
      definitions_used: { type: "array", items: { type: "string" } },
    },
    required: ["status", "entry_codes", "reasoning", "caveats", "definitions_used"],
    additionalProperties: false,
  },
} as const;

export interface Verdict {
  status: "listed" | "not_listed" | "needs_expert";
  entry_codes: string[];
  reasoning: {
    entry_code: string;
    dotted_path: string;
    verbatim_quote: string;
    explanation: string;
  }[];
  caveats: string[];
  definitions_used: string[];
}
