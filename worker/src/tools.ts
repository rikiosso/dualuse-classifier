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
            met: {
              type: "boolean",
              description:
                "true if the facts SATISFY this provision and support listing under " +
                "this entry; false for a rule-out row explaining why a tested entry " +
                "or cross-reference does NOT apply. Ruled-out entries must not " +
                "appear in entry_codes.",
            },
          },
          required: ["entry_code", "dotted_path", "verbatim_quote", "explanation", "met"],
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
    met?: boolean; // absent on legacy transcripts — treated as supporting
  }[];
  caveats: string[];
  definitions_used: string[];
}

export const LOOKUP_GEA_TOOL = {
  name: "lookup_gea",
  description:
    "Retrieve the FULL verbatim text of Union General Export Authorisations from " +
    "Annex II by id, e.g. ['EU001']. Use 'COMMON_LIST' for the Annex II section I " +
    "common excluded-items list (referenced by EU001/EU003/EU004/EU007). Call this " +
    "before reasoning about any licensing pathway — never rely on memory.",
  input_schema: {
    type: "object",
    properties: {
      ids: { type: "array", items: { type: "string" }, description: "EU001..EU008 or COMMON_LIST" },
    },
    required: ["ids"],
    additionalProperties: false,
  },
} as const;

export const LICENSE_PATHWAY_TOOL = {
  name: "license_pathway",
  description:
    "Deliver the licensing-pathway determination for an already-classified LISTED " +
    "item. Call ONLY after destination and end-use are established. Every " +
    "verbatim_quote must be copied exactly from lookup_gea output.",
  strict: true,
  input_schema: {
    type: "object",
    properties: {
      destination: { type: "string" },
      eligible_gea: { type: "string", description: "EU001..EU008, or empty string if none applies" },
      outcome: {
        type: "string",
        enum: ["gea_available", "individual_licence_required", "sanctions_review_required"],
      },
      conditions_quoted: {
        type: "array",
        items: {
          type: "object",
          properties: {
            gea_id: { type: "string", description: "EU001..EU008 or COMMON_LIST" },
            verbatim_quote: { type: "string" },
            explanation: { type: "string" },
          },
          required: ["gea_id", "verbatim_quote", "explanation"],
          additionalProperties: false,
        },
      },
      caveats: { type: "array", items: { type: "string" } },
    },
    required: ["destination", "eligible_gea", "outcome", "conditions_quoted", "caveats"],
    additionalProperties: false,
  },
} as const;

export interface Pathway {
  destination: string;
  eligible_gea: string;
  outcome: "gea_available" | "individual_licence_required" | "sanctions_review_required";
  conditions_quoted: { gea_id: string; verbatim_quote: string; explanation: string }[];
  caveats: string[];
}
