// Node 22.18 or newer is required to run TypeScript directly (type stripping).
//
// Demetrio's P2 probe: does the Anthropic Messages API's strict tool mode
// accept a property typed {"type": ["boolean", "null"]} (a tri-value: true /
// false / "not yet assessed")? worker/src/tools.ts's final_answer.met is
// today a plain boolean, which cannot represent "not assessed" structurally
// — a nullable-boolean property would let it. This script answers the
// question with ONE minimal live call before anything is changed on that
// assumption.
//
// Mirrors the exact wire format worker/src/claudeClient.ts and
// worker/src/tools.ts use: POST https://api.anthropic.com/v1/messages,
// headers x-api-key / anthropic-version: 2023-06-01 / content-type (no beta
// header — claudeClient.ts sends none), a tool object with strict:true
// alongside input_schema (not nested under it), forced via tool_choice.
//
// Never called by tests. Without ANTHROPIC_API_KEY it prints usage and exits
// 1 with no network call.

import process from "node:process";

const USAGE = `Usage: npm run probe-schema

Makes one minimal Anthropic Messages API call with a strict tool whose input
schema has a property {"type": ["boolean", "null"]}, and prints whether the
API accepted it.

Requires ANTHROPIC_API_KEY in the environment. Costs a small amount of real
API spend (max_tokens: 64, one call) against that key's budget.`;

const PROBE_TOOL = {
  name: "probe_tool",
  description: "Report one field, used only to probe strict tool-schema support.",
  strict: true,
  input_schema: {
    type: "object",
    properties: {
      met: {
        type: ["boolean", "null"],
        description: "true, false, or null when not yet assessed.",
      },
    },
    required: ["met"],
    additionalProperties: false,
  },
} as const;

async function main(): Promise<void> {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    console.log(USAGE);
    process.exit(1);
    return;
  }

  const body = {
    model: "claude-sonnet-5",
    max_tokens: 64,
    system: "Call probe_tool with met set to null.",
    messages: [{ role: "user", content: "Call the tool now." }],
    tools: [PROBE_TOOL],
    tool_choice: { type: "tool", name: "probe_tool" },
  };

  const resp = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
      "content-type": "application/json",
    },
    body: JSON.stringify(body),
  });

  const text = await resp.text();
  if (!resp.ok) {
    console.log(`REJECTED: HTTP ${resp.status}`);
    console.log(text);
    process.exit(1);
    return;
  }

  let parsed: { content?: { type: string; input?: unknown }[] };
  try {
    parsed = JSON.parse(text);
  } catch {
    console.log("ACCEPTED (HTTP 200) but the response body was not valid JSON:");
    console.log(text);
    return;
  }
  const toolUse = parsed.content?.find((b) => b.type === "tool_use");
  console.log("ACCEPTED");
  console.log(JSON.stringify(toolUse?.input ?? null, null, 2));
}

main().catch((err) => {
  console.error(err instanceof Error ? err.stack ?? err.message : err);
  process.exit(1);
});
