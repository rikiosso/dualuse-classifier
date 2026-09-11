# CLAUDE.md

## What this is

A classifier for technologies under Annex I of Regulation (EU) 2021/821. A Cloudflare
Worker (`worker/`) runs the interview with Claude and validates every verdict against
the corpus before it goes out. A static page (`docs/`) is the client. The corpus
(`annex.json`) comes from Export Controls Watch and updates itself; nothing here parses
or edits it. See `README.md` for the product and `docs/architecture.md` for the request
flow, streaming protocol and cost design.

## How to work here

- Tests: `cd worker && npm test`. Typecheck: `npx tsc --noEmit`. Both must pass before
  any commit. Neither touches the network.
- Every change on its own branch. Every commit message says why, not just what.
- A failure seen live becomes a test, with the date and origin in its name, before it
  gets fixed. See `docs/how-it-was-built.md` for the pattern.
- Propose the plan before touching more than one file. Do not run destructive commands
  without approval.

## Rules that do not get negotiated

- Literal or nothing: no verdict or pathway ships without passing `validateVerdict` /
  `validatePathway`. Never relax a check to make a case pass.
- No regular expression stands in for a structural fix when one exists (a schema field,
  a validation against the corpus) once the failure is understood. `missing_facts` in
  the verdict schema replaced a prose regex for exactly this reason; see
  `docs/how-it-was-built.md`.
- The disclaimer is appended by the Worker, never by the model.
- Model and dataset text is rendered with `textContent` / `createTextNode`, never
  `innerHTML`.
- No new dependency in the Worker without a stated reason.
- Secrets (`ANTHROPIC_API_KEY`, `IP_SALT`, `TESTER_KEY`, `VERDICT_HMAC_KEY`) live only in
  Wrangler, never in a file in this repo.
- Never weaken a guard to make a demo, a test or a deploy pass. Fix the cause.

## Standing decisions (do not reverse without new evidence)

- The in-app monthly gate (`MONTHLY_BUDGET_USD`) sits ABOVE the Anthropic workspace
  spend limit on purpose (operator decision, 26-08-2026): the workspace limit is the
  enforcing ceiling, the KV counter is only the polite throttle. Do not lower it.
- Claude Sonnet 5 runs the interview, not Haiku: an earlier Haiku interview produced
  legally wrong first-turn verdicts.
- Thinking is disabled on every model call.
- One combined card per conversation; a licensing pathway renders into the same card
  as its classification, not a separate one.
- Streaming, the fused licensing continuation, and the classification-only opt-out all
  stay; they were shipped and hardened together, not experiments to strip out.
- No CAPTCHA. The per-IP cap plus the budget gate is the deliberate abuse control.
- Do not create a new repository, bot or secret for this project without being asked.

## Runbook

```bash
cd worker
npm test                          # offline test suite
npx tsc --noEmit
npx wrangler deploy               # also resets the module-scope annex cache
npx wrangler tail --format json   # per-request "perf" line; look for validation rejections
npx wrangler kv key get <key> --namespace-id <id> --remote    # --remote or you read the local simulator
npx wrangler kv key put <key> <value> --namespace-id <id> --remote
```

Bump `?v=N` on every changed file under `docs/` (`index.html`, `app.js`, `style.css`,
`config.js`) so GitHub Pages does not serve a stale cached copy.

## File map

- `worker/src/validate.ts`: the validation contract for verdicts and pathways. Read
  this before touching what a card is allowed to say.
- `worker/src/questionGate.ts`: detectors for a redundant or defective interview
  question, plus the licensing opt-out gate.
- `worker/src/transcript.ts`: transcript sanitisation, the turn cap, HMAC verdict
  markers.
- `worker/src/stages.ts`: the forced verdict and licensing-pathway stages.
- `worker/src/loop.ts`: the conversation loop that ties the stages together.
- `worker/src/prompt.ts`: the system prompt. Its sha256 is stamped into every card.
- `worker/src/tools.ts`: the tool schemas (`lookup_entries`, `lookup_definitions`,
  `lookup_gea`, `final_answer`, `license_pathway`) and the Verdict/Pathway types.
- `worker/src/claudeClient.ts`: the raw Anthropic API client, no SDK.
- `worker/src/annexData.ts`: loads and caches `annex.json`, resolves a dotted path to
  its provision text, checks whether a quote appears in it.
- `worker/src/rateLimit.ts`: the budget gate (per-IP, day, month) and spend estimation.
- `worker/src/index.ts`: routing, CORS, the buffered and streaming handlers.
- `docs/app.js`: the client. No framework.

## What not to do

- Do not add a framework to the client.
- Do not change the default model without measuring latency and the rejection rate
  first.
- Do not mention exports-watch's notes feed, or anything US-specific, in this
  classifier's interface.
- Do not touch `worker/src/loop.ts` guard logic or `worker/src/prompt.ts` beyond a
  specific, tested change: each rule and guard exists because something failed in
  production.
