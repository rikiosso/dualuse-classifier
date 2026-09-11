# Architecture

How a classification works, the request flow, the streaming protocol, and the design
choices behind the cost model. See the [README](../README.md) for the short version and
the diagram.

## How a classification works

1. The model sees a cached prompt with the index of all 384 entries plus the general
   notes and Articles 2/4/5, never its training memory of the regulation.
2. It narrows candidates and fetches full verbatim entry text through a read-only
   `lookup_entries` tool (definitions via `lookup_definitions`).
3. It interviews you, one discriminating technical question per turn. The prompt asks
   it to quote the threshold it is testing with its dotted path; the server logs whether
   each question actually does (`question_cited` in `wrangler tail`), so the rate is
   measured rather than promised.
4. Claude Sonnet 5 runs the interview and writes the verdicts under a strict JSON schema.
   Every candidate question passes a question gate before it ships: deterministic
   detectors block questions that echo a value you already stated, offer alternatives
   that are the same number, or near-duplicate a question you already answered, and a
   cheap judge model vetoes anything else that is already answered or could not change
   the outcome. A blocked question becomes either a better question or a conclusion.
5. The server validates every verdict against the corpus before you see it: every cited
   entry code must exist, every dotted path must belong to its entry, and every
   "verbatim quote" must actually appear in the specific provision named by that dotted
   path, not merely somewhere in the multi-page entry, which blocks a threshold or
   comparator lifted from a neighbouring clause. Reasoning rows carry a `met` flag, so
   tested-and-ruled-out entries appear on the card without being headlined; where a
   Technical Note defines a term by formula (for example "MRF"), the explanation must
   show the calculation and its result must agree with the claimed outcome; N.B./SEE
   ALSO cross-references on the cited provision must be engaged; a not-listed verdict
   must show the candidates it tested and ruled out; and conclusions can only ever reach
   you as validated cards. Prose verdicts, raw tool syntax, empty replies and dead-air
   turns are all intercepted and escalated by code. A verdict that fails any check is
   rejected and corrected, or the assistant asks instead. No unverifiable classification
   ever ships (the offline test suite pins all of this).
6. The response streams live progress: you watch it consult Annex I, read the cited
   entries and draft the card stage by stage, instead of staring at a spinner.

## Stage 2: the licensing pathway

A classification is only half the journey. After a Listed verdict, the assistant keeps
going in the same breath: it asks for the destination and end-use (one continuous
interview, no separate stage to trigger), retrieves the EU General Export Authorisations
(EU001 to EU008, Annex II of the same Regulation, also auto-updated by the corpus) and
determines the pathway: GEA available (conditions quoted verbatim), individual
authorisation required (quoting the provision that rules the GEAs out), or, for
destinations under an EU sanctions regime, sanctions review required, which the tool
flags loudly and refuses to resolve. Sanctions law is out of scope by design, and the
server rejects any pathway that would green-light a sanctioned destination. Every
pathway is a draft determination that a licensing authority or qualified counsel must
confirm before reliance. The licensing stage cannot run without a validated
classification first, must sweep every GEA whose item scope could reach the entry
(EU008 for any Category 5 Part 2 item), must quote conditions from the authorisation it
grants, and renders into the same card as the classification: one ask, one
determination. Only need the classification? Say so ("just classify it, I don't need
the licence") and the destination questions stop; the classification card ships alone.

## Request flow (one POST = one human turn)

1. **Gate.** A handful of sequential KV reads and writes check the per-IP and global
   budget, then reserve a small fixed amount against the day's spend before the model
   runs.
2. **Corpus.** `annex.json` is read from a module-scope cache (a cold fetch is rare; the
   cache survives across requests on a warm Worker instance).
3. **Turn.** The transcript is sanitised and its verdict-acceptance markers verified;
   the interview model runs, tool results execute server-side, and this repeats for up
   to a few lookup rounds. The loop model signals it is ready to conclude by calling
   `final_answer`, but that draft is discarded: a forced `final_answer` call on the
   verdict model, under the strict schema, produces the card that is actually validated
   (with one retry on failure, carrying the validation feedback). On a Listed verdict,
   the same request continues into the licensing pathway: lookups execute, a genuine
   `license_pathway` call goes to the forced pathway stage (with the full Annex II
   injected once per conversation), a real follow-up question ships through the same
   gates if one is needed, and anything else rolls back rather than fabricate an
   answer. A successful pathway re-attaches the already-validated verdict so the page
   renders one combined card.
4. **Reconcile.** Spend is reconciled against the reservation and the result returns.

## Streaming protocol (NDJSON)

Request: `POST /api/chat`, header `accept: application/x-ndjson` (omit it to get the
older buffered JSON response instead). Body: `{"messages": [...]}`, where `messages` is
the opaque, server-owned transcript returned by the previous turn, with the new
`{"role":"user","content":"..."}` appended.

Response: one `{"type":"progress","stage":"..."}` line per model call start (the stage
names are `interview`, `card:final_answer`, `card:license_pathway`, `ask-fallback`),
followed by exactly one `{"type":"result","data":{...}}` line. A gate failure (rate
limited, daily budget exhausted, bad request) returns plain JSON instead of a stream. A
mid-turn failure rides the final line as `data.type = "error"`.

The result envelope carries a `type` (`question`, `verdict`, `pathway` or `error`), the
assistant `text`, the replacement `messages` transcript to resend verbatim next turn,
and, where relevant, a `verdict` object (status, entry codes, reasoning rows with their
dotted paths and quotes, caveats, corpus and prompt provenance, disclaimer) and a
`pathway` object (destination, outcome, eligible GEA, quoted conditions, caveats,
provenance, disclaimer). `continue_licensing` is set only when a Listed verdict shipped
alone because the turn ran out of time; the page then sends one silent follow-up so the
visitor still ends with a single card. Error reasons include `bad_json`, `bad_request`,
`conversation_too_long`, `forbidden_origin`, `not_found`, `rate_limited`,
`daily_budget_exhausted` and `upstream_error`.

## Models, caching and time budgets

The interview and both card stages run on the same model (Claude Sonnet 5), which lets
them share one prompt cache; an early Haiku-driven interview produced legally wrong
first-turn verdicts, so Sonnet took over that role. A separate, much cheaper Haiku model
judges only whether a candidate question is redundant. Thinking is disabled on every
call: Sonnet's thinking blocks broke text extraction and, once, permanently bricked a
client-held transcript, so the transcript sanitiser also drops any thinking block it
receives regardless of the setting.

Prompt caching keeps the bulk of the system prompt (the rule contract, the entry index,
the general notes and Articles, and the tool schemas) cached across the calls in a
conversation, so a full interview costs a small fraction of what re-sending that context
on every call would. A public demo can therefore serve a meaningful number of full
conversations within a modest monthly key limit; everyone beyond that gets Browse mode,
a fully client-side search of the same dataset that costs nothing and never goes down.

Two layers keep the demo affordable, doing different jobs. The hard ceiling is a
dedicated API key's monthly spend limit, set in the Anthropic console: Anthropic
enforces it server-side and atomically, so total spend cannot exceed it no matter what a
burst of traffic does. In-app throttles (a per-visitor daily conversation cap, global
day and month spend counters in Workers KV, a tight per-turn token bound, and a
conservative reserve-before-spend) keep normal usage far below that ceiling, but they
are best-effort, since Workers KV is only eventually consistent. The key-level limit is
the real backstop, not these counters. A Durable Object would make the in-app counter
strictly atomic; that is a documented follow-up, not something the current demo needs to
be safe.

## Honesty guarantees, in code

- Verbatim-or-nothing: every quote is validated against the exact cited provision
  server-side (`worker/src/validate.ts`, `validateVerdict`), fail-closed.
- Every verdict carries the `corpus_version` it was made against and the sha256 of the
  system prompt, for provenance.
- The disclaimer is appended by the Worker, not the model: it cannot be talked out of
  it.
- Model text is rendered with `textContent`, never `innerHTML`: no markup injection.
