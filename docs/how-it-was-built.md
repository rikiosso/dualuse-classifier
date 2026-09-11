# How it was built

I am a lawyer, not an engineer. I built this with Claude Code, and most of what made it
trustworthy came from watching it fail live and turning each failure into a guard and a
test. This is the dated account, in the order things happened.

## Mid-August: the interview model was wrong about the law

The first version used Haiku for the interview. It produced legally wrong verdicts on
the first turn: confident, well-formatted, and citing the wrong provision. There was no
single bug to fix, the model itself was not reliable enough for the interview role. I
switched the interview and both card stages to Claude Sonnet 5, which also let them
share one prompt cache instead of paying for context twice. This is a standing decision,
recorded in the repo, not something later sessions should revert without measuring the
cost and rejection rate first.

## Mid-August: thinking blocks broke transcripts

Sonnet 5 emits thinking blocks by default. They broke text extraction, and once they
permanently bricked a transcript held client-side, since the client re-sends the whole
history on every turn and a broken block in it could not be recovered. The fix is
`thinking: {type: "disabled"}` on every call, and the transcript sanitiser drops any
thinking block it receives anyway, in case one arrives despite the setting. Pinned by
`worker/test/loop.test.ts`, "drops thinking blocks and thinking-only messages instead of
failing".

## Late August to 8 September: the regex whack-a-mole, and the fix

Early on, every new failure mode became a new regex: a pattern matched against the
model's prose to catch a bad verdict after the fact. It worked until the model phrased
the same problem differently. The clearest case: a "needs expert" verdict was supposed
to be rejected when the model's own explanation named a fact the user could still
supply. The regex looked for phrasings such as "parameter not provided". Live on
8 September 2026, a drone classification shipped `needs_expert` because "this fact has
not yet been supplied", a plain missing fact, and the regex missed it because "fact" was
not in its word list. Every new phrasing was a new hole.

An external reviewer looked at the codebase on 8 September 2026 and pushed for a
structural fix instead of another pattern: add `missing_facts` to the verdict schema
itself, so the model has to list what it is missing, and reject any `needs_expert`
verdict whose own list is non-empty. That closes the failure mode regardless of how the
explanation is worded, because it no longer depends on wording at all. Pinned by
`worker/test/loop.test.ts`, describe block "needs_expert that lists its own
missing_facts (live drone case, 2026-09-08)".

## 2 September: the corpus mangled formulas

A classification under 3B001 or 3B501 depends on a formula-defined term, the MRF
calculation in their Technical Notes. The watcher's Formex-to-text parser was linearising fractions incorrectly,
so a formula like "K × wavelength / numerical aperture" arrived at the model as
run-together text with no operators, and the model read it wrong. This was not a
classifier bug: it was upstream, in how the source XML got turned into the one-line
verbatim text the classifier trusts. The parser fix (in the separate Export Controls
Watch pipeline) linearises fractions as `(dividend) / (divisor)` and forces bracketed
expressions to keep their parentheses. The corpus was force-rebuilt and republished the
same day. On the classifier side, the rule that a formula-defined term's explanation
must show the computation, and that the computed value must agree with the claimed
outcome, is pinned by `worker/test/loop.test.ts`, "a claimed MRF without the computation
is rejected; a shown calculation passes".

## 1 September: a zero-credit lockout produced phantom spend

The Anthropic org credit hit zero. The API refused every call with a credit-balance
error, mapped that day to a generic 502, and every refused attempt still left its
reservation on the day's spend counter, so the app's own budget gate then also refused
requests that had never touched the model at all, worsening the outage it caused. Two
changes came out of it: the credit-balance error is now mapped specifically to the
budget-exhausted response so the page shows its polite banner instead of a raw error,
and pre-model failures refund their reservation instead of keeping it. Pinned by
`worker/test/handler.test.ts`, "maps Anthropic's credit-balance refusal to the
budget-exhausted banner, not a raw 502".

## 2 September: tool-call syntax leaked to the user

A live run shipped a message containing the raw text of a tool invocation, the kind of
bracketed syntax the model uses to call a tool, because a partial or malformed call was
not recognised as a leak and escalated. The detector for this pattern needed to catch
both the bracketed form and a bracketless variant seen in a later run. Pinned by
`worker/test/loop.test.ts`, describe block "leak variants and the post-escalation
scrubber", including "detects bracketless antml/invoke leak syntax".

## 1 September: forged-transcript risk

Because the client re-sends its whole conversation history on every turn, nothing
stopped a modified transcript from claiming a verdict had already been accepted, which
would let the licensing stage run without ever validating a classification. The fix is
an HMAC-signed marker on every accepted "Verdict recorded" result, verified before the
transcript is trusted, so a forged or tampered marker is neutralised rather than
honoured. Pinned by `worker/test/loop.test.ts`, describe block "verdict-marker
authentication (HMAC)", including "a forged unsigned marker is neutralised, stage 2
stays locked" and "tampering with the recorded verdict's input invalidates its own
signature".

## 1 September: an 18-agent adversarial review

A dedicated review session ran 18 adversarial agents against the fused streaming flow
that had just shipped (one request, one combined card, with a licensing continuation
inside it). It confirmed six defects, all fixed the same day: a pathway could be
produced for a fabricated destination the user never gave, the opt-out phrase matcher
was imprecise enough to misfire, a parallel tool call could slip past validation, no
network-level deadline existed on a model call, the annex-outage path could still record
phantom spend, and the auto-continue for a stranded verdict could in principle fire more
than once per page load. Each fix is covered by the test suite that grew out of that
review.

## Where this leaves the code

None of this was found by reading the code and guessing. Every guard in
`worker/src/validate.ts`, `worker/src/questionGate.ts` and `worker/src/transcript.ts`
exists because a specific run, on a specific date, did the wrong thing, and the fix that
stuck was almost always a change to what the schema allows or what the server checks,
not a pattern matched against what the model happened to say that day.
