# Caught: a documented validator rejection

The classifier is built to fail closed: every verdict and every licensing
pathway is checked against the corpus before anyone sees it (see
`worker/src/validate.ts`). When the check fails, the model gets told why and
has to try again: nothing wrong ever ships. This page documents one of those
rejections as it actually happened, so the guard's value is visible rather
than assumed.

## How to catch one

1. While you are testing (chat page, `curl`, or `npm run bench`), in a second
   terminal run:
   ```
   cd worker
   npx wrangler tail --format json
   ```
   This tails the deployed Worker's live logs. `--format json` is what
   `worker/scripts/perf-summary.ts` also expects, so save the output
   (`npx wrangler tail --format json > tail.ndjson`) if you also want a perf
   summary from the same session.
2. Grep the saved output for:
   - `"verdict rejected:"`: logged by `worker/src/loop.ts` next to where a
     `final_answer` call fails `validateVerdict()`. Carries the validation
     problems, truncated to 300 characters.
   - `"pathway rejected:"`: the same, for a `license_pathway` call that fails
     `validatePathway()`.
   These two are short, console-logged summaries: good for spotting THAT a
   rejection happened and roughly why.
3. To see the FULL rejection text the model itself was sent (the sentence
   starting `"Verdict rejected by corpus validation: ..."`: this is the
   `tool_result` content pushed back into the conversation, not a console
   line, so it will not appear in `wrangler tail`), read it off the
   conversation transcript directly: run the case again with the streaming
   `curl` from the project handoff notes, or with `npm run bench`, and inspect
   the `messages` array in the response, or capture the raw NDJSON from the
   terminal. The transcript entry with
   `is_error: true` on the rejected `tool_use_id` has the full text.

## The case

Fill in every section below from a rejection you actually saw. Leave a
section as `(not yet observed)` rather than inventing content: an empty
template is honest; a filled-in guess is not.

**Date:** (not yet observed)

**Case description (anonymised):**
(not yet observed)

**What the model tried to cite:**
(entry code, dotted path, and the exact verbatim_quote or claim it made: not yet observed)

**What the provision actually says:**
(the real corpus text at that dotted path: not yet observed)

**The rejection message (full text sent back to the model):**
(not yet observed)

**The final, correct verdict (after the model corrected itself):**
(not yet observed)
