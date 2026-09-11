# Benchmark

| id | description | expected verdict | obtained verdict | correct | turns | seconds first turn | seconds total | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## How to fill it

1. Write a cases file: a JSON array of `{id, description, answers: [string...],
   expected: {status, entry_codes?, outcome?}}`, or the same array under a
   top-level `cases` key (see `worker/scripts/bench-cases.example.json`).
   `description` must already be anonymised: the script does not anonymise it.
2. Get a tester key (the operator's `TESTER_KEY` Wrangler secret, or your own
   value in `.dev.vars` for a local run) and export it: `export TESTER_KEY=...`.
3. Run `npm run bench -- --cases path/to/cases.json --out docs/benchmark.md`
   from `worker/`. Each case drives the deployed worker through a full
   conversation, feeding the next canned answer whenever it asks a question,
   and appends one table row here.

Nothing from this table is posted publicly (a LinkedIn post, a README claim,
anywhere outside this repo) until at least ten real, anonymised case
descriptions have been run for real and have annotated, human-checked
results in this table. Rows produced from `bench-cases.example.json` do not
count: their `expected` values are placeholders, not ground truth (see that
file's `_comment`).
