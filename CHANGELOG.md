# Changelog

Dates are the day the change landed on `main`.

## Unreleased — review branch (2026-09-08)

- **Fixed:** `needs_expert` can no longer ship while the model itself lists a fact the
  user could supply (`missing_facts` in the verdict schema, rejected structurally).
  Seen live with the "Long-range drone" example.
- **Faster:** a `needs_expert` draft on the opening message, or one that names missing
  facts, skips the forced card stage and goes straight to the question (one fewer
  Sonnet call on the first turn).
- **Measured:** whether each interview question cites a provision is logged
  (`question_cited`); the README no longer claims it is always true.
- **Card:** `met=false` rows are labelled "(not met)" instead of "(tested, ruled out)".
- **Demo:** example chips submit on click.
- **Structure:** `loop.ts` split into `validate.ts`, `questionGate.ts`, `transcript.ts`
  and `loop.ts`. No behaviour change.
- **Docs:** README gains "Known limitations" and "Threat model" sections; `.DS_Store`
  untracked.

## 2026-09 — fused single-card flow

- One interview, one card: a Listed verdict continues into the licensing pathway in
  the same request, with streamed progress.
- HMAC-signed verdict acceptance markers; forged transcripts cannot unlock stage 2.
- Question gate: deterministic detectors plus a cheap judge veto redundant questions.

## 2026-08 — hardening

- Per-IP request metering, nested `cache_control` stripping, CSP without inline script.
- Verbatim quotes are checked against the specific provision named by the dotted path,
  not the whole entry; formula-defined terms (MRF) must show their calculation.

## 2026-07 — first public build

- Cloudflare Worker + static page on GitHub Pages; corpus from Export Controls Watch.
