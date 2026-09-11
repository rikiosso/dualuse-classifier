# Changelog

Dates are the day the change landed on `main`.

## Unreleased (2026-09-11)

- **Structure:** `loop.ts`'s forced verdict and licensing-pathway stages extracted into
  `stages.ts`. No behaviour change.
- **Scripts:** `npm run bench` (runs the annotated cases in `docs/benchmark.md` through
  the API and diffs the result), `npm run perf-summary` (averages the per-stage timings
  from `wrangler tail` into a latency table), `npm run probe-schema` (checks the tool
  schemas still match what the Anthropic API accepts).
- **CI:** a GitHub Actions workflow (`ci.yml`) runs the offline test suite and the
  typecheck on every push; its badge is on the README.
- **Docs:** README rewritten as a front page: the result-card image and tests badge up
  top, the four demo examples as links that pre-fill the page (`?q=`), "Why it exists"
  and "Numbers" sections, the long build explanation moved to the new
  `docs/architecture.md`. New `docs/how-it-was-built.md`: a dated account of building
  this with Claude Code as a lawyer, not an engineer. New `CLAUDE.md` at the repo root.
  New `docs/benchmark.md` and `docs/caught.md` templates, to be filled once the
  annotated cases and a captured rejection exist.
- **Demo page:** `?q=<text>` prefills the chat textarea (value only, never
  auto-submitted) and focuses it; the existing `?tester=` and `#CODE` browse-hash
  behaviour is unchanged. `og:image`/`twitter:image` point at the social preview image
  and `twitter:card` is `summary_large_image`. `app.js` bumped to `?v=9`.
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
