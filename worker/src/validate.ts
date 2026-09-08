// Corpus validation — the guarantee the README makes: no verdict or pathway
// reaches a user unless every code exists, every dotted path belongs to its
// entry and every quote appears verbatim in the provision it cites. Pure
// functions over the dataset; no model, no I/O. Split out of loop.ts so the
// contract can be read (and tested) without the conversation machinery.

import type { AnnexDataset } from "./annexData";
import {
  entryByCode,
  geaById,
  geaScopeText,
  provisionText,
  quoteAppearsIn,
  quoteDivergenceHint,
} from "./annexData";
import type { Pathway, Verdict } from "./tools";

// Quotes shorter than this are too weak to anchor — a 3-char fragment appears
// everywhere. Real thresholds and provisions comfortably clear it.
const MIN_QUOTE_CHARS = 12;

// Destinations whose sanctions regimes this tool must FLAG and never resolve —
// separate regulations with their own complexity; a wrong answer here is the
// most expensive mistake the tool could make. Enforced server-side.
const SANCTIONED_DESTINATIONS =
  /\b(russia|russian?|rusia|russie|russland|moscow|moscú|moskau|belarus|bielorrusia|biélorussie|belarusian|minsk|iran(?:ian)?|irán|tehe?ran|teherán|north[ -]?korea|corea del norte|corée du nord|nordkorea|pyongyang|dprk|democratic people'?s republic of korea|syria|siria|syrie|syrien|damascus|crimea|crimée|donetsk|luhansk|myanmar|burma|birmania|venezuela|caracas)\b/i;

// The verdict model sometimes garbles an intended-empty eligible_gea into
// tool-syntax artifacts (seen live on a sanctions card — and rejection only
// re-triggers the same glitch on retry, fail-closing a correct outcome).
// Outside gea_available the field carries no meaning, so an id that does not
// resolve is normalised to empty instead of rejected; gea_available keeps
// strict validation because the card headlines the id.
export function normalizePathway(pw: Pathway, annex: AnnexDataset): Pathway {
  if (pw.outcome !== "gea_available" && pw.eligible_gea && !geaById(annex, pw.eligible_gea)) {
    return { ...pw, eligible_gea: "" };
  }
  return pw;
}

// Every Category 5 Part 2 item sits inside EU008's subject matter — a live run
// concluded individual_licence_required for a 5A002 item after testing only
// EU001/EU007, with EU008 never retrieved. Enforced in code, not prompt.
const CAT5P2 = /^(5A00[2-4]|5B002|5D002|5E002)/i;

// trimOldToolResults shrinks old lookup outputs to keep long conversations

// Pathway validation — same discipline as verdicts: quotes verbatim-in-scope,
// referenced GEAs must exist, sanctioned destinations MUST carry the sanctions
// outcome (never a green light), a GEA outcome needs quoted conditions.
// verdictCodes (the stage-1 entry codes, when known) gates the EU008 sweep.
export function validatePathway(pw: Pathway, annex: AnnexDataset, verdictCodes: string[] = []): string[] {
  const problems: string[] = [];
  if (!["gea_available", "individual_licence_required", "sanctions_review_required"].includes(String(pw.outcome))) {
    problems.push(`outcome ${JSON.stringify(pw.outcome).slice(0, 40)} is not a valid pathway outcome`);
  }
  if (
    pw.outcome === "individual_licence_required" &&
    verdictCodes.some((c) => CAT5P2.test(c.trim())) &&
    geaById(annex, "EU008") &&
    !pw.conditions_quoted.some((c) => c.gea_id.trim().toUpperCase() === "EU008")
  ) {
    problems.push(
      "the classified item is Category 5 Part 2 (5A002/5D002/5E002) — retrieve EU008 via lookup_gea and either conclude gea_available under it or quote the EU008 scope/exclusion text that rules it out",
    );
  }
  if (pw.caveats.length === 0) problems.push("caveats must not be empty");
  if (!pw.destination.trim()) problems.push("destination must be stated");
  if (SANCTIONED_DESTINATIONS.test(pw.destination) && pw.outcome !== "sanctions_review_required") {
    problems.push(
      `destination "${pw.destination}" is under an EU sanctions regime — outcome must be sanctions_review_required`,
    );
  }
  // eligible_gea is either empty or a REAL GEA id — for every outcome (a live
  // run emitted garbage into this field under an individual_licence outcome)
  if (pw.eligible_gea && !geaById(annex, pw.eligible_gea)) {
    problems.push(`eligible_gea ${JSON.stringify(pw.eligible_gea).slice(0, 60)} does not exist in the corpus`);
  }
  if (pw.outcome === "gea_available") {
    if (!pw.eligible_gea) problems.push("gea_available requires eligible_gea");
    if (pw.conditions_quoted.length === 0) {
      problems.push("gea_available requires quoted conditions");
    } else if (
      pw.eligible_gea &&
      !pw.conditions_quoted.some((c) => c.gea_id.trim().toUpperCase() === pw.eligible_gea.trim().toUpperCase())
    ) {
      problems.push(
        `gea_available under ${pw.eligible_gea} must quote at least one condition from ${pw.eligible_gea} itself`,
      );
    }
  }
  if (pw.outcome === "individual_licence_required" && pw.conditions_quoted.length === 0) {
    problems.push(
      "individual_licence_required must quote the provision that rules the GEAs out (e.g. the coverage clause or exclusion tested)",
    );
  }
  for (const c of pw.conditions_quoted) {
    const scope = geaScopeText(annex, c.gea_id);
    if (!scope) {
      problems.push(`conditions cite nonexistent GEA ${c.gea_id}`);
    } else if (c.verbatim_quote.replace(/\s+/g, " ").trim().length < MIN_QUOTE_CHARS) {
      problems.push(`condition quote for ${c.gea_id} is too short to anchor`);
    } else if (!quoteAppearsIn(c.verbatim_quote, scope)) {
      problems.push(
        `condition quote for ${c.gea_id} not found in that authorisation's text — copy exactly from lookup_gea output${quoteDivergenceHint(c.verbatim_quote, scope)}`,
      );
    }
  }
  return problems;
}

// Server-side verdict validation — the NakedVerdict discipline. Returns a list
// of problems; empty list = acceptable.
export function validateVerdict(v: Verdict, annex: AnnexDataset): string[] {
  const problems: string[] = [];
  if (v.caveats.length === 0) problems.push("caveats must not be empty");

  // Reasoning rows carry a met flag: met=false rows are rule-outs (an entry
  // or cross-reference tested and found NOT to apply). A headline needs at
  // least one SUPPORTING row, and rule-out rows are exempt from the headline
  // requirement — the old symmetric checks (every headline backed by any row,
  // every cited code headlined) forced a live verdict to headline "3B001,
  // 3B501" while its own reasoning ruled 3B001 out.
  if (!["listed", "not_listed", "needs_expert"].includes(String(v.status))) {
    problems.push(`status ${JSON.stringify(v.status).slice(0, 40)} is not a valid verdict status`);
  }
  // a not-listed card is the tool's green light — it must show its work:
  // which candidate entries were tested and why each was ruled out
  if (v.status === "not_listed" && v.reasoning.length === 0) {
    problems.push(
      "a not_listed verdict must include reasoning rows (met=false) showing the candidate entries tested and ruled out",
    );
  }
  const headlined = v.entry_codes.map((c) => c.toUpperCase());
  if (v.status === "listed") {
    if (v.entry_codes.length === 0) problems.push("listed verdict needs entry_codes");
    if (v.reasoning.length === 0) problems.push("listed verdict needs reasoning");
    for (const code of v.entry_codes) {
      const backed = v.reasoning.some(
        (r) => (r.entry_code || "").toUpperCase() === code.toUpperCase() && r.met !== false,
      );
      if (!backed) {
        problems.push(
          `entry_code ${code} is headlined but has no supporting reasoning — an entry whose rows all rule it out (met=false) must be removed from entry_codes`,
        );
      }
    }
    for (const r of v.reasoning) {
      const code = (r.entry_code || "").toUpperCase();
      if (r.met !== false && !headlined.includes(code)) {
        problems.push(`reasoning cites ${code} as met but it is not in entry_codes`);
      }
    }
  }
  for (const code of v.entry_codes) {
    if (!entryByCode(annex, code)) problems.push(`entry_code ${code} does not exist in the corpus`);
  }
  for (const r of v.reasoning) {
    const entry = entryByCode(annex, r.entry_code);
    if (!entry) {
      problems.push(`reasoning cites nonexistent entry ${r.entry_code}`);
      continue;
    }
    // the pinpoint path must belong to the cited entry.
    // Technical Notes belong to their parent provision — the corpus prints
    // them as "<path> Technical Note(s): …" lines, and models cite the path
    // with the suffix attached. Normalise it away so the quote validates
    // against the provision block (which includes its note lines) instead of
    // silently falling through to whole-entry scope, which would defeat the
    // provision-scoping this validator exists for.
    const path = (r.dotted_path || "").trim().replace(/[\s,.]*Technical\s+Notes?\b.*$/i, "").trim();
    if (!path.toUpperCase().startsWith(r.entry_code.toUpperCase())) {
      problems.push(`dotted_path ${path} does not belong to entry ${r.entry_code}`);
      continue;
    }
    // the quote must appear in the SPECIFIC provision named by dotted_path — not
    // merely somewhere in the multi-page entry (blocks comparator/number flips
    // laundered from a sibling clause)
    const resolved = provisionText(entry, path);
    if (resolved === null && /^\d[A-E]\d{3}(\.[a-z0-9]+)+$/i.test(path)) {
      problems.push(
        `dotted_path ${path} does not resolve to a provision of ${r.entry_code} — cite the exact sub-item as printed in lookup_entries output`,
      );
      continue;
    }
    const scope = resolved ?? entry.verbatim_text;
    if (r.verbatim_quote.replace(/\s+/g, " ").trim().length < MIN_QUOTE_CHARS) {
      problems.push(`verbatim_quote for ${path} is too short to anchor a citation`);
    } else if (!quoteAppearsIn(r.verbatim_quote, scope)) {
      problems.push(
        `verbatim_quote for ${path} is not found in that provision's text — quotes must be copied exactly from lookup_entries output for the cited sub-item${quoteDivergenceHint(r.verbatim_quote, scope)}`,
      );
    }
  }
  // FORMULA-DEFINED TERMS (rule 18, enforced in code): where a Technical Note
  // on the cited provision or an ancestor defines a quoted term by formula,
  // any row quoting that term must SHOW the computation. A live first-turn
  // verdict adopted a user-claimed 38 nm MRF that the entry's own K=0.35
  // formula contradicts at any real numerical aperture.
  for (const r of v.reasoning) {
    const entry = entryByCode(annex, r.entry_code);
    if (!entry) continue;
    const cited = (r.dotted_path || "")
      .trim()
      .replace(/[\s,.]*Technical\s+Notes?\b.*$/i, "")
      .trim()
      .toUpperCase();
    if (!cited) continue;
    const definedTerms: string[] = [];
    for (const line of entry.verbatim_text.split("\n")) {
      if (!/technical note/i.test(line) || !/formula/i.test(line)) continue;
      const linePath = (line.split(/\s+/)[0] ?? "").toUpperCase();
      if (!linePath.includes(".")) continue;
      if (!(cited === linePath || cited.startsWith(linePath + "."))) continue;
      for (const m of line.matchAll(/['‘]([^'’]{2,60})['’]/g)) definedTerms.push(m[1]);
    }
    if (!definedTerms.some((t) => (r.verbatim_quote || "").includes(t))) continue;
    const expl = r.explanation || "";
    if (!(/formula|calculat/i.test(expl) && /=/.test(expl))) {
      problems.push(
        `${r.dotted_path} turns on a formula-defined term (see its Technical Note) — compute the value from the underlying parameters with the entry's own formula and constants, showing the calculation (e.g. 'MRF = (wavelength × K)/NA = …') in the explanation. Never adopt a user-claimed value for a defined term; if an input such as the numerical aperture is missing, do not conclude — ask the user for it`,
      );
      continue;
    }
    // arithmetic consistency: a shown computation must AGREE with the claim —
    // a live verdict computed 50,04 nm and declared it "at or below" a 45 nm
    // threshold. Narrow, safe direction only: a supporting row whose computed
    // value EXCEEDS an "…or less" threshold is a false conclusion. (European
    // decimal commas normalised; the last "= N nm" is the final result.)
    if (r.met !== false) {
      const calcs = [...expl.matchAll(/=\s*(\d+(?:[.,]\d+)?)\s*nm/gi)];
      const thr =
        /(\d+(?:[.,]\d+)?)\s*nm\s+or\s+less/i.exec(r.verbatim_quote || "") ??
        /less\s+than\s+or\s+equal\s+to\s+(\d+(?:[.,]\d+)?)\s*nm/i.exec(r.verbatim_quote || "");
      if (calcs.length > 0 && thr) {
        const value = parseFloat(calcs[calcs.length - 1][1].replace(",", "."));
        const threshold = parseFloat(thr[1].replace(",", "."));
        if (value > threshold) {
          problems.push(
            `${r.dotted_path}: the computed ${value} nm EXCEEDS the ${threshold} nm-or-less threshold — this criterion is NOT met. Mark it met=false, do not headline an entry on a failed computation, and if another entry's criteria need a missing parameter (e.g. 'dedicated chuck overlay'), ask the user for it instead of concluding`,
          );
        }
      }
    }
  }

  // N.B. / SEE ALSO cross-references carried by a cited provision (or an
  // ancestor of it) name sibling entries that catch similar equipment on
  // different criteria (3B001.f.1 ↔ 3B501.f: the same defined term with a
  // different K factor plus an overlay criterion). A live first-turn verdict
  // concluded on 3B001.f.1.b without ever testing 3B501 — prompt rules did
  // not stop it, so it is enforced here: every referenced entry must appear
  // somewhere in the verdict (entry_codes, reasoning or caveats), even if
  // only to say why it does not apply.
  if (v.status !== "needs_expert") {
    const mentioned = JSON.stringify(v).toUpperCase();
    const flagged = new Set<string>();
    for (const r of v.reasoning) {
      const entry = entryByCode(annex, r.entry_code);
      if (!entry) continue;
      const cited = (r.dotted_path || "")
        .trim()
        .replace(/[\s,.]*Technical\s+Notes?\b.*$/i, "")
        .trim()
        .toUpperCase();
      for (const line of entry.verbatim_text.split("\n")) {
        if (!/\bN\.B\.|SEE ALSO/i.test(line)) continue;
        const linePath = (line.split(/\s+/)[0] ?? "").toUpperCase();
        // root-level N.B.s ("3B001 N.B. SEE ALSO 2B226") span a whole entry —
        // generic context, not an obligation; requiring them taught the model
        // to interview users about isotope separators on a litho scanner
        if (!linePath.includes(".")) continue;
        if (!(cited === linePath || cited.startsWith(linePath + "."))) continue;
        for (const code of line.toUpperCase().match(/\b\d[A-E]\d{3}\b/g) ?? []) {
          if (code === r.entry_code.toUpperCase() || flagged.has(code)) continue;
          if (!entryByCode(annex, code)) continue;
          if (!mentioned.includes(code)) {
            flagged.add(code);
            problems.push(
              `the cited provision ${r.dotted_path} carries a cross-reference (N.B./SEE ALSO) to ${code} — either include ${code} in the verdict or state in caveats why it does not apply or cannot be assessed on the known facts`,
            );
          }
        }
      }
    }
  }
  return problems;
}

