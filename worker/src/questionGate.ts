// Question-defect detectors and the licensing opt-out — pure, deterministic
// text checks that run on every candidate question before any judge model is
// consulted. Split out of loop.ts: these are the fixtures of live failures
// ("confirm the exact numerical aperture again, e.g. 1.35 or 1.350?", the
// five-times re-asked destination question) and read best on their own.

// candidate question before any judge model is consulted). Fixtures: the
// live failures "confirm the exact numerical aperture again, e.g. 1.35 or
// 1.350?" (echo + equal alternatives) and the five-times re-asked
// destination question (near-duplicate).
const UNIT_FACTORS: Record<string, number> = {
  nm: 1, nanometre: 1, nanometres: 1, nanometer: 1, nanometers: 1,
  um: 1000, µm: 1000, micrometre: 1000, micron: 1000, microns: 1000,
  mm: 1e6, millimetre: 1e6, millimeter: 1e6,
};

function numberTokens(text: string): { value: number; unit: string }[] {
  const out: { value: number; unit: string }[] = [];
  const re = /(\d+(?:[.,]\d+)?)\s*(nm|nanometres?|nanometers?|um|µm|micrometres?|microns?|mm|millimetres?|millimeters?)?\b/gi;
  for (const m of text.matchAll(re)) {
    const value = parseFloat(m[1].replace(",", "."));
    if (!Number.isFinite(value)) continue;
    const unitRaw = (m[2] ?? "").toLowerCase();
    const factor = UNIT_FACTORS[unitRaw];
    out.push(factor ? { value: value * factor, unit: "nm" } : { value, unit: unitRaw || "" });
  }
  return out;
}

function isHedged(text: string, value: number): boolean {
  const re = new RegExp(
    "\\b(about|roughly|approx\\w*|around|circa|~)\\s*" + String(value).replace(".", "[.,]"),
    "i",
  );
  return re.test(text);
}

// a question that echoes a number the user already stated, in the same
// sentence as a confirm-verb, is asking for nothing
export function questionEchoesStatedValue(candidate: string, userTexts: string[]): boolean {
  const stated = userTexts.flatMap((t) => numberTokens(t));
  if (stated.length === 0) return false;
  for (const sentence of candidate.split(/(?<=[.?!])\s+/)) {
    if (!/\b(confirm|verify|double.?check|re.?state|again)\b/i.test(sentence)) continue;
    for (const tok of numberTokens(sentence)) {
      const echoed = stated.some((s) => s.unit === tok.unit && Math.abs(s.value - tok.value) < 1e-9);
      if (echoed && !userTexts.some((t) => isHedged(t, tok.value))) return true;
    }
  }
  return false;
}

// "e.g. 1.35 exactly, or a more precise decimal like 1.350" — alternatives
// that normalise to the same number ask for nothing
export function questionOffersEqualAlternatives(candidate: string): boolean {
  const re = /(\d+(?:[.,]\d+)?)\s*(nm|um|µm|mm)?[^.?\n\d]{0,24}\bor\b[^.?\n\d]{0,40}(\d+(?:[.,]\d+)?)\s*(nm|um|µm|mm)?/gi;
  for (const m of candidate.matchAll(re)) {
    const a = parseFloat(m[1].replace(",", ".")) * (UNIT_FACTORS[(m[2] ?? "").toLowerCase()] ?? 1);
    const b = parseFloat(m[3].replace(",", ".")) * (UNIT_FACTORS[(m[4] ?? "").toLowerCase()] ?? 1);
    if (Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) < 1e-9) return true;
  }
  return false;
}

// "Just classify it — I don't need the licence": the user may opt out of the
// licensing stage entirely; the classification card then ships alone and
// destination questions are blocked. PRECISION over recall: a false opt-out
// silently amputates half the product ("license key", "without licensing
// fees" and the like must never fire), while a missed opt-out costs nothing —
// the model still honours rule 21 on its own. A LATER message that explicitly
// asks about licensing opts back in; last signal wins, so a wrong call in
// either direction is always recoverable in one message.
const OPT_OUT_LICENSING =
  /\b(only|just)\s+(the\s+)?classif|\b(only|just)\b[^.!?\n]{0,15}\b(need|want)\b[^.!?\n]{0,25}\bclassif|\ball\s+i\s+need\b[^.!?\n]{0,25}\bclassif|\bclassif\w*[^.!?\n]{0,25}\bis\s+(all\s+i\s+need|enough)\b|\b(no|don'?t|do not|not)\b[^.!?\n]{0,25}\b(need|want|require|care about|interested in)\b[^.!?\n]{0,15}\b(the|a|any)?\s*(licen[cs]e|licensing|authori[sz]\w*|pathway)\b(?!\s*(key|keys|server|fee|fees|agreement|terms|token))|\b(don'?t|do not)\s+(worry|bother)\s+about\b[^.!?\n]{0,25}\b(licen[cs]\w*|licensing|pathway|authori[sz])|\bskip\b[^.!?\n]{0,20}\b(licen[cs]\w*|licensing|pathway|authori[sz])|\b(solo|s[oó]lo|solamente)\b[^.!?\n]{0,20}\bclasificaci|clasificaci[oó]n\s+(solo|s[oó]lo|solamente)\b/i;

// Re-opt-in must be an explicit licensing ASK, not a stray mention — live
// answers legitimately contain "license key", "authorized personnel", "EU001-
// compliant" and must not silently cancel a genuine opt-out.
const OPT_BACK_IN =
  /(licen[cs]\w*|licensing|authori[sz]\w*|pathway|GEA|EU00[1-8])\b[^.!?\n]{0,40}\?|\b(which|what)\b[^.!?\n]{0,30}\b(licen[cs]e|licensing|authorisation|authorization|pathway|GEA|EU00[1-8])\b|\b(do\s+)?(i|we)\s+(need|want|get|apply\s+for)\b[^.!?\n]{0,25}\b(a\s+|the\s+)?(licen[cs]e|authorisation|authorization|permit)\b/i;

export function wantsClassificationOnly(userTexts: string[]): boolean {
  let only = false;
  for (const t of userTexts) {
    if (OPT_OUT_LICENSING.test(t)) only = true;
    else if (only && OPT_BACK_IN.test(t)) only = false;
  }
  return only;
}

// Questions that only serve the licensing stage — blocked once the user has
// opted out of it. Deliberately narrow: "end-use"/"exported to" appear in
// legitimate ITEM questions (decontrol notes, cryptographic APIs), so only
// unambiguous destination asks are gated; rule 21 covers the rest.
// Rule 2 and the README both promise that every interview question quotes
// the threshold it is testing, with its dotted path. Live questions arrive
// without any entry reference at all ("What is the maximum flight
// endurance…?") — correct, but indistinguishable from a generic chatbot,
// which is the one thing this tool must never look like. A question cites a
// provision when it names an entry code (9A012, 3B001.f.1.b…) or an Article.
export function questionCitesProvision(candidate: string): boolean {
  return /\b\d[A-E]\d{3}\b|\bArticle\s+\d/i.test(candidate);
}

export function questionAsksLicensingFacts(candidate: string): boolean {
  return /\b(destination|destin[oa]\b|country\s+of\s+destination|(which|what)\s+country|consignee|recipient\s+country)\b/i.test(
    candidate,
  );
}

function tokenSet(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter((w) => w.length > 2),
  );
}

// near-duplicate of a question the user has ALREADY answered — re-asking is
// forbidden whatever caused it (an unanswered question may be re-asked)
export function questionNearDuplicate(candidate: string, answeredQuestions: string[]): boolean {
  const c = tokenSet(candidate);
  if (c.size === 0) return false;
  for (const q of answeredQuestions) {
    const s = tokenSet(q);
    if (s.size === 0) continue;
    let inter = 0;
    for (const w of c) if (s.has(w)) inter++;
    const union = c.size + s.size - inter;
    if (union > 0 && inter / union >= 0.8) return true;
  }
  return false;
}
