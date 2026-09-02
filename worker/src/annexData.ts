// Annex I dataset access. The dataset (data/annex.json) is published by
// Export Controls Watch on every pipeline run, so it always reflects the
// latest consolidated version of Regulation (EU) 2021/821 Annex I —
// including in-place corrigenda. We cache it at module scope with a TTL.

export interface AnnexEntry {
  entry_code: string;
  category: string;
  verbatim_text: string;
  parameters: string[];
  applicable_notes: string[];
}

export interface AnnexDoc {
  doc_type: string;
  title: string;
  verbatim_text: string;
}

export interface Gea {
  id: string; // EU001..EU008
  title: string;
  verbatim_text: string;
}

export interface AnnexDataset {
  corpus_version: string;
  celex: string;
  valid_from: string | null;
  sha256: string;
  attribution: string;
  entry_count: number;
  index: { code: string; first_line: string }[];
  entries: AnnexEntry[];
  docs: AnnexDoc[];
  geas?: Gea[];
  gea_common_list?: string | null;
}

const TTL_MS = 30 * 60 * 1000;

let cached: { data: AnnexDataset; at: number } | null = null;

export async function loadAnnex(
  url: string,
  fetcher: typeof fetch = (...args: Parameters<typeof fetch>) => fetch(...args),
): Promise<AnnexDataset> {
  if (cached && Date.now() - cached.at < TTL_MS) return cached.data;
  const resp = await fetcher(url, { headers: { accept: "application/json" } });
  if (!resp.ok) {
    if (cached) return cached.data; // stale beats broken
    throw new Error(`annex.json fetch failed: ${resp.status}`);
  }
  const data = (await resp.json()) as AnnexDataset;
  if (!Array.isArray(data.entries) || data.entries.length === 0) {
    throw new Error("annex.json malformed: no entries");
  }
  cached = { data, at: Date.now() };
  return data;
}

export function resetAnnexCache(): void {
  cached = null;
}

export function entryByCode(annex: AnnexDataset, code: string): AnnexEntry | undefined {
  return annex.entries.find((e) => e.entry_code === code.toUpperCase().trim());
}

// The always-cached prompt context: every corpus doc EXCEPT the long A-Z
// definitions glossary (~12k tokens), which is served on demand through the
// lookup_definitions tool instead.
export function alwaysDocs(annex: AnnexDataset): AnnexDoc[] {
  return annex.docs.filter((d) => d.doc_type !== "definitions_annex");
}

// Case-insensitive lookup of quoted terms in the definitions glossary; returns
// the glossary paragraphs mentioning each term (verbatim, never paraphrased).
export function definitionsFor(annex: AnnexDataset, terms: string[]): string {
  const glossary = annex.docs
    .filter((d) => d.doc_type === "definitions_annex")
    .map((d) => d.verbatim_text)
    .join("\n");
  if (!glossary) return "No definitions annex held in this corpus version.";
  const paragraphs = glossary.split("\n");
  const out: string[] = [];
  for (const term of terms.slice(0, 10)) {
    const needle = term.toLowerCase().replace(/["“”]/g, "");
    const hits = paragraphs.filter((p) => p.toLowerCase().includes(needle));
    out.push(
      hits.length
        ? `Definitions matching "${term}":\n${hits.join("\n")}`
        : `No definition found for "${term}".`,
    );
  }
  return out.join("\n\n");
}

// Whitespace-normalised substring check: is `quote` genuinely somewhere in the
// corpus text? Used to refuse verdicts citing invented text.
export function quoteAppearsIn(quote: string, text: string): boolean {
  // normalise whitespace AND typographic variants: the Formex corpus uses curly
  // quotes/dashes ('MRF', "digital", –) that models render as ASCII — those are
  // the same quote, not an invention
  const norm = (s: string) =>
    s
      .toLowerCase()
      .replace(/[\u2018\u2019\u201A\u201B\u2032]/g, "'")
      .replace(/[\u201C\u201D\u201E\u201F\u2033]/g, '"')
      .replace(/[\u2010-\u2015\u2212]/g, "-")
      .replace(/\u00A0/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  const q = norm(quote);
  return q.length > 0 && norm(text).includes(q);
}

// same normalisation as quoteAppearsIn, reusable for divergence reporting
function normText(s: string): string {
  return s
    .toLowerCase()
    .replace(/[‘’‚‛′]/g, "'")
    .replace(/[“”„‟″]/g, '"')
    .replace(/[‐-―−]/g, "-")
    .replace(/ /g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

// Where does a failed quote stop matching the source? Models fix misquotes
// reliably when shown the divergence point — a bare "not found" leaves them
// re-guessing from memory (live: three attempts at EU001's destination list,
// each recalling "the United Kingdom and the United States" where the source
// carries the Northern Ireland proviso). Returns "" when nothing anchors.
export function quoteDivergenceHint(quote: string, text: string): string {
  const q = normText(quote);
  const t = normText(text);
  if (q.length === 0 || t.length === 0) return "";
  // longest prefix of the quote that still appears in the source
  let lo = 0;
  let hi = q.length;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (t.includes(q.slice(0, mid))) lo = mid;
    else hi = mid - 1;
  }
  if (lo < 20) return ""; // nothing meaningful anchors — the generic message stands
  const at = t.indexOf(q.slice(0, lo)) + lo;
  const anchor = q.slice(Math.max(0, lo - 45), lo);
  return (
    ` — your quote diverges after "...${anchor}"; the source text actually continues: ` +
    `"${t.slice(at, at + 160)}..."`
  );
}

// The verbatim text of exactly one provision plus its descendants, addressed by
// dotted path. Entry text is one line per provision, "<dotted-path> <text>", so
// the block is the line whose first token === path, plus every line whose token
// starts with `path.` (children). Returns null if the path is not in the entry.
// This scopes quote-validation to the CITED sub-item — a comparator or number
// lifted from a different clause of the same (multi-page) entry no longer passes.
export function provisionText(entry: AnnexEntry, dottedPath: string): string | null {
  const path = dottedPath.trim();
  const lines = entry.verbatim_text.split("\n");
  const block: string[] = [];
  for (const line of lines) {
    const token = line.split(/\s/, 1)[0];
    if (token === path || token.startsWith(path + ".")) block.push(line);
  }
  return block.length ? block.join("\n") : null;
}

export function geaById(annex: AnnexDataset, id: string): Gea | undefined {
  return (annex.geas ?? []).find((g) => g.id === id.toUpperCase().trim());
}

// Text a pathway condition may quote from: the named GEA, or the Annex II
// common excluded-items list (cited as COMMON_LIST).
export function geaScopeText(annex: AnnexDataset, id: string): string | null {
  if (id.toUpperCase().trim() === "COMMON_LIST") return annex.gea_common_list ?? null;
  return geaById(annex, id)?.verbatim_text ?? null;
}
