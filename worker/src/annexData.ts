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
}

const TTL_MS = 30 * 60 * 1000;

let cached: { data: AnnexDataset; at: number } | null = null;

export async function loadAnnex(
  url: string,
  fetcher: typeof fetch = fetch,
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
  const norm = (s: string) => s.toLowerCase().replace(/\s+/g, " ").trim();
  const q = norm(quote);
  return q.length > 0 && norm(text).includes(q);
}
