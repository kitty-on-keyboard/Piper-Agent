export interface FuzzyMatch {
  item: string;
  score: number;
}

/** Subsequence fuzzy scorer favoring basename matches, boundary hits, and contiguous runs. */
export function scoreFuzzy(query: string, target: string): number | null {
  const q = query.trim().toLowerCase();
  if (!q) return 0;
  const t = target.toLowerCase();

  let qi = 0;
  let score = 0;
  let run = 0;
  const lastSlash = t.lastIndexOf("/");

  for (let ti = 0; ti < t.length && qi < q.length; ti++) {
    if (t[ti] === q[qi]) {
      let bonus = 10;
      if (ti === 0 || "/_-.".includes(t[ti - 1])) bonus += 16;
      if (ti > lastSlash) bonus += 12;
      if (run > 0) bonus += Math.min(24, run * 6);
      score += bonus;
      run += 1;
      qi += 1;
    } else {
      run = 0;
      score -= 1;
    }
  }

  if (qi < q.length) return null;
  // Shorter paths tie-break slightly ahead of deeply nested paths.
  return score - target.length * 0.15;
}

export function filterFuzzy(query: string, items: string[], limit = 16): string[] {
  const q = query.trim();
  if (!q) return items.slice(0, limit);
  const scored: FuzzyMatch[] = [];
  for (const item of items) {
    const score = scoreFuzzy(q, item);
    if (score !== null) scored.push({ item, score });
  }
  scored.sort((a, b) => b.score - a.score || a.item.localeCompare(b.item));
  return scored.slice(0, limit).map((s) => s.item);
}
