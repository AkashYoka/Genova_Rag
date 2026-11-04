import type { SourceBoundingBox } from "@/store/chat-store";

/** Collapse whitespace / PDF hyphenation so excerpts still match the text layer. */
export function normalizeCitationText(value: string): string {
  return value
    .replace(/\u00a0/g, " ")
    .replace(/\u00ad/g, "") // soft hyphen
    // PDF line-wrap hyphenation: "gener-\nation" → "generation"
    .replace(/-\s+/g, "")
    .replace(/\s+/g, " ")
    .replace(/\.{3}$/, "")
    .replace(/…$/, "")
    .trim()
    .toLowerCase();
}

/**
 * Backend stores bbox as JSON `[[x0,y0,x1,y1], ...]` in normalized 0–1 page space
 * (PyMuPDF). Convert to overlay rects the PDF viewer understands.
 */
export function parseCitationBbox(
  bbox?: string | null,
): SourceBoundingBox[] | undefined {
  if (!bbox || !bbox.trim()) return undefined;

  try {
    const parsed = JSON.parse(bbox) as unknown;
    if (!Array.isArray(parsed) || parsed.length === 0) return undefined;

    const rects: SourceBoundingBox[] = [];
    for (const entry of parsed) {
      if (Array.isArray(entry) && entry.length >= 4) {
        const [x0, y0, x1, y1] = entry.map((n) => Number(n));
        if ([x0, y0, x1, y1].some((n) => !Number.isFinite(n))) continue;
        const left = Math.min(x0, x1);
        const top = Math.min(y0, y1);
        const width = Math.abs(x1 - x0);
        const height = Math.abs(y1 - y0);
        if (width <= 0 || height <= 0) continue;
        rects.push({
          left: Number(left.toFixed(6)),
          top: Number(top.toFixed(6)),
          width: Number(width.toFixed(6)),
          height: Number(height.toFixed(6)),
          unit: "percent",
        });
        continue;
      }

      if (
        entry &&
        typeof entry === "object" &&
        "left" in entry &&
        "top" in entry &&
        "width" in entry &&
        "height" in entry
      ) {
        const box = entry as SourceBoundingBox;
        if (
          ![box.left, box.top, box.width, box.height].every((n) =>
            Number.isFinite(n),
          )
        ) {
          continue;
        }
        rects.push({
          left: box.left,
          top: box.top,
          width: box.width,
          height: box.height,
          unit: box.unit ?? "percent",
        });
      }
    }

    return rects.length > 0 ? rects : undefined;
  } catch {
    return undefined;
  }
}

export function resolveHighlightRects(source: {
  highlightRects?: SourceBoundingBox[];
  bbox?: string | null;
}): SourceBoundingBox[] | undefined {
  if (source.highlightRects && source.highlightRects.length > 0) {
    return source.highlightRects;
  }
  return parseCitationBbox(source.bbox);
}

type NormChar = {
  /** Owning text-layer span; null for inserted synthetic spaces. */
  span: HTMLElement | null;
};

const MIN_MATCH_CHARS = 12;

/**
 * Find the longest flexible match of `needle` inside `haystack`.
 * Skips hyphens and odd extra spaces so PDF line-wraps and slightly
 * truncated excerpts still cover the full passage.
 */
export function findFlexibleNeedleMatch(
  haystack: string,
  needle: string,
): { start: number; end: number; matched: number } | null {
  if (!haystack || !needle || needle.length < 8) return null;

  let best: { start: number; end: number; matched: number } | null = null;

  const seedLen = Math.min(24, needle.length);
  const seed = needle.slice(0, seedLen);

  const tryFrom = (start: number) => {
    let i = start;
    let j = 0;
    while (i < haystack.length && j < needle.length) {
      const h = haystack[i];
      const n = needle[j];
      if (h === n) {
        i += 1;
        j += 1;
        continue;
      }
      if (h === "-") {
        i += 1;
        continue;
      }
      if (h === " " && n !== " ") {
        i += 1;
        continue;
      }
      if (n === " " && h !== " ") {
        j += 1;
        continue;
      }
      // Compound word split across a line: "...protein-" + "protein interactions"
      // after matching the first "protein", skip the repeated half before continuing.
      const matchedSoFar = needle.slice(0, j);
      const lastWord = /([a-z0-9]+)\s*$/i.exec(matchedSoFar)?.[1];
      if (lastWord && lastWord.length >= 3) {
        const repeated = haystack.slice(i, i + lastWord.length);
        if (repeated === lastWord) {
          i += lastWord.length;
          continue;
        }
      }
      break;
    }
    return { end: i, matched: j };
  };

  if (seedLen >= 8) {
    let from = 0;
    while (from < haystack.length) {
      const idx = haystack.indexOf(seed, from);
      if (idx < 0) break;
      const result = tryFrom(idx);
      if (
        result.matched >= MIN_MATCH_CHARS &&
        (!best || result.matched > best.matched)
      ) {
        best = { start: idx, end: result.end, matched: result.matched };
        if (result.matched >= needle.length) return best;
      }
      from = idx + 1;
    }
  }

  // Fallback: longest exact prefix of the needle present in the haystack.
  if (!best) {
    let lo = Math.min(MIN_MATCH_CHARS, needle.length);
    let hi = needle.length;
    let foundStart = -1;
    let foundLen = 0;
    while (lo <= hi) {
      const mid = Math.floor((lo + hi) / 2);
      const prefix = needle.slice(0, mid);
      const idx = haystack.indexOf(prefix);
      if (idx >= 0) {
        foundStart = idx;
        foundLen = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    if (foundStart >= 0 && foundLen >= MIN_MATCH_CHARS) {
      const extended = tryFrom(foundStart);
      best = {
        start: foundStart,
        end: extended.end,
        matched: Math.max(foundLen, extended.matched),
      };
    }
  }

  return best;
}

/**
 * Build a searchable string from PDF.js text-layer spans, tracking which span
 * owns each normalized character. Inserts missing inter-span spaces.
 */
export function buildNormalizedTextLayerIndex(textLayerEl: Element): {
  normalized: string;
  chars: NormChar[];
} {
  const spans = Array.from(
    textLayerEl.querySelectorAll("span"),
  ) as HTMLElement[];

  let raw = "";
  const rawChars: NormChar[] = [];

  const push = (ch: string, span: HTMLElement | null) => {
    raw += ch;
    rawChars.push({ span });
  };

  for (let index = 0; index < spans.length; index += 1) {
    const span = spans[index];
    const text = span.textContent ?? "";
    if (!text) continue;

    const prevSpanText =
      index > 0 ? (spans[index - 1].textContent ?? "") : "";
    const prevEndsWithHyphen = /-\s*$/.test(prevSpanText);
    const lastRaw = raw.length > 0 ? raw[raw.length - 1] : "";
    const startsWithSpace = /^\s/.test(text);

    if (
      raw.length > 0 &&
      lastRaw !== " " &&
      lastRaw !== "-" &&
      !startsWithSpace &&
      !prevEndsWithHyphen
    ) {
      push(" ", null);
    }

    for (let i = 0; i < text.length; i += 1) {
      push(text[i], span);
    }
  }

  // Normalize while preserving the span map 1:1 with surviving characters.
  let normalized = "";
  const chars: NormChar[] = [];
  for (let i = 0; i < raw.length; i += 1) {
    const ch = raw[i];
    const next = raw[i + 1];

    if (ch === "\u00ad") continue;

    // "word-\nnext" / "word- next" → join as "wordnext" (line-wrap hyphen).
    if (ch === "-" && next !== undefined && /\s/.test(next)) {
      continue;
    }

    if (/\s/.test(ch)) {
      if (normalized.length === 0) continue;
      if (normalized[normalized.length - 1] === " ") continue;
      normalized += " ";
      chars.push(rawChars[i]);
      continue;
    }

    normalized += ch.toLowerCase();
    chars.push(rawChars[i]);
  }

  // Trim trailing space if any.
  if (normalized.endsWith(" ")) {
    normalized = normalized.slice(0, -1);
    chars.pop();
  }

  return { normalized, chars };
}

function rectsFromSpans(
  spans: Iterable<HTMLElement>,
  pageEl: HTMLElement,
): SourceBoundingBox[] {
  const pageBox = pageEl.getBoundingClientRect();
  if (pageBox.width <= 0 || pageBox.height <= 0) return [];

  const seen = new Set<HTMLElement>();
  const rects: SourceBoundingBox[] = [];
  for (const span of spans) {
    if (seen.has(span)) continue;
    seen.add(span);
    const r = span.getBoundingClientRect();
    if (r.width <= 1 || r.height <= 1) continue;
    if (
      r.right < pageBox.left ||
      r.left > pageBox.right ||
      r.bottom < pageBox.top ||
      r.top > pageBox.bottom
    ) {
      continue;
    }
    rects.push({
      left: (r.left - pageBox.left) / pageBox.width,
      top: (r.top - pageBox.top) / pageBox.height,
      width: r.width / pageBox.width,
      height: r.height / pageBox.height,
      unit: "percent",
    });
  }

  rects.sort((a, b) => a.top - b.top || a.left - b.left);
  return rects;
}

/**
 * Walk a PDF.js text layer and map a whitespace-normalized quote to overlay
 * rects relative to `pageEl`, including matches that span line breaks.
 *
 * Uses span bounding boxes (not Range.getClientRects) so PDF.js scaleX
 * transforms do not clip or misplace the highlight.
 */
export function findQuoteRectsInTextLayer(
  textLayerEl: Element,
  pageEl: HTMLElement,
  quote: string,
): SourceBoundingBox[] {
  const needle = normalizeCitationText(quote);
  if (!needle || needle.length < 8) return [];

  const { normalized, chars } = buildNormalizedTextLayerIndex(textLayerEl);
  if (!normalized || chars.length === 0) return [];

  const match = findFlexibleNeedleMatch(normalized, needle);
  if (!match) return [];

  const matchedSpans: HTMLElement[] = [];
  const end = Math.min(match.end, chars.length);
  for (let i = match.start; i < end; i += 1) {
    const span = chars[i]?.span;
    if (span) matchedSpans.push(span);
  }

  return rectsFromSpans(matchedSpans, pageEl);
}

export async function findPageForQuote(
  pdfDoc: {
    numPages: number;
    getPage: (pageNumber: number) => Promise<{
      getTextContent: () => Promise<{ items: unknown[] }>;
    }>;
  },
  quote: string,
  preferredPage?: number,
): Promise<number | null> {
  const needle = normalizeCitationText(quote);
  if (!needle || needle.length < 8) return null;

  const pages: number[] = [];
  if (
    typeof preferredPage === "number" &&
    preferredPage >= 1 &&
    preferredPage <= pdfDoc.numPages
  ) {
    pages.push(preferredPage);
  }
  for (let p = 1; p <= pdfDoc.numPages; p += 1) {
    if (!pages.includes(p)) pages.push(p);
  }

  let bestPage: number | null = null;
  let bestScore = 0;

  for (const pageNumber of pages) {
    try {
      const page = await pdfDoc.getPage(pageNumber);
      const content = await page.getTextContent();
      const haystack = normalizeCitationText(
        content.items
          .map((item) =>
            item && typeof item === "object" && "str" in item
              ? String((item as { str: string }).str)
              : "",
          )
          .join(" "),
      );
      const match = findFlexibleNeedleMatch(haystack, needle);
      if (match && match.matched > bestScore) {
        bestScore = match.matched;
        bestPage = pageNumber;
        if (match.matched >= needle.length) return pageNumber;
      }
    } catch {
      // Skip unreadable pages and keep searching.
    }
  }
  return bestPage;
}
