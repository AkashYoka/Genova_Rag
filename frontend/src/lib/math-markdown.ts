/**
 * Prepare LLM markdown for KaTeX-backed rendering.
 * Normalizes \( \) / \[ \] delimiters, gives tables and display math their own
 * block boundaries, and holds back incomplete math while streaming.
 */

const CODE_FENCE = /```[\s\S]*?```/g;
const INLINE_CODE = /`[^`\n]+`/g;

/** GFM delimiter row, e.g. `| --- | :---: |` (leading/trailing pipes optional). */
const TABLE_DELIMITER_ROW =
  /^\s*\|?\s*:?-{1,}:?\s*(?:\|\s*:?-{1,}:?\s*)+\|?\s*$/;

type PlaceholderStore = { token: string; value: string };

function stashMatches(
  text: string,
  pattern: RegExp,
  prefix: string,
): { text: string; placeholders: PlaceholderStore[] } {
  const placeholders: PlaceholderStore[] = [];
  const next = text.replace(pattern, (match) => {
    const token = `\0${prefix}${placeholders.length}\0`;
    placeholders.push({ token, value: match });
    return token;
  });
  return { text: next, placeholders };
}

function restorePlaceholders(
  text: string,
  placeholders: PlaceholderStore[],
): string {
  let result = text;
  for (const { token, value } of placeholders) {
    result = result.split(token).join(value);
  }
  return result;
}

/**
 * Convert TeX-style delimiters to remark-math delimiters.
 *
 * Display math always becomes its own `$$` block so KaTeX renders it in
 * display mode. The negative look-behinds keep LaTeX line breaks such as
 * `\\[2pt]` from being mistaken for a `\[ ... \]` pair.
 */
function normalizeLatexDelimiters(text: string): string {
  return text
    .replace(
      /(?<!\\)\\\[([\s\S]*?)(?<!\\)\\\]/g,
      (_m, inner: string) => `\n$$\n${inner.trim()}\n$$\n`,
    )
    .replace(
      /(?<!\\)\\\(([\s\S]*?)(?<!\\)\\\)/g,
      (_m, inner: string) => `$${inner}$`,
    );
}

/** Expand `$$ ... $$` written on one line into a proper display block. */
function blockifyInlineDisplayMath(text: string): string {
  return text.replace(
    /\$\$([^\n$]+?)\$\$/g,
    (_m, inner: string) => `\n$$\n${inner.trim()}\n$$\n`,
  );
}

function isTableRow(line: string): boolean {
  return line.includes("|");
}

function isTableDelimiterRow(line: string): boolean {
  return TABLE_DELIMITER_ROW.test(line);
}

/**
 * Guarantee that tables and display-math blocks are separated from the prose
 * around them with blank lines.
 *
 * Without this, GFM swallows the sentence that follows a table into a new
 * table row, and a table placed directly after a list is not parsed at all.
 * Keeping intro text before the block and explanation after it mirrors how
 * ChatGPT lays out answers.
 */
function ensureBlockBreathingRoom(text: string): string {
  const lines = text.split("\n");
  const out: string[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const next = lines[i + 1];
    const trimmed = line.trim();

    // Display math: `$$` on its own line, body, closing `$$`.
    if (trimmed === "$$") {
      if (out.length > 0 && out[out.length - 1].trim() !== "") out.push("");
      out.push(line);
      i += 1;
      while (i < lines.length) {
        out.push(lines[i]);
        if (lines[i].trim() === "$$") {
          i += 1;
          break;
        }
        i += 1;
      }
      if (i < lines.length && lines[i].trim() !== "") out.push("");
      continue;
    }

    // GFM table: header row immediately followed by a delimiter row.
    if (next !== undefined && isTableRow(line) && isTableDelimiterRow(next)) {
      if (out.length > 0 && out[out.length - 1].trim() !== "") out.push("");
      out.push(line, next);
      i += 2;
      while (
        i < lines.length &&
        lines[i].trim() !== "" &&
        isTableRow(lines[i])
      ) {
        out.push(lines[i]);
        i += 1;
      }
      if (i < lines.length && lines[i].trim() !== "") out.push("");
      continue;
    }

    out.push(line);
    i += 1;
  }

  return out.join("\n");
}

/** Truncate from the start of an unclosed delimiter pair, if any. */
function truncateUnclosed(
  text: string,
  open: string,
  close: string,
): string {
  let depth = 0;
  let openAt = -1;
  let i = 0;

  while (i < text.length) {
    if (text.startsWith(open, i) && (open !== close || depth === 0)) {
      if (depth === 0) openAt = i;
      depth += 1;
      i += open.length;
      continue;
    }
    if (depth > 0 && text.startsWith(close, i)) {
      depth -= 1;
      i += close.length;
      continue;
    }
    i += 1;
  }

  return depth > 0 && openAt >= 0 ? text.slice(0, openAt) : text;
}

/**
 * Drop trailing incomplete math so partial streams do not flicker or throw.
 * Only applied when `isStreaming` is true.
 */
function stripIncompleteMath(text: string): string {
  let result = text;

  // Display $$...$$ (must run before single-dollar pass)
  result = truncateUnclosed(result, "$$", "$$");
  // Leftover TeX display/inline before $ normalization (defense in depth)
  result = truncateUnclosed(result, "\\[", "\\]");
  result = truncateUnclosed(result, "\\(", "\\)");

  // Single-dollar inline: walk tokens that are not part of $$
  let openAt = -1;
  let i = 0;
  while (i < result.length) {
    if (result.startsWith("$$", i)) {
      i += 2;
      continue;
    }
    if (result[i] === "$") {
      if (openAt < 0) {
        openAt = i;
      } else {
        openAt = -1;
      }
      i += 1;
      continue;
    }
    i += 1;
  }
  if (openAt >= 0) {
    result = result.slice(0, openAt);
  }

  return result;
}

/**
 * Normalize and (optionally) sanitize markdown math for rendering.
 * Code fences and inline code are left untouched.
 */
export function prepareMarkdownForMath(
  content: string,
  options: { isStreaming?: boolean } = {},
): string {
  if (!content) return content;

  const fenced = stashMatches(content, CODE_FENCE, "CODE");
  const inlined = stashMatches(fenced.text, INLINE_CODE, "INLINE");

  let text = normalizeLatexDelimiters(inlined.text);

  if (options.isStreaming) {
    text = stripIncompleteMath(text);
  } else {
    text = blockifyInlineDisplayMath(text);
    text = ensureBlockBreathingRoom(text);
  }

  text = restorePlaceholders(text, inlined.placeholders);
  text = restorePlaceholders(text, fenced.placeholders);
  return text;
}
