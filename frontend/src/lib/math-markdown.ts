/**
 * Prepare LLM markdown for KaTeX-backed rendering.
 * Normalizes \( \) / \[ \] delimiters and holds back incomplete math while streaming.
 */

const CODE_FENCE = /```[\s\S]*?```/g;
const INLINE_CODE = /`[^`\n]+`/g;

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

/** Convert TeX-style \( \) / \[ \] to remark-math $ / $$ delimiters. */
function normalizeLatexDelimiters(text: string): string {
  return text
    .replace(/\\\[([\s\S]*?)\\\]/g, (_m, inner: string) => `$$${inner}$$`)
    .replace(/\\\(([\s\S]*?)\\\)/g, (_m, inner: string) => `$${inner}$`);
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
  }

  text = restorePlaceholders(text, inlined.placeholders);
  text = restorePlaceholders(text, fenced.placeholders);
  return text;
}
