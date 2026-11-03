import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import remarkRehype from "remark-rehype";
import { toHtml } from "hast-util-to-html";
import type { Element, ElementContent, Properties, Root, Text } from "hast";
import { prepareMarkdownForMath } from "./math-markdown";

/**
 * Markdown → CKEditor-compatible HTML for sticky notes.
 *
 * The chat renders assistant answers with remark-gfm + remark-math + KaTeX.
 * Notes accept rich text through CKEditor, which only understands the tags its
 * plugins can upcast (headings, paragraphs, lists, block quotes, tables, code
 * blocks, bold/italic/strikethrough/code, links). So instead of pasting raw
 * Markdown (or KaTeX's span soup) into the editor we convert the message into
 * that supported subset of HTML. Math is turned into readable Unicode text so
 * formulas (and mhchem equations such as `\ce{2H2 + O2 -> 2H2O}`) stay legible
 * and editable inside CKEditor, which has no TeX plugin.
 */

const SUPERSCRIPT: Record<string, string> = {
  "0": "⁰",
  "1": "¹",
  "2": "²",
  "3": "³",
  "4": "⁴",
  "5": "⁵",
  "6": "⁶",
  "7": "⁷",
  "8": "⁸",
  "9": "⁹",
  "+": "⁺",
  "-": "⁻",
  "−": "⁻",
  "=": "⁼",
  "(": "⁽",
  ")": "⁾",
  n: "ⁿ",
  i: "ⁱ",
  a: "ᵃ",
  b: "ᵇ",
  c: "ᶜ",
  d: "ᵈ",
  e: "ᵉ",
  f: "ᶠ",
  g: "ᵍ",
  h: "ʰ",
  j: "ʲ",
  k: "ᵏ",
  l: "ˡ",
  m: "ᵐ",
  o: "ᵒ",
  p: "ᵖ",
  r: "ʳ",
  s: "ˢ",
  t: "ᵗ",
  u: "ᵘ",
  v: "ᵛ",
  w: "ʷ",
  x: "ˣ",
  y: "ʸ",
  z: "ᶻ",
  T: "ᵀ",
};

const SUBSCRIPT: Record<string, string> = {
  "0": "₀",
  "1": "₁",
  "2": "₂",
  "3": "₃",
  "4": "₄",
  "5": "₅",
  "6": "₆",
  "7": "₇",
  "8": "₈",
  "9": "₉",
  "+": "₊",
  "-": "₋",
  "−": "₋",
  "=": "₌",
  "(": "₍",
  ")": "₎",
  a: "ₐ",
  e: "ₑ",
  h: "ₕ",
  i: "ᵢ",
  j: "ⱼ",
  k: "ₖ",
  l: "ₗ",
  m: "ₘ",
  n: "ₙ",
  o: "ₒ",
  p: "ₚ",
  r: "ᵣ",
  s: "ₛ",
  t: "ₜ",
  u: "ᵤ",
  v: "ᵥ",
  x: "ₓ",
};

const GREEK: Record<string, string> = {
  alpha: "α",
  beta: "β",
  gamma: "γ",
  delta: "δ",
  epsilon: "ε",
  varepsilon: "ε",
  zeta: "ζ",
  eta: "η",
  theta: "θ",
  vartheta: "ϑ",
  iota: "ι",
  kappa: "κ",
  lambda: "λ",
  mu: "μ",
  nu: "ν",
  xi: "ξ",
  omicron: "ο",
  pi: "π",
  varpi: "ϖ",
  rho: "ρ",
  sigma: "σ",
  tau: "τ",
  upsilon: "υ",
  phi: "φ",
  varphi: "φ",
  chi: "χ",
  psi: "ψ",
  omega: "ω",
  Gamma: "Γ",
  Delta: "Δ",
  Theta: "Θ",
  Lambda: "Λ",
  Xi: "Ξ",
  Pi: "Π",
  Sigma: "Σ",
  Upsilon: "Υ",
  Phi: "Φ",
  Psi: "Ψ",
  Omega: "Ω",
};

const SYMBOLS: Record<string, string> = {
  times: "×",
  cdot: "·",
  div: "÷",
  pm: "±",
  mp: "∓",
  le: "≤",
  leq: "≤",
  ge: "≥",
  geq: "≥",
  ne: "≠",
  neq: "≠",
  approx: "≈",
  equiv: "≡",
  propto: "∝",
  sim: "∼",
  to: "→",
  rightarrow: "→",
  leftarrow: "←",
  leftrightarrow: "↔",
  Rightarrow: "⇒",
  Leftarrow: "⇐",
  Leftrightarrow: "⇔",
  infty: "∞",
  partial: "∂",
  nabla: "∇",
  sum: "∑",
  prod: "∏",
  int: "∫",
  oint: "∮",
  deg: "°",
  degree: "°",
  circ: "∘",
  angle: "∠",
  perp: "⊥",
  parallel: "∥",
  ldots: "…",
  dots: "…",
  cdots: "⋯",
  uparrow: "↑",
  downarrow: "↓",
  quad: " ",
  qquad: "  ",
  therefore: "∴",
  because: "∵",
  in: "∈",
  notin: "∉",
  subset: "⊂",
  supset: "⊃",
  cup: "∪",
  cap: "∩",
  emptyset: "∅",
};

const TEXT_COMMANDS = new Set([
  "text",
  "textrm",
  "textnormal",
  "mathrm",
  "mathbf",
  "mathit",
  "mathsf",
  "mathtt",
  "operatorname",
  "pu",
]);

interface Argument {
  value: string;
  next: number;
}

/** Read a balanced `{...}` group starting at `index`. */
function readGroup(source: string, index: number): Argument | null {
  if (source[index] !== "{") return null;
  let depth = 0;
  for (let i = index; i < source.length; i += 1) {
    const char = source[i];
    if (char === "\\") {
      i += 1;
      continue;
    }
    if (char === "{") depth += 1;
    else if (char === "}") {
      depth -= 1;
      if (depth === 0) {
        return { value: source.slice(index + 1, i), next: i + 1 };
      }
    }
  }
  return null;
}

/** Read either a braced group or a single command/character argument. */
function readArgument(source: string, index: number): Argument {
  let i = index;
  while (i < source.length && /\s/.test(source[i])) i += 1;

  const group = readGroup(source, i);
  if (group) return group;

  if (source[i] === "\\") {
    const match = /^\\[a-zA-Z]+/.exec(source.slice(i));
    if (match) return { value: match[0], next: i + match[0].length };
  }

  return { value: source[i] ?? "", next: Math.min(i + 1, source.length) };
}

function scriptify(value: string, map: Record<string, string>, marker: string): string {
  const text = convertLatex(value);
  const chars = Array.from(text);
  if (chars.length > 0 && chars.every((char) => map[char])) {
    return chars.map((char) => map[char]).join("");
  }
  return `${marker}(${text})`;
}

/** Group a fraction numerator/denominator when it contains an operator. */
function fractionPart(value: string): string {
  const text = convertLatex(value);
  return /[+\-−=<>]|\s/.test(text) ? `(${text})` : text;
}

/**
 * Turn mhchem source into readable text: arrows become →/⇌ and digits after
 * an element or bracket become subscripts (`2H2 + O2 -> 2H2O` → `2H₂ + O₂ → 2H₂O`).
 */
function convertChemistry(raw: string): string {
  let text = raw
    .replace(/<=>/g, "⇌")
    .replace(/<->/g, "↔")
    .replace(/-->/g, "→")
    .replace(/->/g, "→")
    .replace(/<--/g, "←")
    .replace(/<-/g, "←");

  text = text.replace(/\^\{([^}]*)\}/g, (_match, charge: string) =>
    Array.from(charge)
      .map((char) => SUPERSCRIPT[char] ?? char)
      .join(""),
  );
  text = text.replace(/\^([0-9]*[+-])/g, (_match, charge: string) =>
    Array.from(charge)
      .map((char) => SUPERSCRIPT[char] ?? char)
      .join(""),
  );

  let output = "";
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const previous = output[output.length - 1] ?? "";

    if (char === "_") {
      i += 1;
      while (i < text.length && /[0-9]/.test(text[i])) {
        output += SUBSCRIPT[text[i]] ?? text[i];
        i += 1;
      }
      i -= 1;
      continue;
    }

    if (/[0-9]/.test(char) && /[A-Za-z)\]]/.test(previous)) {
      output += SUBSCRIPT[char] ?? char;
      continue;
    }

    if (char === "{" || char === "}") continue;
    output += char;
  }

  return output.trim();
}

/** Convert a LaTeX snippet into readable, editor-friendly text. */
function convertLatex(source: string): string {
  let output = "";
  let i = 0;

  while (i < source.length) {
    const char = source[i];

    if (char === "\\") {
      const command = /^\\([a-zA-Z]+)\*?/.exec(source.slice(i));
      if (command) {
        const name = command[1];
        i += command[0].length;

        if (name === "frac" || name === "dfrac" || name === "tfrac") {
          const numerator = readArgument(source, i);
          const denominator = readArgument(source, numerator.next);
          i = denominator.next;
          output += `${fractionPart(numerator.value)}/${fractionPart(denominator.value)}`;
          continue;
        }

        if (name === "sqrt") {
          i = skipOptionalArgument(source, i);
          const argument = readArgument(source, i);
          i = argument.next;
          output += `√(${convertLatex(argument.value)})`;
          continue;
        }

        if (name === "ce") {
          const argument = readArgument(source, i);
          i = argument.next;
          output += convertChemistry(argument.value);
          continue;
        }

        if (TEXT_COMMANDS.has(name)) {
          const argument = readArgument(source, i);
          i = argument.next;
          output += convertLatex(argument.value);
          continue;
        }

        if (name === "left" || name === "right") {
          // Drop the sizing delimiter that follows: `\left(` / `\right]`.
          if (source[i] === "\\") {
            const delimiter = /^\\[a-zA-Z]+|^\\./.exec(source.slice(i));
            if (delimiter) i += delimiter[0].length;
          } else if (source[i] === ".") {
            i += 1;
          }
          continue;
        }

        if (name === "begin" || name === "end") {
          const argument = readArgument(source, i);
          i = argument.next;
          continue;
        }

        if (GREEK[name]) {
          output += GREEK[name];
          continue;
        }
        if (SYMBOLS[name]) {
          output += SYMBOLS[name];
          continue;
        }

        // Unknown command: keep the readable name without the backslash.
        output += name;
        continue;
      }

      // Escaped characters (`\%`, `\{`, `\,`, `\\` …).
      const next = source[i + 1] ?? "";
      i += 2;
      if (next === "\\") output += "\n";
      else if (next === "," || next === ";" || next === ":" || next === "!" || next === " ") {
        output += " ";
      } else if (next) {
        output += next;
      }
      continue;
    }

    if (char === "^" || char === "_") {
      const marker = char;
      const argument = readArgument(source, i + 1);
      i = argument.next;
      output +=
        marker === "^"
          ? scriptify(argument.value, SUPERSCRIPT, "^")
          : scriptify(argument.value, SUBSCRIPT, "_");
      continue;
    }

    if (char === "{" || char === "}") {
      i += 1;
      continue;
    }

    if (char === "~" || char === "&") {
      output += " ";
      i += 1;
      continue;
    }

    output += char;
    i += 1;
  }

  return output.replace(/[ \t]{2,}/g, " ").trim();
}

/** Skip an optional `[...]` argument (`\sqrt[3]{x}`). */
function skipOptionalArgument(source: string, index: number): number {
  let i = index;
  while (i < source.length && /\s/.test(source[i])) i += 1;
  if (source[i] !== "[") return index;
  const close = source.indexOf("]", i);
  return close === -1 ? index : close + 1;
}

function classNames(properties: Properties | undefined): string[] {
  const value = properties?.className;
  const list = Array.isArray(value) ? value : value ? [value] : [];
  return list.map((entry) => String(entry));
}

function textOf(node: ElementContent): string {
  if (node.type === "text") return node.value;
  if (node.type !== "element") return "";
  return node.children.map(textOf).join("");
}

function mathElement(node: ElementContent): Element | null {
  if (node.type !== "element") return null;
  return classNames(node.properties).includes("math-display") ||
    classNames(node.properties).includes("math-inline")
    ? node
    : null;
}

function inlineFormula(latex: string): Element {
  return {
    type: "element",
    tagName: "em",
    properties: {},
    children: [{ type: "text", value: convertLatex(latex) }],
  };
}

function displayFormula(latex: string): Element {
  return {
    type: "element",
    tagName: "p",
    properties: { style: "text-align: center;" },
    children: [inlineFormula(latex)],
  };
}

function transformChildren(children: ElementContent[]): ElementContent[] {
  return children.flatMap(transformNode);
}

const INLINE_TAGS = new Set([
  "a",
  "br",
  "code",
  "del",
  "em",
  "s",
  "span",
  "strong",
  "sub",
  "sup",
  "u",
]);

/**
 * CKEditor's data processor expects block elements at the root. Markdown
 * output normally satisfies that, but raw HTML nodes become bare text, so wrap
 * any stray inline content in a paragraph instead of losing it.
 */
function groupRootChildren(children: ElementContent[]): ElementContent[] {
  const output: ElementContent[] = [];
  let inline: ElementContent[] = [];

  const flush = () => {
    if (inline.length === 0) return;
    output.push({
      type: "element",
      tagName: "p",
      properties: {},
      children: inline,
    });
    inline = [];
  };

  for (const child of children) {
    const inlineNode =
      child.type === "text"
        ? child.value.trim().length > 0
        : child.type === "element" && INLINE_TAGS.has(child.tagName);

    if (child.type === "text" && child.value.trim().length === 0) continue;

    if (inlineNode) {
      inline.push(child);
    } else {
      flush();
      output.push(child);
    }
  }

  flush();
  return output;
}

function transformNode(node: ElementContent): ElementContent[] {
  if (node.type !== "element") return [node];

  // GFM task-list checkboxes are not part of CKEditor's schema; keep a glyph.
  if (node.tagName === "input" && node.properties?.type === "checkbox") {
    return [{ type: "text", value: node.properties?.checked ? "☑ " : "☐ " }];
  }

  if (node.tagName === "code") {
    const math = mathElement(node);
    if (math) return [inlineFormula(textOf(math))];
  }

  if (node.tagName === "pre") {
    const math = node.children.map(mathElement).find(Boolean);
    if (math) return [displayFormula(textOf(math))];
  }

  // GFM table alignment arrives as `align`; CKEditor reads `text-align` from
  // the cell style (TableProperties upcasts it to the cell alignment property).
  const properties: Properties = { ...(node.properties ?? {}) };
  if ((node.tagName === "th" || node.tagName === "td") && typeof properties.align === "string") {
    properties.style = `text-align: ${properties.align};`;
    delete properties.align;
  }

  return [
    {
      ...node,
      properties,
      children: transformChildren(node.children),
    },
  ];
}

const processor = unified()
  .use(remarkParse)
  .use(remarkGfm)
  .use(remarkMath)
  .use(remarkRehype, {
    // LLM output can contain raw HTML. Render it as literal text instead of
    // passing it through to the editor.
    handlers: {
      html(_state: unknown, node: { value: string }): Text {
        return { type: "text", value: node.value };
      },
    },
  });

/** Convert a Markdown/structured chat message into CKEditor-safe HTML. */
export function markdownToNoteHtml(markdown: string): string {
  if (!markdown || !markdown.trim()) return "";

  const prepared = prepareMarkdownForMath(markdown);
  const mdast = processor.parse(prepared);
  const hast = processor.runSync(mdast) as Root;
  const tree: Root = {
    type: "root",
    children: groupRootChildren(
      hast.children
        .filter((child) => child.type !== "doctype")
        .flatMap((child) => transformNode(child as ElementContent)),
    ),
  };

  return toHtml(tree);
}
