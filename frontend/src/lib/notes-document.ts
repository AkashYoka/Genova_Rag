/**
 * Structured JSON representation of CKEditor content.
 *
 * CKEditor 5's data API only produces HTML: `editor.data.get()` returns a
 * string and there is no `stringify: false` option (that option exists only in
 * CKEditor 4). To store notes as structured JSON we serialize the editor's
 * model tree instead:
 *
 * ```json
 * {
 *   "format": "ckeditor5-model",
 *   "version": 1,
 *   "main": [
 *     { "name": "heading", "attributes": { "level": 2 },
 *       "children": [{ "data": "Summary" }] },
 *     { "name": "paragraph", "children": [
 *       { "data": "Finding: " },
 *       { "data": "critical", "attributes": { "bold": true } }
 *     ] },
 *     { "name": "list", "attributes": { "listType": "bulleted" },
 *       "children": [{ "name": "listItem", "children": [
 *         { "name": "paragraph", "children": [{ "data": "Item" }] }
 *       ] }] }
 *   ]
 * }
 * ```
 *
 * This mirrors CKEditor's own model JSON (`ModelElement.fromJSON()` /
 * `ModelText.fromJSON()`), so restoring it is lossless: headings, paragraphs,
 * bold/italic, links, list type/indent and block quotes all round-trip.
 */

export const NOTE_DOCUMENT_FORMAT = "ckeditor5-model";
export const NOTE_DOCUMENT_VERSION = 1;

/** A single CKEditor model node in plain-object form. */
export interface NoteModelNode {
  /** Element name (`paragraph`, `heading`, `list`, ...). Missing for text. */
  name?: string;
  /** Text content. Only present on text nodes. */
  data?: string;
  /** Model attributes such as `level`, `bold`, `italic`, `linkHref`. */
  attributes?: Record<string, unknown>;
  /** Child nodes. Only present on element nodes. */
  children?: NoteModelNode[];
}

/** The persisted sticky-notes document. */
export type NoteDocument = {
  format: typeof NOTE_DOCUMENT_FORMAT;
  version: number;
  /** Root `main` children, in document order. */
  main: NoteModelNode[];
};

export interface NormalizedNoteDocument {
  /** Structured document, when the stored value can be restored losslessly. */
  document: NoteDocument | null;
  /** Legacy HTML, when only HTML is available (pre-JSON notes). */
  html: string | null;
}

export function createNoteDocument(main: NoteModelNode[]): NoteDocument {
  return {
    format: NOTE_DOCUMENT_FORMAT,
    version: NOTE_DOCUMENT_VERSION,
    main,
  };
}

/** True when a value looks like a CKEditor model node. */
export function isNoteModelNode(value: unknown): value is NoteModelNode {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const node = value as Record<string, unknown>;
  return typeof node.name === "string" || typeof node.data === "string";
}

function nodesFrom(value: unknown): NoteModelNode[] | null {
  if (!Array.isArray(value)) return null;
  return value.filter(isNoteModelNode);
}

/**
 * Accepts everything a previous client version could have stored and returns
 * the best restorable representation:
 *
 * - `null` / missing → nothing (start with an empty editor)
 * - HTML string → legacy HTML
 * - `{ main: "<p>…</p>" }` → legacy HTML
 * - `{ main: [nodes] }` → structured document (current format)
 * - `{ main: { children: [nodes] } }` → structured document
 * - `{ children: [nodes] }` → structured document
 */
export function normalizeNoteDocument(raw: unknown): NormalizedNoteDocument {
  if (raw == null) return { document: null, html: null };
  if (typeof raw === "string") return { document: null, html: raw };
  if (typeof raw !== "object" || Array.isArray(raw)) {
    return { document: null, html: null };
  }

  const record = raw as Record<string, unknown>;
  const main = record.main;

  if (typeof main === "string") return { document: null, html: main };

  let nodes = nodesFrom(main);
  if (nodes === null && main && typeof main === "object" && !Array.isArray(main)) {
    nodes = nodesFrom((main as Record<string, unknown>).children);
  }
  if (nodes === null) {
    nodes = nodesFrom(record.children);
  }
  if (nodes === null) {
    return { document: null, html: null };
  }

  return { document: createNoteDocument(nodes), html: null };
}
