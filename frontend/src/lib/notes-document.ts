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

/* ── Multiple sticky notes ──────────────────────────────────────────────── */

/**
 * The notes file can hold several independent notes. Each entry keeps its own
 * CKEditor document (or legacy HTML), so closing/creating one never touches
 * the others. Older files store a single document and are migrated on load.
 */
export const NOTES_COLLECTION_FORMAT = "genova-sticky-notes";
export const NOTES_COLLECTION_VERSION = 1;

/** A single persisted sticky note. */
export interface StickyNoteRecord {
  id: string;
  /** User-facing tab label (e.g. "Note 1", "Chemistry"). */
  title: string;
  /** CKEditor model JSON for this note. */
  document: Record<string, unknown> | null;
  /** HTML mirror, used for exports and as the pre-JSON fallback. */
  html: string;
}

export interface NormalizedNotesCollection {
  notes: StickyNoteRecord[];
  activeId: string | null;
  /** At least one note still needs its HTML upgraded to structured JSON. */
  pendingMigration: boolean;
}

/** Build the object written to `{user_id}/sticky_notes/notes.json`. */
export function createNotesEnvelope(
  notes: StickyNoteRecord[],
  activeId: string | null,
): Record<string, unknown> {
  return {
    format: NOTES_COLLECTION_FORMAT,
    version: NOTES_COLLECTION_VERSION,
    activeId,
    notes: notes.map((note) => ({
      id: note.id,
      title: note.title,
      document: note.document,
      html: note.html,
    })),
  };
}

/** True when a stored value is the multi-note envelope (not a bare document). */
export function isNotesCollection(raw: unknown): boolean {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return false;
  const record = raw as Record<string, unknown>;
  return (
    record.format === NOTES_COLLECTION_FORMAT && Array.isArray(record.notes)
  );
}

function noteFrom(
  value: unknown,
  id: string,
  /** Stored HTML mirror, kept alongside a structured document for exports. */
  htmlMirror = "",
  /**
   * Keep an explicitly listed but still-empty note. Closing a note removes its
   * entry, so an entry with no content is a real note the user has not typed in
   * yet and must survive a reload.
   */
  allowEmpty = false,
  title = "",
): { note: StickyNoteRecord; pendingMigration: boolean } | null {
  const normalized = normalizeNoteDocument(value);
  const label = title.trim() || "Note";
  if (!normalized.document && !normalized.html) {
    if (!allowEmpty) return null;
    return {
      note: { id, title: label, document: null, html: htmlMirror },
      pendingMigration: false,
    };
  }

  return {
    note: {
      id,
      title: label,
      document: normalized.document as unknown as Record<string, unknown> | null,
      html: normalized.html ?? htmlMirror,
    },
    // Pre-JSON HTML only becomes lossless once the editor rewrites it.
    pendingMigration: !normalized.document && Boolean(normalized.html),
  };
}

/**
 * Accepts the current multi-note envelope, the previous single-note draft
 * (`{ document, html }`), or a bare CKEditor document.
 */
export function normalizeNotesCollection(raw: unknown): NormalizedNotesCollection {
  if (raw == null) return { notes: [], activeId: null, pendingMigration: false };
  if (typeof raw !== "object" || Array.isArray(raw)) {
    return { notes: [], activeId: null, pendingMigration: false };
  }

  const record = raw as Record<string, unknown>;

  if (Array.isArray(record.notes)) {
    const notes: StickyNoteRecord[] = [];
    let pendingMigration = false;

    record.notes.forEach((item, index) => {
      if (!item || typeof item !== "object" || Array.isArray(item)) return;
      const entry = item as Record<string, unknown>;
      const id =
        typeof entry.id === "string" && entry.id
          ? entry.id
          : `note-${index + 1}`;
      const wrapped = "document" in entry || "html" in entry;
      const source = wrapped ? (entry.document ?? entry.html ?? null) : entry;
      const htmlMirror =
        typeof entry.html === "string" ? entry.html : "";
      const title =
        typeof entry.title === "string" && entry.title.trim()
          ? entry.title.trim()
          : `Note ${index + 1}`;
      const parsed = noteFrom(source, id, htmlMirror, true, title);
      if (!parsed) return;
      notes.push(parsed.note);
      pendingMigration = pendingMigration || parsed.pendingMigration;
    });

    const activeId =
      typeof record.activeId === "string" &&
      notes.some((note) => note.id === record.activeId)
        ? record.activeId
        : (notes[0]?.id ?? null);

    return { notes, activeId, pendingMigration };
  }

  // Previous single-note draft shape: { document, html, savedAt }.
  const legacyHtml = typeof record.html === "string" ? record.html : "";
  const legacy = noteFrom(
    record.document ?? record.html ?? null,
    "note-1",
    legacyHtml,
    false,
    "Note 1",
  );
  if (legacy) {
    return {
      notes: [legacy.note],
      activeId: legacy.note.id,
      pendingMigration: legacy.pendingMigration,
    };
  }

  // Bare CKEditor document / HTML string.
  const single = noteFrom(record, "note-1", "", false, "Note 1");
  if (single) {
    return {
      notes: [single.note],
      activeId: single.note.id,
      pendingMigration: single.pendingMigration,
    };
  }

  return { notes: [], activeId: null, pendingMigration: false };
}
