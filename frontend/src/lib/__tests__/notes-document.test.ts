import { describe, expect, it } from "vitest";
import {
  NOTES_COLLECTION_FORMAT,
  NOTES_COLLECTION_VERSION,
  NOTE_DOCUMENT_FORMAT,
  NOTE_DOCUMENT_VERSION,
  createNoteDocument,
  createNotesEnvelope,
  isNoteModelNode,
  isNotesCollection,
  normalizeNoteDocument,
  normalizeNotesCollection,
} from "../notes-document";

const MAIN = [
  {
    name: "heading",
    attributes: { level: 2 },
    children: [{ data: "Summary" }],
  },
  {
    name: "paragraph",
    children: [
      { data: "Finding: " },
      { data: "critical", attributes: { bold: true } },
    ],
  },
  {
    name: "list",
    attributes: { listType: "bulleted", listIndent: 0 },
    children: [
      {
        name: "listItem",
        children: [{ name: "paragraph", children: [{ data: "Item" }] }],
      },
    ],
  },
];

describe("notes-document", () => {
  it("creates a versioned structured document", () => {
    const doc = createNoteDocument(MAIN);

    expect(doc.format).toBe(NOTE_DOCUMENT_FORMAT);
    expect(doc.version).toBe(NOTE_DOCUMENT_VERSION);
    expect(doc.main).toEqual(MAIN);
  });

  it("recognises model nodes", () => {
    expect(isNoteModelNode({ data: "text" })).toBe(true);
    expect(isNoteModelNode({ name: "paragraph", children: [] })).toBe(true);
    expect(isNoteModelNode(null)).toBe(false);
    expect(isNoteModelNode(["nope"])).toBe(false);
    expect(isNoteModelNode({})).toBe(false);
  });

  it("returns nothing for an empty cloud document", () => {
    expect(normalizeNoteDocument(null)).toEqual({ document: null, html: null });
    expect(normalizeNoteDocument(undefined)).toEqual({
      document: null,
      html: null,
    });
  });

  it("restores the current structured format", () => {
    const restored = normalizeNoteDocument(createNoteDocument(MAIN));

    expect(restored.document?.main).toEqual(MAIN);
    expect(restored.html).toBeNull();
  });

  it("accepts a bare main array", () => {
    const restored = normalizeNoteDocument({ main: MAIN });

    expect(restored.document?.main).toEqual(MAIN);
  });

  it("accepts a main object with children", () => {
    const restored = normalizeNoteDocument({ main: { children: MAIN } });

    expect(restored.document?.main).toEqual(MAIN);
  });

  it("accepts a bare children array", () => {
    const restored = normalizeNoteDocument({ children: MAIN });

    expect(restored.document?.main).toEqual(MAIN);
  });

  it("falls back to legacy HTML strings", () => {
    expect(normalizeNoteDocument("<p>hello</p>")).toEqual({
      document: null,
      html: "<p>hello</p>",
    });
    expect(normalizeNoteDocument({ main: "<p>hello</p>" })).toEqual({
      document: null,
      html: "<p>hello</p>",
    });
  });

  it("ignores values it cannot restore", () => {
    expect(normalizeNoteDocument({ main: 42 })).toEqual({
      document: null,
      html: null,
    });
    expect(normalizeNoteDocument({ blocks: "nope" })).toEqual({
      document: null,
      html: null,
    });
    expect(normalizeNoteDocument(["nope"])).toEqual({
      document: null,
      html: null,
    });
  });
});

const NOTE_A = createNoteDocument([
  { name: "paragraph", children: [{ data: "Alpha" }] },
]);
const NOTE_B = createNoteDocument([
  { name: "heading", attributes: { level: 1 }, children: [{ data: "Beta" }] },
]);

describe("notes collection (multiple sticky notes)", () => {
  it("builds the persisted envelope", () => {
    const envelope = createNotesEnvelope(
      [
        { id: "a", document: NOTE_A, html: "<p>Alpha</p>" },
        { id: "b", document: NOTE_B, html: "<h1>Beta</h1>" },
      ],
      "b",
    );

    expect(envelope).toEqual({
      format: NOTES_COLLECTION_FORMAT,
      version: NOTES_COLLECTION_VERSION,
      activeId: "b",
      notes: [
        { id: "a", document: NOTE_A, html: "<p>Alpha</p>" },
        { id: "b", document: NOTE_B, html: "<h1>Beta</h1>" },
      ],
    });
  });

  it("recognises only the collection envelope", () => {
    const envelope = createNotesEnvelope([], null);

    expect(isNotesCollection(envelope)).toBe(true);
    expect(isNotesCollection(NOTE_A)).toBe(false);
    expect(isNotesCollection({ document: NOTE_A, html: "" })).toBe(false);
    expect(isNotesCollection(null)).toBe(false);
    expect(isNotesCollection([envelope])).toBe(false);
  });

  it("restores every note independently and the active note", () => {
    const normalized = normalizeNotesCollection(
      createNotesEnvelope(
        [
          { id: "note-a", document: NOTE_A, html: "<p>Alpha</p>" },
          { id: "note-b", document: NOTE_B, html: "<h1>Beta</h1>" },
        ],
        "note-b",
      ),
    );

    expect(normalized.activeId).toBe("note-b");
    expect(normalized.pendingMigration).toBe(false);
    expect(normalized.notes).toEqual([
      { id: "note-a", document: NOTE_A, html: "<p>Alpha</p>" },
      { id: "note-b", document: NOTE_B, html: "<h1>Beta</h1>" },
    ]);
  });

  it("falls back to the first note when activeId is unknown", () => {
    const normalized = normalizeNotesCollection(
      createNotesEnvelope(
        [
          { id: "note-a", document: NOTE_A, html: "" },
          { id: "note-b", document: NOTE_B, html: "" },
        ],
        "missing",
      ),
    );

    expect(normalized.activeId).toBe("note-a");
  });

  it("treats an empty envelope as authoritative", () => {
    const normalized = normalizeNotesCollection(
      createNotesEnvelope([], null),
    );

    expect(normalized.notes).toEqual([]);
    expect(normalized.activeId).toBeNull();
    expect(normalized.pendingMigration).toBe(false);
  });

  it("keeps legacy HTML notes and flags them for migration", () => {
    const normalized = normalizeNotesCollection({
      format: NOTES_COLLECTION_FORMAT,
      version: 1,
      activeId: "old",
      notes: [
        { id: "old", html: "<p>Carried over</p>" },
        { id: "new", document: NOTE_A, html: "<p>Alpha</p>" },
      ],
    });

    expect(normalized.notes[0]).toEqual({
      id: "old",
      document: null,
      html: "<p>Carried over</p>",
    });
    expect(normalized.notes[1].document).toEqual(NOTE_A);
    expect(normalized.pendingMigration).toBe(true);
  });

  it("migrates the previous single-note draft into Note 1", () => {
    const normalized = normalizeNotesCollection({
      document: NOTE_A,
      html: "<p>Alpha</p>",
      savedAt: "2026-01-01T00:00:00.000Z",
    });

    expect(normalized.notes).toEqual([
      { id: "note-1", document: NOTE_A, html: "<p>Alpha</p>" },
    ]);
    expect(normalized.activeId).toBe("note-1");
  });

  it("migrates a legacy HTML-only draft into Note 1", () => {
    const normalized = normalizeNotesCollection({
      html: "<p>Old notes</p>",
    });

    expect(normalized.notes).toEqual([
      { id: "note-1", document: null, html: "<p>Old notes</p>" },
    ]);
    expect(normalized.activeId).toBe("note-1");
    expect(normalized.pendingMigration).toBe(true);
  });

  it("wraps a bare CKEditor document into Note 1", () => {
    const normalized = normalizeNotesCollection(NOTE_A);

    expect(normalized.notes).toEqual([
      { id: "note-1", document: NOTE_A, html: "" },
    ]);
    expect(normalized.activeId).toBe("note-1");
    expect(normalized.pendingMigration).toBe(false);
  });

  it("keeps explicitly listed notes that are still empty", () => {
    const normalized = normalizeNotesCollection({
      format: NOTES_COLLECTION_FORMAT,
      version: 1,
      activeId: "note-b",
      notes: [
        { id: "note-a", document: NOTE_A, html: "<p>Alpha</p>" },
        { id: "note-b", document: null, html: "" },
      ],
    });

    expect(normalized.notes).toHaveLength(2);
    expect(normalized.notes[1]).toEqual({
      id: "note-b",
      document: null,
      html: "",
    });
    expect(normalized.activeId).toBe("note-b");
  });

  it("skips malformed note entries without dropping the rest", () => {    const normalized = normalizeNotesCollection({
      format: NOTES_COLLECTION_FORMAT,
      notes: [
        "nope",
        null,
        { id: "kept", document: NOTE_B, html: "" },
      ],
    });

    expect(normalized.notes).toEqual([
      { id: "kept", document: NOTE_B, html: "" },
    ]);
    expect(normalized.activeId).toBe("kept");
  });

  it("normalizes an id-less notes array using positional names", () => {
    const normalized = normalizeNotesCollection({
      notes: [
        { document: NOTE_A, html: "" },
        { document: NOTE_B, html: "" },
      ],
    });

    expect(normalized.notes.map((note) => note.id)).toEqual([
      "note-1",
      "note-2",
    ]);
  });

  it("returns nothing for unrecognised storage", () => {
    expect(normalizeNotesCollection(null)).toEqual({
      notes: [],
      activeId: null,
      pendingMigration: false,
    });
    expect(normalizeNotesCollection([1, 2, 3])).toEqual({
      notes: [],
      activeId: null,
      pendingMigration: false,
    });
    expect(normalizeNotesCollection({ blocks: "nope" })).toEqual({
      notes: [],
      activeId: null,
      pendingMigration: false,
    });
  });
});
