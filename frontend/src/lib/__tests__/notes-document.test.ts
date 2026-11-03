import { describe, expect, it } from "vitest";
import {
  NOTE_DOCUMENT_FORMAT,
  NOTE_DOCUMENT_VERSION,
  createNoteDocument,
  isNoteModelNode,
  normalizeNoteDocument,
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
