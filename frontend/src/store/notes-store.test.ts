/**
 * Tests for the multi-note sticky-notes store.
 *
 * The S3/API layer is mocked: these tests cover the in-memory note collection
 * and the local draft that mirrors it, which is what makes notes independent.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { act } from "@testing-library/react";

vi.mock("@/lib/api", () => ({
  api: {
    get: vi.fn(),
    put: vi.fn(),
    post: vi.fn(),
    patch: vi.fn(),
  },
}));

vi.mock("sonner", () => ({
  toast: {
    success: vi.fn(),
    error: vi.fn(),
    message: vi.fn(),
  },
}));

import { api } from "@/lib/api";
import {
  createNoteDocument,
  createNotesEnvelope,
} from "@/lib/notes-document";
import { useNotesStore } from "./notes-store";

const DRAFT_KEY = "genova-notes-draft:user-1";

const NOTE_A = createNoteDocument([
  { name: "paragraph", children: [{ data: "Alpha" }] },
]);
const NOTE_B = createNoteDocument([
  { name: "paragraph", children: [{ data: "Beta" }] },
]);
const NOTE_C = createNoteDocument([
  { name: "paragraph", children: [{ data: "Gamma" }] },
]);

function envelope(
  notes: Array<{ id: string; title?: string; document: unknown; html: string }>,
  activeId: string | null,
) {
  return createNotesEnvelope(
    notes.map((note, index) => ({
      id: note.id,
      title: note.title ?? `Note ${index + 1}`,
      document: note.document as Record<string, unknown> | null,
      html: note.html,
    })),
    activeId,
  );
}

async function load(noteEnvelope: Record<string, unknown> | null) {
  vi.mocked(api.get).mockResolvedValue({
    notes: noteEnvelope,
    exists: noteEnvelope !== null,
    updated_at: "2026-01-01T00:00:00+00:00",
    storage: "s3",
  });

  await act(async () => {
    await useNotesStore.getState().loadNotes("user-1");
  });
}

function draft() {
  const raw = window.localStorage.getItem(DRAFT_KEY);
  return raw ? JSON.parse(raw) : null;
}

beforeEach(() => {
  useNotesStore.getState().reset();
  window.localStorage.clear();
});

describe("notes-store — multiple notes", () => {
  it("loads every note and the active note from S3", async () => {
    await load(
      envelope(
        [
          { id: "note-a", document: NOTE_A, html: "<p>Alpha</p>" },
          { id: "note-b", document: NOTE_B, html: "<p>Beta</p>" },
        ],
        "note-b",
      ),
    );

    const state = useNotesStore.getState();
    expect(state.notes.map((note) => note.id)).toEqual(["note-a", "note-b"]);
    expect(state.notes[0].html).toBe("<p>Alpha</p>");
    expect(state.notes[1].html).toBe("<p>Beta</p>");
    expect(state.activeNoteId).toBe("note-b");
    expect(state.loaded).toBe(true);
  });

  it("does not replace an explicitly empty collection with a new note", async () => {
    await load(envelope([], null));

    const state = useNotesStore.getState();
    expect(state.notes).toEqual([]);
    expect(state.activeNoteId).toBe("");
  });

  it("creates a new independent note and selects it", async () => {
    await load(
      envelope(
        [{ id: "note-a", document: NOTE_A, html: "<p>Alpha</p>" }],
        "note-a",
      ),
    );

    let createdId = "";
    act(() => {
      createdId = useNotesStore.getState().addNote();
    });

    const state = useNotesStore.getState();
    expect(state.notes).toHaveLength(2);
    expect(state.notes[0]).toEqual({
      id: "note-a",
      title: "Note 1",
      document: NOTE_A,
      html: "<p>Alpha</p>",
    });
    expect(state.notes[1].id).toBe(createdId);
    expect(state.notes[1].title).toBe("Note 2");
    expect(state.notes[1].html).toBe("");
    expect(state.activeNoteId).toBe(createdId);

    const stored = draft();
    expect(stored.notes).toHaveLength(2);
    expect(stored.activeId).toBe(createdId);
  });

  it("restores an empty note created before any typing", async () => {
    await load(
      envelope(
        [{ id: "note-a", document: NOTE_A, html: "<p>Alpha</p>" }],
        "note-a",
      ),
    );

    let createdId = "";
    act(() => {
      createdId = useNotesStore.getState().addNote();
    });

    const saved = draft();
    useNotesStore.getState().reset();
    await load(saved);

    const state = useNotesStore.getState();
    expect(state.notes.map((note) => note.id)).toEqual([
      "note-a",
      createdId,
    ]);
    expect(state.activeNoteId).toBe(createdId);
  });

  it("keeps a queued chat message while notes load", async () => {
    act(() => {
      useNotesStore.getState().appendMessage({
        id: "msg-1",
        role: "user",
        content: "Summarize the chemistry results",
      });
    });

    expect(useNotesStore.getState().pendingAppends).toHaveLength(1);

    await load(envelope([], null));

    const state = useNotesStore.getState();
    expect(state.pendingAppends).toHaveLength(1);
    expect(state.pendingAppends[0].content).toBe(
      "Summarize the chemistry results",
    );
    expect(state.notes).toHaveLength(1);
    expect(state.activeNoteId).toBe(state.notes[0].id);
    expect(state.notesOpen).toBe(true);
  });

  it("switching notes keeps each note's content", async () => {
    await load(
      envelope(
        [
          { id: "note-a", document: NOTE_A, html: "<p>Alpha</p>" },
          { id: "note-b", document: NOTE_B, html: "<p>Beta</p>" },
        ],
        "note-a",
      ),
    );

    act(() => {
      useNotesStore.getState().selectNote("note-b");
    });
    expect(useNotesStore.getState().activeNoteId).toBe("note-b");

    act(() => {
      useNotesStore.getState().commitDocument(NOTE_C, "<p>Gamma</p>");
    });

    const notes = useNotesStore.getState().notes;
    expect(notes[0]).toEqual({
      id: "note-a",
      title: "Note 1",
      document: NOTE_A,
      html: "<p>Alpha</p>",
    });
    expect(notes[1]).toEqual({
      id: "note-b",
      title: "Note 2",
      document: NOTE_C,
      html: "<p>Gamma</p>",
    });

    act(() => {
      useNotesStore.getState().selectNote("note-a");
    });

    // Note A never received Note B's edits.
    expect(useNotesStore.getState().notes[0].html).toBe("<p>Alpha</p>");
    const stored = draft();
    expect(stored.notes[0].html).toBe("<p>Alpha</p>");
    expect(stored.notes[1].html).toBe("<p>Gamma</p>");
  });

  it("closing a note leaves the others untouched and picks a neighbour", async () => {
    await load(
      envelope(
        [
          { id: "note-a", document: NOTE_A, html: "<p>Alpha</p>" },
          { id: "note-b", document: NOTE_B, html: "<p>Beta</p>" },
          { id: "note-c", document: NOTE_C, html: "<p>Gamma</p>" },
        ],
        "note-b",
      ),
    );

    act(() => {
      useNotesStore.getState().closeNote("note-b");
    });

    let state = useNotesStore.getState();
    expect(state.notes.map((note) => note.id)).toEqual(["note-a", "note-c"]);
    expect(state.notes.map((note) => note.html)).toEqual([
      "<p>Alpha</p>",
      "<p>Gamma</p>",
    ]);
    expect(state.activeNoteId).toBe("note-c");

    act(() => {
      useNotesStore.getState().closeNote("note-a");
      useNotesStore.getState().closeNote("note-c");
    });

    state = useNotesStore.getState();
    expect(state.notes).toEqual([]);
    expect(state.activeNoteId).toBe("");
    expect(draft().notes).toEqual([]);
  });

  it("closing a background note keeps the active note", async () => {
    await load(
      envelope(
        [
          { id: "note-a", document: NOTE_A, html: "<p>Alpha</p>" },
          { id: "note-b", document: NOTE_B, html: "<p>Beta</p>" },
        ],
        "note-b",
      ),
    );

    act(() => {
      useNotesStore.getState().closeNote("note-a");
    });

    const state = useNotesStore.getState();
    expect(state.notes.map((note) => note.id)).toEqual(["note-b"]);
    expect(state.activeNoteId).toBe("note-b");
  });

  it("falls back to a local draft when the cloud has no copy", async () => {
    window.localStorage.setItem(
      DRAFT_KEY,
      JSON.stringify(
        envelope(
          [
            { id: "local-a", document: NOTE_A, html: "<p>Alpha</p>" },
            { id: "local-b", document: NOTE_B, html: "<p>Beta</p>" },
          ],
          "local-b",
        ),
      ),
    );

    await load(null);

    const state = useNotesStore.getState();
    expect(state.notes.map((note) => note.id)).toEqual(["local-a", "local-b"]);
    expect(state.activeNoteId).toBe("local-b");
  });

  it("migrates the legacy HTML draft into a single Note 1", async () => {
    window.localStorage.setItem(
      "genova-shared-notes",
      JSON.stringify({ state: { html: "<p>Legacy body</p>" } }),
    );

    await load(null);

    const state = useNotesStore.getState();
    expect(state.notes).toEqual([
      {
        id: "note-1",
        title: "Note 1",
        document: null,
        html: "<p>Legacy body</p>",
      },
    ]);
    expect(state.activeNoteId).toBe("note-1");
    expect(state.pendingMigration).toBe(true);
  });

  it("renames a note and persists the title", async () => {
    await load(
      envelope(
        [{ id: "note-a", document: NOTE_A, html: "<p>Alpha</p>" }],
        "note-a",
      ),
    );

    act(() => {
      useNotesStore.getState().renameNote("note-a", "Chemistry");
    });

    expect(useNotesStore.getState().notes[0].title).toBe("Chemistry");
    expect(draft().notes[0].title).toBe("Chemistry");
  });
});
