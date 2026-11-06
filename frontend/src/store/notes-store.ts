"use client";

import { create } from "zustand";
import { toast } from "sonner";
import { api } from "@/lib/api";
import type { GenovaNotePayload } from "@/lib/notes-mime";
import {
  createNotesEnvelope,
  isNotesCollection,
  normalizeNotesCollection,
  type NormalizedNotesCollection,
  type StickyNoteRecord,
} from "@/lib/notes-document";

/** CKEditor 5 data document (the object form of `editor.data.get()`). */
export type CkEditorDocument = Record<string, unknown>;
export type NotesStorage = "s3" | "local" | "unavailable";

/** One independent sticky note. */
export interface StickyNote {
  id: string;
  title: string;
  document: CkEditorDocument | null;
  html: string;
  /** PDF/document this note is associated with (persisted in notes.json). */
  documentId?: string | null;
  documentName?: string | null;
}

const DRAFT_KEY_PREFIX = "genova-notes-draft:";
const LEGACY_STORAGE_KEY = "genova-shared-notes";
const SAVE_DEBOUNCE_MS = 900;

interface NotesResponse {
  notes: CkEditorDocument | null;
  exists: boolean;
  updated_at: string | null;
  storage: NotesStorage;
}

interface NotesStore {
  /** Every sticky note, in tab order (Note 1, Note 2, …). */
  notes: StickyNote[];
  /** Note currently shown in the editor. Empty when all notes are closed. */
  activeNoteId: string;
  notesOpen: boolean;
  /** Bumped when content is added so the desk can focus the Notes tab */
  focusToken: number;
  /** Bumped to ask the editor for an empty document. */
  clearToken: number;
  /** Chat messages waiting for the editor to append them. */
  pendingAppends: GenovaNotePayload[];
  /** Owner of the currently loaded notes — guards cross-user leakage. */
  userId: string | null;
  storage: NotesStorage;
  loaded: boolean;
  loading: boolean;
  saving: boolean;
  lastSavedAt: string | null;
  error: string | null;
  /** Legacy HTML notes need one save to become CKEditor JSON. */
  pendingMigration: boolean;
  /** Desk PDF currently linked to new / active notes. */
  associatedDocumentId: string | null;
  associatedDocumentName: string | null;
  loadNotes: (userId: string) => Promise<void>;
  commitDocument: (document: CkEditorDocument, html: string) => void;
  syncDocument: (document: CkEditorDocument, html: string) => void;
  appendMessage: (payload: GenovaNotePayload) => void;
  appendsHandled: () => void;
  addNote: () => string;
  selectNote: (noteId: string) => void;
  renameNote: (noteId: string, title: string) => void;
  associateDocument: (
    documentId: string | null,
    documentName?: string | null,
  ) => void;
  closeNote: (noteId: string) => void;
  clear: () => void;
  setNotesOpen: (open: boolean) => void;
  openNotes: () => void;
  reset: () => void;
}

const initialState = {
  notes: [] as StickyNote[],
  activeNoteId: "",
  notesOpen: false,
  focusToken: 0,
  clearToken: 0,
  pendingAppends: [] as GenovaNotePayload[],
  userId: null as string | null,
  storage: "local" as NotesStorage,
  loaded: false,
  loading: false,
  saving: false,
  lastSavedAt: null as string | null,
  error: null as string | null,
  pendingMigration: false,
  associatedDocumentId: null as string | null,
  associatedDocumentName: null as string | null,
};

let saveTimer: ReturnType<typeof setTimeout> | null = null;
let noteSequence = 0;

function cancelScheduledSave() {
  if (saveTimer !== null) {
    clearTimeout(saveTimer);
    saveTimer = null;
  }
}

function createNoteId(): string {
  noteSequence += 1;
  return `note-${Date.now().toString(36)}-${noteSequence}`;
}

function createEmptyNote(
  id = createNoteId(),
  title = "Note",
  documentId: string | null = null,
  documentName: string | null = null,
): StickyNote {
  return {
    id,
    title,
    document: null,
    html: "",
    ...(documentId ? { documentId, documentName } : {}),
  };
}

function readDraft(userId: string): NormalizedNotesCollection | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(DRAFT_KEY_PREFIX + userId);
    if (!raw) return null;
    return normalizeNotesCollection(JSON.parse(raw));
  } catch {
    return null;
  }
}

function writeDraft(
  userId: string | null,
  notes: StickyNoteRecord[],
  activeNoteId: string,
) {
  if (typeof window === "undefined" || !userId) return;
  try {
    window.localStorage.setItem(
      DRAFT_KEY_PREFIX + userId,
      JSON.stringify({
        ...createNotesEnvelope(notes, activeNoteId || null),
        savedAt: new Date().toISOString(),
      }),
    );
  } catch {
    // Quota exceeded / private mode — the S3 copy is still authoritative.
  }
}

/** Notes written before S3 persistence lived in zustand's localStorage key. */
function readLegacyHtml(): string {
  if (typeof window === "undefined") return "";
  try {
    const raw = window.localStorage.getItem(LEGACY_STORAGE_KEY);
    if (!raw) return "";
    const parsed = JSON.parse(raw) as { state?: { html?: string } };
    return typeof parsed?.state?.html === "string" ? parsed.state.html : "";
  } catch {
    return "";
  }
}

function clearLegacyHtml() {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(LEGACY_STORAGE_KEY);
  } catch {
    // ignore
  }
}

function asRecords(notes: StickyNote[]): StickyNoteRecord[] {
  return notes.map((note) => ({
    id: note.id,
    title: note.title,
    document: note.document,
    html: note.html,
    ...(note.documentId
      ? { documentId: note.documentId, documentName: note.documentName ?? null }
      : {}),
  }));
}

function nextNoteTitle(notes: StickyNote[]): string {
  let max = 0;
  for (const note of notes) {
    const match = /^Note\s+(\d+)$/i.exec(note.title.trim());
    if (match) max = Math.max(max, Number(match[1]));
  }
  return `Note ${max + 1}`;
}

function updateActiveNote(
  state: NotesStore,
  patch: Partial<Pick<StickyNote, "document" | "html" | "title">>,
): Pick<NotesStore, "notes"> {
  let changed = false;
  const notes = state.notes.map((note) => {
    if (note.id !== state.activeNoteId) return note;
    changed = true;
    return { ...note, ...patch };
  });
  return { notes: changed ? notes : state.notes };
}

export const useNotesStore = create<NotesStore>((set, get) => {
  const persist = async () => {
    const { userId, notes, activeNoteId, storage } = get();
    if (!userId) return;

    writeDraft(userId, asRecords(notes), activeNoteId);

    // S3 not configured (local dev): keep the draft on this device only.
    if (storage === "unavailable") return;

    set({ saving: true, error: null });
    try {
      const response = await api.put<{ saved: boolean; updated_at: string | null }>(
        "/api/v1/notes",
        { notes: createNotesEnvelope(asRecords(notes), activeNoteId || null) },
      );
      clearLegacyHtml();
      set({
        saving: false,
        storage: "s3",
        lastSavedAt: response?.updated_at ?? new Date().toISOString(),
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Could not save notes";
      const unavailable = /not configured/i.test(message);
      set({
        saving: false,
        error: message,
        storage: unavailable ? "unavailable" : "local",
      });
      if (!unavailable) {
        toast.error("Notes kept locally — cloud sync failed", {
          description: message,
        });
      }
    }
  };

  const scheduleSave = () => {
    const { userId, notes, activeNoteId } = get();
    writeDraft(userId, asRecords(notes), activeNoteId);
    cancelScheduledSave();
    saveTimer = setTimeout(() => {
      saveTimer = null;
      void persist();
    }, SAVE_DEBOUNCE_MS);
  };

  return {
    ...initialState,

    async loadNotes(userId) {
      if (get().userId === userId && get().loaded) return;
      cancelScheduledSave();
      // A chat message can be queued for notes while the initial load is still
      // in flight (e.g. the notes panel was closed when Add to notes was
      // pressed). Keep that queue and the panel state across the reset.
      const { pendingAppends, notesOpen, focusToken } = get();
      set({
        ...initialState,
        userId,
        loading: true,
        loaded: false,
        pendingAppends,
        notesOpen,
        focusToken,
      });

      try {
        const response = await api.get<NotesResponse>("/api/v1/notes");
        const storage = response.storage ?? "local";
        const cloud = normalizeNotesCollection(response.notes);
        const hasCollection = isNotesCollection(response.notes);

        // A saved collection (even an empty one) is authoritative; its `notes`
        // array may legitimately contain zero entries after closing them all.
        if (hasCollection) {
          set({
            notes: cloud.notes,
            activeNoteId: cloud.activeId ?? cloud.notes[0]?.id ?? "",
            storage,
            lastSavedAt: response.updated_at ?? null,
            pendingMigration: cloud.pendingMigration,
            loading: false,
            loaded: true,
          });
          return;
        }

        if (cloud.notes.length > 0) {
          set({
            notes: cloud.notes,
            activeNoteId: cloud.activeId ?? cloud.notes[0].id,
            storage,
            lastSavedAt: response.updated_at ?? null,
            pendingMigration: cloud.pendingMigration,
            loading: false,
            loaded: true,
          });
          return;
        }

        // No cloud copy yet: prefer a local draft, then migrate legacy notes.
        const draft = readDraft(userId);
        if (draft && draft.notes.length > 0) {
          set({
            notes: draft.notes,
            activeNoteId: draft.activeId ?? draft.notes[0].id,
            storage,
            pendingMigration: draft.pendingMigration,
            loading: false,
            loaded: true,
          });
          return;
        }

        const legacyHtml = readLegacyHtml();
        if (legacyHtml) {
          const note: StickyNote = {
            id: "note-1",
            title: "Note 1",
            document: null,
            html: legacyHtml,
          };
          set({
            notes: [note],
            activeNoteId: note.id,
            storage,
            pendingMigration: true,
            loading: false,
            loaded: true,
          });
          return;
        }

        // Fresh account: start with one empty note (Note 1).
        const note = createEmptyNote("note-1", "Note 1");
        set({
          notes: [note],
          activeNoteId: note.id,
          storage,
          loading: false,
          loaded: true,
        });
      } catch (err) {
        // Offline / 5xx: keep the editor usable with whatever is on this device.
        const message =
          err instanceof Error ? err.message : "Could not load notes";
        const draft = readDraft(userId);
        const legacyHtml = draft?.notes.length ? "" : readLegacyHtml();
        const notes =
          draft && draft.notes.length > 0
            ? draft.notes
            : legacyHtml
              ? [
                  {
                    id: "note-1",
                    title: "Note 1",
                    document: null,
                    html: legacyHtml,
                  } satisfies StickyNote,
                ]
              : [createEmptyNote("note-1", "Note 1")];

        set({
          notes,
          activeNoteId: draft?.activeId ?? notes[0]?.id ?? "",
          storage: "local",
          pendingMigration: legacyHtml ? true : (draft?.pendingMigration ?? false),
          loading: false,
          loaded: true,
          error: message,
        });
        toast.message("Notes loaded from this device", {
          description: message,
        });
      } finally {
        // A message can be queued for notes while the cloud/local copy is still
        // loading. If that copy has no notes (all were closed), keep a fresh
        // note so the editor can consume the queued append.
        const current = get();
        if (current.pendingAppends.length > 0 && current.notes.length === 0) {
          const note = createEmptyNote(
            "note-1",
            "Note 1",
            current.associatedDocumentId,
            current.associatedDocumentName,
          );
          set({ notes: [note], activeNoteId: note.id });
        }
      }
    },

    commitDocument(document, html) {
      set((state) => updateActiveNote(state, { document, html }));
      scheduleSave();
    },

    syncDocument(document, html) {
      const { loading, pendingMigration, userId, notes, activeNoteId } = get();
      // The initial S3/local load is still in flight: don't clobber the stored
      // draft with the empty editor that mounted before the data arrived.
      if (loading) return;
      set((state) => updateActiveNote(state, { document, html }));

      const next = get().notes;
      if (pendingMigration) {
        // Legacy HTML just became structured JSON — persist it now.
        scheduleSave();
        // Once every note has a document, stop rewriting the file on each
        // note switch; ordinary edits persist through commitDocument.
        if (next.every((note) => note.document || !note.html)) {
          set({ pendingMigration: false });
        }
      } else {
        writeDraft(userId, asRecords(next.length ? next : notes), activeNoteId);
      }
    },

    appendMessage(payload) {
      set((state) => {
        const notes =
          state.notes.length > 0
            ? state.notes
            : [
                createEmptyNote(
                  "note-1",
                  "Note 1",
                  state.associatedDocumentId,
                  state.associatedDocumentName,
                ),
              ];
        return {
          notes,
          activeNoteId:
            state.notes.length > 0 ? state.activeNoteId : notes[0].id,
          pendingAppends: [...state.pendingAppends, payload],
          notesOpen: true,
          focusToken: state.focusToken + 1,
        };
      });
      scheduleSave();
    },

    appendsHandled() {
      set({ pendingAppends: [] });
    },

    addNote() {
      const { notes, associatedDocumentId, associatedDocumentName } = get();
      const note = createEmptyNote(
        undefined,
        nextNoteTitle(notes),
        associatedDocumentId,
        associatedDocumentName,
      );
      set((state) => ({
        notes: [...state.notes, note],
        activeNoteId: note.id,
        notesOpen: true,
        focusToken: state.focusToken + 1,
      }));
      scheduleSave();
      return note.id;
    },

    selectNote(noteId) {
      const { notes } = get();
      if (!notes.some((note) => note.id === noteId)) return;
      set((state) => ({
        activeNoteId: noteId,
        focusToken: state.focusToken + 1,
      }));
      // Keep the active tab in the persisted envelope as well as the draft.
      scheduleSave();
    },

    renameNote(noteId, title) {
      const nextTitle = title.trim() || "Note";
      const current = get().notes.find((note) => note.id === noteId);
      if (!current || current.title === nextTitle) return;
      set((state) => ({
        notes: state.notes.map((note) =>
          note.id === noteId ? { ...note, title: nextTitle } : note,
        ),
      }));
      scheduleSave();
    },

    associateDocument(documentId, documentName = null) {
      const nextId = documentId || null;
      const nextName = documentName || null;
      const state = get();
      if (
        state.associatedDocumentId === nextId &&
        state.associatedDocumentName === nextName
      ) {
        return;
      }
      set({
        associatedDocumentId: nextId,
        associatedDocumentName: nextName,
      });
      // Stamp the active note when the desk PDF changes so reload restores the link.
      if (state.activeNoteId && nextId) {
        set((s) => ({
          notes: s.notes.map((note) =>
            note.id === s.activeNoteId
              ? { ...note, documentId: nextId, documentName: nextName }
              : note,
          ),
        }));
        scheduleSave();
      }
    },

    closeNote(noteId) {
      set((state) => {
        const index = state.notes.findIndex((note) => note.id === noteId);
        if (index === -1) return {};
        const notes = state.notes.filter((note) => note.id !== noteId);
        const activeNoteId =
          state.activeNoteId === noteId
            ? (notes[Math.min(index, notes.length - 1)]?.id ?? "")
            : state.activeNoteId;
        return { notes, activeNoteId };
      });
      scheduleSave();
    },

    clear() {
      set((state) => ({ clearToken: state.clearToken + 1 }));
    },

    setNotesOpen(open) {
      set({ notesOpen: open });
    },

    openNotes() {
      set((state) => ({
        notesOpen: true,
        focusToken: state.focusToken + 1,
      }));
    },

    reset() {
      cancelScheduledSave();
      set({ ...initialState });
    },
  };
});
