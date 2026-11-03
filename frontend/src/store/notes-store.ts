"use client";

import { create } from "zustand";
import { toast } from "sonner";
import { api } from "@/lib/api";
import type { GenovaNotePayload } from "@/lib/notes-mime";
import { normalizeNoteDocument } from "@/lib/notes-document";

/** CKEditor 5 data document (the object form of `editor.data.get()`). */
export type CkEditorDocument = Record<string, unknown>;
export type NotesStorage = "s3" | "local" | "unavailable";

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
  /** Editor JSON — the persisted representation. */
  document: CkEditorDocument | null;
  /** HTML mirror of the editor, used for exports/empty checks only. */
  html: string;
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
  /** Legacy localStorage notes need one save to become CKEditor JSON. */
  pendingMigration: boolean;
  loadNotes: (userId: string) => Promise<void>;
  commitDocument: (document: CkEditorDocument, html: string) => void;
  syncDocument: (document: CkEditorDocument, html: string) => void;
  appendMessage: (payload: GenovaNotePayload) => void;
  appendsHandled: () => void;
  clear: () => void;
  setNotesOpen: (open: boolean) => void;
  openNotes: () => void;
  reset: () => void;
}

const initialState = {
  document: null as CkEditorDocument | null,
  html: "",
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
};

let saveTimer: ReturnType<typeof setTimeout> | null = null;

function cancelScheduledSave() {
  if (saveTimer !== null) {
    clearTimeout(saveTimer);
    saveTimer = null;
  }
}

function readDraft(userId: string): {
  document: CkEditorDocument | null;
  html: string;
} | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(DRAFT_KEY_PREFIX + userId);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { document?: unknown; html?: unknown };
    return {
      document:
        parsed.document && typeof parsed.document === "object"
          ? (parsed.document as CkEditorDocument)
          : null,
      html: typeof parsed.html === "string" ? parsed.html : "",
    };
  } catch {
    return null;
  }
}

function writeDraft(
  userId: string | null,
  document: CkEditorDocument | null,
  html: string,
) {
  if (typeof window === "undefined" || !userId) return;
  try {
    window.localStorage.setItem(
      DRAFT_KEY_PREFIX + userId,
      JSON.stringify({ document, html, savedAt: new Date().toISOString() }),
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

export const useNotesStore = create<NotesStore>((set, get) => {
  const persist = async () => {
    const { userId, document, html, storage } = get();
    if (!userId) return;

    writeDraft(userId, document, html);

    // S3 not configured (local dev): keep the draft on this device only.
    if (!document || storage === "unavailable") return;

    set({ saving: true, error: null });
    try {
      const response = await api.put<{ saved: boolean; updated_at: string | null }>(
        "/api/v1/notes",
        { notes: document },
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
    const { userId, document, html } = get();
    writeDraft(userId, document, html);
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
      set({
        userId,
        document: null,
        html: "",
        pendingAppends: [],
        clearToken: 0,
        loading: true,
        loaded: false,
        saving: false,
        error: null,
        lastSavedAt: null,
        pendingMigration: false,
      });

      try {
        const response = await api.get<NotesResponse>("/api/v1/notes");
        const storage = response.storage ?? "local";
        const restored = normalizeNoteDocument(response.notes);

        if (restored.document) {
          set({
            document: restored.document,
            html: restored.html ?? "",
            storage,
            lastSavedAt: response.updated_at ?? null,
            loading: false,
            loaded: true,
          });
          return;
        }

        if (restored.html) {
          // Cloud copy predates structured JSON: seed the editor with HTML and
          // let the first ready-sync rewrite it as model JSON.
          set({
            document: null,
            html: restored.html,
            storage,
            lastSavedAt: response.updated_at ?? null,
            pendingMigration: true,
            loading: false,
            loaded: true,
          });
          return;
        }

        // No cloud copy yet: prefer a local draft, then migrate legacy notes.
        const draft = readDraft(userId);
        const legacyHtml = draft?.html ? "" : readLegacyHtml();
        const htmlOnly = Boolean((draft?.html && !draft.document) || legacyHtml);
        set({
          document: draft?.document ?? null,
          html: draft?.html || legacyHtml,
          storage,
          pendingMigration: htmlOnly,
          loading: false,
          loaded: true,
        });
      } catch (err) {
        // Offline / 5xx: keep the editor usable with whatever is on this device.
        const message =
          err instanceof Error ? err.message : "Could not load notes";
        const draft = readDraft(userId);
        set({
          document: draft?.document ?? null,
          html: draft?.html ?? readLegacyHtml(),
          storage: "local",
          loading: false,
          loaded: true,
          error: message,
        });
        toast.message("Notes loaded from this device", {
          description: message,
        });
      }
    },

    commitDocument(document, html) {
      set({ document, html });
      scheduleSave();
    },

    syncDocument(document, html) {
      const { pendingMigration, userId, loading } = get();
      // The initial S3/local load is still in flight: don't clobber the stored
      // draft with the empty editor that mounted before the data arrived.
      if (loading) return;
      set({ document, html, pendingMigration: false });
      if (pendingMigration) {
        // Legacy HTML just became structured JSON — persist it now.
        scheduleSave();
      } else {
        writeDraft(userId, document, html);
      }
    },

    appendMessage(payload) {
      set((state) => ({
        pendingAppends: [...state.pendingAppends, payload],
        notesOpen: true,
        focusToken: state.focusToken + 1,
      }));
    },

    appendsHandled() {
      set({ pendingAppends: [] });
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
