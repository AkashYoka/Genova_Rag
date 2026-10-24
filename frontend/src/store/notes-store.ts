"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";
import { messageToNoteHtml, type GenovaNotePayload } from "@/lib/notes-mime";

interface NotesStore {
  /** Shared HTML document — not keyed by chat session */
  html: string;
  notesOpen: boolean;
  /** Bumped when content is added so the desk can focus the Notes tab */
  focusToken: number;
  setHtml: (html: string) => void;
  appendMessage: (payload: GenovaNotePayload) => void;
  clear: () => void;
  setNotesOpen: (open: boolean) => void;
  openNotes: () => void;
}

export const useNotesStore = create<NotesStore>()(
  persist(
    (set, get) => ({
      html: "",
      notesOpen: false,
      focusToken: 0,
      setHtml: (html) => set({ html }),
      appendMessage: (payload) => {
        const block = messageToNoteHtml(payload);
        const current = get().html?.trim() ?? "";
        set({
          html: current ? `${current}${block}` : block,
          notesOpen: true,
          focusToken: get().focusToken + 1,
        });
      },
      clear: () => set({ html: "" }),
      setNotesOpen: (open) => set({ notesOpen: open }),
      openNotes: () =>
        set((s) => ({ notesOpen: true, focusToken: s.focusToken + 1 })),
    }),
    {
      name: "genova-shared-notes",
      // Persist document content only — panel starts closed each session
      partialize: (state) => ({ html: state.html }),
      merge: (persisted, current) => ({
        ...current,
        ...(persisted as Partial<NotesStore>),
        notesOpen: false,
        focusToken: 0,
      }),
    },
  ),
);
