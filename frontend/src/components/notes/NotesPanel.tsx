"use client";

import { useCallback, useState } from "react";
import {
  Download,
  Eraser,
  FileDown,
  FileText,
  Plus,
  StickyNote,
  X,
} from "lucide-react";
import { toast } from "sonner";
import dynamic from "next/dynamic";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  GENOVA_NOTE_MIME,
  decodeNotePayload,
} from "@/lib/notes-mime";
import {
  exportNotesAsMarkdown,
  exportNotesAsPdf,
  exportNotesAsTxt,
} from "@/lib/notes-export";
import { useNotesStore } from "@/store/notes-store";

const NotesEditor = dynamic(() => import("./NotesEditor"), {
  ssr: false,
  loading: () => (
    <div
      className="flex flex-1 items-center justify-center text-xs text-muted-foreground"
      aria-busy="true"
    >
      Loading editor…
    </div>
  ),
});

export default function NotesPanel() {
  const notes = useNotesStore((s) => s.notes);
  const activeNoteId = useNotesStore((s) => s.activeNoteId);
  const pendingAppends = useNotesStore((s) => s.pendingAppends);
  const clearToken = useNotesStore((s) => s.clearToken);
  const commitDocument = useNotesStore((s) => s.commitDocument);
  const syncDocument = useNotesStore((s) => s.syncDocument);
  const appendsHandled = useNotesStore((s) => s.appendsHandled);
  const appendMessage = useNotesStore((s) => s.appendMessage);
  const addNote = useNotesStore((s) => s.addNote);
  const selectNote = useNotesStore((s) => s.selectNote);
  const closeNote = useNotesStore((s) => s.closeNote);
  const clear = useNotesStore((s) => s.clear);
  const setNotesOpen = useNotesStore((s) => s.setNotesOpen);
  const saving = useNotesStore((s) => s.saving);
  const loading = useNotesStore((s) => s.loading);
  const storage = useNotesStore((s) => s.storage);
  const error = useNotesStore((s) => s.error);
  const [dragOver, setDragOver] = useState(false);
  const [exporting, setExporting] = useState(false);

  const activeIndex = notes.findIndex((note) => note.id === activeNoteId);
  const activeNote = activeIndex >= 0 ? notes[activeIndex] : null;
  const html = activeNote?.html ?? "";

  const handleDragOver = useCallback((e: React.DragEvent) => {
    if (
      e.dataTransfer.types.includes(GENOVA_NOTE_MIME) ||
      e.dataTransfer.types.includes("text/plain")
    ) {
      e.preventDefault();
      e.dataTransfer.dropEffect = "copy";
      setDragOver(true);
    }
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    if (e.currentTarget.contains(e.relatedTarget as Node)) return;
    setDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const raw =
        e.dataTransfer.getData(GENOVA_NOTE_MIME) ||
        e.dataTransfer.getData("text/plain");
      if (!raw) return;

      const payload = decodeNotePayload(raw);
      if (payload) {
        appendMessage(payload);
        toast.success("Added to notes");
        return;
      }

      // Plain text fallback
      const text = raw.trim();
      if (!text) return;
      appendMessage({
        id: `drop-${Date.now()}`,
        role: "assistant",
        content: text,
      });
      toast.success("Added to notes");
    },
    [appendMessage],
  );

  const runExport = async (kind: "pdf" | "txt" | "md") => {
    if (!html.trim()) {
      toast.message("Notes are empty");
      return;
    }
    setExporting(true);
    try {
      if (kind === "txt") exportNotesAsTxt(html);
      else if (kind === "md") exportNotesAsMarkdown(html);
      else await exportNotesAsPdf(html);
      toast.success(`Exported as ${kind.toUpperCase()}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Export failed";
      toast.error(message);
    } finally {
      setExporting(false);
    }
  };

  return (
    <div
      className="flex h-full min-h-0 flex-1 flex-col"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      aria-label="Sticky notes"
    >
      <div className="flex shrink-0 flex-col gap-1.5 border-b border-[var(--gd-border,#e2e5ea)] px-3 py-2">
        <div className="flex items-center justify-between gap-2">
          <div className="flex min-w-0 items-center gap-2">
            <StickyNote className="h-3.5 w-3.5 shrink-0 text-[var(--gd-muted,#6b7280)]" />
            <div className="min-w-0">
              <p className="truncate text-[13px] font-medium leading-tight">
                Notes
              </p>
              <p className="truncate text-[10px] text-muted-foreground">
                {saving
                  ? "Saving…"
                  : storage === "s3"
                    ? "Synced to your private S3 folder"
                    : storage === "unavailable"
                      ? "Stored on this device"
                      : error
                        ? "Kept on this device — sync failed"
                        : "Saved on this device"}
              </p>
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-0.5">
            <DropdownMenu>
              <DropdownMenuTrigger
                render={
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7"
                    disabled={exporting}
                    aria-label="Export notes"
                    title="Export notes"
                  />
                }
              >
                <Download className="h-3.5 w-3.5" />
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="min-w-44">
                <DropdownMenuItem
                  className="cursor-pointer"
                  onClick={() => void runExport("pdf")}
                >
                  <FileDown className="mr-2 h-3.5 w-3.5" />
                  Download PDF
                </DropdownMenuItem>
                <DropdownMenuItem
                  className="cursor-pointer"
                  onClick={() => void runExport("txt")}
                >
                  <FileText className="mr-2 h-3.5 w-3.5" />
                  Download TXT
                </DropdownMenuItem>
                <DropdownMenuItem
                  className="cursor-pointer"
                  onClick={() => void runExport("md")}
                >
                  <FileText className="mr-2 h-3.5 w-3.5" />
                  Download Markdown
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={() => {
                if (!html.trim()) return;
                clear();
                toast.message("Note cleared");
              }}
              aria-label="Clear this note"
              title="Clear this note"
            >
              <Eraser className="h-3.5 w-3.5" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={() => setNotesOpen(false)}
              aria-label="Close notes"
              title="Close notes"
            >
              <X className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>

        {/* Note tabs — one independent note per tab. */}
        <div className="flex min-w-0 items-center gap-1">
          <div
            className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto pb-0.5"
            role="tablist"
            aria-label="Sticky note tabs"
          >
            {notes.map((note, index) => {
              const selected = note.id === activeNoteId;
              return (
                <div
                  key={note.id}
                  className={`flex shrink-0 items-center overflow-hidden rounded-md border text-[11px] transition-colors ${
                    selected
                      ? "border-[var(--gd-data,#0F7C86)]/40 bg-[var(--gd-raised,#eceef2)] text-[var(--gd-text,#14171c)]"
                      : "border-[var(--gd-border,#e2e5ea)] text-[var(--gd-muted,#5c636c)] hover:bg-[var(--gd-raised,#eceef2)]"
                  }`}
                >
                  <button
                    type="button"
                    role="tab"
                    aria-selected={selected}
                    title={`Note ${index + 1}`}
                    onClick={() => selectNote(note.id)}
                    className="cursor-pointer px-2 py-1 font-medium"
                  >
                    Note {index + 1}
                  </button>
                  <button
                    type="button"
                    onClick={() => closeNote(note.id)}
                    aria-label={`Close Note ${index + 1}`}
                    title={`Close Note ${index + 1}`}
                    className="cursor-pointer px-1 py-1 text-[var(--gd-muted,#6b7280)] hover:bg-black/5 hover:text-[var(--gd-text,#14171c)]"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </div>
              );
            })}
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 shrink-0"
            onClick={() => addNote()}
            aria-label="Create a new note"
            title="New note"
          >
            <Plus className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      <div
        className={`relative flex min-h-0 flex-1 flex-col ${
          dragOver
            ? "bg-[color-mix(in_oklab,var(--gd-data,#0F7C86)_8%,transparent)]"
            : ""
        }`}
      >
        {dragOver ? (
          <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center border-2 border-dashed border-[var(--gd-data,#0F7C86)] bg-[color-mix(in_oklab,var(--gd-data,#0F7C86)_6%,var(--gd-surface,#fff))] text-sm font-medium text-[var(--gd-text,#14171c)]">
            Drop message into notes
          </div>
        ) : null}

        {loading ? (
          <div className="flex flex-1 items-center justify-center text-xs text-muted-foreground" aria-busy="true">
            Loading notes…
          </div>
        ) : !activeNote ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-3 px-6 text-center">
            <StickyNote className="h-7 w-7 text-[var(--gd-muted,#6b7280)]" />
            <div>
              <p className="text-[13px] font-medium">No notes open</p>
              <p className="mt-1 text-[11px] text-muted-foreground">
                Create a note or add a chat message to start writing.
              </p>
            </div>
            <Button size="sm" className="text-xs" onClick={() => addNote()}>
              <Plus className="mr-1 h-3.5 w-3.5" />
              New note
            </Button>
          </div>
        ) : (
          <NotesEditor
            noteId={activeNote.id}
            document={activeNote.document}
            html={activeNote.html}
            pendingAppends={pendingAppends}
            clearToken={clearToken}
            onDocumentChange={commitDocument}
            onDocumentReady={syncDocument}
            onAppendsHandled={appendsHandled}
          />
        )}
      </div>
    </div>
  );
}
