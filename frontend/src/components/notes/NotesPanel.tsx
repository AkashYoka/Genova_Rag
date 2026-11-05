"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Download,
  Eraser,
  FileDown,
  FileText,
  Pencil,
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
      className="flex flex-1 items-center justify-center text-xs text-[var(--gd-muted)]"
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
  const renameNote = useNotesStore((s) => s.renameNote);
  const closeNote = useNotesStore((s) => s.closeNote);
  const clear = useNotesStore((s) => s.clear);
  const setNotesOpen = useNotesStore((s) => s.setNotesOpen);
  const saving = useNotesStore((s) => s.saving);
  const loading = useNotesStore((s) => s.loading);
  const storage = useNotesStore((s) => s.storage);
  const error = useNotesStore((s) => s.error);
  const [dragOver, setDragOver] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const renameInputRef = useRef<HTMLInputElement>(null);

  const activeIndex = notes.findIndex((note) => note.id === activeNoteId);
  const activeNote = activeIndex >= 0 ? notes[activeIndex] : null;
  const html = activeNote?.html ?? "";

  useEffect(() => {
    if (editingId) renameInputRef.current?.focus();
  }, [editingId]);

  const startRename = (noteId: string, currentTitle: string) => {
    setEditingId(noteId);
    setEditTitle(currentTitle);
    selectNote(noteId);
  };

  const commitRename = () => {
    if (!editingId) return;
    renameNote(editingId, editTitle);
    setEditingId(null);
    setEditTitle("");
  };

  const cancelRename = () => {
    setEditingId(null);
    setEditTitle("");
  };

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
    if (exporting) return;
    setExporting(true);
    const toastId =
      kind === "pdf"
        ? toast.loading("Preparing ebook PDF…")
        : undefined;
    try {
      if (kind === "txt") exportNotesAsTxt(html);
      else if (kind === "md") exportNotesAsMarkdown(html);
      else {
        await exportNotesAsPdf(html, {
          title: activeNote?.title || "Genova Notes",
          noteId: activeNote?.id,
        });
      }
      if (toastId !== undefined) {
        toast.success("Notes PDF ready", { id: toastId });
      } else {
        toast.success(`Exported as ${kind.toUpperCase()}`);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Export failed";
      if (toastId !== undefined) toast.error(message, { id: toastId });
      else toast.error(message);
    } finally {
      setExporting(false);
    }
  };

  const statusLabel = saving
    ? "Saving…"
    : storage === "s3"
      ? "Synced to your private S3 folder"
      : storage === "unavailable"
        ? "Stored on this device"
        : error
          ? "Kept on this device — sync failed"
          : "Saved on this device";

  return (
    <div
      className="genova-notes flex h-full min-h-0 flex-1 flex-col bg-[var(--gd-surface)] text-[var(--gd-text)]"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      aria-label="Sticky notes"
    >
      <div className="genova-notes-chrome flex shrink-0 flex-col gap-2 border-b border-[var(--gd-border)] px-3 py-2.5">
        <div className="flex items-center justify-between gap-2">
          <div className="flex min-w-0 items-center gap-2">
            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md border border-[var(--gd-border)] bg-[var(--gd-raised)]">
              <StickyNote className="h-3.5 w-3.5 text-[var(--gd-muted)]" />
            </div>
            <div className="min-w-0">
              <p className="truncate text-[13px] font-semibold leading-tight tracking-tight">
                Notes
              </p>
              <p className="truncate text-[11px] text-[var(--gd-muted)]">
                {statusLabel}
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
                    className="h-7 w-7 text-[var(--gd-muted)] hover:bg-[var(--gd-raised)] hover:text-[var(--gd-text)]"
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
                  disabled={exporting}
                  onClick={() => void runExport("pdf")}
                >
                  <FileDown className="mr-2 h-3.5 w-3.5" />
                  {exporting ? "Preparing PDF…" : "Download PDF"}
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
              className="h-7 w-7 text-[var(--gd-muted)] hover:bg-[var(--gd-raised)] hover:text-[var(--gd-text)]"
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
              className="h-7 w-7 text-[var(--gd-muted)] hover:bg-[var(--gd-raised)] hover:text-[var(--gd-text)]"
              onClick={() => setNotesOpen(false)}
              aria-label="Close notes"
              title="Close notes"
            >
              <X className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>

        <div className="flex min-w-0 items-center gap-1">
          <div
            className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto pb-0.5"
            role="tablist"
            aria-label="Sticky note tabs"
          >
            {notes.map((note) => {
              const selected = note.id === activeNoteId;
              const isEditing = editingId === note.id;
              const label = note.title?.trim() || "Note";

              return (
                <div
                  key={note.id}
                  className={`group flex max-w-[11rem] shrink-0 items-center overflow-hidden rounded-lg border text-[12px] transition-colors ${
                    selected
                      ? "border-[var(--gd-border)] bg-[var(--gd-raised)] text-[var(--gd-text)]"
                      : "border-transparent text-[var(--gd-muted)] hover:bg-[var(--gd-raised)] hover:text-[var(--gd-text)]"
                  }`}
                >
                  {isEditing ? (
                    <form
                      className="flex min-w-0 flex-1 items-center"
                      onSubmit={(e) => {
                        e.preventDefault();
                        commitRename();
                      }}
                      onClick={(e) => e.stopPropagation()}
                    >
                      <input
                        ref={renameInputRef}
                        value={editTitle}
                        onChange={(e) => setEditTitle(e.target.value)}
                        onBlur={commitRename}
                        onKeyDown={(e) => {
                          if (e.key === "Escape") {
                            e.preventDefault();
                            cancelRename();
                          }
                        }}
                        maxLength={48}
                        aria-label="Rename note"
                        className="min-w-0 flex-1 bg-transparent px-2 py-1 text-[12px] font-medium text-[var(--gd-text)] outline-none"
                      />
                    </form>
                  ) : (
                    <button
                      type="button"
                      role="tab"
                      aria-selected={selected}
                      title={`${label} — double-click to rename`}
                      onClick={() => selectNote(note.id)}
                      onDoubleClick={() => startRename(note.id, label)}
                      className="min-w-0 cursor-pointer truncate px-2.5 py-1.5 font-medium"
                    >
                      {label}
                    </button>
                  )}
                  {!isEditing && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        startRename(note.id, label);
                      }}
                      aria-label={`Rename ${label}`}
                      title={`Rename ${label}`}
                      className="hidden shrink-0 px-1 py-1.5 text-[var(--gd-muted)] opacity-0 transition-opacity hover:text-[var(--gd-text)] group-hover:inline-flex group-hover:opacity-100 group-focus-within:inline-flex group-focus-within:opacity-100"
                    >
                      <Pencil className="h-3 w-3" />
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => {
                      if (isEditing) cancelRename();
                      closeNote(note.id);
                    }}
                    aria-label={`Close ${label}`}
                    title={`Close ${label}`}
                    className="shrink-0 cursor-pointer px-1.5 py-1.5 text-[var(--gd-muted)] hover:text-[var(--gd-text)]"
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
            className="h-7 w-7 shrink-0 text-[var(--gd-muted)] hover:bg-[var(--gd-raised)] hover:text-[var(--gd-text)]"
            onClick={() => addNote()}
            aria-label="Create a new note"
            title="New note"
          >
            <Plus className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      <div
        className={`relative flex min-h-0 flex-1 flex-col bg-[var(--gd-surface)] ${
          dragOver
            ? "bg-[color-mix(in_oklab,var(--gd-accent)_8%,transparent)]"
            : ""
        }`}
      >
        {dragOver ? (
          <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center border-2 border-dashed border-[var(--gd-accent)] bg-[color-mix(in_oklab,var(--gd-accent)_6%,var(--gd-surface))] text-sm font-medium text-[var(--gd-text)]">
            Drop message into notes
          </div>
        ) : null}

        {loading ? (
          <div
            className="flex flex-1 items-center justify-center text-xs text-[var(--gd-muted)]"
            aria-busy="true"
          >
            Loading notes…
          </div>
        ) : !activeNote ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-3 px-6 text-center">
            <div className="flex h-10 w-10 items-center justify-center rounded-full border border-[var(--gd-border)] bg-[var(--gd-raised)]">
              <StickyNote className="h-4 w-4 text-[var(--gd-muted)]" />
            </div>
            <div>
              <p className="text-[13px] font-medium text-[var(--gd-text)]">
                No notes open
              </p>
              <p className="mt-1 text-[11px] text-[var(--gd-muted)]">
                Create a note or add a chat message to start writing.
              </p>
            </div>
            <Button
              size="sm"
              className="text-xs"
              onClick={() => addNote()}
            >
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
