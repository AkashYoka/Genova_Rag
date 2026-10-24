"use client";

import { useCallback, useState } from "react";
import {
  Download,
  Eraser,
  FileDown,
  FileText,
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
  const html = useNotesStore((s) => s.html);
  const setHtml = useNotesStore((s) => s.setHtml);
  const appendMessage = useNotesStore((s) => s.appendMessage);
  const clear = useNotesStore((s) => s.clear);
  const setNotesOpen = useNotesStore((s) => s.setNotesOpen);
  const [dragOver, setDragOver] = useState(false);
  const [exporting, setExporting] = useState(false);

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
      aria-label="Shared notes"
    >
      <div className="flex shrink-0 items-center justify-between gap-2 border-b border-[var(--gd-border,#e2e5ea)] px-3 py-2">
        <div className="flex min-w-0 items-center gap-2">
          <StickyNote className="h-3.5 w-3.5 shrink-0 text-[var(--gd-muted,#6b7280)]" />
          <div className="min-w-0">
            <p className="truncate text-[13px] font-medium leading-tight">
              Notes
            </p>
            <p className="truncate text-[10px] text-muted-foreground">
              Shared across Chat 1 &amp; Chat 2
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
              toast.message("Notes cleared");
            }}
            aria-label="Clear notes"
            title="Clear notes"
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
        <NotesEditor html={html} onChange={setHtml} />
      </div>
    </div>
  );
}
