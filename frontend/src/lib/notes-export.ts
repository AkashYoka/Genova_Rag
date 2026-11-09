import TurndownService from "turndown";
import { api } from "@/lib/api";

function stripHtml(html: string): string {
  // Prefer DOMParser over innerHTML assignment — avoids executing markup
  // (e.g. <img onerror=...>) while extracting plain text for export.
  if (typeof DOMParser !== "undefined") {
    const doc = new DOMParser().parseFromString(html, "text/html");
    return (doc.body?.textContent || "").replace(/\n{3,}/g, "\n\n").trim();
  }
  return html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
}

function downloadBlob(filename: string, blob: Blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function stamp(): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}`;
}

export function exportNotesAsTxt(html: string) {
  const text = stripHtml(html) || " ";
  downloadBlob(
    `genova-notes-${stamp()}.txt`,
    new Blob([text], { type: "text/plain;charset=utf-8" }),
  );
}

export function exportNotesAsMarkdown(html: string) {
  const turndown = new TurndownService({
    headingStyle: "atx",
    codeBlockStyle: "fenced",
  });
  const md = turndown.turndown(html || "<p></p>");
  downloadBlob(
    `genova-notes-${stamp()}.md`,
    new Blob([md], { type: "text/markdown;charset=utf-8" }),
  );
}

export type NotesPdfOptions = {
  title?: string;
  noteId?: string;
};

/**
 * Export CKEditor HTML via the Genova publishing backend (ReportLab ebook),
 * matching Chat PDF typography and page chrome.
 */
export async function exportNotesAsPdf(
  html: string,
  options: NotesPdfOptions = {},
) {
  // Cookie sessions authenticate via HttpOnly cookies on downloadAuthenticated.
  const hasCookieSession =
    typeof document !== "undefined" && document.cookie.includes("genova_auth=");
  if (!hasCookieSession && !api.getAccessToken()) {
    throw new Error("Sign in required to export notes as PDF");
  }
  if (!html?.trim()) {
    throw new Error("Notes are empty");
  }

  const { blob, filename } = await api.downloadAuthenticated(
    "/api/v1/notes/export/pdf",
    {
      method: "POST",
      body: {
        html,
        title: options.title || undefined,
        note_id: options.noteId || undefined,
      },
    },
  );

  downloadBlob(filename ?? `genova-notes-${stamp()}.pdf`, blob);
}
