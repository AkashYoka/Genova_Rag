import { markdownToNoteHtml } from "./notes-markdown";

/** Custom drag payload so file-drop overlay ignores chat→notes drags. */
export const GENOVA_NOTE_MIME = "application/x-genova-note";

export interface GenovaNotePayload {
  id: string;
  role: "user" | "assistant";
  content: string;
}

export function encodeNotePayload(payload: GenovaNotePayload): string {
  return JSON.stringify(payload);
}

export function decodeNotePayload(raw: string): GenovaNotePayload | null {
  try {
    const data = JSON.parse(raw) as GenovaNotePayload;
    if (!data?.content || typeof data.content !== "string") return null;
    return data;
  } catch {
    return null;
  }
}

/** Escape text for safe HTML insertion into the shared notes document. */
export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * Convert a chat message into an HTML block appended to notes.
 *
 * Chat answers are Markdown (with GFM tables and LaTeX math). Notes are edited
 * in CKEditor, so the content is converted into the HTML subset the editor's
 * plugins understand rather than inserting the raw Markdown source.
 */
export function messageToNoteHtml(payload: GenovaNotePayload): string {
  const label = payload.role === "user" ? "You" : "Assistant";
  const body = markdownToNoteHtml(payload.content) || "<p></p>";
  return `<p><strong>${label}</strong></p>${body}<p>&nbsp;</p>`;
}
