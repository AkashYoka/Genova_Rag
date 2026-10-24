import { jsPDF } from "jspdf";
import TurndownService from "turndown";

function stripHtml(html: string): string {
  if (typeof document === "undefined") {
    return html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
  }
  const el = document.createElement("div");
  el.innerHTML = html;
  return (el.innerText || el.textContent || "").replace(/\n{3,}/g, "\n\n").trim();
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

export async function exportNotesAsPdf(html: string) {
  const text = stripHtml(html) || " ";
  const doc = new jsPDF({ unit: "pt", format: "a4" });
  const margin = 48;
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const maxWidth = pageWidth - margin * 2;
  const lines = doc.splitTextToSize(text, maxWidth) as string[];
  const lineHeight = 14;
  let y = margin;

  doc.setFont("helvetica", "normal");
  doc.setFontSize(11);

  for (const line of lines) {
    if (y + lineHeight > pageHeight - margin) {
      doc.addPage();
      y = margin;
    }
    doc.text(line, margin, y);
    y += lineHeight;
  }

  doc.save(`genova-notes-${stamp()}.pdf`);
}
