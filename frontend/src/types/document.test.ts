import { describe, it, expect } from "vitest";
import { normalizeDocInfo, type DocInfo } from "./document";

describe("normalizeDocInfo", () => {
  it("normalizes a complete document response correctly", () => {
    const raw = {
      id: "doc-123",
      original_name: "Research_Paper.pdf",
      status: "ready",
      file_size: 1048576,
      page_count: 12,
      summary: "A study on agentic RAG workflows.",
      chunk_count: 48,
      error_message: null,
      uploaded_at: "2026-10-30T10:00:00.000Z",
      chunk_size: 1000,
      chunk_overlap: 200,
      keywords: ["ai", "rag", "agents"],
      processing_progress: 100,
      processing_stage: "completed",
    };

    const doc: DocInfo = normalizeDocInfo(raw);

    expect(doc.id).toBe("doc-123");
    expect(doc.original_name).toBe("Research_Paper.pdf");
    expect(doc.status).toBe("ready");
    expect(doc.file_size).toBe(1048576);
    expect(doc.page_count).toBe(12);
    expect(doc.summary).toBe("A study on agentic RAG workflows.");
    expect(doc.chunk_count).toBe(48);
    expect(doc.error_message).toBeNull();
    expect(doc.uploaded_at).toBe("2026-10-30T10:00:00.000Z");
    expect(doc.chunk_size).toBe(1000);
    expect(doc.chunk_overlap).toBe(200);
    expect(doc.keywords).toEqual(["ai", "rag", "agents"]);
    expect(doc.processing_progress).toBe(100);
    expect(doc.processing_stage).toBe("completed");
  });

  it("handles missing or null fields gracefully with safe fallbacks", () => {
    const raw = {
      id: 999,
      original_name: null,
      status: null,
      file_size: null,
      page_count: null,
      summary: null,
      chunk_count: null,
      error_message: null,
      uploaded_at: null,
    };

    const doc: DocInfo = normalizeDocInfo(raw);

    expect(doc.id).toBe("999");
    expect(doc.original_name).toBe("Untitled Document");
    expect(doc.status).toBe("pending");
    expect(doc.file_size).toBe(0);
    expect(doc.page_count).toBe(0);
    expect(doc.summary).toBe("");
    expect(doc.chunk_count).toBe(0);
    expect(doc.error_message).toBeNull();
    expect(doc.uploaded_at).toBe("");
  });

  it("falls back to filename if original_name is not provided", () => {
    const raw = {
      id: "doc-456",
      filename: "backup_notes.md",
    };

    const doc: DocInfo = normalizeDocInfo(raw);
    expect(doc.original_name).toBe("backup_notes.md");
  });
});
