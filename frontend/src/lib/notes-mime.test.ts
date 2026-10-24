import { describe, expect, it } from "vitest";
import {
  decodeNotePayload,
  encodeNotePayload,
  messageToNoteHtml,
} from "./notes-mime";

describe("notes-mime", () => {
  it("round-trips drag payloads", () => {
    const payload = {
      id: "assistant-1",
      role: "assistant" as const,
      content: "Hello **world**",
    };
    const decoded = decodeNotePayload(encodeNotePayload(payload));
    expect(decoded).toEqual(payload);
  });

  it("escapes HTML in note blocks", () => {
    const html = messageToNoteHtml({
      id: "user-1",
      role: "user",
      content: "<script>alert(1)</script>\nline2",
    });
    expect(html).toContain("&lt;script&gt;");
    expect(html).not.toContain("<script>");
    expect(html).toContain("<br>");
    expect(html).toContain("<strong>You</strong>");
  });
});
