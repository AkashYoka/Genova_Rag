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

  it("converts Markdown instead of pasting the raw source", () => {
    const html = messageToNoteHtml({
      id: "assistant-1",
      role: "assistant",
      content: "## Findings\n\n- **Critical** issue\n- Minor issue",
    });

    expect(html).toContain("<strong>Assistant</strong>");
    expect(html).toContain("<h2>Findings</h2>");
    expect(html).toContain("<ul>");
    expect(html).toContain("<strong>Critical</strong>");
    // No Markdown syntax should survive into the editor.
    expect(html).not.toContain("## Findings");
    expect(html).not.toContain("- **Critical**");
  });

  it("keeps HTML in the source message as literal text", () => {
    const html = messageToNoteHtml({
      id: "user-1",
      role: "user",
      content: "<script>alert(1)</script>",
    });

    expect(html).toContain("<strong>You</strong>");
    expect(html).not.toContain("<script>");
    expect(html).toContain("&#x3C;script>alert(1)&#x3C;/script>");
  });

  it("turns hard breaks into <br> and preserves order", () => {
    const html = messageToNoteHtml({
      id: "user-2",
      role: "user",
      content: "line1  \nline2",
    });

    expect(html).toContain("line1<br>");
    expect(html).toContain("line2");
    expect(html.indexOf("line1")).toBeLessThan(html.indexOf("line2"));
  });

  it("renders GFM tables with alignment", () => {
    const html = messageToNoteHtml({
      id: "assistant-2",
      role: "assistant",
      content: [
        "| Metric | Value |",
        "| :--- | ---: |",
        "| Accuracy | 98% |",
      ].join("\n"),
    });

    expect(html).toContain("<table>");
    expect(html).toContain("<th");
    expect(html).toContain("text-align: left;");
    expect(html).toContain("text-align: right;");
    expect(html).toContain("Accuracy");
  });

  it("falls back to an empty paragraph for blank messages", () => {
    const html = messageToNoteHtml({
      id: "assistant-3",
      role: "assistant",
      content: "   ",
    });

    expect(html).toContain("<strong>Assistant</strong>");
    expect(html).toContain("<p></p>");
  });
});
