import { describe, expect, it } from "vitest";
import { markdownToNoteHtml } from "./notes-markdown";

describe("notes-markdown", () => {
  it("returns nothing for blank input", () => {
    expect(markdownToNoteHtml("")).toBe("");
    expect(markdownToNoteHtml("   \n\n ")).toBe("");
  });

  it("converts headings and paragraphs", () => {
    const html = markdownToNoteHtml(
      "# Title\n\nIntro paragraph.\n\n## Section\n\n### Detail",
    );

    expect(html).toContain("<h1>Title</h1>");
    expect(html).toContain("<p>Intro paragraph.</p>");
    expect(html).toContain("<h2>Section</h2>");
    expect(html).toContain("<h3>Detail</h3>");
  });

  it("converts inline emphasis, links, strikethrough and inline code", () => {
    const html = markdownToNoteHtml(
      "A **bold**, *italic*, ~~struck~~ and `code` with [a link](https://example.com).",
    );

    expect(html).toContain("<strong>bold</strong>");
    expect(html).toContain("<em>italic</em>");
    expect(html).toContain("<del>struck</del>");
    expect(html).toContain("<code>code</code>");
    expect(html).toContain('<a href="https://example.com">a link</a>');
  });

  it("converts bulleted and numbered lists", () => {
    const html = markdownToNoteHtml(
      "- First\n- Second\n\n1. Step one\n2. Step two",
    );

    expect(html).toContain("<ul>");
    expect(html).toContain("<li>First</li>");
    expect(html).toContain("<li>Second</li>");
    expect(html).toContain("<ol>");
    expect(html).toContain("<li>Step one</li>");
    expect(html).toContain("<li>Step two</li>");
  });

  it("keeps task-list state as visible glyphs (no form inputs)", () => {
    const html = markdownToNoteHtml("- [x] Done\n- [ ] Todo");

    expect(html).toContain("☑");
    expect(html).toContain("☐");
    expect(html).not.toContain("<input");
  });

  it("converts GFM tables with rows, columns and alignment", () => {
    const html = markdownToNoteHtml(
      [
        "| Metric | Value | Notes |",
        "| :--- | ---: | :---: |",
        "| Water | 18.015 | liquid |",
        "| Benzene | 78.11 | aromatic |",
      ].join("\n"),
    );

    expect(html).toContain("<table>");
    expect(html).toContain("<thead>");
    expect(html).toContain("<tbody>");
    expect(html.match(/<th[\s>]/g)).toHaveLength(3);
    expect(html.match(/<td/g)).toHaveLength(6);
    expect(html).toContain("text-align: left;");
    expect(html).toContain("text-align: right;");
    expect(html).toContain("text-align: center;");
    expect(html).toContain("Water");
    expect(html).toContain("Benzene");
    // Table stays in source order: header, then first row, then second row.
    expect(html.indexOf("Metric")).toBeLessThan(html.indexOf("Water"));
    expect(html.indexOf("Water")).toBeLessThan(html.indexOf("Benzene"));
  });

  it("does not leave Markdown table syntax behind", () => {
    const html = markdownToNoteHtml(
      "| A | B |\n| --- | --- |\n| 1 | 2 |",
    );

    expect(html).not.toContain("| ---");
    expect(html).not.toContain("| A |");
  });

  it("converts inline math into readable text", () => {
    const html = markdownToNoteHtml("Energy is $E = mc^2$.");

    expect(html).toContain("<em>E = mc²</em>");
    expect(html).not.toContain("$E");
    expect(html).not.toContain("mc^2");
  });

  it("converts display math into a centered paragraph", () => {
    const html = markdownToNoteHtml(
      "Compute:\n\n$$\n\\frac{a+b}{c} \\times 10^{3}\n$$",
    );

    expect(html).toContain('style="text-align: center;"');
    expect(html).toContain("<em>(a+b)/c × 10³</em>");
    expect(html).not.toContain("\\frac");
  });

  it("converts chemistry equations (mhchem) into readable formulas", () => {
    const html = markdownToNoteHtml(
      "Water: $\\ce{H2O}$. Reaction: $\\ce{2H2 + O2 -> 2H2O}$.",
    );

    expect(html).toContain("H₂O");
    expect(html).toContain("2H₂ + O₂ → 2H₂O");
    expect(html).not.toContain("\\ce");
  });

  it("converts code blocks with their language and block quotes", () => {
    const html = markdownToNoteHtml(
      "> Quoted finding\n\n```ts\nconst x = 1;\n```",
    );

    expect(html).toContain("<blockquote>");
    expect(html).toContain("<p>Quoted finding</p>");
    expect(html).toContain('<pre><code class="language-ts">');
    expect(html).toContain("const x = 1;");
  });

  it("preserves source citations in order", () => {
    const html = markdownToNoteHtml(
      [
        "The measured yield was 82%.",
        "",
        "[Source: chemistry.pdf, Page 4]",
        "",
        "This matches the expected range.",
      ].join("\n"),
    );

    expect(html).toContain("[Source: chemistry.pdf, Page 4]");
    expect(html.indexOf("measured yield")).toBeLessThan(
      html.indexOf("Source: chemistry.pdf"),
    );
    expect(html.indexOf("Source: chemistry.pdf")).toBeLessThan(
      html.indexOf("expected range"),
    );
  });

  it("escapes raw HTML instead of embedding it", () => {
    const html = markdownToNoteHtml(
      "<div onclick=\"alert(1)\">hi</div>\n\n<script>alert(2)</script>",
    );

    expect(html).not.toContain("<div");
    expect(html).not.toContain("<script>alert");
    expect(html).toContain("&#x3C;div");
    expect(html).toContain("&#x3C;script>alert(2)&#x3C;/script>");
  });

  it("converts horizontal rules", () => {
    expect(markdownToNoteHtml("Above\n\n---\n\nBelow")).toContain("<hr>");
  });

  it("keeps inline citation text inside a paragraph block", () => {
    const html = markdownToNoteHtml(
      "Finding one [Source: a.pdf, Page 1]\n\nFinding two [Source: b.pdf, Page 2]",
    );

    expect(html).toContain(
      "<p>Finding one [Source: a.pdf, Page 1]</p>",
    );
    expect(html).toContain(
      "<p>Finding two [Source: b.pdf, Page 2]</p>",
    );
  });
});
