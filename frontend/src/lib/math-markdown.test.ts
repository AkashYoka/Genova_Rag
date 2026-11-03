import { describe, expect, it } from "vitest";
import { prepareMarkdownForMath } from "./math-markdown";

describe("prepareMarkdownForMath", () => {
  it("converts \\( \\) and \\[ \\] to dollar delimiters", () => {
    expect(prepareMarkdownForMath("Energy \\(E = mc^2\\) is famous.")).toBe(
      "Energy $E = mc^2$ is famous.",
    );
    expect(
      prepareMarkdownForMath("Display:\n\\[\n\\frac{a}{b}\n\\]\nDone."),
    ).toBe("Display:\n\n$$\n\\frac{a}{b}\n$$\n\nDone.");
  });

  it("leaves fenced code and inline code untouched", () => {
    const src =
      "See `\\(x\\)` and:\n```ts\nconst x = '\\(y\\)';\n```\nThen \\(z\\).";
    const out = prepareMarkdownForMath(src);
    expect(out).toContain("`\\(x\\)`");
    expect(out).toContain("```ts\nconst x = '\\(y\\)';\n```");
    expect(out).toContain("$z$");
  });

  it("does not reformat tables inside fenced code", () => {
    const src = "```\n| A | B |\n|---|---|\n| 1 | 2 |\nAfter\n```";
    expect(prepareMarkdownForMath(src)).toBe(src);
  });

  it("preserves complete math while streaming", () => {
    const src = "Inline \\(a+b\\) and display \\[c+d\\] done.";
    expect(prepareMarkdownForMath(src, { isStreaming: true })).toBe(
      "Inline $a+b$ and display \n$$\nc+d\n$$\n done.",
    );
  });

  it("holds back incomplete inline and display math while streaming", () => {
    expect(
      prepareMarkdownForMath("Start \\(E = mc", { isStreaming: true }),
    ).toBe("Start ");
    expect(
      prepareMarkdownForMath("Start \\[\n\\int_0^1", { isStreaming: true }),
    ).toBe("Start ");
    expect(
      prepareMarkdownForMath("Ok $$a$$ then $$b", { isStreaming: true }),
    ).toBe("Ok $$a$$ then ");
  });

  it("supports chemical and medical-style latex after normalization", () => {
    const chem =
      "Reaction \\(2H_2 + O_2 \\rightarrow 2H_2O\\) and \\(\\mathrm{CaCO_3 \\rightarrow CaO + CO_2}\\).";
    expect(prepareMarkdownForMath(chem)).toContain(
      "$2H_2 + O_2 \\rightarrow 2H_2O$",
    );
    expect(prepareMarkdownForMath(chem)).toContain(
      "$\\mathrm{CaCO_3 \\rightarrow CaO + CO_2}$",
    );

    const medical =
      "Clearance \\(CL = \\frac{Dose}{AUC}\\) [Source: pk.pdf, Page 3].";
    expect(prepareMarkdownForMath(medical)).toBe(
      "Clearance $CL = \\frac{Dose}{AUC}$ [Source: pk.pdf, Page 3].",
    );
  });

  it("separates a table from the intro text and the continuation", () => {
    const src = "Intro\n| A | B |\n| --- | --- |\n| 1 | 2 |\nAfter";
    expect(prepareMarkdownForMath(src)).toBe(
      "Intro\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\nAfter",
    );
  });

  it("separates a table that directly follows a list", () => {
    const src = "Items\n- a\n- b\n| A | B |\n|---|---|\n| 1 | 2 |";
    expect(prepareMarkdownForMath(src)).toBe(
      "Items\n- a\n- b\n\n| A | B |\n|---|---|\n| 1 | 2 |",
    );
  });

  it("expands single-line display math into its own block", () => {
    const out = prepareMarkdownForMath("Intro $$E = mc^2$$ after");
    expect(out).toContain("$$\nE = mc^2\n$$");
    expect(out.startsWith("Intro \n\n$$")).toBe(true);
    expect(out.endsWith("$$\n\n after")).toBe(true);
  });

  it("keeps LaTeX line breaks inside display math intact", () => {
    const out = prepareMarkdownForMath("\\[\na \\\\[2pt] b\n\\]");
    expect(out).toContain("a \\\\[2pt] b");
    expect(out.split("$$").length - 1).toBe(2);
  });
});
