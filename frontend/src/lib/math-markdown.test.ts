import { describe, expect, it } from "vitest";
import { prepareMarkdownForMath } from "./math-markdown";

describe("prepareMarkdownForMath", () => {
  it("converts \\( \\) and \\[ \\] to dollar delimiters", () => {
    expect(prepareMarkdownForMath("Energy \\(E = mc^2\\) is famous.")).toBe(
      "Energy $E = mc^2$ is famous.",
    );
    expect(
      prepareMarkdownForMath("Display:\n\\[\n\\frac{a}{b}\n\\]\nDone."),
    ).toBe("Display:\n$$\n\\frac{a}{b}\n$$\nDone.");
  });

  it("leaves fenced code and inline code untouched", () => {
    const src =
      "See `\\(x\\)` and:\n```ts\nconst x = '\\(y\\)';\n```\nThen \\(z\\).";
    const out = prepareMarkdownForMath(src);
    expect(out).toContain("`\\(x\\)`");
    expect(out).toContain("```ts\nconst x = '\\(y\\)';\n```");
    expect(out).toContain("$z$");
  });

  it("preserves complete math while streaming", () => {
    const src = "Inline \\(a+b\\) and display \\[c+d\\] done.";
    expect(prepareMarkdownForMath(src, { isStreaming: true })).toBe(
      "Inline $a+b$ and display $$c+d$$ done.",
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
});
