import { describe, expect, it } from "vitest";
import {
  findFlexibleNeedleMatch,
  normalizeCitationText,
  parseCitationBbox,
  resolveHighlightRects,
} from "./citation-highlight";

describe("normalizeCitationText", () => {
  it("collapses whitespace and strips trailing ellipsis", () => {
    expect(normalizeCitationText("GraphRAG  is\na new\nparadigm...")).toBe(
      "graphrag is a new paradigm",
    );
  });

  it("joins PDF line-wrap hyphenation", () => {
    expect(
      normalizeCitationText("Structure-guided Prompts generate cat-\negorizing reasoning"),
    ).toBe("structure-guided prompts generate categorizing reasoning");
  });
});

describe("findFlexibleNeedleMatch", () => {
  it("matches the full excerpt across hyphenated PDF text", () => {
    const haystack = normalizeCitationText(
      "ProLLM [68] applies graph-based RAG to predict protein-protein interactions, modeling signaling pathways",
    );
    const needle = normalizeCitationText(
      "ProLLM [68] applies graph-based RAG to predict protein interactions",
    );
    const match = findFlexibleNeedleMatch(haystack, needle);
    expect(match).not.toBeNull();
    expect(match!.matched).toBe(needle.length);
    expect(haystack.slice(match!.start, match!.end)).toContain("prollm");
    expect(haystack.slice(match!.start, match!.end)).toContain("interactions");
  });

  it("matches multi-line excerpts after whitespace normalization", () => {
    const haystack = normalizeCitationText(
      "Graph-based Retrieval-Augmented Generation (GraphRAG) is a new paradigm that addresses the limitations of traditional retrieval-augmented generation (RAG) methods",
    );
    const needle = normalizeCitationText(
      "Graph-based Retrieval-Augmented Generation (GraphRAG) is a new\nparadigm that addresses the limitations of traditional retrieval-augmented generation (RAG) methods...",
    );
    const match = findFlexibleNeedleMatch(haystack, needle);
    expect(match).not.toBeNull();
    expect(match!.matched).toBe(needle.length);
  });

  it("falls back to the longest matching prefix when the tail diverges", () => {
    const haystack = "alpha beta gamma delta epsilon zeta";
    const needle = "alpha beta gamma something else entirely";
    const match = findFlexibleNeedleMatch(haystack, needle);
    expect(match).not.toBeNull();
    expect(match!.matched).toBeGreaterThanOrEqual(12);
    expect(haystack.slice(match!.start, match!.end)).toContain("alpha beta");
  });
});

describe("parseCitationBbox", () => {
  it("parses normalized [[x0,y0,x1,y1]] arrays into percent rects", () => {
    expect(
      parseCitationBbox("[[0.1,0.2,0.5,0.35],[0.1,0.4,0.6,0.5]]"),
    ).toEqual([
      {
        left: 0.1,
        top: 0.2,
        width: 0.4,
        height: 0.15,
        unit: "percent",
      },
      {
        left: 0.1,
        top: 0.4,
        width: 0.5,
        height: 0.1,
        unit: "percent",
      },
    ]);
  });

  it("returns undefined for empty or invalid bbox", () => {
    expect(parseCitationBbox("")).toBeUndefined();
    expect(parseCitationBbox("not-json")).toBeUndefined();
    expect(parseCitationBbox("[]")).toBeUndefined();
  });
});

describe("resolveHighlightRects", () => {
  it("prefers explicit highlightRects over bbox", () => {
    const rects = [{ left: 1, top: 2, width: 3, height: 4 }];
    expect(
      resolveHighlightRects({
        highlightRects: rects,
        bbox: "[[0,0,1,1]]",
      }),
    ).toEqual(rects);
  });

  it("falls back to bbox when highlightRects are missing", () => {
    expect(resolveHighlightRects({ bbox: "[[0,0,0.5,0.25]]" })).toEqual([
      {
        left: 0,
        top: 0,
        width: 0.5,
        height: 0.25,
        unit: "percent",
      },
    ]);
  });
});
