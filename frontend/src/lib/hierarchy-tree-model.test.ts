import { describe, expect, it } from "vitest";
import {
  buildEnrichedTree,
  collectExpandableSectionIds,
  countVisibleNodes,
  sectionHasExpandableContent,
} from "./hierarchy-tree-model";
import type { SectionTreeNode } from "@/types/document-hierarchy";

const sections: SectionTreeNode[] = [
  {
    section_id: "sec_root",
    title: "Full PI",
    level: 1,
    ordinal: 0,
    page_start: 1,
    page_end: 4,
    chunk_count: 1,
    tables: [],
    children: [
      {
        section_id: "sec_child",
        title: "Dosage",
        level: 2,
        ordinal: 1,
        page_start: 2,
        page_end: 3,
        chunk_count: 2,
        tables: [
          {
            table_id: "tbl_1",
            caption: "Dose table",
            headers: ["IgE", "Dose"],
            n_rows: 3,
            n_cols: 2,
            page: 2,
          },
        ],
        children: [
          {
            section_id: "sec_grand",
            title: "Adult dosing",
            level: 3,
            ordinal: 2,
            page_start: 3,
            page_end: 3,
            chunk_count: 1,
            tables: [],
            children: [],
          },
        ],
      },
    ],
  },
];

describe("hierarchy-tree-model", () => {
  it("treats sections with nested content as expandable", () => {
    expect(sectionHasExpandableContent(sections[0])).toBe(true);
    expect(sectionHasExpandableContent(sections[0].children![0])).toBe(true);
    expect(
      collectExpandableSectionIds(sections),
    ).toEqual(["sec_root", "sec_child", "sec_grand"]);
  });

  it("recursively expands section nesting and content leaves without a depth cap", () => {
    const root = buildEnrichedTree({
      documentName: "xolair_prescribing.pdf",
      pageCount: 72,
      sectionCount: 3,
      sections,
      expandedIds: new Set(["sec_root", "sec_child", "sec_grand"]),
      chunksBySection: {
        sec_root: [
          {
            chunk_id: "c_root",
            document_id: "d1",
            section_id: "sec_root",
            chunk_index: 0,
            chunk_type: "text",
            page_start: 1,
            content: "Highlights overview paragraph.",
          },
        ],
        sec_child: [
          {
            chunk_id: "c_table",
            document_id: "d1",
            section_id: "sec_child",
            chunk_index: 0,
            chunk_type: "table",
            table_ids: ["tbl_1"],
            page_start: 2,
            content: "Dose table\n| IgE | Dose |\n| --- | --- |\n| 100 | 150 |",
          },
          {
            chunk_id: "c_para",
            document_id: "d1",
            section_id: "sec_child",
            chunk_index: 1,
            chunk_type: "text",
            page_start: 2,
            content: "Administer subcutaneously.",
          },
        ],
        sec_grand: [
          {
            chunk_id: "c_grand",
            document_id: "d1",
            section_id: "sec_grand",
            chunk_index: 0,
            chunk_type: "text",
            page_start: 3,
            content: "Adult patients 12 years and older.",
          },
        ],
      },
      loadingBySection: {},
    });

    expect(root.kind).toBe("document");
    expect(root.children).toHaveLength(1);
    const fullPi = root.children[0];
    expect(fullPi.children.map((c) => c.kind)).toContain("section");
    expect(fullPi.children.map((c) => c.kind)).toContain("paragraph");

    const dosage = fullPi.children.find((c) => c.id === "sec_child")!;
    expect(dosage.children.some((c) => c.kind === "table")).toBe(true);
    expect(dosage.children.some((c) => c.kind === "paragraph")).toBe(true);
    expect(dosage.children.some((c) => c.id === "sec_grand")).toBe(true);

    const adult = dosage.children.find((c) => c.id === "sec_grand")!;
    expect(adult.children).toHaveLength(1);
    expect(adult.children[0].kind).toBe("paragraph");

    // Document + 3 sections + 4 content leaves
    expect(countVisibleNodes(root)).toBe(8);
  });

  it("does not invent content leaves when a branch is collapsed", () => {
    const root = buildEnrichedTree({
      documentName: "doc.pdf",
      sectionCount: 1,
      sections,
      expandedIds: new Set(),
      chunksBySection: {},
      loadingBySection: {},
    });
    expect(root.children[0].children).toHaveLength(0);
    expect(root.children[0].childCount).toBeGreaterThan(0);
  });
});
