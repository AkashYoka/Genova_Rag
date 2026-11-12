import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { DocumentExplorer } from "./DocumentExplorer";

vi.mock("@/lib/document-hierarchy", async () => {
  const actual = await vi.importActual<typeof import("@/lib/document-hierarchy")>(
    "@/lib/document-hierarchy",
  );
  return {
    ...actual,
    fetchDocumentHierarchy: vi.fn(),
    fetchDocumentIndexStatus: vi.fn(),
    fetchSectionChunks: vi.fn(),
    fetchSectionTables: vi.fn(),
    fetchRelatedSections: vi.fn(),
    clearHierarchyCache: vi.fn(),
  };
});

vi.mock("@/lib/chat-attachments", () => ({
  queueDocChat: vi.fn(),
}));

vi.mock("next-themes", () => ({
  useTheme: () => ({ theme: "light", resolvedTheme: "light", setTheme: vi.fn() }),
}));

vi.mock("@xyflow/react", () => {
  const React = require("react") as typeof import("react");
  return {
    ReactFlowProvider: ({ children }: { children: React.ReactNode }) =>
      React.createElement(React.Fragment, null, children),
    ReactFlow: () =>
      React.createElement("div", { "data-testid": "hierarchy-tree-map" }),
    Background: () => null,
    Controls: () => null,
    MiniMap: () => null,
    Handle: () => null,
    MarkerType: { ArrowClosed: "arrowclosed" },
    Position: { Top: "top", Bottom: "bottom", Left: "left", Right: "right" },
    useEdgesState: () => [[], vi.fn(), vi.fn()],
    useNodesState: () => [[], vi.fn(), vi.fn()],
    useReactFlow: () => ({
      fitView: vi.fn(() => Promise.resolve()),
      setViewport: vi.fn(() => Promise.resolve()),
      getViewport: () => ({ x: 0, y: 0, zoom: 0.85 }),
    }),
  };
});

import {
  fetchDocumentHierarchy,
  fetchDocumentIndexStatus,
  fetchRelatedSections,
  fetchSectionChunks,
  fetchSectionTables,
} from "@/lib/document-hierarchy";

const hierarchy = {
  document_id: "doc-1",
  filename: "sample_report.pdf",
  chunk_strategy: "semantic_llm",
  index_version: "semantic-llm-v1",
  section_count: 2,
  chunk_count: 3,
  report: {
    page_count: 10,
    chunk_types: { text: 2, table: 1 },
    source_coverage: 1,
  },
  sections: [
    {
      section_id: "sec_root",
      title: "Executive Summary",
      level: 1,
      ordinal: 0,
      page_start: 1,
      page_end: 2,
      summary: "High-level findings.",
      chunk_count: 1,
      tables: [],
      children: [
        {
          section_id: "sec_child",
          title: "Key Metrics",
          level: 2,
          ordinal: 1,
          page_start: 2,
          page_end: 2,
          summary: "Metrics overview.",
          chunk_count: 2,
          tables: [
            {
              table_id: "tbl_1",
              caption: "Revenue by quarter",
              headers: ["Q1", "Q2"],
              n_rows: 2,
              n_cols: 2,
              page: 2,
            },
          ],
          children: [],
        },
      ],
    },
  ],
};

describe("DocumentExplorer", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(fetchDocumentHierarchy).mockResolvedValue(hierarchy as never);
    vi.mocked(fetchDocumentIndexStatus).mockResolvedValue({
      document_id: "doc-1",
      status: "ready",
      chunk_strategy: "semantic_llm",
      index_version: "semantic-llm-v1",
      section_count: 2,
      chunk_count: 3,
      report: hierarchy.report,
    } as never);
    vi.mocked(fetchSectionChunks).mockResolvedValue({
      document_id: "doc-1",
      section_id: "sec_child",
      title: "Key Metrics",
      chunk_count: 1,
      chunks: [
        {
          chunk_id: "chunk_1",
          document_id: "doc-1",
          section_id: "sec_child",
          chunk_index: 0,
          chunk_type: "text",
          page_start: 2,
          page_end: 2,
          content: "Metric narrative content.",
        },
      ],
    });
    vi.mocked(fetchSectionTables).mockResolvedValue({
      document_id: "doc-1",
      section_id: "sec_child",
      title: "Key Metrics",
      tables: hierarchy.sections[0].children![0].tables!,
    });
    vi.mocked(fetchRelatedSections).mockResolvedValue({
      document_id: "doc-1",
      section_id: "sec_child",
      parent: hierarchy.sections[0],
      previous: null,
      next: null,
      children: [],
      siblings: [],
    });
  });

  it("shows numbered contents and opens a readable part detail", async () => {
    const user = userEvent.setup();
    render(
      <DocumentExplorer
        documentId="doc-1"
        fallbackName="sample_report.pdf"
        pageCount={10}
      />,
    );

    await waitFor(() => {
      expect(
        screen.getByRole("heading", { level: 1, name: "sample_report.pdf" }),
      ).toBeInTheDocument();
    });

    expect(screen.getByText("How to read this page")).toBeInTheDocument();
    expect(screen.getByText(/1\./)).toBeInTheDocument();
    expect(screen.getAllByText("Executive Summary").length).toBeGreaterThan(0);
    expect(screen.getByText("List")).toBeInTheDocument();
    expect(screen.getByText("Tree")).toBeInTheDocument();

    await user.click(screen.getAllByText("Key Metrics")[0]);

    expect(await screen.findByText("In short")).toBeInTheDocument();
    expect(screen.getByText("Metrics overview.")).toBeInTheDocument();
    expect(screen.getByText("Metric narrative content.")).toBeInTheDocument();
    expect(screen.getAllByText("Revenue by quarter").length).toBeGreaterThan(0);
    expect(screen.getByText("Related from")).toBeInTheDocument();
    expect(screen.queryByText(/knowledge graph/i)).not.toBeInTheDocument();

    await user.click(screen.getByText("Tree"));
    expect(screen.getByTestId("hierarchy-tree-map")).toBeInTheDocument();
    expect(screen.getByText("List")).toBeInTheDocument();
    // Tree mode opens part details in a modal so the map keeps full width.
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(
      screen.queryByText(/Click a part to read summary/i),
    ).not.toBeInTheDocument();
  });
});
