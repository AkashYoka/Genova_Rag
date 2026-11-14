/**
 * Frontend-only enrichment of the document hierarchy for the Tree map.
 *
 * The API returns nested SectionTreeNode values (any depth) plus per-section
 * table metadata. Paragraph / table body content comes from section chunk
 * fetches. This module never invents hierarchy — it only attaches content
 * leaves that already exist in those payloads.
 */
import { formatPageRange } from "@/lib/document-hierarchy";
import type {
  HierarchyChunk,
  SectionTableMeta,
  SectionTreeNode,
} from "@/types/document-hierarchy";

export type TreeNodeKind =
  | "document"
  | "section"
  | "paragraph"
  | "table"
  | "loading";

export type EnrichedTreeNode = {
  id: string;
  kind: TreeNodeKind;
  title: string;
  subtitle?: string;
  pageLabel?: string | null;
  preview?: string;
  tableHeaders?: string[];
  tableShape?: string;
  sectionId?: string;
  chunkId?: string;
  tableId?: string;
  childCount: number;
  children: EnrichedTreeNode[];
};

export type SectionChunksMap = Record<string, HierarchyChunk[] | undefined>;
export type SectionChunksLoading = Record<string, boolean | undefined>;

function previewText(raw: string | undefined | null, max = 140): string {
  const text = String(raw || "")
    .replace(/\s+/g, " ")
    .trim();
  if (!text) return "";
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function paragraphTitle(chunk: HierarchyChunk, index: number): string {
  const raw = String(chunk.content || "")
    .replace(/^#+\s*/gm, "")
    .replace(/\s+/g, " ")
    .trim();
  if (!raw) return `Paragraph ${index + 1}`;
  const first = raw.split(/(?<=[.!?])\s+/)[0] || raw;
  return first.length > 72 ? `${first.slice(0, 70)}…` : first;
}

function tableTitle(
  table: SectionTableMeta | null,
  chunk: HierarchyChunk | null,
  index: number,
): string {
  const caption = String(table?.caption || "").trim();
  if (caption) return caption;
  const fromChunk = String(chunk?.content || "")
    .split("\n")
    .map((line) => line.trim())
    .find((line) => line && !line.startsWith("|") && !line.startsWith("-"));
  if (fromChunk && !fromChunk.startsWith("#")) {
    return fromChunk.length > 72 ? `${fromChunk.slice(0, 70)}…` : fromChunk;
  }
  return table?.table_id || chunk?.table_ids?.[0] || `Table ${index + 1}`;
}

function tablePreviewFromChunk(chunk: HierarchyChunk | null): string {
  if (!chunk?.content) return "";
  const lines = String(chunk.content)
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.startsWith("|"))
    .slice(0, 4);
  return lines.join("\n");
}

function chunkMatchesTable(
  chunk: HierarchyChunk,
  table: SectionTableMeta,
): boolean {
  const tableId = String(table.table_id || "");
  if (!tableId) return false;
  const ids = chunk.table_ids ?? [];
  if (ids.some((id) => String(id) === tableId)) return true;
  return String(chunk.content || "").includes(tableId);
}

/** Sections that can expand: nested parts, tables, or attached chunks. */
export function sectionHasExpandableContent(section: SectionTreeNode): boolean {
  if ((section.children?.length ?? 0) > 0) return true;
  if ((section.tables?.length ?? 0) > 0) return true;
  if ((section.chunk_count ?? 0) > 0) return true;
  return false;
}

export function collectExpandableSectionIds(
  sections: SectionTreeNode[],
): string[] {
  const ids: string[] = [];
  const walk = (nodes: SectionTreeNode[]) => {
    for (const node of nodes) {
      if (sectionHasExpandableContent(node)) {
        ids.push(node.section_id);
        if (node.children?.length) walk(node.children);
      }
    }
  };
  walk(sections);
  return ids;
}

function buildContentLeaves(
  section: SectionTreeNode,
  chunks: HierarchyChunk[] | undefined,
  loading: boolean,
): EnrichedTreeNode[] {
  const leaves: EnrichedTreeNode[] = [];
  const tables = section.tables ?? [];
  const matchedTableIds = new Set<string>();

  if (loading && chunks === undefined) {
    leaves.push({
      id: `loading:${section.section_id}`,
      kind: "loading",
      title: "Loading content…",
      subtitle: "Fetching paragraphs and tables",
      sectionId: section.section_id,
      childCount: 0,
      children: [],
    });
    return leaves;
  }

  const list = chunks ?? [];

  list.forEach((chunk, index) => {
    const type = String(chunk.chunk_type || chunk.content_type || "text").toLowerCase();
    const isTable = type.includes("table");
    const matchedTable =
      tables.find((table) => chunkMatchesTable(chunk, table)) ?? null;
    if (matchedTable?.table_id) {
      matchedTableIds.add(String(matchedTable.table_id));
    }

    if (isTable) {
      const headers = matchedTable?.headers?.filter(Boolean).slice(0, 4) ?? [];
      const shape =
        matchedTable?.n_rows != null && matchedTable?.n_cols != null
          ? `${matchedTable.n_rows}×${matchedTable.n_cols}`
          : undefined;
      leaves.push({
        id: `chunk:${chunk.chunk_id}`,
        kind: "table",
        title: tableTitle(matchedTable, chunk, index),
        subtitle: shape ? `Table · ${shape}` : "Table",
        pageLabel: formatPageRange(chunk.page_start, chunk.page_end),
        preview: tablePreviewFromChunk(chunk) || previewText(chunk.content, 160),
        tableHeaders: headers,
        tableShape: shape,
        sectionId: section.section_id,
        chunkId: chunk.chunk_id,
        tableId: matchedTable?.table_id || chunk.table_ids?.[0],
        childCount: 0,
        children: [],
      });
      return;
    }

    leaves.push({
      id: `chunk:${chunk.chunk_id}`,
      kind: "paragraph",
      title: paragraphTitle(chunk, index),
      subtitle: "Paragraph",
      pageLabel: formatPageRange(chunk.page_start, chunk.page_end),
      preview: previewText(chunk.content, 160),
      sectionId: section.section_id,
      chunkId: chunk.chunk_id,
      childCount: 0,
      children: [],
    });
  });

  // Table metadata present in the hierarchy but not yet matched to a chunk.
  tables.forEach((table, index) => {
    const tableId = String(table.table_id || `meta-${index}`);
    if (matchedTableIds.has(tableId)) return;
    if (list.length > 0 && list.some((chunk) => chunkMatchesTable(chunk, table))) {
      return;
    }
    const headers = (table.headers ?? []).filter(Boolean).slice(0, 4);
    const shape =
      table.n_rows != null && table.n_cols != null
        ? `${table.n_rows}×${table.n_cols}`
        : undefined;
    leaves.push({
      id: `table:${section.section_id}:${tableId}`,
      kind: "table",
      title: tableTitle(table, null, index),
      subtitle: shape ? `Table · ${shape}` : "Table",
      pageLabel: formatPageRange(table.page, table.page_end),
      preview: headers.length ? headers.join(" · ") : previewText(table.caption, 120),
      tableHeaders: headers,
      tableShape: shape,
      sectionId: section.section_id,
      tableId,
      childCount: 0,
      children: [],
    });
  });

  return leaves;
}

function enrichSection(
  section: SectionTreeNode,
  opts: {
    expandedIds: Set<string>;
    chunksBySection: SectionChunksMap;
    loadingBySection: SectionChunksLoading;
  },
): EnrichedTreeNode {
  const expanded = opts.expandedIds.has(section.section_id);
  const childSections = (section.children ?? []).map((child) =>
    enrichSection(child, opts),
  );

  let contentLeaves: EnrichedTreeNode[] = [];
  if (expanded) {
    contentLeaves = buildContentLeaves(
      section,
      opts.chunksBySection[section.section_id],
      Boolean(opts.loadingBySection[section.section_id]),
    );
  } else if ((section.tables?.length ?? 0) > 0 && (section.children?.length ?? 0) === 0) {
    // Collapsed leaf still reports expandable tables via childCount.
    contentLeaves = [];
  }

  const children = expanded ? [...childSections, ...contentLeaves] : [];
  const latentChildren =
    childSections.length +
    (section.tables?.length ?? 0) +
    (section.chunk_count ?? 0);

  return {
    id: section.section_id,
    kind: "section",
    title: section.title || "Untitled section",
    subtitle:
      latentChildren > 0
        ? `${latentChildren} nested item${latentChildren === 1 ? "" : "s"}`
        : section.summary
          ? previewText(section.summary, 80)
          : undefined,
    pageLabel: formatPageRange(section.page_start, section.page_end),
    preview: section.summary ? previewText(section.summary, 120) : undefined,
    sectionId: section.section_id,
    childCount: latentChildren,
    children,
  };
}

export function buildEnrichedTree(opts: {
  documentName: string;
  pageCount?: number | null;
  sectionCount: number;
  sections: SectionTreeNode[];
  expandedIds: Set<string>;
  chunksBySection: SectionChunksMap;
  loadingBySection: SectionChunksLoading;
}): EnrichedTreeNode {
  const children = opts.sections.map((section) =>
    enrichSection(section, {
      expandedIds: opts.expandedIds,
      chunksBySection: opts.chunksBySection,
      loadingBySection: opts.loadingBySection,
    }),
  );

  return {
    id: "__doc_root__",
    kind: "document",
    title: opts.documentName,
    subtitle: [
      `${opts.sectionCount} section${opts.sectionCount === 1 ? "" : "s"}`,
      opts.pageCount != null ? `${opts.pageCount} pages` : null,
    ]
      .filter(Boolean)
      .join(" · "),
    childCount: children.length,
    children,
  };
}

export function countVisibleNodes(root: EnrichedTreeNode): number {
  let total = 1;
  for (const child of root.children) {
    total += countVisibleNodes(child);
  }
  return total;
}
