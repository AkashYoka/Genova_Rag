/**
 * Document hierarchy API helpers with in-flight dedupe and short-lived cache.
 */
import { api } from "@/lib/api";
import type {
  DocumentHierarchyResponse,
  DocumentIndexStatusResponse,
  RelatedSectionsResponse,
  SectionChunksResponse,
  SectionTableMeta,
  SectionTablesResponse,
  SectionTreeNode,
} from "@/types/document-hierarchy";

export type { SectionTableMeta };

const CACHE_TTL_MS = 60_000;

type CacheEntry<T> = { value: T; expires: number };

const cache = new Map<string, CacheEntry<unknown>>();
const inflight = new Map<string, Promise<unknown>>();

function readCache<T>(key: string): T | null {
  const hit = cache.get(key);
  if (!hit) return null;
  if (Date.now() > hit.expires) {
    cache.delete(key);
    return null;
  }
  return hit.value as T;
}

function writeCache<T>(key: string, value: T): T {
  cache.set(key, { value, expires: Date.now() + CACHE_TTL_MS });
  return value;
}

async function cachedGet<T>(key: string, path: string): Promise<T> {
  const cached = readCache<T>(key);
  if (cached !== null) return cached;

  const existing = inflight.get(key);
  if (existing) return existing as Promise<T>;

  const promise = api
    .get<T>(path)
    .then((data) => writeCache(key, data))
    .finally(() => {
      inflight.delete(key);
    });
  inflight.set(key, promise);
  return promise;
}

export function clearHierarchyCache(documentId?: string): void {
  if (!documentId) {
    cache.clear();
    return;
  }
  const prefix = `doc:${documentId}:`;
  for (const key of cache.keys()) {
    if (key.startsWith(prefix)) cache.delete(key);
  }
}

export function fetchDocumentHierarchy(
  documentId: string,
): Promise<DocumentHierarchyResponse> {
  return cachedGet(
    `doc:${documentId}:hierarchy`,
    `/api/v1/documents/${encodeURIComponent(documentId)}/hierarchy`,
  );
}

export function fetchDocumentIndexStatus(
  documentId: string,
): Promise<DocumentIndexStatusResponse> {
  return cachedGet(
    `doc:${documentId}:index-status`,
    `/api/v1/documents/${encodeURIComponent(documentId)}/index-status`,
  );
}

export function fetchSectionDetail(
  documentId: string,
  sectionId: string,
): Promise<SectionTreeNode> {
  return cachedGet(
    `doc:${documentId}:section:${sectionId}`,
    `/api/v1/documents/${encodeURIComponent(documentId)}/sections/${encodeURIComponent(sectionId)}`,
  );
}

export function fetchSectionChunks(
  documentId: string,
  sectionId: string,
  opts?: { includeContent?: boolean; limit?: number },
): Promise<SectionChunksResponse> {
  const includeContent = opts?.includeContent ?? true;
  const limit = opts?.limit ?? 500;
  const qs = `?include_content=${includeContent ? "true" : "false"}&limit=${limit}`;
  return cachedGet(
    `doc:${documentId}:chunks:${sectionId}:${includeContent}:${limit}`,
    `/api/v1/documents/${encodeURIComponent(documentId)}/sections/${encodeURIComponent(sectionId)}/chunks${qs}`,
  );
}

export function fetchSectionTables(
  documentId: string,
  sectionId: string,
): Promise<SectionTablesResponse> {
  return cachedGet(
    `doc:${documentId}:tables:${sectionId}`,
    `/api/v1/documents/${encodeURIComponent(documentId)}/sections/${encodeURIComponent(sectionId)}/tables`,
  );
}

export function fetchRelatedSections(
  documentId: string,
  sectionId: string,
): Promise<RelatedSectionsResponse> {
  return cachedGet(
    `doc:${documentId}:related:${sectionId}`,
    `/api/v1/documents/${encodeURIComponent(documentId)}/sections/${encodeURIComponent(sectionId)}/related`,
  );
}

export function formatPageRange(
  start?: number | null,
  end?: number | null,
): string | null {
  if (start == null && end == null) return null;
  if (start != null && (end == null || end === start)) return `p. ${start}`;
  if (start != null && end != null) return `p. ${start}–${end}`;
  if (end != null) return `p. ${end}`;
  return null;
}

export function countDescendants(node: SectionTreeNode): number {
  const children = node.children ?? [];
  return children.reduce(
    (sum, child) => sum + 1 + countDescendants(child),
    0,
  );
}

export function collectAllTables(
  sections: SectionTreeNode[],
): Array<SectionTableMeta & { section_id: string; section_title: string }> {
  const out: Array<
    SectionTableMeta & { section_id: string; section_title: string }
  > = [];
  const walk = (nodes: SectionTreeNode[]) => {
    for (const node of nodes) {
      for (const table of node.tables ?? []) {
        out.push({
          ...table,
          section_id: node.section_id,
          section_title: node.title,
        });
      }
      if (node.children?.length) walk(node.children);
    }
  };
  walk(sections);
  return out;
}

export function findSectionPath(
  sections: SectionTreeNode[],
  sectionId: string,
): SectionTreeNode[] {
  const path: SectionTreeNode[] = [];
  const walk = (nodes: SectionTreeNode[]): boolean => {
    for (const node of nodes) {
      path.push(node);
      if (node.section_id === sectionId) return true;
      if (node.children?.length && walk(node.children)) return true;
      path.pop();
    }
    return false;
  };
  walk(sections);
  return path;
}

export function findSectionById(
  sections: SectionTreeNode[],
  sectionId: string,
): SectionTreeNode | null {
  for (const node of sections) {
    if (node.section_id === sectionId) return node;
    const child = findSectionById(node.children ?? [], sectionId);
    if (child) return child;
  }
  return null;
}

export function countTablesInTree(sections: SectionTreeNode[]): number {
  return collectAllTables(sections).length;
}

/** Flatten the section tree into reading order (ordinal DFS). */
export function flattenSections(
  sections: SectionTreeNode[],
): SectionTreeNode[] {
  const out: SectionTreeNode[] = [];
  const walk = (nodes: SectionTreeNode[]) => {
    for (const node of nodes) {
      out.push(node);
      if (node.children?.length) walk(node.children);
    }
  };
  walk(sections);
  return out;
}

/** Domain-agnostic chips derived only from backend section metadata. */
export function sectionSignalChips(section: SectionTreeNode): string[] {
  const chips: string[] = [];
  const seen = new Set<string>();
  const pushUnique = (label: string) => {
    if (!label || seen.has(label)) return;
    seen.add(label);
    chips.push(label);
  };

  const tables = section.tables ?? [];
  for (const table of tables.slice(0, 3)) {
    const caption = String(table.caption || "").trim();
    if (caption) {
      pushUnique(caption.length > 36 ? `${caption.slice(0, 34)}…` : caption);
    } else if (table.table_id) {
      pushUnique(String(table.table_id));
    } else {
      pushUnique("Table");
    }
  }
  if (tables.length > 3) {
    pushUnique(`+${tables.length - 3} tables`);
  }
  const chunks = section.chunk_count ?? 0;
  if (chunks > 0) pushUnique(`${chunks} chunk${chunks === 1 ? "" : "s"}`);
  return chips;
}
