/** Types mirroring backend document hierarchy / section navigation APIs. */

export interface SectionTableMeta {
  table_id?: string;
  page?: number | null;
  page_end?: number | null;
  caption?: string;
  headers?: string[];
  n_rows?: number;
  n_cols?: number;
  block_id?: string;
  [key: string]: unknown;
}

export interface SectionTreeNode {
  section_id: string;
  document_id?: string | null;
  parent_section_id?: string | null;
  title: string;
  level: number;
  ordinal: number;
  page_start?: number | null;
  page_end?: number | null;
  summary?: string;
  block_ids?: string[];
  tables?: SectionTableMeta[];
  chunk_count?: number;
  index_version?: string | null;
  children?: SectionTreeNode[];
}

export interface ChunkingReport {
  block_count?: number;
  section_count?: number;
  chunk_count?: number;
  page_count?: number;
  pages_covered?: number;
  pages_missing?: number[];
  source_coverage?: number;
  token_limit_violations?: number;
  chunk_types?: Record<string, number>;
  used_llm?: boolean;
  used_fallback?: boolean;
  llm_windows?: number;
  llm_calls?: number;
  llm_retries?: number;
  strategy?: string;
  index_version?: string;
  errors?: string[];
  elapsed_s?: number;
  [key: string]: unknown;
}

export interface DocumentHierarchyResponse {
  document_id: string;
  filename?: string | null;
  chunk_strategy?: string | null;
  index_version?: string | null;
  section_count: number;
  chunk_count: number;
  report?: ChunkingReport | null;
  sections: SectionTreeNode[];
}

export interface DocumentIndexStatusResponse {
  document_id: string;
  status: string;
  processing_stage?: string | null;
  processing_progress?: number | null;
  error_message?: string | null;
  chunk_strategy?: string | null;
  index_version?: string | null;
  section_count: number;
  chunk_count: number;
  report?: ChunkingReport | null;
}

export interface HierarchyChunk {
  chunk_id: string;
  document_id: string;
  section_id?: string | null;
  parent_section_id?: string | null;
  chunk_index: number;
  chunk_type?: string | null;
  content_type?: string | null;
  page_start?: number | null;
  page_end?: number | null;
  token_count?: number | null;
  block_ids?: string[];
  table_ids?: string[];
  index_version?: string | null;
  content?: string;
}

export interface SectionChunksResponse {
  document_id: string;
  section_id: string;
  title: string;
  chunk_count: number;
  chunks: HierarchyChunk[];
}

export interface SectionTablesResponse {
  document_id: string;
  section_id: string;
  title: string;
  tables: SectionTableMeta[];
}

export interface RelatedSectionsResponse {
  document_id: string;
  section_id: string;
  parent?: SectionTreeNode | null;
  previous?: SectionTreeNode | null;
  next?: SectionTreeNode | null;
  children: SectionTreeNode[];
  siblings: SectionTreeNode[];
}

export type ExplorerTab = "contents" | "tables";

export type ContentsViewMode = "list" | "tree";

export type ExplorerSelection =
  | { kind: "document" }
  | { kind: "section"; sectionId: string }
  | { kind: "chunk"; sectionId: string; chunkId: string };
