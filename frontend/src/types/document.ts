export interface DocInfo {
  id: string;
  original_name: string;
  status: string;
  file_size: number;
  page_count: number;
  summary: string;
  chunk_count: number;
  error_message: string | null;
  uploaded_at: string;
  chunk_size?: number;
  chunk_overlap?: number;
  keywords?: string[];
  processing_progress?: number | null;
  processing_stage?: string | null;
}

export interface RawDocumentResponse {
  id?: unknown;
  original_name?: unknown;
  filename?: unknown;
  status?: unknown;
  file_size?: unknown;
  page_count?: unknown;
  summary?: unknown;
  chunk_count?: unknown;
  error_message?: unknown;
  uploaded_at?: unknown;
  chunk_size?: unknown;
  chunk_overlap?: unknown;
  keywords?: unknown;
  processing_progress?: unknown;
  processing_stage?: unknown;
}

export interface DocumentListApiResponse {
  items?: RawDocumentResponse[];
  documents?: RawDocumentResponse[];
  total?: number;
  page?: number;
  pages?: number;
  total_pages?: number;
  limit?: number;
  query?: string | null;
}

export function normalizeDocInfo(raw: RawDocumentResponse): DocInfo {
  const id = typeof raw.id === "string" ? raw.id : String(raw.id ?? "");
  const original_name =
    typeof raw.original_name === "string" && raw.original_name.trim().length > 0
      ? raw.original_name
      : typeof raw.filename === "string" && raw.filename.trim().length > 0
        ? raw.filename
        : "Untitled Document";
  const status = typeof raw.status === "string" ? raw.status : "pending";
  const file_size =
    typeof raw.file_size === "number" && !Number.isNaN(raw.file_size)
      ? raw.file_size
      : 0;
  const page_count =
    typeof raw.page_count === "number" && !Number.isNaN(raw.page_count)
      ? raw.page_count
      : 0;
  const summary = typeof raw.summary === "string" ? raw.summary : "";
  const chunk_count =
    typeof raw.chunk_count === "number" && !Number.isNaN(raw.chunk_count)
      ? raw.chunk_count
      : 0;
  const error_message =
    typeof raw.error_message === "string" ? raw.error_message : null;
  const uploaded_at =
    typeof raw.uploaded_at === "string"
      ? raw.uploaded_at
      : raw.uploaded_at instanceof Date
        ? raw.uploaded_at.toISOString()
        : "";

  const chunk_size =
    typeof raw.chunk_size === "number" ? raw.chunk_size : undefined;
  const chunk_overlap =
    typeof raw.chunk_overlap === "number" ? raw.chunk_overlap : undefined;

  let keywords: string[] | undefined = undefined;
  if (Array.isArray(raw.keywords)) {
    keywords = raw.keywords.filter((k): k is string => typeof k === "string");
  }

  const processing_progress =
    typeof raw.processing_progress === "number"
      ? raw.processing_progress
      : null;
  const processing_stage =
    typeof raw.processing_stage === "string" || typeof raw.processing_stage === "number"
      ? String(raw.processing_stage)
      : null;

  return {
    id,
    original_name,
    status,
    file_size,
    page_count,
    summary,
    chunk_count,
    error_message,
    uploaded_at,
    chunk_size,
    chunk_overlap,
    keywords,
    processing_progress,
    processing_stage,
  };
}
