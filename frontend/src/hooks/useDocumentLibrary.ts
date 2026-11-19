"use client";

/**
 * useDocumentLibrary
 *
 * One source of truth for the unified Workspace + Documents screen.
 * It owns the document list, trash, selection, upload and every management
 * action that used to live inside `app/documents/page.tsx`, so the chat column
 * and the documents view read the same data instead of fetching it twice.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useDropzone } from "react-dropzone";
import { toast } from "sonner";
import {
  api,
  CONNECTION_ERROR_BANNER_MESSAGE,
  CONNECTION_ERROR_MESSAGE,
} from "@/lib/api";
import type { DocInfo, DocumentListApiResponse } from "@/types/document";
import { normalizeDocInfo } from "@/types/document";

export type DocTab = "active" | "trash";
export type DocStatusFilter = "all" | "ready" | "indexing";

const ACCEPTED_FILE_TYPES = {
  "application/pdf": [".pdf"],
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [
    ".docx",
  ],
  "text/plain": [".txt"],
  "text/markdown": [".md"],
};

function matchesSearch(doc: DocInfo, query: string): boolean {
  if (!query) return true;
  const lower = query.toLowerCase();
  return (
    doc.original_name.toLowerCase().includes(lower) ||
    (doc.summary ? doc.summary.toLowerCase().includes(lower) : false)
  );
}

function matchesStatus(doc: DocInfo, filter: DocStatusFilter): boolean {
  if (filter === "ready") return doc.status === "ready";
  if (filter === "indexing")
    return doc.status === "pending" || doc.status === "processing";
  return true;
}

export interface DocumentLibrary {
  documents: DocInfo[];
  trashDocuments: DocInfo[];
  /** The list the table should render, after search + status filter. */
  visibleDocuments: DocInfo[];
  documentsLoading: boolean;
  trashLoading: boolean;
  connectionError: string;
  tab: DocTab;
  setTab: (tab: DocTab) => void;
  statusFilter: DocStatusFilter;
  setStatusFilter: (filter: DocStatusFilter) => void;
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  counts: { active: number; ready: number; indexing: number; trash: number };
  selectedIds: Set<string>;
  isAllSelected: boolean;
  toggleSelectOne: (id: string) => void;
  toggleSelectAll: () => void;
  clearSelection: () => void;
  upload: (files: File[]) => Promise<void>;
  openFilePicker: () => void;
  getInputProps: ReturnType<typeof useDropzone>["getInputProps"];
  uploading: boolean;
  uploadProgress: number;
  uploadError: string;
  clearUploadError: () => void;
  refresh: () => void;
  singleDeletingId: string | null;
  singleRestoringId: string | null;
  singlePermanentDeletingId: string | null;
  bulkDeleting: boolean;
  bulkRestoring: boolean;
  emptyingTrash: boolean;
  deleteDocument: (doc: DocInfo) => Promise<void>;
  restoreDocument: (doc: DocInfo) => Promise<void>;
  deleteDocumentForever: (doc: DocInfo) => Promise<void>;
  deleteSelected: () => Promise<void>;
  restoreSelected: () => Promise<void>;
  deleteSelectedForever: () => Promise<void>;
  emptyTrash: () => Promise<void>;
}

export function useDocumentLibrary(): DocumentLibrary {
  const router = useRouter();

  const [documents, setDocuments] = useState<DocInfo[]>([]);
  const [trashDocuments, setTrashDocuments] = useState<DocInfo[]>([]);
  const [documentsLoading, setDocumentsLoading] = useState(true);
  const [trashLoading, setTrashLoading] = useState(true);
  const [connectionError, setConnectionError] = useState("");

  const [tab, setTabState] = useState<DocTab>("active");
  const [statusFilter, setStatusFilterState] =
    useState<DocStatusFilter>("all");
  const [searchQuery, setSearchQuery] = useState("");

  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [bulkDeleting, setBulkDeleting] = useState(false);
  const [bulkRestoring, setBulkRestoring] = useState(false);
  const [emptyingTrash, setEmptyingTrash] = useState(false);

  const [singleDeletingId, setSingleDeletingId] = useState<string | null>(null);
  const [singleRestoringId, setSingleRestoringId] = useState<string | null>(
    null,
  );
  const [singlePermanentDeletingId, setSinglePermanentDeletingId] = useState<
    string | null
  >(null);


  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadError, setUploadError] = useState("");

  const prevDocsRef = useRef<Record<string, string>>({});

  const loadDocuments = useCallback(
    async (opts?: { silent?: boolean }) => {
      if (!opts?.silent) setDocumentsLoading(true);
      try {
        const queryParam = searchQuery.trim()
          ? `?q=${encodeURIComponent(searchQuery.trim())}`
          : "";
        const data = await api.get<DocumentListApiResponse>(
          `/api/v1/documents/${queryParam}`,
        );
        const rawList = data?.items ?? data?.documents ?? [];
        setDocuments(rawList.map(normalizeDocInfo));
        setConnectionError("");
      } catch (err) {
        const message =
          err instanceof Error ? err.message : CONNECTION_ERROR_MESSAGE;
        if (
          message.includes("Invalid or expired token") ||
          message.includes("Unauthorized") ||
          message.includes("User not found")
        ) {
          router.replace("/login");
          return;
        }
        setConnectionError(
          message === CONNECTION_ERROR_MESSAGE
            ? CONNECTION_ERROR_BANNER_MESSAGE
            : message,
        );
      } finally {
        if (!opts?.silent) setDocumentsLoading(false);
      }
    },
    [router, searchQuery],
  );

  const fetchTrash = useCallback(async (): Promise<DocInfo[]> => {
    const data = await api.get<DocumentListApiResponse>(
      "/api/v1/documents/trash",
    );
    const rawList = data?.items ?? data?.documents ?? [];
    return rawList.map(normalizeDocInfo);
  }, []);

  const loadTrash = useCallback(async () => {
    setTrashLoading(true);
    try {
      setTrashDocuments(await fetchTrash());
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to load trash";
      toast.error(message);
    } finally {
      setTrashLoading(false);
    }
  }, [fetchTrash]);

  // Initial load. Queries are debounced so typing in the search field does not
  // flash the table skeleton on every keystroke.
  useEffect(() => {
    const delay = searchQuery.trim() ? 220 : 0;
    const id = setTimeout(() => {
      void loadDocuments({ silent: Boolean(searchQuery.trim()) });
    }, delay);
    return () => clearTimeout(id);
  }, [loadDocuments, searchQuery]);

  // One-time trash load: resolved in a promise callback so it does not cascade
  // renders inside the effect body.
  useEffect(() => {
    let cancelled = false;
    fetchTrash()
      .then((docs) => {
        if (cancelled) return;
        setTrashDocuments(docs);
      })
      .catch((err) => {
        if (cancelled) return;
        toast.error(
          err instanceof Error ? err.message : "Failed to load trash",
        );
      })
      .finally(() => {
        if (!cancelled) setTrashLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [fetchTrash]);

  // Keep indexing progress live while anything is being prepared.
  useEffect(() => {
    const hasPending = documents.some(
      (d) => d.status === "pending" || d.status === "processing",
    );
    if (!hasPending) return;
    const interval = setInterval(() => {
      void loadDocuments({ silent: true });
    }, 3000);
    return () => clearInterval(interval);
  }, [documents, loadDocuments]);

  // Announce the moment a document becomes usable (or fails).
  useEffect(() => {
    const prev = prevDocsRef.current;
    const nextPrevDocs: Record<string, string> = {};
    for (const doc of documents) {
      nextPrevDocs[doc.id] = doc.status;
      const oldStatus = prev[doc.id];
      if (oldStatus && oldStatus !== doc.status) {
        if (doc.status === "ready") {
          toast.success(`Ingestion complete: '${doc.original_name}' is ready.`);
        } else if (doc.status === "failed") {
          toast.error(
            `Ingestion failed for '${doc.original_name}': ${doc.error_message || "Unknown error"}`,
          );
        }
      }
    }
    prevDocsRef.current = nextPrevDocs;
  }, [documents]);

  const setTab = useCallback(
    (next: DocTab) => {
      setTabState(next);
      setSelectedIds(new Set());
      if (next === "trash") void loadTrash();
      else void loadDocuments();
    },
    [loadDocuments, loadTrash],
  );

  const setStatusFilter = useCallback((filter: DocStatusFilter) => {
    setStatusFilterState(filter);
    setSelectedIds(new Set());
  }, []);

  const refresh = useCallback(() => {
    if (tab === "trash") void loadTrash();
    else void loadDocuments();
  }, [tab, loadDocuments, loadTrash]);

  const visibleDocuments = useMemo(() => {
    const source = tab === "active" ? documents : trashDocuments;
    return source
      .filter((doc) => matchesSearch(doc, searchQuery))
      .filter((doc) => matchesStatus(doc, statusFilter));
  }, [tab, documents, trashDocuments, searchQuery, statusFilter]);

  const counts = useMemo(
    () => ({
      active: documents.length,
      ready: documents.filter((d) => d.status === "ready").length,
      indexing: documents.filter(
        (d) => d.status === "pending" || d.status === "processing",
      ).length,
      trash: trashDocuments.length,
    }),
    [documents, trashDocuments],
  );

  const isAllSelected =
    visibleDocuments.length > 0 &&
    visibleDocuments.every((d) => selectedIds.has(d.id));

  const toggleSelectAll = () => {
    if (isAllSelected) {
      setSelectedIds(new Set());
      return;
    }
    setSelectedIds(new Set(visibleDocuments.map((d) => d.id)));
  };

  const toggleSelectOne = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const clearSelection = useCallback(() => setSelectedIds(new Set()), []);

  const uploadFiles = useCallback(
    async (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) return;
      setUploadError("");
      setUploading(true);
      setUploadProgress(0);

      try {
        for (let i = 0; i < acceptedFiles.length; i++) {
          const file = acceptedFiles[i];
          const formData = new FormData();
          formData.append("file", file);
          toast.info(`Uploading '${file.name}'...`);
          await api.postForm("/api/v1/documents/upload", formData);
          setUploadProgress(((i + 1) / acceptedFiles.length) * 100);
          toast.success(
            `'${file.name}' uploaded. Preparing it for search now.`,
          );
        }
        await loadDocuments({ silent: true });
        void loadTrash();
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Document upload failed";
        setUploadError(message);
        toast.error(`Upload failed: ${message}`);
      } finally {
        setUploading(false);
        setUploadProgress(0);
      }
    },
    [loadDocuments, loadTrash],
  );

  const { getInputProps, open: openFilePicker } = useDropzone({
    onDrop: (files) => {
      void uploadFiles(files);
    },
    noClick: true,
    noKeyboard: true,
    accept: ACCEPTED_FILE_TYPES,
    disabled: uploading,
  });

  const deleteDocument = useCallback(
    async (doc: DocInfo) => {
      if (!window.confirm("Move this document to trash?")) return;
      setSingleDeletingId(doc.id);
      try {
        await api.delete(`/api/v1/documents/${doc.id}`);
        toast.success("Document moved to trash");
        await loadDocuments({ silent: true });
        void loadTrash();
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Failed to delete document";
        toast.error(message);
      } finally {
        setSingleDeletingId(null);
      }
    },
    [loadDocuments, loadTrash],
  );

  const restoreDocument = useCallback(
    async (doc: DocInfo) => {
      setSingleRestoringId(doc.id);
      try {
        await api.post(`/api/v1/documents/${doc.id}/restore`);
        toast.success("Document restored");
        void loadTrash();
        await loadDocuments({ silent: true });
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Failed to restore document";
        toast.error(message);
      } finally {
        setSingleRestoringId(null);
      }
    },
    [loadDocuments, loadTrash],
  );

  const deleteDocumentForever = useCallback(
    async (doc: DocInfo) => {
      if (
        !window.confirm(
          `Permanently delete "${doc.original_name}"? This action cannot be undone.`,
        )
      ) {
        return;
      }
      setSinglePermanentDeletingId(doc.id);
      try {
        await api.delete(`/api/v1/documents/${doc.id}/permanent`);
        toast.success("Document permanently deleted");
        setSelectedIds((prev) => {
          const next = new Set(prev);
          next.delete(doc.id);
          return next;
        });
        void loadTrash();
        await loadDocuments({ silent: true });
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Permanent delete failed";
        toast.error(message);
      } finally {
        setSinglePermanentDeletingId(null);
      }
    },
    [loadDocuments, loadTrash],
  );

  const deleteSelected = useCallback(async () => {
    if (selectedIds.size === 0) return;
    if (
      !window.confirm(
        `Move ${selectedIds.size} document${selectedIds.size > 1 ? "s" : ""} to trash?`,
      )
    ) {
      return;
    }
    setBulkDeleting(true);
    const results = await Promise.allSettled(
      Array.from(selectedIds).map((id) => api.delete(`/api/v1/documents/${id}`)),
    );
    const failed = results.filter((r) => r.status === "rejected").length;
    const succeeded = results.length - failed;
    if (succeeded > 0) {
      toast.success(
        `${succeeded} document${succeeded > 1 ? "s" : ""} moved to trash`,
      );
    }
    if (failed > 0) {
      toast.error(
        `${failed} document${failed > 1 ? "s" : ""} could not be deleted`,
      );
    }
    clearSelection();
    setBulkDeleting(false);
    await loadDocuments({ silent: true });
    void loadTrash();
  }, [selectedIds, clearSelection, loadDocuments, loadTrash]);

  const restoreSelected = useCallback(async () => {
    if (selectedIds.size === 0) return;
    setBulkRestoring(true);
    const results = await Promise.allSettled(
      Array.from(selectedIds).map((id) =>
        api.post(`/api/v1/documents/${id}/restore`),
      ),
    );
    const failed = results.filter((r) => r.status === "rejected").length;
    const succeeded = results.length - failed;
    if (succeeded > 0) {
      toast.success(
        `${succeeded} document${succeeded > 1 ? "s" : ""} restored`,
      );
    }
    if (failed > 0) {
      toast.error(
        `${failed} document${failed > 1 ? "s" : ""} could not be restored`,
      );
    }
    clearSelection();
    setBulkRestoring(false);
    void loadTrash();
    await loadDocuments({ silent: true });
  }, [selectedIds, clearSelection, loadDocuments, loadTrash]);

  const deleteSelectedForever = useCallback(async () => {
    if (selectedIds.size === 0) return;
    if (
      !window.confirm(
        `Permanently delete ${selectedIds.size} document${selectedIds.size > 1 ? "s" : ""}? This cannot be undone.`,
      )
    ) {
      return;
    }
    setBulkDeleting(true);
    const results = await Promise.allSettled(
      Array.from(selectedIds).map((id) =>
        api.delete(`/api/v1/documents/${id}/permanent`),
      ),
    );
    const failed = results.filter((r) => r.status === "rejected").length;
    const succeeded = results.length - failed;
    if (succeeded > 0) {
      toast.success(
        `${succeeded} document${succeeded > 1 ? "s" : ""} permanently deleted`,
      );
    }
    if (failed > 0) {
      toast.error(
        `${failed} document${failed > 1 ? "s" : ""} could not be deleted`,
      );
    }
    clearSelection();
    setBulkDeleting(false);
    void loadTrash();
    await loadDocuments({ silent: true });
  }, [selectedIds, clearSelection, loadDocuments, loadTrash]);

  const emptyTrash = useCallback(async () => {
    if (trashDocuments.length === 0) return;
    if (
      !window.confirm(
        `Empty trash and permanently delete all ${trashDocuments.length} document${trashDocuments.length > 1 ? "s" : ""}? This cannot be undone.`,
      )
    ) {
      return;
    }
    setEmptyingTrash(true);
    try {
      const res = await api.delete<{ message?: string; deleted?: number }>(
        "/api/v1/documents/trash",
      );
      toast.success(res?.message || "Trash emptied");
      clearSelection();
      void loadTrash();
      await loadDocuments({ silent: true });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to empty trash";
      toast.error(message);
    } finally {
      setEmptyingTrash(false);
    }
  }, [trashDocuments.length, clearSelection, loadDocuments, loadTrash]);

  return {
    documents,
    trashDocuments,
    visibleDocuments,
    documentsLoading,
    trashLoading,
    connectionError,
    tab,
    setTab,
    statusFilter,
    setStatusFilter,
    searchQuery,
    setSearchQuery,
    counts,
    selectedIds,
    isAllSelected,
    toggleSelectOne,
    toggleSelectAll,
    clearSelection,
    upload: uploadFiles,
    openFilePicker,
    getInputProps,
    uploading,
    uploadProgress,
    uploadError,
    clearUploadError: () => setUploadError(""),
    refresh,
    singleDeletingId,
    singleRestoringId,
    singlePermanentDeletingId,
    bulkDeleting,
    bulkRestoring,
    emptyingTrash,
    deleteDocument,
    restoreDocument,
    deleteDocumentForever,
    deleteSelected,
    restoreSelected,
    deleteSelectedForever,
    emptyTrash,
  };
}
