"use client";

import { useEffect, useState, useCallback, useRef, useTransition } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useDropzone } from "react-dropzone";
import { toast } from "sonner";
import {
  FileText,
  Upload,
  UploadCloud,
  Trash2,
  AlertCircle,
  Loader2,
  FolderOpen,
  Search,
  X,
  MessageSquare,
  Settings,
  RefreshCw,
  Edit2,
  Check,
  Layers,
  ArrowRight,
  HardDrive,
  FileCode,
} from "lucide-react";
import { useAuth } from "@/lib/auth";
import {
  api,
  CONNECTION_ERROR_BANNER_MESSAGE,
  CONNECTION_ERROR_MESSAGE,
} from "@/lib/api";
import { RequireAuth } from "@/components/auth/RequireAuth";
import Header from "@/components/layout/Header";
import DocumentSettings from "@/components/document/DocumentSettings";
import BulkActionBar from "@/components/document/BulkActionBar";
import ProcessingPipelineStatus from "@/components/document/ProcessingPipelineStatus";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { DocInfo, DocumentListApiResponse } from "@/types/document";
import { normalizeDocInfo } from "@/types/document";
import { queueDocChat } from "@/lib/chat-attachments";
// Shares the desk shell + top bar styles that Header expects (genova-desk-top).
import "../dashboard/dashboard.css";

function formatBytes(bytes: number): string {
  if (!bytes || bytes <= 0) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(dateStr: string): string {
  if (!dateStr) return "—";
  try {
    const d = new Date(dateStr);
    if (Number.isNaN(d.getTime())) return "—";
    return d.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  } catch {
    return "—";
  }
}

function DocumentLibraryPageContent() {
  const { user } = useAuth();
  const router = useRouter();
  const [, startTransition] = useTransition();

  // ── Document states ────────────────────────────────────────────────────────
  const [documents, setDocuments] = useState<DocInfo[]>([]);
  const [trashDocuments, setTrashDocuments] = useState<DocInfo[]>([]);
  const [activeDoc, setActiveDoc] = useState<DocInfo | null>(null);
  const [documentsLoading, setDocumentsLoading] = useState(true);
  const [trashLoading, setTrashLoading] = useState(false);
  const [connectionError, setConnectionError] = useState("");

  // ── Tabs & search ──────────────────────────────────────────────────────────
  const [activeTab, setActiveTab] = useState<"active" | "trash">("active");
  const [searchQuery, setSearchQuery] = useState("");

  // ── Selection & Bulk operations ───────────────────────────────────────────
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [bulkDeleting, setBulkDeleting] = useState(false);
  const [bulkRestoring, setBulkRestoring] = useState(false);
  const [emptyingTrash, setEmptyingTrash] = useState(false);

  // ── Item-level operations ─────────────────────────────────────────────────
  const [singleDeletingId, setSingleDeletingId] = useState<string | null>(null);
  const [singleRestoringId, setSingleRestoringId] = useState<string | null>(null);
  const [singlePermanentDeletingId, setSinglePermanentDeletingId] = useState<string | null>(null);
  const [settingsDoc, setSettingsDoc] = useState<DocInfo | null>(null);
  const [editingDocId, setEditingDocId] = useState<string | null>(null);
  const [draftName, setDraftName] = useState("");
  const [renamingDocId, setRenamingDocId] = useState<string | null>(null);

  // ── Upload state ───────────────────────────────────────────────────────────
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadError, setUploadError] = useState("");

  // ── Status tracker ref ─────────────────────────────────────────────────────
  const prevDocsRef = useRef<Record<string, string>>({});

  // ── Fetch active documents ────────────────────────────────────────────────
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
        const normalized = rawList.map(normalizeDocInfo);
        setDocuments(normalized);

        setActiveDoc((current) => {
          if (!current) return null;
          const found = normalized.find((d) => d.id === current.id);
          return found ?? null;
        });
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

  // ── Fetch trash documents ─────────────────────────────────────────────────
  const loadTrash = useCallback(async () => {
    setTrashLoading(true);
    try {
      const data = await api.get<DocumentListApiResponse>(
        "/api/v1/documents/trash",
      );
      const rawList = data?.items ?? data?.documents ?? [];
      setTrashDocuments(rawList.map(normalizeDocInfo));
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to load trash";
      toast.error(message);
    } finally {
      setTrashLoading(false);
    }
  }, []);

  // Initial load
  useEffect(() => {
    if (user) {
      void loadDocuments();
      void loadTrash();
    }
  }, [user, loadDocuments, loadTrash]);

  // Auto-refresh when any active document is ingesting (polling every 3s)
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

  // Track status transitions for toast notifications
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

  // ── Tab switcher ──────────────────────────────────────────────────────────
  const handleTabChange = (value: string) => {
    const nextTab = value === "trash" ? "trash" : "active";
    setActiveTab(nextTab);
    setSelectedIds(new Set());
    if (nextTab === "trash") {
      void loadTrash();
    } else {
      void loadDocuments();
    }
  };

  // ── Selection helpers ─────────────────────────────────────────────────────
  const currentList = activeTab === "active" ? documents : trashDocuments;
  const filteredList = currentList.filter((doc) => {
    if (!searchQuery.trim()) return true;
    const lower = searchQuery.toLowerCase();
    return (
      doc.original_name.toLowerCase().includes(lower) ||
      (doc.summary && doc.summary.toLowerCase().includes(lower))
    );
  });

  const isAllSelected =
    filteredList.length > 0 &&
    filteredList.every((d) => selectedIds.has(d.id));

  const toggleSelectAll = () => {
    if (isAllSelected) {
      setSelectedIds(new Set());
    } else {
      const allIds = new Set(filteredList.map((d) => d.id));
      setSelectedIds(allIds);
    }
  };

  const toggleSelectOne = (id: string, e?: React.MouseEvent | React.ChangeEvent) => {
    e?.stopPropagation();
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const clearSelection = () => setSelectedIds(new Set());

  // ── Upload handling ───────────────────────────────────────────────────────
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) return;

      void (async () => {
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
            toast.success(`'${file.name}' uploaded successfully. Ingestion started.`);
          }
          await loadDocuments();
        } catch (err) {
          const message =
            err instanceof Error ? err.message : "Document upload failed";
          setUploadError(message);
          toast.error(`Upload failed: ${message}`);
        } finally {
          setUploading(false);
          setUploadProgress(0);
        }
      })();
    },
    [loadDocuments],
  );

  const { getRootProps, getInputProps, open: openFilePicker, isDragActive } =
    useDropzone({
      onDrop,
      noClick: true,
      noKeyboard: true,
      accept: {
        "application/pdf": [".pdf"],
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
          [".docx"],
        "text/plain": [".txt"],
        "text/markdown": [".md"],
      },
      disabled: uploading,
    });

  // ── Renaming handling ─────────────────────────────────────────────────────
  const startRename = (doc: DocInfo, e?: React.MouseEvent) => {
    e?.stopPropagation();
    setEditingDocId(doc.id);
    setDraftName(doc.original_name);
  };

  const cancelRename = () => {
    setEditingDocId(null);
    setDraftName("");
  };

  const submitRename = async (doc: DocInfo) => {
    const nextName = draftName.trim();
    if (!nextName) {
      toast.error("Document name cannot be empty");
      return;
    }
    if (nextName === doc.original_name) {
      cancelRename();
      return;
    }

    const optimisticDoc: DocInfo = { ...doc, original_name: nextName };
    setDocuments((current) =>
      current.map((d) => (d.id === doc.id ? optimisticDoc : d)),
    );
    if (activeDoc?.id === doc.id) {
      setActiveDoc(optimisticDoc);
    }
    setRenamingDocId(doc.id);

    try {
      const updatedRaw = await api.renameDocument<DocInfo>(doc.id, nextName);
      const updated = normalizeDocInfo(updatedRaw);
      setDocuments((current) =>
        current.map((d) => (d.id === doc.id ? updated : d)),
      );
      if (activeDoc?.id === doc.id) {
        setActiveDoc(updated);
      }
      cancelRename();
      toast.success("Document renamed");
    } catch (err) {
      // Rollback
      setDocuments((current) =>
        current.map((d) => (d.id === doc.id ? doc : d)),
      );
      if (activeDoc?.id === doc.id) {
        setActiveDoc(doc);
      }
      const message = err instanceof Error ? err.message : "Rename failed";
      toast.error(message);
    } finally {
      setRenamingDocId(null);
    }
  };

  // ── Single item actions ───────────────────────────────────────────────────
  const handleDeleteSingle = async (docId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm("Move this document to trash?")) return;

    setSingleDeletingId(docId);
    try {
      await api.delete(`/api/v1/documents/${docId}`);
      toast.success("Document moved to trash");
      if (activeDoc?.id === docId) {
        setActiveDoc(null);
      }
      await loadDocuments();
      void loadTrash();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to delete document";
      toast.error(message);
    } finally {
      setSingleDeletingId(null);
    }
  };

  const handleRestoreSingle = async (docId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setSingleRestoringId(docId);
    try {
      await api.post(`/api/v1/documents/${docId}/restore`);
      toast.success("Document restored");
      void loadTrash();
      await loadDocuments();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to restore document";
      toast.error(message);
    } finally {
      setSingleRestoringId(null);
    }
  };

  const handlePermanentDeleteSingle = async (
    doc: DocInfo,
    e: React.MouseEvent,
  ) => {
    e.stopPropagation();
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
      await loadDocuments();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Permanent delete failed";
      toast.error(message);
    } finally {
      setSinglePermanentDeletingId(null);
    }
  };

  // ── Bulk operations ───────────────────────────────────────────────────────
  const handleBulkDelete = async () => {
    if (selectedIds.size === 0) return;
    if (
      !window.confirm(
        `Move ${selectedIds.size} document${selectedIds.size > 1 ? "s" : ""} to trash?`,
      )
    ) {
      return;
    }

    setBulkDeleting(true);
    const ids = Array.from(selectedIds);
    const results = await Promise.allSettled(
      ids.map((id) => api.delete(`/api/v1/documents/${id}`)),
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
    await loadDocuments();
    void loadTrash();
  };

  const handleBulkRestore = async () => {
    if (selectedIds.size === 0) return;

    setBulkRestoring(true);
    const ids = Array.from(selectedIds);
    const results = await Promise.allSettled(
      ids.map((id) => api.post(`/api/v1/documents/${id}/restore`)),
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
    await loadDocuments();
  };

  const handleBulkPermanentDelete = async () => {
    if (selectedIds.size === 0) return;
    if (
      !window.confirm(
        `Permanently delete ${selectedIds.size} document${selectedIds.size > 1 ? "s" : ""}? This cannot be undone.`,
      )
    ) {
      return;
    }

    setBulkDeleting(true);
    const ids = Array.from(selectedIds);
    const results = await Promise.allSettled(
      ids.map((id) => api.delete(`/api/v1/documents/${id}/permanent`)),
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
    await loadDocuments();
  };

  const handleEmptyTrash = async () => {
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
      toast.success(res?.message || "Trash emptied successfully");
      clearSelection();
      void loadTrash();
      await loadDocuments();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to empty trash";
      toast.error(message);
    } finally {
      setEmptyingTrash(false);
    }
  };

  // ── Open in workspace: new chat session with this PDF attached ────────────
  const handleOpenInWorkspace = (doc: DocInfo) => {
    if (doc.status !== "ready") {
      toast.error("Wait until this document finishes indexing.");
      return;
    }
    setActiveDoc(doc);
    queueDocChat({ id: doc.id, filename: doc.original_name });
    startTransition(() => {
      router.push(`/dashboard?docId=${encodeURIComponent(doc.id)}`);
    });
  };

  const isLoading = activeTab === "active" ? documentsLoading : trashLoading;
  const readyCount = documents.filter((d) => d.status === "ready").length;
  const processingCount = documents.filter(
    (d) => d.status === "pending" || d.status === "processing",
  ).length;

  return (
    <div
      {...getRootProps()}
      className="genova-desk relative flex h-dvh flex-col overflow-hidden bg-[var(--gd-ground)] text-[var(--gd-text)]"
    >
      <input {...getInputProps()} />

      {/* Global Application Header */}
      <Header />

      {/* Drag overlay across the entire workspace */}
      {isDragActive && (
        <div className="pointer-events-none fixed inset-0 z-50 flex flex-col items-center justify-center bg-[var(--gd-ground)]/95">
          <UploadCloud className="h-12 w-12 text-[var(--gd-muted)]" strokeWidth={1.5} />
          <p className="mt-4 font-display text-lg font-medium text-[var(--gd-text)]">
            Drop files to upload
          </p>
          <p className="text-sm text-[var(--gd-muted)]">
            PDF, DOCX, TXT, or Markdown
          </p>
        </div>
      )}

      {/* Connection error banner */}
      {connectionError ? (
        <div
          className="flex items-center justify-between border-b border-destructive/30 bg-destructive/10 px-6 py-2.5 text-sm text-destructive"
          role="alert"
        >
          <div className="flex items-center gap-2">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span>{connectionError}</span>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => void loadDocuments()}
            className="h-7 text-xs text-destructive hover:bg-destructive/20"
          >
            Retry
          </Button>
        </div>
      ) : null}

      {/* Main Workspace Body */}
      <main className="mx-auto flex w-full min-h-0 max-w-7xl flex-1 flex-col overflow-hidden px-4 py-6 sm:px-6 lg:px-10">
        {/* Workspace Top Header */}
        <div className="mb-6 flex shrink-0 flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="flex items-baseline gap-3">
              <h1 className="font-display text-2xl font-medium tracking-tight text-[var(--gd-text)]">
                Documents
              </h1>
              <span className="text-sm tabular-nums text-[var(--gd-faint)]">
                {documents.length}
              </span>
            </div>
            <p className="mt-1 text-sm text-[var(--gd-muted)]">
              Upload, index, and manage your sources.
              {processingCount > 0 ? (
                <span className="ml-2 inline-flex items-center gap-1.5 text-xs text-[var(--gd-muted)]">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  {processingCount} indexing
                </span>
              ) : readyCount > 0 ? (
                <span className="ml-2 text-xs text-[var(--gd-faint)]">
                  {readyCount} ready for chat
                </span>
              ) : null}
            </p>
          </div>

          <div className="flex items-center gap-2.5">
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                if (activeTab === "active") void loadDocuments();
                else void loadTrash();
              }}
              disabled={isLoading}
              className="h-9 gap-1.5 rounded-lg text-xs cursor-pointer"
              title="Refresh document list"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${isLoading ? "animate-spin" : ""}`} />
              <span className="hidden sm:inline">Refresh</span>
            </Button>

            <Link href="/dashboard">
              <Button
                variant="outline"
                size="sm"
                className="h-9 gap-1.5 rounded-lg text-xs cursor-pointer"
                title="Open Chat Assistant"
              >
                <MessageSquare className="h-3.5 w-3.5" />
                <span>Open chat</span>
              </Button>
            </Link>

            <Button
              variant="default"
              size="sm"
              onClick={openFilePicker}
              disabled={uploading}
              className="h-9 gap-2 rounded-lg bg-[var(--gd-primary)] font-medium text-[var(--gd-primary-ink)] shadow-none hover:bg-[var(--gd-primary)]/90 cursor-pointer"
              title="Upload new documents"
            >
              {uploading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Upload className="h-4 w-4" />
              )}
              <span>Upload</span>
            </Button>
          </div>
        </div>

        {/* Upload in-progress banner */}
        {uploading && (
          <div className="mb-6 shrink-0 rounded-lg border border-[var(--gd-border)] bg-[var(--gd-surface)] p-4">
            <div className="mb-2 flex items-center justify-between text-xs text-[var(--gd-muted)]">
              <span className="flex items-center gap-1.5">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Uploading…
              </span>
              <span>{Math.round(uploadProgress)}%</span>
            </div>
            <Progress value={uploadProgress} className="h-1.5" />
          </div>
        )}

        {/* Upload error banner */}
        {uploadError && (
          <div className="mb-6 flex shrink-0 items-center justify-between rounded-lg border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive">
            <div className="flex items-center gap-2">
              <AlertCircle className="h-4 w-4 shrink-0" />
              <span>{uploadError}</span>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setUploadError("")}
              className="h-7 text-xs text-destructive hover:bg-destructive/20"
            >
              Dismiss
            </Button>
          </div>
        )}

        {/* Search & Tabs Toolbar */}
        <div className="mb-6 flex shrink-0 flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              type="text"
              placeholder="Search documents"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="h-9 rounded-lg border-[var(--gd-border)] bg-[var(--gd-surface)] pl-9 pr-8 text-sm"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => setSearchQuery("")}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground cursor-pointer"
                title="Clear search"
                aria-label="Clear search"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </div>

          <div className="flex items-center gap-2 self-start sm:self-auto">
            {activeTab === "trash" && trashDocuments.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => void handleEmptyTrash()}
                disabled={emptyingTrash || bulkDeleting || bulkRestoring}
                className="h-8 text-xs text-destructive hover:bg-destructive/10 hover:text-destructive cursor-pointer"
                title="Permanently remove all documents from trash"
              >
                {emptyingTrash ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />
                ) : (
                  <Trash2 className="h-3.5 w-3.5 mr-1.5" />
                )}
                Empty trash
              </Button>
            )}

            <Tabs
              value={activeTab}
              onValueChange={handleTabChange}
              className="w-auto"
            >
              <TabsList className="h-9 rounded-lg bg-[var(--gd-raised)] p-1">
                <TabsTrigger
                  value="active"
                  className="h-7 rounded-md text-xs px-3 data-[state=active]:bg-[var(--gd-surface)] data-[state=active]:text-[var(--gd-text)] data-[state=active]:shadow-none"
                >
                  Active ({documents.length})
                </TabsTrigger>
                <TabsTrigger
                  value="trash"
                  className="h-7 rounded-md text-xs px-3 data-[state=active]:bg-[var(--gd-surface)] data-[state=active]:text-[var(--gd-text)] data-[state=active]:shadow-none"
                >
                  Trash ({trashDocuments.length})
                </TabsTrigger>
              </TabsList>
            </Tabs>
          </div>
        </div>

        {/* Bulk Action Bar */}
        <div className="shrink-0">
          <BulkActionBar
            selectedCount={selectedIds.size}
            onDelete={
              activeTab === "trash"
                ? handleBulkPermanentDelete
                : handleBulkDelete
            }
            onRestore={handleBulkRestore}
            onClear={clearSelection}
            deleting={bulkDeleting || emptyingTrash}
            restoring={bulkRestoring}
            mode={activeTab}
          />
        </div>

        {/* Documents Content View */}
        <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-lg border border-[var(--gd-border)] bg-[var(--gd-surface)]">
          {isLoading ? (
            <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-6" aria-busy="true">
              {Array.from({ length: 5 }).map((_, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between border-b border-border/40 pb-4 last:border-none"
                >
                  <div className="flex items-center gap-3 w-1/3">
                    <Skeleton className="h-4 w-4 rounded-sm" />
                    <Skeleton className="h-8 w-8 rounded-lg" />
                    <div className="space-y-1.5 flex-1">
                      <Skeleton className="h-3.5 w-3/4" />
                      <Skeleton className="h-2.5 w-1/2" />
                    </div>
                  </div>
                  <Skeleton className="h-4 w-16" />
                  <Skeleton className="h-3.5 w-16" />
                  <Skeleton className="h-3.5 w-12" />
                  <Skeleton className="h-3.5 w-24" />
                  <Skeleton className="h-8 w-20 rounded-md" />
                </div>
              ))}
            </div>
          ) : filteredList.length === 0 ? (
            <div className="min-h-0 flex-1 overflow-y-auto px-4 py-20 text-center">
              {searchQuery ? (
                <>
                  <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-[var(--gd-raised)] text-[var(--gd-muted)]">
                    <Search className="h-5 w-5" strokeWidth={1.5} />
                  </div>
                  <h3 className="font-display text-base font-medium text-[var(--gd-text)]">
                    No matches
                  </h3>
                  <p className="mx-auto mt-1 max-w-sm text-sm text-[var(--gd-muted)]">
                    Nothing matched &ldquo;{searchQuery}&rdquo;. Try another term or clear the search.
                  </p>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setSearchQuery("")}
                    className="mt-4 rounded-lg text-xs cursor-pointer"
                  >
                    Clear search
                  </Button>
                </>
              ) : activeTab === "trash" ? (
                <>
                  <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-[var(--gd-raised)] text-[var(--gd-muted)]">
                    <Trash2 className="h-5 w-5" strokeWidth={1.5} />
                  </div>
                  <h3 className="font-display text-base font-medium text-[var(--gd-text)]">
                    Trash is empty
                  </h3>
                  <p className="mx-auto mt-1 max-w-sm text-sm text-[var(--gd-muted)]">
                    Deleted documents appear here. Restore them or delete them permanently.
                  </p>
                </>
              ) : (
                <>
                  <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-[var(--gd-raised)] text-[var(--gd-muted)]">
                    <FolderOpen className="h-5 w-5" strokeWidth={1.5} />
                  </div>
                  <h3 className="font-display text-lg font-medium text-[var(--gd-text)]">
                    No documents yet
                  </h3>
                  <p className="mx-auto mt-1.5 max-w-md text-sm text-[var(--gd-muted)]">
                    Upload a PDF, DOCX, TXT, or Markdown file to ask questions with cited answers.
                  </p>
                  <div className="mt-6 flex items-center justify-center gap-3">
                    <Button
                      variant="default"
                      onClick={openFilePicker}
                      className="gap-2 rounded-lg bg-[var(--gd-primary)] text-[var(--gd-primary-ink)] shadow-none hover:bg-[var(--gd-primary)]/90 cursor-pointer"
                    >
                      <Upload className="h-4 w-4" />
                      <span>Upload</span>
                    </Button>
                  </div>
                </>
              )}
            </div>
          ) : (
            <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden">
              <table className="w-full table-fixed border-collapse text-left text-sm">
                <thead>
                  <tr className="text-xs font-medium text-[var(--gd-muted)]">
                    <th className="sticky top-0 z-10 border-b border-border w-10 bg-[var(--gd-surface)] py-3 pl-4 pr-2">
                      <input
                        type="checkbox"
                        checked={isAllSelected}
                        onChange={toggleSelectAll}
                        aria-label="Select all documents"
                        className="h-4 w-4 rounded-sm border border-muted-foreground/40 accent-primary cursor-pointer"
                      />
                    </th>
                    <th className="sticky top-0 z-10 border-b border-border bg-[var(--gd-surface)] py-3 px-3">
                      Name
                    </th>
                    <th className="sticky top-0 z-10 border-b border-border w-[22%] bg-[var(--gd-surface)] py-3 px-3 sm:w-[15%] md:w-[13%]">
                      Status
                    </th>
                    <th className="sticky top-0 z-10 border-b border-border hidden w-[9%] bg-[var(--gd-surface)] py-3 px-3 text-right sm:table-cell">
                      Size
                    </th>
                    <th className="sticky top-0 z-10 border-b border-border hidden w-[7%] bg-[var(--gd-surface)] py-3 px-3 text-right md:table-cell">
                      Pages
                    </th>
                    <th className="sticky top-0 z-10 border-b border-border hidden w-[8%] bg-[var(--gd-surface)] py-3 px-3 text-right md:table-cell">
                      Chunks
                    </th>
                    <th className="sticky top-0 z-10 border-b border-border hidden w-[11%] bg-[var(--gd-surface)] py-3 px-3 lg:table-cell">
                      Uploaded
                    </th>
                    <th className="sticky top-0 z-10 border-b border-border w-[26%] bg-[var(--gd-surface)] py-3 pl-3 pr-4 text-right sm:w-[20%] lg:w-[15%]">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/60">
                  {filteredList.map((doc) => {
                    const isSelected = selectedIds.has(doc.id);
                    const isEditing = editingDocId === doc.id;
                    const isRenaming = renamingDocId === doc.id;
                    const isActive = activeDoc?.id === doc.id;

                    return (
                      <tr
                        key={doc.id}
                        onClick={() => {
                          if (activeTab === "active" && doc.status === "ready" && !isEditing) {
                            setActiveDoc(doc);
                          }
                        }}
                        className={`group transition-colors duration-150 ${
                          isSelected
                            ? "bg-[var(--gd-active)]"
                            : isActive
                              ? "bg-[var(--gd-raised)]"
                              : "hover:bg-[var(--gd-raised)]/60"
                        } ${activeTab === "active" && doc.status === "ready" ? "cursor-pointer" : ""}`}
                      >
                        {/* Checkbox column */}
                        <td
                          className="py-3 pl-4 pr-2"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <input
                            type="checkbox"
                            checked={isSelected}
                            onChange={(e) => toggleSelectOne(doc.id, e)}
                            aria-label={`Select ${doc.original_name}`}
                            className="h-4 w-4 rounded-sm border border-muted-foreground/40 accent-primary cursor-pointer"
                          />
                        </td>

                        {/* Name & Summary column */}
                        <td className="py-3 px-3">
                          <div className="flex items-start gap-3">
                            <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[var(--gd-raised)] text-[var(--gd-muted)]">
                              {doc.original_name.endsWith(".pdf") ? (
                                <FileText className="h-4 w-4" />
                              ) : (
                                <FileCode className="h-4 w-4" />
                              )}
                            </div>

                            <div className="min-w-0 flex-1">
                              {isEditing ? (
                                <div
                                  className="flex items-center gap-1.5"
                                  onClick={(e) => e.stopPropagation()}
                                >
                                  <Input
                                    value={draftName}
                                    onChange={(e) => setDraftName(e.target.value)}
                                    onKeyDown={(e) => {
                                      if (e.key === "Enter") {
                                        e.preventDefault();
                                        void submitRename(doc);
                                      } else if (e.key === "Escape") {
                                        e.preventDefault();
                                        cancelRename();
                                      }
                                    }}
                                    disabled={isRenaming}
                                    autoFocus
                                    className="h-7 text-xs font-medium"
                                  />
                                  <Button
                                    variant="ghost"
                                    size="icon"
                                    onClick={() => void submitRename(doc)}
                                    disabled={isRenaming}
                                    className="h-7 w-7 text-[var(--gd-text)] hover:bg-[var(--gd-raised)] cursor-pointer"
                                    title="Save name"
                                  >
                                    <Check className="h-3.5 w-3.5" />
                                  </Button>
                                  <Button
                                    variant="ghost"
                                    size="icon"
                                    onClick={cancelRename}
                                    disabled={isRenaming}
                                    className="h-7 w-7 text-muted-foreground hover:bg-muted cursor-pointer"
                                    title="Cancel"
                                  >
                                    <X className="h-3.5 w-3.5" />
                                  </Button>
                                </div>
                              ) : (
                                <div className="flex min-w-0 items-start gap-2">
                                  <p
                                    className="min-w-0 break-words text-xs font-semibold text-foreground"
                                    title={doc.original_name}
                                  >
                                    {doc.original_name}
                                  </p>
                                  {activeTab === "active" && (
                                    <button
                                      type="button"
                                      onClick={(e) => startRename(doc, e)}
                                      className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-foreground transition-opacity cursor-pointer p-0.5"
                                      title="Rename document"
                                      aria-label={`Rename ${doc.original_name}`}
                                    >
                                      <Edit2 className="h-3 w-3" />
                                    </button>
                                  )}
                                </div>
                              )}

                              {doc.summary ? (
                                <p
                                  className="mt-1 line-clamp-2 break-words text-[11px] leading-relaxed text-muted-foreground"
                                  title={doc.summary}
                                >
                                  {doc.summary}
                                </p>
                              ) : null}

                              {/* Keywords tags preview if available */}
                              {doc.keywords && doc.keywords.length > 0 && (
                                <div className="mt-1.5 flex flex-wrap gap-1">
                                  {doc.keywords.slice(0, 3).map((kw, idx) => (
                                    <span
                                      key={idx}
                                      className="inline-block rounded bg-muted/60 px-1.5 py-0.2 text-[9px] text-muted-foreground"
                                    >
                                      #{kw}
                                    </span>
                                  ))}
                                  {doc.keywords.length > 3 && (
                                    <span className="text-[9px] text-muted-foreground">
                                      +{doc.keywords.length - 3} more
                                    </span>
                                  )}
                                </div>
                              )}
                            </div>
                          </div>
                        </td>

                        {/* Status column — hover shows ingest pipeline stages */}
                        <td className="py-3 px-3">
                          <ProcessingPipelineStatus
                            status={doc.status}
                            processingStage={doc.processing_stage}
                            processingProgress={doc.processing_progress}
                            errorMessage={doc.error_message}
                          />
                        </td>

                        {/* File size column */}
                        <td className="hidden py-3 px-3 text-right text-xs font-mono text-muted-foreground sm:table-cell">
                          {formatBytes(doc.file_size)}
                        </td>

                        {/* Page count column */}
                        <td className="hidden py-3 px-3 text-right text-xs font-mono text-muted-foreground md:table-cell">
                          {doc.page_count > 0 ? doc.page_count : "—"}
                        </td>

                        {/* Chunk count column */}
                        <td className="hidden py-3 px-3 text-right text-xs font-mono text-muted-foreground md:table-cell">
                          {doc.chunk_count > 0 ? (
                            <span className="text-[var(--gd-text)]">
                              {doc.chunk_count}
                            </span>
                          ) : (
                            "—"
                          )}
                        </td>

                        {/* Upload date column */}
                        <td className="hidden py-3 px-3 text-xs text-muted-foreground lg:table-cell">
                          {formatDate(doc.uploaded_at)}
                        </td>

                        {/* Actions column */}
                        <td
                          className="py-3 pl-3 pr-4 text-right"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <div className="flex flex-wrap items-center justify-end gap-1">
                            {activeTab === "active" ? (
                              <>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => handleOpenInWorkspace(doc)}
                                  disabled={doc.status !== "ready"}
                                  className="h-8 gap-1.5 text-xs text-[var(--gd-text)] hover:bg-[var(--gd-raised)] cursor-pointer px-2"
                                  title={`Open ${doc.original_name} in Chat Workspace`}
                                >
                                  <MessageSquare className="h-3.5 w-3.5" />
                                  <span className="hidden md:inline">Chat</span>
                                </Button>

                                <Button
                                  variant="ghost"
                                  size="icon"
                                  onClick={() => setSettingsDoc(doc)}
                                  disabled={doc.status !== "ready"}
                                  className="h-8 w-8 text-muted-foreground hover:text-foreground cursor-pointer"
                                  title={`Chunking settings for ${doc.original_name}`}
                                  aria-label={`Chunking settings for ${doc.original_name}`}
                                >
                                  <Settings className="h-3.5 w-3.5" />
                                </Button>

                                <Button
                                  variant="ghost"
                                  size="icon"
                                  onClick={(e) => void handleDeleteSingle(doc.id, e)}
                                  disabled={singleDeletingId === doc.id}
                                  className="h-8 w-8 text-muted-foreground hover:text-destructive hover:bg-destructive/10 cursor-pointer"
                                  title={`Move ${doc.original_name} to trash`}
                                  aria-label={`Move ${doc.original_name} to trash`}
                                >
                                  {singleDeletingId === doc.id ? (
                                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                                  ) : (
                                    <Trash2 className="h-3.5 w-3.5" />
                                  )}
                                </Button>
                              </>
                            ) : (
                              <>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={(e) => void handleRestoreSingle(doc.id, e)}
                                  disabled={singleRestoringId === doc.id}
                                  className="h-8 gap-1.5 text-xs text-[var(--gd-text)] hover:bg-[var(--gd-raised)] cursor-pointer px-2"
                                  title={`Restore ${doc.original_name}`}
                                >
                                  {singleRestoringId === doc.id ? (
                                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                                  ) : (
                                    <RefreshCw className="h-3.5 w-3.5" />
                                  )}
                                  <span>Restore</span>
                                </Button>

                                <Button
                                  variant="ghost"
                                  size="icon"
                                  onClick={(e) =>
                                    void handlePermanentDeleteSingle(doc, e)
                                  }
                                  disabled={singlePermanentDeletingId === doc.id}
                                  className="h-8 w-8 text-muted-foreground hover:text-destructive hover:bg-destructive/10 cursor-pointer"
                                  title={`Permanently delete ${doc.original_name}`}
                                  aria-label={`Permanently delete ${doc.original_name}`}
                                >
                                  {singlePermanentDeletingId === doc.id ? (
                                    <Loader2 className="h-3.5 w-3.5 animate-spin text-destructive" />
                                  ) : (
                                    <Trash2 className="h-3.5 w-3.5 text-destructive" />
                                  )}
                                </Button>
                              </>
                            )}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </main>

      {/* Document Chunking Settings Modal */}
      {settingsDoc && (
        <DocumentSettings
          document={settingsDoc}
          open={Boolean(settingsDoc)}
          onOpenChange={(open) => {
            if (!open) setSettingsDoc(null);
          }}
          onSettingsSaved={() => {
            void loadDocuments();
            setSettingsDoc(null);
          }}
        />
      )}
    </div>
  );
}

export default function DocumentLibraryPage() {
  return (
    <RequireAuth>
      <DocumentLibraryPageContent />
    </RequireAuth>
  );
}
