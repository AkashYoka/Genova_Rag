"use client";

import { useState, useCallback, useEffect } from "react";
import { useTranslation } from "react-i18next";
import type { DocInfo } from "@/types/document";
import { normalizeDocInfo, type DocumentListApiResponse } from "@/types/document";
import { api } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  FileText,
  Upload,
  UploadCloud,
  Trash2,
  FileCheck,
  Clock,
  AlertCircle,
  Loader2,
  FolderOpen,
  Cloud,
  RefreshCw,
} from "lucide-react";
import { useDropzone } from "react-dropzone";
import { Settings } from "lucide-react";
import DocumentSettings from "./DocumentSettings";
import DocumentCard from "./DocumentCard";
import BulkActionBar from "./BulkActionBar";
import { toast } from "sonner";

interface Props {
  documents: DocInfo[];
  activeDoc: DocInfo | null;
  loading?: boolean;
  onSelectDoc: (doc: DocInfo) => void;
  onDocumentsChange: () => void;
  onDocumentRenamed: (doc: DocInfo) => void;
}

function DocumentListSkeleton() {
  return (
    <div className="space-y-1 pb-3" aria-hidden="true">
      {Array.from({ length: 6 }).map((_, index) => (
        <div
          key={index}
          className="rounded-md border border-sidebar-border/60 px-2 py-1.5"
        >
          <div className="flex items-center gap-2">
            <Skeleton className="h-3.5 w-3.5 shrink-0 rounded-full bg-sidebar-accent" />
            <div className="min-w-0 flex-1 space-y-1">
              <Skeleton className="h-3.5 w-3/5 bg-sidebar-accent" />
              <Skeleton className="h-2.5 w-full bg-sidebar-accent/80" />
              <Skeleton className="h-2.5 w-2/5 bg-sidebar-accent/70" />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function DocumentSidebar({
  documents = [],
  activeDoc,
  loading = false,
  onSelectDoc,
  onDocumentsChange,
  onDocumentRenamed,
}: Props) {
  const { t } = useTranslation();
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadError, setUploadError] = useState("");
  const [deleting, setDeleting] = useState<string | null>(null);
  const [settingsDoc, setSettingsDoc] = useState<DocInfo | null>(null);
  const [editingDocId, setEditingDocId] = useState<string | null>(null);
  const [draftName, setDraftName] = useState("");
  const [renamingDocId, setRenamingDocId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("active");
  const [trashDocuments, setTrashDocuments] = useState<DocInfo[]>([]);
  const [loadingTrash, setLoadingTrash] = useState(false);
  const [driveConnected, setDriveConnected] = useState(false);
  const [driveLoading, setDriveLoading] = useState(true);
  const [driveConnecting, setDriveConnecting] = useState(false);
  const [driveError, setDriveError] = useState("");

  // ── Multi-select state ───────────────────────────────────────────────────
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [bulkDeleting, setBulkDeleting] = useState(false);
  const [bulkRestoring, setBulkRestoring] = useState(false);
  const [emptyingTrash, setEmptyingTrash] = useState(false);
  const [permanentDeletingId, setPermanentDeletingId] = useState<string | null>(
    null,
  );

  // Clear selection when switching tabs
  const handleTabChange = (value: string) => {
    setActiveTab(value);
    setSelectedIds(new Set());
    if (value === "trash") {
      void fetchTrash();
    }
  };

  const toggleSelect = (
    docId: string,
    e: React.MouseEvent | React.ChangeEvent,
  ) => {
    e.stopPropagation();
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(docId)) {
        next.delete(docId);
      } else {
        next.add(docId);
      }
      return next;
    });
  };

  const clearSelection = () => setSelectedIds(new Set());

  // ── Bulk delete (active tab) ─────────────────────────────────────────────
  const handleBulkDelete = async () => {
    if (selectedIds.size === 0) return;
    if (
      !confirm(
        `Delete ${selectedIds.size} document${selectedIds.size > 1 ? "s" : ""}? They will be moved to trash.`,
      )
    )
      return;

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
    await onDocumentsChange();
  };

  // ── Bulk restore (trash tab) ─────────────────────────────────────────────
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
    void fetchTrash();
    await onDocumentsChange();
  };

  // ── Bulk permanent delete (trash tab) ────────────────────────────────────
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
    void fetchTrash();
    await onDocumentsChange();
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
      const res = await api.delete<{ message: string; deleted: number }>(
        "/api/v1/documents/trash",
      );
      toast.success(res.message || "Trash emptied");
      clearSelection();
      void fetchTrash();
      await onDocumentsChange();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to empty trash";
      toast.error(message);
    } finally {
      setEmptyingTrash(false);
    }
  };

  const handlePermanentDelete = async (docId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const doc = trashDocuments.find((d) => d.id === docId);
    const label = doc?.original_name || "this document";
    if (
      !window.confirm(
        `Permanently delete "${label}"? This cannot be undone.`,
      )
    ) {
      return;
    }

    setPermanentDeletingId(docId);
    try {
      await api.delete(`/api/v1/documents/${docId}/permanent`);
      toast.success("Document permanently deleted");
      setSelectedIds((prev) => {
        const next = new Set(prev);
        next.delete(docId);
        return next;
      });
      void fetchTrash();
      await onDocumentsChange();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Delete forever failed";
      toast.error(message);
    } finally {
      setPermanentDeletingId(null);
    }
  };

  const fetchTrash = useCallback(async () => {
    setLoadingTrash(true);
    try {
      const res = await api.get<DocumentListApiResponse>(
        "/api/v1/documents/trash",
      );
      const rawItems = res?.items ?? res?.documents ?? [];
      setTrashDocuments(rawItems.map(normalizeDocInfo));
    } catch (err) {
      console.error("Failed to fetch trash:", err);
    } finally {
      setLoadingTrash(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadDriveStatus() {
      try {
        const data = await api.get<{ connected: boolean }>(
          "/api/v1/auth/google-drive/status",
        );
        if (!cancelled) setDriveConnected(data.connected);
      } catch {
        if (!cancelled) setDriveError("Unable to load Google Drive status");
      } finally {
        if (!cancelled) setDriveLoading(false);
      }
    }

    void loadDriveStatus();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleRestore = async (docId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await api.post(`/api/v1/documents/${docId}/restore`);
      toast.success("Document restored");
      void fetchTrash();
      onDocumentsChange();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Restore failed";
      toast.error(`❌ ${message}`);
    }
  };

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

            toast.info(`⏳ Uploading '${file.name}'...`);
            await api.postForm("/api/v1/documents/upload", formData);
            setUploadProgress(((i + 1) / acceptedFiles.length) * 100);
            toast.success(
              `📤 '${file.name}' uploaded successfully! Ingestion started.`,
            );
          }
          await onDocumentsChange();
        } catch (err) {
          const message =
            err instanceof Error ? err.message : t("documents.uploadFailed");
          setUploadError(message);
          toast.error(`❌ Upload failed: ${message}`);
        } finally {
          setUploading(false);
          setUploadProgress(0);
        }
      })();
    },
    [onDocumentsChange, t],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        [".docx"],
      "text/plain": [".txt"],
      "text/markdown": [".md"],
    },
    disabled: uploading,
  });

  const handleDelete = async (docId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm(t("documents.deleteConfirm"))) return;
    setDeleting(docId);
    try {
      await api.delete(`/api/v1/documents/${docId}`);
      await onDocumentsChange();
    } catch (err) {
      console.error("Delete failed:", err);
    } finally {
      setDeleting(null);
    }
  };

  const handleSettingsClick = (doc: DocInfo, e: React.MouseEvent) => {
    e.stopPropagation();
    setSettingsDoc(doc);
  };

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

    const optimisticDoc = { ...doc, original_name: nextName };
    onDocumentRenamed(optimisticDoc);
    setRenamingDocId(doc.id);

    try {
      const updatedDoc = await api.renameDocument<DocInfo>(doc.id, nextName);
      onDocumentRenamed(updatedDoc);
      cancelRename();
      toast.success("Document renamed");
    } catch (err) {
      onDocumentRenamed(doc);
      const message = err instanceof Error ? err.message : "Rename failed";
      toast.error(message);
    } finally {
      setRenamingDocId(null);
    }
  };

  const handleRenameKeyDown = (
    doc: DocInfo,
    e: React.KeyboardEvent<HTMLInputElement>,
  ) => {
    if (e.key === "Enter") {
      e.preventDefault();
      void submitRename(doc);
    } else if (e.key === "Escape") {
      e.preventDefault();
      cancelRename();
    }
  };

  const handleDocumentKeyDown = (doc: DocInfo, e: React.KeyboardEvent) => {
    if (editingDocId === doc.id || doc.status !== "ready") return;
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onSelectDoc(doc);
    }
  };

  const handleConnectDrive = async () => {
    setDriveConnecting(true);
    setDriveError("");

    try {
      const data = await api.get<{ auth_url: string }>(
        "/api/v1/auth/google-drive/connect",
      );
      window.location.assign(data.auth_url);
    } catch (err) {
      setDriveError(
        err instanceof Error ? err.message : "Failed to connect Google Drive",
      );
      setDriveConnecting(false);
    }
  };

  const handleDisconnectDrive = async () => {
    setDriveConnecting(true);
    setDriveError("");

    try {
      const data = await api.delete<{ connected: boolean }>(
        "/api/v1/auth/google-drive/disconnect",
      );
      setDriveConnected(data.connected);
    } catch (err) {
      setDriveError(
        err instanceof Error
          ? err.message
          : "Failed to disconnect Google Drive",
      );
    } finally {
      setDriveConnecting(false);
    }
  };

  const statusIcon = (status: string) => {
    switch (status) {
      case "ready":
        return <FileCheck className="w-3.5 h-3.5 text-emerald-400" />;
      case "processing":
        return <Loader2 className="w-3.5 h-3.5 text-primary animate-spin" />;
      case "pending":
        return <Clock className="w-3.5 h-3.5 text-yellow-400" />;
      case "failed":
        return <AlertCircle className="w-3.5 h-3.5 text-destructive" />;
      default:
        return <FileText className="w-3.5 h-3.5" />;
    }
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1048576).toFixed(1)} MB`;
  };

  return (
    <div className="h-full flex flex-col bg-transparent">
      {/* ── Upload Zone ─────────────────────────────── */}
      <div className="p-3 border-b border-sidebar-border space-y-2">
        {uploadError && (
          <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/30 text-sm text-destructive">
            {uploadError}
          </div>
        )}
        <div
          {...getRootProps()}
          className={`relative rounded-2xl border-2 border-dashed p-4 text-center cursor-pointer transition-all duration-200
            ${isDragActive ? "border-primary bg-primary/10 scale-[1.02]" : "border-sidebar-border hover:border-primary/40 hover:bg-sidebar-accent/50"}
            ${uploading ? "pointer-events-none opacity-60" : ""}`}
          aria-label="Upload documents"
        >
          <input {...getInputProps()} />
          {uploading ? (
            <div className="space-y-2">
              <Loader2 className="w-5 h-5 mx-auto animate-spin text-primary" />
              <p className="text-xs text-muted-foreground">
                {t("documents.uploading")}
              </p>
              <Progress value={uploadProgress} className="h-1" />
            </div>
          ) : (
            <>
              {isDragActive ? (
                <UploadCloud className="w-6 h-6 mx-auto text-primary mb-2 animate-bounce" />
              ) : (
                <Upload className="w-5 h-5 mx-auto text-muted-foreground mb-2" />
              )}
              <p
                className={`text-xs transition-colors ${isDragActive ? "text-primary font-medium" : "text-muted-foreground"}`}
                aria-live="polite"
              >
                {isDragActive
                  ? t("documents.dropHere")
                  : t("documents.dropOrClick")}
              </p>
              <p className="text-[10px] text-muted-foreground/60 mt-1">
                {t("documents.uploadFormats")}
              </p>
            </>
          )}
        </div>


      </div>

      {/* ── Documents List ──────────────────────────── */}
      <Tabs
        value={activeTab}
        onValueChange={handleTabChange}
        className="flex-1 flex flex-col min-h-0"
      >
        <div className="px-3 pt-3 pb-1 flex items-center justify-between gap-2">
          <h3 className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
            {activeTab === "active"
              ? loading
                ? t("documents.documentsTitle", { count: "..." })
                : t("documents.documentsTitle", { count: documents.length })
              : loadingTrash
                ? "TRASH (...)"
                : `TRASH (${trashDocuments.length})`}
          </h3>
          <div className="flex items-center gap-1.5">
            {activeTab === "trash" && trashDocuments.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                className="h-7 text-[10px] px-2 text-destructive hover:text-destructive hover:bg-destructive/10"
                onClick={() => void handleEmptyTrash()}
                disabled={emptyingTrash || bulkDeleting || bulkRestoring}
                aria-label="Empty trash"
                title="Empty trash"
              >
                {emptyingTrash ? (
                  <Loader2 className="w-3 h-3 animate-spin mr-1" />
                ) : (
                  <Trash2 className="w-3 h-3 mr-1" />
                )}
                Empty trash
              </Button>
            )}
            <TabsList className="h-7 bg-sidebar-accent">
              <TabsTrigger value="active" className="text-[10px] px-2 py-0.5">
                Active
              </TabsTrigger>
              <TabsTrigger value="trash" className="text-[10px] px-2 py-0.5">
                Trash
              </TabsTrigger>
            </TabsList>
          </div>
        </div>

        {/* ── Bulk Action Bar ──────────────────────── */}
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
          mode={activeTab as "active" | "trash"}
        />

        <TabsContent
          value="active"
          className="m-0 flex min-h-0 flex-1 flex-col overflow-hidden"
        >
          <div
            className="min-h-0 flex-1 basis-0 overflow-y-auto overscroll-contain px-3"
            aria-busy={loading}
          >
            {loading ? (
              <DocumentListSkeleton />
            ) : documents.length === 0 ? (
              <div className="text-center py-12">
                <FolderOpen className="w-8 h-8 mx-auto text-muted-foreground/40 mb-3" />
                <p className="text-sm text-muted-foreground">
                  {t("documents.noDocuments")}
                </p>
                <p className="text-xs text-muted-foreground/60 mt-1">
                  {t("documents.getStarted")}
                </p>
              </div>
            ) : (
              <div className="space-y-1 pb-3">
                {documents.map((doc) => {
                  const isEditing = editingDocId === doc.id;
                  const isRenaming = renamingDocId === doc.id;
                  const isSelected = selectedIds.has(doc.id);

                  return (
                    <div
                      key={doc.id}
                      role="button"
                      tabIndex={doc.status === "ready" ? 0 : -1}
                      aria-disabled={doc.status !== "ready"}
                      aria-current={
                        activeDoc?.id === doc.id ? "true" : undefined
                      }
                      aria-label={`Select document ${doc.original_name}. Status: ${doc.status}`}
                      onClick={() =>
                        doc.status === "ready" && !isEditing && onSelectDoc(doc)
                      }
                      onKeyDown={(e) => handleDocumentKeyDown(doc, e)}
                      className={`w-full text-left px-2 py-1.5 rounded-md transition-colors duration-150 group
                        ${
                          activeDoc?.id === doc.id
                            ? "bg-primary/15 border border-primary/30"
                            : isSelected
                              ? "bg-primary/8 border border-primary/20"
                              : "hover:bg-sidebar-accent border border-transparent"
                        }
                        ${doc.status !== "ready" ? "opacity-60 cursor-default" : "cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"}`}
                    >
                      <div className="flex items-center gap-2">
                        {/* ── Checkbox ── */}
                        <input
                          type="checkbox"
                          aria-label={`Select ${doc.original_name}`}
                          checked={isSelected}
                          onChange={(e) => toggleSelect(doc.id, e)}
                          onClick={(e) => e.stopPropagation()}
                          className="h-3.5 w-3.5 shrink-0 rounded-sm border border-muted-foreground/40 accent-primary cursor-pointer opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity"
                          style={isSelected ? { opacity: 1 } : undefined}
                        />

                        <span className="shrink-0">{statusIcon(doc.status)}</span>
                        <div className="flex-1 min-w-0">
                          {isEditing ? (
                            <Input
                              value={draftName}
                              onChange={(e) => setDraftName(e.target.value)}
                              onClick={(e) => e.stopPropagation()}
                              onKeyDown={(e) => handleRenameKeyDown(doc, e)}
                              disabled={isRenaming}
                              autoFocus
                              className="h-7 px-2 text-sm font-medium"
                            />
                          ) : (
                            <p
                              className="text-[13px] font-medium truncate leading-tight"
                              onDoubleClick={(e) => startRename(doc, e)}
                              title="Double-click to rename"
                            >
                              {doc.original_name}
                            </p>
                          )}

                          {doc.summary ? (
                            <p
                              className="mt-0.5 text-[11px] leading-snug text-muted-foreground line-clamp-1"
                              title={doc.summary}
                            >
                              {doc.summary}
                            </p>
                          ) : null}

                          <div className="mt-0.5 flex min-w-0 items-center gap-x-1.5 text-[10px] text-muted-foreground">
                            <span className="shrink-0 tabular-nums">
                              {formatSize(doc.file_size)}
                            </span>
                            {doc.status === "ready" && (
                              <>
                                <span className="shrink-0 opacity-40" aria-hidden>
                                  ·
                                </span>
                                <span className="shrink-0 tabular-nums">
                                  {t("documents.pagesShort", {
                                    count: doc.page_count,
                                  })}
                                </span>
                                <span className="shrink-0 opacity-40" aria-hidden>
                                  ·
                                </span>
                                <span className="shrink-0 tabular-nums">
                                  {t("documents.chunks", {
                                    count: doc.chunk_count,
                                  })}
                                </span>
                              </>
                            )}
                            {doc.status === "processing" && (
                              <Badge
                                variant="secondary"
                                className="h-4 px-1.5 text-[9px] tabular-nums"
                                title={
                                  doc.processing_stage
                                    ? `${doc.processing_stage}${
                                        typeof doc.processing_progress ===
                                        "number"
                                          ? ` · ${doc.processing_progress}%`
                                          : ""
                                      }`
                                    : undefined
                                }
                              >
                                {typeof doc.processing_progress === "number"
                                  ? t("documents.processingPercent", {
                                      percent: Math.max(
                                        0,
                                        Math.min(100, doc.processing_progress),
                                      ),
                                    })
                                  : t("documents.processing")}
                              </Badge>
                            )}
                            {doc.status === "failed" && (
                              <Badge
                                variant="destructive"
                                className="h-4 px-1.5 text-[9px]"
                              >
                                {t("documents.failed")}
                              </Badge>
                            )}
                            <DocumentCard document={doc} compact />
                          </div>
                        </div>
                        <div className="flex items-center gap-1 shrink-0">
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity cursor-pointer"
                            onClick={(e) => handleSettingsClick(doc, e)}
                            disabled={doc.status !== "ready"}
                            aria-label="Open chunking settings"
                            title={`Open chunking settings for ${doc.original_name}`}
                          >
                            <Settings className="w-3 h-3" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity shrink-0 cursor-pointer"
                            onClick={(e) => handleDelete(doc.id, e)}
                            disabled={deleting === doc.id}
                            aria-label="Delete document"
                            title={`Delete ${doc.original_name}`}
                          >
                            {deleting === doc.id ? (
                              <Loader2 className="w-3 h-3 animate-spin" />
                            ) : (
                              <Trash2 className="w-3 h-3 text-destructive" />
                            )}
                          </Button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </TabsContent>

        <TabsContent
          value="trash"
          className="m-0 flex min-h-0 flex-1 flex-col overflow-hidden"
        >
          <div
            className="min-h-0 flex-1 basis-0 overflow-y-auto overscroll-contain px-3"
            aria-busy={loadingTrash}
          >
            {loadingTrash ? (
              <DocumentListSkeleton />
            ) : trashDocuments.length === 0 ? (
              <div className="text-center py-12">
                <Trash2 className="w-8 h-8 mx-auto text-muted-foreground/40 mb-3" />
                <p className="text-sm text-muted-foreground">Trash is empty</p>
              </div>
            ) : (
              <div className="space-y-1 pb-3">
                {trashDocuments.map((doc) => {
                  const isSelected = selectedIds.has(doc.id);

                  return (
                    <div
                      key={doc.id}
                      className={`w-full text-left p-2.5 rounded-lg border transition-all duration-200 group
                        ${isSelected ? "bg-primary/8 border-primary/20" : "border-transparent hover:bg-sidebar-accent"}`}
                    >
                      <div className="flex items-start gap-2.5">
                        {/* ── Checkbox ── */}
                        <input
                          type="checkbox"
                          aria-label={`Select ${doc.original_name}`}
                          checked={isSelected}
                          onChange={(e) => toggleSelect(doc.id, e)}
                          onClick={(e) => e.stopPropagation()}
                          className="mt-0.5 h-3.5 w-3.5 shrink-0 rounded-sm border border-muted-foreground/40 accent-primary cursor-pointer opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity"
                          style={isSelected ? { opacity: 1 } : undefined}
                        />

                        {statusIcon(doc.status)}
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium truncate leading-tight opacity-60">
                            {doc.original_name}
                          </p>
                          <div className="flex items-center gap-2 mt-1">
                            <span className="text-[10px] text-muted-foreground">
                              {formatSize(doc.file_size)}
                            </span>
                          </div>
                        </div>
                        <div className="flex items-center gap-1 shrink-0">
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
                            onClick={(e) => handleRestore(doc.id, e)}
                            title={`Restore ${doc.original_name}`}
                            aria-label={`Restore ${doc.original_name}`}
                            disabled={
                              emptyingTrash ||
                              permanentDeletingId === doc.id ||
                              bulkDeleting
                            }
                          >
                            <RefreshCw className="w-3 h-3 text-primary" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
                            onClick={(e) =>
                              void handlePermanentDelete(doc.id, e)
                            }
                            title={`Delete forever ${doc.original_name}`}
                            aria-label={`Delete forever ${doc.original_name}`}
                            disabled={
                              emptyingTrash ||
                              permanentDeletingId === doc.id ||
                              bulkDeleting
                            }
                          >
                            {permanentDeletingId === doc.id ? (
                              <Loader2 className="w-3 h-3 animate-spin text-destructive" />
                            ) : (
                              <Trash2 className="w-3 h-3 text-destructive" />
                            )}
                          </Button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </TabsContent>
      </Tabs>

      {/* ── Settings Modal ───────────────────────────── */}
      {settingsDoc && (
        <DocumentSettings
          document={settingsDoc}
          open={!!settingsDoc}
          onOpenChange={(open) => {
            if (!open) setSettingsDoc(null);
          }}
          onSettingsSaved={() => {
            onDocumentsChange();
            setSettingsDoc(null);
          }}
        />
      )}
    </div>
  );
}
