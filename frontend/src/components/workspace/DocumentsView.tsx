"use client";

/**
 * DocumentsView
 *
 * The library half of the unified screen. Same data, same actions and the same
 * API calls as the old /documents page, presented inside the Workspace shell so
 * people never lose their conversation while managing sources.
 */

import {
  AlertCircle,
  FileCode,
  FileText,
  FolderOpen,
  Info,
  ListTree,
  Loader2,
  MessageSquare,
  RefreshCw,
  Search,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import BulkActionBar from "@/components/document/BulkActionBar";
import ProcessingPipelineStatus from "@/components/document/ProcessingPipelineStatus";
import type { DocumentLibrary } from "@/hooks/useDocumentLibrary";
import type { DocInfo } from "@/types/document";
import { cn } from "@/lib/utils";

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

const FILTER_LABEL: Record<"ready" | "indexing", string> = {
  ready: "Active documents",
  indexing: "Documents being indexed",
};

interface DocumentsViewProps {
  library: DocumentLibrary;
  onOpenDocument: (doc: DocInfo) => void;
  onOpenInChat: (doc: DocInfo) => void;
  onBackToChat: () => void;
}

export default function DocumentsView({
  library,
  onOpenDocument,
  onOpenInChat,
  onBackToChat,
}: DocumentsViewProps) {
  const {
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
    selectedIds,
    isAllSelected,
    toggleSelectOne,
    toggleSelectAll,
  } = library;

  const isLoading = tab === "active" ? documentsLoading : trashLoading;
  const isTrash = tab === "trash";
  const processingCount = library.counts.indexing;

  const resetFilters = () => {
    setStatusFilter("all");
    setSearchQuery("");
  };

  const rowActions = (doc: DocInfo) => (
    <div className="inline-flex items-center justify-end gap-0.5">
      {!isTrash ? (
        <>
          <Button
            variant="ghost"
            size="icon"
            onClick={(e) => {
              e.stopPropagation();
              if (doc.status !== "ready") {
                toast.error("This document is still being indexed.");
                return;
              }
              const params = new URLSearchParams({
                docId: doc.id,
                name: doc.original_name,
              });
              if (doc.page_count > 0) params.set("pages", String(doc.page_count));
              window.location.assign(`/documents/explore?${params.toString()}`);
            }}
            disabled={doc.status !== "ready"}
            className="h-7 w-7 text-[var(--gd-muted)] hover:bg-[var(--gd-raised)] hover:text-[var(--gd-text)]"
            title={`Open the structure of ${doc.original_name}`}
            aria-label={`Open the structure of ${doc.original_name}`}
          >
            <ListTree className="h-3.5 w-3.5" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            onClick={(e) => {
              e.stopPropagation();
              onOpenInChat(doc);
            }}
            disabled={doc.status !== "ready"}
            className="h-7 w-7 text-[var(--gd-muted)] hover:bg-[var(--gd-raised)] hover:text-[var(--gd-text)]"
            title={`Start a new chat about ${doc.original_name}`}
            aria-label={`Start a new chat about ${doc.original_name}`}
          >
            <MessageSquare className="h-3.5 w-3.5" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            onClick={(e) => {
              e.stopPropagation();
              void library.deleteDocument(doc);
            }}
            disabled={library.singleDeletingId === doc.id}
            className="h-7 w-7 text-[var(--gd-muted)] hover:bg-[var(--gd-destructive)]/10 hover:text-[var(--gd-negative)]"
            title={`Move ${doc.original_name} to trash`}
            aria-label={`Move ${doc.original_name} to trash`}
          >
            {library.singleDeletingId === doc.id ? (
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
            size="icon"
            onClick={(e) => {
              e.stopPropagation();
              void library.restoreDocument(doc);
            }}
            disabled={library.singleRestoringId === doc.id}
            className="h-7 w-7 text-[var(--gd-muted)] hover:bg-[var(--gd-raised)] hover:text-[var(--gd-text)]"
            title={`Restore ${doc.original_name}`}
            aria-label={`Restore ${doc.original_name}`}
          >
            {library.singleRestoringId === doc.id ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <RefreshCw className="h-3.5 w-3.5" />
            )}
          </Button>
          <Button
            variant="ghost"
            size="icon"
            onClick={(e) => {
              e.stopPropagation();
              void library.deleteDocumentForever(doc);
            }}
            disabled={library.singlePermanentDeletingId === doc.id}
            className="h-7 w-7 text-[var(--gd-muted)] hover:bg-[var(--gd-destructive)]/10 hover:text-[var(--gd-negative)]"
            title={`Delete ${doc.original_name} permanently`}
            aria-label={`Delete ${doc.original_name} permanently`}
          >
            {library.singlePermanentDeletingId === doc.id ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Trash2 className="h-3.5 w-3.5" />
            )}
          </Button>
        </>
      )}
    </div>
  );

  const nameCell = (doc: DocInfo) => {
    return (
      <div className="flex min-w-0 items-start gap-2.5">
        <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[var(--gd-raised)] text-[var(--gd-muted)]">
          {doc.original_name.toLowerCase().endsWith(".pdf") ? (
            <FileText className="h-3.5 w-3.5" aria-hidden />
          ) : (
            <FileCode className="h-3.5 w-3.5" aria-hidden />
          )}
        </span>
        <div className="min-w-0 flex-1">
          <p
            className="min-w-0 truncate text-[13px] font-medium leading-5 text-[var(--gd-text)]"
            title={doc.original_name}
          >
            {doc.original_name}
          </p>

          {doc.summary ? (
            <p
              className="mt-0.5 line-clamp-1 break-words text-[11px] leading-4 text-[var(--gd-muted)]"
              title={doc.summary}
            >
              {doc.summary}
            </p>
          ) : null}

          {doc.keywords && doc.keywords.length > 0 ? (
            <div className="mt-1 flex flex-wrap items-center gap-1">
              {doc.keywords.slice(0, 2).map((kw, idx) => (
                <span
                  key={idx}
                  className="inline-block rounded bg-[var(--gd-raised)] px-1.5 py-px text-[10px] leading-4 text-[var(--gd-muted)]"
                >
                  #{kw}
                </span>
              ))}
              {doc.keywords.length > 2 ? (
                <span className="text-[10px] leading-4 text-[var(--gd-faint)]">
                  +{doc.keywords.length - 2}
                </span>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>
    );
  };

  return (
    <div className="genova-docs-view">
      <div className="genova-docs-canvas">
        <header className="genova-docs-toolbar shrink-0">
          <div className="flex min-w-0 items-center gap-2">
            <h1 className="font-display text-base font-medium tracking-tight text-[var(--gd-text)]">
              Documents
            </h1>
            <span className="num text-xs text-[var(--gd-faint)]">
              {documents.length}
            </span>
            {processingCount > 0 ? (
              <span className="inline-flex items-center gap-1 text-[11px] text-[var(--gd-warning)]">
                <Loader2 className="h-3 w-3 animate-spin" aria-hidden />
                {processingCount} indexing
              </span>
            ) : null}
          </div>

          <div className="relative min-w-0 flex-1 sm:max-w-sm">
            <Search
              className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-[var(--gd-faint)]"
              aria-hidden
            />
            <Input
              type="text"
              placeholder="Search documents"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              aria-label="Search documents"
              className="h-8 rounded-md border-[var(--gd-border)] bg-[var(--gd-surface)] pl-8 pr-8 text-xs"
            />
            {searchQuery ? (
              <button
                type="button"
                onClick={() => setSearchQuery("")}
                className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-0.5 text-[var(--gd-faint)] hover:text-[var(--gd-text)]"
                title="Clear search"
                aria-label="Clear search"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            ) : null}
          </div>

          <div className="flex shrink-0 flex-wrap items-center gap-1.5">
            {isTrash && trashDocuments.length > 0 ? (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => void library.emptyTrash()}
                disabled={
                  library.emptyingTrash ||
                  library.bulkDeleting ||
                  library.bulkRestoring
                }
                className="h-8 gap-1.5 rounded-md px-2 text-xs text-[var(--gd-negative)] hover:bg-[var(--gd-negative)]/10 hover:text-[var(--gd-negative)]"
                title="Permanently remove all documents in trash"
              >
                {library.emptyingTrash ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
                ) : (
                  <Trash2 className="h-3.5 w-3.5" aria-hidden />
                )}
                Empty trash
              </Button>
            ) : null}

            <Tabs
              value={tab}
              onValueChange={(value) =>
                setTab(value === "trash" ? "trash" : "active")
              }
              className="w-auto"
            >
              <TabsList className="h-8 rounded-md bg-[var(--gd-raised)] p-0.5">
                <TabsTrigger
                  value="active"
                  className="h-7 rounded px-2.5 text-xs data-[state=active]:bg-[var(--gd-surface)] data-[state=active]:text-[var(--gd-text)] data-[state=active]:shadow-none"
                >
                  Active
                  <span className="num ml-1.5 text-[var(--gd-faint)]">
                    {documents.length}
                  </span>
                </TabsTrigger>
                <TabsTrigger
                  value="trash"
                  className="h-7 rounded px-2.5 text-xs data-[state=active]:bg-[var(--gd-surface)] data-[state=active]:text-[var(--gd-text)] data-[state=active]:shadow-none"
                >
                  Trash
                  <span className="num ml-1.5 text-[var(--gd-faint)]">
                    {trashDocuments.length}
                  </span>
                </TabsTrigger>
              </TabsList>
            </Tabs>

            <Button
              variant="outline"
              size="sm"
              onClick={onBackToChat}
              className="h-8 gap-1.5 rounded-md text-xs"
              title="Return to your conversation"
              aria-label="Open chat"
            >
              <MessageSquare className="h-3.5 w-3.5" aria-hidden />
              <span className="hidden sm:inline">Open chat</span>
            </Button>
            <Button
              variant="outline"
              size="icon"
              onClick={library.refresh}
              disabled={isLoading}
              className="h-8 w-8 rounded-md"
              title="Refresh document list"
              aria-label="Refresh document list"
            >
              <RefreshCw
                className={cn("h-3.5 w-3.5", isLoading && "animate-spin")}
              />
            </Button>
            <Button
              onClick={library.openFilePicker}
              disabled={library.uploading}
              className="h-8 gap-2 rounded-md bg-[var(--gd-primary)] px-3 text-xs font-medium text-[var(--gd-primary-ink)] shadow-none hover:bg-[var(--gd-primary)]/90"
              title="Upload documents"
            >
              {library.uploading ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
              ) : (
                <Upload className="h-3.5 w-3.5" aria-hidden />
              )}
              Upload
            </Button>
          </div>
        </header>

        {connectionError ? (
          <div
            className="mt-2 flex shrink-0 items-center justify-between gap-3 rounded-md border border-[var(--gd-negative)]/30 bg-[var(--gd-negative)]/10 px-3 py-1.5 text-xs text-[var(--gd-negative)]"
            role="alert"
          >
            <span className="flex min-w-0 items-center gap-2">
              <AlertCircle className="h-3.5 w-3.5 shrink-0" aria-hidden />
              <span className="truncate">{connectionError}</span>
            </span>
            <Button
              variant="ghost"
              size="sm"
              onClick={library.refresh}
              className="h-6 shrink-0 px-2 text-xs text-[var(--gd-negative)] hover:bg-[var(--gd-negative)]/15"
            >
              Retry
            </Button>
          </div>
        ) : null}

        {library.uploading ? (
          <div className="mt-2 shrink-0 rounded-md border border-[var(--gd-border)] bg-[var(--gd-surface)] px-3 py-2">
            <div className="mb-1.5 flex items-center justify-between text-xs text-[var(--gd-muted)]">
              <span className="flex items-center gap-1.5">
                <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
                Uploading…
              </span>
              <span className="num">{Math.round(library.uploadProgress)}%</span>
            </div>
            <Progress value={library.uploadProgress} className="h-1" />
          </div>
        ) : null}

        {library.uploadError ? (
          <div className="mt-2 flex shrink-0 items-center justify-between gap-3 rounded-md border border-[var(--gd-negative)]/30 bg-[var(--gd-negative)]/10 px-3 py-1.5 text-xs text-[var(--gd-negative)]">
            <span className="flex min-w-0 items-center gap-2">
              <AlertCircle className="h-3.5 w-3.5 shrink-0" aria-hidden />
              <span className="truncate">{library.uploadError}</span>
            </span>
            <Button
              variant="ghost"
              size="sm"
              onClick={library.clearUploadError}
              className="h-6 shrink-0 px-2 text-xs text-[var(--gd-negative)] hover:bg-[var(--gd-negative)]/15"
            >
              Dismiss
            </Button>
          </div>
        ) : null}

        {statusFilter !== "all" && !isTrash ? (
          <div className="mt-2 flex shrink-0 items-center gap-2">
            <span className="inline-flex items-center gap-1.5 rounded-md border border-[var(--gd-border)] bg-[var(--gd-surface)] px-2 py-1 text-[11px] text-[var(--gd-muted)]">
              <Info className="h-3 w-3" aria-hidden />
              Showing {FILTER_LABEL[statusFilter]}
              <button
                type="button"
                onClick={() => setStatusFilter("all")}
                className="ml-0.5 rounded p-0.5 text-[var(--gd-faint)] hover:text-[var(--gd-text)]"
                aria-label="Clear document filter"
              >
                <X className="h-3 w-3" />
              </button>
            </span>
          </div>
        ) : null}

        <div className="mt-2 shrink-0">
          <BulkActionBar
            selectedCount={selectedIds.size}
            onDelete={
              isTrash ? library.deleteSelectedForever : library.deleteSelected
            }
            onRestore={library.restoreSelected}
            onClear={library.clearSelection}
            deleting={library.bulkDeleting || library.emptyingTrash}
            restoring={library.bulkRestoring}
            mode={isTrash ? "trash" : "active"}
          />
        </div>

        <div className="genova-docs-panel mt-2">
          {isLoading ? (
            <div className="space-y-3 p-4" aria-busy="true" aria-label="Loading documents">
              {Array.from({ length: 5 }).map((_, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between border-b border-[var(--gd-rule)] pb-3 last:border-none"
                >
                  <div className="flex w-1/3 items-center gap-3">
                    <Skeleton className="h-4 w-4 rounded-sm" />
                    <Skeleton className="h-7 w-7 rounded-md" />
                    <div className="flex-1 space-y-1.5">
                      <Skeleton className="h-3 w-3/4" />
                      <Skeleton className="h-2.5 w-1/2" />
                    </div>
                  </div>
                  <Skeleton className="h-3.5 w-16" />
                  <Skeleton className="h-3.5 w-16" />
                  <Skeleton className="h-3.5 w-12" />
                  <Skeleton className="h-7 w-24 rounded-md" />
                </div>
              ))}
            </div>
          ) : visibleDocuments.length === 0 ? (
            <div className="px-4 py-16 text-center">
              {searchQuery ? (
                <>
                  <span className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-md bg-[var(--gd-raised)] text-[var(--gd-muted)]">
                    <Search className="h-4 w-4" aria-hidden />
                  </span>
                  <h2 className="font-display text-base font-medium text-[var(--gd-text)]">
                    No matches
                  </h2>
                  <p className="mx-auto mt-1 max-w-sm text-sm text-[var(--gd-muted)]">
                    Nothing matched “{searchQuery}”. Try another term or clear
                    the search.
                  </p>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={resetFilters}
                    className="mt-4 h-8 rounded-md text-xs"
                  >
                    Clear search
                  </Button>
                </>
              ) : isTrash ? (
                <>
                  <span className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-md bg-[var(--gd-raised)] text-[var(--gd-muted)]">
                    <Trash2 className="h-4 w-4" aria-hidden />
                  </span>
                  <h2 className="font-display text-base font-medium text-[var(--gd-text)]">
                    Trash is empty
                  </h2>
                  <p className="mx-auto mt-1 max-w-sm text-sm text-[var(--gd-muted)]">
                    Documents you remove appear here, so you can restore them or
                    delete them for good.
                  </p>
                </>
              ) : statusFilter === "indexing" ? (
                <>
                  <span className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-md bg-[var(--gd-raised)] text-[var(--gd-muted)]">
                    <Loader2 className="h-4 w-4" aria-hidden />
                  </span>
                  <h2 className="font-display text-base font-medium text-[var(--gd-text)]">
                    Nothing is indexing
                  </h2>
                  <p className="mx-auto mt-1 max-w-sm text-sm text-[var(--gd-muted)]">
                    Every document is prepared. New uploads appear here while
                    they are being indexed.
                  </p>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setStatusFilter("all")}
                    className="mt-4 h-8 rounded-md text-xs"
                  >
                    Show all documents
                  </Button>
                </>
              ) : statusFilter === "ready" ? (
                <>
                  <span className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-md bg-[var(--gd-raised)] text-[var(--gd-muted)]">
                    <FileText className="h-4 w-4" aria-hidden />
                  </span>
                  <h2 className="font-display text-base font-medium text-[var(--gd-text)]">
                    No documents are ready yet
                  </h2>
                  <p className="mx-auto mt-1 max-w-sm text-sm text-[var(--gd-muted)]">
                    Finished documents appear here. You can ask about a file as
                    soon as it is ready.
                  </p>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setStatusFilter("all")}
                    className="mt-4 h-8 rounded-md text-xs"
                  >
                    Show all documents
                  </Button>
                </>
              ) : (
                <>
                  <span className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-md bg-[var(--gd-raised)] text-[var(--gd-muted)]">
                    <FolderOpen className="h-4 w-4" aria-hidden />
                  </span>
                  <h2 className="font-display text-base font-medium text-[var(--gd-text)]">
                    No documents yet
                  </h2>
                  <p className="mx-auto mt-1.5 max-w-md text-sm text-[var(--gd-muted)]">
                    Upload a PDF, DOCX, TXT, or Markdown file to ask questions
                    with answers that cite the page.
                  </p>
                  <Button
                    onClick={library.openFilePicker}
                    className="mt-5 h-8 gap-2 rounded-md bg-[var(--gd-primary)] px-3 text-xs font-medium text-[var(--gd-primary-ink)] shadow-none hover:bg-[var(--gd-primary)]/90"
                  >
                    <Upload className="h-3.5 w-3.5" aria-hidden />
                    Upload a document
                  </Button>
                </>
              )}
            </div>
          ) : (
            <>
              {/* Desktop table */}
              <table className="genova-docs-table hidden sm:table">
                <caption className="sr-only">
                  Documents with indexing status, size, pages, searchable
                  sections and upload date
                </caption>
                <thead>
                  <tr>
                    <th scope="col" className="w-10 pl-3 pr-1">
                      <input
                        type="checkbox"
                        checked={isAllSelected}
                        onChange={toggleSelectAll}
                        aria-label="Select all documents"
                        className="h-3.5 w-3.5 cursor-pointer rounded-sm border border-[var(--gd-faint)] accent-[var(--gd-accent)]"
                      />
                    </th>
                    <th scope="col">Name</th>
                    <th scope="col" className="w-[7rem]">
                      Status
                    </th>
                    <th scope="col" className="cell-num hidden w-[5rem] lg:table-cell">
                      Size
                    </th>
                    <th scope="col" className="cell-num hidden w-[4rem] lg:table-cell">
                      Pages
                    </th>
                    <th scope="col" className="cell-num hidden w-[5rem] lg:table-cell">
                      <span
                        className="inline-flex items-center gap-1">
                        Chunks
                      </span>
                    </th>
                    <th scope="col" className="hidden w-[7.5rem] xl:table-cell">
                      Uploaded
                    </th>
                    <th scope="col" className="w-[7rem] pr-3 text-right">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {visibleDocuments.map((doc) => {
                    const isSelected = selectedIds.has(doc.id);
                    const openable = !isTrash && doc.status === "ready";
                    return (
                      <tr
                        key={doc.id}
                        onClick={() => {
                          if (openable) {
                            onOpenDocument(doc);
                          }
                        }}
                        className={cn(
                          "group",
                          isSelected && "is-selected",
                          openable && "cursor-pointer",
                        )}
                      >
                        <td
                          className="pl-3 pr-1"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <input
                            type="checkbox"
                            checked={isSelected}
                            onChange={() => toggleSelectOne(doc.id)}
                            aria-label={`Select ${doc.original_name}`}
                            className="h-3.5 w-3.5 cursor-pointer rounded-sm border border-[var(--gd-faint)] accent-[var(--gd-accent)]"
                          />
                        </td>
                        <td>{nameCell(doc)}</td>
                        <td>
                          <ProcessingPipelineStatus
                            status={doc.status}
                            processingStage={doc.processing_stage}
                            processingProgress={doc.processing_progress}
                            errorMessage={doc.error_message}
                          />
                        </td>
                        <td className="cell-num hidden lg:table-cell">
                          {formatBytes(doc.file_size)}
                        </td>
                        <td className="cell-num hidden lg:table-cell">
                          {doc.page_count > 0 ? doc.page_count : "—"}
                        </td>
                        <td className="cell-num hidden lg:table-cell">
                          {doc.chunk_count > 0 ? doc.chunk_count : "—"}
                        </td>
                        <td className="hidden text-xs text-[var(--gd-muted)] xl:table-cell">
                          {formatDate(doc.uploaded_at)}
                        </td>
                        <td
                          className="pr-3 text-right"
                          onClick={(e) => e.stopPropagation()}
                        >
                          {rowActions(doc)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>

              {/* Phone list — the table would be unreadable at 390px */}
              <ul className="divide-y divide-[var(--gd-rule)] sm:hidden">
                {visibleDocuments.map((doc) => {
                  const isSelected = selectedIds.has(doc.id);
                  const openable = !isTrash && doc.status === "ready";
                  return (
                    <li key={doc.id} className="group p-3">
                      <div className="flex items-start gap-2">
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => toggleSelectOne(doc.id)}
                          aria-label={`Select ${doc.original_name}`}
                          className="mt-1 h-3.5 w-3.5 shrink-0 cursor-pointer rounded-sm border border-[var(--gd-faint)] accent-[var(--gd-accent)]"
                        />
                        <button
                          type="button"
                          disabled={!openable}
                          onClick={() => openable && onOpenDocument(doc)}
                          className="min-w-0 flex-1 text-left disabled:cursor-default"
                        >
                          <p className="truncate text-[13px] font-medium text-[var(--gd-text)]">
                            {doc.original_name}
                          </p>
                          <p className="mt-0.5 truncate text-[11px] text-[var(--gd-muted)]">
                            {formatBytes(doc.file_size)}
                            {doc.page_count > 0
                              ? ` · ${doc.page_count} page${doc.page_count === 1 ? "" : "s"}`
                              : ""}
                            {" · "}
                            {formatDate(doc.uploaded_at)}
                          </p>
                        </button>
                        <ProcessingPipelineStatus
                          status={doc.status}
                          processingStage={doc.processing_stage}
                          processingProgress={doc.processing_progress}
                          errorMessage={doc.error_message}
                          className="mt-0.5 shrink-0"
                        />
                      </div>
                      <div className="mt-1.5 flex items-center justify-end pl-6">
                        {rowActions(doc)}
                      </div>
                    </li>
                  );
                })}
              </ul>
            </>
          )}
        </div>

        <p className="mt-3 text-[11px] leading-relaxed text-[var(--gd-muted)]">
          Click a ready document to open it in your conversation, or use the
          chat icon to start a new one with it attached.
        </p>
      </div>
    </div>
  );
}
