"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  AlertCircle,
  ArrowLeft,
  List,
  ListTree,
  MessageSquare,
  RefreshCw,
} from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import {
  clearHierarchyCache,
  collectAllTables,
  countTablesInTree,
  fetchDocumentHierarchy,
  fetchDocumentIndexStatus,
  fetchRelatedSections,
  fetchSectionChunks,
  fetchSectionTables,
  findSectionById,
  findSectionPath,
  flattenSections,
} from "@/lib/document-hierarchy";
import { queueDocChat } from "@/lib/chat-attachments";
import type {
  ContentsViewMode,
  DocumentHierarchyResponse,
  DocumentIndexStatusResponse,
  ExplorerSelection,
  ExplorerTab,
  HierarchyChunk,
  RelatedSectionsResponse,
  SectionTableMeta,
  SectionTreeNode,
} from "@/types/document-hierarchy";
import { ContentsList } from "./ContentsList";
import { HierarchyTreeMap } from "./HierarchyTreeMap";
import { HowToRead } from "./HowToRead";
import { SectionDetailPanel } from "./SectionDetailPanel";
import { TablesPanel } from "./TablesPanel";

interface DocumentExplorerProps {
  documentId: string;
  fallbackName?: string;
  pageCount?: number;
}

function collectExpandableIds(sections: SectionTreeNode[]): string[] {
  const ids: string[] = [];
  const walk = (nodes: SectionTreeNode[]) => {
    for (const node of nodes) {
      if (node.children?.length) {
        ids.push(node.section_id);
        walk(node.children);
      }
    }
  };
  walk(sections);
  return ids;
}

export function DocumentExplorer({
  documentId,
  fallbackName,
  pageCount,
}: DocumentExplorerProps) {
  const [hierarchy, setHierarchy] = useState<DocumentHierarchyResponse | null>(
    null,
  );
  const [indexStatus, setIndexStatus] =
    useState<DocumentIndexStatusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selection, setSelection] = useState<ExplorerSelection>({
    kind: "document",
  });
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [tab, setTab] = useState<ExplorerTab>("contents");
  const [viewMode, setViewMode] = useState<ContentsViewMode>("list");

  const [chunks, setChunks] = useState<HierarchyChunk[]>([]);
  const [sectionTables, setSectionTables] = useState<SectionTableMeta[]>([]);
  const [related, setRelated] = useState<RelatedSectionsResponse | null>(null);
  const [loadingChunks, setLoadingChunks] = useState(false);
  const [loadingRelated, setLoadingRelated] = useState(false);
  const [chunksError, setChunksError] = useState<string | null>(null);
  const [relatedError, setRelatedError] = useState<string | null>(null);

  const loadRoot = useCallback(
    async (opts?: { bustCache?: boolean }) => {
      setLoading(true);
      setError(null);
      if (opts?.bustCache) clearHierarchyCache(documentId);
      try {
        const [hier, status] = await Promise.all([
          fetchDocumentHierarchy(documentId),
          fetchDocumentIndexStatus(documentId).catch(() => null),
        ]);
        setHierarchy(hier);
        setIndexStatus(status);
        if ((hier.section_count ?? 0) === 0) {
          setError(
            "This document has no hierarchical index yet. Reindex with semantic chunking to generate sections.",
          );
        }
      } catch (err) {
        setHierarchy(null);
        setIndexStatus(null);
        setError(
          err instanceof Error
            ? err.message
            : "Could not load document hierarchy.",
        );
      } finally {
        setLoading(false);
      }
    },
    [documentId],
  );

  useEffect(() => {
    void loadRoot();
  }, [loadRoot]);

  // Seed Tree map with top-level parents expanded for a useful first view.
  useEffect(() => {
    if (!hierarchy?.sections?.length) return;
    setExpandedIds((prev) => {
      if (prev.size > 0) return prev;
      return new Set(
        hierarchy.sections
          .filter((section) => (section.children?.length ?? 0) > 0)
          .map((section) => section.section_id),
      );
    });
  }, [hierarchy]);

  const flatSections = useMemo(
    () => (hierarchy ? flattenSections(hierarchy.sections) : []),
    [hierarchy],
  );

  const selectedSectionId =
    selection.kind === "section" || selection.kind === "chunk"
      ? selection.sectionId
      : null;

  const selectedSection = useMemo(() => {
    if (!hierarchy || !selectedSectionId) return null;
    return findSectionById(hierarchy.sections, selectedSectionId);
  }, [hierarchy, selectedSectionId]);

  const breadcrumb = useMemo(() => {
    if (!hierarchy || !selectedSectionId) return [];
    return findSectionPath(hierarchy.sections, selectedSectionId);
  }, [hierarchy, selectedSectionId]);

  const partNumber = useMemo(() => {
    if (!selectedSectionId) return 0;
    const idx = flatSections.findIndex((s) => s.section_id === selectedSectionId);
    return idx >= 0 ? idx + 1 : 0;
  }, [flatSections, selectedSectionId]);

  useEffect(() => {
    if (!selectedSectionId) {
      setChunks([]);
      setSectionTables([]);
      setRelated(null);
      setChunksError(null);
      setRelatedError(null);
      return;
    }

    let cancelled = false;
    const run = async () => {
      setLoadingChunks(true);
      setLoadingRelated(true);
      setChunksError(null);
      setRelatedError(null);
      try {
        const [chunkResp, tableResp, relatedResp] = await Promise.all([
          fetchSectionChunks(documentId, selectedSectionId),
          fetchSectionTables(documentId, selectedSectionId).catch(() => null),
          fetchRelatedSections(documentId, selectedSectionId),
        ]);
        if (cancelled) return;
        setChunks(chunkResp.chunks ?? []);
        setSectionTables(tableResp?.tables ?? selectedSection?.tables ?? []);
        setRelated(relatedResp);
      } catch (err) {
        if (cancelled) return;
        setChunks([]);
        setRelated(null);
        const message =
          err instanceof Error ? err.message : "Failed to load section details.";
        setChunksError(message);
        setRelatedError(message);
      } finally {
        if (!cancelled) {
          setLoadingChunks(false);
          setLoadingRelated(false);
        }
      }
    };
    void run();
    return () => {
      cancelled = true;
    };
  }, [documentId, selectedSectionId, selectedSection?.tables]);

  const allTables = useMemo(
    () => (hierarchy ? collectAllTables(hierarchy.sections) : []),
    [hierarchy],
  );
  const tableCount = useMemo(
    () => (hierarchy ? countTablesInTree(hierarchy.sections) : 0),
    [hierarchy],
  );

  const report = hierarchy?.report ?? indexStatus?.report ?? null;
  const resolvedPageCount =
    report?.page_count ??
    pageCount ??
    (typeof report?.pages_covered === "number" ? report.pages_covered : null);
  const documentName = hierarchy?.filename || fallbackName || "Document";
  const showTablesTab = tableCount > 0;

  useEffect(() => {
    if (tab === "tables" && !showTablesTab) setTab("contents");
  }, [tab, showTablesTab]);

  const onToggleExpand = (sectionId: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(sectionId)) next.delete(sectionId);
      else next.add(sectionId);
      return next;
    });
  };

  const selectSection = (sectionId: string) => {
    setSelection({ kind: "section", sectionId });
    setTab("contents");
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (hierarchy) {
        for (const node of findSectionPath(hierarchy.sections, sectionId)) {
          next.add(node.section_id);
        }
      }
      return next;
    });
  };

  const selectChunk = (chunkId: string) => {
    if (!selectedSectionId) return;
    if (!chunkId) {
      setSelection({ kind: "section", sectionId: selectedSectionId });
      return;
    }
    setSelection({ kind: "chunk", sectionId: selectedSectionId, chunkId });
  };

  const openInChat = () => {
    queueDocChat({ id: documentId, filename: documentName });
    window.location.href = `/dashboard?docId=${encodeURIComponent(documentId)}`;
  };

  if (loading) {
    return (
      <div className="flex h-full min-h-[60vh] flex-col gap-4 p-6">
        <Skeleton className="h-12 w-full max-w-xl" />
        <Skeleton className="h-[min(70vh,640px)] w-full" />
      </div>
    );
  }

  if (error && !hierarchy) {
    return (
      <div className="flex min-h-[50vh] flex-col items-center justify-center gap-3 px-6 text-center">
        <AlertCircle className="h-7 w-7 text-[var(--gd-negative)]" />
        <p className="max-w-md text-sm text-[var(--gd-text)]">{error}</p>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={() => void loadRoot({ bustCache: true })}
          >
            <RefreshCw className="mr-1.5 h-3.5 w-3.5" />
            Retry
          </Button>
          <Link
            href="/documents"
            className="inline-flex h-9 items-center rounded-md px-3 text-sm text-[var(--gd-muted)] hover:text-[var(--gd-text)]"
          >
            Back to documents
          </Link>
        </div>
      </div>
    );
  }

  if (!hierarchy) return null;

  const showingDetail = Boolean(selectedSection) && tab === "contents";
  const isTreeMap = tab === "contents" && viewMode === "tree";
  const showInlineDetail = showingDetail && !isTreeMap;
  const treeDetailOpen = isTreeMap && Boolean(selectedSection);

  const detailPanel = selectedSection ? (
    <SectionDetailPanel
      section={selectedSection}
      partNumber={partNumber || 1}
      partTotal={flatSections.length || hierarchy.section_count}
      breadcrumb={breadcrumb}
      chunks={chunks}
      tables={sectionTables}
      related={related}
      selectedChunkId={
        selection.kind === "chunk" ? selection.chunkId : null
      }
      loadingChunks={loadingChunks}
      loadingRelated={loadingRelated}
      chunksError={chunksError}
      relatedError={relatedError}
      onSelectChunk={selectChunk}
      onSelectSection={selectSection}
      onBackToContents={() => setSelection({ kind: "document" })}
      onBackToParent={
        selectedSection.parent_section_id
          ? () => selectSection(selectedSection.parent_section_id!)
          : undefined
      }
    />
  ) : null;

  return (
    <div className="flex h-full min-h-0 flex-col bg-[var(--gd-ground)]">
      <header className="shrink-0 border-b border-[var(--gd-border)] bg-[var(--gd-surface)] px-4 py-3 md:px-6">
        <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0 space-y-1">
            <div className="flex items-center gap-2 text-xs text-[var(--gd-muted)]">
              <Link
                href="/documents"
                className="inline-flex items-center gap-1 hover:text-[var(--gd-text)]"
              >
                <ArrowLeft className="h-3.5 w-3.5" />
                Documents
              </Link>
              <span>/</span>
              <span>Explorer</span>
            </div>
            <h1 className="truncate text-lg font-semibold tracking-tight text-[var(--gd-text)]">
              {documentName}
            </h1>
            <p className="text-xs text-[var(--gd-muted)]">
              {[
                `${hierarchy.section_count} parts`,
                `${hierarchy.chunk_count} chunks`,
                resolvedPageCount != null ? `${resolvedPageCount} pages` : null,
                hierarchy.chunk_strategy || null,
              ]
                .filter(Boolean)
                .join(" · ")}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              className="h-8 gap-1.5"
              onClick={() => {
                void loadRoot({ bustCache: true });
                toast.message("Refreshing…");
              }}
            >
              <RefreshCw className="h-3.5 w-3.5" />
              Refresh
            </Button>
            <Button size="sm" className="h-8 gap-1.5" onClick={openInChat}>
              <MessageSquare className="h-3.5 w-3.5" />
              Open chat
            </Button>
          </div>
        </div>

        {error ? (
          <p className="mb-2 text-xs text-[var(--gd-warning)]">{error}</p>
        ) : null}

        <div className="flex flex-wrap items-center justify-between gap-3">
          <Tabs
            value={tab}
            onValueChange={(value) => setTab(value as ExplorerTab)}
          >
            <TabsList variant="line" className="h-9 justify-start gap-1">
              <TabsTrigger value="contents">
                Contents ({hierarchy.section_count})
              </TabsTrigger>
              {showTablesTab ? (
                <TabsTrigger value="tables">Tables ({tableCount})</TabsTrigger>
              ) : null}
            </TabsList>
          </Tabs>

          {tab === "contents" ? (
            <div className="inline-flex rounded-lg border border-[var(--gd-border)] bg-[var(--gd-raised)]/50 p-0.5">
              <button
                type="button"
                onClick={() => setViewMode("list")}
                className={cn(
                  "inline-flex h-7 items-center gap-1.5 rounded-md px-2.5 text-xs cursor-pointer",
                  viewMode === "list"
                    ? "bg-[var(--gd-surface)] font-medium text-[var(--gd-text)] shadow-sm"
                    : "text-[var(--gd-muted)] hover:text-[var(--gd-text)]",
                )}
              >
                <List className="h-3.5 w-3.5" />
                List
              </button>
              <button
                type="button"
                onClick={() => setViewMode("tree")}
                className={cn(
                  "inline-flex h-7 items-center gap-1.5 rounded-md px-2.5 text-xs cursor-pointer",
                  viewMode === "tree"
                    ? "bg-[var(--gd-surface)] font-medium text-[var(--gd-text)] shadow-sm"
                    : "text-[var(--gd-muted)] hover:text-[var(--gd-text)]",
                )}
              >
                <ListTree className="h-3.5 w-3.5" />
                Tree
              </button>
            </div>
          ) : null}
        </div>
      </header>

      {isTreeMap ? (
        <div className="relative min-h-0 flex-1 bg-[var(--gd-surface)]">
          <HierarchyTreeMap
            documentName={documentName}
            partCount={hierarchy.section_count}
            sections={hierarchy.sections}
            flatSections={flatSections}
            expandedIds={expandedIds}
            selectedSectionId={selectedSectionId}
            onToggleExpand={onToggleExpand}
            onSelectSection={selectSection}
            onExpandAll={() =>
              setExpandedIds(new Set(collectExpandableIds(hierarchy.sections)))
            }
            onCollapseAll={() => setExpandedIds(new Set())}
          />
          {!selectedSection ? (
            <div className="pointer-events-none absolute bottom-4 left-4 z-10 max-w-sm rounded-xl border border-[var(--gd-border)] bg-[var(--gd-surface)]/95 px-3 py-2 text-[11px] text-[var(--gd-muted)] shadow-sm backdrop-blur-sm">
              Click a part to read summary, source text, tables, and related
              sections.
            </div>
          ) : null}
        </div>
      ) : (
        <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[minmax(300px,440px)_minmax(0,1fr)]">
          <aside className="min-h-0 border-b border-[var(--gd-border)] bg-[var(--gd-surface)] lg:border-b-0 lg:border-r">
            {tab === "tables" ? (
              <div className="h-full overflow-y-auto p-4">
                <TablesPanel
                  tables={allTables}
                  emptyLabel="No tables were indexed for this document."
                  onSelectSection={selectSection}
                />
              </div>
            ) : (
              <div className="h-[min(48vh,420px)] min-h-0 overflow-y-auto lg:h-full">
                <ContentsList
                  sections={flatSections}
                  selectedSectionId={selectedSectionId}
                  onSelectSection={selectSection}
                />
              </div>
            )}
          </aside>

          <main className="min-h-0 overflow-y-auto px-4 py-5 md:px-8 md:py-6">
            {tab === "tables" && !selectedSection ? (
              <div className="mx-auto max-w-xl">
                <HowToRead />
              </div>
            ) : null}

            {tab === "contents" && !showInlineDetail ? (
              <div className="mx-auto max-w-xl">
                <HowToRead />
              </div>
            ) : null}

            {showInlineDetail ? detailPanel : null}
          </main>
        </div>
      )}

      <Dialog
        open={treeDetailOpen}
        onOpenChange={(open) => {
          if (!open) setSelection({ kind: "document" });
        }}
      >
        <DialogContent
          showCloseButton
          className="flex h-[min(92vh,880px)] w-[calc(100vw-1rem)] max-w-4xl flex-col gap-0 overflow-hidden p-0 sm:max-w-4xl"
        >
          <DialogHeader className="shrink-0 border-b border-[var(--gd-border)] px-5 py-3 pr-12 text-left">
            <DialogTitle className="truncate text-base">
              {selectedSection?.title || "Part details"}
            </DialogTitle>
            <DialogDescription className="text-xs text-[var(--gd-muted)]">
              {[
                partNumber
                  ? `Part ${partNumber} of ${flatSections.length || hierarchy.section_count}`
                  : null,
                selectedSection
                  ? [
                      selectedSection.page_start != null
                        ? `p. ${selectedSection.page_start}`
                        : null,
                      selectedSection.page_end != null &&
                      selectedSection.page_end !== selectedSection.page_start
                        ? `–${selectedSection.page_end}`
                        : null,
                    ]
                      .filter(Boolean)
                      .join("") || null
                  : null,
              ]
                .filter(Boolean)
                .join(" · ")}
            </DialogDescription>
          </DialogHeader>
          <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
            {detailPanel}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
