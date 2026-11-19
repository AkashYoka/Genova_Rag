"use client";

import {
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import dynamic from "next/dynamic";
import { useRouter, useSearchParams } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { toast } from "sonner";
import { api } from "@/lib/api";
import Header from "@/components/layout/Header";
import { RequireAuth } from "@/components/auth/RequireAuth";
import ChatPanel from "@/components/chat/ChatPanel";
import CompareView from "@/components/document/CompareView";
import DashboardDropOverlay from "@/components/document/DashboardDropOverlay";
import DocumentsView from "@/components/workspace/DocumentsView";
import UnifiedSidebar from "@/components/workspace/UnifiedSidebar";
import { useDashboardDrop } from "@/hooks/useDashboardDrop";
import { useDocumentLibrary } from "@/hooks/useDocumentLibrary";
import { Button } from "@/components/ui/button";
import {
  Columns2,
  FolderOpen,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  StickyNote,
} from "lucide-react";
import { useNotesStore } from "@/store/notes-store";
import type { DocInfo } from "@/types/document";
import { normalizeDocInfo, type DocumentListApiResponse } from "@/types/document";
import { useChatStore } from "@/store/chat-store";
import {
  chatTitleFromFilename,
  takePendingDocChat,
} from "@/lib/chat-attachments";
import "./dashboard.css";

export type { DocInfo };

const NotesPanel = dynamic(() => import("@/components/notes/NotesPanel"), {
  ssr: false,
  loading: () => (
    <div className="flex flex-1 items-center justify-center text-xs text-muted-foreground">
      Loading notes…
    </div>
  ),
});

function PDFViewerSkeleton() {
  return (
    <div
      className="flex h-full flex-col bg-[var(--gd-surface,#fff)]"
      aria-busy="true"
      aria-label="Loading PDF viewer"
    >
      <div className="flex shrink-0 items-center justify-between border-b border-[var(--gd-border,#e2e5ea)] px-3 py-2">
        {/* taste-skill-ignore: loading skeleton shimmer */}
        <div className="h-7 w-24 rounded bg-[var(--gd-raised,#eceef2)]" />
        <div className="h-7 w-16 rounded bg-[var(--gd-raised,#eceef2)]" />
      </div>
      <div className="flex-1 p-4">
        <div className="h-full rounded border border-[var(--gd-border,#e2e5ea)] bg-[var(--gd-raised,#eceef2)]" />
      </div>
    </div>
  );
}

const PDFViewer = dynamic(() => import("@/components/document/PDFViewer"), {
  ssr: false,
  loading: () => <PDFViewerSkeleton />,
});

function formatBytes(n: number): string {
  if (!n || n < 0) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function statusClass(status: string): string {
  if (status === "ready") return "status-ready";
  if (status === "failed" || status === "error") return "status-failed";
  if (status === "processing" || status === "pending") return "status-processing";
  return "";
}

const SIDEBAR_OPEN_KEY = "genova:chat-sessions-open";

/**
 * Sidebar visibility lives outside React so the preference can be read during
 * the very first client render (via useSyncExternalStore) without a
 * setState-in-effect and without a hydration mismatch.
 */
let sidebarOpenFallback = true;
const sidebarOpenListeners = new Set<() => void>();

function getDefaultSidebarOpen(): boolean {
  return true;
}

function subscribeSidebarOpen(onStoreChange: () => void): () => void {
  sidebarOpenListeners.add(onStoreChange);
  const onStorage = (event: StorageEvent) => {
    if (event.key === SIDEBAR_OPEN_KEY) onStoreChange();
  };
  window.addEventListener("storage", onStorage);
  return () => {
    sidebarOpenListeners.delete(onStoreChange);
    window.removeEventListener("storage", onStorage);
  };
}

function readSidebarOpen(): boolean {
  try {
    return window.localStorage.getItem(SIDEBAR_OPEN_KEY) !== "false";
  } catch {
    return sidebarOpenFallback;
  }
}

function writeSidebarOpen(open: boolean): void {
  sidebarOpenFallback = open;
  try {
    window.localStorage.setItem(SIDEBAR_OPEN_KEY, String(open));
  } catch {
    // Storage unavailable — the in-memory fallback keeps the toggle working.
  }
  sidebarOpenListeners.forEach((listener) => listener());
}

type DeskView = "workspace" | "documents";

function viewFromSearchParams(
  params: URLSearchParams | { get: (key: string) => string | null },
): DeskView {
  return params.get("view") === "documents" ? "documents" : "workspace";
}

function DashboardPageContent() {
  const { user, initialized } = useAuth();
  const router = useRouter();
  const searchParams = useSearchParams();
  // Optimistic local view so Open chat / New conversation switch immediately.
  // A pending-nav ref ignores stale ?view=documents while the router catches up.
  const [view, setView] = useState<DeskView>(() =>
    viewFromSearchParams(searchParams),
  );
  const pendingViewRef = useRef<DeskView | null>(null);

  useEffect(() => {
    const next = viewFromSearchParams(searchParams);
    if (pendingViewRef.current) {
      if (next !== pendingViewRef.current) return;
      pendingViewRef.current = null;
    }
    setView(next);
  }, [searchParams]);

  const library = useDocumentLibrary();
  const activeSessionTitle = useChatStore(
    (state) =>
      state.sessions.find((session) => session.id === state.activeSessionId)
        ?.title ?? null,
  );
  const { documents, setTab: setDocTab, setStatusFilter } = library;

  const documentsRef = useRef<DocInfo[]>([]);
  useEffect(() => {
    documentsRef.current = library.documents;
  }, [library.documents]);

  const [activeDocId, setActiveDocId] = useState<string | null>(null);
  const [pdfPage, setPdfPage] = useState(1);
  const [pdfHighlightTarget, setPdfHighlightTarget] = useState<{
    page: number;
    rects?: {
      left: number;
      top: number;
      width: number;
      height: number;
      unit?: "percent" | "pixels" | "pdf";
    }[];
    quote?: string;
  } | null>(null);
  const [viewerOpen, setViewerOpen] = useState(true);
  const sidebarOpen = useSyncExternalStore(
    subscribeSidebarOpen,
    readSidebarOpen,
    getDefaultSidebarOpen,
  );
  const [compareOpen, setCompareOpen] = useState(false);
  const [inspectorTab, setInspectorTab] = useState<"notes" | "pdf">("notes");
  const notesOpen = useNotesStore((s) => s.notesOpen);
  const setNotesOpen = useNotesStore((s) => s.setNotesOpen);
  const associateDocument = useNotesStore((s) => s.associateDocument);

  const activeDoc = useMemo(
    () => documents.find((d) => d.id === activeDocId) ?? null,
    [documents, activeDocId],
  );

  const toggleSidebar = useCallback(() => {
    writeSidebarOpen(!readSidebarOpen());
  }, []);

  // Restore this user's sticky notes from S3 on login; drop them on logout so a
  // different account on the same browser never sees another user's notes.
  const loadNotes = useNotesStore((s) => s.loadNotes);
  useEffect(() => {
    if (user?.id) {
      void loadNotes(user.id);
    } else if (initialized) {
      useNotesStore.getState().reset();
    }
  }, [user?.id, initialized, loadNotes]);

  const openWorkspace = useCallback(() => {
    pendingViewRef.current = "workspace";
    setView("workspace");
    router.replace("/dashboard", { scroll: false });
  }, [router]);

  const openDocuments = useCallback(
    (opts?: { tab?: "active" | "trash"; filter?: "all" | "ready" | "indexing" }) => {
      if (opts?.tab) setDocTab(opts.tab);
      if (opts?.filter) setStatusFilter(opts.filter);
      pendingViewRef.current = "documents";
      setView("documents");
      router.replace("/dashboard?view=documents", { scroll: false });
    },
    [router, setDocTab, setStatusFilter],
  );

  // Deep links: /dashboard?docId=… selects that source without a second fetch.
  useEffect(() => {
    if (activeDocId) return;
    const target = searchParams.get("docId");
    if (!target) return;
    const match = documentsRef.current.find((d) => d.id === target);
    if (match) setActiveDocId(match.id);
  }, [activeDocId, searchParams]);

  // Documents → Chat: open a fresh session with the queued PDF already attached.
  // Pending nonce is consumed once (module Set), so Strict Mode remounts are safe.
  useEffect(() => {
    if (!user) return;
    const pending = takePendingDocChat();
    if (!pending) return;

    void (async () => {
      try {
        let matched =
          documentsRef.current.find((d) => d.id === pending.docId) ?? null;

        if (!matched) {
          try {
            const data = await api.get<DocumentListApiResponse>(
              "/api/v1/documents/",
            );
            const rawList = data?.items ?? data?.documents ?? [];
            matched =
              rawList
                .map(normalizeDocInfo)
                .find((d) => d.id === pending.docId) ?? null;
          } catch {
            matched = null;
          }
        }

        // Only attach IDs the authenticated user's document list confirms
        // (ready + owned). Backend still enforces ownership; this avoids
        // poisoning the session with unowned chips.
        if (!matched || matched.status !== "ready") {
          toast.error("Document is not available to chat with yet.");
          return;
        }

        setActiveDocId(matched.id);
        await useChatStore.getState().createSession(
          chatTitleFromFilename(matched.original_name),
          { attachments: [{ id: matched.id, filename: matched.original_name }] },
        );
        openWorkspace();
        toast.success(`Chat started with ${matched.original_name}`);
      } catch {
        toast.error("Could not start chat session");
      }
    })();
  }, [user, openWorkspace]);

  const { isDraggingOver, dropZoneProps } = useDashboardDrop({
    onDrop: (files) => {
      void library.upload(files);
    },
    disabled: !user,
  });

  // The notes store can ask for focus from anywhere in the app. Listening to
  // the store keeps that request out of the render path.
  useEffect(() => {
    const unsubscribe = useNotesStore.subscribe((state, previous) => {
      if (state.focusToken !== previous.focusToken && state.focusToken > 0) {
        setCompareOpen(false);
        setInspectorTab("notes");
      }
    });
    return unsubscribe;
  }, []);

  useEffect(() => {
    associateDocument(activeDoc?.id ?? null, activeDoc?.original_name ?? null);
  }, [activeDoc?.id, activeDoc?.original_name, associateDocument]);

  const openDocumentInWorkspace = useCallback(
    (doc: DocInfo) => {
      setActiveDocId(doc.id);
      const isPdf = doc.original_name.toLowerCase().endsWith(".pdf");
      if (isPdf) {
        setViewerOpen(true);
        setInspectorTab("pdf");
      }
      setCompareOpen(false);
      openWorkspace();
    },
    [openWorkspace],
  );

  const openDocumentInChat = useCallback(
    async (doc: DocInfo) => {
      if (doc.status !== "ready") {
        toast.error("Wait until this document finishes indexing.");
        return;
      }
      setActiveDocId(doc.id);
      try {
        await useChatStore.getState().createSession(
          chatTitleFromFilename(doc.original_name),
          { attachments: [{ id: doc.id, filename: doc.original_name }] },
        );
        setCompareOpen(false);
        openWorkspace();
        toast.success(`Chat started with ${doc.original_name}`);
      } catch {
        toast.error("Could not start chat session");
      }
    },
    [openWorkspace],
  );

  const showPdfDock =
    !compareOpen &&
    viewerOpen &&
    !!activeDoc &&
    activeDoc.original_name.toLowerCase().endsWith(".pdf");

  const showInspector = !compareOpen && (notesOpen || showPdfDock);
  const activeInspectorTab: "notes" | "pdf" =
    showPdfDock && (!notesOpen || inspectorTab === "pdf") ? "pdf" : "notes";

  const titleFinding = activeDoc
    ? activeDoc.status === "ready"
      ? "Ask about this file. Answers cite pages."
      : activeDoc.status === "failed"
        ? `Ingestion failed${activeDoc.error_message ? `: ${activeDoc.error_message}` : "."}`
        : "Still indexing."
    : documents.length === 0
      ? "Upload a document to begin — Genova prepares it before answering."
      : `${library.counts.ready} ready${library.counts.indexing ? `, ${library.counts.indexing} indexing` : ""}. Open a document for page-level citations, or just ask across all of them.`;

  const sidebar = (
    <UnifiedSidebar
      view={view}
      docTab={library.tab}
      docFilter={library.statusFilter}
      counts={library.counts}
      onOpenWorkspace={openWorkspace}
      onOpenDocuments={openDocuments}
    />
  );

  return (
    <div className="genova-desk flex h-dvh min-w-[375px] flex-col overflow-hidden">
      <DashboardDropOverlay
        isDraggingOver={isDraggingOver}
        dropZoneProps={dropZoneProps}
      />
      <input {...library.getInputProps()} />

      <Header
        sidebarOpen={sidebarOpen}
        onToggleSidebar={toggleSidebar}
        mobileSheetContent={
          <UnifiedSidebar
            view={view}
            docTab={library.tab}
            docFilter={library.statusFilter}
            counts={library.counts}
            onOpenWorkspace={openWorkspace}
            onOpenDocuments={openDocuments}
            mobile
          />
        }
      />

      {library.connectionError ? (
        <div className="genova-desk-alert" role="alert">
          {library.connectionError}
        </div>
      ) : null}

      <div className="genova-desk-stage">
        <aside
          id="app-sidebar"
          className={`genova-desk-sidebar${sidebarOpen ? "" : " is-collapsed"}`}
          aria-label="Workspace navigation"
          aria-hidden={!sidebarOpen}
          {...(!sidebarOpen ? { inert: true } : {})}
        >
          {sidebar}
        </aside>

        <div className="genova-desk-main">
          <div
            className={`genova-desk-workspace${view === "workspace" ? "" : " is-hidden"}`}
            aria-hidden={view !== "workspace"}
            {...(view !== "workspace" ? { inert: true } : {})}
          >
            <section className="genova-desk-chat" aria-label="Conversation">
              <header className="genova-desk-title">
                <div className="genova-desk-title-heading">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="hidden h-7 w-7 shrink-0 md:inline-flex"
                    onClick={toggleSidebar}
                    aria-label={
                      sidebarOpen ? "Hide sidebar" : "Show sidebar"
                    }
                    aria-expanded={sidebarOpen}
                    aria-controls="app-sidebar"
                    title={sidebarOpen ? "Hide sidebar" : "Show sidebar"}
                  >
                    {sidebarOpen ? (
                      <PanelLeftClose className="h-3.5 w-3.5" />
                    ) : (
                      <PanelLeftOpen className="h-3.5 w-3.5" />
                    )}
                  </Button>
                  <div className="min-w-0 flex-1">
                    <h1
                      title={
                        activeDoc?.original_name ??
                        activeSessionTitle ??
                        "New conversation"
                      }
                    >
                      {activeDoc?.original_name ??
                        activeSessionTitle ??
                        "New conversation"}
                    </h1>
                    <p
                      className="sub"
                      title={
                        activeDoc
                          ? `${activeDoc.status}, ${formatBytes(activeDoc.file_size)}, ${activeDoc.page_count || "–"} pages, ${activeDoc.chunk_count || 0} chunks. ${titleFinding}`
                          : titleFinding
                      }
                    >
                      {activeDoc ? (
                        <>
                          <span className={statusClass(activeDoc.status)}>
                            {activeDoc.status}
                          </span>
                          <span aria-hidden="true"> · </span>
                          <span className="num">
                            {formatBytes(activeDoc.file_size)}
                          </span>
                          <span aria-hidden="true"> · </span>
                          <span className="num">
                            {activeDoc.page_count || "–"}
                          </span>{" "}
                          pages
                          <span aria-hidden="true"> · </span>
                          <span className="num data-hue">
                            {activeDoc.chunk_count || 0}
                          </span>{" "}
                          chunks
                          <span aria-hidden="true"> · </span>
                          {titleFinding}
                        </>
                      ) : (
                        titleFinding
                      )}
                    </p>
                  </div>
                </div>
                <div className="actions">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 gap-1.5 px-2 text-xs text-[var(--gd-muted)] hover:bg-[var(--gd-raised)] hover:text-[var(--gd-text)]"
                    onClick={() => openDocuments()}
                    aria-label="Browse documents"
                    title="Browse and manage documents"
                  >
                    <FolderOpen className="h-3.5 w-3.5" />
                    <span className="hidden lg:inline">Documents</span>
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7"
                    onClick={() => setViewerOpen(!viewerOpen)}
                    aria-label={viewerOpen ? "Close PDF viewer" : "Open PDF viewer"}
                    aria-pressed={viewerOpen}
                    title={viewerOpen ? "Close viewer" : "Open viewer"}
                  >
                    {viewerOpen ? (
                      <PanelRightClose className="h-3.5 w-3.5" />
                    ) : (
                      <PanelRightOpen className="h-3.5 w-3.5" />
                    )}
                  </Button>
                  <Button
                    variant={compareOpen ? "secondary" : "ghost"}
                    size="icon"
                    className="h-7 w-7"
                    onClick={() => setCompareOpen(!compareOpen)}
                    aria-label={
                      compareOpen ? "Close compare view" : "Open compare view"
                    }
                    aria-pressed={compareOpen}
                    title={
                      compareOpen
                        ? "Close compare view"
                        : "Compare two PDFs side-by-side"
                    }
                  >
                    <Columns2 className="h-3.5 w-3.5" />
                  </Button>
                  <Button
                    variant={notesOpen ? "secondary" : "ghost"}
                    size="icon"
                    className="h-7 w-7"
                    onClick={() => {
                      const next = !notesOpen;
                      setNotesOpen(next);
                      if (next) setInspectorTab("notes");
                    }}
                    aria-label={notesOpen ? "Close notes panel" : "Open notes panel"}
                    aria-pressed={notesOpen}
                    title={notesOpen ? "Close notes" : "Open notes"}
                  >
                    <StickyNote className="h-3.5 w-3.5" />
                  </Button>
                </div>
              </header>

              <div className="genova-desk-chat-body">                      <ChatPanel
                        activeDoc={activeDoc}
                        documents={documents}
                  onCitationClick={(target) => {
                    const filename = target.filename?.trim();
                    const matchedDoc = filename
                      ? documents.find(
                          (doc) =>
                            doc.original_name.toLowerCase() ===
                              filename.toLowerCase() ||
                            doc.original_name
                              .toLowerCase()
                              .endsWith(`/${filename.toLowerCase()}`),
                        )
                      : null;

                    if (filename && !matchedDoc) {
                      toast.error(`Could not open PDF “${filename}”.`);
                      return;
                    }

                    if (matchedDoc) {
                      if (matchedDoc.id !== activeDocId) {
                        setActiveDocId(matchedDoc.id);
                      }
                    } else if (!activeDoc) {
                      toast.error("Select a PDF document to view citations.");
                      return;
                    }

                    const viewerDoc = matchedDoc ?? activeDoc;
                    const maxPage = Math.max(1, viewerDoc?.page_count || 1);
                    const citedPage =
                      typeof target.page === "number" &&
                      Number.isFinite(target.page) &&
                      target.page >= 1
                        ? Math.min(target.page, maxPage)
                        : 0;

                    setPdfPage(citedPage >= 1 ? citedPage : 1);
                    setPdfHighlightTarget({
                      page: citedPage,
                      rects: target.highlightRects,
                      quote: target.excerpt?.trim() || undefined,
                    });
                    if (!viewerOpen) setViewerOpen(true);
                    setCompareOpen(false);
                    setInspectorTab("pdf");
                  }}
                />
              </div>
            </section>

            {compareOpen ? (
              <aside
                className="genova-desk-inspector is-open is-compare"
                aria-label="Compare documents"
              >
                <CompareView
                  documents={documents}
                  onClose={() => setCompareOpen(false)}
                />
              </aside>
            ) : (
              showInspector && (
                <aside
                  className={`genova-desk-inspector is-open${notesOpen ? " is-notes" : ""}${
                    activeInspectorTab === "pdf" && showPdfDock ? " is-pdf" : ""
                  }`}
                  aria-label="Notes and page inspector"
                >
                  {notesOpen && showPdfDock ? (
                    <div
                      className="genova-desk-inspector-tabs"
                      role="tablist"
                      aria-label="Inspector panels"
                    >
                      <button
                        type="button"
                        role="tab"
                        aria-selected={activeInspectorTab === "notes"}
                        className={
                          activeInspectorTab === "notes" ? "is-active" : undefined
                        }
                        onClick={() => setInspectorTab("notes")}
                      >
                        Notes
                      </button>
                      <button
                        type="button"
                        role="tab"
                        aria-selected={activeInspectorTab === "pdf"}
                        className={
                          activeInspectorTab === "pdf" ? "is-active" : undefined
                        }
                        onClick={() => setInspectorTab("pdf")}
                      >
                        PDF
                      </button>
                    </div>
                  ) : null}

                  <div className="flex min-h-0 flex-1 flex-col">
                    {activeInspectorTab === "notes" && notesOpen ? (
                      <NotesPanel />
                    ) : null}
                    {activeInspectorTab === "pdf" && showPdfDock && activeDoc ? (
                      <PDFViewer
                        documentId={activeDoc.id}
                        currentPage={pdfPage}
                        onPageChange={(page) => {
                          setPdfPage(page);
                          setPdfHighlightTarget((current) => {
                            if (!current) return null;
                            // Keep citation overlays while quote-search jumps pages,
                            // or while the user is still on the cited page.
                            if (
                              current.page === 0 ||
                              current.page === page ||
                              Boolean(current.quote)
                            ) {
                              return current;
                            }
                            return null;
                          });
                        }}
                        totalPages={activeDoc.page_count}
                        highlightTarget={pdfHighlightTarget}
                        onClose={() => setViewerOpen(false)}
                      />
                    ) : null}
                  </div>
                </aside>
              )
            )}
          </div>

          {view === "documents" ? (
            <DocumentsView
              library={library}
              onOpenDocument={openDocumentInWorkspace}
              onOpenInChat={openDocumentInChat}
              onBackToChat={openWorkspace}
            />
          ) : null}
        </div>
      </div>
    </div>
  );
}

export default function DashboardPage() {
  return (
    <RequireAuth>
      <Suspense
        fallback={
          <div className="genova-desk flex h-screen items-center justify-center">
            <div
              className="h-8 w-8 animate-pulse rounded-full bg-[var(--gd-raised)]"
              aria-label="Loading workspace"
            />
          </div>
        }
      >
        <DashboardPageContent />
      </Suspense>
    </RequireAuth>
  );
}
