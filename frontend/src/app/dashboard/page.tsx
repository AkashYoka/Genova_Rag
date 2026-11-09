"use client";

import { useEffect, useState, useCallback, useRef, useSyncExternalStore } from "react";
import dynamic from "next/dynamic";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { toast } from "sonner";
import {
  api,
  CONNECTION_ERROR_BANNER_MESSAGE,
  CONNECTION_ERROR_MESSAGE,
} from "@/lib/api";
import Header from "@/components/layout/Header";
import { RequireAuth } from "@/components/auth/RequireAuth";
import ChatSessionSidebar from "@/components/chat/ChatSessionSidebar";
import ChatPanel from "@/components/chat/ChatPanel";
import CompareView from "@/components/document/CompareView";
import DashboardDropOverlay from "@/components/document/DashboardDropOverlay";
import { useDashboardDrop } from "@/hooks/useDashboardDrop";
import { Button } from "@/components/ui/button";
import {
  Columns2,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  StickyNote,
} from "lucide-react";
import { useNotesStore } from "@/store/notes-store";
import "./dashboard.css";

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

// Lazy-load the graph panel — it pulls in @xyflow/react which is sizeable
const KnowledgeGraph = dynamic(
  () => import("@/components/graph/KnowledgeGraph"),
  { ssr: false },
);

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogClose,
} from "@/components/ui/dialog";

import {
  Sheet,
  SheetContent,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";

import type { DocInfo } from "@/types/document";
import { normalizeDocInfo, type DocumentListApiResponse } from "@/types/document";
import { useChatStore } from "@/store/chat-store";
import {
  chatTitleFromFilename,
  takePendingDocChat,
} from "@/lib/chat-attachments";
export type { DocInfo };

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

const SESSIONS_OPEN_KEY = "genova:chat-sessions-open";

/**
 * Chat-session column visibility lives outside React so the preference can be
 * read during the very first client render (via useSyncExternalStore) without
 * a setState-in-effect and without a hydration mismatch.
 */
let sessionsOpenFallback = true;
const sessionsOpenListeners = new Set<() => void>();

function getDefaultSessionsOpen(): boolean {
  return true;
}

function subscribeSessionsOpen(onStoreChange: () => void): () => void {
  sessionsOpenListeners.add(onStoreChange);
  const onStorage = (event: StorageEvent) => {
    if (event.key === SESSIONS_OPEN_KEY) onStoreChange();
  };
  window.addEventListener("storage", onStorage);
  return () => {
    sessionsOpenListeners.delete(onStoreChange);
    window.removeEventListener("storage", onStorage);
  };
}

function readSessionsOpen(): boolean {
  try {
    return window.localStorage.getItem(SESSIONS_OPEN_KEY) !== "false";
  } catch {
    return sessionsOpenFallback;
  }
}

function writeSessionsOpen(open: boolean): void {
  sessionsOpenFallback = open;
  try {
    window.localStorage.setItem(SESSIONS_OPEN_KEY, String(open));
  } catch {
    // Storage unavailable — the in-memory fallback keeps the toggle working.
  }
  sessionsOpenListeners.forEach((listener) => listener());
}

function DashboardPageContent() {
  const { user, initialized } = useAuth();
  const router = useRouter();

  const [documents, setDocuments] = useState<DocInfo[]>([]);
  const documentsRef = useRef<DocInfo[]>([]);
  const prevDocsRef = useRef<Record<string, string>>({});
  const [activeDoc, setActiveDoc] = useState<DocInfo | null>(null);
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
  const [graphOpen, setGraphOpen] = useState(false);
  const sessionsOpen = useSyncExternalStore(
    subscribeSessionsOpen,
    readSessionsOpen,
    getDefaultSessionsOpen,
  );
  const [connectionError, setConnectionError] = useState("");
  const [compareOpen, setCompareOpen] = useState(false);
  const [inspectorTab, setInspectorTab] = useState<"notes" | "pdf">("notes");
  const notesOpen = useNotesStore((s) => s.notesOpen);
  const setNotesOpen = useNotesStore((s) => s.setNotesOpen);
  const associateDocument = useNotesStore((s) => s.associateDocument);
  const notesFocusToken = useNotesStore((s) => s.focusToken);

  const toggleSessions = useCallback(() => {
    writeSessionsOpen(!readSessionsOpen());
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

  const loadDocuments = useCallback(async () => {
    try {
      const data = await api.get<DocumentListApiResponse>(
        "/api/v1/documents/",
      );
      const rawList = data?.items ?? data?.documents ?? [];
      const next = rawList.map(normalizeDocInfo);
      documentsRef.current = next;
      setDocuments(next);
      setActiveDoc((current) => {
        if (!current) {
          if (typeof window !== "undefined") {
            const params = new URLSearchParams(window.location.search);
            const targetId = params.get("docId");
            if (targetId) {
              const matched = next.find((d) => d.id === targetId);
              if (matched) return matched;
            }
          }
          return current;
        }
        const updated = next.find((d) => d.id === current.id);
        if (!updated) return current;
        if (
          updated.status === current.status &&
          updated.page_count === current.page_count &&
          updated.chunk_count === current.chunk_count &&
          updated.original_name === current.original_name &&
          updated.file_size === current.file_size
        ) {
          return current;
        }
        return updated;
      });
      setConnectionError("");
    } catch (err) {
      const message =
        err instanceof Error ? err.message : CONNECTION_ERROR_MESSAGE;
      if (
        (message.includes("Invalid or expired token") ||
          message.includes("Unauthorized") ||
          message.includes("User not found"))
      ) {
        router.replace("/login");
        return;
      }
      setConnectionError(
        message === CONNECTION_ERROR_MESSAGE
          ? CONNECTION_ERROR_BANNER_MESSAGE
          : message,
      );
    }
  }, [router]);

  useEffect(() => {
    if (!user) return;
    void (async () => {
      await loadDocuments();
    })();
  }, [user, loadDocuments]);

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

        setActiveDoc(matched);
        const filename = matched.original_name;
        await useChatStore.getState().createSession(
          chatTitleFromFilename(filename),
          { attachments: [{ id: matched.id, filename }] },
        );

        if (typeof window !== "undefined") {
          const url = new URL(window.location.href);
          if (url.searchParams.get("docId") !== matched.id) {
            url.searchParams.set("docId", matched.id);
            router.replace(`${url.pathname}?${url.searchParams.toString()}`);
          }
        }

        toast.success(`Chat started with ${filename}`);
      } catch {
        toast.error("Could not start chat session");
      }
    })();
  }, [user, router]);

  const handlePageDrop = useCallback(
    async (files: File[]) => {
      for (const file of files) {
        const formData = new FormData();
        formData.append("file", file);
        try {
          await api.postForm("/api/v1/documents/upload", formData);
        } catch (err) {
          const message = err instanceof Error ? err.message : "Upload failed";
          console.error(`Upload failed for ${file.name}:`, message);
        }
      }
      void loadDocuments();
    },
    [loadDocuments],
  );

  const { isDraggingOver, dropZoneProps } = useDashboardDrop({
    onDrop: handlePageDrop,
    disabled: !user,
  });

  useEffect(() => {
    const prev = prevDocsRef.current;
    const nextPrevDocs: Record<string, string> = {};
    (documents || []).forEach((doc) => {
      nextPrevDocs[doc.id] = doc.status;
      const oldStatus = prev[doc.id];
      if (oldStatus && oldStatus !== doc.status) {
        if (doc.status === "ready") {
          toast.success(
            `Ingestion complete: '${doc.original_name}' is ready.`,
          );
        } else if (doc.status === "failed") {
          toast.error(
            `Ingestion failed for '${doc.original_name}': ${doc.error_message || "Unknown error"}`,
          );
        }
      }
    });
    prevDocsRef.current = nextPrevDocs;
  }, [documents]);

  useEffect(() => {
    const hasPending = (documents || []).some(
      (d) => d.status === "pending" || d.status === "processing",
    );
    if (!hasPending) return;
    const interval = setInterval(() => {
      void loadDocuments();
    }, 3000);
    return () => clearInterval(interval);
  }, [documents, loadDocuments]);

  useEffect(() => {
    if (notesFocusToken > 0) {
      setCompareOpen(false);
      setInspectorTab("notes");
    }
  }, [notesFocusToken]);

  // Close the graph only when the selected document actually changes. Comparing
  // normalized ids keeps the null (no document) case stable; otherwise
  // `undefined !== null` fired every render and immediately closed the dialog.
  const prevDocIdRef = useRef<string | null>(null);
  useEffect(() => {
    const activeDocId = activeDoc?.id ?? null;
    if (activeDocId !== prevDocIdRef.current) {
      prevDocIdRef.current = activeDocId;
      setGraphOpen(false);
    }
    associateDocument(activeDoc?.id ?? null, activeDoc?.original_name ?? null);
  }, [activeDoc?.id, activeDoc?.original_name, associateDocument]);

  const showPdfDock =
    !compareOpen &&
    viewerOpen &&
    !!activeDoc &&
    activeDoc.original_name.endsWith(".pdf");

  const showInspector = !compareOpen && (notesOpen || showPdfDock);
  const activeInspectorTab: "notes" | "pdf" =
    showPdfDock && (!notesOpen || inspectorTab === "pdf") ? "pdf" : "notes";

  const readyCount = documents.filter((d) => d.status === "ready").length;
  const processingCount = documents.filter(
    (d) => d.status === "pending" || d.status === "processing",
  ).length;

  const titleFinding = activeDoc
    ? activeDoc.status === "ready"
      ? "Ask about this file. Answers cite pages."
      : activeDoc.status === "failed"
        ? `Ingestion failed${activeDoc.error_message ? `: ${activeDoc.error_message}` : "."}`
        : "Still indexing."
    : documents.length === 0
      ? "Upload a PDF to get started."
      : `${readyCount} ready${processingCount ? `, ${processingCount} indexing` : ""}. Pick one to begin.`;

  return (
    <div className="genova-desk flex h-screen min-w-[375px] flex-col overflow-hidden">
      <DashboardDropOverlay
        isDraggingOver={isDraggingOver}
        dropZoneProps={dropZoneProps}
      />

      <Header
        graphOpen={graphOpen}
        onToggleGraph={() => setGraphOpen(!graphOpen)}
      />

      {connectionError ? (
        <div className="genova-desk-alert" role="alert">
          {connectionError}
        </div>
      ) : null}

      <div className="genova-desk-stage">
        {/* Region 2 — chat sessions (list) */}
        <aside
          id="chat-sessions-panel"
          className={`genova-desk-sessions${sessionsOpen ? "" : " is-collapsed"}`}
          aria-label="Chat sessions"
          aria-hidden={!sessionsOpen}
          {...(!sessionsOpen ? { inert: true } : {})}
        >
          <div className="dashboard-sessions-shell h-full">
            <ChatSessionSidebar />
          </div>
        </aside>

        {/* Region 3 — open conversation */}
        <section className="genova-desk-chat" aria-label="Assistant">
          <header className="genova-desk-title">
            <div className="genova-desk-title-heading">
              <Button
                variant="ghost"
                size="icon"
                className="hidden h-7 w-7 shrink-0 md:inline-flex"
                onClick={toggleSessions}
                aria-label={
                  sessionsOpen ? "Hide chat sessions" : "Show chat sessions"
                }
                aria-expanded={sessionsOpen}
                aria-controls="chat-sessions-panel"
                title={
                  sessionsOpen ? "Hide chat sessions" : "Show chat sessions"
                }
              >
                {sessionsOpen ? (
                  <PanelLeftClose className="h-3.5 w-3.5" />
                ) : (
                  <PanelLeftOpen className="h-3.5 w-3.5" />
                )}
              </Button>
              <div className="min-w-0 flex-1">
                <h1 title={activeDoc?.original_name ?? "No document open"}>
                  {activeDoc?.original_name ?? "No document open"}
                </h1>
                <p className="sub" title={
                  activeDoc
                    ? `${activeDoc.status}, ${formatBytes(activeDoc.file_size)}, ${activeDoc.page_count || "–"} pages, ${activeDoc.chunk_count || 0} chunks. ${titleFinding}`
                    : titleFinding
                }>
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
                      <span className="num">{activeDoc.page_count || "–"}</span>{" "}
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

          <div className="genova-desk-chat-body">
            <ChatPanel
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
                  // Only switch the desk document when the citation points at a
                  // different file — avoid remounting chat state for same-doc views.
                  if (matchedDoc.id !== activeDoc?.id) {
                    setActiveDoc(matchedDoc);
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

        {/* Region 3b — details (Notes / PDF / compare) */}
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
              className={`genova-desk-inspector is-open${notesOpen ? " is-notes" : ""}`}
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
                  />
                ) : null}
              </div>
            </aside>
          )
        )}
      </div>

      <Dialog open={graphOpen} onOpenChange={setGraphOpen}>
        <DialogContent
          className="flex h-[88vh] w-[calc(100vw-1rem)] max-w-none flex-col gap-0 overflow-hidden p-0 outline-none sm:h-[90vh] sm:max-w-[min(1600px,calc(100vw-3rem))]"
          showCloseButton={false}
        >
          <DialogHeader className="sr-only">
            <DialogTitle>Knowledge graph</DialogTitle>
          </DialogHeader>
          {graphOpen ? (
            <div className="flex min-h-0 flex-1 flex-col">
              <KnowledgeGraph onClose={() => setGraphOpen(false)} />
            </div>
          ) : null}
        </DialogContent>
      </Dialog>
    </div>
  );
}

export default function DashboardPage() {
  return (
    <RequireAuth>
      <DashboardPageContent />
    </RequireAuth>
  );
}
