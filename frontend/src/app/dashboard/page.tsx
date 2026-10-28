"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import dynamic from "next/dynamic";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { isDevAuthBypass } from "@/lib/dev-auth";
import { toast } from "sonner";
import {
  api,
  CONNECTION_ERROR_BANNER_MESSAGE,
  CONNECTION_ERROR_MESSAGE,
} from "@/lib/api";
import Header from "@/components/layout/Header";
import DocumentSidebar from "@/components/document/DocumentSidebar";
import ChatSessionSidebar from "@/components/chat/ChatSessionSidebar";
import ChatPanel from "@/components/chat/ChatPanel";
import CompareView from "@/components/document/CompareView";
import DashboardDropOverlay from "@/components/document/DashboardDropOverlay";
import { useDashboardDrop } from "@/hooks/useDashboardDrop";
import { Button } from "@/components/ui/button";
import { Columns2, PanelRightClose, PanelRightOpen } from "lucide-react";
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

export interface DocInfo {
  chunk_size?: number;
  chunk_overlap?: number;
  summary: string;
  id: string;
  original_name: string;
  file_size: number;
  page_count: number;
  chunk_count: number;
  status: string;
  error_message: string | null;
  uploaded_at: string;
  keywords?: string[];
}

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

export default function DashboardPage() {
  const { user, initialized } = useAuth();
  const router = useRouter();

  const [documents, setDocuments] = useState<DocInfo[]>([]);
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
  } | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [viewerOpen, setViewerOpen] = useState(true);
  const [graphOpen, setGraphOpen] = useState(false);
  const [connectionError, setConnectionError] = useState("");
  const [documentsLoading, setDocumentsLoading] = useState(true);
  const [compareOpen, setCompareOpen] = useState(false);
  const [inspectorTab, setInspectorTab] = useState<"notes" | "pdf">("notes");
  const notesOpen = useNotesStore((s) => s.notesOpen);
  const setNotesOpen = useNotesStore((s) => s.setNotesOpen);
  const notesFocusToken = useNotesStore((s) => s.focusToken);

  const handleDocumentRenamed = useCallback((renamedDocument: DocInfo) => {
    setDocuments((current) =>
      current.map((document) =>
        document.id === renamedDocument.id ? renamedDocument : document,
      ),
    );
    setActiveDoc((current) =>
      current?.id === renamedDocument.id ? renamedDocument : current,
    );
  }, []);

  useEffect(() => {
    if (initialized && !user && !isDevAuthBypass()) {
      router.replace("/login");
    }
  }, [user, initialized, router]);

  useEffect(() => {
    if (user) {
      const hasHfToken = !!(user.hf_token || localStorage.getItem("hf_token"));
      if (!hasHfToken) {
        console.info(
          "Hugging Face API token is not configured. Personal model access will fall back to the system default unless set in the user profile menu.",
        );
      }
    }
  }, [user]);

  const loadDocuments = useCallback(async (opts?: { silent?: boolean }) => {
    // Polling must stay silent — toggling loading remounts the list (blink).
    if (!opts?.silent) setDocumentsLoading(true);
    try {
      const data = await api.get<{ documents?: DocInfo[]; items?: DocInfo[] }>(
        "/api/v1/documents/",
      );
      const next = data?.documents ?? data?.items ?? [];
      setDocuments(next);
      setActiveDoc((current) => {
        if (!current) return current;
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
        !isDevAuthBypass() &&
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
    } finally {
      if (!opts?.silent) setDocumentsLoading(false);
    }
  }, [router]);

  useEffect(() => {
    if (!user) return;
    void (async () => {
      await loadDocuments();
    })();
  }, [user, loadDocuments]);

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
      void loadDocuments({ silent: true });
    }, 3000);
    return () => clearInterval(interval);
  }, [documents, loadDocuments]);

  useEffect(() => {
    if (notesFocusToken > 0) {
      setCompareOpen(false);
      setInspectorTab("notes");
    }
  }, [notesFocusToken]);

  const prevDocIdRef = useRef<string | null>(null);
  if (activeDoc?.id !== prevDocIdRef.current) {
    prevDocIdRef.current = activeDoc?.id ?? null;
    if (graphOpen) setGraphOpen(false);
  }

  // Keep KnowledgeGraph in the bundle path used by the product
  void KnowledgeGraph;

  if (!initialized) {
    return (
      <div
        className="flex min-h-screen items-center justify-center bg-background"
        suppressHydrationWarning
      >
        <div
          className="h-8 w-8 animate-pulse rounded-full bg-muted"
          aria-label="Loading"
          suppressHydrationWarning
        />
      </div>
    );
  }

  // Dev bypass can finish auth with no user if the API is down — don't spin forever.
  if (!user) {
    if (isDevAuthBypass()) {
      return (
        <div className="flex min-h-screen flex-col items-center justify-center gap-3 bg-background px-6 text-center text-foreground">
          <p className="text-sm text-muted-foreground">
            Could not sign in to the local API. Is the backend running on port
            7860?
          </p>
          <button
            type="button"
            className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground"
            onClick={() => window.location.reload()}
          >
            Retry
          </button>
        </div>
      );
    }
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <div
          className="h-8 w-8 animate-pulse rounded-full bg-muted"
          aria-label="Loading"
        />
      </div>
    );
  }

  const sidebarContent = (
    <DocumentSidebar
      documents={documents}
      activeDoc={activeDoc}
      loading={documentsLoading}
      onSelectDoc={(doc) => {
        setActiveDoc(doc);
        setPdfPage(1);
      }}
      onDocumentsChange={loadDocuments}
      onDocumentRenamed={handleDocumentRenamed}
    />
  );

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
      ? "Ask this file. Answers cite pages."
      : activeDoc.status === "failed"
        ? `Ingestion failed${activeDoc.error_message ? `: ${activeDoc.error_message}` : "."}`
        : `Ingesting (${activeDoc.status}).`
    : documents.length === 0
      ? "Upload a PDF to start asking."
      : `${readyCount} ready${processingCount ? `, ${processingCount} ingesting` : ""}. Select a document.`;

  return (
    <div className="genova-desk flex h-screen min-w-[375px] flex-col overflow-hidden">
      <DashboardDropOverlay
        isDraggingOver={isDraggingOver}
        dropZoneProps={dropZoneProps}
      />

      <Header
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
        viewerOpen={viewerOpen}
        onToggleViewer={() => setViewerOpen(!viewerOpen)}
        notesOpen={notesOpen}
        onToggleNotes={() => {
          const next = !notesOpen;
          setNotesOpen(next);
          if (next) setInspectorTab("notes");
        }}
        compareOpen={compareOpen}
        onToggleCompare={() => setCompareOpen(!compareOpen)}
        mobileSheetContent={sidebarContent}
      />

      {connectionError ? (
        <div className="genova-desk-alert" role="alert">
          {connectionError}
        </div>
      ) : null}

      <div className="genova-desk-stage">
        {/* Region 1 — document library (nav) */}
        <aside
          className={`genova-desk-docs${sidebarOpen ? " is-open" : ""}`}
          aria-label="Document library"
        >
          {sidebarContent}
        </aside>

        {/* Region 2 — chat sessions (list) */}
        <aside className="genova-desk-sessions" aria-label="Chat sessions">
          <div className="dashboard-sessions-shell h-full">
            <ChatSessionSidebar />
          </div>
        </aside>

        {/* Region 3 — open conversation */}
        <section className="genova-desk-chat" aria-label="Assistant">
          <div className="genova-desk-title">
            <div className="min-w-0">
              <h1>
                {activeDoc?.original_name ?? "Document desk"}
              </h1>
              <p className="sub">
                {activeDoc ? (
                  <>
                    <span className={statusClass(activeDoc.status)}>
                      {activeDoc.status}
                    </span>
                    <span className="text-[var(--gd-faint)]">, </span>
                    <span className="num">{formatBytes(activeDoc.file_size)}</span>
                    <span className="text-[var(--gd-faint)]">, </span>
                    <span className="num">{activeDoc.page_count || "–"}</span> pages
                    <span className="text-[var(--gd-faint)]">, </span>
                    <span className="num data-hue">
                      {activeDoc.chunk_count || 0}
                    </span>{" "}
                    chunks
                    <span className="text-[var(--gd-faint)]">. </span>
                    {titleFinding}
                  </>
                ) : (
                  titleFinding
                )}
              </p>
            </div>
            <div className="actions lg:hidden">
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={() => setViewerOpen(!viewerOpen)}
                aria-label={viewerOpen ? "Close PDF viewer" : "Open PDF viewer"}
              >
                {viewerOpen ? (
                  <PanelRightClose className="h-4 w-4" />
                ) : (
                  <PanelRightOpen className="h-4 w-4" />
                )}
              </Button>
              <Button
                variant={compareOpen ? "secondary" : "ghost"}
                size="icon"
                className="h-8 w-8"
                onClick={() => setCompareOpen(!compareOpen)}
                aria-label={
                  compareOpen ? "Close compare view" : "Open compare view"
                }
              >
                <Columns2 className="h-4 w-4" />
              </Button>
            </div>
          </div>

          <div className="genova-desk-chat-body">
            <ChatPanel
              activeDoc={activeDoc}
              documents={documents}
              onCitationClick={(target) => {
                setPdfPage(target.page);
                setPdfHighlightTarget({
                  page: target.page,
                  rects: target.highlightRects,
                });
                if (!viewerOpen) setViewerOpen(true);
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
                      if (pdfHighlightTarget?.page !== page) {
                        setPdfHighlightTarget(null);
                      }
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
    </div>
  );
}
