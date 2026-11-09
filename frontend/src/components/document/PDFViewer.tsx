"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  ChevronLeft,
  ChevronRight,
  ZoomIn,
  ZoomOut,
  Loader2,
  AlertCircle,
  RotateCw,
} from "lucide-react";
import { API_BASE } from "@/lib/api";
import {
  findPageForQuote,
  findQuoteRectsInTextLayer,
} from "@/lib/citation-highlight";
import { usePdfSearch } from "@/hooks/usePdfSearch";
import { PDFDocumentProxy } from "pdfjs-dist/types/src/display/api";

// Import styles for react-pdf layers
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

// Serve the worker from /public — Turbopack's hashed /_next/static/media/*.mjs
// path breaks pdf.js dynamic import ("Setting up fake worker failed").
pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";

interface SearchRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface PdfHighlightRect {
  left: number;
  top: number;
  width: number;
  height: number;
  unit?: "percent" | "pixels" | "pdf";
}

export interface PdfHighlightTarget {
  /** 1-based page; 0 means locate via quote text. */
  page: number;
  rects?: PdfHighlightRect[];
  /** Citation excerpt used when bbox/rects are missing. */
  quote?: string;
}

interface Props {
  documentId: string;
  currentPage: number;
  onPageChange: (page: number) => void;
  totalPages: number;
  highlightTarget?: PdfHighlightTarget | null;
}

const isNormalizedRect = (rect: PdfHighlightRect) =>
  rect.left >= 0 &&
  rect.left <= 1 &&
  rect.top >= 0 &&
  rect.top <= 1 &&
  rect.width >= 0 &&
  rect.width <= 1 &&
  rect.height >= 0 &&
  rect.height <= 1;

export default function PDFViewer({
  documentId,
  currentPage,
  onPageChange,
  totalPages,
  highlightTarget,
}: Props) {
  const [scale, setScale] = useState(1.0);
  const [rotation, setRotation] = useState(0);
  const [pageInput, setPageInput] = useState(String(currentPage));
  const [pageInputError, setPageInputError] = useState(false);
  const [prevCurrentPage, setPrevCurrentPage] = useState(currentPage);
  const [quoteRects, setQuoteRects] = useState<PdfHighlightRect[]>([]);
  const [resolvedQuotePage, setResolvedQuotePage] = useState<number | null>(
    null,
  );
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const pageAnchorRef = useRef<HTMLDivElement>(null);
  const quoteLocateGenRef = useRef(0);
  const onPageChangeRef = useRef(onPageChange);
  onPageChangeRef.current = onPageChange;
  // Auth: prefer HttpOnly cookie session (withCredentials). Legacy Bearer JWT
  // from localStorage is still honored during migration.
  const tokenRef = useRef<string | null>(null);
  const [authToken, setAuthToken] = useState<string | null>(() => {
    if (typeof window === "undefined") return null;
    const legacy = localStorage.getItem("token");
    if (legacy && legacy !== "cookie") return legacy;
    return null;
  });
  if (typeof window !== "undefined" && tokenRef.current === null) {
    tokenRef.current = authToken;
  }

  useEffect(() => {
    const syncToken = (token: string | null) => {
      // Ignore cookie-session sentinel — pdf.js uses cookies via withCredentials.
      const next = token && token !== "cookie" ? token : null;
      tokenRef.current = next;
      setAuthToken(next);
    };

    const onRefreshed = (e: Event) => {
      const detail = (e as CustomEvent).detail as { accessToken?: string } | undefined;
      if (detail?.accessToken !== undefined) syncToken(detail.accessToken ?? null);
    };
    const onLoggedOut = () => syncToken(null);
    const onStorage = (e: StorageEvent) => {
      if (e.key === "token") syncToken(e.newValue);
    };

    window.addEventListener("auth:tokens-refreshed", onRefreshed);
    window.addEventListener("auth:logged-out", onLoggedOut);
    window.addEventListener("storage", onStorage);
    return () => {
      window.removeEventListener("auth:tokens-refreshed", onRefreshed);
      window.removeEventListener("auth:logged-out", onLoggedOut);
      window.removeEventListener("storage", onStorage);
    };
  }, []);

  // --- NEW: Search State ---
  const [pdfDoc, setPdfDoc] = useState<PDFDocumentProxy | null>(null);
  const {
    searchTerm,
    setSearchTerm,
    matches,
    currentIndex,
    setCurrentIndex,
    performSearch,
  } = usePdfSearch();

  // Sync page input state with current page prop updates during render phase
  if (currentPage !== prevCurrentPage) {
    setPrevCurrentPage(currentPage);
    setPageInput(String(currentPage));
    setPageInputError(false);
  }

  const pdfUrl = `${API_BASE}/api/v1/documents/${documentId}/pdf`;

  // Cookie sessions: withCredentials sends HttpOnly access_token to the API.
  // Legacy JWT sessions: also send Authorization Bearer.
  const fileConfig = useMemo(
    () => ({
      url: pdfUrl,
      withCredentials: true,
      httpHeaders:
        authToken && authToken !== "cookie"
          ? { Authorization: `Bearer ${authToken}` }
          : undefined,
    }),
    [pdfUrl, authToken],
  );

  const handleDocumentLoad = useCallback((pdf: PDFDocumentProxy) => {
    setPdfDoc(pdf);
  }, []);

  // When the citation has no page (or the page misses), locate the excerpt in the PDF.
  useEffect(() => {
    if (!highlightTarget?.quote || !pdfDoc) return;
    if ((highlightTarget.rects?.length ?? 0) > 0) return;

    const preferred =
      highlightTarget.page >= 1 ? highlightTarget.page : undefined;
    const gen = ++quoteLocateGenRef.current;
    let cancelled = false;

    void (async () => {
      const found = await findPageForQuote(
        pdfDoc,
        highlightTarget.quote!,
        preferred,
      );
      if (cancelled || gen !== quoteLocateGenRef.current) return;
      if (!found) {
        setResolvedQuotePage(null);
        return;
      }
      setResolvedQuotePage((prev) => (prev === found ? prev : found));
      if (found !== currentPage) {
        onPageChangeRef.current(found);
      }
    })();

    return () => {
      cancelled = true;
    };
    // currentPage intentionally omitted: re-running on every page flip would
    // restart full-document scans while the user navigates.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    highlightTarget?.quote,
    highlightTarget?.page,
    highlightTarget?.rects,
    pdfDoc,
  ]);

  // Derive highlight overlays from the rendered text layer so matches that
  // span line breaks still light up accurately.
  useEffect(() => {
    setQuoteRects([]);
    if (!highlightTarget?.quote) return;
    if ((highlightTarget.rects?.length ?? 0) > 0) return;

    const targetPage =
      highlightTarget.page >= 1
        ? highlightTarget.page
        : (resolvedQuotePage ?? currentPage);
    if (targetPage !== currentPage) return;

    const container = pageAnchorRef.current;
    if (!container) return;

    let cancelled = false;
    let observer: MutationObserver | null = null;

    const apply = () => {
      if (cancelled) return false;
      const pageEl =
        (container.querySelector(".react-pdf__Page") as HTMLElement | null) ??
        container;
      const textLayer = pageEl.querySelector(
        ".react-pdf__Page__textContent, .textLayer",
      );
      if (!textLayer) return false;
      const found = findQuoteRectsInTextLayer(
        textLayer,
        pageEl,
        highlightTarget.quote!,
      );
      if (found.length === 0) return false;
      setQuoteRects(found);

      const scrollContainer = scrollContainerRef.current;
      if (scrollContainer && found.length > 0) {
        const pageHeight = container.offsetHeight || 1;
        const pageWidth = container.offsetWidth || 1;
        const first = found[0];
        const top = Math.max(
          0,
          container.offsetTop + first.top * pageHeight - 48,
        );
        // Keep the full multi-line passage in view horizontally when the page
        // is wider than the inspector pane.
        const highlightLeft = Math.min(...found.map((r) => r.left)) * pageWidth;
        const highlightRight =
          Math.max(...found.map((r) => r.left + r.width)) * pageWidth;
        const visibleLeft = scrollContainer.scrollLeft;
        const visibleRight = visibleLeft + scrollContainer.clientWidth;
        let nextLeft = visibleLeft;
        if (highlightLeft < visibleLeft + 16) {
          nextLeft = Math.max(0, highlightLeft - 24);
        } else if (highlightRight > visibleRight - 16) {
          nextLeft = Math.max(
            0,
            highlightRight - scrollContainer.clientWidth + 24,
          );
        }
        scrollContainer.scrollTo({
          top,
          left: nextLeft,
          behavior: "smooth",
        });
      }
      return true;
    };

    const raf = window.requestAnimationFrame(() => {
      if (apply()) return;
      observer = new MutationObserver(() => {
        if (apply() && observer) {
          observer.disconnect();
          observer = null;
        }
      });
      observer.observe(container, { childList: true, subtree: true });
      window.setTimeout(() => {
        observer?.disconnect();
        observer = null;
      }, 4000);
    });

    return () => {
      cancelled = true;
      window.cancelAnimationFrame(raf);
      observer?.disconnect();
    };
  }, [
    highlightTarget?.quote,
    highlightTarget?.page,
    highlightTarget?.rects,
    resolvedQuotePage,
    currentPage,
    scale,
    rotation,
    documentId,
  ]);

  // Scroll the page into the viewer pane — never scrollIntoView the whole panel
  // (that fights the layout and can look like continuous blinking).
  useEffect(() => {
    const targetPage =
      highlightTarget?.page && highlightTarget.page >= 1
        ? highlightTarget.page
        : resolvedQuotePage;
    if (!targetPage || targetPage !== currentPage) return;
    if (quoteRects.length > 0) return;
    const container = scrollContainerRef.current;
    const anchor = pageAnchorRef.current;
    if (!container || !anchor) return;
    const top = Math.max(0, anchor.offsetTop - 12);
    container.scrollTo({ top, behavior: "smooth" });
  }, [currentPage, highlightTarget?.page, resolvedQuotePage, quoteRects.length]);

  // Reset captured doc when the file identity changes
  useEffect(() => {
    setPdfDoc(null);
    setQuoteRects([]);
    setResolvedQuotePage(null);
  }, [documentId]);

  // Clear quote-derived state when the citation target changes identity.
  useEffect(() => {
    setQuoteRects([]);
    setResolvedQuotePage(null);
  }, [highlightTarget?.page, highlightTarget?.quote, highlightTarget?.rects]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const isModifier = e.ctrlKey || e.metaKey;
      if (isModifier) {
        if (e.key === "=" || e.key === "+") {
          e.preventDefault();
          setScale((s) => Math.min(2.0, s + 0.1));
        } else if (e.key === "-") {
          e.preventDefault();
          setScale((s) => Math.max(0.5, s - 0.1));
        } else if (e.key === "r" || e.key === "R") {
          e.preventDefault();
          setRotation((r) => (r + 90) % 360);
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  const overlayRects = useMemo(() => {
    if (!highlightTarget) return [];

    const targetPage =
      highlightTarget.page >= 1
        ? highlightTarget.page
        : (resolvedQuotePage ?? currentPage);
    if (targetPage !== currentPage) return [];

    const sourceRects =
      (highlightTarget.rects?.length ?? 0) > 0
        ? highlightTarget.rects!
        : quoteRects;

    return sourceRects.map((rect) => {
      // Percent-based rects scale naturally with the rendered page size
      if (rect.unit === "percent" || isNormalizedRect(rect)) {
        return {
          left: `${rect.left * 100}%`,
          top: `${rect.top * 100}%`,
          width: `${rect.width * 100}%`,
          height: `${rect.height * 100}%`,
        };
      }
      if (rect.unit === "pdf") {
        return {
          left: `${rect.left * scale}px`,
          top: `${rect.top * scale}px`,
          width: `${rect.width * scale}px`,
          height: `${rect.height * scale}px`,
        };
      }
      return {
        left: `${rect.left}px`,
        top: `${rect.top}px`,
        width: `${rect.width}px`,
        height: `${rect.height}px`,
      };
    });
  }, [highlightTarget, currentPage, scale, quoteRects, resolvedQuotePage]);

  // --- NEW: Search Highlights Logic ---
  const searchHighlights = useMemo(() => {
    if (!matches[currentIndex] || matches[currentIndex].page !== currentPage)
      return [];
    return matches[currentIndex].rects.map((r: SearchRect) => ({
      left: `${r.left}px`,
      top: `${r.top}px`,
      width: `${r.width}px`,
      height: `${r.height}px`,
    }));
  }, [matches, currentIndex, currentPage]);

  const handlePageJump = (value: string) => {
    const pageNumber = parseInt(value.trim(), 10);
    if (
      !Number.isNaN(pageNumber) &&
      pageNumber >= 1 &&
      pageNumber <= totalPages
    ) {
      setPageInputError(false);
      onPageChange(pageNumber);
    } else {
      setPageInputError(true);
      setTimeout(() => {
        setPageInput(String(currentPage));
        setPageInputError(false);
      }, 1000);
    }
  };

  return (
    <div className="flex h-full min-h-0 min-w-0 w-full max-w-full flex-col overflow-hidden bg-background">
      <div className="flex min-w-0 shrink-0 flex-wrap items-center justify-between gap-x-2 gap-y-1 border-b border-border/50 bg-card/50 px-2 py-2">
        <div className="flex min-w-0 flex-1 flex-wrap items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 shrink-0"
            onClick={() => {
              const newPage = Math.max(1, currentPage - 1);
              onPageChange(newPage);
              setPageInput(String(newPage));
            }}
            disabled={currentPage <= 1}
            aria-label="Go to previous PDF page"
          >
            <ChevronLeft className="w-4 h-4" />
          </Button>

          {/* --- NEW: Search Integration --- */}
          <div className="flex min-w-0 items-center gap-1 border-l pl-2 sm:gap-2">
            <Input
              placeholder="Search..."
              className="h-7 w-20 min-w-0 max-w-[8rem] flex-1 text-xs sm:w-28"
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                performSearch(pdfDoc, e.target.value);
              }}
            />
            {matches.length > 0 && (
              <div className="flex shrink-0 items-center gap-1">
                <span className="text-[10px] text-muted-foreground">
                  {currentIndex + 1}/{matches.length}
                </span>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7"
                  onClick={() => {
                    const prevIdx = Math.max(0, currentIndex - 1);
                    setCurrentIndex(prevIdx);
                    onPageChange(matches[prevIdx].page);
                  }}
                >
                  <ChevronLeft className="w-3 h-3" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7"
                  onClick={() => {
                    const nextIdx = Math.min(
                      matches.length - 1,
                      currentIndex + 1,
                    );
                    setCurrentIndex(nextIdx);
                    onPageChange(matches[nextIdx].page);
                  }}
                >
                  <ChevronRight className="w-3 h-3" />
                </Button>
              </div>
            )}
          </div>

          {/* Page jump input — Page X of Y */}
          <form
            onSubmit={(event) => {
              event.preventDefault();
              handlePageJump(pageInput);
            }}
            className="flex shrink-0 items-center gap-1 text-xs"
            aria-label="PDF page navigation"
          >
            <span className="text-muted-foreground text-[10px] hidden xl:inline">
              Page
            </span>
            <Input
              value={pageInput}
              onChange={(e) => {
                setPageInput(e.target.value);
                setPageInputError(false);
              }}
              onBlur={() => handlePageJump(pageInput)}
              className={`h-7 text-center text-xs p-0 bg-background/50 transition-colors ${
                String(currentPage).length >= 3 ? "w-14" : "w-10"
              } ${pageInputError ? "border-destructive text-destructive" : ""}`}
              aria-label={`Page number input, current page ${currentPage} of ${totalPages}`}
              aria-invalid={pageInputError}
              title={`Enter page number between 1 and ${totalPages}`}
              inputMode="numeric"
            />
            <span className="text-muted-foreground text-[10px]">
              of {totalPages}
            </span>
          </form>

          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 shrink-0"
            onClick={() => {
              const newPage = Math.min(totalPages, currentPage + 1);
              onPageChange(newPage);
              setPageInput(String(newPage));
            }}
            disabled={currentPage >= totalPages}
            aria-label="Go to next PDF page"
          >
            <ChevronRight className="w-4 h-4" />
          </Button>
        </div>

        <div className="flex shrink-0 items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => setScale((current) => Math.max(0.5, current - 0.1))}
            aria-label="Zoom out PDF"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </Button>
          <span className="text-[10px] text-muted-foreground min-w-[36px] text-center">
            {Math.round(scale * 100)}%
          </span>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => setScale((current) => Math.min(2.0, current + 0.1))}
            aria-label="Zoom in PDF"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => setRotation((current) => (current + 90) % 360)}
            aria-label="Rotate PDF"
            title="Rotate 90°"
          >
            <RotateCw className="w-3.5 h-3.5" />
          </Button>
          <Button
            variant="ghost"
            className="h-7 px-2 text-[10px] font-medium"
            onClick={() => {
              setScale(1.0);
              setRotation(0);
            }}
            aria-label="Reset PDF zoom and rotation"
            title="Reset to 100% and 0°"
          >
            Reset
          </Button>
        </div>
      </div>

      {/* ── PDF Render ──────────────────────────────── */}
      <div
        ref={scrollContainerRef}
        className="genova-pdf-scroll relative flex min-h-0 min-w-0 w-full max-w-full flex-1 items-start overflow-auto bg-muted/30 p-4"
      >
        <Document
          key={documentId}
          file={fileConfig}
          onLoadSuccess={handleDocumentLoad}
          onLoadError={(err) => {
            console.error("PDF load error:", err);
          }}
          loading={
            <div className="flex items-center justify-center p-12">
              <Loader2 className="h-6 w-6 animate-spin text-primary" />
            </div>
          }
          error={
            <div className="mx-auto my-12 flex max-w-md flex-col items-center justify-center gap-3 rounded-lg border border-destructive/20 bg-card p-8 text-center shadow-sm">
              <AlertCircle className="h-8 w-8 text-destructive" />
              <div>
                <p className="mb-1 text-sm font-semibold text-foreground">
                  Failed to load PDF
                </p>
                <p className="text-xs leading-relaxed text-muted-foreground">
                  We encountered an error loading this PDF document. Please
                  verify the document is ready or try refreshing the page.
                </p>
              </div>
            </div>
          }
          noData={
            <div className="mx-auto my-12 flex max-w-md flex-col items-center justify-center gap-2 rounded-lg border border-border bg-card p-8 text-center shadow-sm">
              <p className="text-sm font-semibold text-foreground">
                No PDF document selected
              </p>
              <p className="text-xs text-muted-foreground">
                Select or upload a document to view it here.
              </p>
            </div>
          }
          className="border border-border bg-card shadow-md"
        >
          {/* w-max so the anchor matches the rendered page width — overlays are
              percent-positioned against this box, not the narrower inspector pane. */}
          <div ref={pageAnchorRef} className="relative w-max max-w-none">
            <Page
              pageNumber={currentPage}
              scale={scale}
              rotate={rotation}
              renderAnnotationLayer={false}
              renderTextLayer={true}
              loading={
                <div className="flex items-center justify-center p-8">
                  <Loader2 className="h-6 w-6 animate-spin text-primary" />
                </div>
              }
            />
            {/* Overlay Container (RAG + Search) */}
            <div className="pointer-events-none absolute inset-0 z-10">
              {overlayRects.map((style, index) => (
                <div
                  key={`rag-${index}`}
                  className="absolute rounded-sm border border-yellow-300/50 bg-yellow-400/40"
                  style={style}
                />
              ))}
              {searchHighlights.map((style, index) => (
                <div
                  key={`search-${index}`}
                  className="absolute rounded-sm border border-blue-300/50 bg-blue-400/40"
                  style={style}
                />
              ))}
            </div>
          </div>
        </Document>
      </div>
    </div>
  );
}
