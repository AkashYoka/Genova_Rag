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
  Search,
} from "lucide-react";
import { cn } from "@/lib/utils";
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
  /** User zoom relative to fit-to-width (1 = page fills the pane). */
  const [userZoom, setUserZoom] = useState(1.0);
  const [fitWidth, setFitWidth] = useState(0);
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
  const [nativePageWidth, setNativePageWidth] = useState(612);
  const {
    searchTerm,
    setSearchTerm,
    matches,
    currentIndex,
    setCurrentIndex,
    searching: pdfSearching,
    performSearch,
    clearSearch,
  } = usePdfSearch();
  const pdfDocRef = useRef<PDFDocumentProxy | null>(null);

  // Sync page input state with current page prop updates during render phase
  if (currentPage !== prevCurrentPage) {
    setPrevCurrentPage(currentPage);
    setPageInput(String(currentPage));
    setPageInputError(false);
  }

  // Fit the page to the pane width so the right edge is not clipped.
  useEffect(() => {
    const el = scrollContainerRef.current;
    if (!el) return;

    const measure = () => {
      const styles = window.getComputedStyle(el);
      const padX =
        (Number.parseFloat(styles.paddingLeft) || 0) +
        (Number.parseFloat(styles.paddingRight) || 0);
      setFitWidth(Math.max(0, el.clientWidth - padX));
    };

    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!pdfDoc) return;
    let cancelled = false;
    void pdfDoc
      .getPage(currentPage)
      .then((page) => {
        if (cancelled) return;
        const viewport = page.getViewport({ scale: 1, rotation });
        setNativePageWidth(viewport.width || 612);
      })
      .catch(() => {
        if (!cancelled) setNativePageWidth(612);
      });
    return () => {
      cancelled = true;
    };
  }, [pdfDoc, currentPage, rotation]);

  const pageWidth =
    fitWidth > 0 ? Math.max(160, fitWidth * userZoom) : undefined;
  /** CSS px per PDF unit — used for citation overlays with unit "pdf". */
  const scale =
    pageWidth && nativePageWidth > 0 ? pageWidth / nativePageWidth : userZoom;

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
    pdfDocRef.current = pdf;
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
    pdfDocRef.current = null;
    setPdfDoc(null);
    clearSearch();
    setQuoteRects([]);
    setResolvedQuotePage(null);
  }, [documentId, clearSearch]);

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
          setUserZoom((z) => Math.min(2.5, Number((z + 0.1).toFixed(2))));
        } else if (e.key === "-") {
          e.preventDefault();
          setUserZoom((z) => Math.max(0.5, Number((z - 0.1).toFixed(2))));
        } else if (e.key === "r" || e.key === "R") {
          e.preventDefault();
          setRotation((r) => (r + 90) % 360);
        } else if (e.key === "0") {
          e.preventDefault();
          setUserZoom(1);
          setRotation(0);
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

  const toolBtn =
    "h-7 w-7 shrink-0 rounded-md text-[var(--gd-muted)] hover:bg-[var(--gd-raised)] hover:text-[var(--gd-text)]";

  return (
    <div className="flex h-full min-h-0 min-w-0 w-full max-w-full flex-col overflow-hidden bg-[var(--gd-surface)]">
      <div className="genova-pdf-toolbar" role="toolbar" aria-label="PDF controls">
        <div className="genova-pdf-toolbar-group" aria-label="Page navigation">
          <Button
            variant="ghost"
            size="icon"
            className={toolBtn}
            onClick={() => {
              const newPage = Math.max(1, currentPage - 1);
              onPageChange(newPage);
              setPageInput(String(newPage));
            }}
            disabled={currentPage <= 1}
            aria-label="Go to previous PDF page"
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>

          <form
            onSubmit={(event) => {
              event.preventDefault();
              handlePageJump(pageInput);
            }}
            className="genova-pdf-page-form"
            aria-label="PDF page navigation"
          >
            <Input
              value={pageInput}
              onChange={(e) => {
                setPageInput(e.target.value);
                setPageInputError(false);
              }}
              onBlur={() => handlePageJump(pageInput)}
              className={cn(
                "genova-pdf-page-input",
                String(currentPage).length >= 3 ? "w-12" : "w-9",
                pageInputError && "is-error",
              )}
              aria-label={`Page number input, current page ${currentPage} of ${totalPages}`}
              aria-invalid={pageInputError}
              title={`Enter page number between 1 and ${totalPages}`}
              inputMode="numeric"
            />
            <span className="genova-pdf-page-meta">/ {totalPages}</span>
          </form>

          <Button
            variant="ghost"
            size="icon"
            className={toolBtn}
            onClick={() => {
              const newPage = Math.min(totalPages, currentPage + 1);
              onPageChange(newPage);
              setPageInput(String(newPage));
            }}
            disabled={currentPage >= totalPages}
            aria-label="Go to next PDF page"
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>

        <div className="genova-pdf-toolbar-divider" aria-hidden />

        <div className="genova-pdf-toolbar-group genova-pdf-search" aria-label="Find in PDF">
          <div className="genova-pdf-search-field">
            <Search className="genova-pdf-search-icon" aria-hidden />
            <Input
              placeholder={pdfDoc ? "Find in document" : "Loading PDF…"}
              className="genova-pdf-search-input"
              value={searchTerm}
              disabled={!pdfDoc}
              onChange={(e) => {
                const next = e.target.value;
                setSearchTerm(next);
                performSearch(pdfDocRef.current, next);
              }}
              aria-label="Search PDF text"
            />
          </div>
          {pdfSearching ? (
            <span className="genova-pdf-search-count" aria-live="polite">
              …
            </span>
          ) : null}
          {matches.length > 0 ? (
            <div className="genova-pdf-search-nav">
              <span className="genova-pdf-search-count">
                {currentIndex + 1}/{matches.length}
              </span>
              <Button
                variant="ghost"
                size="icon"
                className={toolBtn}
                onClick={() => {
                  const prevIdx = Math.max(0, currentIndex - 1);
                  setCurrentIndex(prevIdx);
                  onPageChange(matches[prevIdx].page);
                }}
                aria-label="Previous search match"
              >
                <ChevronLeft className="h-3.5 w-3.5" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className={toolBtn}
                onClick={() => {
                  const nextIdx = Math.min(
                    matches.length - 1,
                    currentIndex + 1,
                  );
                  setCurrentIndex(nextIdx);
                  onPageChange(matches[nextIdx].page);
                }}
                aria-label="Next search match"
              >
                <ChevronRight className="h-3.5 w-3.5" />
              </Button>
            </div>
          ) : null}
        </div>

        <div className="genova-pdf-toolbar-divider" aria-hidden />

        <div className="genova-pdf-toolbar-group" aria-label="Zoom and rotate">
          <Button
            variant="ghost"
            size="icon"
            className={toolBtn}
            onClick={() =>
              setUserZoom((z) => Math.max(0.5, Number((z - 0.1).toFixed(2))))
            }
            aria-label="Zoom out PDF"
          >
            <ZoomOut className="h-3.5 w-3.5" />
          </Button>
          <span className="genova-pdf-zoom-label">
            {Math.round(userZoom * 100)}%
          </span>
          <Button
            variant="ghost"
            size="icon"
            className={toolBtn}
            onClick={() =>
              setUserZoom((z) => Math.min(2.5, Number((z + 0.1).toFixed(2))))
            }
            aria-label="Zoom in PDF"
          >
            <ZoomIn className="h-3.5 w-3.5" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className={toolBtn}
            onClick={() => setRotation((current) => (current + 90) % 360)}
            aria-label="Rotate PDF"
            title="Rotate 90°"
          >
            <RotateCw className="h-3.5 w-3.5" />
          </Button>
          <Button
            variant="ghost"
            className="h-7 shrink-0 rounded-md px-2 text-[11px] font-medium text-[var(--gd-muted)] hover:bg-[var(--gd-raised)] hover:text-[var(--gd-text)]"
            onClick={() => {
              setUserZoom(1);
              setRotation(0);
            }}
            aria-label="Fit PDF to panel width"
            title="Fit to width"
          >
            Fit
          </Button>
        </div>
      </div>

      {/* ── PDF Render ──────────────────────────────── */}
      <div
        ref={scrollContainerRef}
        className="genova-pdf-scroll relative flex min-h-0 min-w-0 w-full max-w-full flex-1 items-start justify-center overflow-auto bg-[var(--gd-raised)] p-3"
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
          className="genova-pdf-document"
        >
          {/* Anchor matches rendered page width for percent-based citation overlays. */}
          <div
            ref={pageAnchorRef}
            className={cn(
              "relative",
              userZoom > 1 ? "w-max max-w-none" : "w-full max-w-full",
            )}
            style={pageWidth ? { width: pageWidth } : undefined}
          >
            <Page
              pageNumber={currentPage}
              width={pageWidth}
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
