import { useCallback, useRef, useState } from "react";
import { PDFDocumentProxy } from "pdfjs-dist/types/src/display/api";

export interface SearchMatch {
  page: number;
  rects: { left: number; top: number; width: number; height: number }[];
}

interface TextItem {
  str: string;
  transform: number[];
}

function isUsablePdfDoc(
  pdfDoc: PDFDocumentProxy | null,
): pdfDoc is PDFDocumentProxy {
  if (!pdfDoc) return false;
  const proxy = pdfDoc as PDFDocumentProxy & {
    destroyed?: boolean;
    _transport?: unknown;
  };
  // Destroyed proxies keep a null transport; getPage then throws
  // "Cannot read properties of null (reading 'sendWithPromise')".
  if (proxy.destroyed) return false;
  if (proxy._transport == null) return false;
  return typeof pdfDoc.getPage === "function" && pdfDoc.numPages > 0;
}

export function usePdfSearch() {
  const [searchTerm, setSearchTerm] = useState("");
  const [matches, setMatches] = useState<SearchMatch[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [searching, setSearching] = useState(false);
  const searchGenRef = useRef(0);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearSearch = useCallback(() => {
    searchGenRef.current += 1;
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
      debounceRef.current = null;
    }
    setSearchTerm("");
    setMatches([]);
    setCurrentIndex(0);
    setSearching(false);
  }, []);

  const performSearch = useCallback(
    (pdfDoc: PDFDocumentProxy | null, term: string) => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
        debounceRef.current = null;
      }

      const trimmed = term.trim();
      if (!trimmed) {
        searchGenRef.current += 1;
        setMatches([]);
        setCurrentIndex(0);
        setSearching(false);
        return;
      }

      if (!isUsablePdfDoc(pdfDoc)) {
        setMatches([]);
        setCurrentIndex(0);
        setSearching(false);
        return;
      }

      const gen = ++searchGenRef.current;
      setSearching(true);

      debounceRef.current = setTimeout(() => {
        void (async () => {
          if (gen !== searchGenRef.current || !isUsablePdfDoc(pdfDoc)) {
            if (gen === searchGenRef.current) setSearching(false);
            return;
          }

          const allMatches: SearchMatch[] = [];
          const needle = trimmed.toLowerCase();

          try {
            for (let i = 1; i <= pdfDoc.numPages; i++) {
              if (gen !== searchGenRef.current || !isUsablePdfDoc(pdfDoc)) {
                return;
              }

              let page;
              try {
                page = await pdfDoc.getPage(i);
              } catch {
                // Document destroyed mid-loop — bail quietly.
                if (gen === searchGenRef.current) {
                  setMatches([]);
                  setCurrentIndex(0);
                  setSearching(false);
                }
                return;
              }

              if (gen !== searchGenRef.current) return;

              let content;
              try {
                content = await page.getTextContent();
              } catch {
                continue;
              }

              if (gen !== searchGenRef.current) return;

              const textItems = content.items as TextItem[];
              for (const item of textItems) {
                if (!item?.str?.toLowerCase().includes(needle)) continue;
                allMatches.push({
                  page: i,
                  rects: [
                    {
                      left: item.transform[4],
                      top: item.transform[5],
                      width: 50,
                      height: 10,
                    },
                  ],
                });
              }
            }

            if (gen !== searchGenRef.current) return;
            setMatches(allMatches);
            setCurrentIndex(0);
          } catch (err) {
            if (gen !== searchGenRef.current) return;
            console.warn("PDF text search failed:", err);
            setMatches([]);
            setCurrentIndex(0);
          } finally {
            if (gen === searchGenRef.current) setSearching(false);
          }
        })();
      }, 220);
    },
    [],
  );

  return {
    searchTerm,
    setSearchTerm,
    matches,
    currentIndex,
    setCurrentIndex,
    searching,
    performSearch,
    clearSearch,
  };
}
