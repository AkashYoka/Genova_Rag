"use client";

import { Suspense, useMemo } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { RequireAuth } from "@/components/auth/RequireAuth";
import Header from "@/components/layout/Header";
import { DocumentExplorer } from "@/components/document/explorer/DocumentExplorer";
import { Skeleton } from "@/components/ui/skeleton";
import "../../dashboard/dashboard.css";

function ExplorePageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const documentId = searchParams.get("docId")?.trim() || "";
  const name = searchParams.get("name")?.trim() || undefined;
  const pageCountRaw = searchParams.get("pages");
  const pageCount = useMemo(() => {
    if (!pageCountRaw) return undefined;
    const n = Number(pageCountRaw);
    return Number.isFinite(n) && n > 0 ? n : undefined;
  }, [pageCountRaw]);

  if (!documentId) {
    return (
      <div className="flex min-h-[50vh] flex-col items-center justify-center gap-3 px-6 text-center">
        <p className="text-sm text-[var(--gd-muted)]">
          Choose a document from the library to open the explorer.
        </p>
        <button
          type="button"
          className="text-sm text-[var(--gd-accent)] hover:underline cursor-pointer"
          onClick={() => router.replace("/documents")}
        >
          Back to documents
        </button>
      </div>
    );
  }

  return (
    <DocumentExplorer
      documentId={documentId}
      fallbackName={name}
      pageCount={pageCount}
    />
  );
}

function DocumentExplorePageContent() {
  return (
    <div className="genova-desk flex h-dvh flex-col overflow-hidden">
      <Header />
      <div className="min-h-0 flex-1 overflow-hidden">
        <Suspense
          fallback={
            <div className="space-y-4 p-6">
              <Skeleton className="h-16 w-full" />
              <Skeleton className="h-[420px] w-full" />
            </div>
          }
        >
          <ExplorePageInner />
        </Suspense>
      </div>
    </div>
  );
}

export default function DocumentExplorePage() {
  return (
    <RequireAuth>
      <DocumentExplorePageContent />
    </RequireAuth>
  );
}
