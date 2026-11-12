"use client";

import { ArrowLeft, ArrowUpLeft, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { formatPageRange } from "@/lib/document-hierarchy";
import type {
  HierarchyChunk,
  RelatedSectionsResponse,
  SectionTableMeta,
  SectionTreeNode,
} from "@/types/document-hierarchy";
import { RelatedLists } from "./RelatedLists";

interface SectionDetailPanelProps {
  section: SectionTreeNode;
  partNumber: number;
  partTotal: number;
  breadcrumb: SectionTreeNode[];
  chunks: HierarchyChunk[];
  tables: SectionTableMeta[];
  related: RelatedSectionsResponse | null;
  selectedChunkId: string | null;
  loadingChunks?: boolean;
  loadingRelated?: boolean;
  chunksError?: string | null;
  relatedError?: string | null;
  onSelectChunk: (chunkId: string) => void;
  onSelectSection: (sectionId: string) => void;
  onBackToContents: () => void;
  onBackToParent?: () => void;
}

function ChunkReader({
  chunk,
  onBack,
}: {
  chunk: HierarchyChunk;
  onBack: () => void;
}) {
  const page = formatPageRange(chunk.page_start, chunk.page_end);
  return (
    <div className="space-y-4">
      <Button
        variant="ghost"
        size="sm"
        className="h-8 gap-1.5 px-2 text-xs"
        onClick={onBack}
      >
        <ArrowUpLeft className="h-3.5 w-3.5" />
        Back to part
      </Button>
      <div className="text-xs text-[var(--gd-muted)]">
        Source text · chunk #{chunk.chunk_index}
        {chunk.chunk_type ? ` · ${chunk.chunk_type}` : ""}
        {page ? ` · ${page}` : ""}
      </div>
      <pre className="max-h-[min(64vh,640px)] overflow-auto whitespace-pre-wrap rounded-lg border border-[var(--gd-border)] bg-[var(--gd-surface)] p-5 font-[family-name:var(--font-geist-mono),ui-monospace,monospace] text-[13px] leading-relaxed text-[var(--gd-text)]">
        {chunk.content || "(empty)"}
      </pre>
    </div>
  );
}

export function SectionDetailPanel({
  section,
  partNumber,
  partTotal,
  breadcrumb,
  chunks,
  tables,
  related,
  selectedChunkId,
  loadingChunks,
  loadingRelated,
  chunksError,
  relatedError,
  onSelectChunk,
  onSelectSection,
  onBackToContents,
  onBackToParent,
}: SectionDetailPanelProps) {
  const selectedChunk =
    selectedChunkId != null
      ? chunks.find((c) => c.chunk_id === selectedChunkId) ?? null
      : null;

  if (selectedChunk) {
    return (
      <ChunkReader
        chunk={selectedChunk}
        onBack={() => onSelectChunk("")}
      />
    );
  }

  const page = formatPageRange(section.page_start, section.page_end);
  const pathLabel =
    breadcrumb.length > 1
      ? breadcrumb.map((b) => b.title).join(" › ")
      : null;

  return (
    <div className="mx-auto max-w-3xl space-y-8">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant="ghost"
          size="sm"
          className="h-8 gap-1.5 px-2 text-xs"
          onClick={onBackToContents}
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Contents
        </Button>
        {onBackToParent ? (
          <Button
            variant="ghost"
            size="sm"
            className="h-8 gap-1.5 px-2 text-xs"
            onClick={onBackToParent}
          >
            <ArrowUpLeft className="h-3.5 w-3.5" />
            Parent
          </Button>
        ) : null}
      </div>

      <header className="space-y-2 border-b border-[var(--gd-rule)] pb-6">
        <p className="text-[11px] font-medium uppercase tracking-[0.08em] text-[var(--gd-muted)]">
          Part {partNumber} of {partTotal}
          {page ? ` · ${page}` : ""}
        </p>
        <h2 className="text-[1.75rem] font-semibold leading-tight tracking-tight text-[var(--gd-text)]">
          {section.title || "Untitled section"}
        </h2>
        {pathLabel ? (
          <p className="text-sm text-[var(--gd-muted)]">{pathLabel}</p>
        ) : null}
      </header>

      {section.summary ? (
        <section className="space-y-2">
          <h3 className="text-[11px] font-semibold uppercase tracking-[0.08em] text-[var(--gd-muted)]">
            In short
          </h3>
          <p className="text-[15px] leading-relaxed text-[var(--gd-text)]">
            {section.summary}
          </p>
        </section>
      ) : null}

      <section className="space-y-3">
        <h3 className="text-[11px] font-semibold uppercase tracking-[0.08em] text-[var(--gd-muted)]">
          Source text
        </h3>
        {loadingChunks ? (
          <div className="flex items-center gap-2 py-6 text-sm text-[var(--gd-muted)]">
            <Loader2 className="h-4 w-4 animate-spin" />
            Loading chunks…
          </div>
        ) : chunksError ? (
          <p className="text-sm text-[var(--gd-negative)]">{chunksError}</p>
        ) : chunks.length === 0 ? (
          <p className="text-sm text-[var(--gd-muted)]">
            No semantic chunks are attached directly to this part.
          </p>
        ) : (
          <div className="space-y-3">
            {chunks.map((chunk) => {
              const preview = (chunk.content || "")
                .replace(/\s+/g, " ")
                .trim();
              const chunkPage = formatPageRange(
                chunk.page_start,
                chunk.page_end,
              );
              return (
                <button
                  key={chunk.chunk_id}
                  type="button"
                  onClick={() => onSelectChunk(chunk.chunk_id)}
                  className="w-full rounded-lg border border-[var(--gd-border)] bg-[var(--gd-surface)] px-4 py-3 text-left transition-colors hover:border-[var(--gd-accent)] cursor-pointer"
                >
                  <div className="mb-1.5 flex flex-wrap gap-2 text-[11px] text-[var(--gd-muted)]">
                    <span>Chunk #{chunk.chunk_index}</span>
                    {chunk.chunk_type ? <span>{chunk.chunk_type}</span> : null}
                    {chunkPage ? <span>{chunkPage}</span> : null}
                  </div>
                  <p className="line-clamp-4 text-sm leading-relaxed text-[var(--gd-text)]">
                    {preview || "Open to read source text"}
                  </p>
                </button>
              );
            })}
          </div>
        )}
      </section>

      {tables.length > 0 ? (
        <section className="space-y-3">
          <h3 className="text-[11px] font-semibold uppercase tracking-[0.08em] text-[var(--gd-muted)]">
            Tables
          </h3>
          <div className="flex flex-wrap gap-2">
            {tables.map((table, index) => {
              const label =
                table.caption ||
                table.table_id ||
                `Table ${index + 1}`;
              const pageLabel = formatPageRange(table.page, table.page_end);
              return (
                <span
                  key={String(table.table_id || index)}
                  className="inline-flex max-w-full items-center gap-2 rounded-full border border-[var(--gd-border)] bg-[var(--gd-surface)] px-3 py-1.5 text-xs text-[var(--gd-text)]"
                  title={label}
                >
                  <span className="truncate">{label}</span>
                  {pageLabel ? (
                    <span className="shrink-0 text-[var(--gd-muted)]">
                      {pageLabel}
                    </span>
                  ) : null}
                </span>
              );
            })}
          </div>
        </section>
      ) : null}

      <section className="space-y-3">
        <h3 className="text-[11px] font-semibold uppercase tracking-[0.08em] text-[var(--gd-muted)]">
          Related parts
        </h3>
        <RelatedLists
          related={related}
          loading={loadingRelated}
          error={relatedError}
          onSelectSection={onSelectSection}
        />
      </section>
    </div>
  );
}
