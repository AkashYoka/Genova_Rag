"use client";

import { formatPageRange } from "@/lib/document-hierarchy";
import type { SectionTableMeta } from "@/types/document-hierarchy";

export type ExplorerTableRow = SectionTableMeta & {
  section_id?: string;
  section_title?: string;
};

interface TablesPanelProps {
  tables: ExplorerTableRow[];
  loading?: boolean;
  error?: string | null;
  emptyLabel?: string;
  onSelectSection?: (sectionId: string) => void;
}

export function TablesPanel({
  tables,
  loading,
  error,
  emptyLabel = "No tables associated with this selection.",
  onSelectSection,
}: TablesPanelProps) {
  if (loading) {
    return (
      <p className="py-10 text-center text-sm text-[var(--gd-muted)]">
        Loading tables…
      </p>
    );
  }

  if (error) {
    return (
      <p className="rounded-lg border border-[var(--gd-negative)]/30 bg-[var(--gd-negative)]/5 px-4 py-3 text-sm text-[var(--gd-negative)]">
        {error}
      </p>
    );
  }

  if (!tables.length) {
    return (
      <p className="rounded-lg border border-dashed border-[var(--gd-border)] px-4 py-10 text-center text-sm text-[var(--gd-muted)]">
        {emptyLabel}
      </p>
    );
  }

  return (
    <ol className="m-0 list-none divide-y divide-[var(--gd-rule)] overflow-hidden rounded-xl border border-[var(--gd-border)] bg-[var(--gd-surface)] p-0">
      {tables.map((table, index) => {
        const page = formatPageRange(table.page, table.page_end);
        const headers = Array.isArray(table.headers) ? table.headers : [];
        const title =
          table.caption || table.table_id || `Table ${index + 1}`;
        const key = `${table.table_id || "table"}-${index}`;
        return (
          <li key={key} className="px-4 py-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="text-sm font-medium text-[var(--gd-text)]">
                  <span className="mr-2 tabular-nums text-[var(--gd-muted)]">
                    {index + 1}.
                  </span>
                  {title}
                </p>
                <p className="mt-1 text-[11px] text-[var(--gd-muted)]">
                  {[
                    page,
                    table.n_rows != null ? `${table.n_rows} rows` : null,
                    table.n_cols != null ? `${table.n_cols} cols` : null,
                    table.section_title ? `§ ${table.section_title}` : null,
                  ]
                    .filter(Boolean)
                    .join(" · ")}
                </p>
              </div>
              {table.section_id && onSelectSection ? (
                <button
                  type="button"
                  className="shrink-0 text-xs text-[var(--gd-accent)] hover:underline cursor-pointer"
                  onClick={() => onSelectSection(table.section_id!)}
                >
                  Open part
                </button>
              ) : null}
            </div>
            {headers.length > 0 ? (
              <p className="mt-2 truncate text-xs text-[var(--gd-muted)]">
                Headers: {headers.join(" · ")}
              </p>
            ) : null}
          </li>
        );
      })}
    </ol>
  );
}
