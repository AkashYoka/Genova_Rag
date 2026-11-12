"use client";

import { formatPageRange } from "@/lib/document-hierarchy";
import type {
  RelatedSectionsResponse,
  SectionTreeNode,
} from "@/types/document-hierarchy";

interface RelatedListsProps {
  related: RelatedSectionsResponse | null;
  loading?: boolean;
  error?: string | null;
  onSelectSection: (sectionId: string) => void;
}

function LinkButton({
  section,
  onSelect,
}: {
  section: SectionTreeNode;
  onSelect: () => void;
}) {
  const page = formatPageRange(section.page_start, section.page_end);
  return (
    <button
      type="button"
      onClick={onSelect}
      className="block w-full rounded-md px-2 py-1.5 text-left text-sm text-[var(--gd-accent)] hover:bg-[var(--gd-accent-soft)] cursor-pointer"
    >
      <span className="block truncate font-medium">{section.title}</span>
      {page ? (
        <span className="text-[11px] text-[var(--gd-muted)]">{page}</span>
      ) : null}
    </button>
  );
}

function Column({
  title,
  sections,
  onSelectSection,
}: {
  title: string;
  sections: SectionTreeNode[];
  onSelectSection: (sectionId: string) => void;
}) {
  if (!sections.length) return null;
  return (
    <div className="min-w-0">
      <h4 className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-[var(--gd-muted)]">
        {title}
      </h4>
      <ul className="m-0 space-y-0.5 list-none border-l border-[var(--gd-border)] pl-2 p-0">
        {sections.map((section) => (
          <li key={`${title}:${section.section_id}`}>
            <LinkButton
              section={section}
              onSelect={() => onSelectSection(section.section_id)}
            />
          </li>
        ))}
      </ul>
    </div>
  );
}

/**
 * Plain related-section lists from the hierarchy API.
 * No canvas, no node graph — only navigable text links.
 */
export function RelatedLists({
  related,
  loading,
  error,
  onSelectSection,
}: RelatedListsProps) {
  if (loading) {
    return (
      <p className="text-sm text-[var(--gd-muted)]">Loading related parts…</p>
    );
  }
  if (error) {
    return (
      <p className="text-sm text-[var(--gd-negative)]">{error}</p>
    );
  }
  if (!related) return null;

  const refersTo: SectionTreeNode[] = [
    ...(related.children ?? []),
    ...(related.next ? [related.next] : []),
  ];
  const referredFrom: SectionTreeNode[] = [
    ...(related.parent ? [related.parent] : []),
    ...(related.previous ? [related.previous] : []),
    ...(related.siblings ?? []),
  ];

  // Deduplicate by section_id while preserving order.
  const uniq = (items: SectionTreeNode[]) => {
    const seen = new Set<string>();
    return items.filter((item) => {
      if (seen.has(item.section_id)) return false;
      seen.add(item.section_id);
      return true;
    });
  };

  const outgoing = uniq(refersTo);
  const incoming = uniq(referredFrom);

  if (!outgoing.length && !incoming.length) {
    return (
      <p className="text-sm text-[var(--gd-muted)]">
        No related parts returned for this section.
      </p>
    );
  }

  return (
    <div className="grid gap-6 sm:grid-cols-2">
      <Column
        title="This part relates to"
        sections={outgoing}
        onSelectSection={onSelectSection}
      />
      <Column
        title="Related from"
        sections={incoming}
        onSelectSection={onSelectSection}
      />
    </div>
  );
}
