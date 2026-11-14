"use client";

export function HowToRead() {
  return (
    <aside className="rounded-xl border border-[var(--gd-border)] bg-[var(--gd-surface)] p-5">
      <h2 className="text-base font-semibold text-[var(--gd-text)]">
        How to read this page
      </h2>
      <dl className="mt-4 space-y-4 text-sm leading-relaxed">
        <div>
          <dt className="font-medium text-[var(--gd-text)]">Contents</dt>
          <dd className="mt-1 text-[var(--gd-muted)]">
            Lists document parts in reading order. Click a part to open its
            summary and source text.
          </dd>
        </div>
        <div>
          <dt className="font-medium text-[var(--gd-text)]">List and Tree</dt>
          <dd className="mt-1 text-[var(--gd-muted)]">
            List is a flat numbered index. Tree is a visual map of document
            parts, nested sections, paragraphs, and tables — not an entity
            knowledge graph.
          </dd>
        </div>
        <div>
          <dt className="font-medium text-[var(--gd-text)]">Tables</dt>
          <dd className="mt-1 text-[var(--gd-muted)]">
            Opens a catalog of tables found in the index. Use a table’s section
            link to jump to that part.
          </dd>
        </div>
        <div>
          <dt className="font-medium text-[var(--gd-text)]">Related parts</dt>
          <dd className="mt-1 text-[var(--gd-muted)]">
            Inside a part, related lists show parent, neighbors, and children
            from the hierarchy — not a knowledge graph.
          </dd>
        </div>
      </dl>
    </aside>
  );
}
