"use client";

import { cn } from "@/lib/utils";
import {
  formatPageRange,
  sectionSignalChips,
} from "@/lib/document-hierarchy";
import type { SectionTreeNode } from "@/types/document-hierarchy";

interface ContentsListProps {
  sections: SectionTreeNode[];
  selectedSectionId: string | null;
  onSelectSection: (sectionId: string) => void;
}

/** Flat reading-order list of document parts (screenshot List mode). */
export function ContentsList({
  sections,
  selectedSectionId,
  onSelectSection,
}: ContentsListProps) {
  if (!sections.length) {
    return (
      <p className="px-4 py-10 text-center text-sm text-[var(--gd-muted)]">
        No sections indexed for this document.
      </p>
    );
  }

  return (
    <ol className="m-0 list-none divide-y divide-[var(--gd-rule)] p-0">
      {sections.map((section, index) => {
        const selected = selectedSectionId === section.section_id;
        const page = formatPageRange(section.page_start, section.page_end);
        const chips = sectionSignalChips(section);
        return (
          <li key={section.section_id}>
            <button
              type="button"
              onClick={() => onSelectSection(section.section_id)}
              className={cn(
                "flex w-full items-start gap-3 px-4 py-3.5 text-left transition-colors cursor-pointer",
                selected
                  ? "bg-[var(--gd-accent-soft)]"
                  : "hover:bg-[var(--gd-raised)]/70",
              )}
            >
              <span
                className={cn(
                  "mt-0.5 w-7 shrink-0 text-right text-sm tabular-nums",
                  selected
                    ? "font-semibold text-[var(--gd-accent)]"
                    : "text-[var(--gd-muted)]",
                )}
              >
                {index + 1}.
              </span>
              <span className="min-w-0 flex-1">
                <span className="flex items-start justify-between gap-3">
                  <span
                    className={cn(
                      "text-sm leading-snug text-[var(--gd-text)]",
                      selected ? "font-semibold" : "font-medium",
                    )}
                  >
                    {section.title || "Untitled section"}
                  </span>
                  {page ? (
                    <span className="shrink-0 pt-0.5 text-[11px] tabular-nums text-[var(--gd-muted)]">
                      {page}
                    </span>
                  ) : null}
                </span>
                {chips.length > 0 ? (
                  <span className="mt-2 flex flex-wrap gap-1.5">
                    {chips.map((chip, chipIndex) => (
                      <span
                        key={`${section.section_id}:chip:${chipIndex}`}
                        className="rounded-full border border-[var(--gd-border)] bg-[var(--gd-surface)] px-2 py-0.5 text-[11px] text-[var(--gd-muted)]"
                      >
                        {chip}
                      </span>
                    ))}
                  </span>
                ) : null}
              </span>
            </button>
          </li>
        );
      })}
    </ol>
  );
}
