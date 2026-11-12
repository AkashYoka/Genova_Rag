"use client";

import {
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  formatPageRange,
  sectionSignalChips,
} from "@/lib/document-hierarchy";
import type { SectionTreeNode } from "@/types/document-hierarchy";

interface HierarchyTreeProps {
  sections: SectionTreeNode[];
  expandedIds: Set<string>;
  selectedSectionId: string | null;
  onToggleExpand: (sectionId: string) => void;
  onSelectSection: (sectionId: string) => void;
  onCollapseAll?: () => void;
  onExpandAll?: () => void;
}

function TreeNode({
  node,
  depth,
  expandedIds,
  selectedSectionId,
  onToggleExpand,
  onSelectSection,
}: {
  node: SectionTreeNode;
  depth: number;
  expandedIds: Set<string>;
  selectedSectionId: string | null;
  onToggleExpand: (sectionId: string) => void;
  onSelectSection: (sectionId: string) => void;
}) {
  const children = node.children ?? [];
  const hasChildren = children.length > 0;
  const expanded = expandedIds.has(node.section_id);
  const selected = selectedSectionId === node.section_id;
  const page = formatPageRange(node.page_start, node.page_end);
  const chips = sectionSignalChips(node);

  return (
    <li className="select-none">
      <div
        className={cn(
          "flex items-start gap-1 border-b border-[var(--gd-rule)] py-2.5 pr-3",
          selected ? "bg-[var(--gd-accent-soft)]" : "hover:bg-[var(--gd-raised)]/60",
        )}
        style={{ paddingLeft: `${12 + depth * 16}px` }}
      >
        <button
          type="button"
          className={cn(
            "mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded text-[var(--gd-muted)]",
            hasChildren
              ? "hover:bg-[var(--gd-active)] cursor-pointer"
              : "opacity-0 pointer-events-none",
          )}
          aria-label={expanded ? "Collapse section" : "Expand section"}
          onClick={(e) => {
            e.stopPropagation();
            if (hasChildren) onToggleExpand(node.section_id);
          }}
        >
          {hasChildren ? (
            expanded ? (
              <ChevronDown className="h-3.5 w-3.5" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5" />
            )
          ) : null}
        </button>

        <button
          type="button"
          className="min-w-0 flex-1 text-left cursor-pointer"
          onClick={() => onSelectSection(node.section_id)}
        >
          <span className="flex items-start justify-between gap-3">
            <span
              className={cn(
                "text-sm leading-snug text-[var(--gd-text)]",
                selected ? "font-semibold" : "font-medium",
              )}
            >
              {node.title || "Untitled section"}
            </span>
            {page ? (
              <span className="shrink-0 text-[11px] tabular-nums text-[var(--gd-muted)]">
                {page}
              </span>
            ) : null}
          </span>
          {chips.length > 0 ? (
            <span className="mt-1.5 flex flex-wrap gap-1.5">
              {chips.slice(0, 3).map((chip, chipIndex) => (
                <span
                  key={`${node.section_id}:chip:${chipIndex}`}
                  className="rounded-full border border-[var(--gd-border)] bg-[var(--gd-surface)] px-2 py-0.5 text-[11px] text-[var(--gd-muted)]"
                >
                  {chip}
                </span>
              ))}
            </span>
          ) : null}
        </button>
      </div>

      {hasChildren && expanded ? (
        <ul className="m-0 list-none p-0">
          {children.map((child) => (
            <TreeNode
              key={child.section_id}
              node={child}
              depth={depth + 1}
              expandedIds={expandedIds}
              selectedSectionId={selectedSectionId}
              onToggleExpand={onToggleExpand}
              onSelectSection={onSelectSection}
            />
          ))}
        </ul>
      ) : null}
    </li>
  );
}

/** Nested outline tree (screenshot Tree mode) — not a canvas graph. */
export function HierarchyTree({
  sections,
  expandedIds,
  selectedSectionId,
  onToggleExpand,
  onSelectSection,
  onCollapseAll,
  onExpandAll,
}: HierarchyTreeProps) {
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center justify-end gap-3 border-b border-[var(--gd-rule)] px-4 py-2">
        {onExpandAll ? (
          <button
            type="button"
            className="text-[11px] text-[var(--gd-muted)] hover:text-[var(--gd-text)] cursor-pointer"
            onClick={onExpandAll}
          >
            Expand all
          </button>
        ) : null}
        {onCollapseAll ? (
          <button
            type="button"
            className="text-[11px] text-[var(--gd-muted)] hover:text-[var(--gd-text)] cursor-pointer"
            onClick={onCollapseAll}
          >
            Collapse all
          </button>
        ) : null}
      </div>
      <ul className="m-0 min-h-0 flex-1 list-none overflow-y-auto p-0">
        {sections.length === 0 ? (
          <li className="px-4 py-10 text-center text-sm text-[var(--gd-muted)]">
            No sections indexed for this document.
          </li>
        ) : (
          sections.map((node) => (
            <TreeNode
              key={node.section_id}
              node={node}
              depth={0}
              expandedIds={expandedIds}
              selectedSectionId={selectedSectionId}
              onToggleExpand={onToggleExpand}
              onSelectSection={onSelectSection}
            />
          ))
        )}
      </ul>
    </div>
  );
}
