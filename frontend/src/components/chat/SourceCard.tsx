"use client";

import { useId, useState } from "react";
import type { SourceBoundingBox, SourceChunk } from "@/store/chat-store";
import { resolveHighlightRects } from "@/lib/citation-highlight";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { ChevronDown, ChevronUp, FileText, Eye, TextQuote } from "lucide-react";
import { cn } from "@/lib/utils";

const EXCERPT_THRESHOLD = 200;

type ConfidenceLevel = "High" | "Medium" | "Low" | "Unknown";

interface ConfidenceBadgeMeta {
  label: ConfidenceLevel;
  className: string;
}

const normalizeMetricValue = (value?: number) => {
  if (typeof value !== "number" || Number.isNaN(value)) return undefined;
  return value > 1 ? value / 100 : value;
};

const formatMetricValue = (value?: number) => {
  const normalizedValue = normalizeMetricValue(value);
  if (normalizedValue === undefined) return "N/A";
  return `${Math.round(normalizedValue * 100)}%`;
};

const getConfidenceBadgeMeta = (value?: number): ConfidenceBadgeMeta => {
  const normalizedValue = normalizeMetricValue(value);

  if (normalizedValue === undefined) {
    return {
      label: "Unknown",
      className:
        "border-[var(--gd-border)] bg-[var(--gd-raised)] text-[var(--gd-muted)]",
    };
  }

  if (normalizedValue >= 0.8) {
    return {
      label: "High",
      className:
        "border-[var(--gd-positive)]/30 bg-[color-mix(in_oklab,var(--gd-positive)_10%,transparent)] text-[var(--gd-positive)]",
    };
  }

  if (normalizedValue >= 0.5) {
    return {
      label: "Medium",
      className:
        "border-[var(--gd-warning)]/30 bg-[color-mix(in_oklab,var(--gd-warning)_10%,transparent)] text-[var(--gd-warning)]",
    };
  }

  return {
    label: "Low",
    className:
      "border-[var(--gd-negative)]/30 bg-[color-mix(in_oklab,var(--gd-negative)_10%,transparent)] text-[var(--gd-negative)]",
  };
};

const getPrimarySourceMetric = (source: SourceChunk) =>
  source.confidence ?? source.score;

const hasValidPage = (source: SourceChunk) =>
  typeof source.page === "number" &&
  Number.isFinite(source.page) &&
  source.page >= 1;

const canOpenInViewer = (source: SourceChunk) =>
  hasValidPage(source) || Boolean(source.text?.trim());

const MetricChip = ({
  label,
  value,
}: {
  label: "Score" | "Confidence";
  value?: number;
}) => {
  const badgeMeta = getConfidenceBadgeMeta(value);

  return (
    <span
      className={cn(
        "inline-flex h-5 items-center rounded-md border px-1.5 text-[9px] font-medium",
        badgeMeta.className,
      )}
      title={`${label}: ${formatMetricValue(value)}`}
    >
      {label}: {badgeMeta.label}
    </span>
  );
};

export interface CitationClickPayload {
  page: number;
  filename?: string;
  excerpt?: string;
  highlightRects?: SourceBoundingBox[];
}

interface Props {
  sources: SourceChunk[];
  onPageClick: (payload: CitationClickPayload) => void;
}

export default function SourceCard({ sources = [], onPageClick }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [excerptOpen, setExcerptOpen] = useState<Set<number>>(new Set());
  const sourceListId = useId();
  const excerptBaseId = useId();

  if (sources.length === 0) return null;

  const toggleExcerpt = (index: number) => {
    setExcerptOpen((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  };

  const openCitation = (src: SourceChunk) => {
    if (!canOpenInViewer(src)) return;
    onPageClick({
      page: hasValidPage(src) ? src.page : 0,
      filename: src.filename,
      excerpt: src.text,
      highlightRects: resolveHighlightRects(src),
    });
  };

  return (
    <div className="gc-card">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
        aria-controls={sourceListId}
        aria-label={`${expanded ? "Collapse" : "Expand"} ${sources.length} cited source${sources.length > 1 ? "s" : ""}`}
        className="flex w-full items-center justify-between px-3 py-2.5 text-xs transition-colors hover:bg-[var(--gd-raised)]"
      >
        <span className="flex items-center gap-1.5 font-medium tracking-wide text-[var(--gd-muted)]">
          <FileText className="h-3.5 w-3.5 text-[var(--gd-accent)]" />
          {sources.length} source{sources.length > 1 ? "s" : ""} cited
        </span>
        {expanded ? (
          <ChevronUp className="h-3.5 w-3.5 text-[var(--gd-muted)]" />
        ) : (
          <ChevronDown className="h-3.5 w-3.5 text-[var(--gd-muted)]" />
        )}
      </button>

      {!expanded && (
        <div className="flex flex-wrap gap-1.5 px-3 pb-2.5">
          {sources.map((src, i) => {
            const badgeMeta = getConfidenceBadgeMeta(
              getPrimarySourceMetric(src),
            );
            const validPage = hasValidPage(src);
            const canOpen = canOpenInViewer(src);

            return (
              <Tooltip key={i}>
                <TooltipTrigger
                  type="button"
                  className="inline-flex"
                  disabled={!canOpen}
                  onClick={() => openCitation(src)}
                  aria-label={
                    validPage
                      ? `Go to source page ${src.page}. Confidence ${badgeMeta.label}`
                      : canOpen
                        ? `Open ${src.filename} and locate cited excerpt. Confidence ${badgeMeta.label}`
                        : `Source from ${src.filename}. Page unavailable. Confidence ${badgeMeta.label}`
                  }
                >
                  <span
                    className={cn(
                      "inline-flex h-5 items-center rounded-md border px-1.5 text-[10px] transition-colors",
                      badgeMeta.className,
                      canOpen
                        ? "cursor-pointer hover:opacity-80"
                        : "cursor-default opacity-70",
                    )}
                  >
                    {validPage ? `p.${src.page}` : "Page N/A"} -{" "}
                    {badgeMeta.label}
                  </span>
                </TooltipTrigger>
                <TooltipContent
                  side="top"
                  align="center"
                  className="max-w-xs p-2"
                >
                  <div className="mb-1 flex flex-wrap gap-1">
                    <MetricChip label="Score" value={src.score} />
                    <MetricChip label="Confidence" value={src.confidence} />
                  </div>
                  <p className="line-clamp-6 text-[11px] leading-relaxed">
                    {src.text}
                  </p>
                </TooltipContent>
              </Tooltip>
            );
          })}
        </div>
      )}

      {expanded && (
        <div id={sourceListId} className="border-t border-[var(--gd-border)]">
          {sources.map((src, i) => {
            const validPage = hasValidPage(src);
            const canOpen = canOpenInViewer(src);
            const excerptId = `${excerptBaseId}-excerpt-${i}`;
            const isExcerptOpen = excerptOpen.has(i);

            return (
              <div
                key={i}
                className="border-b border-[var(--gd-border)] px-3 py-2.5 transition-colors last:border-b-0 hover:bg-[var(--gd-raised)]"
              >
                <div className="mb-1.5 flex items-center justify-between gap-2">
                  <div className="flex min-w-0 flex-wrap items-center gap-1.5">
                    <span className="truncate text-[10px] font-medium text-[var(--gd-muted)]">
                      {src.filename}
                    </span>
                    <span className="inline-flex h-5 items-center rounded-md border border-[var(--gd-border)] px-1.5 text-[9px] text-[var(--gd-muted)]">
                      {validPage ? `Page ${src.page}` : "Page N/A"}
                    </span>
                    <MetricChip label="Score" value={src.score} />
                    <MetricChip label="Confidence" value={src.confidence} />
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 shrink-0 px-2 text-[10px] text-[var(--gd-muted)] hover:text-[var(--gd-text)]"
                    disabled={!canOpen}
                    onClick={() => openCitation(src)}
                    aria-label={
                      validPage
                        ? `View source page ${src.page}`
                        : canOpen
                          ? `View source in ${src.filename}`
                          : "Page reference unavailable for this source"
                    }
                  >
                    <Eye className="mr-1 h-3 w-3" />
                    {canOpen ? "View" : "N/A"}
                  </Button>
                </div>
                <p
                  id={excerptId}
                  className={cn(
                    "text-[11px] leading-relaxed text-[var(--gd-muted)]",
                    !isExcerptOpen && "line-clamp-3",
                  )}
                >
                  {src.text}
                </p>
                {src.text.length > EXCERPT_THRESHOLD && (
                  <button
                    type="button"
                    onClick={() => toggleExcerpt(i)}
                    aria-expanded={isExcerptOpen}
                    aria-controls={excerptId}
                    className="mt-1.5 flex items-center gap-1 text-[10px] text-[var(--gd-accent)] transition-colors hover:opacity-80"
                  >
                    <TextQuote className="h-3 w-3" />
                    {isExcerptOpen ? "Hide excerpt" : "Show excerpt"}
                  </button>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
