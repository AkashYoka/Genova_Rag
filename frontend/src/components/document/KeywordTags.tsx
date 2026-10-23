"use client";

import { useState } from "react";

const PALETTE = [
  "bg-rose-100 text-rose-700",
  "bg-amber-100 text-amber-700",
  "bg-emerald-100 text-emerald-700",
  "bg-sky-100 text-sky-700",
  "bg-violet-100 text-violet-700",
  "bg-pink-100 text-pink-700",
  "bg-teal-100 text-teal-700",
];

function colorFor(word: string): string {
  let hash = 0;
  for (let i = 0; i < word.length; i++) {
    hash = (hash * 31 + word.charCodeAt(i)) & 0xffffffff;
  }
  return PALETTE[Math.abs(hash) % PALETTE.length];
}

interface KeywordTagsProps {
  keywords?: string[];
  isIndexing?: boolean;
  maxVisible?: number;
  compact?: boolean;
}

export default function KeywordTags({
  keywords = [],
  isIndexing = false,
  maxVisible = 7,
  compact = false,
}: KeywordTagsProps) {
  const [showAll, setShowAll] = useState(false);

  if (isIndexing) {
    if (compact) return null;
    return (
      <div className="mt-2 flex flex-wrap gap-1">
        {Array.from({ length: 5 }).map((_, i) => (
          <span
            key={i}
            className="inline-block h-5 animate-pulse rounded-full bg-gray-200"
            style={{ width: `${48 + (i % 3) * 16}px` }}
          />
        ))}
      </div>
    );
  }

  if (!keywords.length) return null;

  const visible = showAll ? keywords : keywords.slice(0, maxVisible);
  const overflow = keywords.length - maxVisible;

  if (compact) {
    const label = [
      ...visible,
      !showAll && overflow > 0 ? `+${overflow}` : null,
    ]
      .filter(Boolean)
      .join(" · ");

    return (
      <>
        <span className="shrink-0 opacity-40" aria-hidden>
          ·
        </span>
        <span
          className="min-w-0 truncate"
          title={keywords.join(", ")}
          onClick={
            overflow > 0 && !showAll
              ? (e) => {
                  e.stopPropagation();
                  setShowAll(true);
                }
              : undefined
          }
        >
          {label}
        </span>
      </>
    );
  }

  return (
    <div className="mt-2 flex flex-wrap gap-1">
      {visible.map((kw) => (
        <span
          key={kw}
          className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${colorFor(kw)}`}
        >
          {kw}
        </span>
      ))}
      {!showAll && overflow > 0 && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            setShowAll(true);
          }}
          className="inline-flex items-center rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-500 transition-colors hover:bg-gray-200"
        >
          +{overflow} more
        </button>
      )}
    </div>
  );
}
