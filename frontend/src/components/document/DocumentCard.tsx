"use client";

import type { DocInfo } from "@/app/dashboard/page";
import KeywordTags from "./KeywordTags";

interface DocumentCardProps {
  document: DocInfo;
  isIndexing?: boolean;
  compact?: boolean;
}

export default function DocumentCard({
  document,
  isIndexing,
  compact = false,
}: DocumentCardProps) {
  const indexing =
    isIndexing ??
    (document.status === "processing" || document.status === "pending");

  return (
    <KeywordTags
      keywords={document.keywords}
      isIndexing={indexing}
      compact={compact}
      maxVisible={compact ? 2 : 7}
    />
  );
}
