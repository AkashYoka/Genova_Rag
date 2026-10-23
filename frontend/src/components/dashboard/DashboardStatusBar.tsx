"use client";

import Chip from "@mui/material/Chip";
import Stack from "@mui/material/Stack";
import { FileText } from "lucide-react";
import type { DocInfo } from "@/app/dashboard/page";
import { FadeIn } from "@/components/bits/FadeIn";
import { cn } from "@/lib/utils";

function statusTone(
  status: string,
): "default" | "success" | "warning" | "error" | "info" {
  switch (status) {
    case "ready":
      return "success";
    case "processing":
    case "pending":
      return "warning";
    case "failed":
    case "error":
      return "error";
    default:
      return "info";
  }
}

function formatBytes(n: number): string {
  if (!n || n < 0) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export default function DashboardStatusBar({
  activeDoc,
  documentsCount,
  className,
}: {
  activeDoc: DocInfo | null;
  documentsCount: number;
  className?: string;
}) {
  return (
    <FadeIn
      delay={0.08}
      className={cn(
        "hidden md:flex items-center justify-between gap-3 px-4 py-2 rounded-2xl border border-border/50 bg-card/60 backdrop-blur-xl",
        className,
      )}
    >
      <div className="flex min-w-0 items-center gap-3">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-primary/15 text-primary">
          <FileText className="h-4 w-4" />
        </div>
        <div className="min-w-0">
          <p className="truncate font-display text-sm font-semibold tracking-tight">
            {activeDoc?.original_name ?? "No document selected"}
          </p>
          <p className="truncate text-[11px] text-muted-foreground">
            {activeDoc
              ? `${formatBytes(activeDoc.file_size)} · Library ${documentsCount}`
              : `${documentsCount} document${documentsCount === 1 ? "" : "s"} in library`}
          </p>
        </div>
      </div>

      <Stack
        direction="row"
        spacing={1}
        useFlexGap
        sx={{ flexWrap: "wrap", justifyContent: "flex-end" }}
      >
        {activeDoc ? (
          <>
            <Chip
              size="small"
              label={activeDoc.status}
              color={statusTone(activeDoc.status)}
              variant="outlined"
              sx={{ textTransform: "capitalize", fontWeight: 600 }}
            />
            <Chip
              size="small"
              label={`${activeDoc.page_count || "—"} pages`}
              variant="outlined"
            />
            <Chip
              size="small"
              label={`${activeDoc.chunk_count || 0} chunks`}
              variant="outlined"
            />
            <Chip
              size="small"
              label={
                activeDoc.uploaded_at
                  ? new Date(activeDoc.uploaded_at).toLocaleDateString()
                  : "—"
              }
              variant="outlined"
            />
          </>
        ) : (
          <Chip
            size="small"
            label="Select a PDF to inspect"
            variant="outlined"
          />
        )}
      </Stack>
    </FadeIn>
  );
}
