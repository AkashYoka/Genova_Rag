"use client";

import { useMemo } from "react";
import { Check, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

export type PipelineDocStatus =
  | "pending"
  | "processing"
  | "ready"
  | "failed"
  | string;

type StepState = "done" | "active" | "upcoming" | "failed";

type PipelineStep = {
  id: string;
  label: string;
  detail: string;
};

/** User-facing ingest pipeline aligned with backend processing_stage values. */
export const INGEST_PIPELINE: PipelineStep[] = [
  {
    id: "upload",
    label: "Upload to storage",
    detail: "PDF saved to S3 / local storage",
  },
  {
    id: "extract",
    label: "Extract text",
    detail: "Load file and read pages",
  },
  {
    id: "chunk",
    label: "Chunking",
    detail: "Split document into searchable chunks",
  },
  {
    id: "embed",
    label: "Embedding",
    detail: "Generate vector embeddings",
  },
  {
    id: "store",
    label: "Store in vector DB",
    detail: "Write chunks to Chroma / BM25",
  },
];

function activeStepIndex(
  status: PipelineDocStatus,
  stage: string | null | undefined,
  progress: number | null | undefined,
): number {
  const pct = typeof progress === "number" ? progress : null;
  const s = (stage || "").toLowerCase();

  if (status === "ready" || s === "completed" || pct === 100) {
    return INGEST_PIPELINE.length;
  }
  if (status === "failed") {
    if (s === "embedding" || (pct != null && pct >= 55)) return 3;
    if (s === "chunking" || (pct != null && pct >= 30)) return 2;
    if (s === "extracting" || (pct != null && pct >= 10)) return 1;
    return 0;
  }
  if (status === "pending" || s === "queued") {
    return 1;
  }

  if (s === "extracting") return 1;
  if (s === "chunking") return 2;
  if (s === "embedding") {
    if (pct != null && pct >= 90) return 4;
    return 3;
  }

  if (pct != null) {
    if (pct >= 92) return 4;
    if (pct >= 55) return 3;
    if (pct >= 30) return 2;
    if (pct >= 10) return 1;
    return 0;
  }

  return 1;
}

function stepState(
  index: number,
  active: number,
  status: PipelineDocStatus,
): StepState {
  if (status === "ready" || active >= INGEST_PIPELINE.length) return "done";
  if (status === "failed") {
    if (index < active) return "done";
    if (index === active) return "failed";
    return "upcoming";
  }
  if (status === "pending") {
    if (index === 0) return "done";
    if (index === 1) return "active";
    return "upcoming";
  }
  if (index < active) return "done";
  if (index === active) return "active";
  return "upcoming";
}

type Props = {
  status: PipelineDocStatus;
  processingStage?: string | null;
  processingProgress?: number | null;
  errorMessage?: string | null;
  className?: string;
};

export default function ProcessingPipelineStatus({
  status,
  processingStage,
  processingProgress,
  errorMessage,
  className,
}: Props) {
  const active = useMemo(
    () => activeStepIndex(status, processingStage, processingProgress),
    [status, processingStage, processingProgress],
  );

  const percent =
    typeof processingProgress === "number"
      ? Math.max(0, Math.min(100, Math.round(processingProgress)))
      : null;

  const triggerLabel =
    status === "ready"
      ? "Ready"
      : status === "failed"
        ? "Failed"
        : status === "pending"
          ? "Queued"
          : percent != null
            ? `${percent}%`
            : "Indexing";

  const showSpinner = status === "processing" || status === "pending";

  return (
    <TooltipProvider delay={120}>
      <Tooltip>
        <TooltipTrigger
          className={cn(
            "inline-flex max-w-full cursor-default items-center gap-1.5 text-xs tabular-nums outline-none",
            status === "failed"
              ? "text-[var(--gd-negative)]"
              : "text-[var(--gd-muted)]",
            className,
          )}
          aria-label={
            processingStage
              ? `Processing: ${processingStage}${percent != null ? ` ${percent}%` : ""} — hover for pipeline`
              : `${triggerLabel} — hover for pipeline`
          }
        >
          {showSpinner ? (
            <Loader2 className="h-3 w-3 shrink-0 animate-spin" aria-hidden />
          ) : status === "ready" ? (
            <span
              className="h-1.5 w-1.5 shrink-0 rounded-full bg-[var(--gd-positive)]"
              aria-hidden
            />
          ) : status === "failed" ? (
            <span
              className="h-1.5 w-1.5 shrink-0 rounded-full bg-[var(--gd-negative)]"
              aria-hidden
            />
          ) : (
            <span
              className="h-1.5 w-1.5 shrink-0 rounded-full bg-[var(--gd-warning)]"
              aria-hidden
            />
          )}
          {triggerLabel}
        </TooltipTrigger>

        <TooltipContent
          side="right"
          align="start"
          sideOffset={10}
          className={cn(
            "w-[270px] max-w-[270px] flex-col items-stretch gap-0 rounded-lg border border-[var(--gd-border)]",
            "bg-[var(--gd-surface)] p-3 text-left text-[var(--gd-text)] shadow-md",
            "**:data-[slot=kbd]:hidden",
          )}
        >
          <div className="mb-2.5 flex items-baseline justify-between gap-2">
            <p className="text-[11px] font-medium tracking-wide text-[var(--gd-text)]">
              Ingest pipeline
            </p>
            {percent != null && status !== "ready" && status !== "failed" ? (
              <span className="text-[10px] tabular-nums text-[var(--gd-faint)]">
                {percent}%
              </span>
            ) : null}
          </div>

          <ol className="relative m-0 flex list-none flex-col gap-0 p-0">
            {INGEST_PIPELINE.map((step, index) => {
              const state = stepState(index, active, status);
              const isLast = index === INGEST_PIPELINE.length - 1;
              return (
                <li
                  key={step.id}
                  className="relative flex gap-2.5 pb-3 last:pb-0"
                >
                  {!isLast ? (
                    <span
                      aria-hidden
                      className={cn(
                        "absolute left-[7px] top-4 h-[calc(100%-8px)] w-px",
                        state === "done"
                          ? "bg-[var(--gd-data)]"
                          : "bg-[var(--gd-border)]",
                        state === "active" &&
                          "pipeline-rail-pulse bg-[var(--gd-data)]/40",
                      )}
                    />
                  ) : null}

                  <span
                    className={cn(
                      "relative z-[1] mt-0.5 flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-full border",
                      state === "done" &&
                        "border-[var(--gd-data)] bg-[var(--gd-data)] text-white",
                      state === "active" &&
                        "border-[var(--gd-data)] bg-[var(--gd-accent-soft)] text-[var(--gd-data)]",
                      state === "upcoming" &&
                        "border-[var(--gd-border)] bg-[var(--gd-surface)] text-transparent",
                      state === "failed" &&
                        "border-[var(--gd-negative)] bg-[var(--gd-negative)] text-white",
                    )}
                  >
                    {state === "done" || state === "failed" ? (
                      <Check className="h-2.5 w-2.5" strokeWidth={3} />
                    ) : state === "active" ? (
                      <Loader2 className="h-2.5 w-2.5 animate-spin" />
                    ) : null}
                  </span>

                  <div className="min-w-0 flex-1 pt-px">
                    <p
                      className={cn(
                        "text-[11px] font-medium leading-tight",
                        state === "active"
                          ? "text-[var(--gd-text)]"
                          : state === "done"
                            ? "text-[var(--gd-muted)]"
                            : state === "failed"
                              ? "text-[var(--gd-negative)]"
                              : "text-[var(--gd-faint)]",
                      )}
                    >
                      {step.label}
                      {state === "active" ? (
                        <span className="ml-1 inline-flex gap-0.5 align-middle">
                          <span className="pipeline-dot h-1 w-1 rounded-full bg-[var(--gd-data)]" />
                          <span className="pipeline-dot pipeline-dot-delay-1 h-1 w-1 rounded-full bg-[var(--gd-data)]" />
                          <span className="pipeline-dot pipeline-dot-delay-2 h-1 w-1 rounded-full bg-[var(--gd-data)]" />
                        </span>
                      ) : null}
                    </p>
                    <p className="mt-0.5 text-[10px] leading-snug text-[var(--gd-faint)]">
                      {step.detail}
                    </p>
                  </div>
                </li>
              );
            })}
          </ol>

          {status === "failed" && errorMessage ? (
            <p className="mt-2 border-t border-[var(--gd-rule)] pt-2 text-[10px] leading-snug text-[var(--gd-negative)]">
              {errorMessage}
            </p>
          ) : null}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
