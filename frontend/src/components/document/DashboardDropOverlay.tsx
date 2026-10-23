"use client";

import { UploadCloud } from "lucide-react";

interface DashboardDropOverlayProps {
  isDraggingOver: boolean;
  dropZoneProps: {
    onDragEnter: (e: React.DragEvent) => void;
    onDragOver: (e: React.DragEvent) => void;
    onDragLeave: (e: React.DragEvent) => void;
    onDrop: (e: React.DragEvent) => void;
  };
}

export default function DashboardDropOverlay({
  isDraggingOver,
  dropZoneProps,
}: DashboardDropOverlayProps) {
  if (!isDraggingOver) return null;

  return (
    <div
      aria-label="Drop files here to upload"
      aria-live="assertive"
      role="region"
      className="fixed inset-0 z-[100] flex items-center justify-center"
      {...dropZoneProps}
    >
      {/* Blurred backdrop */}
      <div className="absolute inset-0 bg-background/80 backdrop-blur-sm animate-in fade-in duration-150" />

      {/* Animated dashed border card */}
      <div
        className={[
          "relative z-10 flex flex-col items-center justify-center gap-4",
          "rounded-3xl border-2 border-dashed border-primary",
          "bg-gradient-to-br from-primary/10 to-[oklch(0.86_0.16_125/0.08)] px-16 py-14 shadow-2xl shadow-primary/20",
          "animate-in zoom-in-95 fade-in duration-200",
        ].join(" ")}
      >
        {/* Pulsing icon ring */}
        <div className="relative flex items-center justify-center">
          <span className="absolute h-20 w-20 rounded-full bg-primary/15 animate-ping" />
          <span className="absolute h-16 w-16 rounded-full bg-primary/20" />
          <UploadCloud className="relative w-10 h-10 text-primary animate-bounce" />
        </div>

        <div className="text-center space-y-1">
          <p className="font-display text-lg font-semibold text-foreground tracking-tight">
            Drop files to upload
          </p>
          <p className="text-sm text-muted-foreground">
            PDF, DOCX, TXT, MD supported
          </p>
        </div>
      </div>
    </div>
  );
}
