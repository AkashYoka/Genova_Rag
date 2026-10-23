"use client";

import { cn } from "@/lib/utils";

/** React Bits–style aurora mesh background (CSS-driven). */
export function Aurora({ className }: { className?: string }) {
  return (
    <div
      aria-hidden
      className={cn(
        "pointer-events-none absolute inset-0 overflow-hidden",
        className,
      )}
    >
      <div className="absolute -top-1/4 left-1/4 h-[55vmin] w-[55vmin] rounded-full bg-[radial-gradient(circle,oklch(0.72_0.14_195/0.45),transparent_70%)] blur-3xl animate-[genova-drift_18s_ease-in-out_infinite]" />
      <div className="absolute top-1/3 -right-1/6 h-[50vmin] w-[50vmin] rounded-full bg-[radial-gradient(circle,oklch(0.78_0.16_125/0.35),transparent_70%)] blur-3xl animate-[genova-drift_22s_ease-in-out_infinite_reverse]" />
      <div className="absolute -bottom-1/4 left-1/3 h-[45vmin] w-[45vmin] rounded-full bg-[radial-gradient(circle,oklch(0.65_0.12_250/0.28),transparent_70%)] blur-3xl animate-[genova-drift_26s_ease-in-out_infinite]" />
      <div
        className="absolute inset-0 opacity-[0.035]"
        style={{
          backgroundImage:
            "linear-gradient(to right, currentColor 1px, transparent 1px), linear-gradient(to bottom, currentColor 1px, transparent 1px)",
          backgroundSize: "48px 48px",
        }}
      />
    </div>
  );
}
