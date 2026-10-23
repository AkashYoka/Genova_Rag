"use client";

import type { ReactNode } from "react";
import { cn } from "@/lib/utils";
import { SpotlightCard } from "@/components/bits/SpotlightCard";
import { FadeIn } from "@/components/bits/FadeIn";

export function DashboardPanel({
  children,
  className,
  label,
  delay = 0,
  spotlight = false,
}: {
  children: ReactNode;
  className?: string;
  label?: string;
  delay?: number;
  spotlight?: boolean;
}) {
  const shell = (
    <div
      className={cn(
        "relative flex h-full min-h-0 flex-col overflow-hidden rounded-2xl border border-border/55 bg-card/55 backdrop-blur-xl shadow-[0_24px_80px_-48px_oklch(0.2_0.04_230/0.9)]",
        className,
      )}
    >
      {label ? (
        <div className="pointer-events-none absolute left-3 top-2 z-20">
          <span className="rounded-md bg-background/70 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-[0.16em] text-muted-foreground backdrop-blur-sm">
            {label}
          </span>
        </div>
      ) : null}
      <div className="relative z-10 flex h-full min-h-0 flex-col">{children}</div>
    </div>
  );

  return (
    <FadeIn delay={delay} className="h-full min-h-0">
      {spotlight ? (
        <SpotlightCard className="h-full min-h-0 rounded-2xl border-0 bg-transparent p-0 shadow-none">
          {shell}
        </SpotlightCard>
      ) : (
        shell
      )}
    </FadeIn>
  );
}
