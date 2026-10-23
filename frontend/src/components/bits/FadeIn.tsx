"use client";

import { cn } from "@/lib/utils";
import type { HTMLAttributes, ReactNode } from "react";

/** React Bits–style entrance fade. */
export function FadeIn({
  children,
  className,
  delay = 0,
  ...props
}: HTMLAttributes<HTMLDivElement> & {
  children: ReactNode;
  delay?: number;
  y?: number;
}) {
  return (
    <div
      className={cn("animate-fade-in-up", className)}
      style={{ animationDelay: `${delay}s`, animationFillMode: "both" }}
      {...props}
    >
      {children}
    </div>
  );
}
