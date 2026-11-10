"use client";

import type { ReactNode } from "react";

type IntegrationProviderCardProps = {
  name: string;
  description: string;
  icon: ReactNode;
  status?: ReactNode;
  children: ReactNode;
};

/** Quiet settings card for a single document source. */
export function IntegrationProviderCard({
  name,
  description,
  icon,
  status,
  children,
}: IntegrationProviderCardProps) {
  return (
    <section className="flex h-full flex-col rounded-[12px] border border-[var(--gd-border)] bg-[var(--gd-surface)]">
      <header className="flex items-start gap-3 border-b border-[var(--gd-rule)] px-5 py-4">
        <div
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-[10px] bg-[var(--gd-raised)] text-[var(--gd-text)]"
          aria-hidden
        >
          {icon}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <h3 className="text-[15px] font-medium tracking-tight text-[var(--gd-text)]">
              {name}
            </h3>
            {status}
          </div>
          <p className="mt-1 text-[13px] leading-snug text-[var(--gd-muted)]">
            {description}
          </p>
        </div>
      </header>
      <div className="flex flex-1 flex-col gap-5 px-5 py-5">{children}</div>
    </section>
  );
}

export function IntegrationMethod({
  label,
  hint,
  children,
}: {
  label: string;
  hint: string;
  children: ReactNode;
}) {
  return (
    <div className="space-y-2.5">
      <div>
        <p className="text-[13px] font-medium text-[var(--gd-text)]">{label}</p>
        <p className="mt-0.5 text-[12px] leading-snug text-[var(--gd-muted)]">
          {hint}
        </p>
      </div>
      {children}
    </div>
  );
}

export function IntegrationStatusPill({
  tone,
  children,
}: {
  tone: "idle" | "ok" | "busy" | "warn" | "error";
  children: ReactNode;
}) {
  const className =
    tone === "ok"
      ? "bg-[color-mix(in_oklab,var(--gd-positive)_14%,transparent)] text-[var(--gd-positive)]"
      : tone === "busy"
        ? "bg-[var(--gd-accent-soft)] text-[var(--gd-accent)]"
        : tone === "warn"
          ? "bg-[color-mix(in_oklab,var(--gd-warning)_14%,transparent)] text-[var(--gd-warning)]"
          : tone === "error"
            ? "bg-[color-mix(in_oklab,var(--gd-negative)_14%,transparent)] text-[var(--gd-negative)]"
            : "bg-[var(--gd-raised)] text-[var(--gd-muted)]";

  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium ${className}`}
    >
      {children}
    </span>
  );
}

export function GoogleDriveMark({ className = "h-5 w-5" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} aria-hidden>
      <path fill="#4285F4" d="M12.01 2.5 4.52 15.52h4.98L17 2.5z" />
      <path fill="#34A853" d="m8.5 21.5 3.5-6H2.52L6 21.5z" />
      <path fill="#FBBC04" d="M22 15.52H12.52l-3.5 6H18.5z" />
    </svg>
  );
}

export function DropboxMark({ className = "h-5 w-5" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} aria-hidden>
      <path
        fill="#0061FF"
        d="M6 2.5 1 5.75 6 9l5-3.25L6 2.5zm12 0L13 5.75 18 9l5-3.25L18 2.5zM1 12.25 6 15.5l5-3.25L6 9l-5 3.25zm17 0L13 9l-5 3.25 5 3.25 5-3.25zM6 17.1l5 3.25L16 17.1l-5-3.25L6 17.1z"
      />
    </svg>
  );
}
