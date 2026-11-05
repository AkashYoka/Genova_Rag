"use client";

import {
  type ButtonHTMLAttributes,
  type CSSProperties,
  type ReactNode,
} from "react";
import { Menu, X } from "lucide-react";
import "./ledger.css";

export type LedgerNavItem = {
  id: string;
  label: string;
  icon?: ReactNode;
};

export function LedgerShell({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return <div className={`genova-ledger ${className}`.trim()}>{children}</div>;
}

export function LedgerSidebar({
  brand,
  scope,
  groupLabel,
  items,
  activeId,
  onSelect,
  footer,
}: {
  brand: string;
  scope: string;
  groupLabel?: string;
  items: LedgerNavItem[];
  activeId: string;
  onSelect: (id: string) => void;
  footer?: ReactNode;
}) {
  return (
    <aside className="ledger-side" aria-label="Section navigation">
      <div className="ledger-brand">
        <div className="ledger-brand-mark" aria-hidden>
          G
        </div>
        <div>
          <div className="ledger-brand-name">{brand}</div>
          <div className="ledger-brand-scope">{scope}</div>
        </div>
      </div>
      {groupLabel ? <div className="ledger-nav-group">{groupLabel}</div> : null}
      {items.map((item) => (
        <button
          key={item.id}
          type="button"
          className="ledger-nav-item"
          data-active={activeId === item.id}
          onClick={() => onSelect(item.id)}
        >
          {item.icon}
          <span>{item.label}</span>
        </button>
      ))}
      {footer ? <div className="ledger-side-footer">{footer}</div> : null}
    </aside>
  );
}

export function LedgerMobileBar({
  title,
  open,
  onToggle,
  action,
}: {
  title: string;
  open: boolean;
  onToggle: () => void;
  action?: ReactNode;
}) {
  return (
    <div className="ledger-mobile-bar">
      <button
        type="button"
        className="ledger-btn ledger-btn-ghost"
        onClick={onToggle}
        aria-label={open ? "Close menu" : "Open menu"}
      >
        {open ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
        <span className="font-medium text-[var(--gd-text)]">{title}</span>
      </button>
      {action}
    </div>
  );
}

export function LedgerSheet({
  open,
  onClose,
  children,
}: {
  open: boolean;
  onClose: () => void;
  children: ReactNode;
}) {
  if (!open) return null;
  return (
    <>
      <button
        type="button"
        className="ledger-sheet-backdrop"
        aria-label="Close navigation"
        onClick={onClose}
      />
      <div className="ledger-sheet" role="dialog" aria-modal="true">
        {children}
      </div>
    </>
  );
}

export function LedgerCanvas({ children }: { children: ReactNode }) {
  return <div className="ledger-canvas">{children}</div>;
}

export function LedgerScroll({ children }: { children: ReactNode }) {
  return (
    <div className="ledger-scroll">
      <div className="ledger-frame">{children}</div>
    </div>
  );
}

export function LedgerTitle({
  title,
  finding,
  actions,
}: {
  title: string;
  finding: string;
  actions?: ReactNode;
}) {
  return (
    <div className="ledger-title-row">
      <div>
        <h1 className="ledger-h1">{title}</h1>
        <p className="ledger-finding">{finding}</p>
      </div>
      {actions ? <div className="ledger-actions">{actions}</div> : null}
    </div>
  );
}

export function LedgerButton({
  variant = "secondary",
  className = "",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost";
}) {
  return (
    <button
      type="button"
      className={`ledger-btn ledger-btn-${variant} ${className}`.trim()}
      {...props}
    />
  );
}

export function LedgerKpiStrip({
  items,
}: {
  items: Array<{
    label: string;
    value: string;
    caption: string;
    emphasis?: "data" | "money";
  }>;
}) {
  return (
    <div className="ledger-kpi-strip" role="group" aria-label="Key metrics">
      {items.map((item) => (
        <div key={item.label} className="ledger-kpi">
          <p className="ledger-kpi-label">{item.label}</p>
          <p className="ledger-kpi-value" data-emphasis={item.emphasis}>
            {item.value}
          </p>
          <p className="ledger-kpi-caption">{item.caption}</p>
        </div>
      ))}
    </div>
  );
}

export function LedgerPanel({
  title,
  description,
  action,
  children,
  padded = false,
  className = "",
}: {
  title: string;
  description?: string;
  action?: ReactNode;
  children: ReactNode;
  padded?: boolean;
  className?: string;
}) {
  return (
    <section className={`ledger-panel ${className}`.trim()}>
      <div className="ledger-panel-head">
        <div>
          <h2 className="ledger-panel-title">{title}</h2>
          {description ? (
            <p className="ledger-panel-desc">{description}</p>
          ) : null}
        </div>
        {action}
      </div>
      <div className={padded ? "ledger-panel-pad" : "ledger-panel-body"}>
        {children}
      </div>
    </section>
  );
}

export function LedgerScrollTable({
  children,
  compact = false,
  footer,
}: {
  children: ReactNode;
  compact?: boolean;
  footer?: ReactNode;
}) {
  return (
    <>
      <div
        className={`ledger-table-scroll${compact ? " ledger-table-scroll-sm" : ""}`}
      >
        {children}
      </div>
      {footer}
    </>
  );
}

export function LedgerStatus({
  label,
  tone = "neutral",
}: {
  label: string;
  tone?: "ok" | "bad" | "warn" | "neutral";
}) {
  return (
    <span className="ledger-status" data-tone={tone === "neutral" ? undefined : tone}>
      {label}
    </span>
  );
}

export function LedgerBarChart({
  points,
  valueKey = "total_tokens",
}: {
  points: Array<{ date: string; total_tokens: number; estimated_cost_usd?: number }>;
  valueKey?: "total_tokens" | "estimated_cost_usd";
}) {
  const chartPoints = points.length > 45 ? points.slice(-45) : points;
  const max = Math.max(
    1,
    ...chartPoints.map((p) => Number(p[valueKey] ?? 0)),
  );

  if (!chartPoints.length) {
    return <div className="ledger-empty">No usage in this range.</div>;
  }

  return (
    <div className="ledger-panel-pad">
      <div
        className="ledger-bars"
        role="img"
        aria-label="Daily usage chart"
      >
        {chartPoints.map((point) => {
          const value = Number(point[valueKey] ?? 0);
          const heightPct =
            value <= 0 ? 0 : Math.max(3, Math.round((value / max) * 100));
          return (
            <div
              key={point.date}
              className="ledger-bar-col"
              title={`${point.date}: ${value.toLocaleString()}`}
            >
              <div
                className="ledger-bar"
                style={{ height: `${heightPct}%` } as CSSProperties}
              />
            </div>
          );
        })}
      </div>
      <div className="ledger-bar-meta">
        <span>{chartPoints[0]?.date ?? "—"}</span>
        <span>Daily {valueKey === "total_tokens" ? "tokens" : "cost"} (UTC)</span>
        <span>{chartPoints[chartPoints.length - 1]?.date ?? "—"}</span>
      </div>
    </div>
  );
}

export function LedgerEmpty({ children }: { children: ReactNode }) {
  return <div className="ledger-empty">{children}</div>;
}

export function LedgerError({ children }: { children: ReactNode }) {
  return (
    <div role="alert" className="ledger-error">
      {children}
    </div>
  );
}

export function SettingsSection({
  title,
  description,
  summary,
  children,
}: {
  title: string;
  description: string;
  summary?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="ledger-settings-section">
      <div className="ledger-settings-label">
        <h2>{title}</h2>
        <p>{description}</p>
        {summary}
      </div>
      <div className="ledger-settings-panel">{children}</div>
    </section>
  );
}
