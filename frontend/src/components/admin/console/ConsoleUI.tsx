"use client";

import {
  type ButtonHTMLAttributes,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
  useEffect,
  useId,
  useRef,
} from "react";
import { ChevronLeft, ChevronRight, Inbox, Info, Lock, Search, X } from "lucide-react";
import "./console.css";

/* ------------------------------------------------------------------ utils */

function cx(...parts: Array<string | false | null | undefined>) {
  return parts.filter(Boolean).join(" ");
}

export type Tone = "neutral" | "ok" | "warn" | "danger" | "accent";

/* ----------------------------------------------------------------- button */

export function AdminButton({
  variant = "secondary",
  size = "md",
  className,
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost" | "danger";
  size?: "md" | "sm";
}) {
  return (
    <button
      type="button"
      className={cx("adm-btn", `adm-btn--${variant}`, size === "sm" && "adm-btn--sm", className)}
      {...props}
    >
      {children}
    </button>
  );
}

export function IconButton({
  label,
  className,
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { label: string; children: ReactNode }) {
  return (
    <button
      type="button"
      className={cx("adm-iconbtn", className)}
      aria-label={label}
      title={label}
      {...props}
    >
      {children}
    </button>
  );
}

/* ------------------------------------------------------------------ cards */

export function Card({
  title,
  description,
  action,
  children,
  padded = false,
  className,
  footer,
}: {
  title?: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  padded?: boolean;
  className?: string;
  footer?: ReactNode;
}) {
  return (
    <section className={cx("adm-card", className)}>
      {title || action ? (
        <div className="adm-card__head">
          <div>
            {title ? <h2 className="adm-card__title">{title}</h2> : null}
            {description ? <p className="adm-card__desc">{description}</p> : null}
          </div>
          {action}
        </div>
      ) : null}
      <div className={cx("adm-card__body", padded && "adm-card__body--pad")}>{children}</div>
      {footer ? <div className="adm-card__foot">{footer}</div> : null}
    </section>
  );
}

export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <header className="adm-page-head">
      <div>
        <h1 className="adm-page-title">{title}</h1>
        <p className="adm-page-sub">{description}</p>
      </div>
      {actions ? <div className="adm-page-actions">{actions}</div> : null}
    </header>
  );
}

/* ------------------------------------------------------------------ stats */

export function StatGrid({ children }: { children: ReactNode }) {
  return <div className="adm-stats">{children}</div>;
}

export function Stat({
  label,
  value,
  caption,
  icon,
  attention = false,
  suffix,
}: {
  label: string;
  value: ReactNode;
  caption?: ReactNode;
  icon?: ReactNode;
  attention?: boolean;
  suffix?: ReactNode;
}) {
  return (
    <div className={cx("adm-stat", attention && "adm-stat--attention")}>
      <p className="adm-stat__label">
        {icon}
        {label}
      </p>
      <p className="adm-stat__value">
        {value}
        {suffix ? <small> {suffix}</small> : null}
      </p>
      {caption ? <p className="adm-stat__caption">{caption}</p> : null}
    </div>
  );
}

/* ------------------------------------------------------------------ badge */

export function Badge({
  children,
  tone = "neutral",
  plain = false,
}: {
  children: ReactNode;
  tone?: Tone;
  plain?: boolean;
}) {
  return (
    <span
      className={cx("adm-badge", plain && "adm-badge--plain")}
      data-tone={tone === "neutral" ? undefined : tone}
    >
      {children}
    </span>
  );
}

/* ----------------------------------------------------------------- states */

export function EmptyState({
  title,
  text,
  action,
  icon,
}: {
  title: string;
  text?: ReactNode;
  action?: ReactNode;
  icon?: ReactNode;
}) {
  return (
    <div className="adm-empty">
      <div className="adm-empty__icon" aria-hidden>
        {icon ?? <Inbox />}
      </div>
      <p className="adm-empty__title">{title}</p>
      {text ? <p className="adm-empty__text">{text}</p> : null}
      {action ? <div className="adm-empty__action">{action}</div> : null}
    </div>
  );
}

export function Callout({
  children,
  tone = "neutral",
  icon,
}: {
  children: ReactNode;
  tone?: "neutral" | "warn" | "danger";
  icon?: ReactNode;
}) {
  return (
    <div className="adm-callout" data-tone={tone === "neutral" ? undefined : tone}>
      {icon ?? <Info />}
      <div>{children}</div>
    </div>
  );
}

/* ------------------------------------------------------------------ form */

export function Field({
  label,
  hint,
  optional,
  htmlFor,
  children,
  className,
}: {
  label: ReactNode;
  hint?: ReactNode;
  optional?: boolean;
  htmlFor?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={cx("adm-field", className)}>
      <label className="adm-label" htmlFor={htmlFor}>
        {label}
        {optional ? <span className="adm-label__optional">Optional</span> : null}
      </label>
      {children}
      {hint ? <span className="adm-hint">{hint}</span> : null}
    </div>
  );
}

export function TextInput({ className, ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={cx("adm-input", className)} {...props} />;
}

export function SelectInput({
  className,
  children,
  ...props
}: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select className={cx("adm-select", className)} {...props}>
      {children}
    </select>
  );
}

export function SearchInput({
  value,
  onChange,
  placeholder,
  className,
  label,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
  className?: string;
  label: string;
}) {
  return (
    <div className={cx("adm-input-icon", "adm-search", className)}>
      <Search aria-hidden />
      <input
        className="adm-input"
        type="search"
        value={value}
        aria-label={label}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  );
}

export function Tooltip({ text }: { text: string }) {
  return (
    <span className="adm-tip">
      <button type="button" className="adm-tip__trigger" aria-label={text}>
        ?
      </button>
      <span className="adm-tip__bubble" role="tooltip">
        {text}
      </span>
    </span>
  );
}

/* ------------------------------------------------------------------ table */

export function TableWrap({
  children,
  height = "tall",
}: {
  children: ReactNode;
  height?: "tall" | "short" | "auto";
}) {
  return (
    <div
      className={cx(
        "adm-table-wrap",
        height === "tall" && "adm-table-wrap--tall",
        height === "short" && "adm-table-wrap--short",
      )}
    >
      {children}
    </div>
  );
}

export function Pager({
  page,
  pages,
  total,
  onPage,
  noun = "rows",
  nounOne,
}: {
  page: number;
  pages: number;
  total: number;
  onPage: (page: number) => void;
  noun?: string;
  nounOne?: string;
}) {
  const nounLabel = total === 1 && nounOne ? nounOne : noun;
  return (
    <div className="adm-pager">
      <span className="adm-pager__label">
        {total.toLocaleString()} {nounLabel} · page {page} of {Math.max(1, pages)}
      </span>
      <AdminButton
        size="sm"
        variant="secondary"
        aria-label="Previous page"
        disabled={page <= 1}
        onClick={() => onPage(page - 1)}
      >
        <ChevronLeft />
      </AdminButton>
      <AdminButton
        size="sm"
        variant="secondary"
        aria-label="Next page"
        disabled={page >= pages}
        onClick={() => onPage(page + 1)}
      >
        <ChevronRight />
      </AdminButton>
    </div>
  );
}

/* ------------------------------------------------------------------ chart */

export function BarChart({
  points,
  valueKey,
  formatValue,
  ariaLabel,
}: {
  points: Array<{ date: string; total_tokens: number; estimated_cost_usd?: number }>;
  valueKey: "total_tokens" | "estimated_cost_usd";
  formatValue: (value: number) => string;
  ariaLabel: string;
}) {
  const shown = points.length > 60 ? points.slice(-60) : points;
  const max = Math.max(1, ...shown.map((p) => Number(p[valueKey] ?? 0)));

  if (!shown.length) {
    return <EmptyState title="No usage in this period" text="Usage appears here once people ask questions." />;
  }

  return (
    <div className="adm-chart">
      <div className="adm-chart__plot" role="img" aria-label={ariaLabel}>
        {shown.map((point) => {
          const value = Number(point[valueKey] ?? 0);
          const height = value <= 0 ? 0 : Math.max(3, Math.round((value / max) * 100));
          return (
            <div
              key={point.date}
              className="adm-chart__col"
              data-zero={value <= 0}
              data-date-label={`${point.date} · ${formatValue(value)}`}
            >
              <div className="adm-chart__bar" style={{ height: `${height}%` }} />
            </div>
          );
        })}
      </div>
      <div className="adm-chart__axis">
        <span>{shown[0]?.date}</span>
        <span>Peak {formatValue(max)}</span>
        <span>{shown[shown.length - 1]?.date}</span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ modal */

export function Modal({
  open,
  onClose,
  title,
  description,
  children,
  footer,
  wide = false,
  closeLabel = "Close dialog",
}: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  description?: ReactNode;
  children?: ReactNode;
  footer?: ReactNode;
  wide?: boolean;
  closeLabel?: string;
}) {
  const titleId = useId();
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    panelRef.current?.focus();
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = previous;
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="adm-backdrop adm-scope"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        ref={panelRef}
        className={cx("adm-modal", wide && "adm-modal--wide")}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
      >
        <div className="adm-modal__head">
          <div>
            <h2 className="adm-modal__title" id={titleId}>
              {title}
            </h2>
            {description ? <p className="adm-modal__desc">{description}</p> : null}
          </div>
          <IconButton label={closeLabel} onClick={onClose}>
            <X />
          </IconButton>
        </div>
        {children ? <div className="adm-modal__body">{children}</div> : null}
        {footer ? <div className="adm-modal__foot">{footer}</div> : null}
      </div>
    </div>
  );
}

export function ConfirmDialog({
  open,
  onClose,
  onConfirm,
  title,
  description,
  confirmLabel,
  destructive = false,
  busy = false,
  children,
}: {
  open: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  description: ReactNode;
  confirmLabel: string;
  destructive?: boolean;
  busy?: boolean;
  children?: ReactNode;
}) {
  return (
    <Modal
      open={open}
      onClose={onClose}
      title={title}
      description={description}
      footer={
        <>
          <AdminButton variant="ghost" onClick={onClose} disabled={busy}>
            Cancel
          </AdminButton>
          <AdminButton
            variant={destructive ? "danger" : "primary"}
            onClick={onConfirm}
            disabled={busy}
          >
            {busy ? "Working…" : confirmLabel}
          </AdminButton>
        </>
      }
    >
      {children}
    </Modal>
  );
}

/* ----------------------------------------------------------------- drawer */

export function Drawer({
  open,
  onClose,
  title,
  description,
  children,
  footer,
  labelledBy,
  wide = false,
}: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  description?: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
  labelledBy?: string;
  wide?: boolean;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = previous;
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="adm-drawer-backdrop adm-scope"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className={cx("adm-drawer", wide && "adm-drawer--wide")}
        role="dialog"
        aria-modal="true"
        aria-labelledby={labelledBy}
      >
        <div className="adm-drawer__head">
          <div>
            <h2 className="adm-card__title">{title}</h2>
            {description ? <p className="adm-card__desc">{description}</p> : null}
          </div>
          <IconButton label="Close panel" onClick={onClose}>
            <X />
          </IconButton>
        </div>
        <div className="adm-drawer__body">{children}</div>
        {footer ? <div className="adm-drawer__foot">{footer}</div> : null}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------- misc bits */

export function SkeletonBlock({
  height = 14,
  width = "100%",
  className,
}: {
  height?: number;
  width?: number | string;
  className?: string;
}) {
  return (
    <span
      className={cx("adm-skel", className)}
      style={{ height, width }}
      aria-hidden
    />
  );
}

export function TableSkeleton({ rows = 5 }: { rows?: number }) {
  return (
    <div className="adm-card__body--pad" style={{ display: "grid", gap: 10 }}>
      {Array.from({ length: rows }).map((_, i) => (
        <SkeletonBlock key={i} height={18} />
      ))}
    </div>
  );
}

export function InlineNote({ children }: { children: ReactNode }) {
  return (
    <p className="adm-hint" style={{ display: "flex", alignItems: "center", gap: 5 }}>
      <Lock style={{ width: 12, height: 12 }} aria-hidden />
      {children}
    </p>
  );
}

export function DefinitionList({
  items,
}: {
  items: Array<{ term: string; value: ReactNode }>;
}) {
  return (
    <dl className="adm-kv">
      {items.map((item) => (
        <div key={item.term} style={{ display: "contents" }}>
          <dt>{item.term}</dt>
          <dd>{item.value}</dd>
        </div>
      ))}
    </dl>
  );
}
