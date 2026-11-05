/** Shared admin helpers — never store secrets here. */

export function isAdminUser(user: {
  is_admin?: boolean;
  role?: string | null;
} | null | undefined): boolean {
  if (!user) return false;
  if (user.is_admin) return true;
  return String(user.role || "").toLowerCase() === "admin";
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const index = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1,
  );
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

export function formatTime(milliseconds: number): string {
  if (milliseconds >= 1000) return `${(milliseconds / 1000).toFixed(2)} s`;
  return `${Math.round(milliseconds)} ms`;
}

export function formatUsd(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  if (value === 0) return "$0.00";
  if (Math.abs(value) < 0.0001) return `$${value.toFixed(6)}`;
  if (Math.abs(value) < 0.01) return `$${value.toFixed(4)}`;
  return `$${value.toFixed(2)}`;
}
