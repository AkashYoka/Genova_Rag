/**
 * Local-only auth switch.
 * - ON  (default): auto sign-in via /api/v1/auth/dev-login — no login page
 * - OFF: set NEXT_PUBLIC_DEV_AUTH_BYPASS=false in frontend/.env.local
 *
 * Backend must also allow it (DEV_AUTH_BYPASS=true, ENVIRONMENT != production).
 */
export function isDevAuthBypass(): boolean {
  if (process.env.NEXT_PUBLIC_DEV_AUTH_BYPASS === "false") return false;
  if (typeof window === "undefined") return true;
  return ["localhost", "127.0.0.1"].includes(window.location.hostname);
}
