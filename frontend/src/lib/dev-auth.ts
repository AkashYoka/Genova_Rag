/**
 * Local-only auth switch for calling `/api/v1/auth/dev-login`.
 *
 * Default: OFF (deny). Enable only with an explicit opt-in:
 *   NEXT_PUBLIC_DEV_AUTH_BYPASS=true
 * and only on localhost / 127.0.0.1. Production builds never enable this.
 *
 * Backend must also allow it (`DEV_AUTH_BYPASS=true`, `ENVIRONMENT != production`).
 */
export function isDevAuthBypass(): boolean {
  if (process.env.NODE_ENV === "production") return false;
  if (process.env.NEXT_PUBLIC_DEV_AUTH_BYPASS !== "true") return false;
  if (typeof window === "undefined") return false;
  return ["localhost", "127.0.0.1"].includes(window.location.hostname);
}
