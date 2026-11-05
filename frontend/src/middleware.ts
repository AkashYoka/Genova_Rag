import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

/**
 * Soft client-session gate for protected routes.
 *
 * Checks the `genova_auth` marker cookie (set on login / successful /me).
 * This does not validate JWTs — API `get_current_user` and client `RequireAuth`
 * remain the authoritative checks. The cookie only prevents unauthenticated
 * browsers from painting protected page shells via direct URL entry.
 *
 * Intentionally does NOT bounce logged-in users away from /login|/register —
 * that would risk redirect loops when the marker cookie is stale.
 */

const PROTECTED_PREFIXES = ["/dashboard", "/documents", "/admin"];

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  const hasSession = Boolean(request.cookies.get("genova_auth")?.value);

  const isProtected = PROTECTED_PREFIXES.some(
    (prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`),
  );

  if (isProtected && !hasSession) {
    const loginUrl = request.nextUrl.clone();
    loginUrl.pathname = "/login";
    loginUrl.searchParams.set("next", pathname);
    return NextResponse.redirect(loginUrl);
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/dashboard/:path*", "/documents/:path*", "/admin/:path*"],
};
