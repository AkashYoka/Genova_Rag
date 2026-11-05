"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { isAdminUser } from "@/lib/admin";

interface RequireAuthProps {
  children: React.ReactNode;
  /** When true, non-admin authenticated users are sent to /dashboard */
  requireAdmin?: boolean;
}

/**
 * Blocks protected UI until auth has initialized, then redirects unauthenticated
 * users to /login. Prevents protected content from flashing before verification.
 */
export function RequireAuth({ children, requireAdmin = false }: RequireAuthProps) {
  const { user, initialized } = useAuth();
  const router = useRouter();
  const adminOk = !requireAdmin || isAdminUser(user);

  useEffect(() => {
    if (!initialized) return;
    if (!user) {
      router.replace("/login");
      return;
    }
    if (requireAdmin && !isAdminUser(user)) {
      router.replace("/dashboard");
    }
  }, [initialized, user, requireAdmin, router]);

  if (!initialized || !user || !adminOk) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <div
          className="h-8 w-8 animate-pulse rounded-full bg-muted"
          aria-label="Verifying authentication"
        />
      </div>
    );
  }

  return <>{children}</>;
}
