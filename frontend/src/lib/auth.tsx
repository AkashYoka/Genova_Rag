"use client";

import React, { useEffect } from "react";
import { useAuthStore } from "@/store/auth-store";

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const initializeAuth = useAuthStore((state) => state.initializeAuth);
  const syncTokensRefreshed = useAuthStore((state) => state.syncTokensRefreshed);
  const syncLoggedOut = useAuthStore((state) => state.syncLoggedOut);
  const syncFromStorage = useAuthStore((state) => state.syncFromStorage);

  useEffect(() => {
    void initializeAuth();
  }, [initializeAuth]);

  useEffect(() => {
    const handleTokensRefreshed = (e: Event) => {
      syncTokensRefreshed((e as CustomEvent).detail);
    };

    const handleLoggedOut = () => {
      syncLoggedOut();
    };

    /** Cross-tab: login/logout/refresh in another tab updates localStorage. */
    const handleStorage = (e: StorageEvent) => {
      if (e.key !== "token" && e.key !== "refresh_token" && e.key !== "auth_user") {
        return;
      }
      syncFromStorage();
    };

    window.addEventListener("auth:tokens-refreshed", handleTokensRefreshed);
    window.addEventListener("auth:logged-out", handleLoggedOut);
    window.addEventListener("storage", handleStorage);

    return () => {
      window.removeEventListener("auth:tokens-refreshed", handleTokensRefreshed);
      window.removeEventListener("auth:logged-out", handleLoggedOut);
      window.removeEventListener("storage", handleStorage);
    };
  }, [syncLoggedOut, syncTokensRefreshed, syncFromStorage]);

  return <>{children}</>;
}

export function useAuth() {
  return useAuthStore();
}
