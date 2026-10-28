"use client";

import { create } from "zustand";
import { api } from "@/lib/api";
import { isDevAuthBypass } from "@/lib/dev-auth";

export interface AuthUser {
  id: string;
  username: string;
  email: string;
  is_admin: boolean;
  hf_token?: string;
  created_at: string;
}

interface AuthStore {
  user: AuthUser | null;
  token: string | null;
  loading: boolean;
  initialized: boolean;
  login: (email: string, password: string) => Promise<void>;
  loginWithGoogle: (idToken: string) => Promise<void>;
  register: (
    username: string,
    email: string,
    password: string
  ) => Promise<{ message: string; email: string; verification_url?: string | null }>;
  logout: () => void;
  initializeAuth: () => Promise<void>;
  syncTokensRefreshed: (detail?: { accessToken?: string; user?: AuthUser | null }) => void;
  syncLoggedOut: () => void;
  setHfToken: (hfToken: string) => Promise<void>;
}

const getStoredToken = () =>
  typeof window !== "undefined" ? localStorage.getItem("token") : null;

const clearStoredTokens = () => {
  if (typeof window === "undefined") return;
  localStorage.removeItem("token");
  localStorage.removeItem("refresh_token");
};

export const useAuthStore = create<AuthStore>((set, get) => ({
  // Keep SSR/client first paint identical — read localStorage only in initializeAuth
  user: null,
  token: null,
  loading: true,
  initialized: false,

  async login(email, password) {
    const data = await api.post<{ access_token: string; refresh_token: string; user: AuthUser }>(
      "/api/v1/auth/login",
      { email, password }
    );

    localStorage.setItem("token", data.access_token);
    localStorage.setItem("refresh_token", data.refresh_token);
    set({
      token: data.access_token,
      user: data.user,
      loading: false,
      initialized: true,
    });
  },

  async loginWithGoogle(idToken) {
    const data = await api.post<{ access_token: string; refresh_token: string; user: AuthUser }>(
      "/api/v1/auth/google",
      { id_token: idToken }
    );

    localStorage.setItem("token", data.access_token);
    localStorage.setItem("refresh_token", data.refresh_token);
    set({
      token: data.access_token,
      user: data.user,
      loading: false,
      initialized: true,
    });
  },

  async register(username, email, password) {
    const data = await api.post<{ message: string; email: string; verification_url?: string | null }>(
      "/api/v1/auth/register",
      { username, email, password }
    );

    clearStoredTokens();
    set({
      token: null,
      user: null,
      loading: false,
      initialized: true,
    });
    return data;
  },

  async logout() {
    try {
      await api.post("/api/v1/auth/logout");
    } catch {
      // Ignore network errors on logout
    }
    clearStoredTokens();
    set({
      token: null,
      user: null,
      loading: false,
      initialized: true,
    });
  },

  async initializeAuth() {
    const { initialized, token } = get();
    if (initialized) return;

    // Never leave the dashboard on an infinite loading shell.
    const failSafe =
      typeof window !== "undefined"
        ? window.setTimeout(() => {
            if (!get().initialized) {
              set({ loading: false, initialized: true });
            }
          }, 12000)
        : null;

    try {
      let storedToken = token ?? getStoredToken();

      // 1. Try to extract tokens from URL search params (OAuth redirect)
      if (typeof window !== "undefined") {
        const params = new URLSearchParams(window.location.search);
        const urlToken = params.get("token");
        const urlRefreshToken = params.get("refresh_token");
        if (urlToken && urlRefreshToken) {
          localStorage.setItem("token", urlToken);
          localStorage.setItem("refresh_token", urlRefreshToken);
          storedToken = urlToken;
          // Clean URL parameters without reloading
          const cleanUrl = window.location.origin + window.location.pathname;
          window.history.replaceState({}, document.title, cleanUrl);
        }
      }

      const bypassEnabled = isDevAuthBypass();

      // 2. Local/dev auto sign-in — skip the login form when no token is stored
      if (!storedToken && bypassEnabled) {
        set({ loading: true });
        try {
          const data = await api.post<{
            access_token: string;
            refresh_token: string;
            user: AuthUser;
          }>("/api/v1/auth/dev-login", {});
          localStorage.setItem("token", data.access_token);
          localStorage.setItem("refresh_token", data.refresh_token);
          set({
            token: data.access_token,
            user: data.user,
            loading: false,
            initialized: true,
          });
          return;
        } catch {
          // Fall through to anonymous state if backend bypass is off
          clearStoredTokens();
          set({ user: null, token: null, loading: false, initialized: true });
          return;
        }
      }

      // 3. If no token, don't request /me (prevents CORS preflight credentials error for anonymous users)
      if (!storedToken) {
        set({ user: null, token: null, loading: false, initialized: true });
        return;
      }

      set({ loading: true });

      try {
        const user = await api.get<AuthUser>(
          "/api/v1/auth/me",
          { token: storedToken }
        );
        set({
          user,
          token: storedToken,
          loading: false,
          initialized: true,
        });
      } catch {
        // Stale token: try local bypass once before giving up
        if (bypassEnabled) {
          try {
            clearStoredTokens();
            const data = await api.post<{
              access_token: string;
              refresh_token: string;
              user: AuthUser;
            }>("/api/v1/auth/dev-login", {});
            localStorage.setItem("token", data.access_token);
            localStorage.setItem("refresh_token", data.refresh_token);
            set({
              token: data.access_token,
              user: data.user,
              loading: false,
              initialized: true,
            });
            return;
          } catch {
            // ignore and clear below
          }
        }
        clearStoredTokens();
        set({ user: null, token: null, loading: false, initialized: true });
      }
    } finally {
      if (failSafe != null) window.clearTimeout(failSafe);
    }
  },

  syncTokensRefreshed(detail) {
    if (!detail) return;

    set((state) => ({
      token: detail.accessToken ?? state.token,
      user: detail.user ?? state.user,
      loading: false,
      initialized: true,
    }));
  },

  syncLoggedOut() {
    set({
      token: null,
      user: null,
      loading: false,
      initialized: true,
    });
  },

  async setHfToken(hfToken: string) {
    const response = await api.put<AuthUser>("/api/v1/auth/hf-token", { hf_token: hfToken });
    set({ user: response });
  },
}));
