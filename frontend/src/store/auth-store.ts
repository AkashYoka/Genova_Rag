"use client";

import { create } from "zustand";
import { api, CONNECTION_ERROR_MESSAGE, setAuthMarkerCookie } from "@/lib/api";

export interface AuthUser {
  id: string;
  username: string;
  email: string;
  is_admin: boolean;
  role?: string;
  is_verified?: boolean;
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
  logout: () => Promise<void>;
  initializeAuth: () => Promise<void>;
  syncTokensRefreshed: (detail?: { accessToken?: string; user?: AuthUser | null }) => void;
  syncLoggedOut: () => void;
  syncFromStorage: () => void;
}

const getStoredToken = () =>
  typeof window !== "undefined" ? localStorage.getItem("token") : null;

const getCachedUser = (): AuthUser | null => {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem("auth_user");
    if (!raw) return null;
    return JSON.parse(raw) as AuthUser;
  } catch {
    return null;
  }
};

const cacheUser = (user: AuthUser | null) => {
  if (typeof window === "undefined") return;
  if (user) {
    localStorage.setItem("auth_user", JSON.stringify(user));
  } else {
    localStorage.removeItem("auth_user");
  }
};

const clearStoredTokens = () => {
  if (typeof window === "undefined") return;
  localStorage.removeItem("token");
  localStorage.removeItem("refresh_token");
  localStorage.removeItem("auth_user");
  setAuthMarkerCookie(false);
};

const persistTokens = (_accessToken: string, _refreshToken: string, user: AuthUser) => {
  // JWTs live in HttpOnly cookies set by the API — never in localStorage.
  localStorage.removeItem("token");
  localStorage.removeItem("refresh_token");
  cacheUser(user);
  setAuthMarkerCookie(true);
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

    persistTokens(data.access_token, data.refresh_token, data.user);
    set({
      token: "cookie",
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

    persistTokens(data.access_token, data.refresh_token, data.user);
    set({
      token: "cookie",
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
      // Cookie session: server reads HttpOnly refresh_token; body optional.
      await api.post("/api/v1/auth/logout", {});
    } catch {
      // Ignore network errors on logout — local session must still clear
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

    const failSafe =
      typeof window !== "undefined"
        ? window.setTimeout(() => {
            if (!get().initialized) {
              // Prefer cached session over anonymous if tokens still exist
              const cached = getCachedUser();
              const stillToken = getStoredToken();
              if (stillToken && cached) {
                set({
                  user: cached,
                  token: stillToken,
                  loading: false,
                  initialized: true,
                });
                setAuthMarkerCookie(true);
              } else {
                set({ loading: false, initialized: true });
              }
            }
          }, 12000)
        : null;

    try {
      // Never accept access/refresh JWTs from the query string — they leak via
      // history, Referer, and proxy logs. Strip them if present without storing.
      if (typeof window !== "undefined") {
        const params = new URLSearchParams(window.location.search);
        if (params.has("token") || params.has("refresh_token") || params.has("access_token")) {
          params.delete("token");
          params.delete("refresh_token");
          params.delete("access_token");
          const clean = params.toString();
          const nextUrl =
            window.location.pathname + (clean ? `?${clean}` : "") + window.location.hash;
          window.history.replaceState({}, document.title, nextUrl);
        }
      }

      // Prefer HttpOnly cookie session; fall back to legacy localStorage only
      // long enough to call /me once, then purge JWTs from JS storage.
      const legacyToken = getStoredToken();
      const hasMarker =
        typeof document !== "undefined" && document.cookie.includes(`${"genova_auth"}=`);

      if (!legacyToken && !hasMarker && !token) {
        // Still attempt /me — cookies may exist without the soft marker.
        set({ loading: true });
        try {
          const user = await api.get<AuthUser>("/api/v1/auth/me");
          cacheUser(user);
          setAuthMarkerCookie(true);
          set({ user, token: "cookie", loading: false, initialized: true });
          return;
        } catch {
          clearStoredTokens();
          set({ user: null, token: null, loading: false, initialized: true });
          return;
        }
      }

      set({ loading: true });
      setAuthMarkerCookie(true);

      try {
        const user = await api.get<AuthUser>(
          "/api/v1/auth/me",
          legacyToken ? { token: legacyToken } : undefined
        );
        // Migrate off localStorage JWTs once cookie session is established.
        localStorage.removeItem("token");
        localStorage.removeItem("refresh_token");
        cacheUser(user);
        set({
          user,
          token: "cookie",
          loading: false,
          initialized: true,
        });
      } catch (err) {
        const message = err instanceof Error ? err.message : "";
        const stillHasToken = Boolean(getStoredToken());
        const isNetworkError =
          message === CONNECTION_ERROR_MESSAGE ||
          /network|failed to fetch|could not connect/i.test(message);

        if (stillHasToken && isNetworkError) {
          // Transient outage — restore cached profile so refresh does not log the user out
          const cached = getCachedUser();
          set({
            user: cached,
            token: getStoredToken(),
            loading: false,
            initialized: true,
          });
          return;
        }

        if (!stillHasToken) {
          // Refresh definitively failed and cleared tokens
          clearStoredTokens();
          set({ user: null, token: null, loading: false, initialized: true });
          return;
        }

        // Unexpected error with tokens still present — keep session via cache
        const cached = getCachedUser();
        if (cached) {
          set({
            user: cached,
            token: getStoredToken(),
            loading: false,
            initialized: true,
          });
        } else {
          clearStoredTokens();
          set({ user: null, token: null, loading: false, initialized: true });
        }
      }
    } finally {
      if (failSafe != null) window.clearTimeout(failSafe);
    }
  },

  syncTokensRefreshed(detail) {
    if (!detail) return;

    set((state) => {
      const nextUser = detail.user ?? state.user;
      if (nextUser) cacheUser(nextUser);
      setAuthMarkerCookie(true);
      return {
        token: detail.accessToken ?? state.token,
        user: nextUser,
        loading: false,
        initialized: true,
      };
    });
  },

  syncLoggedOut() {
    cacheUser(null);
    setAuthMarkerCookie(false);
    set({
      token: null,
      user: null,
      loading: false,
      initialized: true,
    });
  },

  /** Cross-tab sync: re-read localStorage after another tab changes auth keys. */
  syncFromStorage() {
    const storedToken = getStoredToken();
    if (!storedToken) {
      set({
        token: null,
        user: null,
        loading: false,
        initialized: true,
      });
      setAuthMarkerCookie(false);
      return;
    }

    const cached = getCachedUser();
    setAuthMarkerCookie(true);
    set({
      token: storedToken,
      user: cached ?? get().user,
      loading: false,
      initialized: true,
    });

    // Refresh profile in background when another tab logged in
    if (!cached) {
      void api
        .get<AuthUser>("/api/v1/auth/me")
        .then((user) => {
          cacheUser(user);
          set({ user, token: getStoredToken(), initialized: true, loading: false });
        })
        .catch(() => {
          /* keep whatever we have */
        });
    }
  },
}));
