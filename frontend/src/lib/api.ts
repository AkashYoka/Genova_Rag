/**
 * API client for the FastAPI backend.
 * Handles authentication headers, base URL configuration,
 * and automatic token refresh on 401 responses.
 */


const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:7860";
const CONNECTION_ERROR_MESSAGE = "Could not connect to the server. Please try again later.";
const CONNECTION_ERROR_BANNER_MESSAGE = `⚠️ ${CONNECTION_ERROR_MESSAGE}`;

const AUTH_MARKER_COOKIE = "genova_auth";
const PROTECTED_PATH_PREFIXES = ["/dashboard", "/documents", "/admin", "/settings"];

interface FetchOptions extends RequestInit {
  token?: string;
  /** Skip auto-refresh for this request (used internally for the refresh call itself) */
  _skipRefresh?: boolean;
}

function setAuthMarkerCookie(present: boolean): void {
  if (typeof document === "undefined") return;
  if (present) {
    // Session marker cookie (not the JWT). Soft client hint only — RequireAuth
    // and API get_current_user remain authoritative under static export.
    document.cookie = `${AUTH_MARKER_COOKIE}=1; Path=/; SameSite=Lax; Max-Age=${60 * 60 * 24 * 7}`;
  } else {
    document.cookie = `${AUTH_MARKER_COOKIE}=; Path=/; SameSite=Lax; Max-Age=0`;
  }
}

class ApiClient {
  private baseUrl: string;
  /** Guards against multiple concurrent refresh attempts */
  private refreshPromise: Promise<string | null> | null = null;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  private getToken(): string | null {
    if (typeof window === "undefined") return null;
    return localStorage.getItem("token");
  }

  private getRefreshToken(): string | null {
    if (typeof window === "undefined") return null;
    return localStorage.getItem("refresh_token");
  }

  /** Public accessor for components that need a fresh access token (PDF, WS). */
  getAccessToken(): string | null {
    return this.getToken();
  }

  private getHeaders(token?: string): HeadersInit {
    const headers: HeadersInit = {
      "Content-Type": "application/json",
    };

    const authToken = token || this.getToken();
    if (authToken && authToken !== "cookie") {
      headers["Authorization"] = `Bearer ${authToken}`;
    }

    return headers;
  }

  private async fetchWithConnectionError(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
    try {
      const host = typeof window !== "undefined" && window.location ? window.location.host : "";
      const isCrossOrigin = typeof window !== "undefined" && Boolean(host) && this.baseUrl.startsWith("http") && !this.baseUrl.includes(host);
      const mergedInit = {
        credentials: isCrossOrigin ? "same-origin" as const : "include" as const,
        ...init,
      };
      return await fetch(input, mergedInit);
    } catch (error) {
      if (error instanceof TypeError) {
        throw new Error(CONNECTION_ERROR_MESSAGE);
      }
      throw error;
    }
  }

  /**
   * Attempt to refresh the access token using the stored refresh token.
   * Uses a mutex so only one refresh happens at a time — concurrent
   * 401s all wait on the same promise.
   *
   * Network failures do NOT clear tokens (avoids unexpected logout).
   * Only definitive auth failures (401/403 from refresh) clear the session.
   */
  private async tryRefreshToken(): Promise<string | null> {
    // If a refresh is already in-flight, wait for it
    if (this.refreshPromise) {
      return this.refreshPromise;
    }

    const refreshToken = this.getRefreshToken();
    if (!refreshToken) return null;

    this.refreshPromise = (async () => {
      try {
        const res = await this.fetchWithConnectionError(`${this.baseUrl}/api/v1/auth/refresh`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ refresh_token: refreshToken }),
        });

        if (!res.ok) {
          // Only force logout when the refresh token is genuinely invalid
          if (res.status === 401 || res.status === 403) {
            this.clearTokens();
          }
          return null;
        }

        const data = await res.json();
        localStorage.setItem("token", data.access_token);
        if (data.refresh_token) {
          localStorage.setItem("refresh_token", data.refresh_token);
        }
        setAuthMarkerCookie(true);

        window.dispatchEvent(
          new CustomEvent("auth:tokens-refreshed", {
            detail: { accessToken: data.access_token, refreshToken: data.refresh_token, user: data.user },
          })
        );

        return data.access_token as string;
      } catch {
        // Network / transient failure — keep existing tokens so the user is not logged out
        return null;
      } finally {
        this.refreshPromise = null;
      }
    })();

    return this.refreshPromise;
  }

  private isProtectedPath(pathname: string): boolean {
    return PROTECTED_PATH_PREFIXES.some(
      (prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`),
    );
  }

  private handleUnauthorized(): void {
    this.clearTokens();
    if (typeof window === "undefined") return;

    if (this.isProtectedPath(window.location.pathname)) {
      window.location.href = "/login";
    }
  }

  private clearTokens(): void {
    localStorage.removeItem("token");
    localStorage.removeItem("refresh_token");
    localStorage.removeItem("auth_user");
    setAuthMarkerCookie(false);
    // Notify AuthProvider (same tab) and other tabs via storage event on token keys
    window.dispatchEvent(new CustomEvent("auth:logged-out"));
  }

  /** Persist session tokens after login/register flows outside this client. */
  persistSession(accessToken: string, refreshToken: string): void {
    localStorage.setItem("token", accessToken);
    localStorage.setItem("refresh_token", refreshToken);
    setAuthMarkerCookie(true);
  }

  clearSession(): void {
    this.clearTokens();
  }

  private getPayloadMessage(payload: unknown): string | null {
    if (typeof payload === "string" && payload.trim()) {
      return payload;
    }

    if (Array.isArray(payload)) {
      const messages = payload
        .map((item) => {
          if (typeof item === "string") return item;
          if (item && typeof item === "object" && "msg" in item && typeof item.msg === "string") {
            return item.msg;
          }
          return null;
        })
        .filter((message): message is string => Boolean(message));

      return messages.length > 0 ? messages.join(", ") : null;
    }

    if (payload && typeof payload === "object") {
      const obj = payload as { message?: unknown; code?: unknown };
      if (typeof obj.message === "string" && obj.message.trim()) {
        return obj.message;
      }
    }

    return null;
  }

  private async getErrorMessage(res: Response, fallback: string): Promise<string> {
    const payload = await res.json().catch(() => null);

    if (payload && typeof payload === "object") {
      const errorPayload = payload as { detail?: unknown; error?: unknown; message?: unknown };
      return (
        this.getPayloadMessage(errorPayload.detail) ||
        this.getPayloadMessage(errorPayload.message) ||
        this.getPayloadMessage(errorPayload.error) ||
        fallback
      );
    }

    return this.getPayloadMessage(payload) || fallback;
  }

  /**
   * On 401: try refresh once. Logout only when the session is definitively gone
   * (no refresh token, or refresh returned 401/403 and cleared tokens).
   */
  private async handleAuthFailure<T>(
    retry: (newToken: string) => Promise<T>,
    skipRefresh?: boolean,
  ): Promise<T | null> {
    if (skipRefresh) return null;

    const newToken = await this.tryRefreshToken();
    if (newToken) {
      return retry(newToken);
    }

    // Definitive auth loss: tokens cleared by refresh, or never had a refresh token
    if (!this.getRefreshToken()) {
      this.handleUnauthorized();
    }
    return null;
  }

  async get<T>(path: string, options?: FetchOptions): Promise<T> {
    const res = await this.fetchWithConnectionError(`${this.baseUrl}${path}`, {
      method: "GET",
      headers: this.getHeaders(options?.token),
      ...options,
    });

    if (res.status === 401 && !options?._skipRefresh) {
      const retried = await this.handleAuthFailure(
        (newToken) => this.get<T>(path, { ...options, token: newToken, _skipRefresh: true }),
        options?._skipRefresh,
      );
      if (retried !== null) return retried;
      // Tokens still present ⇒ refresh failed transiently (network) — do not treat as logout
      if (this.getRefreshToken()) {
        throw new Error(CONNECTION_ERROR_MESSAGE);
      }
      throw new Error(await this.getErrorMessage(res, "Unauthorized"));
    }

    if (!res.ok) {
      throw new Error(await this.getErrorMessage(res, res.statusText || "Request failed"));
    }

    return res.json();
  }

  /** Download binary/file responses (e.g. admin DB export). */
  async getBlob(path: string, options?: FetchOptions): Promise<Blob> {
    const res = await this.fetchWithConnectionError(`${this.baseUrl}${path}`, {
      method: "GET",
      headers: this.getHeaders(options?.token),
      ...options,
    });

    if (res.status === 401 && !options?._skipRefresh) {
      const retried = await this.handleAuthFailure(
        (newToken) =>
          this.getBlob(path, { ...options, token: newToken, _skipRefresh: true }),
        options?._skipRefresh,
      );
      if (retried !== null) return retried;
      if (this.getRefreshToken()) {
        throw new Error(CONNECTION_ERROR_MESSAGE);
      }
      throw new Error(await this.getErrorMessage(res, "Unauthorized"));
    }

    if (!res.ok) {
      throw new Error(await this.getErrorMessage(res, res.statusText || "Request failed"));
    }

    return res.blob();
  }

  async post<T>(path: string, body?: unknown, options?: FetchOptions): Promise<T> {
    const res = await this.fetchWithConnectionError(`${this.baseUrl}${path}`, {
      method: "POST",
      headers: this.getHeaders(options?.token),
      body: body ? JSON.stringify(body) : undefined,
      ...options,
    });

    if (res.status === 401 && !options?._skipRefresh) {
      const retried = await this.handleAuthFailure(
        (newToken) => this.post<T>(path, body, { ...options, token: newToken, _skipRefresh: true }),
        options?._skipRefresh,
      );
      if (retried !== null) return retried;
      throw new Error(await this.getErrorMessage(res, "Unauthorized"));
    }

    if (!res.ok) {
      throw new Error(await this.getErrorMessage(res, res.statusText || "Request failed"));
    }

    return res.json();
  }

  async put<T>(path: string, body?: unknown, options?: FetchOptions): Promise<T> {
    const res = await this.fetchWithConnectionError(`${this.baseUrl}${path}`, {
      method: "PUT",
      headers: this.getHeaders(options?.token),
      body: body ? JSON.stringify(body) : undefined,
      ...options,
    });

    if (res.status === 401 && !options?._skipRefresh) {
      const retried = await this.handleAuthFailure(
        (newToken) => this.put<T>(path, body, { ...options, token: newToken, _skipRefresh: true }),
        options?._skipRefresh,
      );
      if (retried !== null) return retried;
      throw new Error(await this.getErrorMessage(res, "Unauthorized"));
    }

    if (!res.ok) {
      throw new Error(await this.getErrorMessage(res, res.statusText || "Request failed"));
    }

    return res.json();
  }

  async patch<T>(path: string, body?: unknown, options?: FetchOptions): Promise<T> {
    const res = await this.fetchWithConnectionError(`${this.baseUrl}${path}`, {
      method: "PATCH",
      headers: this.getHeaders(options?.token),
      body: body ? JSON.stringify(body) : undefined,
      ...options,
    });

    if (res.status === 401 && !options?._skipRefresh) {
      const retried = await this.handleAuthFailure(
        (newToken) => this.patch<T>(path, body, { ...options, token: newToken, _skipRefresh: true }),
        options?._skipRefresh,
      );
      if (retried !== null) return retried;
      throw new Error(await this.getErrorMessage(res, "Unauthorized"));
    }

    if (!res.ok) {
      throw new Error(await this.getErrorMessage(res, res.statusText || "Request failed"));
    }

    return res.json();
  }

  async renameDocument<T>(documentId: string, name: string): Promise<T> {
    return this.patch<T>(`/api/v1/documents/${documentId}`, { name });
  }

  async postForm<T>(path: string, formData: FormData, options?: FetchOptions): Promise<T> {
    const token = options?.token || this.getToken();
    const headers: HeadersInit = {};
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }

    const res = await this.fetchWithConnectionError(`${this.baseUrl}${path}`, {
      method: "POST",
      headers,
      body: formData,
      ...options,
    });

    if (res.status === 401 && !options?._skipRefresh) {
      const retried = await this.handleAuthFailure(
        (newToken) => this.postForm<T>(path, formData, { ...options, token: newToken, _skipRefresh: true }),
        options?._skipRefresh,
      );
      if (retried !== null) return retried;
      throw new Error(await this.getErrorMessage(res, "Unauthorized"));
    }

    if (!res.ok) {
      throw new Error(await this.getErrorMessage(res, res.statusText || "Upload failed"));
    }

    return res.json();
  }

  async delete<T>(path: string, options?: FetchOptions): Promise<T> {
    const { token, _skipRefresh, ...fetchInit } = options ?? {};
    const res = await this.fetchWithConnectionError(`${this.baseUrl}${path}`, {
      ...fetchInit,
      method: "DELETE",
      headers: this.getHeaders(token),
    });

    if (res.status === 401 && !_skipRefresh) {
      const retried = await this.handleAuthFailure(
        (newToken) => this.delete<T>(path, { ...options, token: newToken, _skipRefresh: true }),
        _skipRefresh,
      );
      if (retried !== null) return retried;
      throw new Error(await this.getErrorMessage(res, "Unauthorized"));
    }

    if (!res.ok) {
      throw new Error(await this.getErrorMessage(res, res.statusText || "Delete failed"));
    }

    if (res.status === 204 || res.status === 205) {
      return undefined as T;
    }
    const text = await res.text();
    if (!text) {
      return undefined as T;
    }
    return JSON.parse(text) as T;
  }

  /**
   * Stream a POST request as Server-Sent Events.
   * Yields parsed SSE data objects.
   */
  async *streamPost(path: string, body: unknown, signal?: AbortSignal): AsyncGenerator<{ type: string; data?: unknown }> {
    let res = await this.fetchWithConnectionError(`${this.baseUrl}${path}`, {
      method: "POST",
      headers: this.getHeaders(),
      body: JSON.stringify(body),
      signal,
    });

    if (res.status === 401) {
      const newToken = await this.tryRefreshToken();
      if (newToken) {
        res = await this.fetchWithConnectionError(`${this.baseUrl}${path}`, {
          method: "POST",
          headers: this.getHeaders(newToken),
          body: JSON.stringify(body),
          signal,
        });
      } else if (!this.getRefreshToken()) {
        this.handleUnauthorized();
      }
    }

    if (!res.ok) {
      throw new Error(await this.getErrorMessage(res, res.statusText || "Stream request failed"));
    }

    const reader = res.body?.getReader();
    if (!reader) throw new Error("No response body");

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const data = JSON.parse(line.slice(6));
            yield data;
          } catch {
            // Skip malformed lines
          }
        }
      }
    }
  }

  /**
   * Download a protected resource with Authorization header (avoids JWT in URL).
   */
  async downloadAuthenticated(
    path: string,
    init?: { method?: "GET" | "POST"; body?: unknown },
  ): Promise<{ blob: Blob; filename: string | null }> {
    const method = init?.method ?? "GET";
    const buildOpts = (token?: string | null): RequestInit => ({
      method,
      headers: this.getHeaders(token ?? undefined),
      body:
        method !== "GET" && init?.body !== undefined
          ? JSON.stringify(init.body)
          : undefined,
    });

    let res = await this.fetchWithConnectionError(
      `${this.baseUrl}${path}`,
      buildOpts(),
    );

    if (res.status === 401) {
      const newToken = await this.tryRefreshToken();
      if (newToken) {
        res = await this.fetchWithConnectionError(
          `${this.baseUrl}${path}`,
          buildOpts(newToken),
        );
      } else if (!this.getRefreshToken()) {
        this.handleUnauthorized();
      }
    }

    if (!res.ok) {
      throw new Error(await this.getErrorMessage(res, "Download failed"));
    }

    const disposition = res.headers.get("Content-Disposition") ?? "";
    const filename = /filename="?([^";]+)"?/.exec(disposition)?.[1] ?? null;
    const blob = await res.blob();
    return { blob, filename };
  }

  /** @deprecated Prefer Authorization header fetches; kept for legacy callers. */
  getPdfUrl(documentId: string): string {
    return `${this.baseUrl}/api/v1/documents/${documentId}/pdf`;
  }
}

export const api = new ApiClient(API_BASE);
export { API_BASE, CONNECTION_ERROR_BANNER_MESSAGE, CONNECTION_ERROR_MESSAGE, setAuthMarkerCookie };
