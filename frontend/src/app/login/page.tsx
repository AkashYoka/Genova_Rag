"use client";

import { useCallback, useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Eye, EyeOff, Loader2 } from "lucide-react";
import Link from "next/link";
import GoogleSignInButton from "@/components/auth/GoogleSignInButton";
import AuthShell from "@/components/auth/AuthShell";

export default function LoginPage() {
  const { login, user, initialized } = useAuth();
  const { t } = useTranslation();
  const router = useRouter();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const postLoginPath = useCallback(() => {
    if (typeof window === "undefined") return "/dashboard";
    const next = new URLSearchParams(window.location.search).get("next");
    if (next && next.startsWith("/") && !next.startsWith("//")) {
      return next;
    }
    return "/dashboard";
  }, []);

  useEffect(() => {
    if (initialized && user) {
      router.replace(postLoginPath());
    }
  }, [user, initialized, router, postLoginPath]);

  const handleGoogleSuccess = useCallback(() => {
    router.replace(postLoginPath());
  }, [router, postLoginPath]);

  if (!initialized || user) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      await login(email, password);
      router.replace(postLoginPath());
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : t("login.fallbackError");
      setError(message);
      setLoading(false);
    }
  };

  return (
    <AuthShell>
      <div className="genova-panel p-6 sm:p-8 space-y-6">
        <div className="space-y-1.5">
          <h1 className="font-display text-2xl font-semibold tracking-tight">
            {t("login.title")}
          </h1>
          <p className="text-sm text-muted-foreground">
            {t("login.description")}
          </p>
        </div>

        <div className="flex flex-col gap-2.5">
          <GoogleSignInButton
            onError={setError}
            onSuccess={handleGoogleSuccess}
          />
        </div>

        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span className="h-px flex-1 bg-border" />
          <span>or with email</span>
          <span className="h-px flex-1 bg-border" />
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {error && (
            <div
              className="p-3 rounded-lg bg-destructive/10 border border-destructive/30 text-sm text-destructive"
              role="alert"
            >
              {error}
            </div>
          )}

          <div className="space-y-2">
            <label htmlFor="login-email" className="text-sm font-medium">
              {t("common.email")}
            </label>
            <Input
              id="login-email"
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="h-11 rounded-lg"
              autoComplete="email"
            />
          </div>

          <div className="space-y-2">
            <label htmlFor="login-password" className="text-sm font-medium">
              {t("common.password")}
            </label>
            <div className="relative">
              <Input
                id="login-password"
                type={showPw ? "text" : "password"}
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="h-11 pr-10 rounded-lg"
                autoComplete="current-password"
              />
              <button
                type="button"
                onClick={() => setShowPw(!showPw)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                aria-label={showPw ? "Hide password" : "Show password"}
                aria-pressed={showPw}
              >
                {showPw ? (
                  <EyeOff className="w-4 h-4" />
                ) : (
                  <Eye className="w-4 h-4" />
                )}
              </button>
            </div>
          </div>

          <Button
            id="sign-in-btn"
            type="submit"
            className="w-full h-11 text-base rounded-lg"
            disabled={loading}
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin" />
                {t("login.submitting")}
              </span>
            ) : (
              t("login.submit")
            )}
          </Button>
        </form>

        <p className="text-sm text-muted-foreground">
          {t("login.noAccount")}{" "}
          <Link
            href="/register"
            className="text-primary hover:underline font-medium"
          >
            {t("login.createOne")}
          </Link>
        </p>
      </div>
    </AuthShell>
  );
}
