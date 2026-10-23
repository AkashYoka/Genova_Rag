"use client";

import { useCallback, useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Eye, EyeOff, Loader2, ArrowLeft } from "lucide-react";
import Link from "next/link";
import GoogleSignInButton from "@/components/auth/GoogleSignInButton";
import HuggingFaceSignInButton from "@/components/auth/HuggingFaceSignInButton";
import Logo from "@/components/layout/Logo";
import { Aurora } from "@/components/bits/Aurora";
import { FadeIn } from "@/components/bits/FadeIn";

export default function LoginPage() {
  const { login, user, initialized } = useAuth();
  const { t } = useTranslation();
  const router = useRouter();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (initialized && user) {
      router.replace("/dashboard");
    }
  }, [user, initialized, router]);

  if (!initialized || user) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  const handleGoogleSuccess = useCallback(() => {
    router.replace("/dashboard");
  }, [router]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      await login(email, password);
      router.replace("/dashboard");
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : t("login.fallbackError");
      setError(message);
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen grid lg:grid-cols-2 relative overflow-hidden">
      <Aurora className="lg:hidden" />

      <aside className="relative hidden lg:flex flex-col justify-between p-10 border-r border-border/40 overflow-hidden bg-[oklch(0.16_0.025_230)] text-[oklch(0.96_0.01_220)]">
        <Aurora />
        <div className="relative z-10 flex items-center gap-3">
          <Logo size={36} />
          <div>
            <p className="font-display text-xl font-semibold">Genova</p>
            <p className="text-[10px] uppercase tracking-[0.22em] text-white/55">
              by Yoka
            </p>
          </div>
        </div>
        <div className="relative z-10 max-w-md space-y-4">
          <h1 className="font-display text-4xl font-bold leading-tight tracking-tight">
            Cite every answer.
            <br />
            Trust every source.
          </h1>
          <p className="text-sm text-white/65 leading-relaxed">
            Sign in to your Genova_Rag workspace — hybrid retrieval, streaming
            chat, and page-accurate citations.
          </p>
        </div>
        <p className="relative z-10 text-xs text-white/40">
          Upload · Embed · Retrieve · Chat
        </p>
      </aside>

      <div className="relative flex items-center justify-center px-4 py-10">
        <FadeIn className="w-full max-w-md space-y-5">
          <Link
            href="/"
            className="inline-flex items-center gap-2 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors group"
          >
            <ArrowLeft className="w-3.5 h-3.5 transition-transform group-hover:-translate-x-1" />
            Back to Home
          </Link>

          <div className="genova-panel p-6 sm:p-8 space-y-6">
            <div className="space-y-2 text-center lg:text-left">
              <div className="flex justify-center lg:justify-start mb-3">
                <Logo size={40} className="lg:hidden" />
              </div>
              <h2 className="font-display text-2xl font-bold tracking-tight">
                {t("login.title")}
              </h2>
              <p className="text-sm text-muted-foreground">
                {t("login.description")}
              </p>
            </div>

            <div className="flex flex-col gap-2.5">
              <HuggingFaceSignInButton onError={setError} />
              <GoogleSignInButton
                onError={setError}
                onSuccess={handleGoogleSuccess}
              />
            </div>

            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t border-border/40" />
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-card px-2.5 text-muted-foreground text-[10px] tracking-wider font-semibold">
                  Or continue with
                </span>
              </div>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
              {error && (
                <div
                  className="p-3 rounded-xl bg-destructive/10 border border-destructive/30 text-sm text-destructive"
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
                  className="h-11 rounded-xl"
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
                    className="h-11 pr-10 rounded-xl"
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
                className="w-full h-11 text-base rounded-full"
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

            <p className="text-center text-sm text-muted-foreground">
              {t("login.noAccount")}{" "}
              <Link
                href="/register"
                className="text-primary hover:underline font-medium"
              >
                {t("login.createOne")}
              </Link>
            </p>
          </div>
        </FadeIn>
      </div>
    </div>
  );
}
