"use client";

import { useCallback, useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import Link from "next/link";
import GoogleSignInButton from "@/components/auth/GoogleSignInButton";
import { PasswordField } from "@/components/auth/PasswordField";
import AuthShell from "@/components/auth/AuthShell";

export default function RegisterPage() {
  const { register, user, initialized } = useAuth();
  const { t } = useTranslation();
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [registeredEmail, setRegisteredEmail] = useState("");
  const [verificationUrl, setVerificationUrl] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (initialized && user) {
      router.replace("/dashboard");
    }
  }, [user, initialized, router]);

  const handleGoogleSuccess = useCallback(() => {
    router.replace("/dashboard");
  }, [router]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess("");
    setVerificationUrl("");
    setLoading(true);

    try {
      const result = await register(username, email, password);
      setRegisteredEmail(result.email);
      setSuccess(result.message);
      setVerificationUrl(result.verification_url ?? "");
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : t("register.fallbackError");
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthShell>
      <div className="genova-panel p-6 sm:p-8 space-y-6">
        <div className="space-y-1.5">
          <h1 className="font-display text-2xl font-semibold tracking-tight">
            {t("register.title")}
          </h1>
          <p className="text-sm text-muted-foreground">
            {t("register.description")}
          </p>
        </div>

        <div className="flex flex-col gap-2.5">
          <GoogleSignInButton
            onError={setError}
            onSuccess={handleGoogleSuccess}
          />
        </div>

        <form onSubmit={handleSubmit} className="space-y-4" noValidate>
          {error && (
            <div
              role="alert"
              className="p-3 rounded-lg bg-destructive/10 border border-destructive/30 text-sm text-destructive"
            >
              {error}
            </div>
          )}
          {success && (
            <div className="p-3 rounded-lg bg-accent border border-primary/30 text-sm text-accent-foreground">
              <p className="font-medium">{t("register.verifyTitle")}</p>
              <p className="mt-1">
                {t("register.verifyMessage", { email: registeredEmail })}
              </p>
              {verificationUrl && (
                <Link
                  href={verificationUrl}
                  className="inline-flex mt-3 font-medium underline"
                >
                  {t("register.openVerification")}
                </Link>
              )}
            </div>
          )}

          <div className="space-y-2">
            <label htmlFor="reg-username" className="text-sm font-medium">
              {t("common.username")}
            </label>
            <Input
              id="reg-username"
              type="text"
              placeholder="akash"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              minLength={3}
              disabled={Boolean(success)}
              className="h-11 rounded-lg"
            />
          </div>

          <div className="space-y-2">
            <label htmlFor="reg-email" className="text-sm font-medium">
              {t("common.email")}
            </label>
            <Input
              id="reg-email"
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              disabled={Boolean(success)}
              className="h-11 rounded-lg"
            />
          </div>

          <PasswordField
            id="reg-password"
            value={password}
            onChange={setPassword}
            placeholder={t("register.passwordPlaceholder")}
            disabled={Boolean(success)}
          />

          <Button
            type="submit"
            className="w-full h-11 text-base rounded-lg"
            disabled={loading || Boolean(success)}
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <span className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                {t("register.submitting")}
              </span>
            ) : (
              t("register.submit")
            )}
          </Button>
        </form>

        <p className="text-sm text-muted-foreground">
          {t("register.hasAccount")}{" "}
          <Link
            href="/login"
            className="text-primary hover:underline font-medium"
          >
            {t("register.signIn")}
          </Link>
        </p>
      </div>
    </AuthShell>
  );
}
