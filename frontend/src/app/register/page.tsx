"use client";

import { useCallback, useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import GoogleSignInButton from "@/components/auth/GoogleSignInButton";
import { PasswordField } from "@/components/auth/PasswordField";
import HuggingFaceSignInButton from "@/components/auth/HuggingFaceSignInButton";
import Logo from "@/components/layout/Logo";
import { Aurora } from "@/components/bits/Aurora";
import { FadeIn } from "@/components/bits/FadeIn";

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
    <div className="min-h-screen flex items-center justify-center px-4 py-8 relative overflow-hidden">
      <Aurora />

      <FadeIn className="w-full max-w-md relative z-10 space-y-4">
        <Link
          href="/"
          className="inline-flex items-center gap-2 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors group"
        >
          <ArrowLeft className="w-3.5 h-3.5 transition-transform group-hover:-translate-x-1" />
          Back to Home
        </Link>

        <Card className="w-full genova-panel border-border/50">
          <CardHeader className="text-center pb-2">
            <div className="flex justify-center mb-4">
              <Logo size={44} />
            </div>
            <CardTitle className="font-display text-2xl font-bold tracking-tight">
              {t("register.title")}
            </CardTitle>
            <CardDescription>{t("register.description")}</CardDescription>
          </CardHeader>

          <CardContent>
            <div className="flex flex-col gap-2.5 mb-4">
              <HuggingFaceSignInButton onError={setError} />
              <GoogleSignInButton
                onError={setError}
                onSuccess={handleGoogleSuccess}
              />
            </div>

            <form onSubmit={handleSubmit} className="space-y-4" noValidate>
              {error && (
                <div
                  role="alert"
                  className="p-3 rounded-xl bg-destructive/10 border border-destructive/30 text-sm text-destructive"
                >
                  {error}
                </div>
              )}
              {success && (
                <div className="p-3 rounded-xl bg-primary/10 border border-primary/30 text-sm text-primary">
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
                  placeholder="paramjit"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                  minLength={3}
                  disabled={Boolean(success)}
                  className="h-11 rounded-xl"
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
                  className="h-11 rounded-xl"
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
                className="w-full h-11 text-base rounded-full"
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

            <p className="text-center text-sm text-muted-foreground mt-6">
              {t("register.hasAccount")}{" "}
              <Link
                href="/login"
                className="text-primary hover:underline font-medium"
              >
                {t("register.signIn")}
              </Link>
            </p>
          </CardContent>
        </Card>
      </FadeIn>
    </div>
  );
}
