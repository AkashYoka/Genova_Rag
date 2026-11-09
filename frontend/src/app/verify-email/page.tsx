"use client";

import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useTranslation } from "react-i18next";
import { api } from "@/lib/api";
import { buttonVariants } from "@/components/ui/button";
import AuthShell from "@/components/auth/AuthShell";

function VerifyEmailContent() {
  const { t } = useTranslation();
  const searchParams = useSearchParams();
  const token = searchParams.get("token");
  const [message, setMessage] = useState(t("verifyEmail.checking"));
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(Boolean(token));

  useEffect(() => {
    if (!token) return;

    let cancelled = false;

    api
      .get<{ message: string }>(`/api/v1/auth/verify?token=${encodeURIComponent(token)}`)
      .then((response) => {
        if (!cancelled) setMessage(response.message || t("verifyEmail.successFallback"));
      })
      .catch((err: unknown) => {
        const fallback = t("verifyEmail.errorFallback");
        if (!cancelled) setError(err instanceof Error ? err.message : fallback);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [token, t]);

  const displayError = token ? error : t("verifyEmail.missingToken");
  const displayMessage = loading ? t("verifyEmail.checking") : displayError || message;

  return (
    <AuthShell showBack={false}>
      <div className="genova-panel p-6 sm:p-8 space-y-6">
        <div className="space-y-1.5">
          <h1 className="font-display text-2xl font-semibold tracking-tight">
            {t("verifyEmail.title")}
          </h1>
        </div>

        <div
          role={displayError ? "alert" : "status"}
          className={
            displayError
              ? "p-3 rounded-lg bg-destructive/10 border border-destructive/30 text-sm text-destructive"
              : "p-3 rounded-lg bg-accent border border-primary/30 text-sm text-accent-foreground"
          }
        >
          {displayMessage}
        </div>

        <Link
          href="/login"
          className={buttonVariants({ className: "w-full h-11 text-base rounded-lg" })}
        >
          {t("verifyEmail.signIn")}
        </Link>
      </div>
    </AuthShell>
  );
}

export default function VerifyEmailPage() {
  return (
    <Suspense fallback={null}>
      <VerifyEmailContent />
    </Suspense>
  );
}
