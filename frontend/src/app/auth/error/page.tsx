"use client";

import { Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { RefreshCcw, Home } from "lucide-react";
import AuthShell from "@/components/auth/AuthShell";

const ERROR_MESSAGES: Record<string, { title: string; description: string }> = {
  csrf_mismatch: {
    title: "Security Check Failed",
    description:
      "The OAuth state did not match. This could indicate a CSRF attack. Please try signing in again.",
  },
  token_exchange_failed: {
    title: "Token Exchange Failed",
    description:
      "We could not exchange your authorization code for an access token. Please try again.",
  },
  userinfo_failed: {
    title: "Profile Fetch Failed",
    description:
      "We could not retrieve your Hugging Face profile. Please check your account and try again.",
  },
  email_required: {
    title: "Email Required",
    description:
      "Your Hugging Face account did not provide an email address. Please ensure your email is public and try again.",
  },
  oauth_not_configured: {
    title: "OAuth Not Configured",
    description:
      "Hugging Face OAuth is not configured on this server. Please contact the administrator.",
  },
  default: {
    title: "Authentication Failed",
    description:
      "Something went wrong during sign-in with Hugging Face. Please try again.",
  },
};

function AuthErrorContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const errorCode = searchParams.get("error") ?? "default";
  const { title, description } =
    ERROR_MESSAGES[errorCode] ?? ERROR_MESSAGES.default;

  return (
    <AuthShell showBack={false}>
      <div className="genova-panel p-6 sm:p-8 space-y-6">
        <div className="space-y-1.5" role="alert">
          <h1 className="font-display text-2xl font-semibold tracking-tight">
            {title}
          </h1>
          <p className="text-sm leading-relaxed text-muted-foreground">
            {description}
          </p>
        </div>

        {errorCode !== "default" && (
          <span className="cite-keycap">error: {errorCode}</span>
        )}

        <div className="space-y-3">
          <Button
            className="w-full h-11 rounded-lg"
            onClick={() => router.push("/login")}
          >
            <RefreshCcw className="w-4 h-4 mr-2" />
            Try Again
          </Button>

          <Button
            variant="outline"
            className="w-full h-11 rounded-lg"
            onClick={() => router.push("/")}
          >
            <Home className="w-4 h-4 mr-2" />
            Go Home
          </Button>
        </div>
      </div>
    </AuthShell>
  );
}

export default function AuthErrorPage() {
  return (
    <Suspense>
      <AuthErrorContent />
    </Suspense>
  );
}
