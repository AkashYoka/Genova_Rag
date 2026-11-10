"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ArrowLeft,
  KeyRound,
  Link2,
  Sparkles,
  UserRound,
} from "lucide-react";

import { RequireAuth } from "@/components/auth/RequireAuth";
import {
  LedgerButton,
  LedgerCanvas,
  LedgerMobileBar,
  LedgerScroll,
  LedgerShell,
  LedgerSheet,
  LedgerSidebar,
  LedgerTitle,
  SettingsSection,
  type LedgerNavItem,
} from "@/components/ledger/LedgerPrimitives";
import AiUsagePanel from "@/components/settings/AiUsagePanel";
import ApiKeysPanel from "@/components/settings/ApiKeysPanel";
import DropboxSettings from "@/components/settings/DropboxSettings";
import GoogleDriveSettings from "@/components/settings/GoogleDriveSettings";
import ProfileSettings from "@/components/settings/ProfileSettings";
import { useAuth } from "@/lib/auth";
import { formatUsd } from "@/lib/admin";
import { api } from "@/lib/api";

type Section = "profile" | "usage" | "keys" | "integrations";

const NAV: LedgerNavItem[] = [
  { id: "profile", label: "Profile", icon: <UserRound /> },
  { id: "usage", label: "AI Usage", icon: <Sparkles /> },
  { id: "keys", label: "API Keys", icon: <KeyRound /> },
  { id: "integrations", label: "Integrations", icon: <Link2 /> },
];

function SettingsPageContent() {
  const router = useRouter();
  const { user } = useAuth();
  const [section, setSection] = useState<Section>("usage");
  const [menuOpen, setMenuOpen] = useState(false);
  const [costHint, setCostHint] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const summary = await api.get<{
          estimated_cost_usd: number;
          today: { estimated_cost_usd: number };
          month_to_date: { estimated_cost_usd: number };
        }>("/api/v1/usage/summary");
        if (!cancelled) {
          setCostHint(
            `${formatUsd(summary.estimated_cost_usd)} estimated · today ${formatUsd(summary.today.estimated_cost_usd)} · MTD ${formatUsd(summary.month_to_date.estimated_cost_usd)}`,
          );
        }
      } catch {
        if (!cancelled) setCostHint(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const finding = useMemo(() => {
    if (section === "profile") {
      return `${user?.email || "Account"} · role ${user?.role || (user?.is_admin ? "admin" : "user")}`;
    }
    if (section === "usage") {
      return costHint || "Token consumption and estimated cost for your account";
    }
    if (section === "keys") {
      return "Programmatic access to your workspace — secrets shown once";
    }
    return "Google Drive and Dropbox sync for document ingestion";
  }, [section, user, costHint]);

  const selectSection = (id: string) => {
    setSection(id as Section);
    setMenuOpen(false);
  };

  const sidebar = (
    <LedgerSidebar
      brand="Genova"
      scope="Settings"
      groupLabel="Account"
      items={NAV}
      activeId={section}
      onSelect={selectSection}
      footer={
        <LedgerButton
          variant="ghost"
          className="w-full justify-start"
          onClick={() => router.push("/dashboard")}
        >
          <ArrowLeft className="h-4 w-4" />
          Back to workspace
        </LedgerButton>
      }
    />
  );

  return (
    <LedgerShell>
      <div className="ledger-shell">
        {sidebar}
        <LedgerSheet open={menuOpen} onClose={() => setMenuOpen(false)}>
          {sidebar}
        </LedgerSheet>

        <LedgerCanvas>
          <LedgerMobileBar
            title="Settings"
            open={menuOpen}
            onToggle={() => setMenuOpen((v) => !v)}
          />
          <LedgerScroll>
            <LedgerTitle
              title={
                section === "profile"
                  ? "Profile"
                  : section === "usage"
                    ? "AI usage"
                    : section === "keys"
                      ? "API keys"
                      : "Integrations"
              }
              finding={finding}
            />

            {section === "profile" ? (
              <SettingsSection
                title="Account"
                description="Update the username and email on this account. Password changes are hashed and never stored in plain text."
                summary={
                  <p className="mt-4 text-[13px] text-[var(--gd-muted)]">
                    Signed in as <span className="text-[var(--gd-text)]">{user?.username}</span>
                  </p>
                }
              >
                <ProfileSettings />
              </SettingsSection>
            ) : null}

            {section === "usage" ? (
              <div className="mt-5">
                <AiUsagePanel />
              </div>
            ) : null}

            {section === "keys" ? (
              <SettingsSection
                title="Developer access"
                description="Create and revoke personal API keys. Full secrets are shown only at creation time."
              >
                <ApiKeysPanel />
              </SettingsSection>
            ) : null}

            {section === "integrations" ? (
              <>
                <SettingsSection
                  title="Google Drive"
                  description="Connect folders for automatic document sync. OAuth tokens stay on the server."
                >
                  <div className="rounded-[10px] border border-[var(--gd-border)] bg-[var(--gd-surface)] p-4">
                    <GoogleDriveSettings />
                  </div>
                </SettingsSection>

                <SettingsSection
                  title="Dropbox"
                  description="Connect Dropbox folders or import a public shared link. PDFs flow through the same ingestion and search pipeline."
                >
                  <div className="rounded-[10px] border border-[var(--gd-border)] bg-[var(--gd-surface)] p-4">
                    <DropboxSettings />
                  </div>
                </SettingsSection>
              </>
            ) : null}
          </LedgerScroll>
        </LedgerCanvas>
      </div>
    </LedgerShell>
  );
}

export default function SettingsPage() {
  return (
    <RequireAuth>
      <SettingsPageContent />
    </RequireAuth>
  );
}
