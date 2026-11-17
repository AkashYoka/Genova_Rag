"use client";

import Link from "next/link";
import { ArrowLeft, Activity } from "lucide-react";

import { RequireAuth } from "@/components/auth/RequireAuth";
import { AiTracesPanel } from "@/components/admin/AiTracesPanel";
import {
  LedgerButton,
  LedgerCanvas,
  LedgerShell,
  LedgerTitle,
} from "@/components/ledger/LedgerPrimitives";

export default function AdminAiTracesPage() {
  return (
    <RequireAuth requireAdmin>
      <LedgerShell>
        <LedgerCanvas>
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <Link href="/admin">
                <LedgerButton>
                  <ArrowLeft className="h-3.5 w-3.5" />
                  Admin
                </LedgerButton>
              </Link>
              <LedgerTitle
                title="AI processing traces"
                finding="Inspect retrieval, LLM calls, and failures for every user question."
              />
            </div>
            <Activity className="h-5 w-5 text-[var(--gd-muted)]" />
          </div>
          <AiTracesPanel syncUrl />
        </LedgerCanvas>
      </LedgerShell>
    </RequireAuth>
  );
}
